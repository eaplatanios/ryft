use crate::arrays::{ArrayType, DataType, Dimension, Shape};
use crate::operations::constants::fill::Fill;
use crate::operations::manipulation::broadcasting::Broadcast;
use crate::operations::manipulation::conversion::ConvertElementType;
use crate::operations::manipulation::reshaping::Reshape;
use crate::operations::math::abs::Abs;
use crate::operations::math::clamp::Clamp;
use crate::operations::math::div::Div;
use crate::operations::math::exp::Exp;
use crate::operations::math::floor::Floor;
use crate::operations::math::log::Log;
use crate::operations::math::max::Max;
use crate::operations::math::mul::Mul;
use crate::operations::math::reduce::{Reduce, ReductionKind};
use crate::operations::math::sub::Sub;
use crate::programs::{ProgramError, TypeError, Value};

/// Value-level block-quantization capability: splits a full-precision (`f32` or `f64`) tensor of rank 1 through 3
/// into a narrow element tensor and a tensor of per-block scales along the trailing dimension (which must be
/// divisible by the block size), producing operands for [`ScaledDot`](crate::ScaledDot). This enables on-the-fly
/// quantization (e.g., of a KV cache) without a dedicated primitive: the recipe is a pure composition of existing
/// operations, so it inherits its transform rules from them.
///
/// Two recipes cover the standard microscaling formats, selected by the scale type:
///
///   - **`f8e4m3fn` scales** (NVIDIA's NVFP4 recipe): each block's scale is `max_abs(block) / element_max`, where
///     `element_max` is the element type's maximum finite magnitude (`6.0` for `f4e2m1fn`), so the block's largest
///     element quantizes to the top of the element grid.
///   - **`f8e8m0fnu` scales** (the [OCP MX](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
///     recipe, e.g. MXFP8): each block's shared scale is the power of two `2^(floor(log2(max_abs(block))) - emax)`,
///     where `emax` is the element type's maximum exponent (`8` for `f8e4m3fn` and `15` for `f8e5m2`), so the
///     block's largest element lands in the element type's top binade. Elements just past the maximum finite
///     magnitude — up to `2^(emax + 1)` — are explicitly clamped before conversion, as the OCP MX specification
///     prescribes; this avoids relying on a floating-point format's overflow conversion policy.
///
/// In both recipes the elements are the input divided by its block's *stored* (already narrowed) scale and
/// converted to the element type, so dequantization (see [`ScaledDotOperation`](crate::ScaledDotOperation))
/// reproduces the input up to element quantization error. The `log2` in the MX recipe is composed as
/// `log(x) / log(2)` plus a `1e-4` nudge before the floor: the nudge keeps block maxima that are exact powers of two
/// in their own binade despite floating-point rounding in the quotient, and the subsequent conversion of
/// `exp(exponent · log(2))` to `f8e8m0fnu` rounds to the nearest power of two, absorbing the remaining approximation
/// error entirely. All-zero (and denormal-tiny) blocks clamp their scale up to a small representable positive value
/// instead of producing a zero or infinite scale.
pub trait BlockQuantize: Sized {
    /// Quantizes `self` into `(elements, scales)` per block of `block_size` trailing-dimension values, where
    /// `elements` carries `element_type` with the shape of `self` and `scales` carries `scale_type` with the
    /// trailing dimension divided by `block_size`. Refer to the trait documentation for the exact recipes. Returns
    /// a [`ProgramError`] if something goes wrong.
    fn block_quantize(
        &self,
        block_size: usize,
        element_type: DataType,
        scale_type: DataType,
    ) -> Result<(Self, Self), ProgramError>;
}

/// Every value with the elementwise, reduction, and reshaping capabilities used by the recipe (which covers both
/// the concrete reference [`Array`](crate::arrays::Array) backend and the transform tracers) quantizes
/// through the shared composition.
impl<V> BlockQuantize for V
where
    V: Value<Type = ArrayType>
        + Abs
        + Clamp
        + Broadcast
        + ConvertElementType
        + Div
        + Exp
        + Floor
        + Log
        + Max
        + Mul
        + Reduce
        + Reshape
        + Sub,
    V::DispatchDomain: Fill<f64, V>,
{
    fn block_quantize(
        &self,
        block_size: usize,
        element_type: DataType,
        scale_type: DataType,
    ) -> Result<(Self, Self), ProgramError> {
        // Max finite magnitude and maximum exponent of the supported microscaling element types.
        let (element_max, element_max_exponent) = match element_type {
            DataType::F4E2M1FN => (6.0, 2.0),
            DataType::F8E4M3FN => (448.0, 8.0),
            DataType::F8E5M2 => (57344.0, 15.0),
            element_type => {
                return Err(TypeError::invalid(format!(
                    "'block_quantize' does not support element data type {element_type}"
                ))
                .into());
            }
        };
        let input_type = self.r#type().into_owned();
        let compute_type = input_type.data_type();
        if !matches!(compute_type, DataType::F32 | DataType::F64) {
            return Err(TypeError::invalid(format!(
                "'block_quantize' expects an f32 or f64 input but got {compute_type}"
            ))
            .into());
        }
        let Some(shape) = input_type.static_shape() else {
            return Err(TypeError::invalid("'block_quantize' input must have a static shape".to_string()).into());
        };
        let dimensions = shape.dimensions().to_vec();
        if dimensions.is_empty() || dimensions.len() > 3 {
            return Err(TypeError::invalid(format!(
                "'block_quantize' input must have rank between 1 and 3 but got rank {}",
                dimensions.len(),
            ))
            .into());
        }
        let trailing_size = *dimensions.last().unwrap();
        if block_size == 0 || trailing_size % block_size != 0 {
            return Err(TypeError::invalid(format!(
                "'block_quantize' trailing dimension size {trailing_size} is not divisible by block size \
                     {block_size}"
            ))
            .into());
        }
        let mut scale_dimensions = dimensions.clone();
        *scale_dimensions.last_mut().unwrap() = trailing_size / block_size;
        let block_shape = Shape::new(
            scale_dimensions
                .iter()
                .map(|&size| Dimension::Static(size))
                .chain(std::iter::once(Dimension::Static(block_size)))
                .collect(),
        );
        let scale_value_type = ArrayType::new(
            compute_type,
            Shape::new(scale_dimensions.iter().map(|&size| Dimension::Static(size)).collect()),
        );
        let domain = self.dispatch_domain();
        let fill = |value: f64| domain.fill(&scale_value_type, value);

        // Per-block maximum magnitude along the trailing dimension.
        let block_max = self.reshape(block_shape.clone())?.abs()?.reduce(&[scale_dimensions.len()], ReductionKind::Max);
        let (scale, smallest_scale) = match scale_type {
            // NVFP4-style linear scaling: the block maximum maps to the element type's maximum magnitude. The clamp
            // floor is the scale type's smallest positive normal, `2^-6`.
            DataType::F8E4M3FN => (block_max.div(&fill(element_max)?)?, (-6.0f64).exp2()),
            // OCP MX power-of-two scaling: `2^(floor(log2(max_abs)) - emax)` with the boundary nudge documented on
            // the trait, folded into one subtraction because `floor(x + ε) - emax = floor(x + ε - emax)` for the
            // integer `emax`. The clamp floor is the scale type's smallest representable value, `2^-127` (which
            // also absorbs the `exp(-inf) = 0` produced by all-zero blocks).
            DataType::F8E8M0FNU => {
                let log_2 = fill(std::f64::consts::LN_2)?;
                let exponent = block_max.log()?.div(&log_2)?.sub(&fill(element_max_exponent - 1e-4)?)?.floor()?;
                (exponent.mul(&log_2)?.exp()?, (-127.0f64).exp2())
            }
            scale_type => {
                return Err(TypeError::invalid(format!(
                    "'block_quantize' does not support scale data type {scale_type}"
                ))
                .into());
            }
        };
        let scales = scale.max(&fill(smallest_scale)?)?.convert_element_type(scale_type)?;

        // Divide by the *stored* scale — exactly the value `scaled_dot` dequantizes with — and narrow the elements.
        let stored_scales = scales.convert_element_type(compute_type)?;
        let expanded_type = ArrayType::new(compute_type, block_shape);
        let scale_axes = (0..scale_dimensions.len()).collect::<Vec<_>>();
        let expanded_scales =
            stored_scales.broadcast(expanded_type, scale_axes.as_slice())?.reshape(input_type.shape().clone())?;
        let fill_scalar = |value: f64| domain.fill(&ArrayType::scalar(compute_type), value);
        let elements = self
            .div(&expanded_scales)?
            .clamp(&fill_scalar(-element_max)?, &fill_scalar(element_max)?)?
            .convert_element_type(element_type)?;
        Ok((elements, scales))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayOperation, ArrayType, DataType, Dimension, Shape};
    use crate::contexts::StagingContext;
    use crate::operations::math::dot::{Dot, DotDimensionNumbers, ScaledDot};
    use crate::programs::Typed;
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_block_quantize_nvfp4() {
        // NVFP4 recipe: `f4e2m1fn` elements with `f8e4m3fn` scales, `scale = max_abs(block) / 6.0`. Every block
        // below is a scaled copy of `f4e2m1fn` grid points whose scale is exactly representable in `f8e4m3fn`, so
        // quantization is exact; the all-zero block exercises the clamp to the smallest normal scale, `2^-6`.
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(8)]));
        let input = Array::from_f64s(
            input_type.clone(),
            vec![
                3.0, 1.5, 0.5, 6.0, 0.5, 1.0, 0.25, 1.5, // Blocks with scales 1.0 and 0.25.
                -12.0, 6.0, 3.0, -1.0, 0.0, 0.0, 0.0, 0.0, // Blocks with scale 2.0 and the clamp floor.
            ],
        );
        let (elements, scales) = input.block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN).unwrap();
        assert_eq!(
            elements.r#type().as_ref(),
            &ArrayType::new(DataType::F4E2M1FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(8)])),
        );
        assert_eq!(
            scales.r#type().as_ref(),
            &ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])),
        );
        assert_eq!(scales.to_f64s(), vec![1.0, 0.25, 2.0, 0.015625]);
        assert_eq!(
            elements.to_f64s(),
            vec![3.0, 1.5, 0.5, 6.0, 2.0, 4.0, 1.0, 6.0, -6.0, 3.0, 1.5, -0.5, 0.0, 0.0, 0.0, 0.0],
        );

        // Round trip: the quantized operands contract through `scaled_dot` to the exact full-precision dot.
        let dimensions = DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new());
        let product = elements
            .scaled_dot(&elements, Some(&scales), Some(&scales), Some(&dimensions), Some(DataType::F32))
            .unwrap();
        let expected = input.dot(&input, &DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new()));
        assert_eq!(product.to_f64s(), expected.to_f64s());

        // Contract violations report clear errors.
        assert!(matches!(
            input.block_quantize(3, DataType::F4E2M1FN, DataType::F8E4M3FN),
            Err(error) if error.to_string().contains("trailing dimension size 8 is not divisible by block size 3"),
        ));
        assert!(matches!(
            input.block_quantize(4, DataType::F16, DataType::F8E4M3FN),
            Err(error) if error.to_string().contains("'block_quantize' does not support element data type f16"),
        ));
        assert!(matches!(
            input.block_quantize(4, DataType::F4E2M1FN, DataType::F16),
            Err(error) if error.to_string().contains("'block_quantize' does not support scale data type f16"),
        ));
        let integer_input =
            Array::from_f64s(ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(8)])), vec![1.0; 8]);
        assert!(matches!(
            integer_input.block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN),
            Err(error) if error.to_string().contains("'block_quantize' expects an f32 or f64 input but got i32"),
        ));
        let scalar_input = Array::from_f64s(ArrayType::scalar(DataType::F32), vec![1.0]);
        assert!(matches!(
            scalar_input.block_quantize(1, DataType::F4E2M1FN, DataType::F8E4M3FN),
            Err(error) if error.to_string().contains("must have rank between 1 and 3 but got rank 0"),
        ));
    }

    #[test]
    fn test_block_quantize_mxfp8() {
        // OCP MX recipe: `f8e4m3fn` elements with power-of-two `f8e8m0fnu` scales,
        // `scale = 2^(floor(log2(max_abs)) - 8)`. The block maxima below sit exactly on powers of two (exercising
        // the boundary nudge in the `log2` composition) and every quotient is exactly representable in `f8e4m3fn`,
        // so quantization is exact; the all-zero block clamps its scale to `2^-127`.
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(8)]));
        let input = Array::from_f64s(
            input_type.clone(),
            vec![
                4.0, 2.0, 1.0, 0.5, 1.75, 0.5, -1.0, 0.25, // Blocks with scales 2^-6 and 2^-8.
                -8.0, 4.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, // Blocks with scale 2^-5 and the clamp floor.
            ],
        );
        let (elements, scales) = input.block_quantize(4, DataType::F8E4M3FN, DataType::F8E8M0FNU).unwrap();
        assert_eq!(
            elements.r#type().as_ref(),
            &ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(8)])),
        );
        assert_eq!(
            scales.r#type().as_ref(),
            &ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])),
        );
        assert_eq!(scales.to_f64s(), vec![(-6.0f64).exp2(), (-8.0f64).exp2(), (-5.0f64).exp2(), (-127.0f64).exp2()],);
        assert_eq!(
            elements.to_f64s(),
            vec![256.0, 128.0, 64.0, 32.0, 448.0, 128.0, -256.0, 64.0, -256.0, 128.0, 64.0, 32.0, 0.0, 0.0, 0.0, 0.0],
        );

        // Round trip: the quantized operands contract through `scaled_dot` to the exact full-precision dot.
        let dimensions = DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new());
        let product = elements
            .scaled_dot(&elements, Some(&scales), Some(&scales), Some(&dimensions), Some(DataType::F32))
            .unwrap();
        let expected = input.dot(&input, &DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new()));
        assert_eq!(product.to_f64s(), expected.to_f64s());
    }

    #[test]
    fn test_block_quantize_round_trip_tolerance() {
        // Values off the storage grids round-trip within the element type's quantization error: `f8e4m3fn` carries
        // three mantissa bits, so each dequantized element is within about 6% of its input (plus the OCP MX
        // saturation of a block maximum landing past the finite range) and the contraction of eight such products
        // stays within a proportional tolerance of the full-precision dot.
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1), Dimension::Static(8)]));
        let input = Array::from_f64s(input_type.clone(), vec![1.1, -2.3, 0.7, 3.9, 0.013, -0.27, 5.4, 8.9]);
        let (elements, scales) = input.block_quantize(4, DataType::F8E4M3FN, DataType::F8E8M0FNU).unwrap();
        assert_eq!(elements.r#type().shape(), input_type.shape());
        assert_eq!(
            scales.r#type().as_ref(),
            &ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(1), Dimension::Static(2)])),
        );
        let dimensions = DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new());
        let product = elements
            .scaled_dot(&elements, Some(&scales), Some(&scales), Some(&dimensions), Some(DataType::F32))
            .unwrap();
        let expected = input.dot(&input, &DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new()));
        let expected_value = expected.to_f64s()[0];
        let actual_value = product.to_f64s()[0];
        assert_abs_diff_eq!(actual_value, expected_value, epsilon = 0.05 * expected_value);

        // Rank-1 inputs quantize per block along their only dimension.
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));
        let vector = Array::from_f64s(vector_type.clone(), vec![1.1, -2.3, 0.7, 3.9, 0.013, -0.27, 5.4, 8.9]);
        let (vector_elements, vector_scales) =
            vector.block_quantize(4, DataType::F8E4M3FN, DataType::F8E8M0FNU).unwrap();
        assert_eq!(vector_elements.r#type().shape(), vector_type.shape());
        assert_eq!(
            vector_scales.r#type().as_ref(),
            &ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(2)])),
        );
        assert_eq!(vector_elements.to_f64s(), elements.to_f64s());
        assert_eq!(vector_scales.to_f64s(), scales.to_f64s());
    }

    #[test]
    fn test_block_quantize_stages_through_tracers() {
        // The composition also covers staging values: quantizing a tracer stages the recipe's operations and the
        // staged outputs carry the quantized element and scale types.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let builder = context.builder().clone();
        let input_atom = builder
            .borrow_mut()
            .add_input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(8)])));
        let input = context.tracer(input_atom, None);
        let (elements, scales) = input.block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN).unwrap();
        assert_eq!(
            elements.r#type().as_ref(),
            &ArrayType::new(DataType::F4E2M1FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(8)])),
        );
        assert_eq!(
            scales.r#type().as_ref(),
            &ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])),
        );
    }
}
