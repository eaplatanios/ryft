//! Block-scaled (i.e., microscaling) quantization through [`BlockQuantize`], [`BlockDequantize`], [`ScaledDot`], and
//! [`BlockScaledDot`]. [`BlockQuantize`] splits a floating-point array into an array of narrow elements and an array
//! of block scales, [`BlockDequantize`] reconstructs their values at a chosen precision, and [`ScaledDotOperation`]
//! contracts such element and scale pairs as a generalized dot product of their dequantized values. Together they
//! cover the NVFP4 format and the [Open Compute Project Microscaling Formats](
//! https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
//! (e.g., MXFP8) that low precision matrix units consume natively.
//!
//! [`BlockQuantize`] and [`BlockDequantize`] compose existing operations and inherit their transform rules.
//! [`ScaledDotOperation`] supports eager arrays, staged programs, array IR projections, partial evaluation, and
//! batching, and it intentionally has no differentiation rule. The XLA backend lowers it to cuDNN's block-scaled
//! matrix product on CUDA where cuDNN supports the call and to the `xla.scaled_dot` composite otherwise.
//! [`block_scaled_dot`] creates the differentiable training entry point, the analogue of `jax.nn.scaled_dot_general`,
//! which quantizes unquantized inputs in its primal and implements both the JAX gradient policies and the NVFP4
//! training recipe of NVIDIA (i.e., current global scaling, stochastic gradient rounding, random Hadamard transforms
//! of the weight gradient inputs, and 16×16 weight scale tiles).
//!
//! # Example
//!
//! ```rust
//! # use ryft_core::{Array, BlockQuantize, DataType, DotDimensionNumbers, ProgramError, ScaledDot};
//! # fn main() -> Result<(), ProgramError> {
//! let input = Array::matrix(1, 8, vec![3.0f32, 1.5, 0.5, 6.0, 0.5, 1.0, 0.25, 1.5])?;
//! let (elements, scales) = input.block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN)?;
//! assert_eq!(scales.to_f64s(), vec![1.0, 0.25]);
//! let dimensions = DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new());
//! let product = elements.scaled_dot(&elements, Some(&scales), Some(&scales), Some(&dimensions), Some(DataType::F32))?;
//! assert_eq!(product.to_f64s(), vec![51.0625]);
//! # Ok(())
//! # }
//! ```

use std::fmt::Display;

use ryft_macros::{Parameterized, capability};

use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrType, ArrayIrValue, ArrayMemberType,
    ArrayType, DataType, Dimension, DimensionType, Shape,
};
use crate::axes::{Axes, Axis};
use crate::batching::{
    BatchAxis, BatchableOperation, BatchingContext, BatchingDriver, BatchingError, InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain, EagerContext, StagingContext};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDriver, DifferentiationDual,
    DifferentiationError, DifferentiationPolicy,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver, MemberInterpretableOperation};
use crate::macros::{check_count, impl_non_transposable_operation};
use crate::operations::Capability;
use crate::operations::arithmetic::{Abs, Add, Div, Mul, Rem, Sub};
use crate::operations::assertions::Assert;
use crate::operations::collectives::parallel_vary::ManualVariationAlignment;
use crate::operations::comparisons::{Compare, ComparisonDirection};
use crate::operations::composites::builders::{Composite, CompositeCallContext, composite};
use crate::operations::composites::configurations::{CompositeVjp, DefaultJvp, SymbolicZeroVjp};
use crate::operations::composites::rules::definitions::CustomRuleDefinition;
use crate::operations::constants::constant::DimensionConstant;
use crate::operations::constants::fill::Fill;
use crate::operations::constants::iota::Iota;
use crate::operations::control_flow::select::Select;
use crate::operations::dimensions::dimension_max::DimensionMax;
use crate::operations::dimensions::dimension_size::DimensionSize;
use crate::operations::dot::{Dot, DotDimensionNumbers, infer_dot_output_type, validate_common_mapped_dimension};
use crate::operations::extrema::Clamp;
use crate::operations::logical::And;
use crate::operations::manipulation::broadcasting::{Broadcast, DynamicBroadcast};
use crate::operations::manipulation::conversions::ConvertElementType;
use crate::operations::manipulation::reshaping::{DynamicReshape, Reshape};
use crate::operations::manipulation::transposition::Transpose;
use crate::operations::random::{Random, RandomAlgorithm};
use crate::operations::reductions::{Reduce, ReductionKind};
use crate::operations::rounding::{Floor, Round};
use crate::parameters::Parameter;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    EmptyRegionDriver, MaybeZero, Operation, OperationFormatter, ProgramError, RegionInterface, TypeError, Typed,
    Value, ValueDomainDispatch, ValueProjection,
};
use crate::tracing::{DomainTracer, DomainTracingContext};

/// Value-level block quantization capability. [`BlockQuantize::block_quantize`] splits an `f32` or `f64` array into
/// an array of narrow elements and an array of scales with one scale per block of consecutive values along the last
/// dimension, which are the inputs that [`ScaledDot`] consumes. This supports quantizing values on the fly (e.g., a KV
/// cache) without a dedicated primitive, because the recipe composes existing operations and inherits their transform
/// rules. Both sides of a [`ScaledDot`] contraction quantized this way must contract their last dimension, which
/// [`ScaledDot::scaled_matmul`] does, while the default dimension numbers of [`ScaledDot::scaled_dot`] contract the
/// penultimate dimension of the right input. [`BlockQuantize::block_quantize_with_options`] additionally places blocks
/// along other axes, scales the input by a global scale first, and rounds stochastically. Its blocks may span several
/// axes (e.g., the 16×16 weight tiles of the NVFP4 training recipe of NVIDIA), with one scale per tile.
///
/// The scale data type selects one of two standard microscaling recipes:
///
///   - **`f8e4m3fn` Scales (i.e., the NVFP4 recipe of NVIDIA):** Set the scale of each block to `max_abs(block)
///     / element_max`, where `element_max` is the largest finite magnitude of the element type (e.g., `6.0` for
///     `f4e2m1fn`), so that the largest element of each block quantizes to the top of the element grid. Scales are
///     clamped to `[2^-9, 448]`, which spans the positive finite `f8e4m3fn` values, including subnormal ones. JAX
///     does not clamp from below, so blocks whose scales round to zero there divide by zero, while here they use
///     the smallest scale.
///   - **`f8e8m0fnu` Scales (i.e., the [Open Compute Project (OCP) Microscaling (MX)](
///     https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) recipe; e.g., MXFP8 and
///     MXFP4):** Set the shared scale of each block to the power of two `2^(floor(log2(max_abs(block))) - emax)`, where
///     `emax` is the largest exponent of the element type (e.g., `2` for `f4e2m1fn`, `8` for `f8e4m3fn`, and `15` for
///     `f8e5m2`), so that the largest element of each block lands in the top binade of the element type. Scales are
///     clamped to `[2^-126, 2^127]`. The upper bound is the largest `f8e8m0fnu` value, and the lower bound is the
///     smallest normal `f32` value rather than the smallest `f8e8m0fnu` value `2^-127`, so that a backend that
///     flushes subnormal values to zero cannot produce a zero scale, which `f8e8m0fnu` cannot represent.
///
/// Both recipes divide each input value by the scale of its block after narrowing that scale to the scale type, which
/// is exactly the scale that [`ScaledDotOperation`] dequantizes with. They then clamp the quotients to the finite range
/// of the element type before converting them, so that dequantization reproduces the input up to the quantization
/// error of the element type and no recipe relies on the overflow behavior of a floating-point conversion. The OCP MX
/// specification prescribes that clamp for the quotients between the largest finite magnitude and `2^(emax + 1)`, and
/// in the NVFP4 recipe it absorbs the rounding of the scales. Blocks whose values are all zero (or tiny) and blocks
/// whose values are huge use the smallest and largest scale, respectively, instead of a zero or non-finite scale.
/// Inputs must not contain NaN values, for which the results are unspecified and may differ across backends.
///
/// The power of two `2^floor(log2(max_abs(block)))` of the MX recipe keeps only the exponent bits of the block maximum,
/// so it is exact for every block maximum, including those just below a power of two. The `quantize` helper of JAX
/// instead rounds `max_abs(block) / element_max` up to a power of two.
///
/// Both recipes round their scales and elements to the grids of the narrow types arithmetically (i.e., with exact
/// power of two spacings derived from exponent bits) before converting them, so that every conversion is exact. This
/// keeps quantization intact under compilers that fold a conversion to a narrow type into a following conversion back,
/// as the XLA backend does on GPUs, where JAX places optimization barriers for the same reason.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and array IR values implement it for [`ArrayIrType`].
#[capability(projection(ArrayIrType => ArrayType))]
pub trait BlockQuantize<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Quantizes `self` into a pair `(elements, scales)`, with one scale per block of `block_size` consecutive values
    /// along the last dimension. `elements` has the shape of `self` and element type `element_type`, while `scales` has
    /// the shape of `self` with its last extent divided by `block_size` and element type `scale_type`. Refer to the
    /// documentation of [`BlockQuantize`] for the exact recipes and to [`Self::block_quantize_with_options`] for the
    /// requirements and errors.
    ///
    /// # Parameters
    ///
    ///   - `block_size`: Number of consecutive values along the last dimension that share one scale.
    ///   - `element_type`: Data type of the quantized elements.
    ///   - `scale_type`: Data type of the scales, which selects the quantization recipe.
    #[inline]
    fn block_quantize(
        &self,
        block_size: usize,
        element_type: DataType,
        scale_type: DataType,
    ) -> Result<(Self, Self), ProgramError> {
        self.block_quantize_with_options(block_size, element_type, scale_type, -1, None, None)
    }

    /// Quantizes `self` into a pair `(elements, scales)` like [`Self::block_quantize`], with blocks of `block_size`
    /// consecutive values along each of `block_axes`. A block therefore spans `block_size` values along every block
    /// axis (e.g., two block axes form `block_size × block_size` tiles). `elements` has the shape of `self`, while
    /// `scales` has the shape of `self` with the extent of every block axis divided by `block_size`.
    ///
    /// A `global_scale` divides `self` before the block scales are computed, as in the NVFP4 recipe, which scales
    /// tensors both globally and per block, and as with the `global_scale` of the JAX `quantize` helper, so the
    /// dequantized value of an element is `element * scale * global_scale`. A `rounding_noise` array of uniform samples
    /// in `[0, 1)` (e.g., drawn with [`Random::random_uniform`]) replaces rounding to the nearest value (with ties to
    /// even) by stochastic rounding. Each scaled value `x` then rounds to `floor(x / ulp + noise) · ulp`, where `ulp`
    /// is the spacing of the element type grid at `x`, so that it rounds up with probability equal to its distance from
    /// the grid value below it, in units of `ulp`, and quantization is unbiased within the finite range of the element
    /// type up to the rounding of the noise addition at the compute type. Results past the largest finite magnitude of
    /// the element type saturate to it. The spacing is computed exactly from the exponent bits of `x`.
    ///
    /// `self` must be an `f32` or `f64` array with a static shape of rank at least one, `block_size` must be at least
    /// two, `block_axes` must name at least one distinct axis, and the extent of every block axis must be a positive
    /// multiple of `block_size`, because [`ScaledDot`] requires non-empty scales whose blocks have at least two values
    /// along each contracting dimension. A `global_scale` must be a rank-0 array of the data type of `self` and
    /// requires `f8e4m3fn` scales, because power of two scales already span the range of their element types (JAX
    /// likewise rejects global scales for MX formats). A `rounding_noise` array must have the data type and shape
    /// of `self`. Returns a [`ProgramError`] if these requirements are not met, if `element_type` is not one of
    /// `f4e2m1fn`, `f8e4m3fn`, and `f8e5m2`, if `scale_type` is not one of `f8e4m3fn` and `f8e8m0fnu`, or if a
    /// composed operation fails.
    ///
    /// # Parameters
    ///
    ///   - `block_size`: Number of consecutive values along each block axis that share one scale.
    ///   - `element_type`: Data type of the quantized elements.
    ///   - `scale_type`: Data type of the scales, which selects the quantization recipe.
    ///   - `block_axes`: Axes along which blocks extend. Negative axes count backward from the input rank.
    ///   - `global_scale`: Optional rank-0 scale that divides `self` before block scaling.
    ///   - `rounding_noise`: Optional uniform samples in `[0, 1)` that select stochastic rounding.
    fn block_quantize_with_options<A: Into<Axes>>(
        &self,
        block_size: usize,
        element_type: DataType,
        scale_type: DataType,
        block_axes: A,
        global_scale: Option<&Self>,
        rounding_noise: Option<&Self>,
    ) -> Result<(Self, Self), ProgramError>;
}

impl<V> BlockQuantize<ArrayType> for V
where
    V: Value<Type = ArrayType>
        + Abs
        + Add
        + And
        + Broadcast
        + Clamp
        + ConvertElementType
        + Div
        + Floor
        + Mul
        + Reduce
        + Reshape
        + Round,
    V::Domain: Fill<f64, V>,
{
    fn block_quantize_with_options<A: Into<Axes>>(
        &self,
        block_size: usize,
        element_type: DataType,
        scale_type: DataType,
        block_axes: A,
        global_scale: Option<&Self>,
        rounding_noise: Option<&Self>,
    ) -> Result<(Self, Self), ProgramError> {
        // Every value with the capabilities used by the recipe, which include both the concrete reference `Array`
        // backend and the transform tracers, quantizes through this composition. All validation happens before the
        // first operation, so that rejected requests stage nothing.
        let element_format = BlockElementFormat::new(element_type)?;
        let (smallest_scale, largest_scale) = match scale_type {
            DataType::F8E4M3FN => ((-9.0f64).exp2(), 448.0),
            DataType::F8E8M0FNU => ((-126.0f64).exp2(), 127.0f64.exp2()),
            scale_type => {
                return Err(TypeError::invalid(format!(
                    "`block_quantize` does not support scale data type `{scale_type}`",
                ))
                .into());
            }
        };

        let input_type = self.r#type().into_owned();
        let compute_type = input_type.data_type();
        if !matches!(compute_type, DataType::F32 | DataType::F64) {
            return Err(TypeError::invalid(format!(
                "`block_quantize` expects an `f32` or `f64` input but got `{compute_type}`",
            ))
            .into());
        }

        let Some(shape) = input_type.static_shape() else {
            return Err(TypeError::invalid("`block_quantize` input must have a static shape").into());
        };

        let dimensions = shape.dimensions().to_vec();
        if dimensions.is_empty() {
            return Err(TypeError::invalid("`block_quantize` does not support rank-0 inputs").into());
        }

        if block_size < 2 {
            return Err(TypeError::invalid(format!(
                "`block_quantize` block size must be at least 2 but got {block_size}"
            ))
            .into());
        }

        let block_axes = block_axes.into().normalize(dimensions.len())?;
        if block_axes.is_empty() {
            return Err(TypeError::invalid("`block_quantize` requires at least one block axis").into());
        }

        for &axis in &block_axes {
            let size = dimensions[axis];
            if size == 0 || size % block_size != 0 {
                return Err(TypeError::invalid(format!(
                    "`block_quantize` block dimension {axis} size {size} is not a positive multiple of block size \
                     {block_size}",
                ))
                .into());
            }
        }

        if let Some(global_scale) = global_scale {
            if scale_type != DataType::F8E4M3FN {
                return Err(TypeError::invalid(format!(
                    "`block_quantize` global scales require `f8e4m3fn` scales but got `{scale_type}` scales",
                ))
                .into());
            }

            let global_scale_type = global_scale.r#type();
            if global_scale_type.rank() != 0 || global_scale_type.data_type() != compute_type {
                return Err(TypeError::invalid(format!(
                    "`block_quantize` global scale must be a rank-0 `{compute_type}` array but got \
                     `{global_scale_type}`",
                ))
                .into());
            }
        }

        if let Some(rounding_noise) = rounding_noise {
            let noise_type = rounding_noise.r#type();
            if noise_type.data_type() != compute_type || noise_type.shape() != input_type.shape() {
                return Err(TypeError::invalid(format!(
                    "`block_quantize` rounding noise must have the data type and shape of the input `{input_type}` \
                     but got `{noise_type}`",
                ))
                .into());
            }
        }

        // Splitting every block axis into a pair of `(block_count, block_size)` dimensions keeps the row-major element
        // order, so each block is the set of values that share all block counts. Reducing the block size dimensions
        // yields one maximum per block.
        let mut scale_dimensions = Vec::with_capacity(dimensions.len());
        let mut block_dimensions = Vec::with_capacity(dimensions.len() + block_axes.len());
        let mut scale_axes = Vec::with_capacity(dimensions.len());
        let mut reduction_axes = Vec::with_capacity(block_axes.len());
        for (axis, &size) in dimensions.iter().enumerate() {
            scale_axes.push(block_dimensions.len());
            if block_axes.contains(&axis) {
                scale_dimensions.push(size / block_size);
                block_dimensions.push(Dimension::Static(size / block_size));
                reduction_axes.push(block_dimensions.len());
                block_dimensions.push(Dimension::Static(block_size));
            } else {
                scale_dimensions.push(size);
                block_dimensions.push(Dimension::Static(size));
            }
        }

        let scale_value_type = ArrayType::new(
            compute_type,
            Shape::new(scale_dimensions.iter().map(|&size| Dimension::Static(size)).collect()),
        );

        let element_value_type = ArrayType::new(compute_type, input_type.shape().clone());
        let domain = self.domain();

        // A global scale divides the input before anything else, so the block scales describe the globally scaled
        // values. The scalar is broadcast with its own type, keeping its placement, and the division aligns it.
        let input = match global_scale {
            Some(global_scale) => {
                let mut expanded_type = global_scale.r#type().into_owned();
                for (axis, &size) in dimensions.iter().enumerate() {
                    expanded_type = expanded_type.with_inserted_dimension(axis, Dimension::Static(size))?;
                }
                self.div(&global_scale.broadcast(expanded_type, &[])?)?
            }
            None => self.clone(),
        };

        // The maximum magnitude of each block determines its scale. Both recipes compute their scales and elements on
        // the grids of the narrow types, so that every conversion to a narrow type is exact. Compilers that fold a
        // conversion to a narrow type into a following conversion back (e.g., XLA's floating-point conversion
        // simplification on GPUs) therefore cannot change the result, because the rounding is explicit.
        let block_max =
            input.reshape(Shape::new(block_dimensions))?.abs()?.reduce(&reduction_axes, ReductionKind::Max)?;
        let stored_scales = if scale_type == DataType::F8E4M3FN {
            // NVFP4 linear scaling maps the block maximum to the largest finite magnitude of the element type,
            // and the clamped scale rounds to the nearest `f8e4m3fn` value.
            let scales = block_max.div(&domain.fill(&scale_value_type, element_format.max)?)?.clamp(
                &domain.fill(&ArrayType::scalar(compute_type), smallest_scale)?,
                &domain.fill(&ArrayType::scalar(compute_type), largest_scale)?,
            )?;
            round_to_floating_point_grid(&scales, -6, 3, None, &|value| domain.fill(&scale_value_type, value))?
        } else {
            // OCP MX power of two scaling computes `2^(floor(log2(max_abs)) - emax)` exactly from the exponent bits of
            // the block maximum. The clamp maps blocks whose maxima are zero or subnormal to the smallest scale and
            // blocks whose maxima are not finite to the largest scale.
            binade_scale(&block_max, &|value| domain.fill(&scale_value_type, value))?
                .mul(&domain.fill(&scale_value_type, (-f64::from(element_format.max_exponent)).exp2())?)?
                .clamp(
                    &domain.fill(&ArrayType::scalar(compute_type), smallest_scale)?,
                    &domain.fill(&ArrayType::scalar(compute_type), largest_scale)?,
                )?
        };
        let scales = stored_scales.convert_element_type(scale_type)?;

        // The elements are divided by the stored scales, which are exactly the values that `ScaledDot` dequantizes
        // with. The scales derive from the input, so the expanded scales keep its memory space, placement, and manual
        // variation and only add replicated block dimensions, because broadcasting cannot change manual variation.
        let mut expanded_type = stored_scales.r#type().into_owned();
        for &axis in &reduction_axes {
            expanded_type = expanded_type.with_inserted_dimension(axis, Dimension::Static(block_size))?;
        }
        let expanded_scales =
            stored_scales.broadcast(expanded_type, scale_axes.as_slice())?.reshape(input_type.shape().clone())?;
        let quotients = input.div(&expanded_scales)?;

        // The quotients are clamped to the finite range of the element type before rounding, which saturates infinite
        // values rather than dividing infinite spacings, and the largest magnitude is on the grid, so rounding to the
        // nearest value keeps the clamped quotients in range. Stochastic rounding can still step past the largest
        // magnitude when adding noise just below one rounds up to the next integer, so its result is clamped again.
        let quotients = quotients.clamp(
            &domain.fill(&ArrayType::scalar(compute_type), -element_format.max)?,
            &domain.fill(&ArrayType::scalar(compute_type), element_format.max)?,
        )?;
        let elements = round_to_floating_point_grid(
            &quotients,
            element_format.min_exponent,
            element_format.mantissa_bits,
            rounding_noise,
            &|value| domain.fill(&element_value_type, value),
        )?;
        let elements = match rounding_noise {
            Some(_) => elements.clamp(
                &domain.fill(&ArrayType::scalar(compute_type), -element_format.max)?,
                &domain.fill(&ArrayType::scalar(compute_type), element_format.max)?,
            )?,
            None => elements,
        };

        Ok((elements.convert_element_type(element_type)?, scales))
    }
}

/// Value-level block dequantization capability. [`BlockDequantize::block_dequantize`] reconstructs an array from
/// elements and scales by repeating each scale over its block and multiplying at a chosen floating-point precision.
/// It accepts the pairs produced by [`BlockQuantize`] using the final axis as the block axis, and it also supports
/// blocks along several axes, as used by [`ScaledDot`]. An axis with equal element and scale extents is left out of
/// the block axes, allowing per-element scaling and scalar inputs with an empty axis list.
///
/// Homogeneous array values require static shapes. Array IR values compose first-class dimension operations directly
/// and support dynamic extents with runtime assertions. Use the parent array IR value when a projected array view
/// has dynamic extents. The composition inherits the transform rules of its component operations and differentiates
/// both elements and scales where their data types are differentiable.
///
/// Blanket implementations derive this capability from conversion, shape, and multiplication capabilities, together
/// with dimension and assertion capabilities for array IR values. Values covered by a blanket implementation cannot
/// supply another implementation of this capability.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and array IR values implement it for [`ArrayIrType`].
///
/// # Example
///
/// ```rust
/// # use ryft_core::{Array, BlockDequantize, BlockQuantize, DataType, ProgramError};
/// # fn main() -> Result<(), ProgramError> {
/// let input = Array::vector(vec![3.0f32, 1.5, 0.5, 6.0])?;
/// let (elements, scales) = input.block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN)?;
/// assert_eq!(elements.block_dequantize(&scales, [-1], DataType::F32)?, input);
/// # Ok(())
/// # }
/// ```
#[capability]
pub trait BlockDequantize<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Repeats each scale over its block, converts both `self` and the expanded scales to `data_type`, and multiplies
    /// them. The output has the shape of `self`. Inputs must have equal ranks and real numeric or Boolean element
    /// types. Non-block dimensions must match exactly. Along each block axis the scale extent must be positive and
    /// divide the element extent with a quotient of at least two, as in [`BlockQuantize`]. Known violations are
    /// rejected before composing operations, and array IR values assert unknown ratios at runtime. Empty non-block
    /// dimensions are supported. Homogeneous values, including projected array views, require static shapes. Use the
    /// parent array IR value for dynamic extents. Returns a [`ProgramError`] for invalid inputs, axes, output
    /// precision, or unsupported shapes, or when a component operation fails.
    ///
    /// # Parameters
    ///
    ///   - `scales`: Scale array with one value per block of elements in `self`.
    ///   - `block_axes`: Axes along which scales are repeated. Negative axes count backward from the input rank.
    ///     Duplicate axes are rejected. An empty list requests elementwise scaling of equal-shaped inputs.
    ///   - `data_type`: Floating-point data type of the multiplication and output. Conversion occurs before
    ///     multiplication, so this choice controls both rounding and output precision. Scales outside the range of
    ///     `data_type` therefore become infinite or zero even when their products with the elements are representable
    ///     (e.g., `f16` turns `f8e8m0fnu` scales of at least `2^16` into infinity and scales below `2^-24` into zero),
    ///     which matches the CPU lowering of the JAX `scaled_matmul` primitive. Use a type with the exponent range of
    ///     the scales, such as `bf16` or `f32`, for such scales.
    fn block_dequantize<A: Into<Axes>>(
        &self,
        scales: &Self,
        block_axes: A,
        data_type: DataType,
    ) -> Result<Self, ProgramError>;
}

impl<V: Value<Type = ArrayType> + Broadcast + ConvertElementType + Mul + Reshape> BlockDequantize<ArrayType> for V {
    fn block_dequantize<A: Into<Axes>>(
        &self,
        scales: &Self,
        block_axes: A,
        data_type: DataType,
    ) -> Result<Self, ProgramError> {
        // Homogeneous values compose the static shape operations directly. Keep the scale's placement and memory
        // through expansion, inserting only replicated block dimensions before reshaping to the element shape.
        let element_type = self.r#type();
        let scale_type = scales.r#type();
        let block_axes = validate_block_dequantize_inputs(&element_type, &scale_type, block_axes.into(), data_type)?;
        let (Some(element_shape), Some(scale_shape)) = (element_type.static_shape(), scale_type.static_shape()) else {
            return Err(TypeError::invalid(
                "`block_dequantize` homogeneous inputs must have static shapes; use array IR values \
                 for dynamic extents",
            )
            .into());
        };
        let mut expanded_type = scale_type.into_owned();
        let mut output_axes = Vec::with_capacity(element_type.rank());
        let mut inserted_dimensions = 0;
        for axis in 0..element_type.rank() {
            let output_axis = axis + inserted_dimensions;
            output_axes.push(output_axis);
            if block_axes.contains(&axis) {
                let block_size = element_shape.dimensions()[axis] / scale_shape.dimensions()[axis];
                expanded_type =
                    expanded_type.with_inserted_dimension(output_axis + 1, Dimension::Static(block_size))?;
                inserted_dimensions += 1;
            }
        }
        let elements = self.convert_element_type(data_type)?;
        let expanded_scales = scales
            .broadcast(expanded_type, &output_axes)?
            .reshape(element_type.shape().clone())?
            .convert_element_type(data_type)?;
        elements.mul(&expanded_scales)
    }
}

impl<V> BlockDequantize<ArrayIrType> for V
where
    V: Value<Type = ArrayIrType>
        + Assert
        + DimensionSize
        + DynamicBroadcast
        + DynamicReshape
        + ValueProjection<ArrayType>
        + ValueProjection<DimensionType>,
    V::Domain: Context<Type = ArrayIrType> + DimensionConstant,
    <V as ValueProjection<ArrayType>>::Projected: Value<Type = ArrayType> + ConvertElementType + Mul,
    <V as ValueProjection<DimensionType>>::Projected:
        Value<Type = DimensionType> + Compare<V> + DimensionMax + Div + Rem,
{
    fn block_dequantize<A: Into<Axes>>(
        &self,
        scales: &Self,
        block_axes: A,
        data_type: DataType,
    ) -> Result<Self, ProgramError> {
        // Array IR values keep shape computations in the parent universe. This composition also serves eager
        // ArrayIrValue arrays, whose dimension values execute as checked host integers.
        let element_type = self.r#type();
        let element_type = <&ArrayType>::try_from(element_type.as_ref())?;
        let scale_type = scales.r#type();
        let scale_type = <&ArrayType>::try_from(scale_type.as_ref())?;
        let block_axes = validate_block_dequantize_inputs(element_type, scale_type, block_axes.into(), data_type)?;
        let context = self.domain();
        let element_dimensions =
            (0..element_type.rank()).map(|axis| self.dimension_size(axis)).collect::<Result<Vec<_>, _>>()?;
        let elements = ValueProjection::<ArrayType>::into_projected(self.clone())?.convert_element_type(data_type)?;
        let scale_dimensions =
            (0..scale_type.rank()).map(|axis| scales.dimension_size(axis)).collect::<Result<Vec<_>, _>>()?;
        let mut expanded_dimensions = Vec::with_capacity(scale_type.rank() + block_axes.len());
        let mut output_axes = Vec::with_capacity(scale_type.rank());
        for (axis, scale_dimension) in scale_dimensions.iter().enumerate() {
            output_axes.push(expanded_dimensions.len());
            expanded_dimensions.push(scale_dimension.clone());
            if block_axes.contains(&axis) {
                // Divide by `max(scale, 1)` so that empty scales report the positivity assertion instead of a failed
                // remainder or division. Each assertion precedes the shape operations that require the ratio.
                let element_extent =
                    ValueProjection::<DimensionType>::into_projected(element_dimensions[axis].clone())?;
                let scale_extent = ValueProjection::<DimensionType>::into_projected(scale_dimension.clone())?;
                let zero = ValueProjection::<DimensionType>::into_projected(context.dimension_constant(0)?)?;
                let one = ValueProjection::<DimensionType>::into_projected(context.dimension_constant(1)?)?;
                scale_extent.compare(&zero, ComparisonDirection::GreaterThan)?.assert(
                    "scale extent must be positive",
                    &[("scale", ValueProjection::<DimensionType>::from_projected(scale_extent.clone()))],
                )?;
                let divisor = scale_extent.dimension_max(&one)?;
                element_extent.rem(&divisor)?.compare(&zero, ComparisonDirection::Equal)?.assert(
                    "element extent must be divisible by scale extent",
                    &[
                        ("elements", ValueProjection::<DimensionType>::from_projected(element_extent.clone())),
                        ("scale", ValueProjection::<DimensionType>::from_projected(scale_extent)),
                    ],
                )?;
                let ratio = element_extent.div(&divisor)?;
                let two = ValueProjection::<DimensionType>::into_projected(context.dimension_constant(2)?)?;
                two.compare(&ratio, ComparisonDirection::LessThanOrEqual)?.assert(
                    "block ratio must be at least two",
                    &[("ratio", ValueProjection::<DimensionType>::from_projected(ratio.clone()))],
                )?;
                expanded_dimensions.push(ValueProjection::<DimensionType>::from_projected(ratio));
            }
        }
        let expanded_scales =
            scales.dynamic_broadcast(&expanded_dimensions, &output_axes)?.dynamic_reshape(&element_dimensions)?;
        let expanded_scales =
            ValueProjection::<ArrayType>::into_projected(expanded_scales)?.convert_element_type(data_type)?;
        Ok(ValueProjection::<ArrayType>::from_projected(elements.mul(&expanded_scales)?))
    }
}

// TODO(eaplatanios): Review from here onwards.

/// Canonical operation name for [`ScaledDotOperation`].
pub const SCALED_DOT_OPERATION_NAME: &str = "scaled_dot";

/// Primitive representing a generalized block-scaled dot product operation. Its first two inputs are the left and right
/// element arrays. The optional left and right scale arrays follow in that order, and [`Self::has_lhs_scale`] and
/// [`Self::has_rhs_scale`] record which of them are present. Element and scale arrays may have any real numeric or
/// Boolean element type, and all inputs must have the same rank and matching varying manual axes. Unreduced inputs are
/// rejected.
///
/// Each scale array has the same rank as its element array. Its non-contracting dimensions match the element array
/// exactly, and the extent of every contracting element dimension is an integer multiple of at least two of the
/// corresponding scale extent. That ratio is the block size, and it is inferred independently for every side and
/// contracting dimension. Type inference validates statically known ratios, and the array IR composition described
/// below asserts the ratios of dynamic extents at runtime.
///
/// Semantically, this operation expands each present scale to the shape of its element array by repeating every scale
/// value over its block, converts the elements and expanded scales to `bf16`, and multiplies them. It then contracts
/// the resulting arrays with a [`DotOperation`](crate::DotOperation) that uses [`Self::dimensions`] and accumulates
/// at [`Self::accumulation_type`], which is also the output element type. It is the accumulation type of
/// [`DotOperation::with_accumulation_type`](crate::DotOperation::with_accumulation_type) and corresponds to the
/// `preferred_element_type` of the JAX `scaled_matmul` primitive and of the `xla.scaled_dot` composite. An absent
/// scale leaves its element array unchanged apart from the `bf16` conversion. The [`MemberInterpretableOperation`]
/// implementation evaluates this composition using the parent array IR domain's dimension values. Operation enums
/// reach this operation through their projected array members and never dispatch to that rule, so its callers invoke it
/// directly. Eager arrays use it, and the XLA backend traces it as the decomposition of the `xla.scaled_dot` composite
/// that it lowers this operation to whenever it does not use cuDNN's block-scaled matrix product. Native block-scaled
/// matrix products (e.g., that of cuDNN) may accumulate at a higher precision than [`Self::accumulation_type`] and may
/// ignore the signs of scales, so they match this composition up to accumulation rounding and only for the
/// non-negative finite scales that [`BlockQuantize`] produces.
///
/// Partial evaluation folds the operation when all inputs are known and stages it otherwise. Batching moves every
/// mapped axis to the front and materializes replicated inputs along a new leading axis, because each scale array
/// must keep the rank of its element array. The lifted operation pairs those leading axes as an additional batching
/// dimension, so repeated batching applies the same lift again without a rank ceiling.
///
/// This operation has no differentiation rule. Its scales are quantization metadata rather than differentiable
/// parameters, and treating them as constants would silently define a straight-through estimator that is not part of
/// the operation's semantics. The `scaled_matmul` primitive of JAX likewise has no differentiation rule. To train with
/// block-scaled formats, use [`block_scaled_dot`], which quantizes unquantized inputs and defines an explicit
/// gradient policy, like `jax.nn.scaled_dot_general`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ScaledDotOperation {
    /// Contracting and batching dimension specification.
    dimensions: DotDimensionNumbers,

    /// Data type of the contraction accumulator and the output.
    accumulation_type: DataType,

    /// Whether the operation has a left scale input.
    has_lhs_scale: bool,

    /// Whether the operation has a right scale input.
    has_rhs_scale: bool,
}

impl ScaledDotOperation {
    /// Creates a new [`ScaledDotOperation`].
    ///
    /// # Parameters
    ///
    ///   - `dimensions`: Contracting and batching dimensions of the generalized dot product.
    ///   - `accumulation_type`: Data type of the contraction accumulator and the output. Type inference validates
    ///     that the `bf16` dequantized inputs can accumulate at this type.
    ///   - `has_lhs_scale`: Whether the inputs include a left scale array after the two element arrays.
    ///   - `has_rhs_scale`: Whether the inputs include a right scale array after the optional left scale array.
    #[inline]
    pub fn new(
        dimensions: DotDimensionNumbers,
        accumulation_type: DataType,
        has_lhs_scale: bool,
        has_rhs_scale: bool,
    ) -> Self {
        Self { dimensions, accumulation_type, has_lhs_scale, has_rhs_scale }
    }

    /// Returns the default dimension numbers that [`ScaledDot::scaled_dot`] uses for element arrays of rank `rank`.
    /// They contract the last left dimension with the penultimate right dimension and pair every preceding dimension
    /// as a batching dimension. For example, rank-two inputs describe the matrix product `[M, K] × [K, N] → [M, N]`
    /// and rank-three inputs describe the batched matrix product `[B, M, K] × [B, K, N] → [B, M, N]`. Returns a
    /// [`TypeError`] when `rank` is smaller than two.
    pub fn default_dimensions(rank: usize) -> Result<DotDimensionNumbers, TypeError> {
        if rank < 2 {
            return Err(TypeError::invalid(format!(
                "default `{SCALED_DOT_OPERATION_NAME}` dimension numbers require inputs of rank at least 2 but got \
                 rank {rank}",
            )));
        }
        Ok(DotDimensionNumbers::new(vec![rank - 1], vec![rank - 2], (0..rank - 2).collect(), (0..rank - 2).collect()))
    }

    /// Returns the contracting and batching dimension specification of this [`ScaledDotOperation`].
    #[inline]
    pub fn dimensions(&self) -> &DotDimensionNumbers {
        &self.dimensions
    }

    /// Returns the data type of the contraction accumulator and the output of this [`ScaledDotOperation`].
    #[inline]
    pub fn accumulation_type(&self) -> DataType {
        self.accumulation_type
    }

    /// Returns whether this [`ScaledDotOperation`] has a left scale input.
    #[inline]
    pub fn has_lhs_scale(&self) -> bool {
        self.has_lhs_scale
    }

    /// Returns whether this [`ScaledDotOperation`] has a right scale input.
    #[inline]
    pub fn has_rhs_scale(&self) -> bool {
        self.has_rhs_scale
    }

    /// Returns the number of inputs of this [`ScaledDotOperation`].
    #[inline]
    fn input_count(&self) -> usize {
        2 + usize::from(self.has_lhs_scale) + usize::from(self.has_rhs_scale)
    }

    /// Splits `inputs` into the left and right element inputs followed by the optional left and right scale inputs.
    fn split_inputs<'o, Input>(
        &self,
        inputs: &'o [Input],
    ) -> Result<(&'o Input, &'o Input, Option<&'o Input>, Option<&'o Input>), TypeError> {
        if inputs.len() != self.input_count() {
            return Err(TypeError::invalid(format!(
                "`{SCALED_DOT_OPERATION_NAME}` expects {} inputs but got {}",
                self.input_count(),
                inputs.len(),
            )));
        }
        let lhs_scale = self.has_lhs_scale.then(|| &inputs[2]);
        let rhs_scale = self.has_rhs_scale.then(|| &inputs[inputs.len() - 1]);
        Ok((&inputs[0], &inputs[1], lhs_scale, rhs_scale))
    }
}

impl Display for ScaledDotOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ScaledDotOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        SCALED_DOT_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        let (lhs, rhs, lhs_scale, rhs_scale) = self.split_inputs(input_types)?;
        ArrayType::check_matching_manual_variation(SCALED_DOT_OPERATION_NAME, &input_types.iter().collect::<Vec<_>>())?;
        let rank = lhs.rank();
        if std::iter::once(rhs).chain(lhs_scale).chain(rhs_scale).any(|input_type| input_type.rank() != rank) {
            return Err(TypeError::invalid(format!("`{SCALED_DOT_OPERATION_NAME}` inputs must have the same rank")));
        }
        for (index, input_type) in input_types.iter().enumerate() {
            let data_type = input_type.data_type();
            // Complex inputs are rejected because the `bf16` dequantization would silently drop their imaginary
            // parts.
            if !data_type.is_real() && data_type != DataType::Boolean {
                return Err(TypeError::invalid(format!(
                    "`{SCALED_DOT_OPERATION_NAME}` input {index} must have a real numeric or Boolean element type but \
                     got `{data_type}`",
                )));
            }
            if !input_type.unreduced_axes().is_empty() {
                return Err(TypeError::invalid(format!(
                    "`{SCALED_DOT_OPERATION_NAME}` does not support unreduced inputs",
                )));
            }
        }

        // The contraction is a dot of the `bf16` dequantized element arrays, so the dot type inference rule validates
        // the dimension numbers and placements and determines the output type.
        let output_type = infer_dot_output_type(
            SCALED_DOT_OPERATION_NAME,
            &lhs.clone().with_data_type(DataType::BF16),
            &rhs.clone().with_data_type(DataType::BF16),
            &self.dimensions,
            Some(self.accumulation_type),
            None,
        )?;

        // Each present scale matches its element array along non-contracting dimensions. Along contracting dimensions,
        // static extents must have an integer ratio of at least two. Ratios that involve a dynamic extent are asserted
        // at runtime, except that two identical extents can never have such a ratio.
        for (side, input_type, scale_type, contracting_dimensions) in [
            ("left", lhs, lhs_scale, self.dimensions.lhs_contracting_dimensions()),
            ("right", rhs, rhs_scale, self.dimensions.rhs_contracting_dimensions()),
        ] {
            let Some(scale_type) = scale_type else { continue };
            for axis in 0..rank {
                let input_dimension = input_type.dimension(axis);
                let scale_dimension = scale_type.dimension(axis);
                if !contracting_dimensions.contains(&axis) {
                    if input_dimension != scale_dimension {
                        return Err(TypeError::invalid(format!(
                            "`{SCALED_DOT_OPERATION_NAME}` {side} dimension {axis} has size `{input_dimension}` but \
                             its scale has size `{scale_dimension}`",
                        )));
                    }
                    continue;
                }
                match (input_dimension.value(), scale_dimension.value()) {
                    (_, Some(0)) => {
                        return Err(TypeError::invalid(format!(
                            "`{SCALED_DOT_OPERATION_NAME}` {side} contracting dimension {axis} has an empty scale, \
                             but scale sizes must be positive",
                        )));
                    }
                    (Some(input_size), Some(scale_size)) if input_size % scale_size != 0 => {
                        return Err(TypeError::invalid(format!(
                            "`{SCALED_DOT_OPERATION_NAME}` {side} contracting dimension {axis} has size {input_size}, \
                             which is not divisible by its scale size {scale_size}",
                        )));
                    }
                    (Some(input_size), Some(scale_size)) if input_size / scale_size < 2 => {
                        return Err(TypeError::invalid(format!(
                            "`{SCALED_DOT_OPERATION_NAME}` {side} contracting dimension {axis} must be at least twice \
                             as large as its scale but has ratio {}",
                            input_size / scale_size,
                        )));
                    }
                    _ if input_dimension == scale_dimension => {
                        return Err(TypeError::invalid(format!(
                            "`{SCALED_DOT_OPERATION_NAME}` {side} contracting dimension {axis} must be at least twice \
                             as large as its scale but both have size `{input_dimension}`",
                        )));
                    }
                    _ => {}
                }
            }
        }
        Ok(vec![output_type])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("dimensions", &self.dimensions)?;
            operation.field("accumulation_type", &self.accumulation_type)?;
            operation.field("lhs_scale", &self.has_lhs_scale)?;
            operation.field("rhs_scale", &self.has_rhs_scale)
        })
    }
}

impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for ScaledDotOperation
where
    C::Value: ScaledDot,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        let (lhs, rhs, lhs_scale, rhs_scale) = self.split_inputs(inputs)?;
        Ok(vec![lhs.scaled_dot(rhs, lhs_scale, rhs_scale, Some(&self.dimensions), Some(self.accumulation_type))?])
    }
}

impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for ScaledDotOperation where
    C::Operation: From<ScaledDotOperation>
{
}

impl<C: Context<Type = ArrayType>, P: ArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>>
    for ScaledDotOperation
where
    ScaledDotOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        if inputs.len() != self.input_count() {
            return Err(ProgramError::InvalidInputCount { expected: self.input_count(), actual: inputs.len() }.into());
        }
        self.dimensions.validate(
            SCALED_DOT_OPERATION_NAME,
            inputs[0].unbatched_type().rank(),
            inputs[1].unbatched_type().rank(),
        )?;
        if inputs.iter().all(|input| input.batch_axis().is_replicated()) {
            return self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()]);
        }
        validate_common_mapped_dimension(SCALED_DOT_OPERATION_NAME, inputs)?;

        // Every scale array must keep the rank of its element array, so every input moves its mapped axis to the front
        // or, when it is replicated, materializes one there through the active policy, which keeps a dynamic mapped
        // extent a first-class value. The lifted dimension numbers then pair those leading axes as a new batching
        // dimension and shift the existing dimension numbers past them.
        let aligned_inputs = inputs
            .iter()
            .map(|input| P::match_axis(context, input, Axis::from(0)))
            .collect::<Result<Vec<_>, _>>()?;
        let (dimensions, output_axis) = self.dimensions.batched(Some(0), Some(0)).unwrap();
        let lifted_operation = Self::new(dimensions, self.accumulation_type, self.has_lhs_scale, self.has_rhs_scale);
        lifted_operation.interpret_with_batch_axes(
            context,
            aligned_inputs.as_slice(),
            &[BatchAxis::from_optional_position(output_axis)],
        )
    }
}

impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for ScaledDotOperation
where
    C::Operation: From<ScaledDotOperation>,
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        _context: &DifferentiationContext<C, P>,
        _driver: &D,
        _inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // Refer to the documentation of `ScaledDotOperation` for why this operation has no differentiation rule.
        // Pairing its outputs with zero tangents instead would silently treat the scales as constants.
        Err(ProgramError::UnsupportedOperation {
            message: format!("`{SCALED_DOT_OPERATION_NAME}` does not support differentiation"),
        }
        .into())
    }
}

impl_non_transposable_operation!(ScaledDotOperation);

impl<C: Domain<Type = ArrayIrType>> MemberInterpretableOperation<C> for ScaledDotOperation
where
    C::Value: BlockDequantize + ValueProjection<ArrayType>,
    <C::Value as ValueProjection<ArrayType>>::Projected: Value<Type = ArrayType> + ConvertElementType + Dot,
{
    fn interpret_in_parent<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        // The array boundary stays homogeneous, but its composition needs the parent domain's dimension values for
        // runtime extent reads and ratio assertions. This rule is shared by eager arrays and XLA decomposition traces.
        let (lhs, rhs, lhs_scale, rhs_scale) = self.split_inputs(inputs)?;
        let input_types = inputs
            .iter()
            .map(|value| Ok(<&ArrayType>::try_from(value.r#type().as_ref())?.clone()))
            .collect::<Result<Vec<_>, TypeError>>()?;
        self.infer_output_types(&input_types, &[])?;

        let lhs = match lhs_scale {
            Some(scale) => ValueProjection::<ArrayType>::into_projected(lhs.block_dequantize(
                scale,
                self.dimensions.lhs_contracting_dimensions(),
                DataType::BF16,
            )?)?,
            None => ValueProjection::<ArrayType>::into_projected(lhs.clone())?.convert_element_type(DataType::BF16)?,
        };
        let rhs = match rhs_scale {
            Some(scale) => ValueProjection::<ArrayType>::into_projected(rhs.block_dequantize(
                scale,
                self.dimensions.rhs_contracting_dimensions(),
                DataType::BF16,
            )?)?,
            None => ValueProjection::<ArrayType>::into_projected(rhs.clone())?.convert_element_type(DataType::BF16)?,
        };
        let output = if self.accumulation_type == DataType::BF16 {
            lhs.dot(&rhs, &self.dimensions)?
        } else {
            lhs.dot_with_accumulation_type(&rhs, &self.dimensions, self.accumulation_type)?
        };
        Ok(vec![ValueProjection::<ArrayType>::from_projected(output)])
    }
}

/// Value-level generalized block-scaled dot product capability. [`ScaledDot`] is the receiver-style entry point for
/// staging or executing [`ScaledDotOperation`]s, which contract element arrays after scaling every block of their
/// contracting dimensions by the corresponding scale value. [`BlockQuantize`] produces element and scale arrays in the
/// layout that this capability consumes.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and array IR values implement it for [`ArrayIrType`].
#[capability(projection(ArrayIrType => ArrayType))]
pub trait ScaledDot<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Computes the generalized dot product `(self * lhs_scale) · (rhs * rhs_scale)`, where each scale is repeated over
    /// its blocks before the multiplication and an absent scale is the multiplicative identity. Refer to
    /// [`ScaledDotOperation`] for the exact semantics and input constraints. Returns a [`ProgramError`] if the inputs
    /// are incompatible with each other or with `dimensions`, or if the operation cannot be recorded in the context of
    /// the value.
    ///
    /// # Parameters
    ///
    ///   - `rhs`: Right element array.
    ///   - `lhs_scale`: Optional scale array of the left element array (i.e., of `self`).
    ///   - `rhs_scale`: Optional scale array of the right element array.
    ///   - `dimensions`: Optional dimension numbers of the generalized dot product. They default to
    ///     [`ScaledDotOperation::default_dimensions`] for the rank of `self`.
    ///   - `accumulation_type`: Optional data type of the contraction accumulator and the output, which defaults to
    ///     `f32` so that the contraction of the `bf16` dequantized inputs does not lose precision. That is the default
    ///     of `jax.nn.scaled_matmul`, whereas `jax.lax.scaled_dot` defaults to `bf16`.
    fn scaled_dot(
        &self,
        rhs: &Self,
        lhs_scale: Option<&Self>,
        rhs_scale: Option<&Self>,
        dimensions: Option<&DotDimensionNumbers>,
        accumulation_type: Option<DataType>,
    ) -> Result<Self, ProgramError>;

    /// Computes the rank-three block-scaled batched matrix product `[B, M, K] × [B, N, K] → [B, M, N]` of `self` and
    /// `rhs`, where both inputs contract their last dimension and pair their first dimension as a batching dimension.
    /// Unlike [`Self::scaled_dot`], both scales are required. Returns a [`ProgramError`] if any input does not have
    /// rank three or if [`Self::scaled_dot`] fails.
    #[inline]
    fn scaled_matmul(
        &self,
        rhs: &Self,
        lhs_scale: &Self,
        rhs_scale: &Self,
        accumulation_type: Option<DataType>,
    ) -> Result<Self, ProgramError>
    where
        Self: Typed<Type: ArrayMemberType>,
    {
        for value in [self, rhs, lhs_scale, rhs_scale] {
            if value.r#type().as_array_type()?.rank() != 3 {
                return Err(TypeError::invalid("`scaled_matmul` expects rank-3 inputs").into());
            }
        }
        self.scaled_dot(
            rhs,
            Some(lhs_scale),
            Some(rhs_scale),
            Some(&DotDimensionNumbers::new(vec![2], vec![2], vec![0], vec![0])),
            accumulation_type,
        )
    }
}

impl ScaledDot for Array {
    fn scaled_dot(
        &self,
        rhs: &Self,
        lhs_scale: Option<&Self>,
        rhs_scale: Option<&Self>,
        dimensions: Option<&DotDimensionNumbers>,
        accumulation_type: Option<DataType>,
    ) -> Result<Self, ProgramError> {
        // Lift concrete arrays into the array IR so that the eager interpretation rule can read their extents as
        // dimension values and validate the inputs before dequantizing and contracting them.
        let dimensions = match dimensions {
            Some(dimensions) => dimensions.clone(),
            None => ScaledDotOperation::default_dimensions(self.r#type().rank())?,
        };
        let accumulation_type = accumulation_type.unwrap_or(DataType::F32);
        let operation =
            ScaledDotOperation::new(dimensions, accumulation_type, lhs_scale.is_some(), rhs_scale.is_some());
        let inputs = std::iter::once(self)
            .chain(std::iter::once(rhs))
            .chain(lhs_scale)
            .chain(rhs_scale)
            .cloned()
            .map(ArrayIrValue::from)
            .collect::<Vec<_>>();
        let mut outputs =
            operation.interpret_in_parent(&EagerContext::<ArrayIrValue<Array>>::new(), &EmptyRegionDriver, &inputs)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(ValueProjection::<ArrayType>::into_projected(outputs.remove(0))?)
    }
}

impl<V: Value<Type = ArrayType, Dispatch = ValueDomainDispatch> + ManualVariationAlignment<ArrayType>>
    ScaledDot<ArrayType> for V
where
    V::Domain: Context<Operation: From<ScaledDotOperation>>,
{
    fn scaled_dot(
        &self,
        rhs: &Self,
        lhs_scale: Option<&Self>,
        rhs_scale: Option<&Self>,
        dimensions: Option<&DotDimensionNumbers>,
        accumulation_type: Option<DataType>,
    ) -> Result<Self, ProgramError> {
        // Values that carry a context stage a scaled dot through that context. The `ValueDomainDispatch` marker keeps
        // this implementation disjoint from eager values, which implement the capability directly.
        let dimensions = match dimensions {
            Some(dimensions) => dimensions.clone(),
            None => ScaledDotOperation::default_dimensions(self.r#type().rank())?,
        };
        let accumulation_type = accumulation_type.unwrap_or(DataType::F32);
        let mut inputs = vec![self.clone(), rhs.clone()];
        inputs.extend(lhs_scale.cloned());
        inputs.extend(rhs_scale.cloned());
        let inputs = ManualVariationAlignment::align_manual_variation(&inputs)?;
        let operation =
            ScaledDotOperation::new(dimensions, accumulation_type, lhs_scale.is_some(), rhs_scale.is_some());
        let mut outputs = self.domain().bind(operation, Vec::new(), inputs.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Element and scale formats of one input of a block-scaled matrix product, which [`BlockScaledDotConfiguration`]
/// assigns to the left input, the right input, and the gradients. It is the analogue of the `BlockScaleConfig` of
/// JAX, except that global scales are inputs of [`BlockScaledDot::call`] or computed by it rather than part of the
/// format. The scale data type selects the [`BlockQuantize`] recipe, and formats with `f8e4m3fn` scales (e.g., NVFP4)
/// use a global scale in addition to their block scales.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BlockScaleFormat {
    /// Number of consecutive values along the contracting dimension that share one scale.
    block_size: usize,

    /// Data type of the quantized elements.
    element_type: DataType,

    /// Data type of the block scales.
    scale_type: DataType,
}

impl BlockScaleFormat {
    /// Creates a new [`BlockScaleFormat`]. [`block_scaled_dot`] validates the format against the requirements of
    /// [`BlockQuantize::block_quantize_with_options`].
    #[inline]
    pub const fn new(block_size: usize, element_type: DataType, scale_type: DataType) -> Self {
        Self { block_size, element_type, scale_type }
    }

    /// Returns the NVFP4 format of NVIDIA, which has blocks of 16 `f4e2m1fn` elements with `f8e4m3fn` scales.
    #[inline]
    pub const fn nvfp4() -> Self {
        Self::new(16, DataType::F4E2M1FN, DataType::F8E4M3FN)
    }

    /// Returns the OCP MXFP8 format, which has blocks of 32 `f8e4m3fn` elements with `f8e8m0fnu` scales.
    #[inline]
    pub const fn mxfp8() -> Self {
        Self::new(32, DataType::F8E4M3FN, DataType::F8E8M0FNU)
    }

    /// Returns the OCP MXFP4 format, which has blocks of 32 `f4e2m1fn` elements with `f8e8m0fnu` scales.
    #[inline]
    pub const fn mxfp4() -> Self {
        Self::new(32, DataType::F4E2M1FN, DataType::F8E8M0FNU)
    }

    /// Returns the number of consecutive values along the contracting dimension that share one scale.
    #[inline]
    pub const fn block_size(&self) -> usize {
        self.block_size
    }

    /// Returns the data type of the quantized elements.
    #[inline]
    pub const fn element_type(&self) -> DataType {
        self.element_type
    }

    /// Returns the data type of the block scales.
    #[inline]
    pub const fn scale_type(&self) -> DataType {
        self.scale_type
    }

    /// Returns whether this format uses a global scale in addition to its block scales (i.e., whether it has
    /// `f8e4m3fn` scales), which corresponds to the `nvfp4` mode of JAX.
    #[inline]
    pub fn has_global_scale(&self) -> bool {
        self.scale_type == DataType::F8E4M3FN
    }

    /// Returns the largest dequantized magnitude of a value that does not clip when scaled by a unit global scale
    /// (i.e., the product of the largest finite element and scale magnitudes).
    fn largest_unclipped_value(&self) -> Result<f64, TypeError> {
        let scale_max = if self.has_global_scale() { 448.0 } else { 127.0f64.exp2() };
        Ok(BlockElementFormat::new(self.element_type)?.max * scale_max)
    }

    /// Validates this format against the requirements of block quantization that do not depend on input shapes.
    fn validate(&self, name: &str) -> Result<(), ProgramError> {
        if BlockElementFormat::new(self.element_type).is_err() {
            return Err(TypeError::invalid(format!(
                "`block_scaled_dot` {name} format does not support element data type `{}`",
                self.element_type,
            ))
            .into());
        }
        if !matches!(self.scale_type, DataType::F8E4M3FN | DataType::F8E8M0FNU) {
            return Err(TypeError::invalid(format!(
                "`block_scaled_dot` {name} format does not support scale data type `{}`",
                self.scale_type,
            ))
            .into());
        }
        if self.block_size < 2 {
            return Err(TypeError::invalid(format!(
                "`block_scaled_dot` {name} format block size must be at least 2 but got {}",
                self.block_size,
            ))
            .into());
        }
        Ok(())
    }
}

/// Source of the global scales of [`BlockScaleFormat`]s that use global scales.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum BlockScaledGlobalScaling {
    /// Global scales are optional inputs of [`BlockScaledDot::call`], and absent ones are `1`. This matches the
    /// `global_scale` of the JAX `BlockScaleConfig`, whose default is `1`. Absent scales suit inputs whose block
    /// maxima lie roughly within `[element_max · 2^-9, element_max · 448]` (e.g., `[6 · 2^-9, 6 · 448]` for NVFP4),
    /// where `element_max` is the largest finite magnitude of the element type. In particular, quantized NVFP4
    /// backward contractions usually need a gradient global scale (or [current](Self::Current) scaling), because
    /// gradients are often much smaller and the smallest block scale flushes them to zero.
    #[default]
    Provided,

    /// Every quantized tensor computes its global scale from its own current maximum magnitude `max_abs` as
    /// `max_abs / (element_max · 448)`, so that the largest value maps to the largest finite element and scale
    /// magnitudes. This is the default NVFP4 recipe of NVIDIA Transformer Engine, which never clips. The maximum is
    /// local to each device shard inside manual regions, whereas Transformer Engine reduces it across the devices that
    /// share a tensor (e.g., a tensor parallel group), so use provided scales where the shards must agree.
    Current,
}

/// Value-independent semantics of a [`BlockScaledDot`], which [`block_scaled_dot`] creates. The defaults follow
/// `jax.nn.scaled_dot_general` and [`Self::nvfp4_training`] selects the NVFP4 training recipe of NVIDIA.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BlockScaledDotConfiguration {
    /// Contracting and batching dimension specification.
    dimensions: DotDimensionNumbers,

    /// Data type of the output (refer to [`Self::with_accumulation_type`]).
    accumulation_type: DataType,

    /// Format of the left input.
    lhs_format: BlockScaleFormat,

    /// Format of the right input.
    rhs_format: BlockScaleFormat,

    /// Format of the output cotangents in quantized backward contractions.
    gradient_format: BlockScaleFormat,

    /// Source of the global scales of formats that use global scales.
    global_scaling: BlockScaledGlobalScaling,

    /// Whether the backward contractions are quantized, or `None` to follow JAX.
    quantized_backward: Option<bool>,

    /// Whether quantized gradients round stochastically.
    stochastic_gradient_rounding: bool,

    /// Whether the weight gradient contraction applies a random Hadamard transform to its inputs.
    hadamard_transform: bool,

    /// Whether the right input uses square tiles of scales.
    tiled_rhs_scales: bool,
}

impl BlockScaledDotConfiguration {
    /// Creates a new [`BlockScaledDotConfiguration`] with the JAX defaults. Every input and the gradients use `format`,
    /// the output is `f32`, global scales are [provided](BlockScaledGlobalScaling::Provided), and the backward
    /// contractions are quantized unless the gradient format uses global scales (refer to
    /// [`Self::quantized_backward`]).
    ///
    /// # Parameters
    ///
    ///   - `dimensions`: Contracting and batching dimensions of the generalized dot product.
    ///   - `format`: Format of both inputs and of the gradients.
    #[inline]
    pub fn new(dimensions: DotDimensionNumbers, format: BlockScaleFormat) -> Self {
        Self {
            dimensions,
            accumulation_type: DataType::F32,
            lhs_format: format,
            rhs_format: format,
            gradient_format: format,
            global_scaling: BlockScaledGlobalScaling::Provided,
            quantized_backward: None,
            stochastic_gradient_rounding: false,
            hadamard_transform: false,
            tiled_rhs_scales: false,
        }
    }

    /// Creates the NVFP4 training recipe of NVIDIA Transformer Engine, which uses NVFP4 for all three contractions
    /// with [current](BlockScaledGlobalScaling::Current) global scaling, rounds gradients stochastically, applies the
    /// random Hadamard transform to the weight gradient inputs, and uses 16×16 scale tiles for the right input, which
    /// should be the weight (e.g., `y = x · Wᵀ` with `dimensions` contracting the last dimensions of `x` and `W`).
    ///
    /// The recipe follows Transformer Engine in which tensors it quantizes and how, but not bit for bit. Global scales
    /// divide the input as in JAX, whereas Transformer Engine multiplies by reciprocals, so roundings can differ.
    /// Stochastic rounding draws a uniform sample at the compute type per value rather than eight random bits, which
    /// keeps rounding unbiased with a finer probability resolution. Hadamard transforms run at the compute type rather
    /// than at `bf16`. Which layers keep higher precision (e.g., the last blocks of a model) is the caller's choice, as
    /// in Transformer Engine.
    #[inline]
    pub fn nvfp4_training(dimensions: DotDimensionNumbers) -> Self {
        Self::new(dimensions, BlockScaleFormat::nvfp4())
            .with_global_scaling(BlockScaledGlobalScaling::Current)
            .with_quantized_backward(true)
            .with_stochastic_gradient_rounding(true)
            .with_hadamard_transform(true)
            .with_tiled_rhs_scales(true)
    }

    /// Returns this configuration with the provided output data type, which must be one of `bf16`, `f16`, `f32`, and
    /// `f64`. Forward contractions accumulate at it when it is `bf16`, `f32`, or `f64` and the left format has no
    /// global scale. Otherwise, as for NVFP4, they accumulate at the compute type (i.e., `f64` when either input is
    /// `f64`, and `f32` otherwise) and convert to it. JAX accumulates at the output type and rejects `f64` outputs.
    #[inline]
    pub fn with_accumulation_type(mut self, accumulation_type: DataType) -> Self {
        self.accumulation_type = accumulation_type;
        self
    }

    /// Returns this configuration with the provided format of the left input.
    #[inline]
    pub fn with_lhs_format(mut self, format: BlockScaleFormat) -> Self {
        self.lhs_format = format;
        self
    }

    /// Returns this configuration with the provided format of the right input.
    #[inline]
    pub fn with_rhs_format(mut self, format: BlockScaleFormat) -> Self {
        self.rhs_format = format;
        self
    }

    /// Returns this configuration with the provided format of the output cotangents in quantized backward
    /// contractions.
    #[inline]
    pub fn with_gradient_format(mut self, format: BlockScaleFormat) -> Self {
        self.gradient_format = format;
        self
    }

    /// Returns this configuration with the provided source of global scales.
    #[inline]
    pub fn with_global_scaling(mut self, global_scaling: BlockScaledGlobalScaling) -> Self {
        self.global_scaling = global_scaling;
        self
    }

    /// Returns this configuration with quantized (`true`) or unquantized (`false`) backward contractions,
    /// overriding the JAX default (refer to [`Self::quantized_backward`]).
    #[inline]
    pub fn with_quantized_backward(mut self, quantized_backward: bool) -> Self {
        self.quantized_backward = Some(quantized_backward);
        self
    }

    /// Returns this configuration with stochastic (`true`) or nearest (`false`) rounding of the output
    /// cotangents in quantized backward contractions. Stochastic rounding requires
    /// [`BlockScaledDotInputs::random_state`].
    #[inline]
    pub fn with_stochastic_gradient_rounding(mut self, stochastic_gradient_rounding: bool) -> Self {
        self.stochastic_gradient_rounding = stochastic_gradient_rounding;
        self
    }

    /// Returns this configuration with or without the random Hadamard transform of the inputs of quantized weight
    /// gradient contractions (refer to [`Self::hadamard_transform`]).
    #[inline]
    pub fn with_hadamard_transform(mut self, hadamard_transform: bool) -> Self {
        self.hadamard_transform = hadamard_transform;
        self
    }

    /// Returns this configuration with square (`true`) or one-dimensional (`false`) scale blocks for the right input
    /// (refer to [`Self::tiled_rhs_scales`]).
    #[inline]
    pub fn with_tiled_rhs_scales(mut self, tiled_rhs_scales: bool) -> Self {
        self.tiled_rhs_scales = tiled_rhs_scales;
        self
    }

    /// Returns the contracting and batching dimension specification.
    #[inline]
    pub fn dimensions(&self) -> &DotDimensionNumbers {
        &self.dimensions
    }

    /// Returns the data type of the output.
    #[inline]
    pub fn accumulation_type(&self) -> DataType {
        self.accumulation_type
    }

    /// Returns the format of the left input.
    #[inline]
    pub fn lhs_format(&self) -> BlockScaleFormat {
        self.lhs_format
    }

    /// Returns the format of the right input.
    #[inline]
    pub fn rhs_format(&self) -> BlockScaleFormat {
        self.rhs_format
    }

    /// Returns the format of the output cotangents in quantized backward contractions.
    #[inline]
    pub fn gradient_format(&self) -> BlockScaleFormat {
        self.gradient_format
    }

    /// Returns the source of global scales.
    #[inline]
    pub fn global_scaling(&self) -> BlockScaledGlobalScaling {
        self.global_scaling
    }

    /// Returns whether the backward contractions are quantized. Unless overridden, they are quantized exactly when
    /// the gradient format does not use global scales, which is when `jax.nn.scaled_dot_general` quantizes them,
    /// so NVFP4 gradients default to unquantized backward contractions.
    #[inline]
    pub fn quantized_backward(&self) -> bool {
        self.quantized_backward.unwrap_or(!self.gradient_format.has_global_scale())
    }

    /// Returns whether quantized gradients round stochastically.
    #[inline]
    pub fn stochastic_gradient_rounding(&self) -> bool {
        self.stochastic_gradient_rounding
    }

    /// Returns whether quantized weight gradient contractions apply the random Hadamard transform of NVIDIA
    /// Transformer Engine to both of their inputs. The transform multiplies every 16 consecutive values along the
    /// contracting dimension by `diag(s) · H / 4`, where `H` is the 16×16 Sylvester Hadamard matrix and `s` is the
    /// fixed sign vector of Transformer Engine. The transform is orthogonal, so the exact product is unchanged, while
    /// outliers spread over their blocks before quantization. It can grow the largest magnitude of a tensor up to four
    /// times, so formats with global scales require [current](BlockScaledGlobalScaling::Current) global scaling, which
    /// computes the global scales of the transformed tensors, rather than provided scales that would clip them.
    #[inline]
    pub fn hadamard_transform(&self) -> bool {
        self.hadamard_transform
    }

    /// Returns whether the right input uses square tiles of `block_size × block_size` values that share one scale,
    /// spanning its (merged) contracting and free dimensions. Its quantization along either of them then yields the
    /// same values, so the forward contraction and the input gradient contraction see the same quantized weight.
    #[inline]
    pub fn tiled_rhs_scales(&self) -> bool {
        self.tiled_rhs_scales
    }

    /// Validates this configuration independently of the input types.
    fn validate(&self) -> Result<(), ProgramError> {
        self.lhs_format.validate("left")?;
        self.rhs_format.validate("right")?;
        self.gradient_format.validate("gradient")?;
        if !matches!(self.accumulation_type, DataType::BF16 | DataType::F16 | DataType::F32 | DataType::F64) {
            return Err(TypeError::invalid(format!(
                "`block_scaled_dot` accumulation type must be one of `bf16`, `f16`, `f32`, and `f64` but got `{}`",
                self.accumulation_type,
            ))
            .into());
        }
        if !self.quantized_backward()
            && (self.stochastic_gradient_rounding || self.hadamard_transform || self.tiled_rhs_scales)
        {
            return Err(TypeError::invalid(
                "`block_scaled_dot` stochastic gradient rounding, Hadamard transforms, and tiled right scales require \
                 quantized backward contractions",
            )
            .into());
        }
        if self.hadamard_transform
            && self.global_scaling == BlockScaledGlobalScaling::Provided
            && (self.lhs_format.has_global_scale() || self.gradient_format.has_global_scale())
        {
            return Err(TypeError::invalid(
                "`block_scaled_dot` Hadamard transforms of formats with global scales require current global scaling",
            )
            .into());
        }
        Ok(())
    }
}

/// Inputs of a [`BlockScaledDot`] call. Only the left and right inputs are differentiable, and the other leaves
/// receive structural zero cotangents.
#[derive(Clone, Debug, PartialEq, Parameterized)]
pub struct BlockScaledDotInputs<P: Parameter> {
    /// Left input of the generalized dot product.
    pub lhs: P,

    /// Right input of the generalized dot product.
    pub rhs: P,

    /// Optional rank-0 global scale of the left input (refer to [`BlockScaledGlobalScaling::Provided`]).
    pub lhs_global_scale: Option<P>,

    /// Optional rank-0 global scale of the right input (refer to [`BlockScaledGlobalScaling::Provided`]).
    pub rhs_global_scale: Option<P>,

    /// Optional rank-0 global scale of the output cotangents in quantized backward contractions (refer to
    /// [`BlockScaledGlobalScaling::Provided`]).
    pub gradient_global_scale: Option<P>,

    /// Random generator state (refer to [`Random`]) that seeds stochastic gradient rounding, which must be present
    /// exactly when [`BlockScaledDotConfiguration::stochastic_gradient_rounding`] is enabled. The call does not return
    /// an advanced state, so every call needs a fresh state (e.g., split from the caller's generator), or it repeats
    /// the same rounding noise.
    pub random_state: Option<P>,
}

impl<P: Parameter> BlockScaledDotInputs<P> {
    /// Creates block-scaled dot product inputs without global scales or a random generator state.
    #[inline]
    pub fn new(lhs: P, rhs: P) -> Self {
        Self {
            lhs,
            rhs,
            lhs_global_scale: None,
            rhs_global_scale: None,
            gradient_global_scale: None,
            random_state: None,
        }
    }

    /// Returns these inputs with the provided global scale of the left input.
    #[inline]
    pub fn with_lhs_global_scale(mut self, global_scale: P) -> Self {
        self.lhs_global_scale = Some(global_scale);
        self
    }

    /// Returns these inputs with the provided global scale of the right input.
    #[inline]
    pub fn with_rhs_global_scale(mut self, global_scale: P) -> Self {
        self.rhs_global_scale = Some(global_scale);
        self
    }

    /// Returns these inputs with the provided global scale of the output cotangents, which requires quantized backward
    /// contractions.
    #[inline]
    pub fn with_gradient_global_scale(mut self, global_scale: P) -> Self {
        self.gradient_global_scale = Some(global_scale);
        self
    }

    /// Returns these inputs with the provided random generator state.
    #[inline]
    pub fn with_random_state(mut self, random_state: P) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Returns the present leaves in field order.
    fn into_values(self) -> Vec<P> {
        [Some(self.lhs), Some(self.rhs), self.lhs_global_scale, self.rhs_global_scale, self.gradient_global_scale]
            .into_iter()
            .chain([self.random_state])
            .flatten()
            .collect()
    }

    /// Rebuilds inputs with the presence pattern of `self` from `values` in field order.
    fn with_values<Q: Parameter>(&self, values: Vec<Q>) -> BlockScaledDotInputs<Q> {
        let mut values = values.into_iter();
        let mut next_if = |present: bool| if present { values.next() } else { None };
        BlockScaledDotInputs {
            lhs: next_if(true).unwrap(),
            rhs: next_if(true).unwrap(),
            lhs_global_scale: next_if(self.lhs_global_scale.is_some()),
            rhs_global_scale: next_if(self.rhs_global_scale.is_some()),
            gradient_global_scale: next_if(self.gradient_global_scale.is_some()),
            random_state: next_if(self.random_state.is_some()),
        }
    }
}

/// Differentiable block-scaled generalized dot product, built by [`block_scaled_dot`]. It quantizes its
/// unquantized inputs with [`BlockQuantize`] and contracts them with [`ScaledDot`] in its primal, and its reverse
/// mode rule implements a quantization-aware gradient policy, which is how models train with block-scaled formats
/// such as NVFP4 and MXFP8. It is the analogue of `jax.nn.scaled_dot_general`, and its rules are retained, so create
/// it once and call it many times.
pub struct BlockScaledDot<D: Domain<Type = ArrayType>>
where
    DomainTracingContext<D>: StagingContext<Type = ArrayType, Constant = D::Constant, Operation = D::Operation>,
{
    /// Configuration that the rules of [`Self::function`] implement.
    configuration: BlockScaledDotConfiguration,

    /// Composite that stages the block-scaled dot product call.
    function: Composite<
        D::Constant,
        D::Operation,
        BlockScaledDotInputs<DomainTracer<D>>,
        DomainTracer<D>,
        DefaultJvp,
        SymbolicZeroVjp<
            D::Constant,
            D::Operation,
            BlockScaledDotInputs<DomainTracer<D>>,
            DomainTracer<D>,
            BlockScaledDotInputs<DomainTracer<D>>,
        >,
    >,
}

impl<D: Domain<Type = ArrayType>> BlockScaledDot<D>
where
    D::Constant: 'static,
    D::Operation: 'static,
    DomainTracingContext<D>: StagingContext<Type = ArrayType, Constant = D::Constant, Operation = D::Operation>,
{
    /// Returns the configuration of this [`BlockScaledDot`].
    #[inline]
    pub fn configuration(&self) -> &BlockScaledDotConfiguration {
        &self.configuration
    }

    /// Validates `inputs`, aligns their manual variation, and stages one block-scaled dot product call, whose output
    /// has the [accumulation type](BlockScaledDotConfiguration::accumulation_type) and the shape of the corresponding
    /// [`Dot`] output.
    ///
    /// The left and right inputs must be real floating-point arrays with static shapes and the extents that the
    /// dimension numbers require. They are quantized at `f64` when either is `f64`, and at `f32` otherwise, whereas
    /// JAX always quantizes at `f32`. The product of the contracting extents must be a positive multiple of the block
    /// sizes of both input formats, and tiled right scales additionally require that of the free extents of the right
    /// input. Quantized backward contractions block the products of the free extents of both inputs, so
    /// differentiation requires each of them to be a multiple of the gradient format's block size and of the block
    /// size of its own input's format (and of `16` for Hadamard transforms of the left free extents). Global scales
    /// must be rank-0 real floating-point arrays that are only present for formats with global scales under
    /// [`BlockScaledGlobalScaling::Provided`], and a gradient global scale additionally requires quantized backward
    /// contractions. A random generator state must be present exactly when stochastic gradient rounding is enabled
    /// and must have a generator state type (refer to [`RandomAlgorithm::from_state_type`]).
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if the inputs do not meet these requirements or if a composed operation fails while
    /// tracing or evaluating a rule.
    pub fn call<C, V>(&self, inputs: BlockScaledDotInputs<V>) -> Result<V, ProgramError>
    where
        C: CompositeCallContext<Type = ArrayType, Value = V, Constant = D::Constant, Operation = D::Operation>,
        V: Value<Type = ArrayType, Domain = C> + ManualVariationAlignment<ArrayType>,
        CustomRuleDefinition<C::Constant, C::Operation>: CompositeVjp<
                SymbolicZeroVjp<
                    D::Constant,
                    D::Operation,
                    BlockScaledDotInputs<DomainTracer<D>>,
                    DomainTracer<D>,
                    BlockScaledDotInputs<DomainTracer<D>>,
                >,
                DomainTracer<D>,
                BlockScaledDotInputs<DomainTracer<D>>,
                DomainTracer<D>,
            >,
    {
        // The global scales and the random state are validated before alignment, so that calls rejected here stage
        // nothing. The rules validate the shapes when the composite traces its primal, after alignment.
        let configuration = &self.configuration;
        let global_scales = [
            ("left", inputs.lhs_global_scale.as_ref(), configuration.lhs_format),
            ("right", inputs.rhs_global_scale.as_ref(), configuration.rhs_format),
            ("gradient", inputs.gradient_global_scale.as_ref(), configuration.gradient_format),
        ];
        for (name, global_scale, format) in global_scales {
            let Some(global_scale) = global_scale else { continue };
            if configuration.global_scaling != BlockScaledGlobalScaling::Provided || !format.has_global_scale() {
                return Err(TypeError::invalid(format!(
                    "`block_scaled_dot` {name} global scale requires provided global scaling and a format with \
                     `f8e4m3fn` scales",
                ))
                .into());
            }
            let r#type = global_scale.r#type();
            if r#type.rank() != 0 || !r#type.data_type().is_floating_point() || !r#type.data_type().is_real() {
                return Err(TypeError::invalid(format!(
                    "`block_scaled_dot` {name} global scale must be a rank-0 real floating-point array but got \
                     `{type}`",
                ))
                .into());
            }
        }
        let requires_random_state = configuration.stochastic_gradient_rounding;
        if inputs.random_state.is_some() != requires_random_state {
            return Err(TypeError::invalid(if requires_random_state {
                "`block_scaled_dot` stochastic gradient rounding requires a random generator state"
            } else {
                "`block_scaled_dot` random generator states require stochastic gradient rounding"
            })
            .into());
        }
        if let Some(random_state) = &inputs.random_state {
            RandomAlgorithm::from_state_type(&random_state.r#type())?;
        }
        if inputs.gradient_global_scale.is_some() && !configuration.quantized_backward() {
            return Err(TypeError::invalid(
                "`block_scaled_dot` gradient global scales require quantized backward contractions",
            )
            .into());
        }
        let aligned = ManualVariationAlignment::align_manual_variation(&inputs.clone().into_values())?;
        self.function.call(inputs.with_values(aligned))
    }
}

/// Creates a differentiable block-scaled generalized dot product (refer to [`BlockScaledDot`]), the analogue of
/// `jax.nn.scaled_dot_general`. Its primal quantizes both inputs along their contracting dimensions with the formats
/// of `configuration`, contracts them with [`ScaledDot`], and multiplies the result by the product of their global
/// scales. Its reverse mode rule saves the inputs at high precision and computes the input gradients with one of two
/// policies.
///
///   - **Unquantized backward contractions** (the JAX default for NVFP4 gradients) contract the output cotangent with
///     the other input at `f32` (or `f64`), so quantization acts as a straight-through estimator. When the gradient
///     format uses global scales (the condition of JAX), gradients of values of inputs whose formats use global scales
///     and that the forward quantization clipped (i.e., whose magnitudes exceed `global_scale · element_max · 448`)
///     are zero.
///   - **Quantized backward contractions** (the JAX default for MX gradients and the NVIDIA recipe) quantize the
///     output cotangent with the gradient format and the other input with its own format along the contracting
///     dimension of each gradient contraction, and contract them with [`ScaledDot`]. The configuration can add
///     stochastic gradient rounding, Hadamard transforms of the weight gradient inputs, and tiled right scales.
///
/// Like JAX, both policies first normalize every input to the rank-three layout `[batch, free, contracting]` by
/// transposing and merging its dimensions, so several contracting dimensions form one blocked dimension. That dimension
/// follows the listed order of the contracting dimensions, while JAX merges them in ascending axis order on each side
/// (which pairs mismatched dimensions when the two sides list them in different orders). Forward mode differentiation
/// is not supported, as with the `custom_vjp` of JAX. The function also differs from JAX where [`BlockQuantize`] does
/// (refer to its documentation), because MX scales round down to the OCP power of two rather than up and NVFP4 scales
/// never round to zero. JAX takes the clipping thresholds of both inputs from the left format, while this function
/// uses the format of each input. Backward contractions accumulate at the compute type, whereas JAX accumulates them
/// at the output type, and MX forward contractions with an output type at which [`ScaledDot`] cannot accumulate
/// (e.g., `f16`) accumulate at the compute type and convert afterwards.
///
/// # Example
///
/// ```rust
/// # use ryft_core::{
/// #     Array, ArrayOperation, BlockScaleFormat, BlockScaledDotConfiguration, BlockScaledDotInputs, DataType,
/// #     DotDimensionNumbers, EagerContext, Reduce, ReductionKind, Transpose, block_scaled_dot, differentiate_at,
/// # };
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // NVFP4 with blocks of four values. Both contracting blocks below contain `6`, so their scales are `1` and these
/// // grid values quantize exactly.
/// let format = BlockScaleFormat::new(4, DataType::F4E2M1FN, DataType::F8E4M3FN);
/// let configuration = BlockScaledDotConfiguration::new(DotDimensionNumbers::matmul(), format);
/// let function = block_scaled_dot::<EagerContext<Array, ArrayOperation<Array>>>(configuration)?;
/// let lhs = Array::matrix(1, 4, vec![6.0f32, 3.0, 1.5, 0.5])?;
/// let rhs = Array::matrix(4, 1, vec![1.0f32, 6.0, 2.0, 0.5])?;
/// let output = function.call(BlockScaledDotInputs::new(lhs.clone(), rhs.clone()))?;
/// assert_eq!(output.to_f64s(), vec![27.25]);
///
/// // NVFP4 gradients use unquantized backward contractions by default.
/// let (lhs_gradient, rhs_gradient) = differentiate_at((lhs.clone(), rhs.clone())).gradient(|(lhs, rhs)| {
///     function.call(BlockScaledDotInputs::new(lhs, rhs))?.reduce(&[0, 1], ReductionKind::Sum)
/// })?;
/// assert_eq!(lhs_gradient, rhs.transpose(vec![1, 0])?);
/// assert_eq!(rhs_gradient, lhs.transpose(vec![1, 0])?);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns a [`ProgramError`] for formats that [`BlockQuantize`] does not support, an accumulation type other than
/// `bf16`, `f16`, `f32`, and `f64`, quantized backward options without quantized backward contractions, or Hadamard
/// transforms of formats with global scales under [provided](BlockScaledGlobalScaling::Provided) global scaling.
pub fn block_scaled_dot<D>(configuration: BlockScaledDotConfiguration) -> Result<BlockScaledDot<D>, ProgramError>
where
    D: Domain<Type = ArrayType>,
    D::Constant: 'static,
    D::Operation: 'static,
    DomainTracer<D>: Abs
        + Add
        + Broadcast
        + BlockQuantize
        + Clamp
        + Compare
        + ConvertElementType
        + Div
        + Dot
        + Floor
        + Mul
        + Random
        + Reduce
        + Rem
        + Reshape
        + ScaledDot
        + Select
        + Sub
        + Transpose,
    DomainTracingContext<D>: StagingContext<Type = ArrayType, Constant = D::Constant, Operation = D::Operation>
        + Fill<f64, DomainTracer<D>>
        + Iota<DomainTracer<D>>,
{
    configuration.validate()?;
    let primal_configuration = configuration.clone();
    let forward_configuration = configuration.clone();
    let backward_configuration = configuration.clone();
    let primal = move |inputs: BlockScaledDotInputs<DomainTracer<D>>| -> Result<_, ProgramError> {
        BlockScaledDotRules::new(&primal_configuration, &inputs)?.forward(&inputs)
    };
    let forward = move |inputs: BlockScaledDotInputs<DomainTracer<D>>| -> Result<_, ProgramError> {
        let output = BlockScaledDotRules::new(&forward_configuration, &inputs)?.forward(&inputs)?;
        Ok((output, inputs))
    };
    let backward = move |inputs: BlockScaledDotInputs<DomainTracer<D>>,
                         seed: MaybeZero<DomainTracer<D>>|
          -> Result<_, ProgramError> {
        let zero = |value: &DomainTracer<D>| Ok::<_, ProgramError>(MaybeZero::Zero(value.r#type().cotangent()?));
        let mut cotangents = inputs.clone().into_values().iter().map(zero).collect::<Result<Vec<_>, _>>()?;
        if let MaybeZero::Value(seed) = seed {
            let (lhs_cotangent, rhs_cotangent) =
                BlockScaledDotRules::new(&backward_configuration, &inputs)?.backward(&inputs, &seed)?;
            cotangents[0] = MaybeZero::Value(lhs_cotangent);
            cotangents[1] = MaybeZero::Value(rhs_cotangent);
        }
        Ok(inputs.with_values(cotangents))
    };
    Ok(BlockScaledDot {
        configuration,
        function: composite(primal).with_symbolic_zero_vjp(forward, backward).with_name(BLOCK_SCALED_DOT_NAME),
    })
}

/// Name of the composite calls that [`BlockScaledDot`] stages.
const BLOCK_SCALED_DOT_NAME: &str = "block_scaled_dot";

/// Fixed sign vector of the random Hadamard transform of NVIDIA Transformer Engine.
const HADAMARD_SIGNS: [f64; 16] =
    [1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0];

/// Grid parameters of a block-scaled element data type.
#[derive(Copy, Clone, Debug, PartialEq)]
struct BlockElementFormat {
    /// Largest finite magnitude.
    max: f64,

    /// Exponent of the largest finite magnitude's binade (i.e., `floor(log2(max))`).
    max_exponent: i32,

    /// Exponent of the smallest normal magnitude, below which the grid spacing stays constant.
    min_exponent: i32,

    /// Number of explicitly stored mantissa bits.
    mantissa_bits: i32,
}

impl BlockElementFormat {
    /// Returns the grid parameters of `element_type`, or a [`TypeError`] for data types that block quantization does
    /// not support as elements.
    fn new(element_type: DataType) -> Result<Self, TypeError> {
        match element_type {
            DataType::F4E2M1FN => Ok(Self { max: 6.0, max_exponent: 2, min_exponent: 0, mantissa_bits: 1 }),
            DataType::F8E4M3FN => Ok(Self { max: 448.0, max_exponent: 8, min_exponent: -6, mantissa_bits: 3 }),
            DataType::F8E5M2 => Ok(Self { max: 57344.0, max_exponent: 15, min_exponent: -14, mantissa_bits: 2 }),
            element_type => {
                Err(TypeError::invalid(format!("`block_quantize` does not support element data type `{element_type}`")))
            }
        }
    }
}

/// Source of the global scale of one quantized tensor in the rules of a [`BlockScaledDot`].
#[derive(Clone)]
enum GlobalScale<V> {
    /// No global scale, for formats without global scales.
    None,

    /// Provided global scale (absent provided scales are `1`, i.e., [`GlobalScale::None`]).
    Provided(V),

    /// Global scale computed from the current maximum magnitude of the tensor.
    Current,
}

/// Shapes and permutations that normalize the inputs of a [`BlockScaledDot`] to the rank-three layout
/// `[batch, free, contracting]`, together with the rules that operate on that layout.
struct BlockScaledDotRules<'o> {
    /// Configuration of the rules.
    configuration: &'o BlockScaledDotConfiguration,

    /// Floating-point type at which inputs are quantized and unquantized contractions accumulate.
    compute_type: DataType,

    /// Permutation of the left input dimensions into batch, free, and contracting order.
    lhs_permutation: Vec<usize>,

    /// Permutation of the right input dimensions into batch, free, and contracting order.
    rhs_permutation: Vec<usize>,

    /// Static shape of the left input.
    lhs_shape: Vec<usize>,

    /// Static shape of the right input.
    rhs_shape: Vec<usize>,

    /// Rank-three shape `[batch, lhs free, contracting]` of the normalized left input.
    lhs_canonical_shape: [usize; 3],

    /// Rank-three shape `[batch, rhs free, contracting]` of the normalized right input.
    rhs_canonical_shape: [usize; 3],

    /// Static shape of the output, which lists the batch, left free, and right free dimensions.
    output_shape: Vec<usize>,
}

impl<'o> BlockScaledDotRules<'o> {
    /// Validates the shapes of `inputs` against `configuration` and computes their normalization.
    fn new<V: Value<Type = ArrayType>>(
        configuration: &'o BlockScaledDotConfiguration,
        inputs: &BlockScaledDotInputs<V>,
    ) -> Result<Self, ProgramError> {
        let lhs_type = inputs.lhs.r#type();
        let rhs_type = inputs.rhs.r#type();
        for (name, r#type) in [("left", &lhs_type), ("right", &rhs_type)] {
            if !r#type.data_type().is_floating_point() || !r#type.data_type().is_real() {
                return Err(TypeError::invalid(format!(
                    "`block_scaled_dot` {name} input must have a real floating-point element type but got `{}`",
                    r#type.data_type(),
                ))
                .into());
            }
        }
        let (Some(lhs_shape), Some(rhs_shape)) = (lhs_type.static_shape(), rhs_type.static_shape()) else {
            return Err(TypeError::invalid("`block_scaled_dot` inputs must have static shapes").into());
        };
        let lhs_shape = lhs_shape.dimensions().to_vec();
        let rhs_shape = rhs_shape.dimensions().to_vec();
        let compute_type = if lhs_type.data_type() == DataType::F64 || rhs_type.data_type() == DataType::F64 {
            DataType::F64
        } else {
            DataType::F32
        };

        // Dot type inference validates the dimension numbers and extents at the common compute type.
        let dimensions = &configuration.dimensions;
        let output_type = infer_dot_output_type(
            BLOCK_SCALED_DOT_NAME,
            &lhs_type.as_ref().clone().with_data_type(compute_type),
            &rhs_type.as_ref().clone().with_data_type(compute_type),
            dimensions,
            None,
            None,
        )?;
        let output_shape = output_type.static_shape().unwrap().dimensions().to_vec();
        let free_axes = |rank: usize, batching: &[usize], contracting: &[usize]| {
            (0..rank).filter(|axis| !batching.contains(axis) && !contracting.contains(axis)).collect::<Vec<_>>()
        };
        let lhs_free_axes =
            free_axes(lhs_shape.len(), dimensions.lhs_batching_dimensions(), dimensions.lhs_contracting_dimensions());
        let rhs_free_axes =
            free_axes(rhs_shape.len(), dimensions.rhs_batching_dimensions(), dimensions.rhs_contracting_dimensions());
        let permutation = |batching: &[usize], free: &[usize], contracting: &[usize]| {
            batching.iter().chain(free).chain(contracting).copied().collect::<Vec<_>>()
        };
        let lhs_permutation =
            permutation(dimensions.lhs_batching_dimensions(), &lhs_free_axes, dimensions.lhs_contracting_dimensions());
        let rhs_permutation =
            permutation(dimensions.rhs_batching_dimensions(), &rhs_free_axes, dimensions.rhs_contracting_dimensions());
        let product = |shape: &[usize], axes: &[usize]| axes.iter().map(|&axis| shape[axis]).product::<usize>();
        let batch = product(&lhs_shape, dimensions.lhs_batching_dimensions());
        let contracting = product(&lhs_shape, dimensions.lhs_contracting_dimensions());
        let lhs_free = product(&lhs_shape, &lhs_free_axes);
        let rhs_free = product(&rhs_shape, &rhs_free_axes);
        let rules = Self {
            configuration,
            compute_type,
            lhs_permutation,
            rhs_permutation,
            lhs_shape,
            rhs_shape,
            lhs_canonical_shape: [batch, lhs_free, contracting],
            rhs_canonical_shape: [batch, rhs_free, contracting],
            output_shape,
        };
        rules.validate_extents(false)?;
        Ok(rules)
    }

    /// Validates that every extent that the forward rule (or, if `backward` is set, the backward rule) blocks is a
    /// positive multiple of the block sizes that apply to it. Like JAX, only differentiation requires the backward
    /// extents.
    fn validate_extents(&self, backward: bool) -> Result<(), ProgramError> {
        let configuration = self.configuration;
        let [_, m, k] = self.lhs_canonical_shape;
        let [_, n, _] = self.rhs_canonical_shape;
        let mut requirements = vec![
            ("contracting", k, configuration.lhs_format.block_size),
            ("contracting", k, configuration.rhs_format.block_size),
        ];
        if backward && configuration.quantized_backward() {
            requirements.extend([
                ("right free", n, configuration.gradient_format.block_size),
                ("right free", n, configuration.rhs_format.block_size),
                ("left free", m, configuration.gradient_format.block_size),
                ("left free", m, configuration.lhs_format.block_size),
            ]);
            if configuration.hadamard_transform {
                requirements.push(("left free", m, 16));
            }
        }
        if configuration.tiled_rhs_scales {
            requirements.push(("right free", n, configuration.rhs_format.block_size));
        }
        for (name, extent, block_size) in requirements {
            if extent == 0 || extent % block_size != 0 {
                return Err(TypeError::invalid(format!(
                    "`block_scaled_dot` {name} extent {extent} is not a positive multiple of block size {block_size}",
                ))
                .into());
            }
        }
        Ok(())
    }

    /// Returns `value` converted to the compute type and normalized to the rank-three `[batch, free, contracting]`
    /// layout through `permutation`.
    fn normalize<V: ConvertElementType + Reshape + Transpose>(
        &self,
        value: &V,
        permutation: &[usize],
        shape: [usize; 3],
    ) -> Result<V, ProgramError> {
        let value = value.convert_element_type(self.compute_type)?;
        let value = if permutation.iter().enumerate().all(|(index, &axis)| index == axis) {
            value
        } else {
            value.transpose(permutation.to_vec())?
        };
        value.reshape(Shape::new(shape.iter().map(|&size| Dimension::Static(size)).collect()))
    }

    /// Restores a rank-three gradient in normalized layout to the layout of an input with `shape`, and converts it to
    /// `data_type`.
    fn denormalize<V: ConvertElementType + Reshape + Transpose>(
        &self,
        value: &V,
        permutation: &[usize],
        shape: &[usize],
        data_type: DataType,
    ) -> Result<V, ProgramError> {
        let permuted_shape = permutation.iter().map(|&axis| Dimension::Static(shape[axis])).collect();
        let value = value.reshape(Shape::new(permuted_shape))?;
        let value = if permutation.iter().enumerate().all(|(index, &axis)| index == axis) {
            value
        } else {
            let mut inverse = vec![0; permutation.len()];
            for (index, &axis) in permutation.iter().enumerate() {
                inverse[axis] = index;
            }
            value.transpose(inverse)?
        };
        value.convert_element_type(data_type)
    }

    /// Computes the primal output.
    fn forward<V>(&self, inputs: &BlockScaledDotInputs<V>) -> Result<V, ProgramError>
    where
        V: Value<Type = ArrayType>
            + Abs
            + Broadcast
            + BlockQuantize
            + Clamp
            + ConvertElementType
            + Div
            + Mul
            + Reduce
            + Reshape
            + ScaledDot
            + Transpose,
        V::Domain: Fill<f64, V>,
    {
        let configuration = self.configuration;
        let lhs = self.normalize(&inputs.lhs, &self.lhs_permutation, self.lhs_canonical_shape)?;
        let rhs = self.normalize(&inputs.rhs, &self.rhs_permutation, self.rhs_canonical_shape)?;
        let lhs_global_scale = self.global_scale(configuration.lhs_format, inputs.lhs_global_scale.as_ref())?;
        let rhs_global_scale = self.global_scale(configuration.rhs_format, inputs.rhs_global_scale.as_ref())?;
        let lhs = self.quantize(&lhs, configuration.lhs_format, 2, false, lhs_global_scale, None)?;
        let rhs =
            self.quantize(&rhs, configuration.rhs_format, 2, configuration.tiled_rhs_scales, rhs_global_scale, None)?;
        let dimensions = DotDimensionNumbers::new(vec![2], vec![2], vec![0], vec![0]);
        let output = self.contract(&lhs, &rhs, &dimensions, configuration.accumulation_type)?;
        let output_shape = Shape::new(self.output_shape.iter().map(|&size| Dimension::Static(size)).collect());
        output.reshape(output_shape)
    }

    /// Computes the cotangents of the left and right inputs for the output cotangent `seed`.
    fn backward<V>(&self, inputs: &BlockScaledDotInputs<V>, seed: &V) -> Result<(V, V), ProgramError>
    where
        V: Value<Type = ArrayType>
            + Abs
            + Add
            + Broadcast
            + BlockQuantize
            + Clamp
            + Compare
            + ConvertElementType
            + Div
            + Dot
            + Floor
            + Mul
            + Random
            + Reduce
            + Rem
            + Reshape
            + ScaledDot
            + Select
            + Sub
            + Transpose,
        V::Domain: Fill<f64, V> + Iota<V>,
    {
        self.validate_extents(true)?;
        let configuration = self.configuration;
        let [batch, m, k] = self.lhs_canonical_shape;
        let [_, n, _] = self.rhs_canonical_shape;
        let lhs = self.normalize(&inputs.lhs, &self.lhs_permutation, self.lhs_canonical_shape)?;
        let rhs = self.normalize(&inputs.rhs, &self.rhs_permutation, self.rhs_canonical_shape)?;
        let seed = seed.convert_element_type(self.compute_type)?.reshape(Shape::new(vec![
            Dimension::Static(batch),
            Dimension::Static(m),
            Dimension::Static(n),
        ]))?;
        let lhs_global_scale = self.global_scale(configuration.lhs_format, inputs.lhs_global_scale.as_ref())?;
        let rhs_global_scale = self.global_scale(configuration.rhs_format, inputs.rhs_global_scale.as_ref())?;
        let (lhs_cotangent, rhs_cotangent) = if configuration.quantized_backward() {
            let gradient_global_scale =
                self.global_scale(configuration.gradient_format, inputs.gradient_global_scale.as_ref())?;
            let (noise, weight_gradient_noise) = match &inputs.random_state {
                Some(state) => {
                    let noise_type = |shape: [usize; 3]| ArrayType::new_static(self.compute_type, shape);
                    let (state, noise) = state.random_uniform(&noise_type([batch, m, n]))?;
                    let (_, weight_gradient_noise) = state.random_uniform(&noise_type([batch, m, n]))?;
                    (Some(noise), Some(weight_gradient_noise))
                }
                None => (None, None),
            };

            // The input gradient contracts the seed `[B, M, N]` with the right input `[B, N, K]` along `N`.
            let seed_along_n = self.quantize(
                &seed,
                configuration.gradient_format,
                2,
                false,
                gradient_global_scale.clone(),
                noise.as_ref(),
            )?;
            let rhs_along_n = self.quantize(
                &rhs,
                configuration.rhs_format,
                1,
                configuration.tiled_rhs_scales,
                rhs_global_scale.clone(),
                None,
            )?;
            let dimensions = DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]);
            let lhs_cotangent = self.contract(&seed_along_n, &rhs_along_n, &dimensions, self.compute_type)?;

            // The weight gradient contracts the seed `[B, M, N]` with the left input `[B, M, K]` along `M`, after the
            // optional Hadamard transform of both along `M`.
            let (seed, lhs) = if configuration.hadamard_transform {
                (self.hadamard_transform(&seed, m)?, self.hadamard_transform(&lhs, m)?)
            } else {
                (seed, lhs.clone())
            };
            let seed_along_m = self.quantize(
                &seed,
                configuration.gradient_format,
                1,
                false,
                gradient_global_scale,
                weight_gradient_noise.as_ref(),
            )?;
            let lhs_along_m = self.quantize(&lhs, configuration.lhs_format, 1, false, lhs_global_scale, None)?;
            let dimensions = DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0]);
            let rhs_cotangent = self.contract(&seed_along_m, &lhs_along_m, &dimensions, self.compute_type)?;
            (lhs_cotangent, rhs_cotangent)
        } else {
            let lhs_cotangent = seed.dot(&rhs, &DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]))?;
            let rhs_cotangent = seed.dot(&lhs, &DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0]))?;

            // Straight-through estimation zeroes the gradients of values that the forward quantization clipped.
            let mask = |cotangent: V, value: &V, format: BlockScaleFormat, global_scale: GlobalScale<V>| {
                let threshold = match global_scale {
                    GlobalScale::Provided(global_scale) if configuration.gradient_format.has_global_scale() => {
                        self.broadcast_scalar(&global_scale, [batch, value.r#type().dimension(1).value().unwrap(), k])?
                    }
                    GlobalScale::None
                        if configuration.gradient_format.has_global_scale() && format.has_global_scale() =>
                    {
                        self.fill(value, 1.0)?
                    }
                    _ => return Ok::<_, ProgramError>(cotangent),
                };
                let threshold = threshold.mul(&self.fill(value, format.largest_unclipped_value()?)?)?;
                let kept = value.abs()?.compare(&threshold, ComparisonDirection::LessThanOrEqual)?;
                kept.select(&cotangent, &self.fill(value, 0.0)?)
            };
            let lhs_cotangent = mask(lhs_cotangent, &lhs, configuration.lhs_format, lhs_global_scale)?;
            let rhs_cotangent = mask(rhs_cotangent, &rhs, configuration.rhs_format, rhs_global_scale)?;
            (lhs_cotangent, rhs_cotangent)
        };
        let lhs_type = inputs.lhs.r#type().cotangent()?.data_type();
        let rhs_type = inputs.rhs.r#type().cotangent()?.data_type();
        Ok((
            self.denormalize(&lhs_cotangent, &self.lhs_permutation, &self.lhs_shape, lhs_type)?,
            self.denormalize(&rhs_cotangent, &self.rhs_permutation, &self.rhs_shape, rhs_type)?,
        ))
    }

    /// Returns the global scale source of a tensor in `format` with the optional provided `global_scale`.
    fn global_scale<V: ConvertElementType>(
        &self,
        format: BlockScaleFormat,
        global_scale: Option<&V>,
    ) -> Result<GlobalScale<V>, ProgramError> {
        if !format.has_global_scale() {
            return Ok(GlobalScale::None);
        }
        match (self.configuration.global_scaling, global_scale) {
            (BlockScaledGlobalScaling::Current, _) => Ok(GlobalScale::Current),
            (BlockScaledGlobalScaling::Provided, Some(global_scale)) => {
                Ok(GlobalScale::Provided(global_scale.convert_element_type(self.compute_type)?))
            }
            (BlockScaledGlobalScaling::Provided, None) => Ok(GlobalScale::None),
        }
    }

    /// Quantizes the rank-three `value` along `axis` and returns its elements, the scales that [`ScaledDot`] consumes
    /// for a contraction along `axis`, and the effective global scale, if any. Tiled scales span both the free and the
    /// contracting dimension (i.e., axes 1 and 2) and are expanded along the dimension other than `axis`.
    fn quantize<V>(
        &self,
        value: &V,
        format: BlockScaleFormat,
        axis: usize,
        tiled: bool,
        global_scale: GlobalScale<V>,
        rounding_noise: Option<&V>,
    ) -> Result<QuantizedInput<V>, ProgramError>
    where
        V: Value<Type = ArrayType> + Abs + Broadcast + BlockQuantize + Clamp + Div + Mul + Reduce + Reshape,
        V::Domain: Fill<f64, V>,
    {
        let global_scale = match global_scale {
            GlobalScale::None => None,
            GlobalScale::Provided(global_scale) => Some(global_scale),
            GlobalScale::Current => {
                // `max_abs / (element_max · 448)` maps the largest magnitude to the largest element and scale. Clamping
                // the scale to the smallest normal value keeps it positive for tensors whose values are all zero.
                let max_abs = value.abs()?.reduce(&[0, 1, 2], ReductionKind::Max)?;
                let scalar = |constant: f64| value.domain().fill(&ArrayType::scalar(self.compute_type), constant);
                let global_scale = max_abs.div(&scalar(format.largest_unclipped_value()?)?)?;
                let smallest = if self.compute_type == DataType::F64 { f64::MIN_POSITIVE } else { 2f64.powi(-126) };
                Some(global_scale.clamp(&scalar(smallest)?, &scalar(f64::INFINITY)?)?)
            }
        };
        let block_axes = if tiled { vec![1, 2] } else { vec![axis] };
        let (elements, scales) = value.block_quantize_with_options(
            format.block_size,
            format.element_type,
            format.scale_type,
            block_axes,
            global_scale.as_ref(),
            rounding_noise,
        )?;
        let scales = if tiled {
            // Every tile scale repeats over the `block_size` rows of its tile along the dimension other than `axis`.
            let repeated_axis = 3 - axis;
            let mut expanded_type = scales.r#type().into_owned();
            expanded_type =
                expanded_type.with_inserted_dimension(repeated_axis + 1, Dimension::Static(format.block_size))?;
            let output_axes = if repeated_axis == 1 { [0, 1, 3] } else { [0, 1, 2] };
            let mut shape = scales.r#type().shape().dimensions().to_vec();
            shape[repeated_axis] = value.r#type().dimension(repeated_axis);
            scales.broadcast(expanded_type, &output_axes)?.reshape(Shape::new(shape))?
        } else {
            scales
        };
        Ok(QuantizedInput { elements, scales, global_scale })
    }

    /// Contracts two quantized inputs with [`ScaledDot`] and multiplies the result by their global scales. When the
    /// left format uses a global scale (i.e., the `nvfp4` mode of JAX), the product accumulates at the compute type
    /// and converts to `data_type` afterwards, as in JAX, which accumulates at `f32`. Other products accumulate at
    /// `data_type` directly, unless [`ScaledDot`] cannot accumulate its `bf16` dequantized inputs at it (e.g., `f16`),
    /// in which case they accumulate at the compute type as well.
    fn contract<V>(
        &self,
        lhs: &QuantizedInput<V>,
        rhs: &QuantizedInput<V>,
        dimensions: &DotDimensionNumbers,
        data_type: DataType,
    ) -> Result<V, ProgramError>
    where
        V: Value<Type = ArrayType> + Broadcast + ConvertElementType + Mul + ScaledDot,
    {
        let global_scale = match (&lhs.global_scale, &rhs.global_scale) {
            (Some(lhs), Some(rhs)) => Some(lhs.mul(rhs)?),
            (Some(scale), None) | (None, Some(scale)) => Some(scale.clone()),
            (None, None) => None,
        };
        let accumulation_type = if self.configuration.lhs_format.has_global_scale()
            || !matches!(data_type, DataType::BF16 | DataType::F32 | DataType::F64)
        {
            self.compute_type
        } else {
            data_type
        };
        let output = lhs.elements.scaled_dot(
            &rhs.elements,
            Some(&lhs.scales),
            Some(&rhs.scales),
            Some(dimensions),
            Some(accumulation_type),
        )?;
        let output = match global_scale {
            Some(global_scale) => {
                let shape = output.r#type().static_shape().unwrap().dimensions().to_vec();
                output.mul(&self.broadcast_scalar(&global_scale, [shape[0], shape[1], shape[2]])?)?
            }
            None => output,
        };
        output.convert_element_type(data_type)
    }

    /// Applies the random Hadamard transform of NVIDIA Transformer Engine along the dimension of extent `extent` (axis
    /// 1) of the rank-three `value`, multiplying every 16 consecutive values by `diag(s) · H / 4`. The 16×16 Sylvester
    /// Hadamard matrix `H` is the Kronecker product of four 2×2 Hadamard matrices, so the transform applies one 2×2
    /// butterfly along each of the four bits of the position within a chunk. Every butterfly adds or subtracts two
    /// values, so the transform rounds identically on every backend, unlike a matrix product that may use reduced
    /// precision (e.g., TF32 on GPUs).
    fn hadamard_transform<V>(&self, value: &V, extent: usize) -> Result<V, ProgramError>
    where
        V: Value<Type = ArrayType> + Add + Broadcast + Div + Floor + Mul + Reduce + Rem + Reshape + Sub,
        V::Domain: Fill<f64, V> + Iota<V>,
    {
        let shape = value.r#type().static_shape().unwrap().dimensions().to_vec();
        let [batch, other] = [shape[0], shape[2]];
        let static_shape =
            |dimensions: &[usize]| Shape::new(dimensions.iter().map(|&size| Dimension::Static(size)).collect());
        let chunks = [batch, extent / 16, 16, other];
        let signs = self.hadamard_signs(value)?;
        let mut signs_type = signs.r#type().into_owned();
        for (axis, size) in [(0, batch), (1, extent / 16), (3, other)] {
            signs_type = signs_type.with_inserted_dimension(axis, Dimension::Static(size))?;
        }
        let mut transformed = value
            .reshape(static_shape(&chunks))?
            .mul(&signs.broadcast(signs_type, &[2])?)?
            .reshape(static_shape(&[batch, extent / 16, 2, 2, 2, 2, other]))?;

        // The butterfly along bit axis `a` computes `y[.., k, ..] = Σ_j x[.., j, ..] · H2[j, k]` with
        // `H2[j, k] = 1 - 2 · j · k`, by broadcasting the input over a new output bit axis and reducing the input one.
        let domain = value.domain();
        let butterfly_type = ArrayType::new_static(self.compute_type, [2, 2]);
        let butterfly = domain.fill(&butterfly_type, 1.0)?.sub(
            &domain
                .fill(&butterfly_type, 2.0)?
                .mul(&domain.iota(&butterfly_type, 0)?.mul(&domain.iota(&butterfly_type, 1)?)?)?,
        )?;
        for axis in 2..6 {
            let mut expanded_type = transformed.r#type().into_owned();
            expanded_type = expanded_type.with_inserted_dimension(axis + 1, Dimension::Static(2))?;
            let input_axes = (0..7)
                .map(|input_axis| if input_axis <= axis { input_axis } else { input_axis + 1 })
                .collect::<Vec<_>>();
            let mut butterfly_expanded_type = butterfly.r#type().into_owned();
            for (inserted_axis, size) in expanded_type.static_shape().unwrap().dimensions().iter().enumerate() {
                if inserted_axis != axis && inserted_axis != axis + 1 {
                    butterfly_expanded_type =
                        butterfly_expanded_type.with_inserted_dimension(inserted_axis, Dimension::Static(*size))?;
                }
            }
            transformed = transformed
                .broadcast(expanded_type.clone(), &input_axes)?
                .mul(&butterfly.broadcast(butterfly_expanded_type, &[axis, axis + 1])?)?
                .reduce(&[axis], ReductionKind::Sum)?;
        }
        transformed.reshape(static_shape(&shape))?.div(&self.fill(value, 4.0)?)
    }

    /// Builds the sign vector `s` of the random Hadamard transform in the domain of `value`. The signs `1 - 2 · b_i`
    /// derive from the bits `b_i` of the constant whose set bits mark the negative signs, extracted from coordinate
    /// values with exact power of two arithmetic, because staged rules can only create constants by filling.
    fn hadamard_signs<V>(&self, value: &V) -> Result<V, ProgramError>
    where
        V: Value<Type = ArrayType> + Add + Div + Floor + Mul + Rem + Sub,
        V::Domain: Fill<f64, V> + Iota<V>,
    {
        let domain = value.domain();
        let vector_type = ArrayType::new_static(self.compute_type, [16]);
        let fill = |constant: f64| domain.fill(&vector_type, constant);
        let positions = domain.iota(&vector_type, 0)?;
        let mut powers = fill(1.0)?;
        for bit in 0..4 {
            let position_bit = positions.div(&fill(2f64.powi(bit))?)?.floor()?.rem(&fill(2.0)?)?;
            powers = powers.mul(&fill(1.0)?.add(&position_bit.mul(&fill(2f64.powi(1 << bit) - 1.0)?)?)?)?;
        }
        let negative_positions = HADAMARD_SIGNS
            .iter()
            .enumerate()
            .filter(|(_, sign)| **sign < 0.0)
            .map(|(position, _)| 2f64.powi(position as i32))
            .sum::<f64>();
        let negative = fill(negative_positions)?.div(&powers)?.floor()?.rem(&fill(2.0)?)?;
        fill(1.0)?.sub(&fill(2.0)?.mul(&negative)?)
    }

    /// Returns a value of the compute type with the shape of `value` and every element set to `constant`.
    fn fill<V: Value<Type = ArrayType>>(&self, value: &V, constant: f64) -> Result<V, ProgramError>
    where
        V::Domain: Fill<f64, V>,
    {
        value.domain().fill(&ArrayType::new(self.compute_type, value.r#type().shape().clone()), constant)
    }

    /// Broadcasts the rank-0 `scalar` to a rank-three array of `shape`, keeping its placement.
    fn broadcast_scalar<V: Value<Type = ArrayType> + Broadcast>(
        &self,
        scalar: &V,
        shape: [usize; 3],
    ) -> Result<V, ProgramError> {
        let mut expanded_type = scalar.r#type().into_owned();
        for (axis, size) in shape.into_iter().enumerate() {
            expanded_type = expanded_type.with_inserted_dimension(axis, Dimension::Static(size))?;
        }
        scalar.broadcast(expanded_type, &[])
    }
}

/// Quantized rank-three input of a contraction in the rules of a [`BlockScaledDot`].
struct QuantizedInput<V> {
    /// Quantized elements.
    elements: V,

    /// Scales expanded to one per block along the contracting dimension.
    scales: V,

    /// Global scale that multiplies the dequantized values, if any.
    global_scale: Option<V>,
}

/// Rounds the `f32` or `f64` `value` to the grid of a narrow floating-point type whose normal values have
/// `mantissa_bits` explicit mantissa bits and whose smallest normal value is `2^min_exponent`, below which the grid
/// spacing stays constant (i.e., subnormal values). Rounding is to the nearest grid value with ties to even, or
/// stochastic with `rounding_noise` (refer to [`BlockQuantize::block_quantize_with_options`]). `value` must be finite,
/// and the result is not clamped to the finite range of the narrow type.
///
/// The grid spacing at `x` derives from the power of two of [`binade_scale`], which is zero for zero and subnormal
/// values, where the clamp to the smallest normal value applies. Rounding to the nearest value is exact, because it
/// only multiplies or divides by powers of two and rounds to an integer. Stochastic rounding adds the noise first,
/// and that addition itself rounds, so values within one unit in the last place of the compute type below an integer
/// round up to it.
///
/// # Parameters
///
///   - `value`: Value to round, whose element type is the compute type.
///   - `min_exponent`: Exponent of the smallest normal value of the narrow type.
///   - `mantissa_bits`: Number of explicit mantissa bits of the narrow type.
///   - `rounding_noise`: Optional uniform samples in `[0, 1)` that select stochastic rounding.
///   - `fill`: Function that returns a value with the type of `value` and every element set to its argument.
fn round_to_floating_point_grid<V, F>(
    value: &V,
    min_exponent: i32,
    mantissa_bits: i32,
    rounding_noise: Option<&V>,
    fill: &F,
) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType> + Add + And + Clamp + ConvertElementType + Div + Floor + Mul + Round,
    F: Fn(f64) -> Result<V, ProgramError>,
{
    let spacing = binade_scale(value, fill)?
        .clamp(&fill(2f64.powi(min_exponent))?, &fill(f64::INFINITY)?)?
        .mul(&fill(2f64.powi(-mantissa_bits))?)?;
    let quotients = value.div(&spacing)?;
    let integers = match rounding_noise {
        Some(rounding_noise) => quotients.add(rounding_noise)?.floor()?,
        None => quotients.round()?,
    };
    integers.mul(&spacing)
}

/// Returns the power of two `2^floor(log2(|x|))` of every normal `f32` or `f64` value `x` in `value`, which keeps only
/// the exponent bits of `x` (i.e., the bits of positive infinity) and is therefore exact. The result is zero for zero
/// and subnormal values, whose exponent bits are zero, and positive infinity for infinite and NaN values.
///
/// # Parameters
///
///   - `value`: Value whose element type is the compute type.
///   - `fill`: Function that returns a value with the type of `value` and every element set to its argument.
fn binade_scale<V, F>(value: &V, fill: &F) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType> + And + ConvertElementType,
    F: Fn(f64) -> Result<V, ProgramError>,
{
    let compute_type = value.r#type().data_type();
    let bits_type = if compute_type == DataType::F64 { DataType::U64 } else { DataType::U32 };
    value
        .bitcast_element_type(bits_type)?
        .and(&fill(f64::INFINITY)?.bitcast_element_type(bits_type)?)?
        .bitcast_element_type(compute_type)
}

/// Validates block dequantization's array types and returns normalized, distinct block axes before any operations
/// execute or stage. Dynamic block ratios are checked by the array IR composition at runtime.
fn validate_block_dequantize_inputs(
    elements: &ArrayType,
    scales: &ArrayType,
    block_axes: Axes,
    data_type: DataType,
) -> Result<Vec<usize>, ProgramError> {
    if !data_type.is_floating_point() {
        return Err(TypeError::invalid(format!(
            "`block_dequantize` output data type must be floating-point but got `{data_type}`",
        ))
        .into());
    }
    for (name, input_type) in [("elements", elements), ("scales", scales)] {
        let data_type = input_type.data_type();
        if !data_type.is_real() && data_type != DataType::Boolean {
            return Err(TypeError::invalid(format!(
                "`block_dequantize` {name} must have a real numeric or Boolean element type but got `{data_type}`",
            ))
            .into());
        }
    }
    if elements.rank() != scales.rank() {
        return Err(TypeError::invalid("`block_dequantize` inputs must have the same rank").into());
    }
    let block_axes = block_axes.normalize(elements.rank())?;
    for axis in 0..elements.rank() {
        let element_dimension = elements.dimension(axis);
        let scale_dimension = scales.dimension(axis);
        if !block_axes.contains(&axis) {
            if element_dimension != scale_dimension {
                return Err(TypeError::invalid(format!(
                    "`block_dequantize` non-block dimension {axis} has element extent `{element_dimension}` but scale \
                     extent `{scale_dimension}`",
                ))
                .into());
            }
            continue;
        }
        match (element_dimension.value(), scale_dimension.value()) {
            (_, Some(0)) => {
                return Err(TypeError::invalid(format!(
                    "`block_dequantize` block dimension {axis} has an empty scale but scale extents must be positive",
                ))
                .into());
            }
            (Some(element_extent), Some(scale_extent)) if element_extent % scale_extent != 0 => {
                return Err(TypeError::invalid(format!(
                    "`block_dequantize` element extent {element_extent} of block dimension {axis} is not divisible \
                     by scale extent {scale_extent}",
                ))
                .into());
            }
            (Some(element_extent), Some(scale_extent)) if element_extent / scale_extent < 2 => {
                return Err(TypeError::invalid(format!(
                    "`block_dequantize` block dimension {axis} must have ratio at least 2 but got {}",
                    element_extent / scale_extent,
                ))
                .into());
            }
            _ if element_dimension == scale_dimension => {
                return Err(TypeError::invalid(format!(
                    "`block_dequantize` block dimension {axis} must have ratio at least 2 but both extents are \
                     `{element_dimension}`",
                ))
                .into());
            }
            _ => {}
        }
    }
    Ok(block_axes)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayIrOperation, ArrayOperation, DimensionBounds, DimensionValue, DimensionVariable, LogicalMesh, MeshAxis,
        MeshAxisType, Sharding, ShardingDimension,
    };
    use crate::axes::{AxisError, NamedAxis};
    use crate::batching::batch;
    use crate::contexts::ProjectedContext;
    use crate::differentiation::differentiate_at;
    use crate::macros::{
        check_gradient, check_operation_batching, check_operation_partial_evaluation, check_operation_type_inference,
    };
    use crate::operations::assertions::AssertionError;
    use crate::operations::sharding::shard_map::{ShardMapTracer, shard_map};
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::tracing::{Tracer, TracingContext};

    use super::*;

    /// Returns a `[rows, columns]` matrix whose values cycle through `values` in row-major order.
    fn cyclic_matrix(rows: usize, columns: usize, values: &[f32]) -> Array {
        Array::matrix(rows, columns, (0..rows * columns).map(|index| values[index % values.len()]).collect()).unwrap()
    }

    type EagerArrayContext = EagerContext<Array, ArrayOperation<Array>>;

    #[test]
    fn test_block_quantize() {
        // The NVFP4 recipe uses `f4e2m1fn` elements with `f8e4m3fn` scales `max_abs(block) / 6`. Every block below is a
        // scaled copy of `f4e2m1fn` grid points whose scale is exactly representable in `f8e4m3fn`, so quantization is
        // exact. The block whose values are all zero uses the smallest positive scale, `2^-9`.
        let input = Array::matrix(
            2,
            8,
            vec![
                3.0f32, 1.5, 0.5, 6.0, 0.5, 1.0, 0.25, 1.5, // Blocks with scales `1.0` and `0.25`.
                -12.0, 6.0, 3.0, -1.0, 0.0, 0.0, 0.0, 0.0, // Blocks with scale `2.0` and the smallest scale.
            ],
        )
        .unwrap();
        let (elements, scales) = input.block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN).unwrap();
        assert_eq!(elements.r#type().into_owned(), ArrayType::new_static(DataType::F4E2M1FN, [2, 8]));
        assert_eq!(scales.r#type().into_owned(), ArrayType::new_static(DataType::F8E4M3FN, [2, 2]));
        assert_eq!(scales.to_f64s(), vec![1.0, 0.25, 2.0, 0.001953125]);
        assert_eq!(
            elements.to_f64s(),
            vec![3.0, 1.5, 0.5, 6.0, 2.0, 4.0, 1.0, 6.0, -6.0, 3.0, 1.5, -0.5, 0.0, 0.0, 0.0, 0.0],
        );

        // The quantized inputs contract through `scaled_dot` to the exact full precision dot product.
        let dimensions = DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new());
        assert_eq!(
            elements.scaled_dot(&elements, Some(&scales), Some(&scales), Some(&dimensions), Some(DataType::F32)),
            input.dot(&input, &dimensions),
        );
    }

    #[test]
    fn test_block_quantize_mxfp8() {
        // The OCP MX recipe uses `f8e4m3fn` elements with power of two `f8e8m0fnu` scales
        // `2^(floor(log2(max_abs(block))) - 8)`. Several block maxima below are exact powers of two, and every
        // quotient is exactly representable in `f8e4m3fn`, so quantization is exact. The block whose values are all
        // zero uses the smallest scale, `2^-126`.
        let input = Array::matrix(
            2,
            8,
            vec![
                4.0f32, 2.0, 1.0, 0.5, 1.75, 0.5, -1.0, 0.25, // Blocks with scales `2^-6` and `2^-8`.
                -8.0, 4.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, // Blocks with scale `2^-5` and the smallest scale.
            ],
        )
        .unwrap();
        let (elements, scales) = input.block_quantize(4, DataType::F8E4M3FN, DataType::F8E8M0FNU).unwrap();
        assert_eq!(elements.r#type().into_owned(), ArrayType::new_static(DataType::F8E4M3FN, [2, 8]));
        assert_eq!(scales.r#type().into_owned(), ArrayType::new_static(DataType::F8E8M0FNU, [2, 2]));
        assert_eq!(scales.to_f64s(), vec![(-6.0f64).exp2(), (-8.0f64).exp2(), (-5.0f64).exp2(), (-126.0f64).exp2()]);
        assert_eq!(
            elements.to_f64s(),
            vec![256.0, 128.0, 64.0, 32.0, 448.0, 128.0, -256.0, 64.0, -256.0, 128.0, 64.0, 32.0, 0.0, 0.0, 0.0, 0.0],
        );

        // The quantized inputs contract through `scaled_dot` to the exact full precision dot product.
        let dimensions = DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new());
        assert_eq!(
            elements.scaled_dot(&elements, Some(&scales), Some(&scales), Some(&dimensions), Some(DataType::F32)),
            input.dot(&input, &dimensions),
        );

        // A block maximum just below a power of two stays in its own binade, so `3.9998` uses the scale
        // `2^(1 - 2) = 0.5` with `f4e2m1fn` elements, and its quotient `7.9996` saturates at `6`.
        let input = Array::vector(vec![3.9998f32, 0.75, 0.0, 0.0]).unwrap();
        let (elements, scales) = input.block_quantize(4, DataType::F4E2M1FN, DataType::F8E8M0FNU).unwrap();
        assert_eq!(scales.to_f64s(), vec![0.5]);
        assert_eq!(elements.to_f64s(), vec![6.0, 1.5, 0.0, 0.0]);
    }

    #[test]
    fn test_block_quantize_clamps_scales() {
        // An NVFP4 block maximum above `6 · 448` would need a scale beyond the largest finite `f8e4m3fn` value, so the
        // scale saturates at `448` and the quotients saturate at the largest element magnitude instead of becoming NaN.
        let input = Array::vector(vec![6000.0f32, 3000.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
        let (elements, scales) = input.block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN).unwrap();
        assert_eq!(scales.to_f64s(), vec![448.0, 0.171875]);
        assert_eq!(elements.to_f64s(), vec![6.0, 6.0, 0.0, 0.0, 6.0, 6.0, 6.0, 6.0]);

        // Likewise, an `f64` MX block maximum beyond the `f8e8m0fnu` range saturates at the largest scale, `2^127`.
        let input = Array::vector(vec![1e300f64, 0.0, 0.0, 0.0]).unwrap();
        let (elements, scales) = input.block_quantize(4, DataType::F8E4M3FN, DataType::F8E8M0FNU).unwrap();
        assert_eq!(scales.to_f64s(), vec![127.0f64.exp2()]);
        assert_eq!(elements.to_f64s(), vec![448.0, 0.0, 0.0, 0.0]);

        // Infinite values saturate to the largest element magnitude under both recipes.
        let input = Array::vector(vec![f32::INFINITY, 1.0, f32::NEG_INFINITY, 0.0]).unwrap();
        let (elements, _) = input.block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN).unwrap();
        assert_eq!(elements.to_f64s(), vec![6.0, 0.0, -6.0, 0.0]);
        let (elements, _) = input.block_quantize(4, DataType::F8E4M3FN, DataType::F8E8M0FNU).unwrap();
        assert_eq!(elements.to_f64s(), vec![448.0, 0.0, -448.0, 0.0]);
    }

    #[test]
    fn test_block_quantize_round_trip_tolerance() {
        // Values off the storage grids round-trip within the quantization error of the element type. `f8e4m3fn` has
        // three mantissa bits, so each dequantized element is within about 6% of its input (apart from the OCP MX
        // saturation of a block maximum that lands past the finite range), and the contraction of eight such products
        // stays within a proportional tolerance of the full precision dot product.
        let values = [1.1f32, -2.3, 0.7, 3.9, 0.013, -0.27, 5.4, 8.9];
        let input = Array::matrix(1, 8, values.to_vec()).unwrap();
        let (elements, scales) = input.block_quantize(4, DataType::F8E4M3FN, DataType::F8E8M0FNU).unwrap();
        assert_eq!(elements.r#type().into_owned(), ArrayType::new_static(DataType::F8E4M3FN, [1, 8]));
        assert_eq!(scales.r#type().into_owned(), ArrayType::new_static(DataType::F8E8M0FNU, [1, 2]));
        let dimensions = DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new());
        let product = elements
            .scaled_dot(&elements, Some(&scales), Some(&scales), Some(&dimensions), Some(DataType::F32))
            .unwrap()
            .to_f64s()[0];
        let expected = input.dot(&input, &dimensions).unwrap().to_f64s()[0];
        assert_abs_diff_eq!(product, expected, epsilon = 0.05 * expected);
    }

    #[test]
    fn test_block_quantize_ranks() {
        // Inputs of any positive rank quantize each block of their last dimension identically.
        let values = [1.1f32, -2.3, 0.7, 3.9, 0.013, -0.27, 5.4, 8.9];
        let (elements, scales) = Array::vector(values.to_vec())
            .unwrap()
            .block_quantize(4, DataType::F8E4M3FN, DataType::F8E8M0FNU)
            .unwrap();
        assert_eq!(elements.r#type().into_owned(), ArrayType::new_static(DataType::F8E4M3FN, [8]));
        assert_eq!(scales.r#type().into_owned(), ArrayType::new_static(DataType::F8E8M0FNU, [2]));
        let rank_four =
            Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [1, 1, 1, 8]), &values).unwrap();
        let (rank_four_elements, rank_four_scales) =
            rank_four.block_quantize(4, DataType::F8E4M3FN, DataType::F8E8M0FNU).unwrap();
        assert_eq!(rank_four_elements.r#type().into_owned(), ArrayType::new_static(DataType::F8E4M3FN, [1, 1, 1, 8]));
        assert_eq!(rank_four_scales.r#type().into_owned(), ArrayType::new_static(DataType::F8E8M0FNU, [1, 1, 1, 2]));
        assert_eq!(rank_four_elements.to_f64s(), elements.to_f64s());
        assert_eq!(rank_four_scales.to_f64s(), scales.to_f64s());
    }

    #[test]
    fn test_block_quantize_array_ir() {
        // Concrete array IR values quantize their array members exactly like the arrays themselves.
        let input = Array::matrix(1, 8, vec![3.0f32, 1.5, 0.5, 6.0, 0.5, 1.0, 0.25, 1.5]).unwrap();
        let (elements, scales) = input.block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN).unwrap();
        assert_eq!(
            ArrayIrValue::Array(input).block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN),
            Ok((ArrayIrValue::Array(elements), ArrayIrValue::Array(scales))),
        );
    }

    #[test]
    fn test_block_quantize_validation() {
        let input = Array::matrix(2, 8, vec![1.0f32; 16]).unwrap();
        assert_eq!(
            input.block_quantize(4, DataType::F16, DataType::F8E4M3FN),
            Err(ProgramError::from(TypeError::invalid("`block_quantize` does not support element data type `f16`"))),
        );
        assert_eq!(
            input.block_quantize(4, DataType::F4E2M1FN, DataType::F16),
            Err(ProgramError::from(TypeError::invalid("`block_quantize` does not support scale data type `f16`"))),
        );
        assert_eq!(
            Array::vector(vec![1i32; 8]).unwrap().block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN),
            Err(ProgramError::from(TypeError::invalid(
                "`block_quantize` expects an `f32` or `f64` input but got `i32`",
            ))),
        );
        assert_eq!(
            Array::scalar(1.0f32).unwrap().block_quantize(2, DataType::F4E2M1FN, DataType::F8E4M3FN),
            Err(ProgramError::from(TypeError::invalid("`block_quantize` does not support rank-0 inputs"))),
        );

        // `ScaledDot` requires blocks of at least two values and non-empty scales.
        assert_eq!(
            input.block_quantize(1, DataType::F4E2M1FN, DataType::F8E4M3FN),
            Err(ProgramError::from(TypeError::invalid("`block_quantize` block size must be at least 2 but got 1"))),
        );
        assert_eq!(
            input.block_quantize(3, DataType::F4E2M1FN, DataType::F8E4M3FN),
            Err(ProgramError::from(TypeError::invalid(
                "`block_quantize` block dimension 1 size 8 is not a positive multiple of block size 3",
            ))),
        );
        assert_eq!(
            Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [2, 0]), &[])
                .unwrap()
                .block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN),
            Err(ProgramError::from(TypeError::invalid(
                "`block_quantize` block dimension 1 size 0 is not a positive multiple of block size 4",
            ))),
        );

        // Staged inputs must have a static shape, and rejected requests stage nothing.
        let extent = Dimension::Dynamic(DimensionVariable::new("n", DimensionBounds::new(0, Some(8)).unwrap()));
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |input: Tracer<_>| {
                assert!(matches!(
                    input.block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN),
                    Err(error) if error
                        == ProgramError::from(TypeError::invalid("`block_quantize` input must have a static shape")),
                ));
                Ok(input)
            },
            ArrayType::new(DataType::F32, Shape::new(vec![extent])),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[n] .
                in (%0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_block_quantize_staging() {
        // Quantizing a tracer stages the composition, and the staged outputs carry the element and scale types.
        let (output_types, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |input: Tracer<_>| {
                let (elements, scales) = input.block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN)?;
                Ok(vec![elements, scales])
            },
            ArrayType::new_static(DataType::F32, [2, 8]),
        )
        .unwrap();
        assert_eq!(
            output_types,
            vec![ArrayType::new_static(DataType::F4E2M1FN, [2, 8]), ArrayType::new_static(DataType::F8E4M3FN, [2, 2])],
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 8] .
                let %1:f32[2, 2, 4] = reshape [shape=[2, 2, 4]] %0
                    %2:f32[2, 2, 4] = abs %1
                    %3:f32[2, 2] = reduce [kind=max, axes=[2]] %2
                    %4:f32[] = constant [value=6.0]
                    %5:f32[2, 2] = broadcast [output_type=f32[2, 2], output_axes=[]] %4
                    %6:f32[2, 2] = div %3 %5
                    %7:f32[] = constant [value=0.001953125]
                    %8:f32[] = constant [value=448.0]
                    %9:f32[2, 2] = clamp %7 %6 %8
                    %10:u32[2, 2] = convert_element_type [data_type=u32, bitcast=true] %9
                    %11:f32[] = constant [value=inf]
                    %12:f32[2, 2] = broadcast [output_type=f32[2, 2], output_axes=[]] %11
                    %13:u32[2, 2] = convert_element_type [data_type=u32, bitcast=true] %12
                    %14:u32[2, 2] = and %10 %13
                    %15:f32[2, 2] = convert_element_type [data_type=f32, bitcast=true] %14
                    %16:f32[] = constant [value=0.015625]
                    %17:f32[2, 2] = broadcast [output_type=f32[2, 2], output_axes=[]] %16
                    %18:f32[] = constant [value=inf]
                    %19:f32[2, 2] = broadcast [output_type=f32[2, 2], output_axes=[]] %18
                    %20:f32[2, 2] = clamp %17 %15 %19
                    %21:f32[] = constant [value=0.125]
                    %22:f32[2, 2] = broadcast [output_type=f32[2, 2], output_axes=[]] %21
                    %23:f32[2, 2] = mul %20 %22
                    %24:f32[2, 2] = div %9 %23
                    %25:f32[2, 2] = round %24
                    %26:f32[2, 2] = mul %25 %23
                    %27:f8e4m3fn[2, 2] = convert_element_type [data_type=f8e4m3fn] %26
                    %28:f32[2, 2, 4] = broadcast [output_type=f32[2, 2, 4], output_axes=[0, 1]] %26
                    %29:f32[2, 8] = reshape [shape=[2, 8]] %28
                    %30:f32[2, 8] = div %0 %29
                    %31:f32[] = constant [value=-6.0]
                    %32:f32[] = constant [value=6.0]
                    %33:f32[2, 8] = clamp %31 %30 %32
                    %34:u32[2, 8] = convert_element_type [data_type=u32, bitcast=true] %33
                    %35:f32[] = constant [value=inf]
                    %36:f32[2, 8] = broadcast [output_type=f32[2, 8], output_axes=[]] %35
                    %37:u32[2, 8] = convert_element_type [data_type=u32, bitcast=true] %36
                    %38:u32[2, 8] = and %34 %37
                    %39:f32[2, 8] = convert_element_type [data_type=f32, bitcast=true] %38
                    %40:f32[] = constant [value=1.0]
                    %41:f32[2, 8] = broadcast [output_type=f32[2, 8], output_axes=[]] %40
                    %42:f32[] = constant [value=inf]
                    %43:f32[2, 8] = broadcast [output_type=f32[2, 8], output_axes=[]] %42
                    %44:f32[2, 8] = clamp %41 %39 %43
                    %45:f32[] = constant [value=0.5]
                    %46:f32[2, 8] = broadcast [output_type=f32[2, 8], output_axes=[]] %45
                    %47:f32[2, 8] = mul %44 %46
                    %48:f32[2, 8] = div %33 %47
                    %49:f32[2, 8] = round %48
                    %50:f32[2, 8] = mul %49 %47
                    %51:f4e2m1fn[2, 8] = convert_element_type [data_type=f4e2m1fn] %50
                in (%51, %27)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_block_quantize_sharding() {
        // Inside a manual region, the elements and scales of a varying input keep its variation for both recipes,
        // because the scales are broadcast back over their blocks with their own variation rather than to an invariant
        // type and the recipe constants are aligned with the varying values that they combine with.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying = |data_type: DataType, size: usize| {
            ArrayType::new_static(data_type, [size])
                .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
                .unwrap()
        };
        let (output_types, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input: Tracer<_>| {
                let (elements, scales) = input.block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN)?;
                Ok(vec![elements, scales])
            },
            varying(DataType::F32, 8),
            vec![("x".to_string(), NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2 })],
        )
        .unwrap();
        assert_eq!(output_types, vec![varying(DataType::F4E2M1FN, 8), varying(DataType::F8E4M3FN, 2)]);
        assert_eq!(
            program.to_string(),
            indoc! {r#"
                lambda %0:f32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                let %1:f32[2, 4][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = reshape [shape=[2, 4]] %0
                    %2:f32[2, 4][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = abs %1
                    %3:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = reduce [kind=max, axes=[1]] %2
                    %4:f32[] = constant [value=6.0]
                    %5:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %4
                    %6:f32[2][sharding={mesh<['x'=2:manual]>, [{}]}] = broadcast [output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}]}], output_axes=[0]] %5
                    %7:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = parallel_vary [axis_name="x"] %6
                    %8:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = div %3 %7
                    %9:f32[] = constant [value=0.001953125]
                    %10:f32[] = constant [value=448.0]
                    %11:f32[][sharding={mesh<['x'=2:manual]>, []}] = broadcast [output_type=f32[][sharding={mesh<['x'=2:manual]>, []}], output_axes=[]] %9
                    %12:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = parallel_vary [axis_name="x"] %11
                    %13:f32[][sharding={mesh<['x'=2:manual]>, []}] = broadcast [output_type=f32[][sharding={mesh<['x'=2:manual]>, []}], output_axes=[]] %10
                    %14:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = parallel_vary [axis_name="x"] %13
                    %15:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = clamp %12 %8 %14
                    %16:u32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = convert_element_type [data_type=u32, bitcast=true] %15
                    %17:f32[] = constant [value=inf]
                    %18:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %17
                    %19:u32[2] = convert_element_type [data_type=u32, bitcast=true] %18
                    %20:u32[2][sharding={mesh<['x'=2:manual]>, [{}]}] = broadcast [output_type=u32[2][sharding={mesh<['x'=2:manual]>, [{}]}], output_axes=[0]] %19
                    %21:u32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = parallel_vary [axis_name="x"] %20
                    %22:u32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = and %16 %21
                    %23:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = convert_element_type [data_type=f32, bitcast=true] %22
                    %24:f32[] = constant [value=0.015625]
                    %25:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %24
                    %26:f32[] = constant [value=inf]
                    %27:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %26
                    %28:f32[2][sharding={mesh<['x'=2:manual]>, [{}]}] = broadcast [output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}]}], output_axes=[0]] %25
                    %29:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = parallel_vary [axis_name="x"] %28
                    %30:f32[2][sharding={mesh<['x'=2:manual]>, [{}]}] = broadcast [output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}]}], output_axes=[0]] %27
                    %31:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = parallel_vary [axis_name="x"] %30
                    %32:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = clamp %29 %23 %31
                    %33:f32[] = constant [value=0.125]
                    %34:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %33
                    %35:f32[2][sharding={mesh<['x'=2:manual]>, [{}]}] = broadcast [output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}]}], output_axes=[0]] %34
                    %36:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = parallel_vary [axis_name="x"] %35
                    %37:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = mul %32 %36
                    %38:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = div %15 %37
                    %39:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = round %38
                    %40:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = mul %39 %37
                    %41:f8e4m3fn[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = convert_element_type [data_type=f8e4m3fn] %40
                    %42:f32[2, 4][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = broadcast [
                        output_type=f32[2, 4][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}],
                        output_axes=[0],
                    ] %40
                    %43:f32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = reshape [shape=[8]] %42
                    %44:f32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = div %0 %43
                    %45:f32[] = constant [value=-6.0]
                    %46:f32[] = constant [value=6.0]
                    %47:f32[][sharding={mesh<['x'=2:manual]>, []}] = broadcast [output_type=f32[][sharding={mesh<['x'=2:manual]>, []}], output_axes=[]] %45
                    %48:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = parallel_vary [axis_name="x"] %47
                    %49:f32[][sharding={mesh<['x'=2:manual]>, []}] = broadcast [output_type=f32[][sharding={mesh<['x'=2:manual]>, []}], output_axes=[]] %46
                    %50:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = parallel_vary [axis_name="x"] %49
                    %51:f32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = clamp %48 %44 %50
                    %52:u32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = convert_element_type [data_type=u32, bitcast=true] %51
                    %53:f32[] = constant [value=inf]
                    %54:f32[8] = broadcast [output_type=f32[8], output_axes=[]] %53
                    %55:u32[8] = convert_element_type [data_type=u32, bitcast=true] %54
                    %56:u32[8][sharding={mesh<['x'=2:manual]>, [{}]}] = broadcast [output_type=u32[8][sharding={mesh<['x'=2:manual]>, [{}]}], output_axes=[0]] %55
                    %57:u32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = parallel_vary [axis_name="x"] %56
                    %58:u32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = and %52 %57
                    %59:f32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = convert_element_type [data_type=f32, bitcast=true] %58
                    %60:f32[] = constant [value=1.0]
                    %61:f32[8] = broadcast [output_type=f32[8], output_axes=[]] %60
                    %62:f32[] = constant [value=inf]
                    %63:f32[8] = broadcast [output_type=f32[8], output_axes=[]] %62
                    %64:f32[8][sharding={mesh<['x'=2:manual]>, [{}]}] = broadcast [output_type=f32[8][sharding={mesh<['x'=2:manual]>, [{}]}], output_axes=[0]] %61
                    %65:f32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = parallel_vary [axis_name="x"] %64
                    %66:f32[8][sharding={mesh<['x'=2:manual]>, [{}]}] = broadcast [output_type=f32[8][sharding={mesh<['x'=2:manual]>, [{}]}], output_axes=[0]] %63
                    %67:f32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = parallel_vary [axis_name="x"] %66
                    %68:f32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = clamp %65 %59 %67
                    %69:f32[] = constant [value=0.5]
                    %70:f32[8] = broadcast [output_type=f32[8], output_axes=[]] %69
                    %71:f32[8][sharding={mesh<['x'=2:manual]>, [{}]}] = broadcast [output_type=f32[8][sharding={mesh<['x'=2:manual]>, [{}]}], output_axes=[0]] %70
                    %72:f32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = parallel_vary [axis_name="x"] %71
                    %73:f32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = mul %68 %72
                    %74:f32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = div %51 %73
                    %75:f32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = round %74
                    %76:f32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = mul %75 %73
                    %77:f4e2m1fn[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = convert_element_type [data_type=f4e2m1fn] %76
                in (%77, %41)
            "#}
            .trim_end(),
        );
        let (output_types, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input: Tracer<_>| {
                let (elements, scales) = input.block_quantize(4, DataType::F8E4M3FN, DataType::F8E8M0FNU)?;
                Ok(vec![elements, scales])
            },
            varying(DataType::F32, 8),
            vec![("x".to_string(), NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2 })],
        )
        .unwrap();
        assert_eq!(output_types, vec![varying(DataType::F8E4M3FN, 8), varying(DataType::F8E8M0FNU, 2)]);
        assert_eq!(
            program.to_string(),
            indoc! {r#"
                lambda %0:f32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                let %1:f32[2, 4][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = reshape [shape=[2, 4]] %0
                    %2:f32[2, 4][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = abs %1
                    %3:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = reduce [kind=max, axes=[1]] %2
                    %4:u32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = convert_element_type [data_type=u32, bitcast=true] %3
                    %5:f32[] = constant [value=inf]
                    %6:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %5
                    %7:u32[2] = convert_element_type [data_type=u32, bitcast=true] %6
                    %8:u32[2][sharding={mesh<['x'=2:manual]>, [{}]}] = broadcast [output_type=u32[2][sharding={mesh<['x'=2:manual]>, [{}]}], output_axes=[0]] %7
                    %9:u32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = parallel_vary [axis_name="x"] %8
                    %10:u32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = and %4 %9
                    %11:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = convert_element_type [data_type=f32, bitcast=true] %10
                    %12:f32[] = constant [value=0.00390625]
                    %13:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %12
                    %14:f32[2][sharding={mesh<['x'=2:manual]>, [{}]}] = broadcast [output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}]}], output_axes=[0]] %13
                    %15:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = parallel_vary [axis_name="x"] %14
                    %16:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = mul %11 %15
                    %17:f32[] = constant [value=1.1754944e-38]
                    %18:f32[] = constant [value=1.7014118e38]
                    %19:f32[][sharding={mesh<['x'=2:manual]>, []}] = broadcast [output_type=f32[][sharding={mesh<['x'=2:manual]>, []}], output_axes=[]] %17
                    %20:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = parallel_vary [axis_name="x"] %19
                    %21:f32[][sharding={mesh<['x'=2:manual]>, []}] = broadcast [output_type=f32[][sharding={mesh<['x'=2:manual]>, []}], output_axes=[]] %18
                    %22:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = parallel_vary [axis_name="x"] %21
                    %23:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = clamp %20 %16 %22
                    %24:f8e8m0fnu[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = convert_element_type [data_type=f8e8m0fnu] %23
                    %25:f32[2, 4][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = broadcast [
                        output_type=f32[2, 4][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}],
                        output_axes=[0],
                    ] %23
                    %26:f32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = reshape [shape=[8]] %25
                    %27:f32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = div %0 %26
                    %28:f32[] = constant [value=-448.0]
                    %29:f32[] = constant [value=448.0]
                    %30:f32[][sharding={mesh<['x'=2:manual]>, []}] = broadcast [output_type=f32[][sharding={mesh<['x'=2:manual]>, []}], output_axes=[]] %28
                    %31:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = parallel_vary [axis_name="x"] %30
                    %32:f32[][sharding={mesh<['x'=2:manual]>, []}] = broadcast [output_type=f32[][sharding={mesh<['x'=2:manual]>, []}], output_axes=[]] %29
                    %33:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = parallel_vary [axis_name="x"] %32
                    %34:f32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = clamp %31 %27 %33
                    %35:u32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = convert_element_type [data_type=u32, bitcast=true] %34
                    %36:f32[] = constant [value=inf]
                    %37:f32[8] = broadcast [output_type=f32[8], output_axes=[]] %36
                    %38:u32[8] = convert_element_type [data_type=u32, bitcast=true] %37
                    %39:u32[8][sharding={mesh<['x'=2:manual]>, [{}]}] = broadcast [output_type=u32[8][sharding={mesh<['x'=2:manual]>, [{}]}], output_axes=[0]] %38
                    %40:u32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = parallel_vary [axis_name="x"] %39
                    %41:u32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = and %35 %40
                    %42:f32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = convert_element_type [data_type=f32, bitcast=true] %41
                    %43:f32[] = constant [value=0.015625]
                    %44:f32[8] = broadcast [output_type=f32[8], output_axes=[]] %43
                    %45:f32[] = constant [value=inf]
                    %46:f32[8] = broadcast [output_type=f32[8], output_axes=[]] %45
                    %47:f32[8][sharding={mesh<['x'=2:manual]>, [{}]}] = broadcast [output_type=f32[8][sharding={mesh<['x'=2:manual]>, [{}]}], output_axes=[0]] %44
                    %48:f32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = parallel_vary [axis_name="x"] %47
                    %49:f32[8][sharding={mesh<['x'=2:manual]>, [{}]}] = broadcast [output_type=f32[8][sharding={mesh<['x'=2:manual]>, [{}]}], output_axes=[0]] %46
                    %50:f32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = parallel_vary [axis_name="x"] %49
                    %51:f32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = clamp %48 %42 %50
                    %52:f32[] = constant [value=0.125]
                    %53:f32[8] = broadcast [output_type=f32[8], output_axes=[]] %52
                    %54:f32[8][sharding={mesh<['x'=2:manual]>, [{}]}] = broadcast [output_type=f32[8][sharding={mesh<['x'=2:manual]>, [{}]}], output_axes=[0]] %53
                    %55:f32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = parallel_vary [axis_name="x"] %54
                    %56:f32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = mul %51 %55
                    %57:f32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = div %34 %56
                    %58:f32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = round %57
                    %59:f32[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = mul %58 %56
                    %60:f8e4m3fn[8][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = convert_element_type [data_type=f8e4m3fn] %59
                in (%60, %24)
            "#}
            .trim_end(),
        );

        // Explicit placement of the leading dimension is likewise kept rather than dropped by the scale broadcast.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded = |data_type: DataType, columns: usize| {
            ArrayType::new_static(data_type, [2, columns])
                .with_sharding(
                    Sharding::new(
                        mesh.clone(),
                        vec![ShardingDimension::sharded(["y"]), ShardingDimension::replicated()],
                    )
                    .unwrap(),
                )
                .unwrap()
        };
        let (output_types, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |input: Tracer<_>| {
                let (elements, scales) = input.block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN)?;
                Ok(vec![elements, scales])
            },
            sharded(DataType::F32, 8),
        )
        .unwrap();
        assert_eq!(output_types, vec![sharded(DataType::F4E2M1FN, 8), sharded(DataType::F8E4M3FN, 2)]);
        assert_eq!(
            program.to_string(),
            indoc! {r#"
                lambda %0:f32[2, 8][sharding={mesh<['y'=2:explicit]>, [{'y'}, {}]}] .
                let %1:f32[2, 2, 4][sharding={mesh<['y'=2:explicit]>, [{'y'}, {}, {}]}] = reshape [shape=[2, 2, 4]] %0
                    %2:f32[2, 2, 4][sharding={mesh<['y'=2:explicit]>, [{'y'}, {}, {}]}] = abs %1
                    %3:f32[2, 2][sharding={mesh<['y'=2:explicit]>, [{'y'}, {}]}] = reduce [kind=max, axes=[2]] %2
                    %4:f32[] = constant [value=6.0]
                    %5:f32[2, 2] = broadcast [output_type=f32[2, 2], output_axes=[]] %4
                    %6:f32[2, 2][sharding={mesh<['y'=2:explicit]>, [{'y'}, {}]}] = div %3 %5
                    %7:f32[] = constant [value=0.001953125]
                    %8:f32[] = constant [value=448.0]
                    %9:f32[2, 2][sharding={mesh<['y'=2:explicit]>, [{'y'}, {}]}] = clamp %7 %6 %8
                    %10:u32[2, 2][sharding={mesh<['y'=2:explicit]>, [{'y'}, {}]}] = convert_element_type [data_type=u32, bitcast=true] %9
                    %11:f32[] = constant [value=inf]
                    %12:f32[2, 2] = broadcast [output_type=f32[2, 2], output_axes=[]] %11
                    %13:u32[2, 2] = convert_element_type [data_type=u32, bitcast=true] %12
                    %14:u32[2, 2][sharding={mesh<['y'=2:explicit]>, [{'y'}, {}]}] = and %10 %13
                    %15:f32[2, 2][sharding={mesh<['y'=2:explicit]>, [{'y'}, {}]}] = convert_element_type [data_type=f32, bitcast=true] %14
                    %16:f32[] = constant [value=0.015625]
                    %17:f32[2, 2] = broadcast [output_type=f32[2, 2], output_axes=[]] %16
                    %18:f32[] = constant [value=inf]
                    %19:f32[2, 2] = broadcast [output_type=f32[2, 2], output_axes=[]] %18
                    %20:f32[2, 2][sharding={mesh<['y'=2:explicit]>, [{'y'}, {}]}] = clamp %17 %15 %19
                    %21:f32[] = constant [value=0.125]
                    %22:f32[2, 2] = broadcast [output_type=f32[2, 2], output_axes=[]] %21
                    %23:f32[2, 2][sharding={mesh<['y'=2:explicit]>, [{'y'}, {}]}] = mul %20 %22
                    %24:f32[2, 2][sharding={mesh<['y'=2:explicit]>, [{'y'}, {}]}] = div %9 %23
                    %25:f32[2, 2][sharding={mesh<['y'=2:explicit]>, [{'y'}, {}]}] = round %24
                    %26:f32[2, 2][sharding={mesh<['y'=2:explicit]>, [{'y'}, {}]}] = mul %25 %23
                    %27:f8e4m3fn[2, 2][sharding={mesh<['y'=2:explicit]>, [{'y'}, {}]}] = convert_element_type [data_type=f8e4m3fn] %26
                    %28:f32[2, 2, 4][sharding={mesh<['y'=2:explicit]>, [{'y'}, {}, {}]}] = broadcast [
                        output_type=f32[2, 2, 4][sharding={mesh<['y'=2:explicit]>, [{'y'}, {}, {}]}],
                        output_axes=[0, 1],
                    ] %26
                    %29:f32[2, 8][sharding={mesh<['y'=2:explicit]>, [{'y'}, {}]}] = reshape [shape=[2, 8]] %28
                    %30:f32[2, 8][sharding={mesh<['y'=2:explicit]>, [{'y'}, {}]}] = div %0 %29
                    %31:f32[] = constant [value=-6.0]
                    %32:f32[] = constant [value=6.0]
                    %33:f32[2, 8][sharding={mesh<['y'=2:explicit]>, [{'y'}, {}]}] = clamp %31 %30 %32
                    %34:u32[2, 8][sharding={mesh<['y'=2:explicit]>, [{'y'}, {}]}] = convert_element_type [data_type=u32, bitcast=true] %33
                    %35:f32[] = constant [value=inf]
                    %36:f32[2, 8] = broadcast [output_type=f32[2, 8], output_axes=[]] %35
                    %37:u32[2, 8] = convert_element_type [data_type=u32, bitcast=true] %36
                    %38:u32[2, 8][sharding={mesh<['y'=2:explicit]>, [{'y'}, {}]}] = and %34 %37
                    %39:f32[2, 8][sharding={mesh<['y'=2:explicit]>, [{'y'}, {}]}] = convert_element_type [data_type=f32, bitcast=true] %38
                    %40:f32[] = constant [value=1.0]
                    %41:f32[2, 8] = broadcast [output_type=f32[2, 8], output_axes=[]] %40
                    %42:f32[] = constant [value=inf]
                    %43:f32[2, 8] = broadcast [output_type=f32[2, 8], output_axes=[]] %42
                    %44:f32[2, 8][sharding={mesh<['y'=2:explicit]>, [{'y'}, {}]}] = clamp %41 %39 %43
                    %45:f32[] = constant [value=0.5]
                    %46:f32[2, 8] = broadcast [output_type=f32[2, 8], output_axes=[]] %45
                    %47:f32[2, 8][sharding={mesh<['y'=2:explicit]>, [{'y'}, {}]}] = mul %44 %46
                    %48:f32[2, 8][sharding={mesh<['y'=2:explicit]>, [{'y'}, {}]}] = div %33 %47
                    %49:f32[2, 8][sharding={mesh<['y'=2:explicit]>, [{'y'}, {}]}] = round %48
                    %50:f32[2, 8][sharding={mesh<['y'=2:explicit]>, [{'y'}, {}]}] = mul %49 %47
                    %51:f4e2m1fn[2, 8][sharding={mesh<['y'=2:explicit]>, [{'y'}, {}]}] = convert_element_type [data_type=f4e2m1fn] %50
                in (%51, %27)
            "#}
            .trim_end(),
        );
    }

    #[test]
    fn test_block_quantize_with_options() {
        // Blocks along the leading axis quantize like the transposed input with blocks along its last axis.
        let input = Array::matrix(
            4,
            4,
            vec![1.0f32, 2.0, 0.5, 0.25, 3.0, 6.0, 1.0, 1.5, 3.0, 0.5, 12.0, 2.0, 1.0, 1.0, 3.0, 6.0],
        )
        .unwrap();
        let (elements, scales) =
            input.block_quantize_with_options(2, DataType::F4E2M1FN, DataType::F8E4M3FN, 0, None, None).unwrap();
        let (transposed_elements, transposed_scales) = input
            .transpose(vec![1, 0])
            .unwrap()
            .block_quantize(2, DataType::F4E2M1FN, DataType::F8E4M3FN)
            .unwrap();
        assert_eq!(scales.r#type().into_owned(), ArrayType::new_static(DataType::F8E4M3FN, [2, 4]));
        assert_eq!(elements, transposed_elements.transpose(vec![1, 0]).unwrap());
        assert_eq!(scales, transposed_scales.transpose(vec![1, 0]).unwrap());

        // Two block axes form tiles that share one scale. Every tile below is a scaled copy of `f4e2m1fn` grid points,
        // so quantization is exact and dequantization over both block axes reproduces the input.
        let (elements, scales) = input
            .block_quantize_with_options(2, DataType::F4E2M1FN, DataType::F8E4M3FN, [0, 1], None, None)
            .unwrap();
        assert_eq!(scales.r#type().into_owned(), ArrayType::new_static(DataType::F8E4M3FN, [2, 2]));
        assert_eq!(scales.to_f64s(), vec![1.0, 0.25, 0.5, 2.0]);
        assert_eq!(
            elements.to_f64s(),
            vec![1.0, 2.0, 2.0, 1.0, 3.0, 6.0, 4.0, 6.0, 6.0, 1.0, 6.0, 1.0, 2.0, 2.0, 1.5, 3.0],
        );
        assert_eq!(elements.block_dequantize(&scales, [0, 1], DataType::F32), Ok(input));
    }

    #[test]
    fn test_block_quantize_with_options_global_scale() {
        // A global scale divides the input before block scaling, so a scaled copy of the module example quantizes to
        // the same elements and scales, and the global scale restores the input after dequantization.
        let input = Array::vector(vec![12.0f32, 6.0, 2.0, 24.0, 2.0, 4.0, 1.0, 6.0]).unwrap();
        let global_scale = Array::scalar(4.0f32).unwrap();
        let (elements, scales) = input
            .block_quantize_with_options(4, DataType::F4E2M1FN, DataType::F8E4M3FN, -1, Some(&global_scale), None)
            .unwrap();
        assert_eq!(scales.to_f64s(), vec![1.0, 0.25]);
        assert_eq!(elements.to_f64s(), vec![3.0, 1.5, 0.5, 6.0, 2.0, 4.0, 1.0, 6.0]);
        assert_eq!(
            elements
                .block_dequantize(&scales, [0], DataType::F32)
                .unwrap()
                .mul(&Array::vector(vec![4.0f32; 8]).unwrap()),
            Ok(input),
        );

        // Without the global scale, the block maximum `6000` exceeds `6 · 448` and saturates, while the global scale
        // brings it back into range.
        let input = Array::vector(vec![6000.0f32, 3000.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
        let (elements, scales) = input
            .block_quantize_with_options(4, DataType::F4E2M1FN, DataType::F8E4M3FN, -1, Some(&global_scale), None)
            .unwrap();
        assert_eq!(scales.to_f64s(), vec![256.0, 0.04296875]);
        assert_eq!(elements.to_f64s(), vec![6.0, 3.0, 0.0, 0.0, 6.0, 6.0, 6.0, 6.0]);
    }

    #[test]
    fn test_block_quantize_with_options_stochastic_rounding() {
        // The block scale is `1`, so the quotients equal the inputs. `6` is on the grid, `1.3` lies between `1.0` and
        // `1.5` (spacing `0.5`), and `0.2` lies below the smallest normal value `1.0`, where the spacing stays `0.5`.
        // Each value rounds up exactly when its noise reaches its distance from the grid value below it.
        let input = Array::vector(vec![6.0f32, 1.3, -1.3, 0.2]).unwrap();
        let noise = Array::vector(vec![0.0f32, 0.3, 0.3, 0.7]).unwrap();
        let (elements, scales) = input
            .block_quantize_with_options(4, DataType::F4E2M1FN, DataType::F8E4M3FN, -1, None, Some(&noise))
            .unwrap();
        assert_eq!(scales.to_f64s(), vec![1.0]);
        assert_eq!(elements.to_f64s(), vec![6.0, 1.0, -1.5, 0.5]);

        // Adding noise just below one to the largest grid value rounds up to the next integer at `f32`, so the result
        // saturates at the largest element magnitude rather than overflowing to NaN (`f8e4m3fn`) or infinity
        // (`f8e5m2`). The scales are `1`, and the quotients `448 / 32 = 14` and `57344 / 8192 = 7` are integers.
        let noise = Array::vector(vec![1.0f32 - f32::EPSILON / 2.0, 0.0, 0.0, 0.0]).unwrap();
        let input = Array::vector(vec![448.0f32, 0.0, 0.0, 0.0]).unwrap();
        let (elements, scales) = input
            .block_quantize_with_options(4, DataType::F8E4M3FN, DataType::F8E8M0FNU, -1, None, Some(&noise))
            .unwrap();
        assert_eq!(scales.to_f64s(), vec![1.0]);
        assert_eq!(elements.to_f64s(), vec![448.0, 0.0, 0.0, 0.0]);
        let input = Array::vector(vec![57344.0f32, 0.0, 0.0, 0.0]).unwrap();
        let (elements, scales) = input
            .block_quantize_with_options(4, DataType::F8E5M2, DataType::F8E8M0FNU, -1, None, Some(&noise))
            .unwrap();
        assert_eq!(scales.to_f64s(), vec![1.0]);
        assert_eq!(elements.to_f64s(), vec![57344.0, 0.0, 0.0, 0.0]);

        // Stratified noise over ten rows makes the column sums of the rounded values exactly ten times the inputs,
        // which shows that rounding is unbiased at both `f64` and `f32`. Zero never rounds away from zero.
        let input = Array::from_elements::<f64>(
            ArrayType::new_static(DataType::F64, [10, 8]),
            &(0..10).flat_map(|_| [6.0, 1.3, -1.3, 0.2, 0.0, -6.0, 2.5, 0.75]).collect::<Vec<_>>(),
        )
        .unwrap();
        let noise = Array::from_elements::<f64>(
            ArrayType::new_static(DataType::F64, [10, 8]),
            &(0..10).flat_map(|row| [(row as f64 + 0.5) / 10.0; 8]).collect::<Vec<_>>(),
        )
        .unwrap();
        let (elements, scales) = input
            .block_quantize_with_options(4, DataType::F4E2M1FN, DataType::F8E4M3FN, -1, None, Some(&noise))
            .unwrap();
        assert_eq!(scales.to_f64s(), vec![1.0; 20]);
        assert_eq!(
            elements
                .convert_element_type(DataType::F64)
                .unwrap()
                .reduce(&[0], ReductionKind::Sum)
                .unwrap()
                .to_f64s(),
            vec![60.0, 13.0, -13.0, 2.0, 0.0, -60.0, 25.0, 7.5],
        );
        let (elements, scales) = input
            .convert_element_type(DataType::F32)
            .unwrap()
            .block_quantize_with_options(
                4,
                DataType::F4E2M1FN,
                DataType::F8E4M3FN,
                -1,
                None,
                Some(&noise.convert_element_type(DataType::F32).unwrap()),
            )
            .unwrap();
        assert_eq!(scales.to_f64s(), vec![1.0; 20]);
        assert_eq!(
            elements
                .convert_element_type(DataType::F64)
                .unwrap()
                .reduce(&[0], ReductionKind::Sum)
                .unwrap()
                .to_f64s(),
            vec![60.0, 13.0, -13.0, 2.0, 0.0, -60.0, 25.0, 7.5],
        );
    }

    #[test]
    fn test_block_quantize_with_options_staging() {
        // Stochastic rounding computes the grid spacing from the exponent bits of the quotients.
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(input, global_scale, noise): (Tracer<_>, Tracer<_>, Tracer<_>)| {
                let (elements, scales) = input.block_quantize_with_options(
                    2,
                    DataType::F4E2M1FN,
                    DataType::F8E4M3FN,
                    [0],
                    Some(&global_scale),
                    Some(&noise),
                )?;
                Ok(vec![elements, scales])
            },
            (
                ArrayType::new_static(DataType::F32, [2]),
                ArrayType::scalar(DataType::F32),
                ArrayType::new_static(DataType::F32, [2]),
            ),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[], %2:f32[2] .
                let %3:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %1
                    %4:f32[2] = div %0 %3
                    %5:f32[1, 2] = reshape [shape=[1, 2]] %4
                    %6:f32[1, 2] = abs %5
                    %7:f32[1] = reduce [kind=max, axes=[1]] %6
                    %8:f32[] = constant [value=6.0]
                    %9:f32[1] = broadcast [output_type=f32[1], output_axes=[]] %8
                    %10:f32[1] = div %7 %9
                    %11:f32[] = constant [value=0.001953125]
                    %12:f32[] = constant [value=448.0]
                    %13:f32[1] = clamp %11 %10 %12
                    %14:u32[1] = convert_element_type [data_type=u32, bitcast=true] %13
                    %15:f32[] = constant [value=inf]
                    %16:f32[1] = broadcast [output_type=f32[1], output_axes=[]] %15
                    %17:u32[1] = convert_element_type [data_type=u32, bitcast=true] %16
                    %18:u32[1] = and %14 %17
                    %19:f32[1] = convert_element_type [data_type=f32, bitcast=true] %18
                    %20:f32[] = constant [value=0.015625]
                    %21:f32[1] = broadcast [output_type=f32[1], output_axes=[]] %20
                    %22:f32[] = constant [value=inf]
                    %23:f32[1] = broadcast [output_type=f32[1], output_axes=[]] %22
                    %24:f32[1] = clamp %21 %19 %23
                    %25:f32[] = constant [value=0.125]
                    %26:f32[1] = broadcast [output_type=f32[1], output_axes=[]] %25
                    %27:f32[1] = mul %24 %26
                    %28:f32[1] = div %13 %27
                    %29:f32[1] = round %28
                    %30:f32[1] = mul %29 %27
                    %31:f8e4m3fn[1] = convert_element_type [data_type=f8e4m3fn] %30
                    %32:f32[1, 2] = broadcast [output_type=f32[1, 2], output_axes=[0]] %30
                    %33:f32[2] = reshape [shape=[2]] %32
                    %34:f32[2] = div %4 %33
                    %35:f32[] = constant [value=-6.0]
                    %36:f32[] = constant [value=6.0]
                    %37:f32[2] = clamp %35 %34 %36
                    %38:u32[2] = convert_element_type [data_type=u32, bitcast=true] %37
                    %39:f32[] = constant [value=inf]
                    %40:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %39
                    %41:u32[2] = convert_element_type [data_type=u32, bitcast=true] %40
                    %42:u32[2] = and %38 %41
                    %43:f32[2] = convert_element_type [data_type=f32, bitcast=true] %42
                    %44:f32[] = constant [value=1.0]
                    %45:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %44
                    %46:f32[] = constant [value=inf]
                    %47:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %46
                    %48:f32[2] = clamp %45 %43 %47
                    %49:f32[] = constant [value=0.5]
                    %50:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %49
                    %51:f32[2] = mul %48 %50
                    %52:f32[2] = div %37 %51
                    %53:f32[2] = add %52 %2
                    %54:f32[2] = floor %53
                    %55:f32[2] = mul %54 %51
                    %56:f32[] = constant [value=-6.0]
                    %57:f32[] = constant [value=6.0]
                    %58:f32[2] = clamp %56 %55 %57
                    %59:f4e2m1fn[2] = convert_element_type [data_type=f4e2m1fn] %58
                in (%59, %31)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_block_quantize_with_options_validation() {
        let input = Array::matrix(2, 8, vec![1.0f32; 16]).unwrap();
        assert_eq!(
            input.block_quantize_with_options(4, DataType::F4E2M1FN, DataType::F8E4M3FN, Axes::default(), None, None),
            Err(ProgramError::from(TypeError::invalid("`block_quantize` requires at least one block axis"))),
        );
        assert_eq!(
            input.block_quantize_with_options(4, DataType::F4E2M1FN, DataType::F8E4M3FN, [0], None, None),
            Err(ProgramError::from(TypeError::invalid(
                "`block_quantize` block dimension 0 size 2 is not a positive multiple of block size 4",
            ))),
        );
        assert_eq!(
            input.block_quantize_with_options(4, DataType::F4E2M1FN, DataType::F8E4M3FN, [2], None, None),
            Err(ProgramError::Axis(AxisError::OutOfBounds { axis: Axis::from(2), rank: 2 })),
        );
        assert_eq!(
            input.block_quantize_with_options(4, DataType::F4E2M1FN, DataType::F8E4M3FN, [1, -1], None, None),
            Err(ProgramError::Axis(AxisError::DuplicateAxis { axis: 1 })),
        );
        assert_eq!(
            input.block_quantize_with_options(
                4,
                DataType::F4E2M1FN,
                DataType::F8E4M3FN,
                [1],
                Some(&Array::vector(vec![1.0f32]).unwrap()),
                None,
            ),
            Err(ProgramError::from(TypeError::invalid(
                "`block_quantize` global scale must be a rank-0 `f32` array but got `f32[1]`",
            ))),
        );
        assert_eq!(
            input.block_quantize_with_options(
                4,
                DataType::F4E2M1FN,
                DataType::F8E4M3FN,
                [1],
                Some(&Array::scalar(1.0f64).unwrap()),
                None,
            ),
            Err(ProgramError::from(TypeError::invalid(
                "`block_quantize` global scale must be a rank-0 `f32` array but got `f64[]`",
            ))),
        );
        assert_eq!(
            input.block_quantize_with_options(
                4,
                DataType::F8E4M3FN,
                DataType::F8E8M0FNU,
                -1,
                Some(&Array::scalar(1.0f32).unwrap()),
                None,
            ),
            Err(ProgramError::from(TypeError::invalid(
                "`block_quantize` global scales require `f8e4m3fn` scales but got `f8e8m0fnu` scales",
            ))),
        );
        assert_eq!(
            input.block_quantize_with_options(
                4,
                DataType::F4E2M1FN,
                DataType::F8E4M3FN,
                [1],
                None,
                Some(&Array::vector(vec![0.5f32; 16]).unwrap()),
            ),
            Err(ProgramError::from(TypeError::invalid(
                "`block_quantize` rounding noise must have the data type and shape of the input `f32[2, 8]` but got \
                 `f32[16]`",
            ))),
        );
    }

    #[test]
    fn test_block_dequantize() {
        let elements = Array::vector(vec![1i32, 2, 3, 4]).unwrap();
        let scales = Array::vector(vec![2.0f32, 3.0]).unwrap();
        let expected = Array::vector(vec![2.0f64, 4.0, 9.0, 12.0]).unwrap();
        assert_eq!(elements.block_dequantize(&scales, [-1], DataType::F64), Ok(expected.clone()));
        assert_eq!(
            ArrayIrValue::Array(elements).block_dequantize(&ArrayIrValue::Array(scales), [-1], DataType::F64),
            Ok(ArrayIrValue::Array(expected)),
        );

        // Both inputs are converted before multiplication. The midpoint rounds to one in `bf16`.
        let elements = Array::scalar(1.00390625f64).unwrap();
        let scales = elements.clone();
        assert_eq!(
            elements.block_dequantize(&scales, Axes::default(), DataType::BF16),
            Ok(Array::scalar(1.0f32).unwrap().convert_element_type(DataType::BF16).unwrap()),
        );
        assert_eq!(
            elements.block_dequantize(&scales, Axes::default(), DataType::F32),
            Ok(Array::scalar(1.0078277587890625f32).unwrap()),
        );
        assert_eq!(
            ArrayIrValue::Array(elements).block_dequantize(
                &ArrayIrValue::Array(scales),
                Axes::default(),
                DataType::F32,
            ),
            Ok(ArrayIrValue::Array(Array::scalar(1.0078277587890625f32).unwrap())),
        );
        let elements = Array::vector(vec![true, false, true, false]).unwrap();
        let scales = Array::vector(vec![2.0f32, 3.0]).unwrap();
        assert_eq!(
            elements.block_dequantize(&scales, [0], DataType::F32),
            Ok(Array::vector(vec![2.0f32, 0.0, 3.0, 0.0]).unwrap()),
        );
    }

    #[test]
    fn test_block_dequantize_axes_and_empty_arrays() {
        // Axis order does not alter block layout, and negative axes select from the end of the array.
        let elements = Array::matrix(4, 4, (1..=16).map(|value| value as f32).collect()).unwrap();
        let scales = Array::matrix(2, 2, vec![2.0f32, 3.0, 4.0, 5.0]).unwrap();
        let expected = Array::matrix(
            4,
            4,
            vec![2.0f32, 4.0, 9.0, 12.0, 10.0, 12.0, 21.0, 24.0, 36.0, 40.0, 55.0, 60.0, 52.0, 56.0, 75.0, 80.0],
        )
        .unwrap();
        assert_eq!(elements.block_dequantize(&scales, [-1, 0], DataType::F32), Ok(expected.clone()));
        assert_eq!(
            ArrayIrValue::Array(elements).block_dequantize(&ArrayIrValue::Array(scales), [-1, 0], DataType::F32),
            Ok(ArrayIrValue::Array(expected)),
        );

        // Equal-sized dimensions are omitted from the block axes for per-element scaling.
        let elements = Array::vector(vec![2.0f32, 3.0]).unwrap();
        let scales = Array::vector(vec![4.0f32, 5.0]).unwrap();
        let expected = Array::vector(vec![8.0f32, 15.0]).unwrap();
        assert_eq!(elements.block_dequantize(&scales, Axes::default(), DataType::F32), Ok(expected.clone()));
        assert_eq!(
            ArrayIrValue::Array(elements).block_dequantize(
                &ArrayIrValue::Array(scales),
                Axes::default(),
                DataType::F32,
            ),
            Ok(ArrayIrValue::Array(expected)),
        );

        // Empty non-block dimensions remain empty without rejecting the positive block ratio.
        let elements = Array::matrix(0, 4, Vec::<f32>::new()).unwrap();
        let scales = Array::matrix(0, 2, Vec::<f32>::new()).unwrap();
        assert_eq!(elements.block_dequantize(&scales, [1], DataType::F32), Ok(elements.clone()));
        assert_eq!(
            ArrayIrValue::Array(elements.clone()).block_dequantize(&ArrayIrValue::Array(scales), [1], DataType::F32),
            Ok(ArrayIrValue::Array(elements)),
        );
    }

    #[test]
    fn test_block_dequantize_quantized_values() {
        let input = Array::vector(vec![3.0f32, 1.5, 0.5, 6.0, 0.0, 0.25, 0.5, 1.5]).unwrap();
        let (elements, scales) = input.block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN).unwrap();
        assert_eq!(elements.block_dequantize(&scales, [-1], DataType::F32), Ok(input.clone()));
        let input = ArrayIrValue::Array(input);
        let (elements, scales) = input.block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN).unwrap();
        assert_eq!(elements.block_dequantize(&scales, [-1], DataType::F32), Ok(input));
    }

    #[test]
    fn test_block_dequantize_validation() {
        let elements = Array::vector(vec![1.0f32; 4]).unwrap();
        let scales = Array::vector(vec![2.0f32; 2]).unwrap();
        assert_eq!(
            elements.block_dequantize(&scales, [0, -1], DataType::F32),
            Err(ProgramError::Axis(AxisError::DuplicateAxis { axis: 0 })),
        );
        assert_eq!(
            elements.block_dequantize(&scales, [1], DataType::F32),
            Err(ProgramError::Axis(AxisError::OutOfBounds { axis: Axis::from(1), rank: 1 })),
        );
        assert_eq!(
            elements.block_dequantize(&scales, [0], DataType::I32),
            Err(TypeError::invalid("`block_dequantize` output data type must be floating-point but got `i32`").into()),
        );
        assert_eq!(
            elements.convert_element_type(DataType::C64).unwrap().block_dequantize(&scales, [0], DataType::F32),
            Err(TypeError::invalid(
                "`block_dequantize` elements must have a real numeric or Boolean element type but got `c64`",
            )
            .into()),
        );
        assert_eq!(
            elements.block_dequantize(&scales.convert_element_type(DataType::C64).unwrap(), [0], DataType::F32),
            Err(TypeError::invalid(
                "`block_dequantize` scales must have a real numeric or Boolean element type but got `c64`",
            )
            .into()),
        );
        assert_eq!(
            elements.block_dequantize(&Array::scalar(2.0f32).unwrap(), [0], DataType::F32),
            Err(TypeError::invalid("`block_dequantize` inputs must have the same rank").into()),
        );
        assert_eq!(
            elements.block_dequantize(&scales, Axes::default(), DataType::F32),
            Err(TypeError::invalid(
                "`block_dequantize` non-block dimension 0 has element extent `4` but scale extent `2`",
            )
            .into()),
        );
        assert_eq!(
            elements.block_dequantize(&Array::vector(Vec::<f32>::new()).unwrap(), [0], DataType::F32),
            Err(TypeError::invalid(
                "`block_dequantize` block dimension 0 has an empty scale but scale extents must be positive",
            )
            .into()),
        );
        assert_eq!(
            elements.block_dequantize(&Array::vector(vec![1.0f32; 3]).unwrap(), [0], DataType::F32),
            Err(TypeError::invalid(
                "`block_dequantize` element extent 4 of block dimension 0 is not divisible by scale extent 3",
            )
            .into()),
        );
        assert_eq!(
            elements.block_dequantize(&elements, [0], DataType::F32),
            Err(TypeError::invalid("`block_dequantize` block dimension 0 must have ratio at least 2 but got 1").into()),
        );
        assert_eq!(
            ArrayIrValue::Array(elements.clone()).block_dequantize(&ArrayIrValue::Array(elements), [0], DataType::F32,),
            Err(TypeError::invalid("`block_dequantize` block dimension 0 must have ratio at least 2 but got 1").into()),
        );
        assert_eq!(
            ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap()).block_dequantize(
                &ArrayIrValue::Array(scales),
                [0],
                DataType::F32,
            ),
            Err(TypeError::invalid("expected array type but got dimension type").into()),
        );
    }

    #[test]
    fn test_block_dequantize_staging() {
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(elements, scales): (Tracer<_>, Tracer<_>)| elements.block_dequantize(&scales, [-1], DataType::F32),
            (ArrayType::new_static(DataType::F4E2M1FN, [4]), ArrayType::new_static(DataType::F8E4M3FN, [2])),
        )
        .unwrap();
        let expected = indoc! {"
            lambda %0:f4e2m1fn[4], %1:f8e4m3fn[2] .
            let %2:f32[4] = convert_element_type [data_type=f32] %0
                %3:f8e4m3fn[2, 2] = broadcast [output_type=f8e4m3fn[2, 2], output_axes=[0]] %1
                %4:f8e4m3fn[4] = reshape [shape=[4]] %3
                %5:f32[4] = convert_element_type [data_type=f32] %4
                %6:f32[4] = mul %2 %5
            in (%6)
        "};
        assert_eq!(program.to_string(), expected.trim_end());
        assert_eq!(
            program.interpret((
                Array::vector(vec![1.0f32, 2.0, 3.0, 4.0])
                    .unwrap()
                    .convert_element_type(DataType::F4E2M1FN)
                    .unwrap(),
                Array::vector(vec![2.0f32, 3.0]).unwrap().convert_element_type(DataType::F8E4M3FN).unwrap(),
            )),
            Ok(Array::vector(vec![2.0f32, 4.0, 9.0, 12.0]).unwrap()),
        );

        // A projected static array view preserves the parent trace while using the homogeneous composition.
        let (_, projected_program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(elements, scales): (Tracer<_>, Tracer<_>)| {
                let elements = ValueProjection::<ArrayType>::into_projected(elements)?;
                let scales = ValueProjection::<ArrayType>::into_projected(scales)?;
                Ok(elements.block_dequantize(&scales, [-1], DataType::F32)?.into_value())
            },
            (
                ArrayIrType::Array(ArrayType::new_static(DataType::F4E2M1FN, [4])),
                ArrayIrType::Array(ArrayType::new_static(DataType::F8E4M3FN, [2])),
            ),
        )
        .unwrap();
        assert_eq!(projected_program.to_string(), expected.trim_end());
    }

    #[test]
    fn test_block_dequantize_staging_rejects_dynamic_homogeneous_shapes() {
        let extent = DimensionVariable::new("elements", DimensionBounds::new(4, Some(9)).unwrap());
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(elements, scales): (Tracer<_>, Tracer<_>)| {
                assert_eq!(
                    elements.block_dequantize(&scales, [0], DataType::F32),
                    Err(TypeError::invalid(
                        "`block_dequantize` homogeneous inputs must have static shapes; use array IR values for \
                         dynamic extents",
                    )
                    .into()),
                );
                Ok(elements)
            },
            (
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)])),
                ArrayType::new_static(DataType::F32, [2]),
            ),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[elements], %1:f32[2] .
                in (%0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_block_dequantize_dynamic_extents() {
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let elements = context.input(ArrayType::new_static(DataType::F32, [8]).into());
        let extent = DimensionVariable::new("scale", DimensionBounds::new(0, Some(9)).unwrap());
        let scales = context.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)])).into());
        let output = elements.block_dequantize(&scales, [0], DataType::BF16).unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {r#"
            lambda %0:f32[8], %1:f32[scale] .
            let %2:dimension<8> = constant [value=8]
                %3:bf16[8] = convert_element_type [data_type=bf16] %0
                %4:dimension<scale ∈ [0, 9)> = dimension_size [axis=0] %1
                %5:dimension<0> = constant [value=0]
                %6:dimension<1> = constant [value=1]
                %7:bool[] = compare [direction=GreaterThan] %4 %5
                () = assert [message="scale extent must be positive", labels=["scale"]] %7 %4
                %8:dimension<max(scale, 1) ∈ [1, 9)> = dimension_max %4 %6
                %9:dimension<8 % max(scale, 1) ∈ [0, 8)> = dimension_rem %2 %8
                %10:bool[] = compare [direction=Equal] %9 %5
                () = assert [
                    message="element extent must be divisible by scale extent",
                    labels=["elements", "scale"],
                ] %10 %2 %4
                %11:dimension<8 / max(scale, 1) ∈ [1, 9)> = dimension_div %2 %8
                %12:dimension<2> = constant [value=2]
                %13:bool[] = compare [direction=LessThanOrEqual] %12 %11
                () = assert [message="block ratio must be at least two", labels=["ratio"]] %13 %11
                %14:f32[scale, 8 / max(scale, 1)] = broadcast [output_axes=[0]] %1 %4 %11
                %15:f32[8] = reshape %14 %2
                %16:bf16[8] = convert_element_type [data_type=bf16] %15
                %17:bf16[8] = mul %3 %16
            in (%17)
        "#}
            .trim_end(),
        );
        let evaluation = program
            .partially_evaluate(&program.input_types().into_iter().map(PartialValue::Unknown).collect::<Vec<_>>())
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {r#"
            lambda %0:f32[8], %1:f32[scale] .
            let %2:dimension<scale ∈ [0, 9)> = dimension_size [axis=0] %1
                %3:dimension<0> = const 0
                %4:bool[] = compare [direction=GreaterThan] %2 %3
                %5:dimension<8> = const 8
                %6:dimension<1> = const 1
                () = assert [message="scale extent must be positive", labels=["scale"]] %4 %2
                %7:dimension<max(scale, 1) ∈ [1, 9)> = dimension_max %2 %6
                %8:dimension<8 % max(scale, 1) ∈ [0, 8)> = dimension_rem %5 %7
                %9:bool[] = compare [direction=Equal] %8 %3
                %10:dimension<2> = const 2
                () = assert [
                    message="element extent must be divisible by scale extent",
                    labels=["elements", "scale"],
                ] %9 %5 %2
                %11:dimension<8 / max(scale, 1) ∈ [1, 9)> = dimension_div %5 %7
                %12:bool[] = compare [direction=LessThanOrEqual] %10 %11
                () = assert [message="block ratio must be at least two", labels=["ratio"]] %12 %11
                %13:f32[scale, 8 / max(scale, 1)] = broadcast [output_axes=[0]] %1 %2 %11
                %14:f32[8] = reshape %13 %5
                %15:bf16[8] = convert_element_type [data_type=bf16] %0
                %16:bf16[8] = convert_element_type [data_type=bf16] %14
                %17:bf16[8] = mul %15 %16
            in (%17)
        "#}
            .trim_end(),
        );
        let elements = ArrayIrValue::Array(Array::vector(vec![1.0f32; 8]).unwrap());
        let scales = ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap());
        let expected = ArrayIrValue::Array(
            Array::vector(vec![1.0f32, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
                .unwrap()
                .convert_element_type(DataType::BF16)
                .unwrap(),
        );
        assert_eq!(program.interpret(vec![elements.clone(), scales.clone()]), Ok(vec![expected.clone()]));
        assert_eq!(
            evaluation.interpret(
                &EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
                &[elements.clone(), scales],
            ),
            Ok(vec![expected]),
        );
        // Invalid scale extents fail the staged assertions both before and after partial evaluation.
        let scales = ArrayIrValue::Array(Array::vector(Vec::<f32>::new()).unwrap());
        let expected = AssertionError::Failed {
            message: "scale extent must be positive".to_string(),
            observations: vec![("scale".to_string(), "0".to_string())],
        };
        assert_eq!(
            program
                .interpret(vec![elements.clone(), scales.clone()])
                .unwrap_err()
                .downcast_custom::<AssertionError>(),
            Some(&expected),
        );
        assert_eq!(
            evaluation
                .interpret(
                    &EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
                    &[elements.clone(), scales],
                )
                .unwrap_err()
                .downcast_custom::<AssertionError>(),
            Some(&expected),
        );
        let scales = ArrayIrValue::Array(Array::vector(vec![1.0f32; 3]).unwrap());
        let expected = AssertionError::Failed {
            message: "element extent must be divisible by scale extent".to_string(),
            observations: vec![("elements".to_string(), "8".to_string()), ("scale".to_string(), "3".to_string())],
        };
        assert_eq!(
            program
                .interpret(vec![elements.clone(), scales.clone()])
                .unwrap_err()
                .downcast_custom::<AssertionError>(),
            Some(&expected),
        );
        assert_eq!(
            evaluation
                .interpret(
                    &EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
                    &[elements.clone(), scales],
                )
                .unwrap_err()
                .downcast_custom::<AssertionError>(),
            Some(&expected),
        );
        let scales = ArrayIrValue::Array(Array::vector(vec![1.0f32; 8]).unwrap());
        let expected = AssertionError::Failed {
            message: "block ratio must be at least two".to_string(),
            observations: vec![("ratio".to_string(), "1".to_string())],
        };
        assert_eq!(
            program
                .interpret(vec![elements.clone(), scales.clone()])
                .unwrap_err()
                .downcast_custom::<AssertionError>(),
            Some(&expected),
        );
        assert_eq!(
            evaluation
                .interpret(&EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(), &[elements, scales])
                .unwrap_err()
                .downcast_custom::<AssertionError>(),
            Some(&expected),
        );
    }

    #[test]
    fn test_block_dequantize_batching() {
        let elements = Array::matrix(2, 4, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let scales = Array::matrix(2, 2, vec![2.0f32, 3.0, 4.0, 5.0]).unwrap();
        assert_eq!(
            batch(
                |(elements, scales)| elements.block_dequantize(&scales, [-1], DataType::F32),
                (elements.clone(), scales.clone()),
                (BatchAxis::new(0), BatchAxis::new(0)),
                BatchAxis::new(0),
                2,
            ),
            Ok(Array::matrix(2, 4, vec![2.0f32, 4.0, 9.0, 12.0, 20.0, 24.0, 35.0, 40.0]).unwrap()),
        );
        assert_eq!(
            batch(
                |(elements, scales)| elements.block_dequantize(&scales, [-1], DataType::F32),
                (ArrayIrValue::Array(elements), ArrayIrValue::Array(scales.clone())),
                (BatchAxis::new(0), BatchAxis::new(0)),
                BatchAxis::new(0),
                None,
            ),
            Ok(
                ArrayIrValue::Array(Array::matrix(2, 4, vec![2.0f32, 4.0, 9.0, 12.0, 20.0, 24.0, 35.0, 40.0]).unwrap(),)
            ),
        );
        assert_eq!(
            batch(
                |(elements, scales)| elements.block_dequantize(&scales, [-1], DataType::F32),
                (Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap(), scales),
                (BatchAxis::replicated(), BatchAxis::new(0)),
                BatchAxis::new(0),
                2,
            ),
            Ok(Array::matrix(2, 4, vec![2.0f32, 4.0, 9.0, 12.0, 4.0, 8.0, 15.0, 20.0]).unwrap()),
        );
    }

    #[test]
    fn test_block_dequantize_differentiation() {
        let elements = Array::vector(vec![1.0f64, 2.0, 3.0, 4.0]).unwrap();
        let scales = Array::vector(vec![2.0f64, 3.0]).unwrap();
        assert_eq!(
            differentiate_at((elements.clone(), scales.clone())).value_and_gradient(|(elements, scales)| {
                elements.block_dequantize(&scales, [0], DataType::F64)?.reduce(&[0], ReductionKind::Sum)
            }),
            Ok((
                Array::scalar(27.0f64).unwrap(),
                (Array::vector(vec![2.0f64, 2.0, 3.0, 3.0]).unwrap(), Array::vector(vec![3.0f64, 7.0]).unwrap()),
            )),
        );
        assert_eq!(
            differentiate_at((ArrayIrValue::Array(elements.clone()), ArrayIrValue::Array(scales.clone())))
                .value_and_gradient(|(elements, scales)| {
                    elements.block_dequantize(&scales, [0], DataType::F64)?.reduce(&[0], ReductionKind::Sum)
                }),
            Ok((
                ArrayIrValue::Array(Array::scalar(27.0f64).unwrap()),
                (
                    ArrayIrValue::Array(Array::vector(vec![2.0f64, 2.0, 3.0, 3.0]).unwrap()),
                    ArrayIrValue::Array(Array::vector(vec![3.0f64, 7.0]).unwrap()),
                ),
            )),
        );
        check_gradient!(
            |elements, scales| elements.block_dequantize(&scales, [0], DataType::F64)?.reduce(&[0], ReductionKind::Sum),
            at = elements.clone(),
            with = scales.clone(),
            step = 1e-5,
            tolerance = 1e-8,
        );
        check_gradient!(
            |scales, elements| elements.block_dequantize(&scales, [0], DataType::F64)?.reduce(&[0], ReductionKind::Sum),
            at = scales,
            with = elements,
            step = 1e-5,
            tolerance = 1e-8,
        );
    }

    #[test]
    fn test_block_dequantize_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let element_type = ArrayType::new_static(DataType::F32, [2, 4])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        let scale_type = ArrayType::new_static(DataType::F32, [2, 2])
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap(),
            )
            .unwrap();
        let elements = Array::from_elements::<f32>(element_type.clone(), &[1.0; 8]).unwrap();
        let scales = Array::from_elements::<f32>(scale_type, &[2.0, 3.0, 4.0, 5.0]).unwrap();
        let output = elements.block_dequantize(&scales, [1], DataType::F32).unwrap();
        assert_eq!(output.r#type().as_ref(), &element_type);
        assert_eq!(output.to_f64s(), vec![2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0]);
        assert_eq!(
            ArrayIrValue::Array(elements).block_dequantize(&ArrayIrValue::Array(scales), [1], DataType::F32),
            Ok(ArrayIrValue::Array(output)),
        );

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let element_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let scale_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh, 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let elements = Array::from_elements::<f32>(element_type.clone(), &[1.0; 4]).unwrap();
        let scales = Array::from_elements::<f32>(scale_type, &[2.0, 3.0]).unwrap();
        let output = elements.block_dequantize(&scales, [0], DataType::F32).unwrap();
        assert_eq!(output.r#type().as_ref(), &element_type);
        assert_eq!(output.to_f64s(), vec![2.0, 2.0, 3.0, 3.0]);
        assert_eq!(
            ArrayIrValue::Array(elements).block_dequantize(&ArrayIrValue::Array(scales), [0], DataType::F32),
            Ok(ArrayIrValue::Array(output)),
        );
    }

    #[test]
    fn test_scaled_dot() {
        let dimensions = DotDimensionNumbers::matmul();
        let operation = ScaledDotOperation::new(dimensions.clone(), DataType::F32, true, false);

        // Operation identity, accessors, and formatting.
        assert_eq!(operation.name(), SCALED_DOT_OPERATION_NAME);
        assert_eq!(operation.dimensions(), &dimensions);
        assert_eq!(operation.accumulation_type(), DataType::F32);
        assert!(operation.has_lhs_scale());
        assert!(!operation.has_rhs_scale());
        assert_eq!(
            operation.to_string(),
            indoc! {"
                scaled_dot [
                    dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    accumulation_type=f32,
                    lhs_scale=true,
                    rhs_scale=false,
                ]
            "}
            .trim_end(),
        );

        // Staging records the default dimension numbers, the default `f32` accumulation type, and the scales.
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(lhs, rhs, lhs_scale, rhs_scale): (Tracer<_>, Tracer<_>, Tracer<_>, Tracer<_>)| {
                lhs.scaled_dot(&rhs, Some(&lhs_scale), Some(&rhs_scale), None, None)
            },
            (
                ArrayType::new_static(DataType::F8E4M3FN, [2, 4]),
                ArrayType::new_static(DataType::F8E5M2, [4, 3]),
                ArrayType::new_static(DataType::F8E8M0FNU, [2, 2]),
                ArrayType::new_static(DataType::F8E8M0FNU, [2, 3]),
            ),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f8e4m3fn[2, 4], %1:f8e5m2[4, 3], %2:f8e8m0fnu[2, 2], %3:f8e8m0fnu[2, 3] .
                let %4:f32[2, 3] = scaled_dot [
                    dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    accumulation_type=f32,
                    lhs_scale=true,
                    rhs_scale=true,
                ] %0 %1 %2 %3
                in (%4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_scaled_dot_operation_default_dimensions() {
        assert_eq!(ScaledDotOperation::default_dimensions(2), Ok(DotDimensionNumbers::matmul()));
        assert_eq!(
            ScaledDotOperation::default_dimensions(4),
            Ok(DotDimensionNumbers::new(vec![3], vec![2], vec![0, 1], vec![0, 1])),
        );
        assert_eq!(
            ScaledDotOperation::default_dimensions(1),
            Err(TypeError::invalid(format!(
                "default `{SCALED_DOT_OPERATION_NAME}` dimension numbers require inputs of rank at least 2 but got \
                 rank 1",
            ))),
        );
    }

    #[test]
    fn test_scaled_dot_type_inference() {
        let dimensions = DotDimensionNumbers::new(vec![2, 3], vec![1, 2], vec![0], vec![0]);
        let lhs = ArrayType::new_static(DataType::F8E4M3FN, [2, 3, 4, 6]);
        let rhs = ArrayType::new_static(DataType::F8E5M2, [2, 4, 6, 5]);
        let lhs_scale = ArrayType::new_static(DataType::F8E8M0FNU, [2, 3, 2, 2]);
        let rhs_scale = ArrayType::new_static(DataType::F8E8M0FNU, [2, 2, 2, 5]);
        check_operation_type_inference!(
            operation = ScaledDotOperation::new(dimensions.clone(), DataType::BF16, true, true),
            cases = [
                {
                    input_types = [lhs.clone(), rhs.clone(), lhs_scale.clone(), rhs_scale.clone()],
                    output_types = [ArrayType::new_static(DataType::BF16, [2, 3, 5])],
                },
                {
                    input_types = [lhs.clone(), rhs.clone(), lhs_scale.clone()],
                    error = format!("`{SCALED_DOT_OPERATION_NAME}` expects 4 inputs but got 3"),
                },
                {
                    input_types = [
                        lhs.clone(),
                        rhs.clone(),
                        ArrayType::new_static(DataType::F8E8M0FNU, [2, 3, 2]),
                        rhs_scale.clone(),
                    ],
                    error = format!("`{SCALED_DOT_OPERATION_NAME}` inputs must have the same rank"),
                },
                {
                    input_types = [
                        lhs.clone(),
                        rhs.clone(),
                        lhs_scale.clone(),
                        ArrayType::new_static(DataType::Token, [2, 2, 2, 5]),
                    ],
                    error = format!(
                        "`{SCALED_DOT_OPERATION_NAME}` input 3 must have a real numeric or Boolean element type but \
                         got `token`",
                    ),
                },
                {
                    input_types = [
                        ArrayType::new_static(DataType::C64, [2, 3, 4, 6]),
                        rhs.clone(),
                        lhs_scale.clone(),
                        rhs_scale.clone(),
                    ],
                    error = format!(
                        "`{SCALED_DOT_OPERATION_NAME}` input 0 must have a real numeric or Boolean element type but \
                         got `c64`",
                    ),
                },
                {
                    input_types = [
                        ArrayType::new_static(DataType::Boolean, [2, 3, 4, 6]),
                        rhs.clone(),
                        lhs_scale.clone(),
                        rhs_scale.clone(),
                    ],
                    output_types = [ArrayType::new_static(DataType::BF16, [2, 3, 5])],
                },
                {
                    input_types = [
                        lhs.clone(),
                        ArrayType::new_static(DataType::F8E5M2, [2, 5, 6, 5]),
                        lhs_scale.clone(),
                        rhs_scale.clone(),
                    ],
                    error = format!(
                        "`{SCALED_DOT_OPERATION_NAME}` contracting dimension sizes do not match (left-hand side axis \
                         2, right-hand side axis 1)",
                    ),
                },
                {
                    input_types = [
                        lhs.clone(),
                        rhs.clone(),
                        ArrayType::new_static(DataType::F8E8M0FNU, [2, 1, 2, 2]),
                        rhs_scale.clone(),
                    ],
                    error = format!(
                        "`{SCALED_DOT_OPERATION_NAME}` left dimension 1 has size `3` but its scale has size `1`",
                    ),
                },
                {
                    input_types = [
                        lhs.clone(),
                        rhs.clone(),
                        ArrayType::new_static(DataType::F8E8M0FNU, [2, 3, 3, 2]),
                        rhs_scale.clone(),
                    ],
                    error = format!(
                        "`{SCALED_DOT_OPERATION_NAME}` left contracting dimension 2 has size 4, which is not \
                         divisible by its scale size 3",
                    ),
                },
                {
                    input_types = [
                        lhs.clone(),
                        rhs.clone(),
                        ArrayType::new_static(DataType::F8E8M0FNU, [2, 3, 0, 2]),
                        rhs_scale.clone(),
                    ],
                    error = format!(
                        "`{SCALED_DOT_OPERATION_NAME}` left contracting dimension 2 has an empty scale, but scale \
                         sizes must be positive",
                    ),
                },
                {
                    input_types = [
                        lhs.clone(),
                        rhs.clone(),
                        lhs_scale.clone(),
                        ArrayType::new_static(DataType::F8E8M0FNU, [2, 4, 2, 5]),
                    ],
                    error = format!(
                        "`{SCALED_DOT_OPERATION_NAME}` right contracting dimension 1 must be at least twice as large \
                         as its scale but has ratio 1",
                    ),
                },
            ],
        );

        // Without scales, the operation is a dot of the `bf16` element arrays at the accumulation type.
        check_operation_type_inference!(
            operation = ScaledDotOperation::new(dimensions.clone(), DataType::F64, false, false),
            cases = [{
                input_types = [lhs.clone(), rhs.clone()],
                output_types = [ArrayType::new_static(DataType::F64, [2, 3, 5])],
            }],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = ScaledDotOperation::new(dimensions, DataType::BF16, true, true),
            input_types = [lhs, rhs, lhs_scale, rhs_scale],
        );

        // Ratios that involve a dynamic extent are asserted at runtime, but identical extents are rejected statically.
        let block = Dimension::Dynamic(DimensionVariable::new("block", DimensionBounds::new(1, Some(8)).unwrap()));
        let lhs = ArrayType::new(DataType::F32, Shape::new(vec![2.into(), block.clone()]));
        let rhs = ArrayType::new(DataType::F32, Shape::new(vec![block.clone(), 3.into()]));
        check_operation_type_inference!(
            operation = ScaledDotOperation::new(DotDimensionNumbers::matmul(), DataType::F32, true, false),
            cases = [
                {
                    input_types = [lhs.clone(), rhs.clone(), ArrayType::new_static(DataType::F32, [2, 1])],
                    output_types = [ArrayType::new_static(DataType::F32, [2, 3])],
                },
                {
                    input_types = [lhs.clone(), rhs, lhs],
                    error = format!(
                        "`{SCALED_DOT_OPERATION_NAME}` left contracting dimension 1 must be at least twice as large \
                         as its scale but both have size `block`",
                    ),
                },
            ],
        );
    }

    #[test]
    fn test_scaled_dot_type_inference_manual_variation() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 2).with_varying_manual_axes(["x"]).unwrap();
        let lhs = ArrayType::new_static(DataType::F32, [2, 4]);
        let rhs = ArrayType::new_static(DataType::F32, [4, 2]);
        let scale = ArrayType::new_static(DataType::F32, [2, 2]).with_sharding(sharding.clone()).unwrap();
        check_operation_type_inference!(
            operation = ScaledDotOperation::new(DotDimensionNumbers::matmul(), DataType::F32, true, false),
            cases = [
                {
                    input_types = [lhs.clone(), rhs.clone(), scale.clone()],
                    error = format!(
                        "`{SCALED_DOT_OPERATION_NAME}` inputs must have matching varying manual axes; insert \
                         `parallel_vary` on the inputs that lack an axis, as `align_manual_variation` does",
                    ),
                },
                {
                    input_types = [
                        lhs.with_sharding(sharding.clone()).unwrap(),
                        rhs.with_sharding(sharding.clone()).unwrap(),
                        scale,
                    ],
                    output_types = [ArrayType::new_static(DataType::F32, [2, 2]).with_sharding(sharding).unwrap()],
                },
            ],
        );
    }

    #[test]
    fn test_scaled_dot_interpretation() {
        // Rank-two inputs use the default dimension numbers, which contract the last left dimension with the first
        // right dimension. The two sides infer independent block sizes of two along different scale dimensions.
        let lhs = Array::matrix(2, 4, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let rhs = Array::matrix(4, 3, (1..=12).map(|value| value as f32).collect()).unwrap();
        let lhs_scale = Array::matrix(2, 2, vec![1.0f32, 2.0, 0.5, 1.0]).unwrap();
        let rhs_scale = Array::matrix(2, 3, vec![1.0f32, 1.0, 1.0, 2.0, 2.0, 2.0]).unwrap();
        assert_eq!(
            lhs.scaled_dot(&rhs, Some(&lhs_scale), Some(&rhs_scale), None, Some(DataType::F32)),
            Ok(Array::matrix(2, 3, vec![253.0f32, 284.0, 315.0, 272.5, 308.0, 343.5]).unwrap()),
        );

        // Each scale is independently optional, and an absent scale is the multiplicative identity.
        assert_eq!(
            lhs.scaled_dot(&rhs, Some(&lhs_scale), None, None, Some(DataType::F32)),
            Ok(Array::matrix(2, 3, vec![131.0f32, 148.0, 165.0, 143.5, 164.0, 184.5]).unwrap()),
        );
        assert_eq!(
            lhs.scaled_dot(&rhs, None, None, None, Some(DataType::F32)),
            Ok(Array::matrix(2, 3, vec![70.0f32, 80.0, 90.0, 158.0, 184.0, 210.0]).unwrap()),
        );

        // Without an accumulation type, the contraction accumulates at and returns `f32`, while an explicit `bf16`
        // accumulation type keeps the contraction at the dequantized input type.
        assert_eq!(
            lhs.scaled_dot(&rhs, Some(&lhs_scale), Some(&rhs_scale), None, None),
            Ok(Array::matrix(2, 3, vec![253.0f32, 284.0, 315.0, 272.5, 308.0, 343.5]).unwrap()),
        );
        let product = lhs.scaled_dot(&rhs, None, None, None, Some(DataType::BF16)).unwrap();
        assert_eq!(product.r#type().into_owned(), ArrayType::new_static(DataType::BF16, [2, 3]));
        assert_eq!(product.to_f64s(), vec![70.0, 80.0, 90.0, 158.0, 184.0, 210.0]);

        // The type inference rule rejects invalid inputs before anything is evaluated.
        assert_eq!(
            lhs.scaled_dot(&rhs, Some(&rhs_scale), None, None, None),
            Err(ProgramError::from(TypeError::invalid(format!(
                "`{SCALED_DOT_OPERATION_NAME}` left contracting dimension 1 has size 4, which is not divisible by its \
                 scale size 3",
            )))),
        );

        // Both contracting dimensions of each side carry independent block sizes (two and three on the left), and every
        // scale value is repeated over its two-dimensional block.
        let lhs = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [1, 2, 4, 6]), &[1.0; 48]).unwrap();
        let rhs = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [1, 4, 6, 3]), &[1.0; 72]).unwrap();
        let lhs_scale = Array::from_elements::<f32>(
            ArrayType::new_static(DataType::F32, [1, 2, 2, 2]),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        let rhs_scale =
            Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [1, 2, 2, 3]), &[1.0; 12]).unwrap();
        let dimensions = DotDimensionNumbers::new(vec![2, 3], vec![1, 2], vec![0], vec![0]);
        assert_eq!(
            lhs.scaled_dot(&rhs, Some(&lhs_scale), Some(&rhs_scale), Some(&dimensions), Some(DataType::F32)),
            Ok(Array::from_elements::<f32>(
                ArrayType::new_static(DataType::F32, [1, 2, 3]),
                &[60.0, 60.0, 60.0, 156.0, 156.0, 156.0],
            )
            .unwrap()),
        );
    }

    #[test]
    fn test_scaled_dot_interpretation_array_ir() {
        // Concrete array IR values contract their array members.
        let lhs = Array::matrix(2, 4, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let rhs = Array::matrix(4, 2, vec![1.0f32; 8]).unwrap();
        let lhs_scale = Array::matrix(2, 2, vec![1.0f32, 2.0, 0.5, 1.0]).unwrap();
        let expected = lhs.scaled_dot(&rhs, Some(&lhs_scale), None, None, Some(DataType::F32)).unwrap();
        assert_eq!(
            ArrayIrValue::Array(lhs).scaled_dot(
                &ArrayIrValue::Array(rhs),
                Some(&ArrayIrValue::Array(lhs_scale)),
                None,
                None,
                Some(DataType::F32),
            ),
            Ok(ArrayIrValue::Array(expected)),
        );
    }

    #[test]
    fn test_scaled_dot_interpret_in_parent() {
        // A right-only scale follows the two element arrays, and interpretation returns an array IR member.
        let operation = ScaledDotOperation::new(DotDimensionNumbers::inner_product(), DataType::F32, false, true);
        let context = EagerContext::<ArrayIrValue<Array>>::new();
        let inputs = [
            ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap()),
            ArrayIrValue::Array(Array::vector(vec![1.0f32; 4]).unwrap()),
            ArrayIrValue::Array(Array::vector(vec![2.0f32, 3.0]).unwrap()),
        ];
        assert_eq!(
            operation.interpret_in_parent(&context, &EmptyRegionDriver, &inputs),
            Ok(vec![ArrayIrValue::Array(Array::scalar(27.0f32).unwrap())]),
        );

        // Validate the operation boundary before reading extents or evaluating array operations.
        assert_eq!(
            operation.interpret_in_parent(&context, &EmptyRegionDriver, &inputs[..2]),
            Err(ProgramError::from(TypeError::invalid(format!(
                "`{SCALED_DOT_OPERATION_NAME}` expects 3 inputs but got 2",
            )))),
        );
        assert_eq!(
            operation.interpret_in_parent(
                &context,
                &EmptyRegionDriver,
                &[inputs[0].clone(), inputs[1].clone(), inputs[1].clone()],
            ),
            Err(ProgramError::from(TypeError::invalid(format!(
                "`{SCALED_DOT_OPERATION_NAME}` right contracting dimension 0 must be at least twice as large as its \
                 scale but has ratio 1",
            )))),
        );
    }

    #[test]
    fn test_scaled_dot_interpret_in_parent_dynamic_extents() {
        // A dynamic scale extent is validated at runtime by the staged assertions.
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let lhs = context.input(ArrayType::new_static(DataType::F32, [8]).into());
        let rhs = context.input(ArrayType::new_static(DataType::F32, [8]).into());
        let scale_extent = DimensionVariable::new("scale", DimensionBounds::new(0, Some(5)).unwrap());
        let lhs_scale =
            context.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(scale_extent)])).into());
        let operation = ScaledDotOperation::new(DotDimensionNumbers::inner_product(), DataType::F32, true, false);
        let output = operation
            .interpret_in_parent(&context, &EmptyRegionDriver, &[lhs, rhs, lhs_scale])
            .unwrap()
            .remove(0);
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {r#"
                lambda %0:f32[8], %1:f32[8], %2:f32[scale] .
                let %3:dimension<8> = constant [value=8]
                    %4:bf16[8] = convert_element_type [data_type=bf16] %0
                    %5:dimension<scale ∈ [0, 5)> = dimension_size [axis=0] %2
                    %6:dimension<0> = constant [value=0]
                    %7:dimension<1> = constant [value=1]
                    %8:bool[] = compare [direction=GreaterThan] %5 %6
                    () = assert [message="scale extent must be positive", labels=["scale"]] %8 %5
                    %9:dimension<max(scale, 1) ∈ [1, 5)> = dimension_max %5 %7
                    %10:dimension<8 % max(scale, 1) ∈ [0, 4)> = dimension_rem %3 %9
                    %11:bool[] = compare [direction=Equal] %10 %6
                    () = assert [
                        message="element extent must be divisible by scale extent",
                        labels=["elements", "scale"],
                    ] %11 %3 %5
                    %12:dimension<8 / max(scale, 1) ∈ [2, 9)> = dimension_div %3 %9
                    %13:dimension<2> = constant [value=2]
                    %14:bool[] = const true
                    %15:f32[scale, 8 / max(scale, 1)] = broadcast [output_axes=[0]] %2 %5 %12
                    %16:f32[8] = reshape %15 %3
                    %17:bf16[8] = convert_element_type [data_type=bf16] %16
                    %18:bf16[8] = mul %4 %17
                    %19:bf16[8] = convert_element_type [data_type=bf16] %1
                    %20:f32[] = dot [
                        dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        accumulation_type=f32,
                    ] %18 %19
                in (%20)
            "#}
            .trim_end(),
        );

        // Valid inputs dequantize with the observed block size and contract.
        let ones = ArrayIrValue::Array(Array::vector(vec![1.0f32; 8]).unwrap());
        assert_eq!(
            program.interpret(vec![
                ones.clone(),
                ones.clone(),
                ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap()),
            ]),
            Ok(vec![ArrayIrValue::Array(Array::scalar(20.0f32).unwrap())]),
        );

        // An empty scale reports the positivity assertion rather than a failed remainder or division, and a positive
        // scale extent that does not divide the element extent reports the divisibility assertion. Partially evaluating
        // the program with unknown inputs preserves both diagnostics.
        let evaluation = program
            .partially_evaluate(&program.input_types().into_iter().map(PartialValue::Unknown).collect::<Vec<_>>())
            .unwrap();
        let inputs = vec![ones.clone(), ones.clone(), ArrayIrValue::Array(Array::vector(Vec::<f32>::new()).unwrap())];
        let expected = AssertionError::Failed {
            message: "scale extent must be positive".to_string(),
            observations: vec![("scale".to_string(), "0".to_string())],
        };
        assert_eq!(program.interpret(inputs.clone()).unwrap_err().downcast_custom::<AssertionError>(), Some(&expected));
        assert_eq!(
            evaluation
                .interpret(&EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(), &inputs)
                .unwrap_err()
                .downcast_custom::<AssertionError>(),
            Some(&expected),
        );
        let inputs = vec![ones.clone(), ones, ArrayIrValue::Array(Array::vector(vec![1.0f32; 3]).unwrap())];
        let expected = AssertionError::Failed {
            message: "element extent must be divisible by scale extent".to_string(),
            observations: vec![("elements".to_string(), "8".to_string()), ("scale".to_string(), "3".to_string())],
        };
        assert_eq!(program.interpret(inputs.clone()).unwrap_err().downcast_custom::<AssertionError>(), Some(&expected));
        assert_eq!(
            evaluation
                .interpret(&EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(), &inputs)
                .unwrap_err()
                .downcast_custom::<AssertionError>(),
            Some(&expected),
        );
    }

    #[test]
    fn test_scaled_dot_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = ScaledDotOperation::new(DotDimensionNumbers::inner_product(), DataType::F32, true, false),
            inputs = [
                Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap(),
                Array::vector(vec![1.0f32; 4]).unwrap(),
                Array::vector(vec![2.0f32, 0.5]).unwrap(),
            ],
            expected = Array::scalar(9.5f32).unwrap(),
        );
    }

    #[test]
    fn test_scaled_dot_batching() {
        check_operation_batching!(
            @exact,
            operation = ScaledDotOperation::new(DotDimensionNumbers::inner_product(), DataType::F32, true, true),
            axis_size = 2,
            cases = [
                {
                    // Mapped scales keep one set of block scales per batch item, and replicated inputs are shared.
                    inputs = [
                        (
                            @mapped(axis = 0),
                            Array::matrix(2, 4, vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0]).unwrap(),
                        ),
                        (@replicated, Array::vector(vec![1.0f32; 4]).unwrap()),
                        (@mapped(axis = 1), Array::matrix(2, 2, vec![1.0f32, 2.0, 2.0, 1.0]).unwrap()),
                        (@replicated, Array::vector(vec![1.0f32, 1.0]).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![6.0f32, 13.0]).unwrap())],
                },
                {
                    inputs = [
                        (@replicated, Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap()),
                        (@replicated, Array::vector(vec![1.0f32; 4]).unwrap()),
                        (@replicated, Array::vector(vec![2.0f32, 0.5]).unwrap()),
                        (@replicated, Array::vector(vec![1.0f32, 1.0]).unwrap()),
                    ],
                    outputs = [(@replicated, Array::scalar(9.5f32).unwrap())],
                },
            ],
        );

        // Existing batching dimensions shift past the new leading batching dimension.
        check_operation_batching!(
            @exact,
            operation = ScaledDotOperation::new(
                DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]),
                DataType::F32,
                true,
                false,
            ),
            axis_size = 2,
            cases = [{
                inputs = [
                    (
                        @mapped(axis = 0),
                        Array::from_elements::<f32>(
                            ArrayType::new_static(DataType::F32, [2, 1, 1, 4]),
                            &[1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0],
                        )
                        .unwrap(),
                    ),
                    (
                        @replicated,
                        Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [1, 4, 1]), &[1.0; 4])
                            .unwrap(),
                    ),
                    (
                        @mapped(axis = 0),
                        Array::from_elements::<f32>(
                            ArrayType::new_static(DataType::F32, [2, 1, 1, 2]),
                            &[1.0, 2.0, 2.0, 1.0],
                        )
                        .unwrap(),
                    ),
                ],
                outputs = [(
                    @mapped(axis = 0),
                    Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [2, 1, 1, 1]), &[6.0, 13.0])
                        .unwrap(),
                )],
            }],
        );
    }

    #[test]
    fn test_scaled_dot_batching_validates_original_dimensions() {
        let input = ArrayBatch::new(Array::matrix(2, 1, vec![1.0f32, 2.0]).unwrap(), BatchAxis::new(0)).unwrap();
        let operation = ScaledDotOperation::new(
            DotDimensionNumbers::new(vec![usize::MAX], vec![0], Vec::new(), Vec::new()),
            DataType::F32,
            false,
            false,
        );
        assert_eq!(
            operation.batch(
                &BatchingContext::new(EagerContext::<Array>::new(), 2),
                &EmptyRegionDriver,
                &[input.clone(), input],
            ),
            Err(BatchingError::Type(TypeError::invalid(format!(
                "`{SCALED_DOT_OPERATION_NAME}` left-hand side dimension index out of bounds",
            )))),
        );
    }

    #[test]
    fn test_scaled_dot_batching_validates_mapped_extents() {
        let lhs = ArrayBatch::new(Array::matrix(2, 4, vec![1.0f32; 8]).unwrap(), BatchAxis::new(0)).unwrap();
        let rhs = ArrayBatch::new(Array::matrix(3, 4, vec![1.0f32; 12]).unwrap(), BatchAxis::new(0)).unwrap();
        let operation = ScaledDotOperation::new(DotDimensionNumbers::inner_product(), DataType::F32, false, false);
        assert_eq!(
            operation.batch(&BatchingContext::new(EagerContext::<Array>::new(), 2), &EmptyRegionDriver, &[lhs, rhs]),
            Err(BatchingError::MismatchedBatchSizes { expected: 2, actual: 3 }),
        );
    }

    #[test]
    fn test_scaled_dot_differentiation() {
        // Scaled dot has no differentiation rule, so it rejects live tangents instead of treating its scales as
        // constants.
        let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let dual = DifferentiationDual::new(input.clone(), input).unwrap();
        assert!(matches!(
            ScaledDotOperation::new(DotDimensionNumbers::inner_product(), DataType::F32, false, false).jvp(
                &DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new()),
                &EmptyRegionDriver,
                &[dual.clone(), dual],
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == format!("`{SCALED_DOT_OPERATION_NAME}` does not support differentiation"),
        ));
    }

    #[test]
    fn test_scaled_dot_scaled_matmul() {
        // Both inputs contract their last dimension, and the output defaults to `f32`.
        let lhs = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [1, 1, 4]), &[1.0, 2.0, 3.0, 4.0])
            .unwrap();
        let rhs = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [1, 2, 4]), &[1.0; 8]).unwrap();
        let lhs_scale =
            Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [1, 1, 2]), &[1.0, 2.0]).unwrap();
        let rhs_scale =
            Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [1, 2, 2]), &[1.0, 1.0, 2.0, 2.0])
                .unwrap();
        assert_eq!(
            lhs.scaled_matmul(&rhs, &lhs_scale, &rhs_scale, None),
            Ok(Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [1, 1, 2]), &[17.0, 34.0]).unwrap()),
        );

        // Each side infers its own block size, so the right scale may use larger blocks than the left one.
        let rhs_scale =
            Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [1, 2, 1]), &[3.0, 1.0]).unwrap();
        assert_eq!(
            lhs.scaled_matmul(&rhs, &lhs_scale, &rhs_scale, Some(DataType::F64)),
            Ok(Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [1, 1, 2]), &[51.0, 17.0]).unwrap()),
        );

        // Inputs of any other rank are rejected.
        let matrix = Array::matrix(2, 2, vec![1.0f32; 4]).unwrap();
        assert_eq!(
            matrix.scaled_matmul(&matrix, &matrix, &matrix, None),
            Err(ProgramError::from(TypeError::invalid("`scaled_matmul` expects rank-3 inputs"))),
        );
    }

    #[test]
    fn test_block_scale_format() {
        let format = BlockScaleFormat::new(4, DataType::F8E5M2, DataType::F8E8M0FNU);
        assert_eq!(format.block_size(), 4);
        assert_eq!(format.element_type(), DataType::F8E5M2);
        assert_eq!(format.scale_type(), DataType::F8E8M0FNU);
        assert!(!format.has_global_scale());
        assert_eq!(format.largest_unclipped_value(), Ok(57344.0 * 127.0f64.exp2()));

        // Only formats with `f8e4m3fn` scales use global scales, which bound their scales at `448` rather than `2^127`.
        let nvfp4 = BlockScaleFormat::nvfp4();
        assert_eq!(nvfp4, BlockScaleFormat::new(16, DataType::F4E2M1FN, DataType::F8E4M3FN));
        assert!(nvfp4.has_global_scale());
        assert_eq!(nvfp4.largest_unclipped_value(), Ok(2688.0));
        let mxfp8 = BlockScaleFormat::mxfp8();
        assert_eq!(mxfp8, BlockScaleFormat::new(32, DataType::F8E4M3FN, DataType::F8E8M0FNU));
        assert!(!mxfp8.has_global_scale());
        assert_eq!(mxfp8.largest_unclipped_value(), Ok(448.0 * 127.0f64.exp2()));
        let mxfp4 = BlockScaleFormat::mxfp4();
        assert_eq!(mxfp4, BlockScaleFormat::new(32, DataType::F4E2M1FN, DataType::F8E8M0FNU));
        assert!(!mxfp4.has_global_scale());
        assert_eq!(mxfp4.largest_unclipped_value(), Ok(6.0 * 127.0f64.exp2()));

        // Construction does not validate formats, but element types without a grid have no largest value.
        assert_eq!(
            BlockScaleFormat::new(4, DataType::F16, DataType::F8E4M3FN).largest_unclipped_value(),
            Err(TypeError::invalid("`block_quantize` does not support element data type `f16`")),
        );
    }

    #[test]
    fn test_block_scaled_dot_configuration() {
        // The JAX defaults use the provided format everywhere, produce `f32` outputs, take provided global scales, and
        // enable none of the options of the NVIDIA recipe.
        let dimensions = DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new());
        let (nvfp4, mxfp8, mxfp4) = (BlockScaleFormat::nvfp4(), BlockScaleFormat::mxfp8(), BlockScaleFormat::mxfp4());
        let configuration = BlockScaledDotConfiguration::new(dimensions.clone(), mxfp8);
        assert_eq!(configuration.dimensions(), &dimensions);
        assert_eq!(configuration.accumulation_type(), DataType::F32);
        assert_eq!(configuration.lhs_format(), mxfp8);
        assert_eq!(configuration.rhs_format(), mxfp8);
        assert_eq!(configuration.gradient_format(), mxfp8);
        assert_eq!(configuration.global_scaling(), BlockScaledGlobalScaling::Provided);
        assert_eq!(BlockScaledGlobalScaling::default(), BlockScaledGlobalScaling::Provided);
        assert!(configuration.quantized_backward());
        assert!(!configuration.stochastic_gradient_rounding());
        assert!(!configuration.hadamard_transform());
        assert!(!configuration.tiled_rhs_scales());

        // Each builder sets the value that the corresponding accessor returns.
        let configuration = configuration
            .with_accumulation_type(DataType::BF16)
            .with_lhs_format(nvfp4)
            .with_rhs_format(mxfp4)
            .with_gradient_format(nvfp4)
            .with_global_scaling(BlockScaledGlobalScaling::Current)
            .with_quantized_backward(true)
            .with_stochastic_gradient_rounding(true)
            .with_hadamard_transform(true)
            .with_tiled_rhs_scales(true);
        assert_eq!(configuration.dimensions(), &dimensions);
        assert_eq!(configuration.accumulation_type(), DataType::BF16);
        assert_eq!(configuration.lhs_format(), nvfp4);
        assert_eq!(configuration.rhs_format(), mxfp4);
        assert_eq!(configuration.gradient_format(), nvfp4);
        assert_eq!(configuration.global_scaling(), BlockScaledGlobalScaling::Current);
        assert!(configuration.quantized_backward());
        assert!(configuration.stochastic_gradient_rounding());
        assert!(configuration.hadamard_transform());
        assert!(configuration.tiled_rhs_scales());

        // Unless overridden, the backward contractions are quantized exactly when the gradient format has no global
        // scale, regardless of the input formats.
        assert!(!BlockScaledDotConfiguration::new(dimensions.clone(), nvfp4).quantized_backward());
        assert!(BlockScaledDotConfiguration::new(dimensions.clone(), mxfp4).quantized_backward());
        assert!(
            !BlockScaledDotConfiguration::new(dimensions.clone(), mxfp8)
                .with_gradient_format(nvfp4)
                .quantized_backward(),
        );
        assert!(
            BlockScaledDotConfiguration::new(dimensions.clone(), nvfp4)
                .with_gradient_format(mxfp8)
                .quantized_backward(),
        );
        assert!(
            BlockScaledDotConfiguration::new(dimensions.clone(), nvfp4)
                .with_quantized_backward(true)
                .quantized_backward(),
        );
        assert!(
            !BlockScaledDotConfiguration::new(dimensions, mxfp8)
                .with_quantized_backward(false)
                .quantized_backward(),
        );
    }

    #[test]
    fn test_block_scaled_dot_configuration_nvfp4_training() {
        // The NVIDIA recipe quantizes all three contractions with NVFP4 and current global scales, rounds gradients
        // stochastically, transforms the weight gradient inputs, and tiles the scales of the weight.
        let dimensions = DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new());
        let configuration = BlockScaledDotConfiguration::nvfp4_training(dimensions.clone());
        assert_eq!(
            configuration,
            BlockScaledDotConfiguration::new(dimensions, BlockScaleFormat::nvfp4())
                .with_global_scaling(BlockScaledGlobalScaling::Current)
                .with_quantized_backward(true)
                .with_stochastic_gradient_rounding(true)
                .with_hadamard_transform(true)
                .with_tiled_rhs_scales(true),
        );

        // The same random state reproduces the gradients of the recipe, while a different random state changes them.
        let function = block_scaled_dot::<EagerArrayContext>(configuration).unwrap();
        let values = |rows: usize, columns: usize, phase: f64| {
            Array::from_elements::<f64>(
                ArrayType::new_static(DataType::F64, [rows, columns]),
                &(0..rows * columns).map(|index| (index as f64 * 0.37 + phase).sin()).collect::<Vec<_>>(),
            )
            .unwrap()
            .convert_element_type(DataType::F32)
            .unwrap()
        };
        let (lhs, rhs, seed) = (values(32, 32, 0.1), values(16, 32, 0.7), values(32, 16, 1.3));
        let gradients = |state: u64| {
            let state = Array::from_elements(RandomAlgorithm::ThreeFry.state_type(), &[state, 0u64]).unwrap();
            let (_, pullback) = differentiate_at((lhs.clone(), rhs.clone()))
                .with_captures(state)
                .vjp(|(lhs, rhs), state| function.call(BlockScaledDotInputs::new(lhs, rhs).with_random_state(state)))
                .unwrap();
            pullback.apply_materialized(seed.clone()).unwrap()
        };
        let (lhs_cotangent, rhs_cotangent) = gradients(1);
        assert_eq!(gradients(1), (lhs_cotangent.clone(), rhs_cotangent.clone()));
        assert_ne!(gradients(2).0, lhs_cotangent);

        // The gradients equal a reference built from the public quantization capabilities. The input gradient
        // contracts the seed, quantized along `N` with the first noise draw, with the tiled weight, which forward
        // quantization also uses. The weight gradient contracts the Hadamard transforms of the seed and the left input,
        // quantized along `M` with the second noise draw. Every tensor uses its current global scale.
        let quantize = |value: &Array, axes: Vec<isize>, noise: Option<&Array>| {
            let global_scale = value
                .abs()
                .unwrap()
                .reduce(&(0..value.r#type().rank()).collect::<Vec<_>>(), ReductionKind::Max)
                .unwrap()
                .div(&Array::scalar(6.0f32 * 448.0).unwrap())
                .unwrap();
            let (elements, scales) = value
                .block_quantize_with_options(
                    16,
                    DataType::F4E2M1FN,
                    DataType::F8E4M3FN,
                    axes.clone(),
                    Some(&global_scale),
                    noise,
                )
                .unwrap();
            let dequantized = elements.block_dequantize(&scales, axes, DataType::F32).unwrap();
            dequantized.mul(&global_scale.broadcast(dequantized.r#type().into_owned(), &[]).unwrap()).unwrap()
        };
        let rank_three = |value: &Array| {
            let dimensions = value.r#type().static_shape().unwrap().dimensions().to_vec();
            value
                .reshape(Shape::new(vec![
                    Dimension::Static(1),
                    Dimension::Static(dimensions[0]),
                    Dimension::Static(dimensions[1]),
                ]))
                .unwrap()
        };
        let noise_type = ArrayType::new_static(DataType::F32, [1, 32, 16]);
        let state = Array::from_elements(RandomAlgorithm::ThreeFry.state_type(), &[1u64, 0]).unwrap();
        let (state, input_gradient_noise) = state.random_uniform(&noise_type).unwrap();
        let (_, weight_gradient_noise) = state.random_uniform(&noise_type).unwrap();
        let weight = quantize(&rank_three(&rhs), vec![1, 2], None);
        let expected_lhs = quantize(&rank_three(&seed), vec![2], Some(&input_gradient_noise))
            .dot(&weight, &DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]))
            .unwrap();
        let rules =
            BlockScaledDotRules::new(function.configuration(), &BlockScaledDotInputs::new(lhs.clone(), rhs.clone()))
                .unwrap();
        let expected_rhs =
            quantize(&rules.hadamard_transform(&rank_three(&seed), 32).unwrap(), vec![1], Some(&weight_gradient_noise))
                .dot(
                    &quantize(&rules.hadamard_transform(&rank_three(&lhs), 32).unwrap(), vec![1], None),
                    &DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0]),
                )
                .unwrap();
        let close = |actual: &Array, expected: &Array| {
            let expected = expected.to_f64s();
            let magnitude = expected.iter().fold(0.0f64, |maximum, &value| maximum.max(f64::abs(value)));
            for (actual, expected) in actual.to_f64s().into_iter().zip(expected) {
                assert_abs_diff_eq!(actual, expected, epsilon = 1e-5 * magnitude);
            }
        };
        close(&lhs_cotangent, &expected_lhs);
        close(&rhs_cotangent, &expected_rhs);
    }

    #[test]
    fn test_block_scaled_dot_inputs() {
        let lhs = Array::matrix(1, 2, vec![1.0f32, 2.0]).unwrap();
        let rhs = Array::matrix(2, 1, vec![3.0f32, 4.0]).unwrap();
        let state = Array::from_elements(RandomAlgorithm::ThreeFry.state_type(), &[1u64, 0]).unwrap();
        let inputs = BlockScaledDotInputs::new(lhs.clone(), rhs.clone());
        assert_eq!(
            inputs,
            BlockScaledDotInputs {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                lhs_global_scale: None,
                rhs_global_scale: None,
                gradient_global_scale: None,
                random_state: None,
            },
        );

        // Each builder sets its own optional leaf.
        assert_eq!(
            inputs
                .with_lhs_global_scale(Array::scalar(2.0f32).unwrap())
                .with_rhs_global_scale(Array::scalar(3.0f32).unwrap())
                .with_gradient_global_scale(Array::scalar(4.0f32).unwrap())
                .with_random_state(state.clone()),
            BlockScaledDotInputs {
                lhs,
                rhs,
                lhs_global_scale: Some(Array::scalar(2.0f32).unwrap()),
                rhs_global_scale: Some(Array::scalar(3.0f32).unwrap()),
                gradient_global_scale: Some(Array::scalar(4.0f32).unwrap()),
                random_state: Some(state),
            },
        );
    }

    #[test]
    fn test_block_scaled_dot() {
        // Every block of four contracting values contains `6`, so NVFP4 scales are `1` and quantization is exact.
        let nvfp4 = BlockScaleFormat::new(4, DataType::F4E2M1FN, DataType::F8E4M3FN);
        let configuration = BlockScaledDotConfiguration::new(DotDimensionNumbers::matmul(), nvfp4);
        let function = block_scaled_dot::<EagerArrayContext>(configuration.clone()).unwrap();
        assert_eq!(function.configuration(), &configuration);
        let lhs = Array::matrix(
            2,
            8,
            vec![6.0f32, 1.0, 0.5, 2.0, 3.0, 6.0, 1.5, 4.0, 1.0, 6.0, 2.0, 3.0, 6.0, 0.5, 1.0, 4.0],
        )
        .unwrap();
        let rhs = Array::matrix(
            8,
            3,
            vec![
                6.0f32, 1.0, 0.5, 1.0, 6.0, 1.0, 2.0, 1.0, 6.0, 3.0, 1.0, 2.0, 6.0, 2.0, 1.0, 1.0, 6.0, 1.0, 1.0, 3.0,
                1.0, 1.0, 4.0, 6.0,
            ],
        )
        .unwrap();
        let expected = lhs.dot(&rhs, &DotDimensionNumbers::matmul()).unwrap();
        assert_eq!(function.call(BlockScaledDotInputs::new(lhs.clone(), rhs.clone())), Ok(expected.clone()));

        // A provided global scale divides its input before quantization and multiplies the product afterwards, so a
        // doubled input with a global scale of `2` quantizes exactly and yields the doubled product.
        assert_eq!(
            function.call(
                BlockScaledDotInputs::new(
                    lhs.mul(&Array::matrix(2, 8, vec![2.0f32; 16]).unwrap()).unwrap(),
                    rhs.clone(),
                )
                .with_lhs_global_scale(Array::scalar(2.0f32).unwrap()),
            ),
            Ok(expected.mul(&Array::matrix(2, 3, vec![2.0f32; 6]).unwrap()).unwrap()),
        );

        // MXFP8 contractions accumulate at the output type directly, unless `ScaledDot` cannot accumulate at it, as for
        // `f16`, where they accumulate at the compute type and convert afterwards.
        let mxfp8 = BlockScaleFormat::new(4, DataType::F8E4M3FN, DataType::F8E8M0FNU);
        let function = block_scaled_dot::<EagerArrayContext>(
            BlockScaledDotConfiguration::new(DotDimensionNumbers::matmul(), mxfp8)
                .with_accumulation_type(DataType::F64),
        )
        .unwrap();
        assert_eq!(
            function.call(BlockScaledDotInputs::new(lhs.clone(), rhs.clone())),
            Ok(expected.convert_element_type(DataType::F64).unwrap()),
        );
        let function = block_scaled_dot::<EagerArrayContext>(
            BlockScaledDotConfiguration::new(DotDimensionNumbers::matmul(), mxfp8)
                .with_accumulation_type(DataType::F16),
        )
        .unwrap();
        assert_eq!(
            function.call(BlockScaledDotInputs::new(lhs.clone(), rhs.clone())),
            Ok(expected.convert_element_type(DataType::F16).unwrap()),
        );

        // NVFP4 contractions accumulate at the compute type even without global scales and convert to the output type
        // afterwards, as in JAX.
        let function = block_scaled_dot::<EagerArrayContext>(
            BlockScaledDotConfiguration::new(DotDimensionNumbers::matmul(), nvfp4)
                .with_accumulation_type(DataType::BF16),
        )
        .unwrap();
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(lhs, rhs): (Tracer<_>, Tracer<_>)| function.call(BlockScaledDotInputs::new(lhs, rhs)),
            (ArrayType::new_static(DataType::F32, [2, 8]), ArrayType::new_static(DataType::F32, [8, 3])),
        )
        .unwrap();
        assert_eq!(output_type, ArrayType::new_static(DataType::BF16, [2, 3]));
        assert_eq!(
            program.to_string(),
            indoc! {r#"
                lambda %0:f32[2, 8], %1:f32[8, 3] .
                let %2:bf16[2, 3] = composite [name="block_scaled_dot"] %0 %1 [
                    primal={
                        lambda %0:f32[2, 8], %1:f32[8, 3] .
                        let %2:f32[1, 2, 8] = reshape [shape=[1, 2, 8]] %0
                            %3:f32[3, 8] = transpose [permutation=[1, 0]] %1
                            %4:f32[1, 3, 8] = reshape [shape=[1, 3, 8]] %3
                            %5:f32[1, 2, 2, 4] = reshape [shape=[1, 2, 2, 4]] %2
                            %6:f32[1, 2, 2, 4] = abs %5
                            %7:f32[1, 2, 2] = reduce [kind=max, axes=[3]] %6
                            %8:f32[] = constant [value=6.0]
                            %9:f32[1, 2, 2] = broadcast [output_type=f32[1, 2, 2], output_axes=[]] %8
                            %10:f32[1, 2, 2] = div %7 %9
                            %11:f32[] = constant [value=0.001953125]
                            %12:f32[] = constant [value=448.0]
                            %13:f32[1, 2, 2] = clamp %11 %10 %12
                            %14:u32[1, 2, 2] = convert_element_type [data_type=u32, bitcast=true] %13
                            %15:f32[] = constant [value=inf]
                            %16:f32[1, 2, 2] = broadcast [output_type=f32[1, 2, 2], output_axes=[]] %15
                            %17:u32[1, 2, 2] = convert_element_type [data_type=u32, bitcast=true] %16
                            %18:u32[1, 2, 2] = and %14 %17
                            %19:f32[1, 2, 2] = convert_element_type [data_type=f32, bitcast=true] %18
                            %20:f32[] = constant [value=0.015625]
                            %21:f32[1, 2, 2] = broadcast [output_type=f32[1, 2, 2], output_axes=[]] %20
                            %22:f32[] = constant [value=inf]
                            %23:f32[1, 2, 2] = broadcast [output_type=f32[1, 2, 2], output_axes=[]] %22
                            %24:f32[1, 2, 2] = clamp %21 %19 %23
                            %25:f32[] = constant [value=0.125]
                            %26:f32[1, 2, 2] = broadcast [output_type=f32[1, 2, 2], output_axes=[]] %25
                            %27:f32[1, 2, 2] = mul %24 %26
                            %28:f32[1, 2, 2] = div %13 %27
                            %29:f32[1, 2, 2] = round %28
                            %30:f32[1, 2, 2] = mul %29 %27
                            %31:f8e4m3fn[1, 2, 2] = convert_element_type [data_type=f8e4m3fn] %30
                            %32:f32[1, 2, 2, 4] = broadcast [output_type=f32[1, 2, 2, 4], output_axes=[0, 1, 2]] %30
                            %33:f32[1, 2, 8] = reshape [shape=[1, 2, 8]] %32
                            %34:f32[1, 2, 8] = div %2 %33
                            %35:f32[] = constant [value=-6.0]
                            %36:f32[] = constant [value=6.0]
                            %37:f32[1, 2, 8] = clamp %35 %34 %36
                            %38:u32[1, 2, 8] = convert_element_type [data_type=u32, bitcast=true] %37
                            %39:f32[] = constant [value=inf]
                            %40:f32[1, 2, 8] = broadcast [output_type=f32[1, 2, 8], output_axes=[]] %39
                            %41:u32[1, 2, 8] = convert_element_type [data_type=u32, bitcast=true] %40
                            %42:u32[1, 2, 8] = and %38 %41
                            %43:f32[1, 2, 8] = convert_element_type [data_type=f32, bitcast=true] %42
                            %44:f32[] = constant [value=1.0]
                            %45:f32[1, 2, 8] = broadcast [output_type=f32[1, 2, 8], output_axes=[]] %44
                            %46:f32[] = constant [value=inf]
                            %47:f32[1, 2, 8] = broadcast [output_type=f32[1, 2, 8], output_axes=[]] %46
                            %48:f32[1, 2, 8] = clamp %45 %43 %47
                            %49:f32[] = constant [value=0.5]
                            %50:f32[1, 2, 8] = broadcast [output_type=f32[1, 2, 8], output_axes=[]] %49
                            %51:f32[1, 2, 8] = mul %48 %50
                            %52:f32[1, 2, 8] = div %37 %51
                            %53:f32[1, 2, 8] = round %52
                            %54:f32[1, 2, 8] = mul %53 %51
                            %55:f4e2m1fn[1, 2, 8] = convert_element_type [data_type=f4e2m1fn] %54
                            %56:f32[1, 3, 2, 4] = reshape [shape=[1, 3, 2, 4]] %4
                            %57:f32[1, 3, 2, 4] = abs %56
                            %58:f32[1, 3, 2] = reduce [kind=max, axes=[3]] %57
                            %59:f32[] = constant [value=6.0]
                            %60:f32[1, 3, 2] = broadcast [output_type=f32[1, 3, 2], output_axes=[]] %59
                            %61:f32[1, 3, 2] = div %58 %60
                            %62:f32[] = constant [value=0.001953125]
                            %63:f32[] = constant [value=448.0]
                            %64:f32[1, 3, 2] = clamp %62 %61 %63
                            %65:u32[1, 3, 2] = convert_element_type [data_type=u32, bitcast=true] %64
                            %66:f32[] = constant [value=inf]
                            %67:f32[1, 3, 2] = broadcast [output_type=f32[1, 3, 2], output_axes=[]] %66
                            %68:u32[1, 3, 2] = convert_element_type [data_type=u32, bitcast=true] %67
                            %69:u32[1, 3, 2] = and %65 %68
                            %70:f32[1, 3, 2] = convert_element_type [data_type=f32, bitcast=true] %69
                            %71:f32[] = constant [value=0.015625]
                            %72:f32[1, 3, 2] = broadcast [output_type=f32[1, 3, 2], output_axes=[]] %71
                            %73:f32[] = constant [value=inf]
                            %74:f32[1, 3, 2] = broadcast [output_type=f32[1, 3, 2], output_axes=[]] %73
                            %75:f32[1, 3, 2] = clamp %72 %70 %74
                            %76:f32[] = constant [value=0.125]
                            %77:f32[1, 3, 2] = broadcast [output_type=f32[1, 3, 2], output_axes=[]] %76
                            %78:f32[1, 3, 2] = mul %75 %77
                            %79:f32[1, 3, 2] = div %64 %78
                            %80:f32[1, 3, 2] = round %79
                            %81:f32[1, 3, 2] = mul %80 %78
                            %82:f8e4m3fn[1, 3, 2] = convert_element_type [data_type=f8e4m3fn] %81
                            %83:f32[1, 3, 2, 4] = broadcast [output_type=f32[1, 3, 2, 4], output_axes=[0, 1, 2]] %81
                            %84:f32[1, 3, 8] = reshape [shape=[1, 3, 8]] %83
                            %85:f32[1, 3, 8] = div %4 %84
                            %86:f32[] = constant [value=-6.0]
                            %87:f32[] = constant [value=6.0]
                            %88:f32[1, 3, 8] = clamp %86 %85 %87
                            %89:u32[1, 3, 8] = convert_element_type [data_type=u32, bitcast=true] %88
                            %90:f32[] = constant [value=inf]
                            %91:f32[1, 3, 8] = broadcast [output_type=f32[1, 3, 8], output_axes=[]] %90
                            %92:u32[1, 3, 8] = convert_element_type [data_type=u32, bitcast=true] %91
                            %93:u32[1, 3, 8] = and %89 %92
                            %94:f32[1, 3, 8] = convert_element_type [data_type=f32, bitcast=true] %93
                            %95:f32[] = constant [value=1.0]
                            %96:f32[1, 3, 8] = broadcast [output_type=f32[1, 3, 8], output_axes=[]] %95
                            %97:f32[] = constant [value=inf]
                            %98:f32[1, 3, 8] = broadcast [output_type=f32[1, 3, 8], output_axes=[]] %97
                            %99:f32[1, 3, 8] = clamp %96 %94 %98
                            %100:f32[] = constant [value=0.5]
                            %101:f32[1, 3, 8] = broadcast [output_type=f32[1, 3, 8], output_axes=[]] %100
                            %102:f32[1, 3, 8] = mul %99 %101
                            %103:f32[1, 3, 8] = div %88 %102
                            %104:f32[1, 3, 8] = round %103
                            %105:f32[1, 3, 8] = mul %104 %102
                            %106:f4e2m1fn[1, 3, 8] = convert_element_type [data_type=f4e2m1fn] %105
                            %107:f32[1, 2, 3] = scaled_dot [
                                dimensions=(lhs_contracting=[2], rhs_contracting=[2], lhs_batching=[0], rhs_batching=[0]),
                                accumulation_type=f32,
                                lhs_scale=true,
                                rhs_scale=true,
                            ] %55 %106 %31 %82
                            %108:bf16[1, 2, 3] = convert_element_type [data_type=bf16] %107
                            %109:bf16[2, 3] = reshape [shape=[2, 3]] %108
                        in (%109)
                    },
                ]
                in (%2)
            "#}
            .trim_end(),
        );
        assert_eq!(
            function.call(BlockScaledDotInputs::new(lhs, rhs)),
            Ok(expected.convert_element_type(DataType::BF16).unwrap()),
        );
    }

    #[test]
    fn test_block_scaled_dot_staging() {
        let nvfp4 = BlockScaleFormat::new(4, DataType::F4E2M1FN, DataType::F8E4M3FN);
        let function = block_scaled_dot::<EagerArrayContext>(BlockScaledDotConfiguration::new(
            DotDimensionNumbers::matmul(),
            nvfp4,
        ))
        .unwrap();
        let (output_type, program) = DomainTracingContext::<EagerArrayContext>::trace(
            |(lhs, rhs)| function.call(BlockScaledDotInputs::new(lhs, rhs)),
            (ArrayType::new_static(DataType::BF16, [2, 8]), ArrayType::new_static(DataType::BF16, [8, 4])),
        )
        .unwrap();
        assert_eq!(output_type, ArrayType::new_static(DataType::F32, [2, 4]));
        assert_eq!(
            program.to_string(),
            indoc! {r#"
                lambda %0:bf16[2, 8], %1:bf16[8, 4] .
                let %2:f32[2, 4] = composite [name="block_scaled_dot"] %0 %1 [
                    primal={
                        lambda %0:bf16[2, 8], %1:bf16[8, 4] .
                        let %2:f32[2, 8] = convert_element_type [data_type=f32] %0
                            %3:f32[1, 2, 8] = reshape [shape=[1, 2, 8]] %2
                            %4:f32[8, 4] = convert_element_type [data_type=f32] %1
                            %5:f32[4, 8] = transpose [permutation=[1, 0]] %4
                            %6:f32[1, 4, 8] = reshape [shape=[1, 4, 8]] %5
                            %7:f32[1, 2, 2, 4] = reshape [shape=[1, 2, 2, 4]] %3
                            %8:f32[1, 2, 2, 4] = abs %7
                            %9:f32[1, 2, 2] = reduce [kind=max, axes=[3]] %8
                            %10:f32[] = constant [value=6.0]
                            %11:f32[1, 2, 2] = broadcast [output_type=f32[1, 2, 2], output_axes=[]] %10
                            %12:f32[1, 2, 2] = div %9 %11
                            %13:f32[] = constant [value=0.001953125]
                            %14:f32[] = constant [value=448.0]
                            %15:f32[1, 2, 2] = clamp %13 %12 %14
                            %16:u32[1, 2, 2] = convert_element_type [data_type=u32, bitcast=true] %15
                            %17:f32[] = constant [value=inf]
                            %18:f32[1, 2, 2] = broadcast [output_type=f32[1, 2, 2], output_axes=[]] %17
                            %19:u32[1, 2, 2] = convert_element_type [data_type=u32, bitcast=true] %18
                            %20:u32[1, 2, 2] = and %16 %19
                            %21:f32[1, 2, 2] = convert_element_type [data_type=f32, bitcast=true] %20
                            %22:f32[] = constant [value=0.015625]
                            %23:f32[1, 2, 2] = broadcast [output_type=f32[1, 2, 2], output_axes=[]] %22
                            %24:f32[] = constant [value=inf]
                            %25:f32[1, 2, 2] = broadcast [output_type=f32[1, 2, 2], output_axes=[]] %24
                            %26:f32[1, 2, 2] = clamp %23 %21 %25
                            %27:f32[] = constant [value=0.125]
                            %28:f32[1, 2, 2] = broadcast [output_type=f32[1, 2, 2], output_axes=[]] %27
                            %29:f32[1, 2, 2] = mul %26 %28
                            %30:f32[1, 2, 2] = div %15 %29
                            %31:f32[1, 2, 2] = round %30
                            %32:f32[1, 2, 2] = mul %31 %29
                            %33:f8e4m3fn[1, 2, 2] = convert_element_type [data_type=f8e4m3fn] %32
                            %34:f32[1, 2, 2, 4] = broadcast [output_type=f32[1, 2, 2, 4], output_axes=[0, 1, 2]] %32
                            %35:f32[1, 2, 8] = reshape [shape=[1, 2, 8]] %34
                            %36:f32[1, 2, 8] = div %3 %35
                            %37:f32[] = constant [value=-6.0]
                            %38:f32[] = constant [value=6.0]
                            %39:f32[1, 2, 8] = clamp %37 %36 %38
                            %40:u32[1, 2, 8] = convert_element_type [data_type=u32, bitcast=true] %39
                            %41:f32[] = constant [value=inf]
                            %42:f32[1, 2, 8] = broadcast [output_type=f32[1, 2, 8], output_axes=[]] %41
                            %43:u32[1, 2, 8] = convert_element_type [data_type=u32, bitcast=true] %42
                            %44:u32[1, 2, 8] = and %40 %43
                            %45:f32[1, 2, 8] = convert_element_type [data_type=f32, bitcast=true] %44
                            %46:f32[] = constant [value=1.0]
                            %47:f32[1, 2, 8] = broadcast [output_type=f32[1, 2, 8], output_axes=[]] %46
                            %48:f32[] = constant [value=inf]
                            %49:f32[1, 2, 8] = broadcast [output_type=f32[1, 2, 8], output_axes=[]] %48
                            %50:f32[1, 2, 8] = clamp %47 %45 %49
                            %51:f32[] = constant [value=0.5]
                            %52:f32[1, 2, 8] = broadcast [output_type=f32[1, 2, 8], output_axes=[]] %51
                            %53:f32[1, 2, 8] = mul %50 %52
                            %54:f32[1, 2, 8] = div %39 %53
                            %55:f32[1, 2, 8] = round %54
                            %56:f32[1, 2, 8] = mul %55 %53
                            %57:f4e2m1fn[1, 2, 8] = convert_element_type [data_type=f4e2m1fn] %56
                            %58:f32[1, 4, 2, 4] = reshape [shape=[1, 4, 2, 4]] %6
                            %59:f32[1, 4, 2, 4] = abs %58
                            %60:f32[1, 4, 2] = reduce [kind=max, axes=[3]] %59
                            %61:f32[] = constant [value=6.0]
                            %62:f32[1, 4, 2] = broadcast [output_type=f32[1, 4, 2], output_axes=[]] %61
                            %63:f32[1, 4, 2] = div %60 %62
                            %64:f32[] = constant [value=0.001953125]
                            %65:f32[] = constant [value=448.0]
                            %66:f32[1, 4, 2] = clamp %64 %63 %65
                            %67:u32[1, 4, 2] = convert_element_type [data_type=u32, bitcast=true] %66
                            %68:f32[] = constant [value=inf]
                            %69:f32[1, 4, 2] = broadcast [output_type=f32[1, 4, 2], output_axes=[]] %68
                            %70:u32[1, 4, 2] = convert_element_type [data_type=u32, bitcast=true] %69
                            %71:u32[1, 4, 2] = and %67 %70
                            %72:f32[1, 4, 2] = convert_element_type [data_type=f32, bitcast=true] %71
                            %73:f32[] = constant [value=0.015625]
                            %74:f32[1, 4, 2] = broadcast [output_type=f32[1, 4, 2], output_axes=[]] %73
                            %75:f32[] = constant [value=inf]
                            %76:f32[1, 4, 2] = broadcast [output_type=f32[1, 4, 2], output_axes=[]] %75
                            %77:f32[1, 4, 2] = clamp %74 %72 %76
                            %78:f32[] = constant [value=0.125]
                            %79:f32[1, 4, 2] = broadcast [output_type=f32[1, 4, 2], output_axes=[]] %78
                            %80:f32[1, 4, 2] = mul %77 %79
                            %81:f32[1, 4, 2] = div %66 %80
                            %82:f32[1, 4, 2] = round %81
                            %83:f32[1, 4, 2] = mul %82 %80
                            %84:f8e4m3fn[1, 4, 2] = convert_element_type [data_type=f8e4m3fn] %83
                            %85:f32[1, 4, 2, 4] = broadcast [output_type=f32[1, 4, 2, 4], output_axes=[0, 1, 2]] %83
                            %86:f32[1, 4, 8] = reshape [shape=[1, 4, 8]] %85
                            %87:f32[1, 4, 8] = div %6 %86
                            %88:f32[] = constant [value=-6.0]
                            %89:f32[] = constant [value=6.0]
                            %90:f32[1, 4, 8] = clamp %88 %87 %89
                            %91:u32[1, 4, 8] = convert_element_type [data_type=u32, bitcast=true] %90
                            %92:f32[] = constant [value=inf]
                            %93:f32[1, 4, 8] = broadcast [output_type=f32[1, 4, 8], output_axes=[]] %92
                            %94:u32[1, 4, 8] = convert_element_type [data_type=u32, bitcast=true] %93
                            %95:u32[1, 4, 8] = and %91 %94
                            %96:f32[1, 4, 8] = convert_element_type [data_type=f32, bitcast=true] %95
                            %97:f32[] = constant [value=1.0]
                            %98:f32[1, 4, 8] = broadcast [output_type=f32[1, 4, 8], output_axes=[]] %97
                            %99:f32[] = constant [value=inf]
                            %100:f32[1, 4, 8] = broadcast [output_type=f32[1, 4, 8], output_axes=[]] %99
                            %101:f32[1, 4, 8] = clamp %98 %96 %100
                            %102:f32[] = constant [value=0.5]
                            %103:f32[1, 4, 8] = broadcast [output_type=f32[1, 4, 8], output_axes=[]] %102
                            %104:f32[1, 4, 8] = mul %101 %103
                            %105:f32[1, 4, 8] = div %90 %104
                            %106:f32[1, 4, 8] = round %105
                            %107:f32[1, 4, 8] = mul %106 %104
                            %108:f4e2m1fn[1, 4, 8] = convert_element_type [data_type=f4e2m1fn] %107
                            %109:f32[1, 2, 4] = scaled_dot [
                                dimensions=(lhs_contracting=[2], rhs_contracting=[2], lhs_batching=[0], rhs_batching=[0]),
                                accumulation_type=f32,
                                lhs_scale=true,
                                rhs_scale=true,
                            ] %57 %108 %33 %84
                            %110:f32[2, 4] = reshape [shape=[2, 4]] %109
                        in (%110)
                    },
                ]
                in (%2)
            "#}
            .trim_end(),
        );
    }

    #[test]
    fn test_block_scaled_dot_full_precision_differentiation() {
        // NVFP4 gradients default to unquantized backward contractions, which equal the gradients of the plain dot
        // product, except that values that the forward quantization clipped receive zero gradients.
        let nvfp4 = BlockScaleFormat::new(4, DataType::F4E2M1FN, DataType::F8E4M3FN);
        let configuration = BlockScaledDotConfiguration::new(DotDimensionNumbers::matmul(), nvfp4);
        assert!(!configuration.quantized_backward());
        let function = block_scaled_dot::<EagerArrayContext>(configuration).unwrap();
        let lhs = cyclic_matrix(2, 8, &[6.0, 1.0, 0.5, 2.0, 3.0, 4.0, 1.5]);
        let rhs = cyclic_matrix(8, 3, &[1.0, 6.0, 2.0, 0.5, 3.0]);
        let seed = cyclic_matrix(2, 3, &[1.0, -2.0, 0.5]);
        let (_, pullback) = differentiate_at((lhs.clone(), rhs.clone()))
            .vjp(|(lhs, rhs)| function.call(BlockScaledDotInputs::new(lhs, rhs)))
            .unwrap();
        let (_, expected_pullback) = differentiate_at((lhs.clone(), rhs.clone()))
            .vjp(|(lhs, rhs)| lhs.dot(&rhs, &DotDimensionNumbers::matmul()))
            .unwrap();
        let expected = expected_pullback.apply_materialized(seed.clone()).unwrap();
        assert_eq!(pullback.apply_materialized(seed.clone()), Ok(expected.clone()));

        // Global scales of `1 / 896` clip values above `6 · 448 / 896 = 3` on both sides.
        let global_scale = Array::scalar(1.0f32 / 896.0).unwrap();
        let (_, pullback) = differentiate_at((lhs.clone(), rhs.clone()))
            .with_captures(global_scale)
            .vjp(|(lhs, rhs), global_scale| {
                function.call(
                    BlockScaledDotInputs::new(lhs, rhs)
                        .with_lhs_global_scale(global_scale.clone())
                        .with_rhs_global_scale(global_scale),
                )
            })
            .unwrap();
        let (lhs_cotangent, rhs_cotangent) = pullback.apply_materialized(seed).unwrap();
        let masked = |value: &Array, cotangent: &Array, rows: usize, columns: usize| {
            let values = value
                .to_f64s()
                .into_iter()
                .zip(cotangent.to_f64s())
                .map(|(value, cotangent)| if f64::abs(value) <= 3.0 { cotangent as f32 } else { 0.0 })
                .collect::<Vec<_>>();
            Array::matrix(rows, columns, values).unwrap()
        };
        assert_eq!(lhs_cotangent, masked(&lhs, &expected.0, 2, 8));
        assert_eq!(rhs_cotangent, masked(&rhs, &expected.1, 8, 3));
    }

    #[test]
    fn test_block_scaled_dot_dimension_numbers() {
        // Several contracting dimensions, listed in different orders on both sides, merge into one blocked dimension,
        // and batching dimensions need not lead. Every value is `±6`, so every block scale is `1` and quantization is
        // exact, and the gradients restore the input layouts and data types.
        let dimensions = DotDimensionNumbers::new(vec![3, 1], vec![0, 2], vec![0], vec![1]);
        let nvfp4 = BlockScaleFormat::new(4, DataType::F4E2M1FN, DataType::F8E4M3FN);
        let function = block_scaled_dot::<EagerArrayContext>(
            BlockScaledDotConfiguration::new(dimensions.clone(), nvfp4).with_accumulation_type(DataType::BF16),
        )
        .unwrap();
        let signs = |shape: &[usize], phase: f64| {
            let count = shape.iter().product::<usize>();
            Array::from_elements::<f64>(
                ArrayType::new_static(DataType::F64, shape.to_vec()),
                &(0..count)
                    .map(|index| if (index as f64 * 1.7 + phase).sin() < 0.0 { -6.0 } else { 6.0 })
                    .collect::<Vec<_>>(),
            )
            .unwrap()
            .convert_element_type(DataType::BF16)
            .unwrap()
        };
        let lhs = signs(&[3, 4, 2, 2], 0.3);
        let rhs = signs(&[2, 3, 4, 5], 0.9);
        let seed = signs(&[3, 2, 5], 1.4);
        let (output, pullback) = differentiate_at((lhs.clone(), rhs.clone()))
            .vjp(|(lhs, rhs)| function.call(BlockScaledDotInputs::new(lhs, rhs)))
            .unwrap();
        let (expected_output, expected_pullback) =
            differentiate_at((lhs, rhs)).vjp(|(lhs, rhs)| lhs.dot(&rhs, &dimensions)).unwrap();
        assert_eq!(output, expected_output);
        let (lhs_cotangent, rhs_cotangent) = pullback.apply_materialized(seed.clone()).unwrap();
        assert_eq!(lhs_cotangent.r#type().data_type(), DataType::BF16);
        assert_eq!((lhs_cotangent, rhs_cotangent), expected_pullback.apply_materialized(seed).unwrap());
    }

    #[test]
    fn test_block_scaled_dot_quantized_differentiation() {
        // MX gradients default to quantized backward contractions. Every value below is exactly representable after
        // power of two scaling, so the quantized gradients equal the gradients of the plain dot product.
        let mxfp8 = BlockScaleFormat::new(4, DataType::F8E4M3FN, DataType::F8E8M0FNU);
        let configuration = BlockScaledDotConfiguration::new(DotDimensionNumbers::matmul(), mxfp8);
        assert!(configuration.quantized_backward());
        let function = block_scaled_dot::<EagerArrayContext>(configuration).unwrap();
        let lhs = cyclic_matrix(4, 8, &[6.0, 1.0, 0.5, 2.0, 3.0, 4.0, 1.5]);
        let rhs = cyclic_matrix(8, 4, &[1.0, 6.0, 2.0, 0.5, 3.0]);
        let seed = cyclic_matrix(4, 4, &[1.0, -2.0, 0.5, 3.0, -6.0]);
        let (output, pullback) = differentiate_at((lhs.clone(), rhs.clone()))
            .vjp(|(lhs, rhs)| function.call(BlockScaledDotInputs::new(lhs, rhs)))
            .unwrap();
        let (expected_output, expected_pullback) = differentiate_at((lhs, rhs))
            .vjp(|(lhs, rhs)| lhs.dot(&rhs, &DotDimensionNumbers::matmul()))
            .unwrap();
        assert_eq!(output, expected_output);
        assert_eq!(pullback.apply_materialized(seed.clone()), expected_pullback.apply_materialized(seed));

        // Forward mode differentiation is not supported, as with the `custom_vjp` of JAX.
        let lhs = cyclic_matrix(4, 8, &[1.0]);
        let rhs = cyclic_matrix(8, 4, &[1.0]);
        assert!(matches!(
            differentiate_at((lhs.clone(), rhs.clone()))
                .jvp((lhs, rhs), |(lhs, rhs)| function.call(BlockScaledDotInputs::new(lhs, rhs))),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "cannot apply forward mode differentiation to a `composite` call of `block_scaled_dot` \
                               that has only reverse mode rules; it supports only reverse mode differentiation (e.g., \
                               `vjp`, `value_and_gradient`, or `jacobian_reverse`)",
        ));
    }

    #[test]
    fn test_block_scaled_dot_manual_variation() {
        // The NVIDIA recipe combines values that its rules create (e.g., the Hadamard signs and butterflies, the bounds
        // of current global scales, and the stochastic rounding noise drawn from an invariant random state) with the
        // varying left input and output cotangent. Inside a manual region whose rows are sharded over `x`, every device
        // therefore computes the forward and backward rules of its own shard, so the outputs and the left gradient
        // concatenate the results of the two shards, and the gradient of the invariant right input sums them.
        type ArrayIrContext = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let configuration = BlockScaledDotConfiguration::nvfp4_training(DotDimensionNumbers::new(
            vec![1],
            vec![1],
            Vec::new(),
            Vec::new(),
        ));
        let function = block_scaled_dot::<ProjectedContext<ArrayIrContext, ArrayType>>(configuration.clone()).unwrap();
        let reference = block_scaled_dot::<EagerArrayContext>(configuration).unwrap();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharded_rows =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap();
        let values = |count: usize, phase: f64| {
            (0..count).map(|index| (index as f64 * 0.37 + phase).sin() as f32).collect::<Vec<_>>()
        };
        let (lhs, rhs, seed) = (values(512, 0.1), values(256, 0.7), values(512, 1.3));
        let state = Array::from_elements(RandomAlgorithm::ThreeFry.state_type(), &[1u64, 0]).unwrap();
        let (output, pullback) = differentiate_at((
            ArrayIrValue::Array(Array::matrix(32, 16, lhs.clone()).unwrap()),
            ArrayIrValue::Array(Array::matrix(16, 16, rhs.clone()).unwrap()),
        ))
        .with_captures(ArrayIrValue::Array(state.clone()))
        .vjp(|(lhs, rhs), state| {
            let output = shard_map(
                |(lhs, rhs, state): (
                    ShardMapTracer<ArrayIrContext>,
                    ShardMapTracer<ArrayIrContext>,
                    ShardMapTracer<ArrayIrContext>,
                )| {
                    function.call(BlockScaledDotInputs::new(lhs, rhs).with_random_state(state)).unwrap()
                },
                (
                    ValueProjection::<ArrayType>::into_projected(lhs)?,
                    ValueProjection::<ArrayType>::into_projected(rhs)?,
                    ValueProjection::<ArrayType>::into_projected(state)?,
                ),
                mesh.clone(),
                (sharded_rows.clone(), Sharding::replicated(mesh.clone(), 2), Sharding::replicated(mesh.clone(), 1)),
                sharded_rows.clone(),
            )?;
            Ok::<_, ProgramError>(output.into_value())
        })
        .unwrap();
        let output_type = ArrayType::new_static(DataType::F32, [32, 16]).with_sharding(sharded_rows).unwrap();
        let (lhs_cotangent, rhs_cotangent) = pullback
            .apply_materialized(ArrayIrValue::Array(Array::from_elements(output_type.clone(), &seed).unwrap()))
            .unwrap();

        // Each device runs the eager rules on its own shard with the same random state.
        let shard = |lhs: &[f32], seed: &[f32]| {
            let (output, pullback) = differentiate_at((
                Array::matrix(16, 16, lhs.to_vec()).unwrap(),
                Array::matrix(16, 16, rhs.clone()).unwrap(),
            ))
            .with_captures(state.clone())
            .vjp(|(lhs, rhs), state| reference.call(BlockScaledDotInputs::new(lhs, rhs).with_random_state(state)))
            .unwrap();
            let (lhs_cotangent, rhs_cotangent) =
                pullback.apply_materialized(Array::matrix(16, 16, seed.to_vec()).unwrap()).unwrap();
            (output, lhs_cotangent, rhs_cotangent)
        };
        let (first_output, first_lhs_cotangent, first_rhs_cotangent) = shard(&lhs[..256], &seed[..256]);
        let (second_output, second_lhs_cotangent, second_rhs_cotangent) = shard(&lhs[256..], &seed[256..]);
        assert_eq!(
            output,
            ArrayIrValue::Array(
                Array::from_elements(
                    output_type,
                    &[first_output.elements::<f32>().unwrap(), second_output.elements::<f32>().unwrap()].concat(),
                )
                .unwrap(),
            ),
        );
        assert_eq!(
            lhs_cotangent,
            ArrayIrValue::Array(
                Array::matrix(
                    32,
                    16,
                    [first_lhs_cotangent.elements::<f32>().unwrap(), second_lhs_cotangent.elements::<f32>().unwrap()]
                        .concat(),
                )
                .unwrap(),
            ),
        );
        assert_eq!(rhs_cotangent, ArrayIrValue::Array(first_rhs_cotangent.add(&second_rhs_cotangent).unwrap()));
    }

    #[test]
    fn test_block_scaled_dot_batching() {
        let nvfp4 = BlockScaleFormat::new(4, DataType::F4E2M1FN, DataType::F8E4M3FN);
        let function = block_scaled_dot::<EagerArrayContext>(BlockScaledDotConfiguration::new(
            DotDimensionNumbers::matmul(),
            nvfp4,
        ))
        .unwrap();
        let lhs = cyclic_matrix(2, 8, &[6.0, 1.0, 0.5, 2.0, 3.0, 4.0, 1.5]);
        let rhs = cyclic_matrix(8, 3, &[1.0, 6.0, 2.0, 0.5, 3.0]);
        let expected = function.call(BlockScaledDotInputs::new(lhs.clone(), rhs.clone())).unwrap();
        let stacked = |value: &Array| {
            let mut r#type = value.r#type().into_owned();
            r#type = r#type.with_inserted_dimension(0, Dimension::Static(2)).unwrap();
            value.broadcast(r#type, &[1, 2]).unwrap()
        };
        assert_eq!(
            batch(
                |(lhs, rhs)| function.call(BlockScaledDotInputs::new(lhs, rhs)),
                (stacked(&lhs), rhs),
                (BatchAxis::new(0), BatchAxis::replicated()),
                BatchAxis::new(0),
                2,
            ),
            Ok(stacked(&expected)),
        );
    }

    #[test]
    fn test_block_scaled_dot_validation() {
        let nvfp4 = BlockScaleFormat::new(4, DataType::F4E2M1FN, DataType::F8E4M3FN);
        let configuration = BlockScaledDotConfiguration::new(DotDimensionNumbers::matmul(), nvfp4);
        assert!(matches!(
            block_scaled_dot::<EagerArrayContext>(configuration.clone().with_lhs_format(BlockScaleFormat::new(
                4,
                DataType::F16,
                DataType::F8E4M3FN,
            ))),
            Err(error) if error == ProgramError::from(TypeError::invalid(
                "`block_scaled_dot` left format does not support element data type `f16`",
            )),
        ));
        assert!(matches!(
            block_scaled_dot::<EagerArrayContext>(configuration.clone().with_gradient_format(BlockScaleFormat::new(
                1,
                DataType::F4E2M1FN,
                DataType::F8E4M3FN,
            ))),
            Err(error) if error == ProgramError::from(TypeError::invalid(
                "`block_scaled_dot` gradient format block size must be at least 2 but got 1",
            )),
        ));
        assert!(matches!(
            block_scaled_dot::<EagerArrayContext>(configuration.clone().with_accumulation_type(DataType::I32)),
            Err(error) if error == ProgramError::from(TypeError::invalid(
                "`block_scaled_dot` accumulation type must be one of `bf16`, `f16`, `f32`, and `f64` but got `i32`",
            )),
        ));
        assert!(matches!(
            block_scaled_dot::<EagerArrayContext>(configuration.clone().with_tiled_rhs_scales(true)),
            Err(error) if error == ProgramError::from(TypeError::invalid(
                "`block_scaled_dot` stochastic gradient rounding, Hadamard transforms, and tiled right scales require \
                 quantized backward contractions",
            )),
        ));
        assert!(matches!(
            block_scaled_dot::<EagerArrayContext>(
                configuration.clone().with_quantized_backward(true).with_hadamard_transform(true),
            ),
            Err(error) if error == ProgramError::from(TypeError::invalid(
                "`block_scaled_dot` Hadamard transforms of formats with global scales require current global scaling",
            )),
        ));

        let function = block_scaled_dot::<EagerArrayContext>(configuration).unwrap();
        let lhs = cyclic_matrix(2, 8, &[1.0]);
        let rhs = cyclic_matrix(8, 3, &[1.0]);
        assert_eq!(
            function.call(BlockScaledDotInputs::new(cyclic_matrix(2, 6, &[1.0]), cyclic_matrix(6, 3, &[1.0]))),
            Err(ProgramError::from(TypeError::invalid(
                "`block_scaled_dot` contracting extent 6 is not a positive multiple of block size 4",
            ))),
        );
        assert_eq!(
            function.call(
                BlockScaledDotInputs::new(lhs.clone(), rhs.clone())
                    .with_lhs_global_scale(Array::vector(vec![1.0f32]).unwrap()),
            ),
            Err(ProgramError::from(TypeError::invalid(
                "`block_scaled_dot` left global scale must be a rank-0 real floating-point array but got `f32[1]`",
            ))),
        );
        assert_eq!(
            function.call(
                BlockScaledDotInputs::new(lhs.clone(), rhs.clone()).with_random_state(
                    Array::from_elements(RandomAlgorithm::ThreeFry.state_type(), &[1u64, 0]).unwrap(),
                ),
            ),
            Err(ProgramError::from(TypeError::invalid(
                "`block_scaled_dot` random generator states require stochastic gradient rounding",
            ))),
        );
        assert_eq!(
            function.call(BlockScaledDotInputs::new(Array::matrix(2, 8, vec![1i32; 16]).unwrap(), rhs)),
            Err(ProgramError::from(TypeError::invalid(
                "`block_scaled_dot` left input must have a real floating-point element type but got `i32`",
            ))),
        );
        let current = block_scaled_dot::<EagerArrayContext>(
            BlockScaledDotConfiguration::new(DotDimensionNumbers::matmul(), nvfp4)
                .with_global_scaling(BlockScaledGlobalScaling::Current),
        )
        .unwrap();
        assert_eq!(
            current.call(
                BlockScaledDotInputs::new(lhs.clone(), cyclic_matrix(8, 3, &[1.0]))
                    .with_rhs_global_scale(Array::scalar(1.0f32).unwrap()),
            ),
            Err(ProgramError::from(TypeError::invalid(
                "`block_scaled_dot` right global scale requires provided global scaling and a format with `f8e4m3fn` \
                 scales",
            ))),
        );

        // Gradient global scales require quantized backward contractions, and random states must have a generator
        // state type.
        assert_eq!(
            function.call(
                BlockScaledDotInputs::new(lhs.clone(), cyclic_matrix(8, 3, &[1.0]))
                    .with_gradient_global_scale(Array::scalar(1.0f32).unwrap()),
            ),
            Err(ProgramError::from(TypeError::invalid(
                "`block_scaled_dot` gradient global scales require quantized backward contractions",
            ))),
        );
        let rounding = block_scaled_dot::<EagerArrayContext>(
            BlockScaledDotConfiguration::new(DotDimensionNumbers::matmul(), nvfp4)
                .with_quantized_backward(true)
                .with_stochastic_gradient_rounding(true),
        )
        .unwrap();
        assert_eq!(
            rounding.call(
                BlockScaledDotInputs::new(lhs.clone(), cyclic_matrix(8, 3, &[1.0]))
                    .with_random_state(Array::vector(vec![1u32, 0]).unwrap()),
            ),
            Err(ProgramError::from(TypeError::invalid(
                "random generator states must have type `u64[2]` (i.e., for `three_fry`) or `u64[3]` (i.e., for \
                 `philox`) but got `u32[2]`",
            ))),
        );

        // Extents that only backward contractions block are validated when differentiating.
        let mxfp8 = BlockScaleFormat::new(4, DataType::F8E4M3FN, DataType::F8E8M0FNU);
        let function = block_scaled_dot::<EagerArrayContext>(BlockScaledDotConfiguration::new(
            DotDimensionNumbers::matmul(),
            mxfp8,
        ))
        .unwrap();
        assert_eq!(
            function.call(BlockScaledDotInputs::new(lhs.clone(), cyclic_matrix(8, 3, &[1.0]))),
            Ok(Array::matrix(2, 3, vec![8.0f32; 6]).unwrap()),
        );
        let (_, pullback) = differentiate_at((lhs, cyclic_matrix(8, 3, &[1.0])))
            .vjp(|(lhs, rhs)| function.call(BlockScaledDotInputs::new(lhs, rhs)))
            .unwrap();
        assert_eq!(
            pullback.apply_materialized(cyclic_matrix(2, 3, &[1.0])),
            Err(ProgramError::from(TypeError::invalid(
                "`block_scaled_dot` right free extent 3 is not a positive multiple of block size 4",
            ))),
        );
    }

    #[test]
    fn test_block_scaled_dot_tiled_rhs_scales() {
        // Tiled scales quantize the right input identically along either of its tile dimensions, so the forward and
        // input gradient contractions see the same quantized weight.
        let nvfp4 = BlockScaleFormat::new(4, DataType::F4E2M1FN, DataType::F8E4M3FN);
        let configuration = BlockScaledDotConfiguration::new(DotDimensionNumbers::matmul(), nvfp4)
            .with_quantized_backward(true)
            .with_tiled_rhs_scales(true);
        let inputs = BlockScaledDotInputs::new(cyclic_matrix(4, 8, &[1.0]), cyclic_matrix(8, 4, &[1.0]));
        let rules = BlockScaledDotRules::new(&configuration, &inputs).unwrap();
        let weight = Array::from_elements::<f64>(
            ArrayType::new_static(DataType::F64, [1, 8, 8]),
            &(0..64).map(|index| (index as f64 * 0.91).cos() * (1.0 + (index / 16) as f64)).collect::<Vec<_>>(),
        )
        .unwrap()
        .convert_element_type(DataType::F32)
        .unwrap();
        let dequantize = |axis: usize| {
            let quantized = rules.quantize(&weight, nvfp4, axis, true, GlobalScale::None, None).unwrap();
            quantized.elements.block_dequantize(&quantized.scales, [axis as isize], DataType::F32).unwrap()
        };
        assert_eq!(dequantize(1), dequantize(2));
        let quantized = rules.quantize(&weight, nvfp4, 2, false, GlobalScale::None, None).unwrap();
        assert_ne!(quantized.elements.block_dequantize(&quantized.scales, [2], DataType::F32).unwrap(), dequantize(2));
    }

    #[test]
    fn test_block_scaled_dot_hadamard_transform() {
        // Transforming the standard basis along the transformed dimension yields the transposed transform matrix, which
        // is `diag(s) · H / 4` for the Sylvester Hadamard matrix `H` and the sign vector of NVIDIA Transformer Engine.
        // The matrix is orthogonal.
        let configuration = BlockScaledDotConfiguration::nvfp4_training(DotDimensionNumbers::matmul());
        let inputs = BlockScaledDotInputs::new(cyclic_matrix(16, 16, &[1.0]), cyclic_matrix(16, 16, &[1.0]));
        let rules = BlockScaledDotRules::new(&configuration, &inputs).unwrap();
        let basis = Array::from_elements::<f32>(
            ArrayType::new_static(DataType::F32, [1, 16, 16]),
            &(0..256).map(|index| if index % 17 == 0 { 1.0 } else { 0.0 }).collect::<Vec<_>>(),
        )
        .unwrap();
        let matrix = rules
            .hadamard_transform(&basis, 16)
            .unwrap()
            .reshape(Shape::new(vec![Dimension::Static(16), Dimension::Static(16)]))
            .unwrap()
            .transpose(vec![1, 0])
            .unwrap();
        let expected = (0..16)
            .flat_map(|row: u32| {
                (0..16u32).map(move |column| {
                    HADAMARD_SIGNS[row as usize] * if (row & column).count_ones() % 2 == 0 { 0.25 } else { -0.25 }
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(matrix.to_f64s(), expected);
        let identity =
            matrix.dot(&matrix, &DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new())).unwrap();
        assert_eq!(
            identity.to_f64s(),
            (0..256).map(|index| if index % 17 == 0 { 1.0 } else { 0.0 }).collect::<Vec<_>>(),
        );

        // The transform therefore preserves products along the transformed dimension.
        let value = Array::from_elements::<f64>(
            ArrayType::new_static(DataType::F64, [1, 32, 2]),
            &(0..64).map(|index| (index as f64 * 0.53).sin()).collect::<Vec<_>>(),
        )
        .unwrap()
        .convert_element_type(DataType::F32)
        .unwrap();
        let transformed = rules.hadamard_transform(&value, 32).unwrap();
        let dimensions = DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0]);
        assert_abs_diff_eq!(
            transformed.dot(&transformed, &dimensions).unwrap(),
            value.dot(&value, &dimensions).unwrap(),
            epsilon = 1e-5,
        );
    }
}
