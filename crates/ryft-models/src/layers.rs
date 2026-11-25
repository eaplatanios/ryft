//! Numerical building blocks that model definitions share.
//!
//! The functions of this module are written once over the [`ArrayOperations`] capability bundle of `ryft-core`, so
//! they run eagerly on host arrays, under the program transformations of `ryft-core` (e.g., reverse mode
//! differentiation), and inside programs that a backend compiles. They hold no parameters themselves. A model passes
//! its parameter leaves in, which keeps the ownership of every leaf with the parameter structure of the model.
//!
//! Precision is explicit throughout. Every function states the data type that it computes in and the data type of its
//! result, because the models built from these functions pin where values are rounded to a narrower type.

use ryft_core::{
    Add, ArrayMemberType, ArrayOperations, DataType, Div, DotDimensionNumbers, Max, Mul, ProgramError, ReductionKind,
};

// TODO(eaplatanios): Review this module.

/// Returns the extents of the dimensions of `value`, which must have a static shape.
///
/// # Errors
///
/// Returns a [`ProgramError`] if `value` is not an array or if one of its extents is not statically known.
pub fn static_shape<V: ArrayOperations>(value: &V) -> Result<Vec<usize>, ProgramError>
where
    V::Type: ArrayMemberType,
{
    let r#type = value.r#type();
    let array_type = r#type.as_array_type()?;
    array_type
        .static_shape()
        .map(|shape| shape.dimensions().to_vec())
        .ok_or_else(|| ProgramError::InvalidArgument {
            message: format!("expected a value with a static shape but got one of type `{array_type}`"),
        })
}

/// Multiplies every element of `value` by the constant `factor`, keeping the data type of `value`.
pub fn scale<V: ArrayOperations>(value: &V, factor: f64) -> Result<V, ProgramError> {
    Mul::mul(value, &value.fill_like(factor)?)
}

/// Applies the projection `weight` to the trailing axis of `input` (i.e., computes `input · weightᵀ`), with both
/// operands converted to `compute_type` and the result accumulated at `accumulation_type`.
///
/// # Parameters
///
///   - `input`: Array of shape `[..., input_size]`.
///   - `weight`: Array of shape `[output_size, input_size]`.
///   - `compute_type`: Data type that both operands are converted to before the contraction.
///   - `accumulation_type`: Data type of the accumulator and of the result, which has shape `[..., output_size]`.
pub fn linear<V: ArrayOperations>(
    input: &V,
    weight: &V,
    compute_type: DataType,
    accumulation_type: DataType,
) -> Result<V, ProgramError>
where
    V::Type: ArrayMemberType,
{
    let rank = static_shape(input)?.len();
    let dimensions = DotDimensionNumbers::new(vec![rank - 1], vec![1], Vec::new(), Vec::new());
    input.convert_element_type(compute_type)?.dot_with_accumulation_type(
        &weight.convert_element_type(compute_type)?,
        &dimensions,
        accumulation_type,
    )
}

/// Normalizes the trailing axis of `input` by its root mean square, computing in `f32` and returning an `f32` array.
/// For an input `x` with `n` trailing elements, the result is `x / sqrt(mean(x²) + epsilon)`, multiplied elementwise
/// by `weight` when one is provided.
///
/// # Parameters
///
///   - `input`: Array of shape `[..., n]` with any real floating-point data type.
///   - `weight`: Optional gain of shape `[n]`, which is converted to `f32`.
///   - `epsilon`: Stabilizer that is added to the mean of the squares.
pub fn rms_norm<V: ArrayOperations>(input: &V, weight: Option<&V>, epsilon: f64) -> Result<V, ProgramError>
where
    V::Type: ArrayMemberType,
{
    let input = input.convert_element_type(DataType::F32)?;
    let mut shape = static_shape(&input)?;
    let axis = shape.len() - 1;
    shape[axis] = 1;
    let mean_square = Mul::mul(&input, &input)?.reduce_mean(&[axis])?.reshape(shape)?;
    let normalized = Mul::mul(&input, &Add::add(&mean_square, &mean_square.fill_like(epsilon)?)?.rsqrt()?)?;
    match weight {
        Some(weight) => Mul::mul(&normalized, &weight.convert_element_type(DataType::F32)?),
        None => Ok(normalized),
    }
}

/// Soft caps `input` to the open interval `(-cap, cap)` by computing `cap · tanh(input / cap)` at the data type of
/// `input`. The function is close to the identity for magnitudes well below `cap`.
pub fn softcap<V: ArrayOperations>(input: &V, cap: f64) -> Result<V, ProgramError> {
    scale(&scale(input, 1.0 / cap)?.tanh()?, cap)
}

/// Shortens the vectors along the trailing axis of `input` whose L2 length exceeds `threshold` to that length,
/// computing in `f32` and returning an `f32` array. A vector `x` becomes `x / max((‖x‖ + 1e-6) / threshold, 1)`, so
/// vectors within the threshold are unchanged. When `gain` is provided, `x` is first multiplied elementwise by it.
/// A zero vector stays zero and receives a zero gradient through its length, whose derivative is otherwise
/// undefined there, so padding positions do not turn gradients into NaN values.
///
/// # Parameters
///
///   - `input`: Array of shape `[..., n]` with any real floating-point data type.
///   - `gain`: Optional array that broadcasts against `input` and is converted to `f32`.
///   - `threshold`: Largest L2 length of the resulting vectors.
pub fn clip_norm<V: ArrayOperations>(input: &V, gain: Option<&V>, threshold: f64) -> Result<V, ProgramError>
where
    V::Type: ArrayMemberType,
{
    let input = input.convert_element_type(DataType::F32)?;
    let input = match gain {
        Some(gain) => Mul::mul(&input, &gain.convert_element_type(DataType::F32)?)?,
        None => input,
    };
    let mut shape = static_shape(&input)?;
    let axis = shape.len() - 1;
    shape[axis] = 1;
    // The square root has an infinite derivative at zero, which would multiply the zero cotangent of a zero vector
    // into a NaN value. Selecting a positive placeholder before the square root keeps both the value and the
    // gradient of that case at zero.
    let squares = Mul::mul(&input, &input)?.reduce(&[axis], ReductionKind::Sum)?.reshape(shape)?;
    let positive = squares.greater_than(&squares.zero_like()?)?;
    let norm =
        positive.select(&positive.select(&squares, &squares.fill_like(1.0f32)?)?.sqrt()?, &squares.zero_like()?)?;
    let denominator = scale(&Add::add(&norm, &norm.fill_like(1e-6f64)?)?, 1.0 / threshold)?;
    Div::div(&input, &Max::max(&denominator, &denominator.fill_like(1.0f64)?)?)
}

/// Reads a learned mixture of `sources`, which is how layers of a model with attention residuals read the states of
/// earlier blocks. Every source is first normalized by its root mean square over the trailing axis (without a gain).
/// The weight of a source is the softmax over sources of `⟨normalized source, query⟩ / sqrt(n)`, computed separately
/// for every leading position, where `n` is the trailing extent. The result is the weighted sum of the normalized
/// sources, or of the sources themselves when `normalize_values` is false. The computation runs in `f32` and returns
/// an `f32` array.
///
/// # Parameters
///
///   - `sources`: Arrays of one shared shape `[..., n]` with any real floating-point data types. At least one source
///     is required.
///   - `query`: Array of shape `[n]`, which is converted to `f32`.
///   - `normalize_values`: Whether the mixed values are the normalized sources rather than the sources themselves.
///   - `epsilon`: Stabilizer of the root mean square normalization of the sources.
pub fn residual_read<V: ArrayOperations>(
    sources: &[&V],
    query: &V,
    normalize_values: bool,
    epsilon: f64,
) -> Result<V, ProgramError>
where
    V::Type: ArrayMemberType,
{
    let Some(first) = sources.first() else {
        return Err(ProgramError::InvalidArgument { message: "a residual read needs at least one source".to_string() });
    };
    let shape = static_shape(*first)?;
    let axis = shape.len() - 1;
    let size = shape[axis];
    let mut logit_shape = shape.clone();
    logit_shape[axis] = 1;

    let query = query.convert_element_type(DataType::F32)?;
    let keys = sources.iter().map(|source| rms_norm(*source, None, epsilon)).collect::<Result<Vec<_>, _>>()?;
    let logits = keys
        .iter()
        .map(|key| {
            let logits = Mul::mul(key, &query)?.reduce(&[axis], ReductionKind::Sum)?.reshape(logit_shape.clone())?;
            scale(&logits, 1.0 / (size as f64).sqrt())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let weights = V::concatenate(logits.iter(), axis)?.softmax(axis as i64)?;

    let mut output = None;
    for (index, key) in keys.iter().enumerate() {
        let value = if normalize_values { key.clone() } else { sources[index].convert_element_type(DataType::F32)? };
        let term = Mul::mul(&value, &weights.slice_axis(axis, index, index + 1, 1)?)?;
        output = Some(match output {
            Some(output) => Add::add(&output, &term)?,
            None => term,
        });
    }
    Ok(output.unwrap())
}

/// Delays `input` by `lag` positions along axis 1, which is the sequence axis of arrays of shape
/// `[batch, sequence, ...]`. Position `t` of the result holds position `t − lag` of `input`, and the first `lag`
/// positions hold zeros, so a sum of delayed copies is a causal convolution. A lag of at least the sequence length
/// gives an array of zeros.
pub fn delay<V: ArrayOperations>(input: &V, lag: usize) -> Result<V, ProgramError>
where
    V::Type: ArrayMemberType,
{
    let length = static_shape(input)?[1];
    if lag == 0 {
        return Ok(input.clone());
    }
    if lag >= length {
        return input.zero_like();
    }
    let zeros = input.slice_axis(1, 0, lag, 1)?.zero_like()?;
    zeros.concatenate_with([&input.slice_axis(1, 0, length - lag, 1)?], 1)
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::{Array, ArrayType, ConvertElementType, Reduce, Typed, bf16, differentiate_at};

    use super::*;

    /// Returns an `f32` array of the provided shape.
    fn array(shape: &[usize], elements: &[f32]) -> Array {
        Array::from_elements(ArrayType::new_static(DataType::F32, shape.to_vec()), elements).unwrap()
    }

    /// Asserts that the `f32` elements of `actual` are within `tolerance` of `expected`.
    fn assert_close(actual: &Array, expected: &[f32], tolerance: f32) {
        let actual = actual.elements::<f32>().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!((actual - expected).abs() <= tolerance, "element {index}: {actual} is not close to {expected}");
        }
    }

    #[test]
    fn test_static_shape() {
        assert_eq!(static_shape(&array(&[2, 3], &[0.0; 6])), Ok(vec![2, 3]));
        assert_eq!(static_shape(&Array::scalar(1.0f32).unwrap()), Ok(Vec::new()));
    }

    #[test]
    fn test_scale() {
        assert_eq!(scale(&array(&[2], &[1.0, -3.0]), 0.5), Ok(array(&[2], &[0.5, -1.5])));
    }

    #[test]
    fn test_linear() {
        // The weight has shape `[output_size, input_size]`, so each output is the dot product with one weight row.
        let input = array(&[1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let weight = array(&[3, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        assert_eq!(
            linear(&input, &weight, DataType::F32, DataType::F32),
            Ok(array(&[1, 2, 3], &[1.0, 2.0, 3.0, 3.0, 4.0, 7.0])),
        );

        // Operands are rounded to the compute type before the contraction, and the result has the accumulation type.
        let input = array(&[1, 1], &[1.0 + 1.0 / 1024.0]);
        let weight = array(&[1, 1], &[3.0]);
        let output = linear(&input, &weight, DataType::BF16, DataType::F32).unwrap();
        assert_eq!(output, array(&[1, 1], &[3.0]));
        let output = linear(&input, &weight, DataType::BF16, DataType::BF16).unwrap();
        assert_eq!(output.elements::<bf16>().unwrap(), vec![bf16::from_f32(3.0)]);
    }

    #[test]
    fn test_rms_norm() {
        // The root mean square of `[3, 4]` is `sqrt(12.5)`.
        let input = array(&[1, 2], &[3.0, 4.0]);
        let root_mean_square = 12.5f32.sqrt();
        assert_close(&rms_norm(&input, None, 0.0).unwrap(), &[3.0 / root_mean_square, 4.0 / root_mean_square], 1e-6);
        assert_close(
            &rms_norm(&input, Some(&array(&[2], &[2.0, -1.0])), 0.0).unwrap(),
            &[6.0 / root_mean_square, -4.0 / root_mean_square],
            1e-6,
        );

        // The stabilizer keeps an all-zero vector at zero, and narrow inputs are normalized in `f32`.
        assert_close(&rms_norm(&array(&[1, 2], &[0.0, 0.0]), None, 1e-6).unwrap(), &[0.0, 0.0], 0.0);
        let narrow = input.convert_element_type(DataType::BF16).unwrap();
        let output = rms_norm(&narrow, None, 0.0).unwrap();
        assert_eq!(output.r#type().data_type(), DataType::F32);
        assert_close(&output, &[3.0 / root_mean_square, 4.0 / root_mean_square], 1e-6);
    }

    #[test]
    fn test_softcap() {
        assert_close(
            &softcap(&array(&[3], &[0.0, 30.0, -1e6]), 30.0).unwrap(),
            &[0.0, 30.0 * 1.0f32.tanh(), -30.0],
            1e-5,
        );
    }

    #[test]
    fn test_clip_norm() {
        // The first vector has length 5 and is shortened to length 2.5, and the second one is within the threshold.
        let input = array(&[2, 2], &[3.0, 4.0, 0.3, 0.4]);
        assert_close(&clip_norm(&input, None, 2.5).unwrap(), &[1.5, 2.0, 0.3, 0.4], 1e-5);

        // A gain is applied before the length is measured.
        assert_close(&clip_norm(&input, Some(&array(&[2], &[2.0, 0.0])), 2.5).unwrap(), &[2.5, 0.0, 0.6, 0.0], 1e-5);
    }

    #[test]
    fn test_clip_norm_differentiation_at_zero() {
        // A zero vector has a zero gradient instead of an undefined one, and other vectors are unaffected.
        let input = array(&[2, 2], &[0.0, 0.0, 0.3, 0.4]);
        let gradient = differentiate_at(input)
            .gradient(|input| clip_norm(&input, None, 2.5)?.reduce(&[0, 1], ReductionKind::Sum))
            .unwrap();
        assert_close(&gradient, &[1.0, 1.0, 1.0, 1.0], 1e-6);
    }

    #[test]
    fn test_residual_read() {
        // Both sources normalize to unit vectors along the two axes, so the query `[4, 0]` gives the logits
        // `4 / sqrt(2)` and `0`.
        let first = array(&[1, 2], &[2.0, 0.0]);
        let second = array(&[1, 2], &[0.0, 3.0]);
        let query = array(&[2], &[4.0, 0.0]);
        let first_weight = 1.0 / (1.0 + (-(4.0f32 * 2.0f32.sqrt()) / 2.0f32.sqrt()).exp());
        let root_two = 2.0f32.sqrt();
        assert_close(
            &residual_read(&[&first, &second], &query, true, 0.0).unwrap(),
            &[first_weight * root_two, (1.0 - first_weight) * root_two],
            1e-5,
        );
        assert_close(
            &residual_read(&[&first, &second], &query, false, 0.0).unwrap(),
            &[first_weight * 2.0, (1.0 - first_weight) * 3.0],
            1e-5,
        );

        // A single source has weight one, and no source is an error.
        assert_close(&residual_read(&[&first], &query, false, 0.0).unwrap(), &[2.0, 0.0], 1e-6);
        assert!(matches!(
            residual_read::<Array>(&[], &query, true, 0.0),
            Err(ProgramError::InvalidArgument { message }) if message == "a residual read needs at least one source",
        ));
    }

    #[test]
    fn test_delay() {
        let input = array(&[1, 3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(delay(&input, 0), Ok(input.clone()));
        assert_eq!(delay(&input, 1), Ok(array(&[1, 3, 2], &[0.0, 0.0, 1.0, 2.0, 3.0, 4.0])));
        assert_eq!(delay(&input, 2), Ok(array(&[1, 3, 2], &[0.0, 0.0, 0.0, 0.0, 1.0, 2.0])));
        assert_eq!(delay(&input, 3), Ok(array(&[1, 3, 2], &[0.0; 6])));
        assert_eq!(delay(&input, 7), Ok(array(&[1, 3, 2], &[0.0; 6])));
    }
}
