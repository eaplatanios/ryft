//! Reference [`Array`] kernels for the cumulative reduction operation family contracts.
//!
//! Each kernel decodes its operand's logical elements, runs the shared sequential prefix scan of
//! [`crate::operations::cumulative`] over them with the member's element-level combining operator, and re-encodes the
//! result into the operand's own type. Accumulation happens in the operand's element data type, so a low-precision
//! payload rounds every partial result exactly as a staged program does.

// TODO(eaplatanios): Review this module.

use crate::arrays::arrays::Array;
use crate::arrays::macros::dispatch_on_array_element_type;
use crate::arrays::operations::math::{ElementAdd, ElementExtremum, ElementMul, ElementRealFloatMath};
use crate::arrays::types::data::DataType;
use crate::operations::cumulative::cumulative_log_sum_exp::cumulative_log_sum_exp_abstract;
use crate::operations::cumulative::cumulative_max::cumulative_max_abstract;
use crate::operations::cumulative::cumulative_min::cumulative_min_abstract;
use crate::operations::cumulative::cumulative_product::cumulative_product_abstract;
use crate::operations::cumulative::cumulative_sum::cumulative_sum_abstract;
use crate::operations::{
    CUMULATIVE_LOG_SUM_EXP_OPERATION_NAME, CUMULATIVE_MAX_OPERATION_NAME, CUMULATIVE_MIN_OPERATION_NAME,
    CUMULATIVE_PRODUCT_OPERATION_NAME, CUMULATIVE_SUM_OPERATION_NAME, CumulativeLogSumExp, CumulativeMax,
    CumulativeMin, CumulativeProduct, CumulativeSum, cumulative_evaluate,
};
use crate::programs::{ProgramError, TypeError, Typed};

impl Array {
    /// Scans this array along `axis` with the element data type's addition, or with its multiplication when
    /// `multiply` is set, in the requested direction. Validation and the complete result metadata come from the
    /// operation's own abstract rule, so this kernel accepts exactly what a staged `cumulative_sum` or
    /// `cumulative_product` accepts.
    fn scan_arithmetic(&self, axis: usize, reverse: bool, multiply: bool) -> Result<Self, ProgramError> {
        let input_type = self.r#type().into_owned();
        let (output_type, operation_name) = match multiply {
            true => (cumulative_product_abstract(&input_type, axis)?, CUMULATIVE_PRODUCT_OPERATION_NAME),
            false => (cumulative_sum_abstract(&input_type, axis)?, CUMULATIVE_SUM_OPERATION_NAME),
        };
        // The structural-zero element type has no payload bytes, and every prefix sum or product of a zero is a zero.
        if input_type.data_type() == DataType::Zero {
            return Self::new(output_type, Vec::new());
        }
        let shape = input_type.static_shape().ok_or_else(|| {
            TypeError::invalid(format!("`{operation_name}` requires a statically shaped operand but got {input_type}"))
        })?;
        dispatch_on_array_element_type!(@numeric input_type.data_type(), |Element| {
            let elements = self.elements::<Element>()?;
            let scanned = cumulative_evaluate(elements.as_slice(), &shape, axis, reverse, |left, right| {
                match multiply {
                    true => <Element as ElementMul>::mul(left, right),
                    false => <Element as ElementAdd>::add(left, right),
                }
            })?;
            Self::from_elements(output_type, scanned.as_slice())
        })
    }

    /// Scans this array along `axis` with the element data type's extremum selection, in the requested direction and
    /// polarity. Validation and the complete result metadata come from the operation's own abstract rule, so this
    /// kernel accepts exactly what a staged `cumulative_max` or `cumulative_min` accepts.
    fn scan_extremum(&self, axis: usize, reverse: bool, maximum: bool) -> Result<Self, ProgramError> {
        let input_type = self.r#type().into_owned();
        let output_type = match maximum {
            true => cumulative_max_abstract(&input_type, axis)?,
            false => cumulative_min_abstract(&input_type, axis)?,
        };
        let operation_name = match maximum {
            true => CUMULATIVE_MAX_OPERATION_NAME,
            false => CUMULATIVE_MIN_OPERATION_NAME,
        };
        let shape = input_type.static_shape().ok_or_else(|| {
            TypeError::invalid(format!("`{operation_name}` requires a statically shaped operand but got {input_type}"))
        })?;
        dispatch_on_array_element_type!(@real input_type.data_type(), |Element| {
            let elements = self.elements::<Element>()?;
            let scanned = cumulative_evaluate(elements.as_slice(), &shape, axis, reverse, |left, right| {
                Ok(match maximum {
                    true => <Element as ElementExtremum>::maximum(left, right),
                    false => <Element as ElementExtremum>::minimum(left, right),
                })
            })?;
            Self::from_elements(output_type, scanned.as_slice())
        })
    }

    /// Scans this array along `axis` with the element data type's stable `log_add_exp`, in the requested direction.
    /// Validation and the complete result metadata come from the operation's own abstract rule, so this kernel
    /// accepts exactly what a staged `cumulative_log_sum_exp` accepts.
    fn scan_log_sum_exp(&self, axis: usize, reverse: bool) -> Result<Self, ProgramError> {
        let input_type = self.r#type().into_owned();
        let output_type = cumulative_log_sum_exp_abstract(&input_type, axis)?;
        let shape = input_type.static_shape().ok_or_else(|| {
            TypeError::invalid(format!(
                "`{CUMULATIVE_LOG_SUM_EXP_OPERATION_NAME}` requires a statically shaped operand but got {input_type}"
            ))
        })?;
        dispatch_on_array_element_type!(@float input_type.data_type(), |Element| {
            let elements = self.elements::<Element>()?;
            let scanned = cumulative_evaluate(
                elements.as_slice(),
                &shape,
                axis,
                reverse,
                <Element as ElementRealFloatMath>::log_add_exp,
            )?;
            Self::from_elements(output_type, scanned.as_slice())
        })
    }
}

impl CumulativeProduct for Array {
    #[inline]
    fn cumulative_product(&self, axis: usize) -> Result<Self, ProgramError> {
        self.scan_arithmetic(axis, false, true)
    }

    #[inline]
    fn reverse_cumulative_product(&self, axis: usize) -> Result<Self, ProgramError> {
        self.scan_arithmetic(axis, true, true)
    }
}

impl CumulativeMax for Array {
    #[inline]
    fn cumulative_max(&self, axis: usize) -> Result<Self, ProgramError> {
        self.scan_extremum(axis, false, true)
    }

    #[inline]
    fn reverse_cumulative_max(&self, axis: usize) -> Result<Self, ProgramError> {
        self.scan_extremum(axis, true, true)
    }
}

impl CumulativeMin for Array {
    #[inline]
    fn cumulative_min(&self, axis: usize) -> Result<Self, ProgramError> {
        self.scan_extremum(axis, false, false)
    }

    #[inline]
    fn reverse_cumulative_min(&self, axis: usize) -> Result<Self, ProgramError> {
        self.scan_extremum(axis, true, false)
    }
}

impl CumulativeLogSumExp for Array {
    #[inline]
    fn cumulative_log_sum_exp(&self, axis: usize) -> Result<Self, ProgramError> {
        self.scan_log_sum_exp(axis, false)
    }

    #[inline]
    fn reverse_cumulative_log_sum_exp(&self, axis: usize) -> Result<Self, ProgramError> {
        self.scan_log_sum_exp(axis, true)
    }
}

impl CumulativeSum for Array {
    #[inline]
    fn cumulative_sum(&self, axis: usize) -> Result<Self, ProgramError> {
        self.scan_arithmetic(axis, false, false)
    }

    #[inline]
    fn reverse_cumulative_sum(&self, axis: usize) -> Result<Self, ProgramError> {
        self.scan_arithmetic(axis, true, false)
    }
}

#[cfg(test)]
mod tests {
    use half::bf16;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::types::arrays::ArrayType;
    use crate::arrays::types::layouts::{Layout, StridedLayout};

    use super::*;

    #[test]
    fn test_array_cumulative_sum() {
        // Forward scans accumulate prefixes along the selected axis only, and reverse scans accumulate suffixes.
        let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(input.cumulative_sum(1), Ok(Array::matrix(2, 3, vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0])));
        assert_eq!(input.reverse_cumulative_sum(1), Ok(Array::matrix(2, 3, vec![6.0, 5.0, 3.0, 15.0, 11.0, 6.0])));
        assert_eq!(input.cumulative_sum(0), Ok(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0])));

        // Integer payloads accumulate with the element type's own wrapping arithmetic.
        assert_eq!(Array::vector(vec![1_i32, 2, 3, 4]).cumulative_sum(0), Ok(Array::vector(vec![1_i32, 3, 6, 10])));

        // A non-numeric payload has no summation, and the scan geometry is validated against the operand type.
        assert_eq!(
            Array::vector(vec![true, false]).cumulative_sum(0),
            Err(ProgramError::Type(TypeError::invalid(
                "`cumulative_sum` requires numeric inputs but got bool".to_string(),
            ))),
        );
        assert_eq!(
            input.cumulative_sum(2),
            Err(ProgramError::Type(TypeError::invalid(
                "`cumulative_sum` axis 2 is out of bounds for rank 2".to_string(),
            ))),
        );
    }

    #[test]
    fn test_array_cumulative_sum_element_encodings() {
        // Accumulation happens in the operand's own encoding, so every partial sum is re-encoded rather than only
        // the final one. Each increment below is smaller than half a `bf16` step next to one, yet the running sum
        // still climbs by a full step each time, because each partial sum rounds up on its own. Summing the four
        // increments exactly and rounding once would stop one step short, which is what pins per-step re-encoding.
        let low_precision_type = ArrayType::new_static(DataType::BF16, [5]);
        let increment = f64::from(bf16::from_f64(0.005));
        assert_eq!(
            Array::from_f64s(low_precision_type.clone(), vec![1.0, 0.005, 0.005, 0.005, 0.005]).cumulative_sum(0),
            Ok(Array::from_f64s(low_precision_type, vec![1.0, 1.0078125, 1.015625, 1.0234375, 1.03125])),
        );
        // Each of those partial sums is the re-encoding of the previous one plus the increment, and the exact sum of
        // all four increments rounds one step below the scan's last element.
        assert_eq!(f64::from(bf16::from_f64(1.0 + increment)), 1.0078125);
        assert_eq!(f64::from(bf16::from_f64(1.0078125 + increment)), 1.015625);
        assert_eq!(f64::from(bf16::from_f64(1.015625 + increment)), 1.0234375);
        assert_eq!(f64::from(bf16::from_f64(1.0234375 + increment)), 1.03125);
        assert_eq!(f64::from(bf16::from_f64(1.0 + 4.0 * increment)), 1.0234375);

        // The payload-free structural zero has no bytes to scan, and every prefix sum of a zero is a zero.
        let structural_zero = Array::new(ArrayType::new_static(DataType::Zero, [3]), Vec::new()).unwrap();
        assert_eq!(structural_zero.cumulative_sum(0), Ok(structural_zero));

        // Complex payloads accumulate both components.
        assert_eq!(
            Array::vector(vec![
                ComplexNumber::new(1.0_f64, 1.0),
                ComplexNumber::new(2.0, -1.0),
                ComplexNumber::new(3.0, 5.0),
            ])
            .cumulative_sum(0),
            Ok(Array::vector(vec![
                ComplexNumber::new(1.0_f64, 1.0),
                ComplexNumber::new(3.0, 0.0),
                ComplexNumber::new(6.0, 5.0),
            ])),
        );

        // The result carries the operand's complete type, including a non-default physical layout.
        let laid_out =
            ArrayType::new_static(DataType::F32, [2, 2]).with_layout(Layout::Strided(StridedLayout::new(vec![4, 8])));
        assert_eq!(
            Array::from_f64s(laid_out.clone(), vec![1.0, 2.0, 3.0, 4.0]).cumulative_sum(1),
            Ok(Array::from_f64s(laid_out, vec![1.0, 3.0, 3.0, 7.0])),
        );

        // A zero-length scanned axis has nothing to accumulate.
        let empty = Array::new(ArrayType::new_static(DataType::F32, [0, 2]), Vec::new()).unwrap();
        assert_eq!(empty.cumulative_sum(0), Ok(empty));
    }

    #[test]
    fn test_array_cumulative_product() {
        // Forward scans accumulate prefixes along the selected axis only, and reverse scans accumulate suffixes.
        let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(input.cumulative_product(1), Ok(Array::matrix(2, 3, vec![1.0, 2.0, 6.0, 4.0, 20.0, 120.0])));
        let reversed = Array::matrix(2, 3, vec![6.0, 6.0, 3.0, 120.0, 30.0, 6.0]);
        assert_eq!(input.reverse_cumulative_product(1), Ok(reversed));

        // Integer payloads accumulate with the element type's own arithmetic, and complex payloads multiply as
        // complex numbers.
        assert_eq!(Array::vector(vec![1_i32, 2, 3, 4]).cumulative_product(0), Ok(Array::vector(vec![1_i32, 2, 6, 24])));
        assert_eq!(
            Array::vector(vec![ComplexNumber::new(0.0_f64, 1.0); 3]).cumulative_product(0),
            Ok(Array::vector(vec![
                ComplexNumber::new(0.0_f64, 1.0),
                ComplexNumber::new(-1.0, 0.0),
                ComplexNumber::new(0.0, -1.0),
            ])),
        );

        // Accumulation happens in the operand's own encoding, so every partial product is re-encoded rather than only
        // the final one. The third prefix below is exactly halfway between two `f8e4m3fn` values and rounds down to
        // an even mantissa, which drags the fourth prefix one step below the exactly accumulated product.
        let low_precision_type = ArrayType::new_static(DataType::F8E4M3FN, [4]);
        assert_eq!(
            Array::from_f64s(low_precision_type.clone(), vec![2.0, 1.25, 1.25, 1.25]).cumulative_product(0),
            Ok(Array::from_f64s(low_precision_type, vec![2.0, 2.5, 3.0, 3.75])),
        );
        assert_eq!(2.0 * 1.25 * 1.25 * 1.25, 3.90625);
        assert_eq!(
            Array::from_f64s(ArrayType::new_static(DataType::F8E4M3FN, [1]), vec![3.90625]).to_f64s(),
            vec![4.0],
        );

        // The payload-free structural zero has no bytes to scan, and every prefix product of a zero is a zero.
        let structural_zero = Array::new(ArrayType::new_static(DataType::Zero, [3]), Vec::new()).unwrap();
        assert_eq!(structural_zero.cumulative_product(0), Ok(structural_zero));

        // A non-numeric payload has no multiplication, and the scan geometry is validated against the operand type.
        assert_eq!(
            Array::vector(vec![true, false]).cumulative_product(0),
            Err(ProgramError::Type(TypeError::invalid(
                "`cumulative_product` requires numeric inputs but got bool".to_string(),
            ))),
        );
        assert_eq!(
            input.cumulative_product(2),
            Err(ProgramError::Type(TypeError::invalid(
                "`cumulative_product` axis 2 is out of bounds for rank 2".to_string(),
            ))),
        );
    }

    #[test]
    fn test_array_cumulative_extrema() {
        // Both extrema scan the selected axis in both directions, selecting rather than combining elements.
        let input = Array::matrix(2, 3, vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0]);
        assert_eq!(input.cumulative_max(1), Ok(Array::matrix(2, 3, vec![3.0, 3.0, 4.0, 1.0, 5.0, 9.0])));
        assert_eq!(input.reverse_cumulative_max(1), Ok(Array::matrix(2, 3, vec![4.0, 4.0, 4.0, 9.0, 9.0, 9.0])));
        assert_eq!(input.cumulative_min(1), Ok(Array::matrix(2, 3, vec![3.0, 1.0, 1.0, 1.0, 1.0, 1.0])));
        assert_eq!(input.reverse_cumulative_min(1), Ok(Array::matrix(2, 3, vec![1.0, 1.0, 4.0, 1.0, 5.0, 9.0])));

        // Selection happens in the operand's own element type, so a low-precision payload is returned bit for bit
        // rather than through a widened intermediate, and signed integers order below zero as expected.
        let low_precision_type = ArrayType::new_static(DataType::F8E5M2, [3]);
        assert_eq!(
            Array::from_f64s(low_precision_type.clone(), vec![0.5, 6.0, 1.5]).cumulative_max(0),
            Ok(Array::from_f64s(low_precision_type, vec![0.5, 6.0, 6.0])),
        );
        assert_eq!(Array::vector(vec![7_i32, -2, 3]).cumulative_min(0), Ok(Array::vector(vec![7_i32, -2, -2])));

        // Complex numbers are unordered and Booleans are not numeric, so neither has a running extremum.
        assert_eq!(
            Array::vector(vec![ComplexNumber::new(1.0_f64, 0.0); 2]).cumulative_max(0),
            Err(ProgramError::Type(TypeError::invalid(
                "`cumulative_max` requires real numeric inputs but got c128".to_string(),
            ))),
        );
        assert_eq!(
            Array::vector(vec![true, false]).cumulative_min(0),
            Err(ProgramError::Type(TypeError::invalid(
                "`cumulative_min` requires real numeric inputs but got bool".to_string(),
            ))),
        );
    }

    #[test]
    fn test_array_cumulative_log_sum_exp() {
        // Folding the guarded pairwise primitive keeps the scan exact where exponentiating directly would overflow:
        // two equal operands add exactly `log(2)` at any magnitude, in both directions.
        assert_eq!(
            Array::vector(vec![1000.0, 1000.0]).cumulative_log_sum_exp(0),
            Ok(Array::vector(vec![1000.0, 1000.0 + std::f64::consts::LN_2])),
        );
        assert_eq!(
            Array::vector(vec![1000.0, 1000.0]).reverse_cumulative_log_sum_exp(0),
            Ok(Array::vector(vec![1000.0 + std::f64::consts::LN_2, 1000.0])),
        );

        // Negative infinity is the combining operator's identity, so it neither contributes to nor poisons a later
        // prefix, while a NaN operand propagates.
        assert_eq!(
            Array::vector(vec![f64::NEG_INFINITY, f64::NEG_INFINITY, 2.0]).cumulative_log_sum_exp(0),
            Ok(Array::vector(vec![f64::NEG_INFINITY, f64::NEG_INFINITY, 2.0])),
        );
        let with_nan = Array::vector(vec![1.0, f64::NAN, 2.0]).cumulative_log_sum_exp(0).unwrap().to_f64s();
        assert_eq!(with_nan[0], 1.0);
        assert!(with_nan[1].is_nan() && with_nan[2].is_nan());

        // Accumulation happens in the operand's own encoding, so each partial result is rounded to it.
        let single_precision = ArrayType::new_static(DataType::F32, [2]);
        assert_eq!(
            Array::from_f64s(single_precision.clone(), vec![0.0, 0.0]).cumulative_log_sum_exp(0),
            Ok(Array::from_f64s(single_precision, vec![0.0, f64::from(std::f32::consts::LN_2)])),
        );

        // The exponential and the logarithm have no meaning for the integer element types.
        assert_eq!(
            Array::vector(vec![1_i32, 2]).cumulative_log_sum_exp(0),
            Err(ProgramError::Type(TypeError::invalid(
                "`cumulative_log_sum_exp` requires real floating-point inputs but got i32".to_string(),
            ))),
        );
    }
}
