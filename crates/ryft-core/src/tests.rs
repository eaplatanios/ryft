// TODO(eaplatanios): This module needs careful review.

//! Reference test value types for exercising `ryft` programs without a real backend.
//!
//! This module provides [`TestArray`], a deliberately simple dense array value backed by a flat row-major `Vec<f64>`
//! payload, together with [`EagerContext<TestArray, ArrayOperation<TestArray>>`], a minimal interpreting [`Domain`](crate::Domain) whose operation set is
//! [`ArrayOperation`]. They implement every value-level capability that [`ArrayOperation`] interpretation requires,
//! so unit tests, doctests, and downstream crates can stage, transform, and interpret programs end-to-end without
//! depending on an optimized backend such as `ryft-xla`.
//!
//! These types prioritize transparency over performance: payloads are plain `f64` vectors with public fields, and
//! every operation is implemented with straightforward index arithmetic. Do not use them outside of tests and
//! documentation examples.
//!
//! The module is part of `ryft-core`'s public API so downstream tests and documentation examples can use it without
//! feature configuration.

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::fmt::Display;

use crate::backends::scalars::Scalar;
use crate::broadcasting::Broadcastable;
#[cfg(test)]
use crate::contexts::Context;
use crate::contexts::EagerContext;
use crate::macros::check_count;
use crate::operations::BooleanLike;
use crate::operations::complex::{Conjugate, Imaginary, Real};
use crate::operations::constants::{Fill, One, OneLike, Zero, ZeroLike};
use crate::operations::logical::{And, Not, Or, Xor};
use crate::operations::manipulation::{
    Concatenate, ConvertElementType, DynamicSlice, DynamicUpdateSlice, Gather, GatherOperation, GatherScatterMode, Pad,
    Reshape, Scatter, ScatterOperation, ScatterReductionKind, Slice, Transpose, UpdateSlice,
};
use crate::operations::math::{Abs, Add, Atan2, Cos, Div, Exp, Log, Mul, Neg, Sin, Sqrt, Sub};
use crate::operations::tag::Tag;
use crate::parameters::Parameter;
use crate::programs::ProgramError;
use crate::programs::effects::{Effect, Effects};
use crate::programs::operations::Operation;
use crate::programs::regions::RegionInterface;
use crate::programs::types::{TypeError, Typed};
use crate::programs::values::Value;
use crate::tracing_v2::ArrayOperation;
use crate::tracing_v2::operations::TransferToMemory;
use crate::types::{ArrayType, DataType, Shape, Size, StaticShape};
use crate::{Broadcast, Compare, ComparisonDirection, Select, SelectCondition};

/// Asserts that the reverse-mode gradient of `$function` at `$input` matches a central finite-difference estimate
/// of its derivative within absolute tolerance `$tolerance`. This is the standard oracle for testing operation
/// gradient rules without hand-deriving the expected derivative — and without trusting the machinery under test: the
/// gradient side runs `$function` through [`gradient`](crate::differentiation::gradient), while the
/// finite-difference side evaluates `$function` directly on concrete [`Scalar`](crate::backends::scalars::Scalar) values at
/// the perturbed points, never touching the differentiation machinery it is checking. That double instantiation is
/// why this is a macro: `$function` must be a closure literal (or a generic function), and the expansion
/// instantiates it once over [`LinearizationTracer`](crate::differentiation::LinearizationTracer) inputs and once
/// over concrete [`Scalar`](crate::backends::scalars::Scalar) inputs.
///
/// An `f64` input estimates the ordinary derivative `(f(x + h) - f(x - h)) / (2h)`. A `c128` input requires a
/// ℂ → ℝ `$function` (the only shape the plain [`gradient`](crate::differentiation::gradient) entry point accepts)
/// and estimates both real partials with central differences along the real and imaginary axes, assembling
/// `complex(∂f/∂re, -∂f/∂im)` — the conjugate steepest-ascent gradient the bilinear transposition pairing returns
/// (e.g., `2z̄` for `f(z) = |z|²`). Other input data types (including `c64`, whose `f32` precision cannot support a
/// meaningful central difference) panic. Pick an `$input` away from any non-differentiable point of `$function`
/// (e.g., the kink of `abs` at zero) and a `$tolerance` compatible with the O(`$step`²) truncation error of the
/// central difference.
#[macro_export]
macro_rules! check_gradient {
    ($function:expr, $input:expr, $step:expr, $tolerance:expr $(,)?) => {{
        // Closure parameter types infer from an expected type, so each instantiation of `$function` flows through
        // an identity function pinning the signature that instantiation is used at.
        type EagerScalarContext = $crate::contexts::EagerContext<
            $crate::backends::scalars::Scalar,
            $crate::backends::scalars::ScalarOperation<$crate::backends::scalars::Scalar>,
        >;
        fn pin_traced<F>(function: F) -> F
        where
            F: Fn(
                $crate::differentiation::LinearizationTracer<EagerScalarContext>,
            ) -> $crate::differentiation::LinearizationTracer<EagerScalarContext>,
        {
            function
        }
        fn pin_eager<F: Fn($crate::backends::scalars::Scalar) -> $crate::backends::scalars::Scalar>(function: F) -> F {
            function
        }
        let input: $crate::backends::scalars::Scalar = ::core::convert::Into::into($input);
        let step: f64 = $step;
        let tolerance: f64 = $tolerance;
        let gradient = $crate::differentiation::gradient(pin_traced($function), input).unwrap();
        let evaluate = pin_eager($function);
        let central_difference = |plus: $crate::backends::scalars::Scalar, minus: $crate::backends::scalars::Scalar| {
            (evaluate(plus) - evaluate(minus)) / $crate::backends::scalars::Scalar::from(2.0 * step)
        };
        match input {
            $crate::backends::scalars::Scalar::F64(input) => {
                let estimate = central_difference(
                    $crate::backends::scalars::Scalar::from(input + step),
                    $crate::backends::scalars::Scalar::from(input - step),
                );
                ::approx::assert_abs_diff_eq!(gradient, estimate, epsilon = tolerance);
            }
            $crate::backends::scalars::Scalar::C128(_) => {
                // Both perturbation steps are built as `c128` values (binary `Scalar` arithmetic requires
                // same-variant operands), and the two central differences estimate the two real partials that
                // assemble the conjugate steepest-ascent gradient.
                let real_step = $crate::operations::complex::Complex::complex(
                    &$crate::backends::scalars::Scalar::from(step),
                    &$crate::backends::scalars::Scalar::from(0.0),
                )
                .unwrap();
                let imaginary_step = $crate::operations::complex::Complex::complex(
                    &$crate::backends::scalars::Scalar::from(0.0),
                    &$crate::backends::scalars::Scalar::from(step),
                )
                .unwrap();
                let real_estimate = central_difference(input + real_step, input - real_step);
                let imaginary_estimate = central_difference(input + imaginary_step, input - imaginary_step);
                let estimate =
                    $crate::operations::complex::Complex::complex(&real_estimate, &(-imaginary_estimate)).unwrap();
                ::approx::assert_abs_diff_eq!(gradient, estimate, epsilon = tolerance);
            }
            other => panic!(
                "finite-difference gradient checking requires an f64 or c128 input but got {}",
                $crate::programs::types::Typed::r#type(&other).into_owned(),
            ),
        }
    }};
}

pub use check_gradient;

// TODO(eaplatanios): Promote to a simple built-in `Array` type in `arrays.rs` parallel to `scalars.rs`.
/// Minimal dense array value used by `ryft` tests and documentation examples. Refer to the [module
/// documentation](crate::tests) for more information.
#[derive(Clone, Debug, PartialEq)]
pub struct TestArray {
    /// Staged array type of this test value.
    pub r#type: ArrayType,

    /// Row-major payload used by tests that need concrete interpretation.
    pub values: Vec<f64>,
}

impl TestArray {
    /// Creates a test array from its staged array type and row-major payload.
    pub fn new(r#type: ArrayType, values: Vec<f64>) -> Self {
        Self { r#type, values }
    }

    /// Creates a rank-0 scalar test array.
    pub fn scalar(value: f64) -> Self {
        Self::new(ArrayType::scalar(DataType::F64), vec![value])
    }

    /// Creates a rank-1 test array.
    pub fn vector(values: Vec<f64>) -> Self {
        let r#type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(values.len())]));
        Self::new(r#type, values)
    }

    /// Creates a rank-2 test array.
    pub fn matrix(rows: usize, cols: usize, values: Vec<f64>) -> Self {
        assert_eq!(values.len(), rows * cols);
        let r#type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(rows), Size::Static(cols)]));
        Self::new(r#type, values)
    }

    /// Returns the row-major payload used by concrete test interpretation.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Returns the number of elements represented by `type`. Panics if the type has dynamic dimensions, so this
    /// helper is reserved for types of already-materialized values (which are always fully static); kernels that
    /// materialize values from payload types use [`TestArray::materialized_element_count`] instead.
    pub fn element_count(r#type: &ArrayType) -> usize {
        r#type.element_count().unwrap().unwrap()
    }

    /// Returns the number of elements represented by `type`, or an error when `type` has dynamic dimensions and
    /// therefore cannot be materialized into a concrete payload.
    pub fn materialized_element_count(r#type: &ArrayType) -> Result<usize, ProgramError> {
        r#type.element_count().map_err(|error| TypeError { message: error.to_string() })?.ok_or_else(|| {
            TypeError { message: format!("cannot materialize a value of dynamically sized type {}", r#type) }.into()
        })
    }

    /// Applies an elementwise binary function using scalar broadcasting.
    fn binary(self, rhs: Self, function: impl Fn(f64, f64) -> f64) -> Self {
        let output_type = Broadcastable::broadcast(&self.r#type, &rhs.r#type).unwrap();
        let output_len = Self::element_count(&output_type);
        let left = self.broadcast_values(output_len);
        let right = rhs.broadcast_values(output_len);
        Self {
            r#type: output_type,
            values: left.into_iter().zip(right).map(|(left, right)| function(left, right)).collect(),
        }
    }

    /// Broadcasts the payload to `output_len`.
    fn broadcast_values(&self, output_len: usize) -> Vec<f64> {
        if self.values.len() == output_len {
            self.values.clone()
        } else if self.values.len() == 1 {
            vec![self.values[0]; output_len]
        } else {
            panic!("cannot broadcast {} values to {output_len}", self.values.len());
        }
    }
}

impl Parameter for TestArray {}

impl Display for TestArray {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:?}", self.values)
    }
}

impl Typed for TestArray {
    type Type = ArrayType;

    fn r#type(&self) -> Cow<'_, ArrayType> {
        Cow::Borrowed(&self.r#type)
    }
}

impl Value for TestArray {
    type DispatchDomain = EagerContext<Self>;
    // A concrete `TestArray`'s active context is the test backend's rich eager domain (unlike the constant-only
    // `EagerContext<TestArray>` it declares as its `Value::DispatchDomain`, which cannot bind operations), so free transform
    // entry points such as `crate::batching::batch` serve top-level concrete values.
    type ExecutionDomain = EagerContext<Self, ArrayOperation<Self>>;

    fn dispatch_domain(&self) -> EagerContext<Self> {
        EagerContext::new()
    }

    fn execution_domain(&self) -> EagerContext<Self, ArrayOperation<Self>> {
        EagerContext::new()
    }
}

impl Tag for TestArray {
    #[inline]
    fn tag(self, _key: &str) -> Self {
        self
    }
}

impl TransferToMemory for TestArray {
    /// Re-places this [`TestArray`] in `destination` by updating the [`Memory`](crate::types::Memory) carried by its
    /// type. The payload is host-resident either way, but the carried type must reflect the transfer so that staged
    /// programs whose declared types park values in other memories (e.g., offloaded residuals) accept the
    /// interpreted value.
    #[inline]
    fn transfer_to_memory(&self, destination: crate::types::Memory) -> Self {
        Self { r#type: self.r#type.clone().with_memory(destination), values: self.values.clone() }
    }
}

impl BooleanLike for TestArray {
    /// Returns a [`TestArray`] with a Boolean-typed counterpart of this array's type and with every in-band `f64`
    /// element reinterpreted as Boolean (i.e., `0.0` maps to `0.0`/false and any nonzero element maps to `1.0`/true).
    fn as_boolean(&self) -> Self {
        Self {
            r#type: self.r#type.as_boolean(),
            values: self.values.iter().map(|value| if *value != 0.0 { 1.0 } else { 0.0 }).collect(),
        }
    }

    fn boolean(&self) -> Result<bool, ProgramError> {
        // Accept scalar Boolean predicates (rank-0, one element, encoded as 0.0=false / nonzero=true)
        // so that batch-varying while can extract a final `any(mask)` result. Higher-rank predicates
        // still error because they cannot collapse to a single Boolean.
        if self.r#type.rank() == 0 && self.r#type.data_type() == DataType::Boolean && self.values.len() == 1 {
            return Ok(self.values[0] != 0.0);
        }
        Err(ProgramError::Concretization {
            message: format!(
                "cannot extract a concrete boolean from a value of type {}; expected bool[]",
                self.r#type()
            ),
        })
    }
}

/// Batched while-predicate semantics for [`TestArray`]: `any_true` reduces the whole Boolean payload with `or`, and
/// `mask_select` broadcasts the predicate against the operands along its leading (prefix) axes, so predicate item `i`
/// masks the contiguous per-item block of `on_true` / `on_false` elements it governs.
impl crate::operations::control_flow::WhilePredicate for TestArray {
    fn any_true(&self) -> Result<bool, ProgramError> {
        if self.r#type.data_type() != DataType::Boolean {
            return Err(ProgramError::Concretization {
                message: format!("cannot use a value of type {} as a Boolean while predicate", self.r#type),
            });
        }
        Ok(self.values.iter().any(|value| *value != 0.0))
    }

    fn mask_select(&self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        if self.r#type.data_type() != DataType::Boolean
            || on_true.r#type != on_false.r#type
            || self.values.is_empty()
            || on_true.values.len() % self.values.len() != 0
        {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "mask_select requires a Boolean predicate whose element count divides congruent operands, but \
                     got predicate {} with operands {} and {}",
                    self.r#type, on_true.r#type, on_false.r#type,
                ),
            });
        }
        let block = on_true.values.len() / self.values.len();
        let values = on_true
            .values
            .iter()
            .zip(on_false.values.iter())
            .enumerate()
            .map(|(index, (on_true, on_false))| if self.values[index / block] != 0.0 { *on_true } else { *on_false })
            .collect();
        Ok(Self { r#type: on_true.r#type.clone(), values })
    }
}

impl<O: crate::programs::operations::Operation<ArrayType>> Zero<TestArray> for EagerContext<TestArray, O> {
    fn zero(&self, r#type: &ArrayType) -> Result<TestArray, ProgramError> {
        Ok(TestArray { r#type: r#type.clone(), values: vec![0.0; TestArray::materialized_element_count(r#type)?] })
    }
}

impl<O: crate::programs::operations::Operation<ArrayType>> One<TestArray> for EagerContext<TestArray, O> {
    fn one(&self, r#type: &ArrayType) -> Result<TestArray, ProgramError> {
        Ok(TestArray { r#type: r#type.clone(), values: vec![1.0; TestArray::materialized_element_count(r#type)?] })
    }
}

impl<O: crate::programs::operations::Operation<ArrayType>> Fill<Scalar, TestArray> for EagerContext<TestArray, O> {
    fn fill(&self, r#type: &ArrayType, value: Scalar) -> Result<TestArray, ProgramError> {
        // `TestArray` stores `f64` elements, so any fill value that widens to `f64` is representable and complex
        // fill values surface the cast's promotion error.
        let Scalar::F64(value) = value.cast(DataType::F64)? else { unreachable!("a cast to f64 yields an f64 scalar") };
        Ok(TestArray { r#type: r#type.clone(), values: vec![value; TestArray::materialized_element_count(r#type)?] })
    }
}

impl<O: crate::programs::operations::Operation<ArrayType>> crate::operations::constants::Iota<TestArray>
    for EagerContext<TestArray, O>
{
    fn iota(&self, r#type: &ArrayType, dimension: usize) -> Result<TestArray, ProgramError> {
        let sizes = r#type
            .shape()
            .dimensions()
            .iter()
            .map(|dimension| {
                dimension.value().ok_or_else(|| TypeError {
                    message: format!("cannot materialize an iota of dynamically sized type {type}"),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if dimension >= sizes.len() {
            return Err(TypeError {
                message: format!("iota dimension {dimension} is out of bounds for array type {type}"),
            }
            .into());
        }
        // In row-major order, the index along `dimension` at flat position `flat` is `(flat / stride) % size`, where
        // `stride` is the product of the sizes of the dimensions after `dimension`.
        let size = sizes[dimension];
        let stride: usize = sizes[dimension + 1..].iter().product();
        let element_count = TestArray::materialized_element_count(r#type)?;
        let values = (0..element_count).map(|flat| ((flat / stride) % size) as f64).collect();
        Ok(TestArray { r#type: r#type.clone(), values })
    }
}

impl ZeroLike for TestArray {
    fn zero_like(&self) -> Self {
        Self { r#type: self.r#type.clone(), values: vec![0.0; self.values.len()] }
    }
}

impl OneLike for TestArray {
    fn one_like(&self) -> Self {
        Self { r#type: self.r#type.clone(), values: vec![1.0; self.values.len()] }
    }
}

impl std::ops::Add for TestArray {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.binary(rhs, |left, right| left + right)
    }
}

impl std::ops::Sub for TestArray {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.binary(rhs, |left, right| left - right)
    }
}

impl std::ops::Mul for TestArray {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.binary(rhs, |left, right| left * right)
    }
}

impl std::ops::Mul<f64> for TestArray {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self { r#type: self.r#type, values: self.values.into_iter().map(|value| value * rhs).collect() }
    }
}

impl std::ops::Div for TestArray {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.binary(rhs, |left, right| left / right)
    }
}

impl std::ops::Neg for TestArray {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self { r#type: self.r#type, values: self.values.into_iter().map(|value| -value).collect() }
    }
}

// Fallible Ryft arithmetic capabilities used by operation interpretation. A `TestArray` is always `f64`-backed, so
// these never fail; they delegate to the ergonomic `std::ops` operators and wrap the result.
impl Add for TestArray {
    fn add(&self, rhs: &Self) -> Result<Self, ProgramError> {
        Ok(self.clone() + rhs.clone())
    }
}

impl Sub for TestArray {
    fn sub(&self, rhs: &Self) -> Result<Self, ProgramError> {
        Ok(self.clone() - rhs.clone())
    }
}

impl Mul for TestArray {
    fn mul(&self, rhs: &Self) -> Result<Self, ProgramError> {
        Ok(self.clone() * rhs.clone())
    }
}

impl Div for TestArray {
    fn div(&self, rhs: &Self) -> Result<Self, ProgramError> {
        Ok(self.clone() / rhs.clone())
    }
}

impl Neg for TestArray {
    fn neg(&self) -> Result<Self, ProgramError> {
        Ok(-self.clone())
    }
}

impl ConvertElementType for TestArray {
    fn convert_element_type(&self, data_type: DataType) -> Result<Self, ProgramError> {
        if self.r#type.data_type() == DataType::Token || data_type == DataType::Token {
            return Err(
                TypeError { message: "cannot convert values to or from the token data type".to_string() }.into()
            );
        }
        Ok(Self { r#type: self.r#type.clone().with_data_type(data_type), values: self.values.clone() })
    }
}

impl Sin for TestArray {
    fn sin(&self) -> Result<Self, ProgramError> {
        Ok(Self { r#type: self.r#type.clone(), values: self.values.iter().copied().map(f64::sin).collect() })
    }
}

impl Cos for TestArray {
    fn cos(&self) -> Result<Self, ProgramError> {
        Ok(Self { r#type: self.r#type.clone(), values: self.values.iter().copied().map(f64::cos).collect() })
    }
}

impl Exp for TestArray {
    fn exp(&self) -> Result<Self, ProgramError> {
        Ok(Self { r#type: self.r#type.clone(), values: self.values.iter().copied().map(f64::exp).collect() })
    }
}

impl Log for TestArray {
    fn log(&self) -> Result<Self, ProgramError> {
        Ok(Self { r#type: self.r#type.clone(), values: self.values.iter().copied().map(f64::ln).collect() })
    }
}

impl Sqrt for TestArray {
    fn sqrt(&self) -> Result<Self, ProgramError> {
        Ok(Self { r#type: self.r#type.clone(), values: self.values.iter().copied().map(f64::sqrt).collect() })
    }
}

impl Atan2 for TestArray {
    fn atan2(&self, x: &Self) -> Result<Self, ProgramError> {
        let values = self.values.iter().zip(x.values.iter()).map(|(y, x)| y.atan2(*x)).collect();
        Ok(Self { r#type: self.r#type.clone(), values })
    }
}

impl Abs for TestArray {
    fn abs(&self) -> Result<Self, ProgramError> {
        Ok(Self { r#type: self.r#type.clone(), values: self.values.iter().copied().map(f64::abs).collect() })
    }
}

// `TestArray` stores real `f64` payloads only, so the complex capabilities are uniformly rejected: complex array
// coverage lives in the XLA-backed array domain instead.
impl crate::operations::complex::Complex for TestArray {
    fn complex(&self, _imaginary: &Self) -> Result<Self, ProgramError> {
        Err(TypeError { message: "TestArray does not support complex values".to_string() }.into())
    }
}

impl Conjugate for TestArray {
    fn conjugate(&self) -> Result<Self, ProgramError> {
        Err(TypeError { message: "TestArray does not support complex values".to_string() }.into())
    }
}

impl Real for TestArray {
    fn real(&self) -> Result<Self, ProgramError> {
        Err(TypeError { message: "TestArray does not support complex values".to_string() }.into())
    }
}

impl Imaginary for TestArray {
    fn imaginary(&self) -> Result<Self, ProgramError> {
        Err(TypeError { message: "TestArray does not support complex values".to_string() }.into())
    }
}

impl Broadcast for TestArray {
    fn broadcast(&self, output_type: ArrayType, output_axes: &[usize]) -> Result<Self, ProgramError> {
        let r#type = Broadcast::broadcast(&self.r#type, output_type, output_axes)?;
        let input_shape = self.r#type.static_shape().unwrap();
        let Some(target_shape) = r#type.static_shape() else {
            return Err(TypeError {
                message: format!("cannot materialize a value of dynamically sized type {}", r#type),
            }
            .into());
        };
        let input_rank = input_shape.rank();
        let target_rank = target_shape.rank();
        let input_strides = input_shape.row_major_strides();
        let output_count: usize = target_shape.dimensions().iter().product();
        let mut values = Vec::with_capacity(output_count);
        let mut target_index = vec![0usize; target_rank];
        while values.len() < output_count {
            let mut input_flat = 0usize;
            for input_axis in 0..input_rank {
                let target_axis = output_axes[input_axis];
                let coordinate = if input_shape[input_axis] == 1 { 0 } else { target_index[target_axis] };
                input_flat += coordinate * input_strides[input_axis];
            }
            values.push(self.values[input_flat]);
            for position in (0..target_rank).rev() {
                target_index[position] += 1;
                if target_index[position] < target_shape[position] {
                    break;
                }
                target_index[position] = 0;
            }
        }
        Ok(Self { r#type, values })
    }
}

impl crate::tracing_v2::operations::dot::Dot for TestArray {
    fn dot(&self, rhs: &Self, dimensions: &crate::tracing_v2::operations::dot::DotDimensionNumbers) -> Self {
        let lhs_shape = self.r#type.static_shape().unwrap();
        let rhs_shape = rhs.r#type.static_shape().unwrap();
        let (values, output_shape) = crate::tracing_v2::operations::dot::dot_general_evaluate(
            self.values.as_slice(),
            &lhs_shape,
            rhs.values.as_slice(),
            &rhs_shape,
            dimensions,
            || 0.0f64,
            |accumulator, lhs_value, rhs_value| accumulator + lhs_value * rhs_value,
        );
        let output_type = ArrayType::new(self.r#type.data_type(), Shape::from(&output_shape));
        Self { r#type: output_type, values }
    }
}

impl Transpose for TestArray {
    fn transpose<P: AsRef<[usize]>>(&self, permutation: P) -> Result<Self, ProgramError> {
        // Validate the permutation and compute the output type (including sharding) via the type-level rule, so an
        // out-of-range or duplicated axis is a clean error rather than an out-of-bounds panic.
        let permutation = permutation.as_ref();
        let output_type = self.r#type.transpose(permutation)?;
        if permutation.iter().enumerate().all(|(index, axis)| index == *axis) {
            return Ok(self.clone());
        }
        let shape = self.r#type.static_shape().unwrap();
        let rank = shape.rank();
        let permuted_shape = StaticShape::new(permutation.iter().map(|axis| shape[*axis]).collect());
        let input_strides = shape.row_major_strides();
        let element_count: usize = shape.dimensions().iter().product();
        let mut values = Vec::with_capacity(element_count);
        let mut permuted_index = vec![0usize; rank];
        while values.len() < element_count {
            let mut input_flat = 0usize;
            for (position, &input_axis) in permutation.iter().enumerate() {
                input_flat += permuted_index[position] * input_strides[input_axis];
            }
            values.push(self.values[input_flat]);
            for position in (0..rank).rev() {
                permuted_index[position] += 1;
                if permuted_index[position] < permuted_shape[position] {
                    break;
                }
                permuted_index[position] = 0;
            }
        }
        Ok(Self { r#type: output_type, values })
    }
}

impl Reshape for TestArray {
    fn reshape(&self, target_shape: Shape) -> Result<Self, ProgramError> {
        // Delegate to the type-level reshape so that element-count mismatches and dynamic target shapes surface the
        // canonical reshape errors instead of panicking, and reinterpret the row-major payload under the result.
        let output_type = self.r#type.reshape(target_shape)?;
        Ok(Self { r#type: output_type, values: self.values.clone() })
    }
}

// `TestArray` is a concrete single-device value, so resharding is a no-op on its payload. Its type still records the
// requested distribution metadata so interpreted programs preserve their declared boundaries exactly.
impl crate::operations::sharding::Reshard for TestArray {
    fn reshard(&self, sharding: &crate::Sharding) -> Self {
        let mut output = self.clone();
        output.r#type.sharding = Some(sharding.clone());
        output
    }
}

impl crate::operations::sharding::ConstrainSharding for TestArray {}

impl TestArray {
    /// Copies the row-major block of shape `sizes` out of this array's payload, reading the element at index
    /// `start_indices + block_index * strides` along each axis. The caller guarantees that the block lies in bounds.
    fn copy_block(&self, start_indices: &[usize], strides: &[usize], sizes: &[usize]) -> Vec<f64> {
        let input_shape = self.r#type.static_shape().unwrap();
        let input_strides = input_shape.row_major_strides();
        let rank = input_shape.rank();
        let output_count: usize = sizes.iter().product();
        let mut values = Vec::with_capacity(output_count);
        let mut block_index = vec![0usize; rank];
        while values.len() < output_count {
            let mut input_flat = 0usize;
            for axis in 0..rank {
                input_flat += (start_indices[axis] + block_index[axis] * strides[axis]) * input_strides[axis];
            }
            values.push(self.values[input_flat]);
            for position in (0..rank).rev() {
                block_index[position] += 1;
                if block_index[position] < sizes[position] {
                    break;
                }
                block_index[position] = 0;
            }
        }
        values
    }

    /// Overwrites the row-major block of `update`'s shape starting at `start_indices` in this array's payload with
    /// `update`'s payload. The caller guarantees that the block lies in bounds.
    fn replace_block(mut self, update: &TestArray, start_indices: &[usize]) -> Self {
        let input_shape = self.r#type.static_shape().unwrap();
        let update_shape = update.r#type.static_shape().unwrap();
        let input_strides = input_shape.row_major_strides();
        let rank = input_shape.rank();
        let update_count: usize = update_shape.dimensions().iter().product();
        let mut block_index = vec![0usize; rank];
        let mut written = 0usize;
        while written < update_count {
            let mut input_flat = 0usize;
            for axis in 0..rank {
                input_flat += (start_indices[axis] + block_index[axis]) * input_strides[axis];
            }
            self.values[input_flat] = update.values[written];
            written += 1;
            for position in (0..rank).rev() {
                block_index[position] += 1;
                if block_index[position] < update_shape[position] {
                    break;
                }
                block_index[position] = 0;
            }
        }
        self
    }

    /// Extracts the in-band scalar start indices of a dynamic slicing operation and clamps them per StableHLO
    /// semantics: the effective start index along axis `d` is
    /// `clamp(0, start_indices[d], input_dimension[d] - block_sizes[d])`.
    fn clamped_start_indices(
        start_indices: &[TestArray],
        input_shape: &StaticShape,
        block_sizes: &[usize],
    ) -> Vec<usize> {
        start_indices
            .iter()
            .enumerate()
            .map(|(axis, index)| {
                let raw = index.values[0] as i64;
                let maximum = (input_shape[axis] - block_sizes[axis]) as i64;
                raw.clamp(0, maximum) as usize
            })
            .collect()
    }
}

impl Slice for TestArray {
    fn slice(&self, start_indices: &[usize], limit_indices: &[usize], strides: &[usize]) -> Result<Self, ProgramError> {
        let output_type = self.r#type.slice(start_indices, limit_indices, strides)?;
        let sizes: Vec<usize> = start_indices
            .iter()
            .zip(limit_indices.iter())
            .zip(strides.iter())
            .map(|((start, limit), stride)| (limit - start).div_ceil(*stride))
            .collect();
        let values = self.copy_block(start_indices, strides, sizes.as_slice());
        Ok(Self { r#type: output_type, values })
    }
}

impl UpdateSlice for TestArray {
    fn update_slice(&self, update: &Self, start_indices: &[usize]) -> Result<Self, ProgramError> {
        self.r#type.update_slice(&update.r#type, start_indices)?;
        Ok(self.clone().replace_block(update, start_indices))
    }
}

impl DynamicSlice for TestArray {
    fn dynamic_slice(&self, start_indices: &[Self], sizes: &[usize]) -> Result<Self, ProgramError> {
        let index_types: Vec<ArrayType> = start_indices.iter().map(|index| index.r#type.clone()).collect();
        let output_type = self.r#type.dynamic_slice(&index_types, sizes)?;
        let input_shape = self.r#type.static_shape().unwrap();
        let starts = Self::clamped_start_indices(start_indices, &input_shape, sizes);
        let unit_strides = vec![1; sizes.len()];
        let values = self.copy_block(starts.as_slice(), unit_strides.as_slice(), sizes);
        Ok(Self { r#type: output_type, values })
    }
}

impl DynamicUpdateSlice for TestArray {
    fn dynamic_update_slice(&self, update: &Self, start_indices: &[Self]) -> Result<Self, ProgramError> {
        let index_types: Vec<ArrayType> = start_indices.iter().map(|index| index.r#type.clone()).collect();
        self.r#type.dynamic_update_slice(&update.r#type, &index_types)?;
        let input_shape = self.r#type.static_shape().unwrap();
        let update_shape = update.r#type.static_shape().unwrap();
        let starts = Self::clamped_start_indices(start_indices, &input_shape, update_shape.dimensions());
        Ok(self.clone().replace_block(update, starts.as_slice()))
    }
}

impl Pad for TestArray {
    fn pad(
        &self,
        padding_value: &Self,
        edge_padding_low: &[usize],
        edge_padding_high: &[usize],
        interior_padding: &[usize],
    ) -> Result<Self, ProgramError> {
        let output_type =
            self.r#type.pad(&padding_value.r#type, edge_padding_low, edge_padding_high, interior_padding)?;
        let input_shape = self.r#type.static_shape().unwrap();
        let output_shape = output_type.static_shape().unwrap();
        let output_strides = output_shape.row_major_strides();
        let rank = input_shape.rank();
        let mut values = vec![padding_value.values[0]; Self::element_count(&output_type)];
        let mut input_index = vec![0usize; rank];
        let mut written = 0usize;
        while written < self.values.len() {
            let mut output_flat = 0usize;
            for axis in 0..rank {
                output_flat +=
                    (edge_padding_low[axis] + input_index[axis] * (interior_padding[axis] + 1)) * output_strides[axis];
            }
            values[output_flat] = self.values[written];
            written += 1;
            for position in (0..rank).rev() {
                input_index[position] += 1;
                if input_index[position] < input_shape[position] {
                    break;
                }
                input_index[position] = 0;
            }
        }
        Ok(Self { r#type: output_type, values })
    }
}

impl Concatenate for TestArray {
    fn concatenate(operands: &[Self], axis: usize) -> Result<Self, ProgramError> {
        let operand_types: Vec<ArrayType> = operands.iter().map(|operand| operand.r#type.clone()).collect();
        let output_type = ArrayType::concatenate(&operand_types, axis)?;
        // Each operand owns a contiguous run of `axis` coordinates; writing its block at the running offset along
        // `axis` (and offset zero on every other axis) into a zero-initialized output reuses the row-major
        // odometer in `replace_block`.
        let mut output = Self { r#type: output_type.clone(), values: vec![0.0; Self::element_count(&output_type)] };
        let mut offset = 0usize;
        for operand in operands {
            let operand_axis_size = operand.r#type.static_shape().unwrap()[axis];
            let mut start_indices = vec![0usize; output_type.rank()];
            start_indices[axis] = offset;
            output = output.replace_block(operand, start_indices.as_slice());
            offset += operand_axis_size;
        }
        Ok(output)
    }
}

impl Gather for TestArray {
    fn gather(&self, indices: &Self, operation: &GatherOperation) -> Result<Self, ProgramError> {
        let output_type = self.r#type.gather(&indices.r#type, operation)?;
        let dimensions = operation.dimensions();
        let slice_sizes = operation.slice_sizes();
        let operand_shape = self.r#type.static_shape().unwrap();
        let operand_strides = operand_shape.row_major_strides();
        let indices_shape = indices.r#type.static_shape().unwrap();
        let indices_strides = indices_shape.row_major_strides();
        let output_shape = output_type.static_shape().unwrap();
        let operand_rank = operand_shape.rank();
        let indices_rank = indices_shape.rank();
        let output_rank = output_shape.rank();
        let index_vector_dimension = indices_rank - 1;
        let index_vector_extent = indices_shape[index_vector_dimension];

        // Classify operand axes (window axes carry the slice; collapsed/batching do not) and output axes (offset
        // positions carry the window, the rest carry the indices' batch coordinates).
        let collapsed: BTreeSet<usize> = dimensions.collapsed_slice_dimensions().iter().copied().collect();
        let batching: BTreeSet<usize> = dimensions.operand_batching_dimensions().iter().copied().collect();
        let operand_window_axes: Vec<usize> =
            (0..operand_rank).filter(|axis| !collapsed.contains(axis) && !batching.contains(axis)).collect();
        let offset_positions: BTreeSet<usize> = dimensions.offset_dimensions().iter().copied().collect();
        let batch_output_positions: Vec<usize> =
            (0..output_rank).filter(|position| !offset_positions.contains(position)).collect();
        let indices_batch_axes: Vec<usize> = (0..indices_rank).filter(|axis| *axis != index_vector_dimension).collect();

        let extents = output_shape.dimensions();
        let output_count: usize = extents.iter().product();
        let mut values = Vec::with_capacity(output_count);
        let mut output_index = vec![0usize; output_rank];
        for _ in 0..output_count {
            // Place the output's batch coordinates into the indices multi-index and read this query's start vector.
            let mut indices_index = vec![0usize; indices_rank];
            for (position, &output_position) in batch_output_positions.iter().enumerate() {
                indices_index[indices_batch_axes[position]] = output_index[output_position];
            }
            let mut starts = vec![0i64; index_vector_extent];
            for (component, start) in starts.iter_mut().enumerate() {
                indices_index[index_vector_dimension] = component;
                let flat: usize = (0..indices_rank).map(|axis| indices_index[axis] * indices_strides[axis]).sum();
                *start = indices.values[flat] as i64;
            }
            // Assemble the operand multi-index: window offsets, then batching coordinates, then start offsets.
            let mut operand_index = vec![0i64; operand_rank];
            for (window, &operand_axis) in operand_window_axes.iter().enumerate() {
                operand_index[operand_axis] = output_index[dimensions.offset_dimensions()[window]] as i64;
            }
            for (batch, &operand_axis) in dimensions.operand_batching_dimensions().iter().enumerate() {
                operand_index[operand_axis] =
                    indices_index[dimensions.start_indices_batching_dimensions()[batch]] as i64;
            }
            let mut dropped = false;
            for (component, &operand_axis) in dimensions.start_index_map().iter().enumerate() {
                let raw = starts[component];
                let maximum = (operand_shape[operand_axis] - slice_sizes[operand_axis]) as i64;
                match operation.mode() {
                    GatherScatterMode::FillOrDrop => {
                        if raw < 0 || raw > maximum {
                            dropped = true;
                        }
                        operand_index[operand_axis] += raw;
                    }
                    GatherScatterMode::PromiseInBounds | GatherScatterMode::Clip => {
                        operand_index[operand_axis] += raw.clamp(0, maximum)
                    }
                }
            }
            let value = if dropped {
                0.0
            } else {
                let flat: usize =
                    (0..operand_rank).map(|axis| operand_index[axis] as usize * operand_strides[axis]).sum();
                self.values[flat]
            };
            values.push(value);
            for position in (0..output_rank).rev() {
                output_index[position] += 1;
                if output_index[position] < extents[position] {
                    break;
                }
                output_index[position] = 0;
            }
        }
        Ok(Self { r#type: output_type, values })
    }
}

impl Scatter for TestArray {
    fn scatter(&self, indices: &Self, updates: &Self, operation: &ScatterOperation) -> Result<Self, ProgramError> {
        let output_type = self.r#type.scatter(&indices.r#type, &updates.r#type, operation)?;
        let dimensions = operation.dimensions();
        let operand_shape = self.r#type.static_shape().unwrap();
        let operand_strides = operand_shape.row_major_strides();
        let indices_shape = indices.r#type.static_shape().unwrap();
        let indices_strides = indices_shape.row_major_strides();
        let updates_shape = updates.r#type.static_shape().unwrap();
        let operand_rank = operand_shape.rank();
        let indices_rank = indices_shape.rank();
        let updates_rank = updates_shape.rank();
        let index_vector_dimension = indices_rank - 1;
        let index_vector_extent = indices_shape[index_vector_dimension];

        let inserted: BTreeSet<usize> = dimensions.inserted_window_dimensions().iter().copied().collect();
        let batching: BTreeSet<usize> = dimensions.operand_batching_dimensions().iter().copied().collect();
        let operand_window_axes: Vec<usize> =
            (0..operand_rank).filter(|axis| !inserted.contains(axis) && !batching.contains(axis)).collect();
        let update_window: BTreeSet<usize> = dimensions.update_window_dimensions().iter().copied().collect();
        let update_scatter_axes: Vec<usize> = (0..updates_rank).filter(|axis| !update_window.contains(axis)).collect();
        let indices_batch_axes: Vec<usize> = (0..indices_rank).filter(|axis| *axis != index_vector_dimension).collect();
        // Window size per operand axis (the update extent on window axes, 1 elsewhere), used to clamp the start so the
        // whole window stays in bounds.
        let mut operand_window_size = vec![1usize; operand_rank];
        for (window, &operand_axis) in operand_window_axes.iter().enumerate() {
            operand_window_size[operand_axis] = updates_shape[dimensions.update_window_dimensions()[window]];
        }

        let mut values = self.values.clone();
        let extents = updates_shape.dimensions();
        let update_count: usize = extents.iter().product();
        let mut update_index = vec![0usize; updates_rank];
        for written in 0..update_count {
            let mut indices_index = vec![0usize; indices_rank];
            for (position, &update_axis) in update_scatter_axes.iter().enumerate() {
                indices_index[indices_batch_axes[position]] = update_index[update_axis];
            }
            let mut starts = vec![0i64; index_vector_extent];
            for (component, start) in starts.iter_mut().enumerate() {
                indices_index[index_vector_dimension] = component;
                let flat: usize = (0..indices_rank).map(|axis| indices_index[axis] * indices_strides[axis]).sum();
                *start = indices.values[flat] as i64;
            }
            let mut operand_index = vec![0i64; operand_rank];
            for (window, &operand_axis) in operand_window_axes.iter().enumerate() {
                operand_index[operand_axis] = update_index[dimensions.update_window_dimensions()[window]] as i64;
            }
            for (batch, &operand_axis) in dimensions.operand_batching_dimensions().iter().enumerate() {
                operand_index[operand_axis] =
                    indices_index[dimensions.scatter_indices_batching_dimensions()[batch]] as i64;
            }
            let mut dropped = false;
            for (component, &operand_axis) in dimensions.scatter_dimensions_to_operand_dimensions().iter().enumerate() {
                let raw = starts[component];
                let maximum = (operand_shape[operand_axis] - operand_window_size[operand_axis]) as i64;
                match operation.mode() {
                    GatherScatterMode::FillOrDrop => {
                        if raw < 0 || raw > maximum {
                            dropped = true;
                        }
                        operand_index[operand_axis] += raw;
                    }
                    GatherScatterMode::PromiseInBounds | GatherScatterMode::Clip => {
                        operand_index[operand_axis] += raw.clamp(0, maximum)
                    }
                }
            }
            if !dropped {
                let flat: usize =
                    (0..operand_rank).map(|axis| operand_index[axis] as usize * operand_strides[axis]).sum();
                values[flat] = combine_scatter(operation.kind(), values[flat], updates.values[written]);
            }
            for position in (0..updates_rank).rev() {
                update_index[position] += 1;
                if update_index[position] < extents[position] {
                    break;
                }
                update_index[position] = 0;
            }
        }
        Ok(Self { r#type: output_type, values })
    }
}

/// Combines an existing operand value with a scattered update under the given [`ScatterReductionKind`].
fn combine_scatter(kind: ScatterReductionKind, current: f64, update: f64) -> f64 {
    match kind {
        ScatterReductionKind::Overwrite => update,
        ScatterReductionKind::Add => current + update,
        ScatterReductionKind::Mul => current * update,
        ScatterReductionKind::Min => current.min(update),
        ScatterReductionKind::Max => current.max(update),
    }
}

impl Select for TestArray {
    type Condition = Self;

    fn select(condition: &Self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        // Mirrors the broadcasting `SelectOperation` type-inference contract: the condition must be Boolean-typed,
        // the three operand shapes broadcast together, and the two branch data types promote together to the output
        // data type. The condition is retyped to a branch data type before broadcasting so its Boolean data type
        // acts as a mask rather than promoting into the output.
        assert_eq!(condition.r#type.data_type(), DataType::Boolean, "select condition must have a Boolean data type",);
        let output_type = Broadcastable::broadcast(
            &Broadcastable::broadcast(
                &condition.r#type.clone().with_data_type(on_true.r#type.data_type()),
                &on_true.r#type,
            )
            .unwrap(),
            &on_false.r#type,
        )
        .unwrap();
        let output_len = Self::element_count(&output_type);
        let condition = condition.broadcast_values(output_len);
        let on_true = on_true.broadcast_values(output_len);
        let on_false = on_false.broadcast_values(output_len);
        let values: Vec<f64> = condition
            .into_iter()
            .zip(on_true)
            .zip(on_false)
            .map(|((condition, t), f)| if condition != 0.0 { t } else { f })
            .collect();
        Ok(Self { r#type: output_type, values })
    }
}

impl SelectCondition for TestArray {
    type Condition = Self;

    fn select_condition(&self) -> Result<Self, ProgramError> {
        Ok(self.clone())
    }
}

impl Compare for TestArray {
    type Output = Self;

    fn compare(&self, rhs: &Self, direction: ComparisonDirection) -> Result<Self::Output, ProgramError> {
        let output_shape = self.r#type.shape().clone();
        let output_len = Self::element_count(&self.r#type);
        let left = self.broadcast_values(output_len);
        let right = rhs.broadcast_values(output_len);
        let values: Vec<f64> = left
            .into_iter()
            .zip(right)
            .map(|(left, right)| {
                let predicate = match direction {
                    ComparisonDirection::Equal => left == right,
                    ComparisonDirection::NotEqual => left != right,
                    ComparisonDirection::LessThan => left < right,
                    ComparisonDirection::LessThanOrEqual => left <= right,
                    ComparisonDirection::GreaterThan => left > right,
                    ComparisonDirection::GreaterThanOrEqual => left >= right,
                };
                if predicate { 1.0 } else { 0.0 }
            })
            .collect();
        let output_type = ArrayType::new(DataType::Boolean, output_shape);
        Ok(Self { r#type: output_type, values })
    }
}

impl TestArray {
    /// Applies one elementwise binary logical operator, treating nonzero elements as logically true.
    fn binary_logical(self, rhs: Self, operator: impl Fn(bool, bool) -> bool) -> Self {
        let output_len = Self::element_count(&self.r#type);
        let left = self.broadcast_values(output_len);
        let right = rhs.broadcast_values(output_len);
        let values: Vec<f64> = left
            .into_iter()
            .zip(right)
            .map(|(left, right)| if operator(left != 0.0, right != 0.0) { 1.0 } else { 0.0 })
            .collect();
        Self { r#type: self.r#type, values }
    }
}

impl std::ops::BitAnd for TestArray {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        self.binary_logical(rhs, |left, right| left && right)
    }
}

impl std::ops::BitOr for TestArray {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.binary_logical(rhs, |left, right| left || right)
    }
}

impl std::ops::BitXor for TestArray {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        self.binary_logical(rhs, |left, right| left ^ right)
    }
}

impl std::ops::Not for TestArray {
    type Output = Self;

    fn not(self) -> Self::Output {
        let values: Vec<f64> = self.values.into_iter().map(|value| if value != 0.0 { 0.0 } else { 1.0 }).collect();
        Self { r#type: self.r#type, values }
    }
}

// Fallible Ryft logical capabilities used by operation interpretation. A `TestArray` is always `f64`-backed with
// non-zero values treated as `true`, so these never fail; they delegate to the ergonomic `std::ops` operators and
// wrap the result.
impl And for TestArray {
    fn and(&self, rhs: &Self) -> Result<Self, ProgramError> {
        Ok(self.clone() & rhs.clone())
    }
}

impl Or for TestArray {
    fn or(&self, rhs: &Self) -> Result<Self, ProgramError> {
        Ok(self.clone() | rhs.clone())
    }
}

impl Xor for TestArray {
    fn xor(&self, rhs: &Self) -> Result<Self, ProgramError> {
        Ok(self.clone() ^ rhs.clone())
    }
}

impl Not for TestArray {
    fn not(&self) -> Result<Self, ProgramError> {
        Ok(!self.clone())
    }
}

impl crate::tracing_v2::operations::reduce::Reduce for TestArray {
    fn reduce(&self, axes: &[usize], kind: crate::tracing_v2::operations::reduce::ReductionKind) -> Self {
        use crate::tracing_v2::operations::reduce::{ReductionKind, reduce_evaluate};
        if axes.is_empty() {
            return self.clone();
        }
        let shape = self.r#type.static_shape().unwrap();
        let (reduced_values, reduced_shape) = match kind {
            ReductionKind::Sum | ReductionKind::Mean => {
                reduce_evaluate(self.values.as_slice(), &shape, axes, || 0.0, |acc, value| acc + value)
            }
            ReductionKind::Max => {
                reduce_evaluate(self.values.as_slice(), &shape, axes, || f64::NEG_INFINITY, |acc, value| acc.max(value))
            }
            ReductionKind::Min => {
                reduce_evaluate(self.values.as_slice(), &shape, axes, || f64::INFINITY, |acc, value| acc.min(value))
            }
            ReductionKind::Any => reduce_evaluate(
                self.values.as_slice(),
                &shape,
                axes,
                || 0.0,
                |acc, value| if acc != 0.0 || value != 0.0 { 1.0 } else { 0.0 },
            ),
            ReductionKind::All => reduce_evaluate(
                self.values.as_slice(),
                &shape,
                axes,
                || 1.0,
                |acc, value| if acc != 0.0 && value != 0.0 { 1.0 } else { 0.0 },
            ),
        };
        let mut values = reduced_values;
        if matches!(kind, ReductionKind::Mean) {
            let reduced_count: usize = axes.iter().map(|axis| shape[*axis]).product();
            let divisor = reduced_count.max(1) as f64;
            for value in values.iter_mut() {
                *value /= divisor;
            }
        }
        let data_type = self.r#type.data_type();
        let output_type = ArrayType::new(data_type, Shape::from(&reduced_shape));
        Self { r#type: output_type, values }
    }
}

/// Test [`Operation`] with declared attached-region slots, used to exercise the region-carrying construction,
/// inference, validation, effects, rendering, and rebuild paths before any production operation family migrates onto
/// attached regions. Like the rest of this module, it exists only for tests and documentation examples.
#[derive(Clone, Debug, PartialEq)]
pub enum TestRegionOperation {
    /// Region-free binary addition stand-in used inside region bodies.
    Add,

    /// Region-free unary identity stand-in with an observable ordered-IO effect.
    Effectful,

    /// Region-carrying operation declaring the provided region slot names. Its inferred output types are the first
    /// attached region's output types, which pins that region interfaces are derived and delivered during inference.
    WithRegions(&'static [&'static str]),
}

impl Operation<DataType> for TestRegionOperation {
    fn name(&self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Effectful => "effectful",
            Self::WithRegions(_) => "with_regions",
        }
    }

    fn infer_output_types(
        &self,
        input_types: &[DataType],
        region_interfaces: &[RegionInterface<DataType>],
    ) -> Result<Vec<DataType>, TypeError> {
        match self {
            Self::Add => {
                check_count!("input", input_types, 2, TypeError);
                Ok(vec![input_types[0]])
            }
            Self::Effectful => {
                check_count!("input", input_types, 1, TypeError);
                Ok(vec![input_types[0]])
            }
            Self::WithRegions(names) => {
                check_count!("input", input_types, 1, TypeError);
                if region_interfaces.len() != names.len() {
                    return Err(TypeError {
                        message: format!(
                            "expected {} region interfaces but got {}",
                            names.len(),
                            region_interfaces.len(),
                        ),
                    });
                }
                Ok(region_interfaces[0].output_types().to_vec())
            }
        }
    }

    fn region_names(&self) -> &'static [&'static str] {
        match self {
            Self::Add | Self::Effectful => &[],
            Self::WithRegions(names) => names,
        }
    }

    fn effects(&self) -> Effects {
        match self {
            Self::Add | Self::WithRegions(_) => Effects::PURE,
            Self::Effectful => Effects::single(Effect::OrderedIo),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_test_array_constant_kernels_reject_dynamically_sized_types() {
        // Kernels that materialize a payload from a type cannot do so for dynamically sized types and must error
        // instead of panicking.
        let dynamic_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None), Size::Static(3)]));
        let expected_message = "cannot materialize a value of dynamically sized type f64[*, 3]";
        let context = EagerContext::<TestArray>::new();
        assert!(matches!(
            context.zero(&dynamic_type),
            Err(ProgramError::Type(TypeError { message })) if message == expected_message,
        ));
        assert!(matches!(
            context.one(&dynamic_type),
            Err(ProgramError::Type(TypeError { message })) if message == expected_message,
        ));
        assert!(matches!(
            context.fill(&dynamic_type, Scalar::from(42.0)),
            Err(ProgramError::Type(TypeError { message })) if message == expected_message,
        ));
    }
}

#[cfg(test)]
mod differentiation_tests {
    use half::{bf16, f16};
    use pretty_assertions::assert_eq;

    use crate::backends::scalars::{Scalar, ScalarOperation};
    use crate::contexts::{Context, EagerContext};
    use crate::differentiation::{DifferentiationTracer, LinearizationTracer};
    use crate::tracing_v2::{ForwardModeDifferentiate, ReverseModeDifferentiate};

    #[test]
    fn test_scalar_domain_half_precision_variants_run_jvp() {
        // The unified domain interprets each half-precision `DataType` through the matching `Scalar` variant, so
        // the former `bf16`- and `f16`-specific domains are now two variants exercised over the one domain.
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        assert_eq!(
            domain.jvp(|x| Ok(x.clone() + x), Scalar::BF16(bf16::from_f32(3.0)), Scalar::BF16(bf16::ONE)),
            Ok((Scalar::BF16(bf16::from_f32(6.0)), Scalar::BF16(bf16::from_f32(2.0))))
        );
        assert_eq!(
            domain.jvp(|x| Ok(x.clone() + x), Scalar::F16(f16::from_f32(3.0)), Scalar::F16(f16::ONE)),
            Ok((Scalar::F16(f16::from_f32(6.0)), Scalar::F16(f16::from_f32(2.0))))
        );
    }

    #[test]
    fn test_jvp_takes_the_symbolic_zero_fast_path_for_rule_less_operations() {
        use crate::contexts::EagerContext;
        use crate::operations::differentiation::StopGradient;
        use crate::tests::TestArray;
        use crate::tracing_v2::ArrayOperation;
        use crate::tracing_v2::operations::collective::{CollectiveKind, CollectiveOperation};

        // `stop_gradient` severs the incoming tangent, making every downstream collective input tangent a symbolic
        // zero, so `DifferentiationContext::bind`'s all-zero fast path computes the primal by binding the operation and emits a
        // zero output tangent without consulting the per-operation rule. The function is
        // `f(x) = x + psum(stop_gradient(x))`, which differentiates like `x + c`, so the tangent equals the input
        // tangent. `jvp` runs the closure directly on duals, so the fast path fires as the closure binds the
        // collective operation.
        let (primal, tangent) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jvp(
                |x: DifferentiationTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| {
                    let severed = x.stop_gradient();
                    let mut outputs = severed.context().bind(
                        CollectiveOperation::new("batch".to_string(), CollectiveKind::PSum),
                        Vec::new(),
                        &[severed.clone()],
                    )?;
                    Ok(x + outputs.remove(0))
                },
                TestArray::scalar(2.0),
                TestArray::scalar(1.0),
            )
            .unwrap();
        assert_eq!(primal.values, vec![4.0]);
        assert_eq!(tangent.values, vec![1.0]);
    }

    #[test]
    fn test_linearize_scalar_straight_line_matches_jvp() {
        use crate::backends::scalars::ScalarOperation;
        use crate::contexts::EagerContext;
        use crate::operations::math::Sin;

        // `f(x) = x * sin(x)`: the linearized map's primal output equals `jvp`'s primal output, and applying it to two
        // distinct tangents reproduces `jvp`'s tangent output each time. Linearizing once and applying many times is
        // the headline `linearize` capability.
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let function = |x: LinearizationTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>| Ok(x.clone() * x.sin()?);
        let jvp_function =
            |x: DifferentiationTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>| Ok(x.clone() * x.sin()?);
        let (output, forward) = domain.linearize(function, Scalar::from(0.7)).unwrap();

        let (reference_primal, _) = domain.jvp(jvp_function, Scalar::from(0.7), Scalar::from(1.0)).unwrap();
        assert_eq!(output, reference_primal);

        for tangent in [1.0, -2.5] {
            let (_, reference_tangent) = domain.jvp(jvp_function, Scalar::from(0.7), Scalar::from(tangent)).unwrap();
            assert_eq!(forward.apply(Scalar::from(tangent)).unwrap(), reference_tangent);
        }
    }

    #[test]
    fn test_linearize_scalar_multi_input_matches_jvp() {
        use crate::backends::scalars::ScalarOperation;
        use crate::contexts::EagerContext;
        use crate::operations::math::Sin;

        // `f(a, b) = a * b + sin(a)`: a two-input function whose linearization is applied at two distinct tangent
        // pairs, exercising the residual-input routing for several primal inputs.
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let function = |(a, b): (
            LinearizationTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>,
            LinearizationTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>,
        )| Ok(a.clone() * b + a.sin()?);
        let jvp_function = |(a, b): (
            DifferentiationTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>,
            DifferentiationTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>,
        )| Ok(a.clone() * b + a.sin()?);
        let (output, forward) = domain.linearize(function, (Scalar::from(0.5), Scalar::from(1.3))).unwrap();

        let (reference_primal, _) = domain
            .jvp(jvp_function, (Scalar::from(0.5), Scalar::from(1.3)), (Scalar::from(1.0), Scalar::from(1.0)))
            .unwrap();
        assert_eq!(output, reference_primal);

        for (da, db) in [(1.0, 0.0), (0.0, 1.0), (2.0, -1.0)] {
            let (_, reference_tangent) = domain
                .jvp(jvp_function, (Scalar::from(0.5), Scalar::from(1.3)), (Scalar::from(da), Scalar::from(db)))
                .unwrap();
            let tangent = forward.apply((Scalar::from(da), Scalar::from(db))).unwrap();
            assert_eq!(tangent, reference_tangent);
        }
    }

    #[test]
    fn test_linearize_array_straight_line_matches_jvp() {
        use crate::contexts::EagerContext;
        use crate::operations::math::Sin;
        use crate::tests::TestArray;
        use crate::tracing_v2::ArrayOperation;

        // The array-domain counterpart of the straight-line scalar test: `f(x) = x * sin(x)` over a `TestArray`. Two
        // distinct tangents are applied through the one linearization and each matches `jvp`.
        let function =
            |x: LinearizationTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| Ok(x.clone() * x.sin()?);
        let jvp_function =
            |x: DifferentiationTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| Ok(x.clone() * x.sin()?);
        let (output, forward) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .linearize(function, TestArray::scalar(0.7))
            .unwrap();

        let (reference_primal, _) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jvp(jvp_function, TestArray::scalar(0.7), TestArray::scalar(1.0))
            .unwrap();
        assert_eq!(output.values, reference_primal.values);

        for tangent in [1.0, -2.5] {
            let (_, reference_tangent) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
                .jvp(jvp_function, TestArray::scalar(0.7), TestArray::scalar(tangent))
                .unwrap();
            let result = forward.apply(TestArray::scalar(tangent)).unwrap();
            assert_eq!(result.values, reference_tangent.values);
        }
    }

    #[test]
    fn test_linearize_through_condition_matches_jvp() {
        use crate::contexts::EagerContext;
        use crate::operations::control_flow::ConditionOperation;
        use crate::tests::TestArray;
        use crate::tracing_v2::ArrayOperation;
        use crate::tracing_v2::test_util::scalar_scale_branch;
        use crate::types::{ArrayType, DataType};

        // Control-flow correctness signal: a `condition` whose predicate is a constant `true` selects the scale-by-2
        // branch. Forward linearization fuses the JVP program (which stages a `condition`), then partially evaluates it
        // with the input tangent unknown — the known-predicate `condition` inlines its selected branch through its
        // executable partial-evaluation rule, so the residual tangent map is the scale-by-2 linear map. The result
        // must match `jvp` both for the primal and the tangent.
        let condition_function = |x: LinearizationTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| {
            let condition_regions = vec![scalar_scale_branch(2.0), scalar_scale_branch(3.0)];
            let condition = ConditionOperation::new();
            let predicate = x.context().lift(TestArray::new(ArrayType::scalar(DataType::Boolean), vec![1.0]))?;
            let mut outputs = x.context().bind(
                ArrayOperation::Condition(condition),
                condition_regions.clone(),
                &[predicate, x.clone()],
            )?;
            Ok(outputs.remove(0))
        };
        let condition_jvp_function = |x: DifferentiationTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| {
            let condition_regions = vec![scalar_scale_branch(2.0), scalar_scale_branch(3.0)];
            let condition = ConditionOperation::new();
            let predicate = x.context().lift(TestArray::new(ArrayType::scalar(DataType::Boolean), vec![1.0]))?;
            let mut outputs = x.context().bind(
                ArrayOperation::Condition(condition),
                condition_regions.clone(),
                &[predicate, x.clone()],
            )?;
            Ok(outputs.remove(0))
        };

        let (output, forward) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .linearize(condition_function, TestArray::scalar(4.0))
            .unwrap();
        let (reference_primal, _) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jvp(condition_jvp_function, TestArray::scalar(4.0), TestArray::scalar(1.0))
            .unwrap();
        assert_eq!(output.values, reference_primal.values);
        assert_eq!(output.values, vec![8.0]);

        for tangent in [1.0, 3.0] {
            let (_, reference_tangent) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
                .jvp(condition_jvp_function, TestArray::scalar(4.0), TestArray::scalar(tangent))
                .unwrap();
            let result = forward.apply(TestArray::scalar(tangent)).unwrap();
            assert_eq!(result.values, reference_tangent.values);
            // The selected branch scales by 2, so the directional derivative is `2 * tangent`.
            assert_eq!(result.values, vec![2.0 * tangent]);
        }
    }

    #[test]
    fn test_linearize_through_eager_unbounded_while_matches_jvp() {
        // Eager forward linearization through an unbounded `while x < 8 { x = x + x }`: the eager unroll-then-fuse
        // pre-pass unrolls the loop at the concrete primal, so the fused JVP program is straight-line and partially
        // evaluates cleanly. From `x = 1` the loop doubles three times, so `f(x) = 8 x` locally; the primal is 8 and
        // every directional derivative is `8 * tangent`. This matches `jvp`, whose eager `while` rule differentiates
        // the same loop directly at the concrete primal.
        use crate::contexts::EagerContext;
        use crate::tests::TestArray;
        use crate::tracing_v2::ArrayOperation;

        let while_function = |x: LinearizationTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| {
            let (while_operation, while_regions) = doubling_while_for_linearize();
            let mut outputs = x.context().bind(ArrayOperation::While(while_operation), while_regions, &[x.clone()])?;
            Ok(outputs.remove(0))
        };
        let while_jvp_function = |x: DifferentiationTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| {
            let (while_operation, while_regions) = doubling_while_for_linearize();
            let mut outputs = x.context().bind(ArrayOperation::While(while_operation), while_regions, &[x.clone()])?;
            Ok(outputs.remove(0))
        };

        let (output, forward) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .linearize(while_function, TestArray::scalar(1.0))
            .unwrap();
        assert_eq!(output.values, vec![8.0]);

        let (reference_primal, _) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jvp(while_jvp_function, TestArray::scalar(1.0), TestArray::scalar(1.0))
            .unwrap();
        assert_eq!(output.values, reference_primal.values);

        for tangent in [1.0, 2.0] {
            let (_, reference_tangent) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
                .jvp(while_jvp_function, TestArray::scalar(1.0), TestArray::scalar(tangent))
                .unwrap();
            let result = forward.apply(TestArray::scalar(tangent)).unwrap();
            assert_eq!(result.values, reference_tangent.values);
            assert_eq!(result.values, vec![8.0 * tangent]);
        }
    }

    #[test]
    fn test_linearize_through_scan_matches_jvp() {
        use crate::contexts::EagerContext;
        use crate::tests::TestArray;
        use crate::tracing_v2::ArrayOperation;

        // Forward linearization through a statically-sized cumulative-product `scan` (body `[carry, x] -> [carry*x,
        // carry*x]`, length 3): the fused JVP program retains a `scan`, so partial evaluation exercises the scan's
        // value-carrying partial-evaluation rule with the carry/scanned tangents unknown. `f(init, xs)` is the final
        // carry `init * xs[0] * xs[1] * xs[2] = 24` at `init = 1, xs = [2, 3, 4]`. The result must match `jvp` for the
        // primal and for several distinct tangent pairs.
        let scan_function = |(init, xs): (
            LinearizationTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>,
            LinearizationTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>,
        )| {
            let (scan, scan_body) = product_scan_for_linearize();
            let mut outputs = init.context().bind(ArrayOperation::Scan(scan), vec![scan_body], &[init.clone(), xs])?;
            Ok(outputs.remove(0))
        };
        let scan_jvp_function = |(init, xs): (
            DifferentiationTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>,
            DifferentiationTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>,
        )| {
            let (scan, scan_body) = product_scan_for_linearize();
            let mut outputs = init.context().bind(ArrayOperation::Scan(scan), vec![scan_body], &[init.clone(), xs])?;
            Ok(outputs.remove(0))
        };

        let primals = (TestArray::scalar(1.0), TestArray::vector(vec![2.0, 3.0, 4.0]));
        let (output, forward) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .linearize(scan_function, primals.clone())
            .unwrap();
        assert_eq!(output.values, vec![24.0]);

        let (reference_primal, _) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jvp(scan_jvp_function, primals.clone(), (TestArray::scalar(0.0), TestArray::vector(vec![0.0, 0.0, 0.0])))
            .unwrap();
        assert_eq!(output.values, reference_primal.values);

        let tangents = [
            (TestArray::scalar(1.0), TestArray::vector(vec![0.0, 0.0, 0.0])),
            (TestArray::scalar(0.0), TestArray::vector(vec![1.0, 0.0, 0.0])),
            (TestArray::scalar(0.5), TestArray::vector(vec![1.0, -1.0, 2.0])),
        ];
        for (init_tangent, xs_tangent) in tangents {
            let (_, reference_tangent) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
                .jvp(scan_jvp_function, primals.clone(), (init_tangent.clone(), xs_tangent.clone()))
                .unwrap();
            let result = forward.apply((init_tangent, xs_tangent)).unwrap();
            assert_eq!(result.values, reference_tangent.values);
        }
    }

    #[test]
    fn test_vjp_pullback_apply_scalar_matches_raw_parts() {
        use crate::backends::scalars::ScalarOperation;
        use crate::contexts::EagerContext;
        use crate::operations::math::Sin;

        // The `Pullback::apply` callable surface must reproduce the raw opened parts: interpreting the raw pullback manually at
        // `[cotangent ++ residuals]` and applying `Pullback::apply` to the same cotangent must agree, for two distinct
        // cotangents. `f(x) = x * sin(x)`.
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let context = crate::contexts::EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let function = |x: LinearizationTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>| Ok(x.clone() * x.sin()?);

        let (output, pullback) = domain.vjp(function, Scalar::from(0.7)).unwrap();
        let (reference_output, reference_pullback) = domain.vjp(function, Scalar::from(0.7)).unwrap();
        let (reference_pullback, reference_residuals) = reference_pullback.into_parts();
        assert_eq!(output, reference_output);

        for cotangent in [1.0, 2.5] {
            let mut reference_inputs = vec![Scalar::from(cotangent)];
            reference_inputs.extend(reference_residuals.iter().cloned());
            let reference_cotangents = reference_pullback.interpret_in_context(&context, reference_inputs).unwrap();
            let input_cotangent = pullback.apply(Scalar::from(cotangent)).unwrap();
            assert_eq!(vec![input_cotangent], reference_cotangents);
        }
    }

    #[test]
    fn test_vjp_pullback_apply_array_multi_input_matches_raw_parts() {
        use crate::contexts::EagerContext;
        use crate::tests::TestArray;
        use crate::tracing_v2::ArrayOperation;

        // The array-domain, multi-input `Pullback::apply`: `f(a, b) = a * b` returns one scalar output whose pullback maps the
        // output cotangent to `(b * cotangent, a * cotangent)`. The callable's reshaped input cotangents must match the
        // raw pullback interpreted manually.
        let context = crate::contexts::EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let function = |(a, b): (
            LinearizationTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>,
            LinearizationTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>,
        )| Ok(a * b);
        let (output, pullback) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .vjp(function, (TestArray::scalar(3.0), TestArray::scalar(2.0)))
            .unwrap();
        let (_, reference_pullback) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .vjp(function, (TestArray::scalar(3.0), TestArray::scalar(2.0)))
            .unwrap();
        let (reference_pullback, reference_residuals) = reference_pullback.into_parts();
        assert_eq!(output.values, vec![6.0]);

        for cotangent in [1.0, 4.0] {
            let mut reference_inputs = vec![TestArray::scalar(cotangent)];
            reference_inputs.extend(reference_residuals.iter().cloned());
            let reference_cotangents = reference_pullback.interpret_in_context(&context, reference_inputs).unwrap();
            let (cotangent_a, cotangent_b) = pullback.apply(TestArray::scalar(cotangent)).unwrap();
            assert_eq!(cotangent_a.values, reference_cotangents[0].values);
            assert_eq!(cotangent_b.values, reference_cotangents[1].values);
            // `d(a*b)/da = b = 2` and `d(a*b)/db = a = 3`, scaled by the cotangent.
            assert_eq!(cotangent_a.values, vec![2.0 * cotangent]);
            assert_eq!(cotangent_b.values, vec![3.0 * cotangent]);
        }
    }

    /// Builds the eager unbounded `while x < 8 { x = x + x }` doubling loop over the array domain used by the
    /// forward-linearization control-flow test.
    fn doubling_while_for_linearize() -> (
        crate::operations::control_flow::WhileOperation,
        Vec<
            crate::Program<
                crate::tests::TestArray,
                crate::tracing_v2::ArrayOperation<crate::tests::TestArray>,
                Vec<crate::tests::TestArray>,
                Vec<crate::tests::TestArray>,
            >,
        >,
    ) {
        use crate::operations::compare::{CompareOperation, ComparisonDirection};
        use crate::operations::math::AddOperation;
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;
        use crate::tests::TestArray;
        use crate::tracing_v2::ArrayOperation;
        use crate::types::{ArrayType, DataType};

        type TestOp = ArrayOperation<TestArray>;
        let scalar_f64 = ArrayType::scalar(DataType::F64);

        let mut condition_builder = ProgramBuilder::<TestArray, TestOp>::new();
        let condition_state = condition_builder.add_input(scalar_f64.clone());
        let threshold = condition_builder.add_constant(TestArray::scalar(8.0));
        let predicate = condition_builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::LessThan),
                Vec::new(),
                vec![condition_state, threshold],
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestArray, TestOp>::new();
        let body_state = body_builder.add_input(scalar_f64);
        let doubled = body_builder.add_instruction(AddOperation, Vec::new(), vec![body_state, body_state]).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();

        (crate::operations::control_flow::WhileOperation::new(), vec![condition, body])
    }

    /// Builds the statically-sized cumulative-product [`ScanOperation`](crate::operations::control_flow::ScanOperation)
    /// (body `[carry, x] -> [carry * x, carry * x]`, one scanned input, length 3) used by the forward-linearization
    /// scan test.
    fn product_scan_for_linearize() -> (
        crate::operations::control_flow::ScanOperation<crate::tests::TestArray>,
        crate::Program<
            crate::tests::TestArray,
            crate::tracing_v2::ArrayOperation<crate::tests::TestArray>,
            Vec<crate::tests::TestArray>,
            Vec<crate::tests::TestArray>,
        >,
    ) {
        use crate::operations::math::MulOperation;
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;
        use crate::tests::TestArray;
        use crate::tracing_v2::ArrayOperation;
        use crate::types::{ArrayType, DataType};

        type TestOp = ArrayOperation<TestArray>;
        let mut body_builder = ProgramBuilder::<TestArray, TestOp>::new();
        let carry = body_builder.add_input(ArrayType::scalar(DataType::F64));
        let x = body_builder.add_input(ArrayType::scalar(DataType::F64));
        let product = body_builder.add_instruction(MulOperation, Vec::new(), vec![carry, x]).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(
                vec![product, product],
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        (crate::operations::control_flow::ScanOperation::new(1, 3), body)
    }
}

#[cfg(test)]
mod linearization_tests {
    use crate::backends::scalars::{Scalar, ScalarOperation};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{
        DifferentiationTracer, ForwardModeDifferentiate, LinearizationTracer, ReverseModeDifferentiate,
    };
    use crate::operations::compare::Compare;
    use crate::operations::control_flow::{Select, WhileOperation};
    use crate::operations::differentiation::StopGradient;
    use crate::operations::math::{Cos, Sin};
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::{Program, ProgramBuilder};
    use crate::tracing::{NestedTracingContext, Trace, Tracer};
    use crate::tracing_v2::unroll::unroll_concretizable_whiles;
    use crate::types::DataType;

    use super::*;

    /// Tracer leaf seen by the scalar test closures.
    type ScalarTracer = Tracer<NestedTracingContext<EagerContext<Scalar, ScalarOperation<Scalar>>>>;

    /// Forward-mode dual leaf seen by the scalar `jvp` closures.
    type ScalarJvpTracer = DifferentiationTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>;

    /// Forward-mode dual leaf seen by the scalar closures handed to [`ForwardModeDifferentiate::linearize`] and
    /// [`ReverseModeDifferentiate::vjp`].
    type ScalarLinearizationTracer = LinearizationTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>;

    /// Absolute tolerance for comparing the path against the established transforms.
    const TOLERANCE: f64 = 1e-12;

    /// Extracts the `f64` payload of a floating-point [`Scalar`], panicking on any other variant.
    fn scalar_f64(scalar: &Scalar) -> f64 {
        match scalar {
            Scalar::F64(value) => *value,
            other => panic!("expected an f64 scalar but got {}", other.r#type().into_owned()),
        }
    }

    /// Asserts that every element of `left` is within [`TOLERANCE`] of the corresponding element of `right`.
    fn assert_close(left: &[Scalar], right: &[Scalar], label: &str) {
        assert_eq!(left.len(), right.len(), "{label}: length mismatch ({left:?} vs {right:?})");
        for (index, (a, b)) in left.iter().zip(right).enumerate() {
            let (a, b) = (scalar_f64(a), scalar_f64(b));
            assert!((a - b).abs() <= TOLERANCE, "{label}: element {index} differs ({a} vs {b})");
        }
    }

    /// Asserts forward equivalence: the primal and tangent sub-programs, reassembled, equal the outputs of
    /// [`ForwardModeDifferentiate::jvp`] for `function` at `primals` with the given `tangents`.
    fn assert_forward_equivalent<JvpFunction, LinearizeFunction>(
        jvp_function: JvpFunction,
        linearize_function: LinearizeFunction,
        primals: Vec<Scalar>,
        tangents: Vec<Scalar>,
    ) where
        JvpFunction: FnOnce(Vec<ScalarJvpTracer>) -> Result<Vec<ScalarJvpTracer>, ProgramError>,
        LinearizeFunction: FnOnce(Vec<ScalarTracer>) -> Result<Vec<ScalarTracer>, ProgramError>,
    {
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (reference_primals, reference_tangents) =
            domain.jvp(jvp_function, primals.clone(), tangents.clone()).unwrap();

        let input_types = primals.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
        let (_, primal_program) = NestedTracingContext::trace(domain.clone(), linearize_function, input_types).unwrap();
        let linearization = primal_program.into_simplified().unwrap().linearize().unwrap();
        let context = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();

        // The known side computes the primal outputs followed by the residuals; interpreting it recovers the concrete
        // primal outputs that the linearization core no longer caches.
        let mut known_outputs = linearization.primal().interpret_in_context(&context, primals).unwrap();
        let residuals = known_outputs.split_off(known_outputs.len() - linearization.residual_count());
        assert_close(&known_outputs, &reference_primals, "forward primal");

        // The unknown side is the linear tangent map, taking the tangents followed by the residuals. Canonical arity
        // places all tangent outputs on the unknown side in original order, so they are compared directly.
        let mut tangent_inputs = tangents;
        tangent_inputs.extend(residuals);
        let unknown_outputs = linearization.tangent().interpret_in_context(&context, tangent_inputs).unwrap();
        assert_close(&unknown_outputs, &reference_tangents, "forward tangent");
    }

    /// Runs the raw fused-JVP program pipeline — trace, eager `while` unroll, fused JVP program build,
    /// simplification, and direct interpretation at `(primals ++ tangents)` — as an independent oracle for
    /// [`ForwardModeDifferentiate::jvp`]: the dual-interpreter entry point (including its eager `while` rule) must
    /// agree with the program-level pipeline. Returns the flat primal and tangent outputs.
    fn fused_pipeline_jvp<Function>(
        domain: &EagerContext<Scalar, ScalarOperation<Scalar>>,
        function: Function,
        primals: Vec<Scalar>,
        tangents: Vec<Scalar>,
    ) -> Result<(Vec<Scalar>, Vec<Scalar>), ProgramError>
    where
        Function: FnOnce(Vec<ScalarTracer>) -> Vec<ScalarTracer>,
    {
        let input_types = primals.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
        let (_, program) = NestedTracingContext::trace(domain.clone(), |inputs| Ok(function(inputs)), input_types)?;
        let program = unroll_concretizable_whiles(domain, program.into_simplified()?, primals.clone())?;
        let jvp_program = program.jvp()?.into_simplified()?;
        let mut combined_inputs = primals;
        combined_inputs.extend(tangents);
        let mut outputs = jvp_program.interpret_in_context(domain, combined_inputs)?;
        let tangent_outputs = outputs.split_off(outputs.len() / 2);
        Ok((outputs, tangent_outputs))
    }

    /// Reverse-mode-differentiates `function` at `primals` through the raw program pipeline — trace, eager `while`
    /// unroll, direct program linearization, primal replay, and partition-aware transposition — so the packaged
    /// closure-level [`ReverseModeDifferentiate::vjp`] surface can be compared against the independently staged
    /// program-level path. Returns the flat primal outputs, the pullback over `(output_cotangents ++ residuals)`, and
    /// the residuals.
    fn vjp_direct<Function>(
        domain: &EagerContext<Scalar, ScalarOperation<Scalar>>,
        function: Function,
        primals: Vec<Scalar>,
    ) -> Result<
        (Vec<Scalar>, Program<Scalar, ScalarOperation<Scalar>, Vec<Scalar>, Vec<Scalar>>, Vec<Scalar>),
        ProgramError,
    >
    where
        Function: FnOnce(Vec<ScalarTracer>) -> Result<Vec<ScalarTracer>, ProgramError>,
    {
        let input_types = primals.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
        let (_, program) = NestedTracingContext::trace(domain.clone(), function, input_types)?;
        let program = unroll_concretizable_whiles(domain, program.into_simplified()?, primals.clone())?;
        let linearization = program.linearize()?;
        let primal_side = linearization.primal().interpret_in_context(domain, primals)?;
        let primal_output_count = primal_side.len() - linearization.residual_count();
        let residuals = primal_side[primal_output_count..].to_vec();
        let primal_outputs = primal_side[..primal_output_count].to_vec();
        let pullback = linearization.pullback()?;
        Ok((primal_outputs, pullback, residuals))
    }

    /// Asserts reverse equivalence: the raw-pipeline pullback (built by [`vjp_direct`], which transposes the tangent
    /// sub-program in the primal `ScalarOperation` enum) yields the same input cotangents as the packaged
    /// [`ReverseModeDifferentiate::vjp`] pullback for `function` at `primals`, for the given `output_cotangents`. The
    /// same `function` is supplied twice because each consuming entry point traces it once into a primal program.
    ///
    /// Both pullbacks consume the residuals as ordinary pullback inputs, so each is interpreted at
    /// `output_cotangents ++ residuals`.
    fn assert_reverse_equivalent<VjpFunction, LinearizeFunction>(
        vjp_function: VjpFunction,
        linearize_function: LinearizeFunction,
        primals: Vec<Scalar>,
        output_cotangents: Vec<Scalar>,
    ) where
        VjpFunction: FnOnce(Vec<ScalarLinearizationTracer>) -> Result<Vec<ScalarLinearizationTracer>, ProgramError>,
        LinearizeFunction: FnOnce(Vec<ScalarTracer>) -> Result<Vec<ScalarTracer>, ProgramError>,
    {
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let context = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();

        let (_, pullback) = domain.vjp(vjp_function, primals.clone()).unwrap();
        let (pullback, vjp_residuals) = pullback.into_parts();
        let mut reference_inputs = output_cotangents.clone();
        reference_inputs.extend(vjp_residuals);
        let reference_cotangents = pullback.interpret_in_context(&context, reference_inputs).unwrap();

        let (_, pullback, residuals) = vjp_direct(&domain, linearize_function, primals).unwrap();
        let mut pullback_inputs = output_cotangents;
        pullback_inputs.extend(residuals);
        let input_cotangents = pullback.interpret_in_context(&context, pullback_inputs).unwrap();
        assert_close(&input_cotangents, &reference_cotangents, "reverse cotangent");
    }

    /// Builds the eager, data-dependent `while x < 100 { x = x * x }` loop over the `f64` scalar domain. Its trip count
    /// depends on the runtime value and the loop carries no iteration bound, so it is the kind of unbounded loop the
    /// front end rejects unless the eager unroll-then-fuse pre-pass first unrolls it at the concrete primal.
    fn scalar_squaring_while()
    -> (WhileOperation, Vec<Program<Scalar, ScalarOperation<Scalar>, Vec<Scalar>, Vec<Scalar>>>) {
        use crate::operations::compare::{CompareOperation, ComparisonDirection};
        use crate::operations::math::MulOperation;

        let condition = {
            let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
            let carry = builder.add_input(DataType::F64);
            let threshold = builder.add_constant(Scalar::from(100.0));
            let predicate = builder
                .add_instruction(
                    CompareOperation::new(ComparisonDirection::LessThan),
                    Vec::new(),
                    vec![carry, threshold],
                )
                .unwrap()[0];
            builder
                .build::<Vec<Scalar>, Vec<Scalar>>(vec![predicate], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
            let carry = builder.add_input(DataType::F64);
            let squared = builder.add_instruction(MulOperation, Vec::new(), vec![carry, carry]).unwrap()[0];
            builder
                .build::<Vec<Scalar>, Vec<Scalar>>(vec![squared], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        (WhileOperation::new(), vec![condition, body])
    }

    #[test]
    fn test_jvp_pipeline_unrolls_eager_unbounded_while() {
        // The eager unroll-then-fuse pre-pass unrolls the unbounded `while x < 100 { x = x * x }` at the concrete
        // primal, so forward mode through it now succeeds on the capture-free path and must reproduce the
        // established eager `jvp`, whose eager `while` rule differentiates the same loop directly at the concrete
        // primal. From `x = 1.5` the loop runs four squarings.
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (reference_primals, reference_tangents) = domain
            .jvp(
                |inputs: Vec<ScalarJvpTracer>| {
                    let (while_operation, while_regions) = scalar_squaring_while();
                    inputs[0].context().bind(while_operation, while_regions, &[inputs[0].clone()])
                },
                vec![Scalar::from(1.5)],
                vec![Scalar::from(1.0)],
            )
            .unwrap();

        let (primal_outputs, tangent_outputs) = fused_pipeline_jvp(
            &domain,
            |inputs| {
                let (while_operation, while_regions) = scalar_squaring_while();
                inputs[0].context().bind(while_operation, while_regions, &[inputs[0].clone()]).unwrap()
            },
            vec![Scalar::from(1.5)],
            vec![Scalar::from(1.0)],
        )
        .unwrap();
        assert_close(&primal_outputs, &reference_primals, "eager unbounded while jvp pipeline primal");
        assert_close(&tangent_outputs, &reference_tangents, "eager unbounded while jvp pipeline tangent");
    }

    /// Builds a nested eager `while` loop: the outer loop `while s < 5000 { s = s + inner(s) }` runs while its carry is
    /// below the threshold, and its body stages an inner unbounded `while t < 100 { t = t + t }` started from the outer
    /// carry. Both loops are unbounded, so this exercises recursive nested-`while` unrolling: the inner loop is just
    /// another instruction encountered while the outer body is rewritten, so it is unrolled by the same pre-pass.
    fn scalar_nested_while() -> (WhileOperation, Vec<Program<Scalar, ScalarOperation<Scalar>, Vec<Scalar>, Vec<Scalar>>>)
    {
        use crate::operations::compare::{CompareOperation, ComparisonDirection};
        use crate::operations::math::AddOperation;

        let inner = {
            let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
            let carry = builder.add_input(DataType::F64);
            let threshold = builder.add_constant(Scalar::from(100.0));
            let predicate = builder
                .add_instruction(
                    CompareOperation::new(ComparisonDirection::LessThan),
                    Vec::new(),
                    vec![carry, threshold],
                )
                .unwrap()[0];
            builder
                .build::<Vec<Scalar>, Vec<Scalar>>(vec![predicate], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let inner_body = {
            let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
            let carry = builder.add_input(DataType::F64);
            let doubled = builder.add_instruction(AddOperation, Vec::new(), vec![carry, carry]).unwrap()[0];
            builder
                .build::<Vec<Scalar>, Vec<Scalar>>(vec![doubled], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let inner_while = WhileOperation::new();

        let outer = {
            let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
            let carry = builder.add_input(DataType::F64);
            let threshold = builder.add_constant(Scalar::from(5000.0));
            let predicate = builder
                .add_instruction(
                    CompareOperation::new(ComparisonDirection::LessThan),
                    Vec::new(),
                    vec![carry, threshold],
                )
                .unwrap()[0];
            builder
                .build::<Vec<Scalar>, Vec<Scalar>>(vec![predicate], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let outer_body = {
            let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
            let inner_condition_region = builder.import_region(inner.entry_region_ref());
            let inner_body_region = builder.import_region(inner_body.entry_region_ref());
            let carry = builder.add_input(DataType::F64);
            let inner_output = builder
                .add_instruction(inner_while, vec![inner_condition_region, inner_body_region], vec![carry])
                .unwrap()[0];
            let next = builder.add_instruction(AddOperation, Vec::new(), vec![carry, inner_output]).unwrap()[0];
            builder.build::<Vec<Scalar>, Vec<Scalar>>(vec![next], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        (WhileOperation::new(), vec![outer, outer_body])
    }

    #[test]
    fn test_unroll_concretizable_whiles_unrolls_nested_while() {
        // Nested unbounded `while`s unroll recursively: the inner loop is unrolled while the outer body is rewritten,
        // so the rewritten program is control-flow-free and computes the same concrete value as the original. This is
        // the capability that lets the path differentiate nested loops the legacy eager `while` JVP rule cannot
        // (it linearizes the body symbolically, which has no staged form for a nested unbounded loop).
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let input_values = vec![Scalar::from(1.5)];
        let (_, program) = NestedTracingContext::trace(
            domain.clone(),
            |inputs| {
                let (while_operation, while_regions) = scalar_nested_while();
                inputs[0].context().bind(while_operation, while_regions, &[inputs[0].clone()])
            },
            vec![DataType::F64],
        )
        .unwrap();
        let program = program.into_simplified().unwrap();

        // The original nested-`while` program interprets eagerly to a concrete value.
        let expected = program.interpret(input_values.clone()).unwrap();

        // After unrolling, no `while` instruction remains and the straight-line program reproduces the same value.
        let unrolled = unroll_concretizable_whiles(&domain, program, input_values.clone()).unwrap();
        assert!(
            unrolled
                .instructions()
                .iter()
                .all(|instruction| <&WhileOperation>::try_from(instruction.operation()).is_err()),
            "the unrolled program still contains a `while` instruction",
        );
        let actual = unrolled.interpret(input_values).unwrap();
        assert_close(&actual, &expected, "nested while unroll concrete value");
    }

    #[test]
    fn test_jvp_runs_eager_nested_unbounded_while() {
        // Forward mode through *nested* data-dependent loops under an eager receiver: the `while` JVP rule runs the
        // outer loop at the concrete duals and unrolls the inner loop at each iteration's concrete carries, so
        // neither loop needs an iteration bound. The primal must match the traced program's own eager
        // interpretation, and because every operation on the branch taken at `x = 1.5` (adds and doublings) is
        // linear in the carry, the pushforward of a unit tangent is exactly `f(1.5) / 1.5`.
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let input_values = vec![Scalar::from(1.5)];
        let (_, program) = NestedTracingContext::trace(
            domain.clone(),
            |inputs| {
                let (while_operation, while_regions) = scalar_nested_while();
                inputs[0].context().bind(while_operation, while_regions, &[inputs[0].clone()])
            },
            vec![DataType::F64],
        )
        .unwrap();
        let expected = program.into_simplified().unwrap().interpret(input_values).unwrap();

        let (primal, tangent): (Vec<Scalar>, Vec<Scalar>) = domain
            .jvp(
                |inputs: Vec<_>| {
                    let (while_operation, while_regions) = scalar_nested_while();
                    let mut outputs = inputs[0].context().bind(while_operation, while_regions, &[inputs[0].clone()])?;
                    Ok(vec![outputs.remove(0)])
                },
                vec![Scalar::from(1.5)],
                vec![Scalar::from(1.0)],
            )
            .unwrap();
        assert_close(&primal, &expected, "eager nested unbounded while jvp primal");
        let Scalar::F64(value) = expected[0] else {
            panic!("nested while primal should be an f64 scalar");
        };
        assert_close(&tangent, &[Scalar::from(value / 1.5)], "eager nested unbounded while jvp tangent");
    }

    #[test]
    fn test_vjp_pipeline_unrolls_eager_unbounded_while() {
        // The unrolled straight-line primal program produces a control-flow-free tangent program that transposes via
        // the existing partitioned transposition, so reverse mode through the unbounded `while x < 100 { x = x * x }`
        // now succeeds and must reproduce the established eager `vjp`. The direct-transpose pullback consumes the
        // residuals as ordinary inputs, so it is interpreted at `output_cotangents ++ residuals`.
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let context = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();

        let (_, reference_pullback) = domain
            .vjp(
                |inputs| {
                    let (while_operation, while_regions) = scalar_squaring_while();
                    inputs[0].context().bind(while_operation, while_regions, &[inputs[0].clone()])
                },
                vec![Scalar::from(1.5)],
            )
            .unwrap();
        let (reference_pullback, reference_residuals) = reference_pullback.into_parts();
        let mut reference_inputs = vec![Scalar::from(1.0)];
        reference_inputs.extend(reference_residuals);
        let reference_cotangents = reference_pullback.interpret_in_context(&context, reference_inputs).unwrap();

        let (_, pullback, residuals) = vjp_direct(
            &domain,
            |inputs| {
                let (while_operation, while_regions) = scalar_squaring_while();
                inputs[0].context().bind(while_operation, while_regions, &[inputs[0].clone()])
            },
            vec![Scalar::from(1.5)],
        )
        .unwrap();
        let mut pullback_inputs = vec![Scalar::from(1.0)];
        pullback_inputs.extend(residuals);
        let input_cotangents = pullback.interpret_in_context(&context, pullback_inputs).unwrap();
        assert_close(&input_cotangents, &reference_cotangents, "eager unbounded while vjp pipeline");
    }

    #[test]
    fn test_forward_equivalent_to_jvp() {
        // f(x) = x * sin(x): exercises mul (operand-factor terms) feeding sin (fresh-coefficient chain rule).
        assert_forward_equivalent(
            |inputs| Ok(vec![inputs[0].clone() * inputs[0].sin()?]),
            |inputs| Ok(vec![inputs[0].clone() * inputs[0].sin()?]),
            vec![Scalar::from(0.7)],
            vec![Scalar::from(1.0)],
        );

        // f(x) = x * x + 3 * x: exercises mul against a constant tangent and add.
        assert_forward_equivalent(
            |inputs| {
                let three = inputs[0].context().lift(Scalar::from(3.0))?;
                Ok(vec![inputs[0].clone() * inputs[0].clone() + three * inputs[0].clone()])
            },
            |inputs| {
                let three = inputs[0].context().constant(Scalar::from(3.0));
                Ok(vec![inputs[0].clone() * inputs[0].clone() + three * inputs[0].clone()])
            },
            vec![Scalar::from(2.0)],
            vec![Scalar::from(1.0)],
        );

        // f(x) = sin(x) * cos(x): exercises both fresh-coefficient chain rules feeding a product.
        assert_forward_equivalent(
            |inputs| Ok(vec![inputs[0].sin()? * inputs[0].cos()?]),
            |inputs| Ok(vec![inputs[0].sin()? * inputs[0].cos()?]),
            vec![Scalar::from(1.2)],
            vec![Scalar::from(1.0)],
        );

        // f(a, b) = a * b + sin(a): two inputs, mixing a bilinear product with a unary chain rule.
        assert_forward_equivalent(
            |inputs| Ok(vec![inputs[0].clone() * inputs[1].clone() + inputs[0].sin()?]),
            |inputs| Ok(vec![inputs[0].clone() * inputs[1].clone() + inputs[0].sin()?]),
            vec![Scalar::from(0.5), Scalar::from(1.3)],
            vec![Scalar::from(1.0), Scalar::from(1.0)],
        );

        // f(a, b) = a * b + sin(a), tangent only along b, to exercise the partial all-zero tangent path.
        assert_forward_equivalent(
            |inputs| Ok(vec![inputs[0].clone() * inputs[1].clone() + inputs[0].sin()?]),
            |inputs| Ok(vec![inputs[0].clone() * inputs[1].clone() + inputs[0].sin()?]),
            vec![Scalar::from(0.5), Scalar::from(1.3)],
            vec![Scalar::from(0.0), Scalar::from(2.0)],
        );

        // f(a, b) = a / b: exercises the quotient rule (fresh reciprocal and `-(a/b²)` coefficients) with both terms
        // live.
        assert_forward_equivalent(
            |inputs| Ok(vec![inputs[0].clone() / inputs[1].clone()]),
            |inputs| Ok(vec![inputs[0].clone() / inputs[1].clone()]),
            vec![Scalar::from(3.0), Scalar::from(2.0)],
            vec![Scalar::from(1.0), Scalar::from(1.0)],
        );

        // f(a, b) = sin(a) / b, tangent only along a, to exercise the quotient rule's dropped right term.
        assert_forward_equivalent(
            |inputs| Ok(vec![inputs[0].sin()? / inputs[1].clone()]),
            |inputs| Ok(vec![inputs[0].sin()? / inputs[1].clone()]),
            vec![Scalar::from(0.9), Scalar::from(1.4)],
            vec![Scalar::from(1.0), Scalar::from(0.0)],
        );
    }

    #[test]
    fn test_reverse_equivalent_to_vjp() {
        // f(x) = x * sin(x)
        assert_reverse_equivalent(
            |inputs| Ok(vec![inputs[0].clone() * inputs[0].sin()?]),
            |inputs| Ok(vec![inputs[0].clone() * inputs[0].sin()?]),
            vec![Scalar::from(0.7)],
            vec![Scalar::from(1.0)],
        );

        // f(x) = x * x + 3 * x
        assert_reverse_equivalent(
            |inputs| {
                let three = inputs[0].context().lift(Scalar::from(3.0))?;
                Ok(vec![inputs[0].clone() * inputs[0].clone() + three * inputs[0].clone()])
            },
            |inputs| {
                let three = inputs[0].context().constant(Scalar::from(3.0));
                Ok(vec![inputs[0].clone() * inputs[0].clone() + three * inputs[0].clone()])
            },
            vec![Scalar::from(2.0)],
            vec![Scalar::from(1.0)],
        );

        // f(x) = sin(x) * cos(x), with a non-unit output cotangent.
        assert_reverse_equivalent(
            |inputs| Ok(vec![inputs[0].sin()? * inputs[0].cos()?]),
            |inputs| Ok(vec![inputs[0].sin()? * inputs[0].cos()?]),
            vec![Scalar::from(1.2)],
            vec![Scalar::from(2.5)],
        );

        // f(a, b) = a * b + sin(a)
        assert_reverse_equivalent(
            |inputs| Ok(vec![inputs[0].clone() * inputs[1].clone() + inputs[0].sin()?]),
            |inputs| Ok(vec![inputs[0].clone() * inputs[1].clone() + inputs[0].sin()?]),
            vec![Scalar::from(0.5), Scalar::from(1.3)],
            vec![Scalar::from(1.0)],
        );

        // f(a, b) = a / b: transposes the quotient-rule tangent map (two captured constant factors).
        assert_reverse_equivalent(
            |inputs| Ok(vec![inputs[0].clone() / inputs[1].clone()]),
            |inputs| Ok(vec![inputs[0].clone() / inputs[1].clone()]),
            vec![Scalar::from(3.0), Scalar::from(2.0)],
            vec![Scalar::from(1.5)],
        );
    }

    #[test]
    fn test_linearize_restores_pruned_tangent_input() {
        // f(x, y) = sin(x) + stop_gradient(y) uses both inputs in the primal (so the primal sub-program keeps both),
        // but `stop_gradient` blocks `y`'s tangent: `dy` reaches no tangent output, so partial-evaluation liveness
        // pruning drops it from the unknown sub-program. The canonical tangent arity must be restored (a fresh
        // zero-typed `dy` slot reinserted) so the tangent program still presents `[dx, dy, residuals...]`. This
        // exercises the input-restoration branch of `linearize` that straight-line all-differentiable programs do not.
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (_, primal_program) = NestedTracingContext::trace(
            domain.clone(),
            |inputs| Ok(vec![inputs[0].sin()? + inputs[1].stop_gradient()]),
            vec![DataType::F64, DataType::F64],
        )
        .unwrap();
        let primal_program = primal_program.into_simplified().unwrap();

        let linearization = primal_program.linearize().unwrap();

        // Structural asserts: the function uses both inputs primally (so `sin(x)` threads at least one residual) and
        // the primal sub-program produces the single primal output followed by those residuals. The pruned `dy` slot
        // is restored, so the tangent program presents both tangent inputs (`dx`, the restored `dy`) ahead of its
        // residuals — `2 + residual_count`.
        assert!(
            linearization.residual_count() > 0,
            "the chosen function must produce a non-empty residual environment"
        );
        assert_eq!(
            linearization.primal().output_ids().len(),
            1 + linearization.residual_count(),
            "primal program output count",
        );
        assert_eq!(
            linearization.tangent().input_ids().len(),
            2 + linearization.residual_count(),
            "tangent program input count (restored dy slot)",
        );

        // Behavioral assert (absolute, hand-computed): `d(sin x + stop_gradient(y)) = cos(x) * dx`, with `dy` ignored.
        // Recover the residuals by interpreting the primal sub-program at the primals, then feed `(dx, dy, residuals)`
        // through the tangent sub-program. The restored `dy` slot must be reinserted for the tangent program to accept
        // both tangent inputs even though `stop_gradient` blocks `y`'s tangent and `dy` feeds nothing.
        let context = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let primal_outputs = linearization
            .primal()
            .interpret_in_context(&context, vec![Scalar::from(0.7), Scalar::from(1.3)])
            .unwrap();
        let residuals = primal_outputs[primal_outputs.len() - linearization.residual_count()..].to_vec();

        let mut tangent_inputs = vec![Scalar::from(1.0), Scalar::from(1.0)];
        tangent_inputs.extend(residuals);
        let tangent_outputs = linearization.tangent().interpret_in_context(&context, tangent_inputs).unwrap();
        assert_close(&tangent_outputs, &[Scalar::from(0.7_f64.cos())], "restored-input tangent program");
    }

    #[test]
    fn test_rejects_nested_program_operations() {
        use crate::contexts::StagingContext;
        use crate::operations::compare::{CompareOperation, ComparisonDirection};
        use crate::operations::control_flow::WhileOperation;
        use crate::operations::math::AddOperation;
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;

        // A `while` loop is a nested-program operation: its tangent is a linearized loop, not primal-enum operand
        // arithmetic, so the front end must reject it rather than mis-evaluate it.
        let condition = {
            let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
            let carry = builder.add_input(DataType::F64);
            let eight = builder.add_constant(Scalar::from(8.0));
            let predicate = builder
                .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), Vec::new(), vec![carry, eight])
                .unwrap()[0];
            builder
                .build::<Vec<Scalar>, Vec<Scalar>>(vec![predicate], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
            let carry = builder.add_input(DataType::F64);
            let doubled = builder.add_instruction(AddOperation, Vec::new(), vec![carry, carry]).unwrap()[0];
            builder
                .build::<Vec<Scalar>, Vec<Scalar>>(vec![doubled], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };

        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (_, primal_program) = NestedTracingContext::trace(
            domain.clone(),
            move |inputs| {
                let operation = WhileOperation::new();
                let mut outputs =
                    inputs[0].context().stage_operation(operation, vec![condition, body], &[inputs[0].clone()])?;
                Ok(vec![outputs.remove(0)])
            },
            vec![DataType::F64],
        )
        .unwrap();
        let primal_program = primal_program.into_simplified().unwrap();

        let error = primal_program.jvp().unwrap_err();
        assert!(
            matches!(
                error,
                crate::differentiation::DifferentiationError::Program(ProgramError::UnsupportedOperation { .. }),
            ),
            "expected the front end to reject a `while` loop, but got {error:?}",
        );
    }

    #[test]
    fn test_program_covers_compare_select_and_stop_gradient_directly() {
        // `compare`/`select` introduce a Boolean codomain and `stop_gradient` injects a structural zero into a tangent
        // computed from a non-input expression; neither round-trips cleanly through the `f64` partial-evaluation
        // harness (a Boolean condition would become a typed residual, and the discarded tangent subtree desynchronizes
        // the split's residual accounting). Their rules are nonetheless correct, which these checks prove by
        // interpreting the whole program directly at `(primal, tangent)`. The program is simplified first to
        // prune the structurally dead zero tangents the replay synthesizes for the Boolean codomain and the
        // severed branch — including a `zero` of Boolean type that the `f64` interpreter cannot evaluate.
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let context = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();

        // f(x) = select(x > 1, x * x, x + x): the comparison contributes a zero tangent and `select` routes the chosen
        // branch's tangent. For x > 1 the derivative is `2x`; otherwise it is `2`.
        let (_, select_program) = NestedTracingContext::trace(
            domain.clone(),
            |inputs| {
                let one = inputs[0].context().constant(Scalar::from(1.0));
                let mask = inputs[0].clone().greater_than(&one)?;
                let on_true = inputs[0].clone() * inputs[0].clone();
                let on_false = inputs[0].clone() + inputs[0].clone();
                Ok(vec![Select::select(&mask, &on_true, &on_false)?])
            },
            vec![DataType::F64],
        )
        .unwrap();
        let select_program = select_program.into_simplified().unwrap();
        let select_jvp = select_program.jvp().unwrap().simplified().unwrap();

        // Selected branch (x = 3 > 1): primal x*x = 9, tangent 2x*dx = 6.
        let outputs = select_jvp.interpret_in_context(&context, vec![Scalar::from(3.0), Scalar::from(1.0)]).unwrap();
        assert_close(&outputs, &[Scalar::from(9.0), Scalar::from(6.0)], "select true-branch");

        // Other branch (x = 0.5 <= 1): primal x+x = 1, tangent 2*dx = 2.
        let outputs = select_jvp.interpret_in_context(&context, vec![Scalar::from(0.5), Scalar::from(1.0)]).unwrap();
        assert_close(&outputs, &[Scalar::from(1.0), Scalar::from(2.0)], "select false-branch");

        // f(x) = stop_gradient(x * x) + x: the stopped term contributes no tangent, so the derivative is 1. The JVP program
        // `add` drops the severed zero tangent, leaving the trailing `x`'s tangent as the only contribution.
        let (_, stop_gradient_program) = NestedTracingContext::trace(
            domain.clone(),
            |inputs| Ok(vec![(inputs[0].clone() * inputs[0].clone()).stop_gradient() + inputs[0].clone()]),
            vec![DataType::F64],
        )
        .unwrap();
        let stop_gradient_program = stop_gradient_program.into_simplified().unwrap();
        let stop_gradient_jvp = stop_gradient_program.jvp().unwrap().simplified().unwrap();
        // primal = stop_gradient(2.25) + 1.5 = 3.75; tangent = 0 + dx = 1.
        let outputs = stop_gradient_jvp
            .interpret_in_context(&context, vec![Scalar::from(1.5), Scalar::from(1.0)])
            .unwrap();
        assert_close(&outputs, &[Scalar::from(3.75), Scalar::from(1.0)], "stop_gradient");
    }

    #[test]
    fn test_jvp_zero_tangent_flows_the_all_zero_fast_path() {
        use crate::operations::math::{Mul, Sin};

        // A zero input tangent flows through the all-zero fast path of `DifferentiationContext::bind` (the rule is skipped and
        // the primal operation binds directly): the derivative is zero and the value matches the primal evaluation.
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (value, tangent): (Scalar, Scalar) =
            domain.jvp(|x: ScalarJvpTracer| x.mul(&x.sin()?), Scalar::from(0.7), Scalar::from(0.0)).unwrap();
        assert_close(&[value], &[Scalar::from(0.7 * 0.7_f64.sin())], "zero-tangent value");
        assert_close(&[tangent], &[Scalar::from(0.0)], "zero-tangent tangent");
    }

    #[test]
    fn test_jvp_branches_on_concrete_primal() {
        use crate::operations::BooleanLike;
        use crate::operations::math::{Mul, Neg, Sin};

        // Over an eager receiver `jvp` runs the closure directly on concrete duals, so ordinary Rust control flow can
        // branch on the primal — impossible under a staging receiver, whose duals carry tracers.
        // `f(x) = if x != 0 { x * sin(x) } else { -x }`.
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let branching = |x: ScalarJvpTracer| -> Result<ScalarJvpTracer, ProgramError> {
            if x.primal().boolean()? { x.mul(&x.sin()?) } else { x.neg() }
        };
        let (primal, tangent): (Scalar, Scalar) = domain.jvp(branching, Scalar::from(0.7), Scalar::from(1.0)).unwrap();

        // `0.7` decodes as `true`, so the `x * sin(x)` branch runs; its primal and tangent match `jvp` of that
        // straight-line function.
        let (reference_primal, reference_tangent): (Scalar, Scalar) = domain
            .jvp(|x: ScalarJvpTracer| Ok(x.clone() * x.sin()?), Scalar::from(0.7), Scalar::from(1.0))
            .unwrap();
        assert_eq!(primal, reference_primal);
        assert_eq!(tangent, reference_tangent);
    }

    #[test]
    fn test_program_keeps_structural_zero_tangents_symbolic() {
        // f(x) = stop_gradient(x * x) + x: the severed tangent is a structural zero that must stay symbolic — the
        // `add` rule drops the zero term instead of staging `add(zero, dx)`, so the *unsimplified* fused program
        // contains no `zero` instruction at all and its tangent output is the tangent input directly.
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (_, primal_program) = NestedTracingContext::trace(
            domain.clone(),
            |inputs| Ok(vec![(inputs[0].clone() * inputs[0].clone()).stop_gradient() + inputs[0].clone()]),
            vec![DataType::F64],
        )
        .unwrap();
        let primal_program = primal_program.into_simplified().unwrap();
        let jvp_program = primal_program.jvp().unwrap();
        use crate::programs::operations::Operation;
        assert!(
            !jvp_program.instructions().iter().any(|instruction| instruction.operation().name() == "zero"),
            "expected no staged zero instructions in the fused jvp program, but got:\n{jvp_program}",
        );
    }

    #[test]
    fn test_program_covers_constant_scaling_and_nullary_constants_directly() {
        use crate::operations::constants::{OneLikeOperation, OneOperation, ZeroLikeOperation};
        use crate::operations::math::{AddOperation, MulOperation};
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;

        // The exemplar-derived `one`/`one_like`/`zero_like` constants do not arise in a traced primal program, so they
        // are staged manually here alongside a multiply-by-a-closed-constant. Interpreting the program directly
        // verifies their rules: multiplying by a constant is linear in its input, and the constants contribute zero
        // tangents.
        //
        // f(x) = 2*x + one_like(x) + one() + zero_like(x): primal = 2x + 1 + 1 + 0 = 2x + 2; tangent = 2*dx (only the
        // product term carries a tangent).
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::F64);
        let factor = builder.add_constant(Scalar::from(2.0));
        let scaled = builder.add_instruction(MulOperation, Vec::new(), vec![input, factor]).unwrap()[0];
        let one_like = builder.add_instruction(OneLikeOperation, Vec::new(), vec![input]).unwrap()[0];
        let one = builder
            .add_instruction(OneOperation::<DataType>::new(DataType::F64), Vec::new(), Vec::new())
            .unwrap()[0];
        let zero_like = builder.add_instruction(ZeroLikeOperation, Vec::new(), vec![input]).unwrap()[0];
        let first = builder.add_instruction(AddOperation, Vec::new(), vec![scaled, one_like]).unwrap()[0];
        let second = builder.add_instruction(AddOperation, Vec::new(), vec![first, one]).unwrap()[0];
        let total = builder.add_instruction(AddOperation, Vec::new(), vec![second, zero_like]).unwrap()[0];
        let primal_program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![total], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let jvp_program = primal_program.jvp().unwrap();
        let context = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let outputs = jvp_program.interpret_in_context(&context, vec![Scalar::from(4.0), Scalar::from(1.0)]).unwrap();
        // primal = 2*4 + 2 = 10; tangent = 2*1 = 2.
        assert_close(&outputs, &[Scalar::from(10.0), Scalar::from(2.0)], "constant scaling and nullary constants");
    }

    #[test]
    fn test_jvp_computes_analytic_forward_derivatives() {
        // The single `jvp` entry point runs the closure directly on duals; each block below asserts the hand-computed
        // primal and directional derivative for one of the function shapes the pipeline-equivalence harness used to
        // exercise.
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();

        // f(x) = x * sin(x): mul feeding sin; d/dx = sin(x) + x * cos(x).
        let (primals, tangents) = domain
            .jvp(
                |inputs: Vec<ScalarJvpTracer>| Ok(vec![inputs[0].clone() * inputs[0].sin()?]),
                vec![Scalar::from(0.7)],
                vec![Scalar::from(1.0)],
            )
            .unwrap();
        assert_close(&primals, &[Scalar::from(0.7 * 0.7_f64.sin())], "x sin(x) primal");
        assert_close(&tangents, &[Scalar::from(0.7_f64.sin() + 0.7 * 0.7_f64.cos())], "x sin(x) tangent");

        // f(x) = x * x + 3 * x at x = 2: primal 10 and derivative 2x + 3 = 7.
        let (primals, tangents) = domain
            .jvp(
                |inputs: Vec<ScalarJvpTracer>| {
                    let three = inputs[0].context().lift(Scalar::from(3.0))?;
                    Ok(vec![inputs[0].clone() * inputs[0].clone() + three * inputs[0].clone()])
                },
                vec![Scalar::from(2.0)],
                vec![Scalar::from(1.0)],
            )
            .unwrap();
        assert_close(&primals, &[Scalar::from(10.0)], "x^2 + 3x primal");
        assert_close(&tangents, &[Scalar::from(7.0)], "x^2 + 3x tangent");

        // f(a, b) = a * b + sin(a) at (0.5, 1.3) with tangents (1, 1): the directional derivative is
        // b * da + a * db + cos(a) * da = 1.3 + 0.5 + cos(0.5).
        let (primals, tangents) = domain
            .jvp(
                |inputs: Vec<ScalarJvpTracer>| Ok(vec![inputs[0].clone() * inputs[1].clone() + inputs[0].sin()?]),
                vec![Scalar::from(0.5), Scalar::from(1.3)],
                vec![Scalar::from(1.0), Scalar::from(1.0)],
            )
            .unwrap();
        assert_close(&primals, &[Scalar::from(0.5 * 1.3 + 0.5_f64.sin())], "a b + sin(a) primal");
        assert_close(&tangents, &[Scalar::from(1.3 + 0.5 + 0.5_f64.cos())], "a b + sin(a) tangent");

        // Same function with the tangent only along `b`, exercising the partial all-zero tangent path: the `sin`
        // term drops out and the derivative is a * db = 0.5 * 2 = 1.
        let (_, tangents) = domain
            .jvp(
                |inputs: Vec<ScalarJvpTracer>| Ok(vec![inputs[0].clone() * inputs[1].clone() + inputs[0].sin()?]),
                vec![Scalar::from(0.5), Scalar::from(1.3)],
                vec![Scalar::from(0.0), Scalar::from(2.0)],
            )
            .unwrap();
        assert_close(&tangents, &[Scalar::from(1.0)], "a b + sin(a) partial-zero tangent");
    }

    #[test]
    fn test_jvp_stages_into_an_enclosing_trace() {
        use crate::tracing::DomainTracingContext;

        // jvp-under-tracing duality: running `jvp` against an enclosing `TracingContext` (whose values are
        // tracers) must splice both the primal replay and the tangent replay into the enclosing trace through the same
        // `bind` path the eager domain uses to compute. We trace `f(a, b) = a * b + sin(a)` under an outer trace and
        // assert the staged program, interpreted eagerly, equals the eager `EagerContext<Scalar, ScalarOperation<Scalar>>` jvp at a sample point.
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let outer_context = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let outer_builder = outer_context.builder().clone();

        // The outer trace's inputs are the two primals followed by the two tangents, so that interpreting the staged
        // program at `(a, b, da, db)` mirrors the eager jvp inputs.
        let primal_a = outer_context.input(DataType::F64);
        let primal_b = outer_context.input(DataType::F64);
        let tangent_a = outer_context.input(DataType::F64);
        let tangent_b = outer_context.input(DataType::F64);

        let (primal_outputs, tangent_outputs) = outer_context
            .jvp(
                |inputs: Vec<
                    DifferentiationTracer<DomainTracingContext<EagerContext<Scalar, ScalarOperation<Scalar>>>>,
                >| { Ok(vec![inputs[0].clone() * inputs[1].clone() + inputs[0].sin()?]) },
                vec![primal_a, primal_b],
                vec![tangent_a, tangent_b],
            )
            .unwrap();
        assert_eq!(primal_outputs.len(), 1);
        assert_eq!(tangent_outputs.len(), 1);

        // Build the staged outer program over its four inputs, producing the staged primal output followed by the
        // staged tangent output, then interpret it eagerly.
        let output_atoms = vec![primal_outputs[0].atom_id().unwrap(), tangent_outputs[0].atom_id().unwrap()];
        let staged = outer_builder
            .borrow()
            .clone()
            .build::<Vec<Scalar>, Vec<Scalar>>(output_atoms, vec![Placeholder; 4], vec![Placeholder; 2])
            .unwrap();

        let context = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let staged_outputs = staged
            .interpret_in_context(
                &context,
                vec![Scalar::from(0.5), Scalar::from(1.3), Scalar::from(1.0), Scalar::from(1.0)],
            )
            .unwrap();

        let (reference_primal, reference_tangent) = domain
            .jvp(
                |inputs: Vec<ScalarJvpTracer>| Ok(vec![inputs[0].clone() * inputs[1].clone() + inputs[0].sin()?]),
                vec![Scalar::from(0.5), Scalar::from(1.3)],
                vec![Scalar::from(1.0), Scalar::from(1.0)],
            )
            .unwrap();
        assert_close(&staged_outputs[..1], &reference_primal, "jvp-under-tracing primal");
        assert_close(&staged_outputs[1..], &reference_tangent, "jvp-under-tracing tangent");
    }

    #[test]
    fn test_vjp_stages_into_an_enclosing_trace() {
        use crate::tracing::DomainTracingContext;

        // Reverse-mode-under-tracing duality: running `vjp` against an enclosing
        // `TracingContext` (whose values are tracers) must produce a tracer-valued pullback that splices into the
        // enclosing trace (the pullback's constants are lifted into the enclosing trace's tracer value space). We
        // trace `f(x) = x * sin(x)` (a nonlinear scalar function) under an outer trace, interpret the tracer-valued
        // pullback at an outer cotangent tracer to stage the backward pass into that trace, then interpret the staged
        // program eagerly and assert the input cotangent equals the established `vjp` pullback at the same point.
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let outer_context = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let outer_builder = outer_context.builder().clone();

        // The outer trace's input is the primal `x` followed by the output cotangent, so interpreting the staged
        // program at `(x, cotangent)` mirrors seeding the established `vjp` pullback with the same cotangent.
        let primal_x = outer_context.input(DataType::F64);
        let cotangent = outer_context.input(DataType::F64);

        let (primal_outputs, pullback) = outer_context
            .vjp(
                |inputs: Vec<
                    LinearizationTracer<DomainTracingContext<EagerContext<Scalar, ScalarOperation<Scalar>>>>,
                >| { Ok(vec![inputs[0].clone() * inputs[0].sin()?]) },
                vec![primal_x],
            )
            .unwrap();
        let (pullback, residuals) = pullback.into_parts();
        assert_eq!(primal_outputs.len(), 1);

        // The pullback is genuinely tracer-valued (its value type is the enclosing trace's `Tracer`), so interpreting
        // it through the enclosing `TracingContext` splices the backward pass into the outer trace. The direct-
        // transpose pullback consumes the residuals as ordinary inputs, so it is interpreted at `[cotangent, residuals]`
        // — and under tracing those residuals are themselves outer tracers recovered from the primal replay.
        let mut pullback_inputs = vec![cotangent];
        pullback_inputs.extend(residuals);
        let input_cotangents = pullback.interpret_in_context(&outer_context, pullback_inputs).unwrap();
        assert_eq!(input_cotangents.len(), 1);

        // Build the staged outer program over its two inputs `(x, cotangent)`, producing the staged input cotangent,
        // then interpret it eagerly at a sample point.
        let output_atoms = vec![input_cotangents[0].atom_id().unwrap()];
        let staged = outer_builder
            .borrow()
            .clone()
            .build::<Vec<Scalar>, Vec<Scalar>>(output_atoms, vec![Placeholder; 2], vec![Placeholder; 1])
            .unwrap();

        let context = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let staged_cotangents =
            staged.interpret_in_context(&context, vec![Scalar::from(0.7), Scalar::from(1.0)]).unwrap();

        // Reference: the established `vjp` pullback at `x = 0.7`, seeded with the same cotangent `1.0`. For
        // `f(x) = x * sin(x)` the gradient is `sin(x) + x * cos(x)`.
        let (_, reference_pullback) = domain
            .vjp(|inputs| Ok(vec![inputs[0].clone() * inputs[0].sin()?]), vec![Scalar::from(0.7)])
            .unwrap();
        let (reference_pullback, reference_residuals) = reference_pullback.into_parts();
        let mut reference_inputs = vec![Scalar::from(1.0)];
        reference_inputs.extend(reference_residuals);
        let reference_cotangents = reference_pullback.interpret_in_context(&context, reference_inputs).unwrap();
        assert_close(&staged_cotangents, &reference_cotangents, "reverse-under-tracing cotangent");
        assert_close(
            &staged_cotangents,
            &[Scalar::from(0.7f64.sin() + 0.7 * 0.7f64.cos())],
            "reverse-under-tracing gradient",
        );
    }

    /// Builds the scalar primal program `x -> sin(x)`, the primal half of the deliberately wrong custom-JVP oracle.
    fn scalar_custom_jvp_sin_primal() -> Program<Scalar, ScalarOperation<Scalar>, Vec<Scalar>, Vec<Scalar>> {
        use crate::operations::math::SinOperation;
        use crate::programs::ProgramBuilder;

        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(SinOperation, Vec::new(), vec![input]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds the deliberately wrong scalar custom-JVP rule `(x, dx) -> (sin(x), 2 * cos(x) * dx)`, detectably
    /// different from the true `cos(x) * dx`, so a passing equivalence proves the spliced rule governs both forward and
    /// reverse mode.
    fn scalar_custom_jvp_sin_doubled_rule() -> Program<Scalar, ScalarOperation<Scalar>, Vec<Scalar>, Vec<Scalar>> {
        use crate::operations::math::{CosOperation, MulOperation, SinOperation};
        use crate::programs::ProgramBuilder;

        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let x = builder.add_input(DataType::F64);
        let dx = builder.add_input(DataType::F64);
        let y = builder.add_instruction(SinOperation, Vec::new(), vec![x]).unwrap()[0];
        let cosine = builder.add_instruction(CosOperation, Vec::new(), vec![x]).unwrap()[0];
        let two = builder.add_constant(Scalar::from(2.0));
        let scaled = builder.add_instruction(MulOperation, Vec::new(), vec![cosine, two]).unwrap()[0];
        let tangent = builder.add_instruction(MulOperation, Vec::new(), vec![scaled, dx]).unwrap()[0];
        builder
            .build(vec![y, tangent], vec![Placeholder, Placeholder], vec![Placeholder, Placeholder])
            .unwrap()
    }

    /// Stages `custom_jvp(sin, doubled_rule)(x)` over the single closure input `[x]`, the shared body of the scalar
    /// custom-JVP equivalence closures. Generic over the value type so it serves both the staged trace pipelines
    /// (over [`ScalarTracer`]) and the dual-running entry points (over [`ScalarLinearizationTracer`]).
    fn scalar_custom_jvp_function<V>(inputs: Vec<V>) -> Result<Vec<V>, ProgramError>
    where
        V: Value<Type = DataType>,
        V::DispatchDomain: Context<Type = DataType, Constant = Scalar, Operation = ScalarOperation<Scalar>>,
    {
        use crate::tracing_v2::operations::custom_derivatives::CustomJvpOperation;

        let operation = CustomJvpOperation::new();
        inputs[0].dispatch_domain().bind(
            ScalarOperation::CustomJvp(operation),
            vec![scalar_custom_jvp_sin_primal(), scalar_custom_jvp_sin_doubled_rule()],
            &[inputs[0].clone()],
        )
    }

    #[test]
    fn test_forward_equivalent_to_jvp_for_custom_jvp() {
        use crate::tracing_v2::operations::custom_derivatives::CustomJvpOperation;

        // f(x) = custom_jvp(sin, doubled_rule)(x). The spliced JVP program maps `(x, dx) -> (sin(x), 2*cos(x)*dx)`, so
        // the tangent is the deliberately doubled `2*cos(x)*dx` rather than the primal body's `cos(x)*dx`.
        assert_forward_equivalent(
            |inputs: Vec<ScalarJvpTracer>| {
                let operation = CustomJvpOperation::new();
                inputs[0].context().bind(
                    ScalarOperation::CustomJvp(operation),
                    vec![scalar_custom_jvp_sin_primal(), scalar_custom_jvp_sin_doubled_rule()],
                    &[inputs[0].clone()],
                )
            },
            scalar_custom_jvp_function,
            vec![Scalar::from(0.7)],
            vec![Scalar::from(1.5)],
        );
    }

    #[test]
    fn test_reverse_equivalent_to_vjp_for_custom_jvp() {
        // Reverse mode transposes the spliced (straight-line) JVP program, so the doubled derivative carries over: the
        // pullback of a cotangent is `2*cos(x)*cotangent`.
        assert_reverse_equivalent(
            scalar_custom_jvp_function,
            scalar_custom_jvp_function,
            vec![Scalar::from(0.7)],
            vec![Scalar::from(2.5)],
        );
    }

    /// Derives the scalar [`RematerializeOperation`] for `rematerialize(x -> (x*x).sin())` at an `f64` input under the
    /// default `NothingSaveable` policy and stages it directly, mirroring how the array tests stage their
    /// rematerialize operation. The derivation runs once through a `TracingContext::trace`.
    fn scalar_rematerialize_function(inputs: Vec<ScalarTracer>) -> Result<Vec<ScalarTracer>, ProgramError> {
        use crate::contexts::StagingContext;
        use crate::operations::math::Sin;
        use crate::tracing::DomainTracer;
        use crate::tracing_v2::rematerialize;

        let function = rematerialize::<EagerContext<Scalar, ScalarOperation<Scalar>>, _, _, _>(
            |x: DomainTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>| Ok((x.clone() * x).sin()?),
        );
        let (_, program) =
            EagerContext::<Scalar, ScalarOperation<Scalar>>::trace(|x| function.call(x), DataType::F64).unwrap();
        let instruction = &program.instructions()[0];
        let ScalarOperation::Rematerialize(operation) = instruction.operation() else {
            panic!("rematerialize should stage a rematerialize call");
        };
        let operation_regions = instruction
            .regions()
            .iter()
            .map(|region| program.region_ref(*region).map(|region| region.to_program()))
            .collect::<Result<Vec<_>, _>>()?;
        inputs[0].context().stage_operation(
            ScalarOperation::Rematerialize(*operation),
            operation_regions,
            &[&inputs[0]],
        )
    }

    #[test]
    fn test_forward_equivalent_to_jvp_for_rematerialize() {
        // f(x) = rematerialize(sin(x*x)) under the default `NothingSaveable` policy. The rule splices the derived
        // forward program (recovering the region input as the residual tail) and the derived tangent program (which
        // recomputes the interior residuals from it), so the forward reproduces `jvp` of the un-rematerialized
        // body. At x = 2 the directional derivative is `cos(4) * 4 * dx`.
        assert_forward_equivalent(
            |inputs| Ok(vec![(inputs[0].clone() * inputs[0].clone()).sin()?]),
            scalar_rematerialize_function,
            vec![Scalar::from(2.0)],
            vec![Scalar::from(1.0)],
        );
    }

    #[test]
    fn test_reverse_equivalent_to_vjp_for_rematerialize() {
        // Reverse mode transposes the spliced recompute-and-pushforward tangent program, so the pullback matches
        // `vjp` of the un-rematerialized body: the pullback of a cotangent is `cos(x*x) * 2x * cotangent`.
        assert_reverse_equivalent(
            |inputs| Ok(vec![(inputs[0].clone() * inputs[0].clone()).sin()?]),
            scalar_rematerialize_function,
            vec![Scalar::from(2.0)],
            vec![Scalar::from(1.5)],
        );
    }

    /// Builds the scalar `custom_vjp(sin, forward, tripled_backward)` operation. The forward program is
    /// `x -> (sin(x), cos(x))` and the deliberately wrong backward program is
    /// `(residual, cotangent) -> 3 * residual * cotangent`, detectably different from the true `cos(x) * cotangent`, so
    /// a passing equivalence proves the opaque carrier actually replays `backward` rather than folding to zero or to
    /// the primal derivative. Shares its oracle shape with the `custom_derivatives` tests.
    fn scalar_custom_vjp_operation() -> Result<
        (
            crate::tracing_v2::operations::custom_derivatives::CustomVjpOperation,
            Vec<Program<Scalar, ScalarOperation<Scalar>, Vec<Scalar>, Vec<Scalar>>>,
        ),
        ProgramError,
    > {
        use crate::operations::math::{CosOperation, MulOperation, SinOperation};
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;
        use crate::tracing_v2::operations::custom_derivatives::CustomVjpOperation;

        let mut primal_builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let primal_input = primal_builder.add_input(DataType::F64);
        let primal_output = primal_builder.add_instruction(SinOperation, Vec::new(), vec![primal_input]).unwrap()[0];
        let primal = primal_builder.build(vec![primal_output], vec![Placeholder], vec![Placeholder]).unwrap();

        let mut forward_builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let x = forward_builder.add_input(DataType::F64);
        let y = forward_builder.add_instruction(SinOperation, Vec::new(), vec![x]).unwrap()[0];
        let residual = forward_builder.add_instruction(CosOperation, Vec::new(), vec![x]).unwrap()[0];
        let forward =
            forward_builder.build(vec![y, residual], vec![Placeholder], vec![Placeholder, Placeholder]).unwrap();

        let mut backward_builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let backward_residual = backward_builder.add_input(DataType::F64);
        let cotangent = backward_builder.add_input(DataType::F64);
        let three = backward_builder.add_constant(Scalar::from(3.0));
        let scaled =
            backward_builder.add_instruction(MulOperation, Vec::new(), vec![backward_residual, three]).unwrap()[0];
        let gradient = backward_builder.add_instruction(MulOperation, Vec::new(), vec![scaled, cotangent]).unwrap()[0];
        let backward =
            backward_builder.build(vec![gradient], vec![Placeholder, Placeholder], vec![Placeholder]).unwrap();

        Ok((CustomVjpOperation::new(), vec![primal, forward, backward]))
    }

    /// Stages the [`scalar_custom_vjp_operation`] call `custom_vjp(sin, forward, tripled_backward)(x)` over the
    /// single closure input `[x]`, the shared body of the scalar custom-VJP equivalence closures. Generic over the
    /// value type so it serves both the staged trace pipelines (over [`ScalarTracer`]) and the dual-running entry
    /// points (over [`ScalarLinearizationTracer`]).
    fn scalar_custom_vjp_function<V>(inputs: Vec<V>) -> Result<Vec<V>, ProgramError>
    where
        V: Value<Type = DataType>,
        V::DispatchDomain: Context<Type = DataType, Constant = Scalar, Operation = ScalarOperation<Scalar>>,
    {
        let (operation, operation_regions) = scalar_custom_vjp_operation()?;
        inputs[0]
            .dispatch_domain()
            .bind(ScalarOperation::CustomVjp(operation), operation_regions, &[inputs[0].clone()])
    }

    #[test]
    fn test_direct_transpose_equivalent_to_vjp_for_custom_vjp() {
        // f(x) = custom_vjp(sin, forward, tripled_backward)(x). The forward splices the forward program and stages
        // one opaque `CustomVjpTangent` carrier over `[dx, residual=cos(x)]`; the direct transpose replays the tripled
        // backward program on `[residual, cotangent]`, yielding `3*cos(x)*cotangent`. This must equal `vjp` of the same
        // custom-VJP call, proving the carrier actually runs `backward` (a folded zero tangent would silently give a
        // wrong zero gradient). The re-key path is intentionally not exercised: the primal-enum carrier has no linear
        // operation family variant to re-key into.
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let context = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();

        let (_, reference_pullback) = domain.vjp(scalar_custom_vjp_function, vec![Scalar::from(0.7)]).unwrap();
        let (reference_pullback, reference_residuals) = reference_pullback.into_parts();
        let mut reference_inputs = vec![Scalar::from(2.5)];
        reference_inputs.extend(reference_residuals);
        let reference_cotangents = reference_pullback.interpret_in_context(&context, reference_inputs).unwrap();

        let (_, direct_pullback, residuals) =
            vjp_direct(&domain, scalar_custom_vjp_function, vec![Scalar::from(0.7)]).unwrap();
        let mut direct_inputs = vec![Scalar::from(2.5)];
        direct_inputs.extend(residuals);
        let direct_cotangents = direct_pullback.interpret_in_context(&context, direct_inputs).unwrap();

        assert_close(&direct_cotangents, &reference_cotangents, "direct-transpose vs vjp cotangent");
        // The tripled oracle gives `3 * cos(0.7) * 2.5`; assert the concrete value so a silently-zero or primal-rule
        // gradient is caught even if `vjp` itself regressed.
        assert_close(
            &direct_cotangents,
            &[Scalar::from(3.0 * 0.7f64.cos() * 2.5)],
            "direct-transpose tripled gradient",
        );
    }

    #[test]
    fn test_custom_vjp_backward_zero_outputs_are_recovered_as_structural_zeros() {
        use crate::operations::constants::ZeroOperation;
        use crate::operations::math::AddOperation;
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;
        use crate::programs::operations::Operation;
        use crate::tracing_v2::operations::custom_derivatives::CustomVjpOperation;

        // f(x) = custom_vjp(add)(sin(x), cos(x)) with a user backward that returns `[cotangent, zero]`: the second
        // primal input is declared non-differentiated through a canonical `zero` output. The transpose splice must
        // recover that zero *structurally*, so the `cos(x)` branch contributes no adjoint work: the pullback contains
        // no `add` accumulation (only the `sin` branch contributes to `x`) and the gradient is `cos(x) * cotangent`.
        fn function(inputs: Vec<ScalarTracer>) -> Result<Vec<ScalarTracer>, ProgramError> {
            use crate::contexts::StagingContext;
            use crate::operations::math::{CosOperation, SinOperation};

            let mut primal_builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
            let a = primal_builder.add_input(DataType::F64);
            let b = primal_builder.add_input(DataType::F64);
            let y = primal_builder.add_instruction(AddOperation, Vec::new(), vec![a, b]).unwrap()[0];
            let primal = primal_builder.build(vec![y], vec![Placeholder, Placeholder], vec![Placeholder]).unwrap();

            let mut forward_builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
            let a = forward_builder.add_input(DataType::F64);
            let b = forward_builder.add_input(DataType::F64);
            let y = forward_builder.add_instruction(AddOperation, Vec::new(), vec![a, b]).unwrap()[0];
            let forward = forward_builder.build(vec![y], vec![Placeholder, Placeholder], vec![Placeholder]).unwrap();

            let mut backward_builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
            let cotangent = backward_builder.add_input(DataType::F64);
            let zero =
                backward_builder.add_instruction(ZeroOperation::new(DataType::F64), Vec::new(), Vec::new()).unwrap()[0];
            let backward = backward_builder
                .build(vec![cotangent, zero], vec![Placeholder], vec![Placeholder, Placeholder])
                .unwrap();

            let operation = CustomVjpOperation::new();
            let a = inputs[0].context().stage_operation(SinOperation, Vec::new(), &[&inputs[0]])?.remove(0);
            let b = inputs[0].context().stage_operation(CosOperation, Vec::new(), &[&inputs[0]])?.remove(0);
            inputs[0].context().stage_operation(
                ScalarOperation::CustomVjp(operation),
                vec![primal, forward, backward],
                &[&a, &b],
            )
        }

        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let context = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (_, pullback, residuals) = vjp_direct(&domain, function, vec![Scalar::from(0.7)]).unwrap();
        let mut pullback_inputs = vec![Scalar::from(2.5)];
        pullback_inputs.extend(residuals);
        let cotangents = pullback.interpret_in_context(&context, pullback_inputs).unwrap();
        assert_close(&cotangents, &[Scalar::from(0.7f64.cos() * 2.5)], "zero-backward gradient");

        // With the zero recovered structurally, the `cos` branch contributes nothing to `x`, so the pullback never
        // stages an `add` accumulation (a lost zero would flow as a live cotangent through the `cos` branch's
        // bilinear rule and accumulate against the `sin` branch's contribution).
        assert!(
            !pullback.instructions().iter().any(|instruction| instruction.operation().name() == "add"),
            "expected no adjoint accumulation in the pullback, but got:\n{pullback}",
        );
    }

    #[test]
    fn test_forward_through_custom_vjp_is_rejected() {
        // `custom_vjp` is reverse-mode-only. The forward builds a tangent program containing the opaque
        // `CustomVjpTangent` carrier, but interpreting that tangent program (which forward mode does) replays the
        // carrier, whose interpretation rejects forward mode with the canonical reverse-only error.
        use crate::programs::types::TypeError;

        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        match domain.jvp(
            |inputs: Vec<ScalarJvpTracer>| {
                let (operation, operation_regions) = scalar_custom_vjp_operation()?;
                inputs[0]
                    .context()
                    .bind(ScalarOperation::CustomVjp(operation), operation_regions, &[inputs[0].clone()])
            },
            vec![Scalar::from(0.7)],
            vec![Scalar::from(1.5)],
        ) {
            Err(crate::differentiation::DifferentiationError::Program(ProgramError::Type(TypeError { message })))
                if message.starts_with("custom_vjp does not support forward-mode differentiation") => {}
            Err(other) => panic!("expected the reverse-only TypeError from the forward but got {other:?}"),
            Ok(_) => panic!("expected the forward through custom_vjp to be rejected but it succeeded"),
        }
    }

    /// Linearization composes under a live outer staging trace — the first recorded consumer of
    /// parent-context-polymorphic partial evaluation: `linearize` invoked with outer tracers as primals stages the
    /// primal computation into the outer program, keeps the tangent map residual, and `apply` stages tangent pushes
    /// into the same outer trace. Building and interpreting the outer program end-to-end reproduces the eager
    /// linearization at the same point.
    #[test]
    fn test_linearize_composes_under_an_outer_staging_trace() {
        use crate::tracing::TracingContext;

        type Outer = TracingContext<Scalar, ScalarOperation<Scalar>>;

        let primal_values = vec![Scalar::from(0.7), Scalar::from(1.3)];
        let tangent_values = vec![Scalar::from(1.0), Scalar::from(0.5)];

        // Reference: eager linearize at the concrete primals.
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (reference_primal, reference_forward) = domain
            .linearize::<_, Vec<Scalar>, Vec<ScalarLinearizationTracer>>(
                |inputs| Ok(vec![inputs[0].sin()? * inputs[1].clone()]),
                primal_values.clone(),
            )
            .unwrap();
        let reference_tangent = reference_forward.apply(tangent_values.clone()).unwrap();

        // Staged: the same linearize invoked under a live outer trace, with outer tracers as primals and tangents.
        let outer = Outer::new();
        let primals = vec![outer.input(DataType::F64), outer.input(DataType::F64)];
        let (staged_primal, staged_forward) = outer
            .linearize::<_, Vec<Tracer<Outer>>, Vec<LinearizationTracer<Outer>>>(
                |inputs| Ok(vec![inputs[0].sin()? * inputs[1].clone()]),
                primals,
            )
            .unwrap();
        let tangents = vec![outer.input(DataType::F64), outer.input(DataType::F64)];
        let staged_tangent = staged_forward.apply(tangents).unwrap();

        // The outer program now contains the staged primal computation plus the staged tangent map; interpreting it
        // end-to-end at `[primals ++ tangents]` reproduces the eager linearization.
        let output_atoms = staged_primal
            .iter()
            .chain(staged_tangent.iter())
            .map(|tracer| tracer.atom_id().unwrap())
            .collect::<Vec<_>>();
        let output_count = output_atoms.len();
        let outer_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Scalar>, Vec<Scalar>>(output_atoms, vec![Placeholder; 4], vec![Placeholder; output_count])
            .unwrap()
            .into_simplified()
            .unwrap();
        let mut runtime_inputs = primal_values;
        runtime_inputs.extend(tangent_values);
        let values = outer_program.interpret(runtime_inputs).unwrap();
        assert_close(&values[..reference_primal.len()], &reference_primal, "staged linearize primal");
        assert_close(&values[reference_primal.len()..], &reference_tangent, "staged linearize tangent");
    }

    /// Staged linearization through control flow: under a live outer trace, a fused `condition` with a
    /// literal-backed predicate still inlines its taken branch — the predicate concretizes through the staging
    /// context even though it flows as a tracer — so linearize-under-trace handles control flow end-to-end.
    #[test]
    fn test_linearize_composes_under_an_outer_staging_trace_through_condition() {
        use crate::operations::control_flow::ConditionOperation;
        use crate::tests::TestArray;
        use crate::tracing::TracingContext;
        use crate::tracing_v2::ArrayOperation;
        use crate::tracing_v2::test_util::scalar_scale_branch;
        use crate::types::ArrayType;

        type Outer = TracingContext<TestArray, ArrayOperation<TestArray>>;

        let condition_function = |x: LinearizationTracer<Outer>| {
            let condition_regions = vec![scalar_scale_branch(2.0), scalar_scale_branch(3.0)];
            let condition = ConditionOperation::new();
            let predicate = x.context().lift(TestArray::new(ArrayType::scalar(DataType::Boolean), vec![1.0]))?;
            let mut outputs = x.context().bind(
                ArrayOperation::Condition(condition),
                condition_regions.clone(),
                &[predicate, x.clone()],
            )?;
            Ok(outputs.remove(0))
        };

        // The eager behavior of the same function is pinned by `test_linearize_through_condition_matches_jvp`:
        // `f(4.0) = 8.0` and the tangent map is scale-by-2. Staged: the primal condition inlines its taken branch
        // into the outer program, and `apply` stages the scale-by-2 tangent map into the same trace.
        let outer = Outer::new();
        let primal = outer.input(ArrayType::scalar(DataType::F64));
        let (staged_primal, staged_forward) =
            outer.linearize::<_, Tracer<Outer>, LinearizationTracer<Outer>>(condition_function, primal).unwrap();
        let tangent = outer.input(ArrayType::scalar(DataType::F64));
        let staged_tangent = staged_forward.apply(tangent).unwrap();

        let output_atoms = vec![staged_primal.atom_id().unwrap(), staged_tangent.atom_id().unwrap()];
        let outer_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestArray>, Vec<TestArray>>(output_atoms, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap()
            .into_simplified()
            .unwrap();
        let values = outer_program.interpret(vec![TestArray::scalar(4.0), TestArray::scalar(1.5)]).unwrap();
        assert_eq!(values[0].values, vec![8.0]);
        assert_eq!(values[1].values, vec![3.0]);
    }

    /// Equivalence of the raw context-generic partial-evaluation trace with `Program::linearize`: partially evaluating
    /// a fused JVP with primals seeded as inputs of a *fresh staging context* and tangents unknown reproduces the same
    /// split — primal work folds into the outer program, the residual program is the linear tangent map, and known
    /// feeders are residual edges. `Program::linearize` now obtains that split directly by composing differentiation
    /// over partial evaluation rather than constructing this fused intermediary. The test pins their semantic
    /// equivalence: interpreting both splits at concrete primals and tangents yields identical results.
    #[test]
    fn test_staging_partial_evaluation_reproduces_the_linearize_split() {
        use crate::contexts::StagingContext;
        use crate::partial::{PartialEvaluationInput, PartialEvaluationOutput};
        use crate::tracing::TracingContext;

        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let context = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let primals = vec![Scalar::from(0.7), Scalar::from(1.3)];
        let tangents = vec![Scalar::from(1.0), Scalar::from(0.5)];

        // f(x, y) = (sin(x) * y, x * x + y): shared primal work with several residual edges (sin(x), cos(x), y, x).
        let function = |inputs: Vec<ScalarTracer>| -> Result<Vec<ScalarTracer>, ProgramError> {
            Ok(vec![inputs[0].sin()? * inputs[1].clone(), inputs[0].clone() * inputs[0].clone() + inputs[1].clone()])
        };

        // Reference: the direct `Program::linearize` split.
        let input_types = primals.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
        let (_, primal_program) = NestedTracingContext::trace(domain.clone(), function, input_types).unwrap();
        let primal_program = primal_program.into_simplified().unwrap();
        let linearization = primal_program.linearize().unwrap();
        let mut reference_known = linearization.primal().interpret_in_context(&context, primals.clone()).unwrap();
        let reference_residuals = reference_known.split_off(reference_known.len() - linearization.residual_count());
        let mut reference_tangent_inputs = tangents.clone();
        reference_tangent_inputs.extend(reference_residuals);
        let reference_tangents =
            linearization.tangent().interpret_in_context(&context, reference_tangent_inputs).unwrap();

        // Generic trace: partially evaluate the same fused JVP program with `C` = a fresh staging context, primals
        // seeded `Known(C.input(..))` and tangents `Unknown(type)`.
        let jvp_program = primal_program.jvp().unwrap().into_simplified().unwrap();
        let outer = TracingContext::<Scalar, ScalarOperation<Scalar>>::new();
        let primal_count = primals.len();
        let mut knowledge = Vec::with_capacity(2 * primal_count);
        for _ in 0..primal_count {
            knowledge.push(PartialValue::Known(outer.input(DataType::F64)));
        }
        for _ in 0..primal_count {
            knowledge.push(PartialValue::Unknown(DataType::F64));
        }
        let evaluation = jvp_program.partially_evaluate_in_context(&outer, knowledge.as_slice()).unwrap();

        // The outer program plays `Linearization::primal`'s role: its outputs are the folded primal outputs
        // followed by the known feeders (the residual edges), in feeder order.
        let (primal_outputs, tangent_outputs) = evaluation.outputs.split_at(evaluation.outputs.len() / 2);
        let mut outer_output_atoms = Vec::new();
        for output in primal_outputs {
            match output {
                PartialEvaluationOutput::Known(value) => outer_output_atoms.push(value.atom_id().unwrap()),
                PartialEvaluationOutput::Unknown(_) => panic!("a primal output did not fold under the staging trace"),
            }
        }
        let mut residual_edge_count = 0;
        for input in evaluation.inputs.iter() {
            if let PartialEvaluationInput::Known(value) = input {
                outer_output_atoms.push(value.atom_id().unwrap());
                residual_edge_count += 1;
            }
        }
        let outer_output_count = outer_output_atoms.len();
        let known_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Scalar>, Vec<Scalar>>(
                outer_output_atoms,
                vec![Placeholder; primal_count],
                vec![Placeholder; outer_output_count],
            )
            .unwrap()
            .into_simplified()
            .unwrap();
        let mut known_values = known_program.interpret_in_context(&context, primals).unwrap();
        let residual_values = known_values.split_off(known_values.len() - residual_edge_count);
        assert_close(&known_values, &reference_known, "staging-trace primal outputs");

        // The residual program plays `Linearization::tangent`'s role: interpret it at the tangents plus the
        // residual-edge values, in feeder order, and reassemble the tangent outputs.
        let mut remaining_tangents = tangents.into_iter();
        let mut remaining_residuals = residual_values.into_iter();
        let residual_inputs = evaluation
            .inputs
            .iter()
            .map(|input| match input {
                PartialEvaluationInput::Unknown(_) => remaining_tangents.next().unwrap(),
                PartialEvaluationInput::Known(_) => remaining_residuals.next().unwrap(),
            })
            .collect::<Vec<_>>();
        let staged_tangent_values = evaluation.program.interpret(residual_inputs).unwrap();
        let reassembled_tangents = tangent_outputs
            .iter()
            .map(|output| match output {
                PartialEvaluationOutput::Known(_) => panic!("a tangent output unexpectedly folded"),
                PartialEvaluationOutput::Unknown(index) => staged_tangent_values[*index],
            })
            .collect::<Vec<_>>();
        assert_close(&reassembled_tangents, &reference_tangents, "staging-trace tangent outputs");
    }
}

#[cfg(test)]
mod array_linearization_tests {
    /// Linearizes `function` at array `primals` through the front end and partial evaluation.
    ///
    /// This traces `function` into a primal program through `domain` and hands it to the generic
    /// [`Program::linearize`](crate::Program::linearize) core, which builds the capture-free JVP program over the array
    /// slice and partially evaluates it. The returned [`Linearization`] carries the known primal sub-program, the unknown
    /// linear tangent sub-program, and the metadata needed to reassemble their outputs and transpose the tangent side; the
    /// concrete primal outputs are recovered by interpreting [`primal`](Linearization::primal).
    ///
    /// This entry point is specialized to [`EagerContext<TestArray, ArrayOperation<TestArray>>`] and to straight-line array functions over the supported
    /// slice, mirroring the array partial-evaluation driver whose array linearization-context obligations also do not
    /// discharge generically; functions reaching unsupported operations fail with an
    /// [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
    ///
    /// # Parameters
    ///
    ///   - `domain`: Differentiation context supplying the primal value semantics and operation family.
    ///   - `function`: Array closure to linearize; it is traced once into a primal program.
    ///   - `primals`: Structured primal input values at which to linearize.
    fn array_linearize<Function>(
        domain: &EagerContext<TestArray, ArrayOperation<TestArray>>,
        function: Function,
        primals: Vec<TestArray>,
    ) -> Result<Linearization<TestArray, ArrayOperation<TestArray>>, crate::differentiation::DifferentiationError>
    where
        Function: FnOnce(
            Vec<Tracer<NestedTracingContext<EagerContext<TestArray, ArrayOperation<TestArray>>>>>,
        ) -> Result<
            Vec<Tracer<NestedTracingContext<EagerContext<TestArray, ArrayOperation<TestArray>>>>>,
            ProgramError,
        >,
    {
        let input_types = primals.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
        let (_, primal_program) = NestedTracingContext::trace(domain.clone(), function, input_types)?;
        primal_program.into_simplified()?.linearize()
    }

    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{DifferentiationTracer, Linearization, LinearizationTracer};
    use crate::operations::compare::{Compare, CompareOperation, ComparisonDirection};
    use crate::operations::constants::ZeroLike;
    use crate::operations::control_flow::{Select, WhileOperation};
    use crate::operations::manipulation::{Broadcast, Reshape};
    use crate::operations::math::Sin;
    use crate::programs::Program;
    use crate::programs::operations::Operation;
    use crate::tracing::{NestedTracingContext, Tracer};
    use crate::tracing_v2::operations::dot::{Dot, DotDimensionNumbers};
    use crate::tracing_v2::operations::reduce::{Reduce, ReductionKind};
    use crate::tracing_v2::unroll::unroll_concretizable_whiles;
    use crate::tracing_v2::{ForwardModeDifferentiate, ReverseModeDifferentiate};

    use super::*;

    /// Tracer leaf seen by the array test closures.
    type ArrayTracer = Tracer<NestedTracingContext<EagerContext<TestArray, ArrayOperation<TestArray>>>>;

    /// Forward-mode dual leaf seen by the array `jvp` closures.
    type ArrayJvpTracer = DifferentiationTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>;

    /// Forward-mode dual leaf seen by the array closures handed to [`ForwardModeDifferentiate::linearize`] and
    /// [`ReverseModeDifferentiate::vjp`].
    type ArrayLinearizationTracer = LinearizationTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>;

    /// Absolute tolerance for comparing the path against the established transforms.
    const TOLERANCE: f64 = 1e-12;

    /// Asserts that every element of `left` is within [`TOLERANCE`] of the corresponding element of `right`.
    fn assert_close(left: &[f64], right: &[f64], label: &str) {
        assert_eq!(left.len(), right.len(), "{label}: length mismatch ({left:?} vs {right:?})");
        for (index, (a, b)) in left.iter().zip(right).enumerate() {
            assert!((a - b).abs() <= TOLERANCE, "{label}: element {index} differs ({a} vs {b})");
        }
    }

    /// Asserts that every element of `left` is within [`TOLERANCE`] of the corresponding element of `right`, treating
    /// each [`TestArray`] as its flat row-major payload.
    fn assert_arrays_close(left: &[TestArray], right: &[TestArray], label: &str) {
        assert_eq!(left.len(), right.len(), "{label}: count mismatch ({} vs {})", left.len(), right.len());
        for (index, (a, b)) in left.iter().zip(right).enumerate() {
            assert_eq!(a.r#type, b.r#type, "{label}: output {index} type mismatch ({:?} vs {:?})", a.r#type, b.r#type);
            assert_close(&a.values, &b.values, &format!("{label} (output {index})"));
        }
    }

    /// Asserts forward equivalence for array functions: the primal and tangent sub-programs, reassembled, equal
    /// the outputs of [`ForwardModeDifferentiate::jvp`] for `function` at `primals` with the given `tangents`.
    fn assert_array_forward_equivalent<JvpFunction, LinearizeFunction>(
        jvp_function: JvpFunction,
        linearize_function: LinearizeFunction,
        primals: Vec<TestArray>,
        tangents: Vec<TestArray>,
    ) where
        JvpFunction: FnOnce(Vec<ArrayJvpTracer>) -> Result<Vec<ArrayJvpTracer>, ProgramError>,
        LinearizeFunction: FnOnce(Vec<ArrayTracer>) -> Result<Vec<ArrayTracer>, ProgramError>,
    {
        let domain = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let (reference_primals, reference_tangents) =
            domain.jvp(jvp_function, primals.clone(), tangents.clone()).unwrap();

        let linearization = array_linearize(&domain, linearize_function, primals.clone()).unwrap();
        let context = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();

        // The known side computes the primal outputs followed by the residuals; interpreting it recovers the concrete
        // primal outputs that the linearization core no longer caches.
        let mut known_outputs = linearization.primal().interpret_in_context(&context, primals).unwrap();
        let residuals = known_outputs.split_off(known_outputs.len() - linearization.residual_count());
        assert_arrays_close(&known_outputs, &reference_primals, "forward primal");

        // The unknown side is the linear tangent map, taking the tangents followed by the residuals. Canonical arity
        // places all tangent outputs on the unknown side in original order, so they are compared directly.
        let mut tangent_inputs = tangents;
        tangent_inputs.extend(residuals);
        let unknown_outputs = linearization.tangent().interpret_in_context(&context, tangent_inputs).unwrap();
        assert_arrays_close(&unknown_outputs, &reference_tangents, "forward tangent");
    }

    /// Asserts reverse equivalence for array functions: transposing the tangent sub-program yields the same input
    /// cotangents as the [`ReverseModeDifferentiate::vjp`] pullback for `function` at `primals`, for the given
    /// `output_cotangents`.
    fn assert_array_reverse_equivalent<VjpFunction, LinearizeFunction>(
        vjp_function: VjpFunction,
        linearize_function: LinearizeFunction,
        primals: Vec<TestArray>,
        output_cotangents: Vec<TestArray>,
    ) where
        VjpFunction: FnOnce(Vec<ArrayLinearizationTracer>) -> Result<Vec<ArrayLinearizationTracer>, ProgramError>,
        LinearizeFunction: FnOnce(Vec<ArrayTracer>) -> Result<Vec<ArrayTracer>, ProgramError>,
    {
        let domain = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let context = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();

        let (_, pullback) = domain.vjp(vjp_function, primals.clone()).unwrap();
        let (pullback, vjp_residuals) = pullback.into_parts();
        let mut reference_inputs = output_cotangents.clone();
        reference_inputs.extend(vjp_residuals);
        let reference_cotangents = pullback.interpret_in_context(&context, reference_inputs).unwrap();

        let (_, pullback, residuals) = vjp_direct(&domain, linearize_function, primals).unwrap();
        let mut pullback_inputs = output_cotangents;
        pullback_inputs.extend(residuals);
        let input_cotangents = pullback.interpret_in_context(&context, pullback_inputs).unwrap();
        assert_arrays_close(&input_cotangents, &reference_cotangents, "reverse cotangent");
    }

    /// Runs the raw fused-JVP program pipeline — trace, eager `while` unroll, fused JVP program build,
    /// simplification, and direct interpretation at `(primals ++ tangents)` — as an independent oracle for
    /// [`ForwardModeDifferentiate::jvp`]: the dual-interpreter entry point (including its eager `while` rule) must
    /// agree with the program-level pipeline. Returns the flat primal and tangent outputs.
    fn fused_pipeline_jvp<Function>(
        domain: &EagerContext<TestArray, ArrayOperation<TestArray>>,
        function: Function,
        primals: Vec<TestArray>,
        tangents: Vec<TestArray>,
    ) -> Result<(Vec<TestArray>, Vec<TestArray>), ProgramError>
    where
        Function: FnOnce(Vec<ArrayTracer>) -> Vec<ArrayTracer>,
    {
        let input_types = primals.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
        let (_, program) = NestedTracingContext::trace(domain.clone(), |inputs| Ok(function(inputs)), input_types)?;
        let program = unroll_concretizable_whiles(domain, program.into_simplified()?, primals.clone())?;
        let jvp_program = program.jvp()?.into_simplified()?;
        let mut combined_inputs = primals;
        combined_inputs.extend(tangents);
        let mut outputs = jvp_program.interpret_in_context(domain, combined_inputs)?;
        let tangent_outputs = outputs.split_off(outputs.len() / 2);
        Ok((outputs, tangent_outputs))
    }

    /// Reverse-mode-differentiates `function` at `primals` through the raw program pipeline — trace, eager `while`
    /// unroll, direct program linearization, primal replay, and partition-aware transposition — so the packaged
    /// closure-level [`ReverseModeDifferentiate::vjp`] surface can be compared against the independently staged
    /// program-level path. Returns the flat primal outputs, the pullback over `(output_cotangents ++ residuals)`, and
    /// the residuals.
    fn vjp_direct<Function>(
        domain: &EagerContext<TestArray, ArrayOperation<TestArray>>,
        function: Function,
        primals: Vec<TestArray>,
    ) -> Result<
        (Vec<TestArray>, Program<TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>>, Vec<TestArray>),
        ProgramError,
    >
    where
        Function: FnOnce(Vec<ArrayTracer>) -> Result<Vec<ArrayTracer>, ProgramError>,
    {
        let input_types = primals.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
        let (_, program) = NestedTracingContext::trace(domain.clone(), function, input_types)?;
        let program = unroll_concretizable_whiles(domain, program.into_simplified()?, primals.clone())?;
        let linearization = program.linearize()?;
        let primal_side = linearization.primal().interpret_in_context(domain, primals)?;
        let primal_output_count = primal_side.len() - linearization.residual_count();
        let residuals = primal_side[primal_output_count..].to_vec();
        let primal_outputs = primal_side[..primal_output_count].to_vec();
        let pullback = linearization.pullback()?;
        Ok((primal_outputs, pullback, residuals))
    }

    /// Asserts the control-flow reverse-mode equivalence gate: the direct-transpose pullback (built by
    /// [`vjp_direct`], which transposes the primal `Scan` / `Condition` operations of the tangent
    /// sub-program directly) produces the same linear-input cotangents as the established
    /// [`ReverseModeDifferentiate::vjp`] pullback, for a control-flow `function` at `primals` and the given
    /// `output_cotangents`. The same `function` is supplied twice because each consuming entry point traces it once
    /// into a primal program.
    ///
    /// The direct pullback consumes the residuals as ordinary pullback inputs, so it is interpreted at
    /// `output_cotangents ++ residuals`. It prunes any dead operand tangent — for example the Boolean predicate of a
    /// `condition`, which has no tangent space — so it emits one cotangent per *live* operand, whereas
    /// [`ReverseModeDifferentiate::vjp`] emits a typed zero for every primal input including the predicate. The direct-transpose
    /// path is correct precisely when its cotangents equal the trailing (live-operand) cotangents of `vjp`, which this
    /// asserts by comparing against the last `direct.len()` entries of the reference cotangents.
    fn assert_array_control_flow_reverse_equivalent_to_vjp<VjpFunction, DirectFunction>(
        vjp_function: VjpFunction,
        direct_function: DirectFunction,
        primals: Vec<TestArray>,
        output_cotangents: Vec<TestArray>,
    ) where
        VjpFunction: FnOnce(Vec<ArrayLinearizationTracer>) -> Result<Vec<ArrayLinearizationTracer>, ProgramError>,
        DirectFunction: FnOnce(Vec<ArrayTracer>) -> Result<Vec<ArrayTracer>, ProgramError>,
    {
        let domain = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let context = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();

        // Reference: the established `vjp` pullback emits one input cotangent per primal input and consumes
        // `output_cotangents ++ residuals`.
        let (_, reference_pullback) = domain.vjp(vjp_function, primals.clone()).unwrap();
        let (reference_pullback, reference_residuals) = reference_pullback.into_parts();
        let mut reference_inputs = output_cotangents.clone();
        reference_inputs.extend(reference_residuals);
        let reference_cotangents = reference_pullback.interpret_in_context(&context, reference_inputs).unwrap();

        // Direct path: the pullback is over the primal `ArrayOperation` enum and consumes the residuals as inputs, so
        // it is fed `output_cotangents ++ residuals`.
        let (_, direct_pullback, residuals) = vjp_direct(&domain, direct_function, primals).unwrap();
        let mut direct_inputs = output_cotangents;
        direct_inputs.extend(residuals);
        let direct_cotangents = direct_pullback.interpret_in_context(&context, direct_inputs).unwrap();

        // The pullback's live-operand cotangents equal the trailing (live-operand) cotangents of `vjp` — the
        // leading entries `vjp` emits for any pruned operand (such as a condition predicate) are dropped here.
        let reference_live = &reference_cotangents[reference_cotangents.len() - direct_cotangents.len()..];
        assert_arrays_close(&direct_cotangents, reference_live, "control-flow direct-transpose vs vjp cotangent");
    }

    #[test]
    fn test_array_control_flow_reverse_equivalent_to_vjp() {
        use crate::types::DataType;

        // Condition: the direct path transposes the primal `condition` operand-form — reading the known predicate and
        // the joined residuals from the pullback and recursing into each branch through `transpose_with_respect_to` — so it
        // matches `vjp`'s operand cotangent.
        assert_array_control_flow_reverse_equivalent_to_vjp(
            condition_function,
            condition_function,
            vec![TestArray::new(ArrayType::scalar(DataType::Boolean), vec![1.0]), TestArray::scalar(0.7)],
            vec![TestArray::scalar(2.5)],
        );
        assert_array_control_flow_reverse_equivalent_to_vjp(
            condition_function,
            condition_function,
            vec![TestArray::new(ArrayType::scalar(DataType::Boolean), vec![0.0]), TestArray::scalar(0.7)],
            vec![TestArray::scalar(2.5)],
        );

        // Scan: the direct path transposes the primal `scan` operand-form — reading the known residual stacks from the
        // pullback, recursing into the body through `transpose_with_respect_to`, and re-staging a flipped-`reverse` scan —
        // so it matches `vjp` one-to-one (no predicate to prune).
        assert_array_control_flow_reverse_equivalent_to_vjp(
            scan_function,
            scan_function,
            vec![TestArray::scalar(1.0), TestArray::vector(vec![2.0, 3.0, 4.0])],
            vec![TestArray::scalar(1.0), TestArray::vector(vec![1.0, 1.0, 1.0])],
        );

        // Bounded while: the forward lowers it to a primal `scan`, so the direct path is covered by the `Scan`
        // rule automatically and matches `vjp`.
        assert_array_control_flow_reverse_equivalent_to_vjp(
            bounded_while_function,
            bounded_while_function,
            vec![TestArray::scalar(2.0)],
            vec![TestArray::scalar(1.0)],
        );
    }

    /// Builds the eager, data-dependent `while x < 100 { x = x * x }` loop over [`EagerContext<TestArray, ArrayOperation<TestArray>>`]'s scalar arrays.
    /// Its trip count depends on the runtime value and the loop carries no
    /// [`iteration_bound`](crate::operations::control_flow::WhileOperation::iteration_bound), so it is the kind of
    /// unbounded loop the front end rejects unless the eager unroll-then-fuse pre-pass first unrolls it at the
    /// concrete primal.
    fn array_squaring_while()
    -> (WhileOperation, Vec<Program<TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>>>) {
        use crate::operations::math::MulOperation;
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;
        use crate::types::DataType;

        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let condition_state = condition_builder.add_input(scalar_f64.clone());
        let threshold = condition_builder.add_constant(TestArray::scalar(100.0));
        let predicate = condition_builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::LessThan),
                Vec::new(),
                vec![condition_state, threshold],
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let state = body_builder.add_input(scalar_f64);
        let squared = body_builder.add_instruction(MulOperation, Vec::new(), vec![state, state]).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![squared], vec![Placeholder], vec![Placeholder])
            .unwrap();
        (WhileOperation::new(), vec![condition, body])
    }

    #[test]
    fn test_array_jvp_pipeline_unrolls_eager_unbounded_while() {
        // The eager unroll-then-fuse pre-pass unrolls the unbounded `while x < 100 { x = x * x }` at the concrete
        // primal, so forward mode through it now succeeds on the capture-free path and must reproduce the
        // established eager `jvp`, whose eager `while` rule differentiates the same loop directly at the concrete
        // primal. From `x = 1.5` the loop runs four squarings.
        let domain = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let (reference_primals, reference_tangents) = domain
            .jvp(
                |inputs: Vec<ArrayJvpTracer>| {
                    let (while_operation, while_regions) = array_squaring_while();
                    inputs[0].context().bind(while_operation, while_regions, &[inputs[0].clone()])
                },
                vec![TestArray::scalar(1.5)],
                vec![TestArray::scalar(1.0)],
            )
            .unwrap();

        let (primal_outputs, tangent_outputs) = fused_pipeline_jvp(
            &domain,
            |inputs| {
                let (while_operation, while_regions) = array_squaring_while();
                inputs[0].context().bind(while_operation, while_regions, &[inputs[0].clone()]).unwrap()
            },
            vec![TestArray::scalar(1.5)],
            vec![TestArray::scalar(1.0)],
        )
        .unwrap();
        assert_arrays_close(&primal_outputs, &reference_primals, "eager unbounded while jvp pipeline primal");
        assert_arrays_close(&tangent_outputs, &reference_tangents, "eager unbounded while jvp pipeline tangent");
    }

    #[test]
    fn test_array_vjp_pipeline_unrolls_eager_unbounded_while() {
        // The unrolled straight-line primal program produces a control-flow-free tangent program that transposes via
        // the existing partitioned transposition, so reverse mode through the unbounded `while x < 100 { x = x * x }`
        // now succeeds and must reproduce the established eager `vjp`. The direct-transpose pullback consumes the
        // residuals as ordinary inputs, so it is interpreted at `output_cotangents ++ residuals`.
        let domain = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let context = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();

        let (_, reference_pullback) = domain
            .vjp(
                |inputs| {
                    let (while_operation, while_regions) = array_squaring_while();
                    inputs[0].context().bind(while_operation, while_regions, &[inputs[0].clone()])
                },
                vec![TestArray::scalar(1.5)],
            )
            .unwrap();
        let (reference_pullback, reference_residuals) = reference_pullback.into_parts();
        let mut reference_inputs = vec![TestArray::scalar(1.0)];
        reference_inputs.extend(reference_residuals);
        let reference_cotangents = reference_pullback.interpret_in_context(&context, reference_inputs).unwrap();

        let (_, pullback, residuals) = vjp_direct(
            &domain,
            |inputs| {
                let (while_operation, while_regions) = array_squaring_while();
                inputs[0].context().bind(while_operation, while_regions, &[inputs[0].clone()])
            },
            vec![TestArray::scalar(1.5)],
        )
        .unwrap();
        let mut pullback_inputs = vec![TestArray::scalar(1.0)];
        pullback_inputs.extend(residuals);
        let input_cotangents = pullback.interpret_in_context(&context, pullback_inputs).unwrap();
        assert_arrays_close(&input_cotangents, &reference_cotangents, "eager unbounded while vjp pipeline cotangent");
    }

    #[test]
    fn test_array_forward_equivalent_to_jvp() {
        // Elementwise: f(x) = x * sin(x) over a vector (Mul + Sin), with a non-trivial tangent.
        assert_array_forward_equivalent(
            |inputs| Ok(vec![inputs[0].clone() * inputs[0].sin()?]),
            |inputs| Ok(vec![inputs[0].clone() * inputs[0].sin()?]),
            vec![TestArray::vector(vec![0.7, -1.2, 2.0])],
            vec![TestArray::vector(vec![1.0, 0.5, -2.0])],
        );

        // Inner product: f(x) = x . x (Dot with both operands the differentiated input, so the product rule stages a
        // left and a right Dot).
        assert_array_forward_equivalent(
            |inputs| Ok(vec![inputs[0].dot(&inputs[0], &DotDimensionNumbers::inner_product())]),
            |inputs| Ok(vec![inputs[0].dot(&inputs[0], &DotDimensionNumbers::inner_product())]),
            vec![TestArray::vector(vec![1.0, 2.0, 3.0])],
            vec![TestArray::vector(vec![0.5, -1.0, 2.0])],
        );

        // Matrix multiply against a constant: f(x) = x @ c (Dot with the right operand held constant).
        assert_array_forward_equivalent(
            |inputs| {
                let constant = inputs[0].context().lift(TestArray::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]))?;
                Ok(vec![inputs[0].dot(&constant, &DotDimensionNumbers::matmul())])
            },
            |inputs| {
                let constant = inputs[0].context().constant(TestArray::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]));
                Ok(vec![inputs[0].dot(&constant, &DotDimensionNumbers::matmul())])
            },
            vec![TestArray::matrix(2, 2, vec![0.5, 1.0, -1.0, 2.0])],
            vec![TestArray::matrix(2, 2, vec![1.0, 0.0, 0.5, -2.0])],
        );

        // Structural reduce: f(x) = reduce_sum(x * x, axis 0), reducing a vector to a scalar. This exercises the
        // elementwise product feeding a structural (linear) sum reduction.
        assert_array_forward_equivalent(
            |inputs| Ok(vec![(inputs[0].clone() * inputs[0].clone()).reduce(&[0], ReductionKind::Sum)]),
            |inputs| Ok(vec![(inputs[0].clone() * inputs[0].clone()).reduce(&[0], ReductionKind::Sum)]),
            vec![TestArray::vector(vec![1.0, -2.0, 3.0])],
            vec![TestArray::vector(vec![0.5, 1.0, -1.0])],
        );

        // Select: f(x) = select(x > 0, 2x, 3x) over a vector, masking per element. The comparison is a
        // non-differentiated Boolean operand edge that becomes the primal select condition.
        assert_array_forward_equivalent(
            |inputs| {
                let mask = inputs[0].compare(&inputs[0].zero_like(), ComparisonDirection::GreaterThan)?;
                let on_true = inputs[0].clone() + inputs[0].clone();
                let on_false = inputs[0].clone() + inputs[0].clone() + inputs[0].clone();
                Ok(vec![Select::select(&mask, &on_true, &on_false)?])
            },
            |inputs| {
                let mask = inputs[0].compare(&inputs[0].zero_like(), ComparisonDirection::GreaterThan)?;
                let on_true = inputs[0].clone() + inputs[0].clone();
                let on_false = inputs[0].clone() + inputs[0].clone() + inputs[0].clone();
                Ok(vec![Select::select(&mask, &on_true, &on_false).unwrap()])
            },
            vec![TestArray::vector(vec![1.0, -1.0, 2.0])],
            vec![TestArray::vector(vec![1.0, 1.0, 1.0])],
        );

        // Structural broadcast + reshape: f(x) = reshape(x, [4]) + broadcast(reduce_sum(x)) over a 2x2 matrix.
        // Exercises a reshape and a broadcast of a reduced scalar feeding an elementwise add.
        assert_array_forward_equivalent(
            |inputs| Ok(vec![reshape_broadcast_jvp(&inputs[0])?]),
            |inputs| Ok(vec![reshape_broadcast(&inputs[0])]),
            vec![TestArray::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0])],
            vec![TestArray::matrix(2, 2, vec![0.5, -1.0, 2.0, 1.0])],
        );
    }

    #[test]
    fn test_array_reverse_equivalent_to_vjp() {
        // Elementwise: f(x) = x * sin(x) over a vector, with a non-unit output cotangent.
        assert_array_reverse_equivalent(
            |inputs| Ok(vec![inputs[0].clone() * inputs[0].sin()?]),
            |inputs| Ok(vec![inputs[0].clone() * inputs[0].sin()?]),
            vec![TestArray::vector(vec![0.7, -1.2, 2.0])],
            vec![TestArray::vector(vec![2.5, 1.0, -0.5])],
        );

        // Inner product: f(x) = x . x produces a scalar; the pullback of a scalar cotangent recovers 2 * cotangent * x.
        assert_array_reverse_equivalent(
            |inputs| Ok(vec![inputs[0].dot(&inputs[0], &DotDimensionNumbers::inner_product())]),
            |inputs| Ok(vec![inputs[0].dot(&inputs[0], &DotDimensionNumbers::inner_product())]),
            vec![TestArray::vector(vec![1.0, 2.0, 3.0])],
            vec![TestArray::scalar(1.5)],
        );

        // Matrix multiply against a constant: the pullback transposes through the captured right dot.
        assert_array_reverse_equivalent(
            |inputs| {
                let constant = inputs[0].context().lift(TestArray::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]))?;
                Ok(vec![inputs[0].dot(&constant, &DotDimensionNumbers::matmul())])
            },
            |inputs| {
                let constant = inputs[0].context().constant(TestArray::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]));
                Ok(vec![inputs[0].dot(&constant, &DotDimensionNumbers::matmul())])
            },
            vec![TestArray::matrix(2, 2, vec![0.5, 1.0, -1.0, 2.0])],
            vec![TestArray::matrix(2, 2, vec![1.0, -1.0, 0.5, 2.0])],
        );

        // Structural reduce: f(x) = reduce_sum(x * x, axis 0); the pullback broadcasts the scalar cotangent back.
        assert_array_reverse_equivalent(
            |inputs| Ok(vec![(inputs[0].clone() * inputs[0].clone()).reduce(&[0], ReductionKind::Sum)]),
            |inputs| Ok(vec![(inputs[0].clone() * inputs[0].clone()).reduce(&[0], ReductionKind::Sum)]),
            vec![TestArray::vector(vec![1.0, -2.0, 3.0])],
            vec![TestArray::scalar(2.0)],
        );

        // Select: f(x) = select(x > 0, 2x, 3x); the pullback routes each output cotangent to the selected branch.
        assert_array_reverse_equivalent(
            |inputs| {
                let mask = inputs[0].compare(&inputs[0].zero_like(), ComparisonDirection::GreaterThan)?;
                let on_true = inputs[0].clone() + inputs[0].clone();
                let on_false = inputs[0].clone() + inputs[0].clone() + inputs[0].clone();
                Ok(vec![Select::select(&mask, &on_true, &on_false).unwrap()])
            },
            |inputs| {
                let mask = inputs[0].compare(&inputs[0].zero_like(), ComparisonDirection::GreaterThan)?;
                let on_true = inputs[0].clone() + inputs[0].clone();
                let on_false = inputs[0].clone() + inputs[0].clone() + inputs[0].clone();
                Ok(vec![Select::select(&mask, &on_true, &on_false).unwrap()])
            },
            vec![TestArray::vector(vec![1.0, -1.0, 2.0])],
            vec![TestArray::vector(vec![1.0, 1.0, 1.0])],
        );

        // Structural broadcast + reshape: the pullback reshapes and reduces the cotangent back.
        assert_array_reverse_equivalent(
            |inputs| Ok(vec![reshape_broadcast(&inputs[0])]),
            |inputs| Ok(vec![reshape_broadcast(&inputs[0])]),
            vec![TestArray::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0])],
            vec![TestArray::vector(vec![1.0, -1.0, 0.5, 2.0])],
        );
    }

    /// Stages `f(x) = x + fill([3], 2.0)` over the closure inputs `[x]`. The `fill` is a nullary constant carrying a
    /// zero tangent, so the directional derivative of `f` is `dx` — the shared rule body for `Fill`.
    fn fill_function<V>(inputs: Vec<V>) -> Result<Vec<V>, ProgramError>
    where
        V: Value<Type = ArrayType> + std::ops::Add<Output = V>,
        V::DispatchDomain: Context<Type = ArrayType, Constant = TestArray, Operation = ArrayOperation<TestArray>>,
    {
        use crate::operations::constants::FillOperation;
        use crate::types::{DataType, Shape, Size};

        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        let mut filled =
            inputs[0]
                .dispatch_domain()
                .bind(FillOperation::new(vector_type, Scalar::from(2.0)), Vec::new(), &[])?;
        Ok(vec![inputs[0].clone() + filled.remove(0)])
    }

    /// Binds `f(x) = x + fill([3], 2.0)` through the forward-mode dual context, the `jvp` twin of [`fill_function`].
    fn fill_jvp_function(inputs: Vec<ArrayJvpTracer>) -> Result<Vec<ArrayJvpTracer>, ProgramError> {
        use crate::operations::constants::FillOperation;
        use crate::types::{DataType, Shape, Size};

        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        let mut filled =
            inputs[0].context().bind(FillOperation::new(vector_type, Scalar::from(2.0)), Vec::new(), &[])?;
        Ok(vec![inputs[0].clone() + filled.remove(0)])
    }

    #[test]
    fn test_array_forward_and_reverse_equivalent_for_fill() {
        // f(x) = x + fill([3], 2.0): the fill constant carries a zero tangent, so the forward tangent is `dx` and
        // the pullback passes the cotangent straight through, matching `jvp` / `vjp`.
        assert_array_forward_equivalent(
            fill_jvp_function,
            fill_function,
            vec![TestArray::vector(vec![1.0, -2.0, 3.0])],
            vec![TestArray::vector(vec![0.5, 1.0, -1.0])],
        );
        assert_array_reverse_equivalent(
            fill_function,
            fill_function,
            vec![TestArray::vector(vec![1.0, -2.0, 3.0])],
            vec![TestArray::vector(vec![2.5, 1.0, -0.5])],
        );
    }

    #[test]
    fn test_array_forward_and_reverse_equivalent_for_slice() {
        use crate::operations::manipulation::Slice;

        // f(x) = slice(x, [1], [4], strides=[1]): slicing is linear, so the tangent is the same slice of the
        // operand tangent and the pullback writes the cotangent back into the read window, matching `jvp` / `vjp`.
        assert_array_forward_equivalent(
            |inputs| Ok(vec![inputs[0].slice(&[1], &[4], &[1])?]),
            |inputs| Ok(vec![inputs[0].slice(&[1], &[4], &[1]).unwrap()]),
            vec![TestArray::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0])],
            vec![TestArray::vector(vec![0.5, -1.0, 2.0, 1.0, -0.5])],
        );
        assert_array_reverse_equivalent(
            |inputs| Ok(vec![inputs[0].slice(&[1], &[4], &[1]).unwrap()]),
            |inputs| Ok(vec![inputs[0].slice(&[1], &[4], &[1]).unwrap()]),
            vec![TestArray::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0])],
            vec![TestArray::vector(vec![2.5, 1.0, -0.5])],
        );

        // A strided slice exercises the pad-based pullback geometry: reads positions 1, 3, 5 of a length-6 vector.
        assert_array_reverse_equivalent(
            |inputs| Ok(vec![inputs[0].slice(&[1], &[6], &[2]).unwrap()]),
            |inputs| Ok(vec![inputs[0].slice(&[1], &[6], &[2]).unwrap()]),
            vec![TestArray::vector(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])],
            vec![TestArray::vector(vec![1.0, -1.0, 0.5])],
        );
    }

    #[test]
    fn test_array_forward_and_reverse_equivalent_for_update_slice() {
        use crate::operations::manipulation::UpdateSlice;

        // f(x, u) = update_slice(x, u, [1]): the operation is jointly linear in the operand and the update, so the
        // tangent updates the operand tangent with the update tangent and the pullback splits the cotangent into
        // the zeroed-window operand cotangent and the windowed update cotangent, matching `jvp` / `vjp`.
        assert_array_forward_equivalent(
            |inputs| Ok(vec![inputs[0].update_slice(&inputs[1], &[1])?]),
            |inputs| Ok(vec![inputs[0].update_slice(&inputs[1], &[1]).unwrap()]),
            vec![TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]), TestArray::vector(vec![7.0, 8.0])],
            vec![TestArray::vector(vec![0.5, -1.0, 2.0, 1.0]), TestArray::vector(vec![1.0, -0.5])],
        );
        assert_array_reverse_equivalent(
            |inputs| Ok(vec![inputs[0].update_slice(&inputs[1], &[1]).unwrap()]),
            |inputs| Ok(vec![inputs[0].update_slice(&inputs[1], &[1]).unwrap()]),
            vec![TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]), TestArray::vector(vec![7.0, 8.0])],
            vec![TestArray::vector(vec![2.5, 1.0, -0.5, 2.0])],
        );
    }

    /// Stages `f(x, u) = scatter_add(x, [[1], [3]], u)` over the closure inputs `[x, u]`. The integer indices are a
    /// constant of the trace, so the operation is jointly linear in `x` and `u` — the shared rule body for the
    /// scatter-add `Scatter`.
    fn scatter_add_function<V>(inputs: Vec<V>) -> Result<Vec<V>, ProgramError>
    where
        V: Value<Type = ArrayType> + crate::operations::manipulation::Scatter,
        V::DispatchDomain: Context<Type = ArrayType, Constant = TestArray, Operation = ArrayOperation<TestArray>>,
    {
        use crate::operations::manipulation::{ScatterDimensionNumbers, ScatterOperation, ScatterReductionKind};
        use crate::types::{DataType, Shape, Size};

        let indices_type = ArrayType::new(DataType::I32, Shape::new(vec![Size::Static(2), Size::Static(1)]));
        let indices = inputs[0].dispatch_domain().lift(TestArray::new(indices_type, vec![1.0, 3.0]))?;
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Add);
        Ok(vec![inputs[0].scatter(&indices, &inputs[1], &operation)?])
    }

    /// Binds `f(x, u) = scatter_add(x, [[1], [3]], u)` through the forward-mode dual context, the `jvp` twin of
    /// [`scatter_add_function`].
    fn scatter_add_jvp_function(inputs: Vec<ArrayJvpTracer>) -> Result<Vec<ArrayJvpTracer>, ProgramError> {
        use crate::operations::manipulation::{
            Scatter, ScatterDimensionNumbers, ScatterOperation, ScatterReductionKind,
        };
        use crate::types::{DataType, Shape, Size};

        let indices_type = ArrayType::new(DataType::I32, Shape::new(vec![Size::Static(2), Size::Static(1)]));
        let indices = inputs[0].context().lift(TestArray::new(indices_type, vec![1.0, 3.0]))?;
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Add);
        Ok(vec![inputs[0].scatter(&indices, &inputs[1], &operation)?])
    }

    #[test]
    fn test_array_forward_and_reverse_equivalent_for_scatter_add() {
        // f(x, u) = scatter_add(x, [[1], [3]], u): scatter-add is the identity in its operand and accumulates the
        // updates, so the tangent scatter-adds the operand and update tangents and the pullback carries the
        // operand cotangent through while gathering the update cotangent at the captured indices, matching `jvp` /
        // `vjp`.
        assert_array_forward_equivalent(
            scatter_add_jvp_function,
            scatter_add_function,
            vec![TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]), TestArray::vector(vec![10.0, 20.0])],
            vec![TestArray::vector(vec![0.5, -1.0, 2.0, 1.0]), TestArray::vector(vec![1.0, -0.5])],
        );
        assert_array_reverse_equivalent(
            scatter_add_function,
            scatter_add_function,
            vec![TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]), TestArray::vector(vec![10.0, 20.0])],
            vec![TestArray::vector(vec![2.5, 1.0, -0.5, 2.0])],
        );
    }

    #[test]
    fn test_array_forward_equivalent_for_reduce_max_and_min() {
        // f(x) = reduce_max(x, axis 0): the tangent is the masked sum `reduce_sum(mask * dx)`, where `mask`
        // selects the extremal coordinate, so the directional derivative equals `dx` at the argmax. The argmax mask is
        // staged capture-free as a `compare`/`mul`, and the result matches the established `jvp`.
        assert_array_forward_equivalent(
            |inputs| Ok(vec![inputs[0].reduce(&[0], ReductionKind::Max)]),
            |inputs| Ok(vec![inputs[0].reduce(&[0], ReductionKind::Max)]),
            vec![TestArray::vector(vec![1.0, 4.0, 2.0, 3.0])],
            vec![TestArray::vector(vec![0.5, -1.0, 2.0, 1.0])],
        );
        assert_array_forward_equivalent(
            |inputs| Ok(vec![inputs[0].reduce(&[0], ReductionKind::Min)]),
            |inputs| Ok(vec![inputs[0].reduce(&[0], ReductionKind::Min)]),
            vec![TestArray::vector(vec![4.0, 1.0, 2.0, 3.0])],
            vec![TestArray::vector(vec![0.5, -1.0, 2.0, 1.0])],
        );

        // Matrix reduction along axis 1 keeps the broadcast-back-and-mask geometry honest across a non-trivial output
        // shape; the per-row argmaxes pick distinct columns.
        assert_array_forward_equivalent(
            |inputs| Ok(vec![inputs[0].reduce(&[1], ReductionKind::Max)]),
            |inputs| Ok(vec![inputs[0].reduce(&[1], ReductionKind::Max)]),
            vec![TestArray::matrix(2, 3, vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0])],
            vec![TestArray::matrix(2, 3, vec![0.5, -1.0, 2.0, 1.0, -0.5, 0.25])],
        );
    }

    #[test]
    fn test_array_reverse_equivalent_for_reduce_max() {
        // f(x) = reduce_max(x, axis 0): the tangent program holds the argmax mask as a residual, so transposing
        // it routes the scalar cotangent to the extremal coordinate, matching the established `vjp`. The reduce-max
        // pullback is only wired on the direct-transpose path, so the equivalence is gated against the
        // forward tangent reproduced in reverse rather than the capture-based `vjp` (whose reduce-max transpose is not
        // implemented).
        let domain = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let context = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();

        let primals = vec![TestArray::vector(vec![1.0, 4.0, 2.0, 3.0])];
        let output_cotangents = vec![TestArray::scalar(2.5)];
        let (_, pullback, residuals) =
            vjp_direct(&domain, |inputs| Ok(vec![inputs[0].reduce(&[0], ReductionKind::Max)]), primals).unwrap();
        let mut inputs = output_cotangents;
        inputs.extend(residuals);
        let input_cotangents = pullback.interpret_in_context(&context, inputs).unwrap();
        // The argmax is coordinate 1, so the cotangent lands entirely there.
        assert_arrays_close(&input_cotangents, &[TestArray::vector(vec![0.0, 2.5, 0.0, 0.0])], "reduce-max reverse");
    }

    #[test]
    fn test_array_program_is_capture_free_and_has_expected_shape() {
        // f(x) = x . x over a vector has one primal input and one (scalar) primal output, so the program takes
        // two inputs (primal + tangent) and produces two outputs (primal + tangent).
        let domain = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let (_, primal_program) = NestedTracingContext::trace(
            domain.clone(),
            |inputs: Vec<ArrayTracer>| Ok(vec![inputs[0].dot(&inputs[0], &DotDimensionNumbers::inner_product())]),
            vec![TestArray::vector(vec![1.0, 2.0, 3.0]).r#type().into_owned()],
        )
        .unwrap();
        let primal_program = primal_program.into_simplified().unwrap();
        let jvp_program = primal_program.jvp().unwrap();
        assert_eq!(jvp_program.input_ids().len(), 2);
        assert_eq!(jvp_program.output_ids().len(), 2);

        // The crux: the program is expressed entirely in the primal `ArrayOperation` enum, so it carries no
        // captured-factor linear operation (no captured-factor linear dot operation carrying a residual factor). The tangent
        // dots are ordinary binary `Dot`s referencing primal and tangent SSA values directly. The type system proves
        // the absence of symbolic captures: the program's operation family is `ArrayOperation<TestArray>`, the ordinary
        // primal operation family, rather than a capture-keyed linear operation family. This binding documents that
        // distinction at the type level.
        let _capture_free: &Program<TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>> =
            &jvp_program;

        // Interpreting the program at primal x = [1, 2, 3] and tangent t = [1, 1, 1] reproduces the value
        // x . x = 14 and its directional derivative 2 * (x . t) = 2 * 6 = 12.
        let context = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let inputs = vec![TestArray::vector(vec![1.0, 2.0, 3.0]), TestArray::vector(vec![1.0, 1.0, 1.0])];
        let outputs = jvp_program.interpret_in_context(&context, inputs).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_close(&outputs[0].values, &[14.0], "array jvp primal");
        assert_close(&outputs[1].values, &[12.0], "array jvp tangent");
    }

    #[test]
    fn test_array_jvp_computes_analytic_forward_derivatives() {
        // The single `jvp` entry point runs the closure directly on duals; each block below asserts the hand-computed
        // primal and directional derivative for one of the array function shapes the pipeline-equivalence harness
        // used to exercise.
        let domain = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();

        // Elementwise: f(x) = x * sin(x) over a vector (Mul + Sin); tangent_i = (sin(x_i) + x_i cos(x_i)) * t_i.
        let x = [0.7, -1.2, 2.0];
        let t = [1.0, 0.5, -2.0];
        let (primals, tangents) = domain
            .jvp(
                |inputs: Vec<ArrayJvpTracer>| Ok(vec![inputs[0].clone() * inputs[0].sin()?]),
                vec![TestArray::vector(x.to_vec())],
                vec![TestArray::vector(t.to_vec())],
            )
            .unwrap();
        let expected_primals = x.iter().map(|x| x * x.sin()).collect::<Vec<_>>();
        let expected_tangents = x.iter().zip(t).map(|(x, t)| (x.sin() + x * x.cos()) * t).collect::<Vec<_>>();
        assert_close(&primals[0].values, &expected_primals, "elementwise x sin(x) primal");
        assert_close(&tangents[0].values, &expected_tangents, "elementwise x sin(x) tangent");

        // Inner product: f(x) = x . x (Dot product rule staging a left and a right Dot); the tangent is
        // 2 * (x . t) = 2 * (0.5 - 2 + 6) = 9.
        let (primals, tangents) = domain
            .jvp(
                |inputs: Vec<ArrayJvpTracer>| {
                    Ok(vec![inputs[0].dot(&inputs[0], &DotDimensionNumbers::inner_product())])
                },
                vec![TestArray::vector(vec![1.0, 2.0, 3.0])],
                vec![TestArray::vector(vec![0.5, -1.0, 2.0])],
            )
            .unwrap();
        assert_close(&primals[0].values, &[14.0], "inner-product primal");
        assert_close(&tangents[0].values, &[9.0], "inner-product tangent");

        // Select: f(x) = select(x > 0, 2x, 3x) over a vector, masking per element: the primal is [2, -3, 4] and the
        // tangent routes each element through the selected branch's slope, [2, 3, 2].
        let (primals, tangents) = domain
            .jvp(
                |inputs: Vec<ArrayJvpTracer>| {
                    let mask = inputs[0].compare(&inputs[0].zero_like(), ComparisonDirection::GreaterThan)?;
                    let on_true = inputs[0].clone() + inputs[0].clone();
                    let on_false = inputs[0].clone() + inputs[0].clone() + inputs[0].clone();
                    Ok(vec![Select::select(&mask, &on_true, &on_false)?])
                },
                vec![TestArray::vector(vec![1.0, -1.0, 2.0])],
                vec![TestArray::vector(vec![1.0, 1.0, 1.0])],
            )
            .unwrap();
        assert_close(&primals[0].values, &[2.0, -3.0, 4.0], "select primal");
        assert_close(&tangents[0].values, &[2.0, 3.0, 2.0], "select tangent");
    }

    /// Builds the flat scalar branch `x -> x * 2 + 1`, which linearizes with no residuals (its tangent `2 * dx`
    /// holds no primal-derived coefficient). Shared by the condition equivalence tests as the true branch.
    fn affine_branch() -> Program<TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>> {
        use crate::operations::math::{AddOperation, MulOperation};
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;
        use crate::types::DataType;

        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let two = builder.add_constant(TestArray::scalar(2.0));
        let scaled = builder.add_instruction(MulOperation, Vec::new(), vec![input, two]).unwrap()[0];
        let one = builder.add_constant(TestArray::scalar(1.0));
        let output = builder.add_instruction(AddOperation, Vec::new(), vec![scaled, one]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds the flat scalar branch `x -> sin(x)`, which linearizes with one residual (its tangent
    /// `cos(x) * dx` carries the primal-derived `cos(x)` coefficient). Shared by the condition equivalence tests as the
    /// false branch, so the two branches have asymmetric residual counts and exercise the residual join.
    fn sine_branch() -> Program<TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>> {
        use crate::operations::math::SinOperation;
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;
        use crate::types::DataType;

        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(SinOperation, Vec::new(), vec![input]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Stages `condition(predicate, x*2+1, sin(x))` over the closure inputs `[predicate, x]`, the shared body of every
    /// condition equivalence closure. The predicate is the scalar-boolean first input and carries no tangent; the
    /// scalar operand `x` flows into the selected branch.
    fn condition_function<V>(inputs: Vec<V>) -> Result<Vec<V>, ProgramError>
    where
        V: Value<Type = ArrayType>,
        V::DispatchDomain: Context<Type = ArrayType, Constant = TestArray, Operation = ArrayOperation<TestArray>>,
    {
        let condition = crate::operations::control_flow::ConditionOperation::new();
        inputs[0].dispatch_domain().bind(
            ArrayOperation::Condition(condition),
            vec![affine_branch(), sine_branch()],
            &[inputs[0].clone(), inputs[1].clone()],
        )
    }

    /// Asserts that the full JVP program for `condition_function` at `(predicate, x)` with tangent `(0, dx)` computes
    /// the analytic primal and tangent outputs.
    ///
    /// The condition's predicate is a scalar-boolean operand whose tangent input is dead (Boolean predicates have
    /// no tangent space), so the partial-evaluation split prunes it and the reassembling
    /// [`assert_array_forward_equivalent`] harness — which assumes every input tangent survives into the tangent
    /// sub-program — does not apply. Interpreting the whole JVP program directly at `(primals ++ tangents)`
    /// instead proves the rule end to end, exactly as the scalar front end verifies its Boolean-codomain rules.
    fn assert_condition_forward_equivalent_to_jvp(predicate: bool, x: f64, dx: f64) {
        use crate::types::DataType;

        let domain = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let predicate_value = TestArray::new(ArrayType::scalar(DataType::Boolean), vec![predicate as u8 as f64]);
        let predicate_tangent = TestArray::new(ArrayType::scalar(DataType::Boolean), vec![0.0]);
        let error = domain
            .jvp(
                |inputs: Vec<ArrayJvpTracer>| {
                    let condition = crate::operations::control_flow::ConditionOperation::new();
                    inputs[0].context().bind(
                        ArrayOperation::Condition(condition),
                        vec![affine_branch(), sine_branch()],
                        &[inputs[0].clone(), inputs[1].clone()],
                    )
                },
                vec![predicate_value.clone(), TestArray::scalar(x)],
                vec![predicate_tangent.clone(), TestArray::scalar(dx)],
            )
            .unwrap_err();
        assert_eq!(
            error,
            crate::differentiation::DifferentiationError::Program(ProgramError::MalformedProgram(
                "JVP input 0 has live tangent type bool[] but primal type bool[] has no tangent space".to_string(),
            )),
        );

        let reference_primals = vec![TestArray::scalar(if predicate { 2.0 * x + 1.0 } else { x.sin() })];
        let reference_tangents = vec![TestArray::scalar(if predicate { 2.0 * dx } else { x.cos() * dx })];

        let (_, primal_program) = NestedTracingContext::trace(
            domain.clone(),
            condition_function,
            vec![predicate_value.r#type().into_owned(), TestArray::scalar(x).r#type().into_owned()],
        )
        .unwrap();
        let primal_program = primal_program.into_simplified().unwrap();
        let jvp_program = primal_program.jvp().unwrap();

        // The program takes `(primals ++ tangents)` and produces `(primal_outputs ++ tangent_outputs)`.
        let context = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let inputs = vec![predicate_value, TestArray::scalar(x), predicate_tangent, TestArray::scalar(dx)];
        let outputs = jvp_program.interpret_in_context(&context, inputs).unwrap();
        let (primal_outputs, tangent_outputs) = outputs.split_at(reference_primals.len());
        assert_arrays_close(primal_outputs, &reference_primals, "condition forward primal");
        assert_arrays_close(tangent_outputs, &reference_tangents, "condition forward tangent");
    }

    #[test]
    fn test_array_forward_equivalent_to_jvp_for_condition() {
        // f(predicate, x) = condition(predicate, x*2+1, sin(x)). The branches linearize with asymmetric residual
        // counts (the affine branch has none, the sine branch has one), so this exercises the residual join on both the
        // primal and the tangent side of the staged condition.

        // Predicate true: the affine branch is taken, so the directional derivative is `2 * dx`.
        assert_condition_forward_equivalent_to_jvp(true, 0.7, 1.5);

        // Predicate false: the sine branch is taken, so the directional derivative is `cos(x) * dx`.
        assert_condition_forward_equivalent_to_jvp(false, 0.7, 1.5);
    }

    /// Builds the cumulative-product scan body `[carry, x] -> [carry * x, carry * x]`. Its single `Mul` linearizes
    /// with two per-iteration residuals (the primal `carry` and `x`), so the scan rule stacks both into
    /// `[length, ...]` residual outputs and re-keys two scan-local captures on reverse. Shared by the scan equivalence
    /// tests.
    fn scan_product_body() -> Program<TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>> {
        use crate::operations::math::MulOperation;
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;
        use crate::types::DataType;

        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let carry = builder.add_input(ArrayType::scalar(DataType::F64));
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let product = builder.add_instruction(MulOperation, Vec::new(), vec![carry, x]).unwrap()[0];
        builder
            .build(vec![product, product], vec![Placeholder, Placeholder], vec![Placeholder, Placeholder])
            .unwrap()
    }

    /// Stages a length-3 cumulative-product `scan(init, xs)` over the closure inputs `[init, xs]`, the shared body of
    /// every scan equivalence closure. The scalar `init` is the loop-carried state and the `[3]` vector `xs` is the
    /// scanned input; the scan returns the final carry and the `[3]` stack of running products.
    fn scan_function<V>(inputs: Vec<V>) -> Result<Vec<V>, ProgramError>
    where
        V: Value<Type = ArrayType>,
        V::DispatchDomain: Context<Type = ArrayType, Constant = TestArray, Operation = ArrayOperation<TestArray>>,
    {
        let scan = crate::operations::control_flow::ScanOperation::new(1, 3);
        inputs[0].dispatch_domain().bind(
            ArrayOperation::Scan(scan),
            vec![scan_product_body()],
            &[inputs[0].clone(), inputs[1].clone()],
        )
    }

    /// Binds the same length-3 cumulative-product `scan(init, xs)` through the forward-mode dual context, the `jvp`
    /// twin of [`scan_function`].
    fn scan_jvp_function(inputs: Vec<ArrayJvpTracer>) -> Result<Vec<ArrayJvpTracer>, ProgramError> {
        let scan = crate::operations::control_flow::ScanOperation::new(1, 3);
        inputs[0].context().bind(
            ArrayOperation::Scan(scan),
            vec![scan_product_body()],
            &[inputs[0].clone(), inputs[1].clone()],
        )
    }

    #[test]
    fn test_array_forward_equivalent_to_jvp_for_scan() {
        use crate::types::{DataType, Shape, Size};

        // f(init, xs) = scan(init, xs) cumulative product. Both operand tangents are live (a scan has no Boolean
        // operand to prune), so the reassembling forward harness applies directly: the rule stages a
        // residual-extended primal scan storing the per-iteration `carry`/`x` residual stacks and a tangent scan
        // consuming them as extra scanned inputs.
        let stacked_f64 = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));

        // A unit tangent on `init` propagates the cumulative product's derivative; over `xs = [2, 3, 4]` from
        // `init = 1` the running products are `[2, 6, 24]` and `d/d(init)` matches them.
        assert_array_forward_equivalent(
            scan_jvp_function,
            scan_function,
            vec![TestArray::scalar(1.0), TestArray::vector(vec![2.0, 3.0, 4.0])],
            vec![TestArray::scalar(1.0), TestArray::new(stacked_f64.clone(), vec![0.0, 0.0, 0.0])],
        );

        // A unit tangent on `xs[1]` exercises the per-iteration residual stacking: only iterations at or past
        // iteration 1 depend on `x1`, so the stacked-output tangent is `[0, 2, 8]`.
        assert_array_forward_equivalent(
            scan_jvp_function,
            scan_function,
            vec![TestArray::scalar(1.0), TestArray::vector(vec![2.0, 3.0, 4.0])],
            vec![TestArray::scalar(0.0), TestArray::new(stacked_f64, vec![0.0, 1.0, 0.0])],
        );
    }

    #[test]
    fn test_array_reverse_equivalent_to_vjp_for_scan() {
        // f(init, xs) = scan(init, xs) cumulative product. Reverse mode re-keys the tangent scan into a captured-stack
        // linear scan whose body folds the trailing residual-slice scanned inputs into scan-local captures, then the
        // single outer transpose flips `reverse` and transposes the body — pairing cotangent iteration `i` with
        // residual stack iteration `i`. Both operand cotangents are real (no predicate to prune), so the pullback
        // matches `vjp`'s cotangents one-to-one.
        assert_array_reverse_equivalent(
            scan_function,
            scan_function,
            vec![TestArray::scalar(1.0), TestArray::vector(vec![2.0, 3.0, 4.0])],
            vec![TestArray::scalar(1.0), TestArray::vector(vec![1.0, 1.0, 1.0])],
        );
    }

    /// Computes `reshape(x, [4]) + broadcast(reduce_sum(x, axes 0,1), [4])` for a 2x2 matrix input, exercising a
    /// reshape and a broadcast of a reduced scalar feeding an elementwise add. Shared by the forward and reverse tests.
    fn reshape_broadcast<V>(input: &V) -> V
    where
        V: Value<Type = ArrayType> + Reshape + Reduce + Broadcast + std::ops::Add<Output = V>,
    {
        use crate::types::{ArrayType, DataType, Shape, Size};

        let flat = input.reshape(Shape::new(vec![Size::Static(4)])).unwrap();
        let total = input.reduce(&[0, 1], ReductionKind::Sum);
        let broadcast_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(4)]));
        let broadcast_total = total.broadcast(broadcast_type, &[]).unwrap();
        flat + broadcast_total
    }

    /// Computes the [`reshape_broadcast`] body over forward-mode duals, the `jvp` twin of that helper.
    fn reshape_broadcast_jvp(input: &ArrayJvpTracer) -> Result<ArrayJvpTracer, ProgramError> {
        use crate::types::{ArrayType, DataType, Shape, Size};

        let flat = input.reshape(Shape::new(vec![Size::Static(4)]))?;
        let total = input.reduce(&[0, 1], ReductionKind::Sum);
        let broadcast_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(4)]));
        let broadcast_total = total.broadcast(broadcast_type, &[])?;
        Ok(flat + broadcast_total)
    }

    /// Control-flow de-risking: the payload-free [`ArrayOperation`] enum must satisfy the direct forward-mode and
    /// partial-evaluation bounds used by program transforms. Higher-order rules recurse through instruction-scoped
    /// drivers, so the bounds below stay finite even though condition, while, and scan regions contain further
    /// `ArrayOperation` instructions.
    #[test]
    fn array_operation_satisfies_the_direct_program_bounds() {
        use crate::differentiation::DifferentiableOperation;
        use crate::operations::constants::ZeroOperation;
        use crate::partial::{PartialEvaluationContext, PartiallyEvaluatableOperation};
        use crate::programs::Value;
        use crate::tracing::TracingContext;

        fn assert_program_bounds<V: Value, O>()
        where
            O: Operation<V::Type>
                + From<ZeroOperation<V::Type>>
                + DifferentiableOperation<TracingContext<V, O>>
                + PartiallyEvaluatableOperation<TracingContext<V, O>>
                + DifferentiableOperation<PartialEvaluationContext<TracingContext<V, O>>>,
        {
        }
        assert_program_bounds::<TestArray, ArrayOperation<TestArray>>();
    }

    /// Builds the `while (x < threshold) { x = x * x }` loop with the provided semantic iteration bound. Squaring
    /// captures the loop state itself as a per-iteration residual, so the rule stacks the residual into a
    /// `[bound, ...]` stack and the masked tangent scan is exercised on the iterations beyond the actual trip count.
    fn bounded_squaring_while_operation(
        threshold: f64,
        bound: usize,
    ) -> (
        crate::operations::control_flow::WhileOperation,
        Vec<Program<TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>>>,
    ) {
        use crate::operations::compare::CompareOperation;
        use crate::operations::control_flow::WhileOperation;
        use crate::operations::math::MulOperation;
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;
        use crate::types::DataType;

        let scalar_f64 = ArrayType::scalar(DataType::F64);

        let mut condition_builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let condition_state = condition_builder.add_input(scalar_f64.clone());
        let threshold = condition_builder.add_constant(TestArray::scalar(threshold));
        let predicate = condition_builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::LessThan),
                Vec::new(),
                vec![condition_state, threshold],
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let state = body_builder.add_input(scalar_f64);
        let squared = body_builder.add_instruction(MulOperation, Vec::new(), vec![state, state]).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![squared], vec![Placeholder], vec![Placeholder])
            .unwrap();

        (WhileOperation::new().with_iteration_bound(bound).unwrap(), vec![condition, body])
    }

    /// Stages the bounded squaring `while (x < 100) { x = x * x }` loop (iteration bound `5`) over the single closure
    /// input `[x]`, the shared body of the bounded-while equivalence closures. Starting from `x = 2` the loop runs the
    /// actual trip count `3` (`2 -> 4 -> 16 -> 256`), so iterations `3` and `4` are inactive and the validity mask is
    /// exercised; the loop computes `x^8` and its derivative is `8 * x^7`.
    fn bounded_while_function<V>(inputs: Vec<V>) -> Result<Vec<V>, ProgramError>
    where
        V: Value<Type = ArrayType>,
        V::DispatchDomain: Context<Type = ArrayType, Constant = TestArray, Operation = ArrayOperation<TestArray>>,
    {
        let (while_operation, while_regions) = bounded_squaring_while_operation(100.0, 5);
        inputs[0]
            .dispatch_domain()
            .bind(ArrayOperation::While(while_operation), while_regions, &[inputs[0].clone()])
    }

    /// Binds the same bounded squaring `while` through the forward-mode dual context, the `jvp` twin of
    /// [`bounded_while_function`].
    fn bounded_while_jvp_function(inputs: Vec<ArrayJvpTracer>) -> Result<Vec<ArrayJvpTracer>, ProgramError> {
        let (while_operation, while_regions) = bounded_squaring_while_operation(100.0, 5);
        inputs[0]
            .context()
            .bind(ArrayOperation::While(while_operation), while_regions, &[inputs[0].clone()])
    }

    #[test]
    fn test_array_forward_equivalent_to_jvp_for_bounded_while() {
        // f(x) = (((x^2)^2)^2) = x^8 via the bounded squaring loop. The rule stages an augmented primal while
        // that stores the per-iteration `x` residual into a `[5, ...]` stack plus a `[5]` validity mask, then a
        // length-5 masked tangent scan whose iterations beyond the actual trip count (3) pass the carried tangent
        // through unchanged. At x = 2 the loop value is 256 and the directional derivative is `8 * x^7 = 1024`.
        assert_array_forward_equivalent(
            bounded_while_jvp_function,
            bounded_while_function,
            vec![TestArray::scalar(2.0)],
            vec![TestArray::scalar(1.0)],
        );
    }

    #[test]
    fn test_array_reverse_equivalent_to_vjp_for_bounded_while() {
        // Reverse mode through the bounded while re-keys the masked tangent scan into a captured-stack linear scan with
        // no while-specific transpose code: the per-iteration `select` over the mask-iteration capture transposes so
        // the inactive iterations pass cotangents through unchanged. A unit output cotangent pulls back to
        // `8 * x^7 = 1024`.
        assert_array_reverse_equivalent(
            bounded_while_function,
            bounded_while_function,
            vec![TestArray::scalar(2.0)],
            vec![TestArray::scalar(1.0)],
        );
    }

    #[test]
    fn test_array_unbounded_while_has_no_differentiation_rule() {
        use crate::operations::compare::CompareOperation;
        use crate::operations::control_flow::WhileOperation;
        use crate::operations::math::MulOperation;
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;
        use crate::types::DataType;

        // An unbounded while loop (no iteration bound) has no statically shaped residual stack and no transposable
        // forward-mode form, so the front end reports `UnsupportedOperation` rather than mis-evaluating.
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let condition_state = condition_builder.add_input(scalar_f64.clone());
        let threshold = condition_builder.add_constant(TestArray::scalar(100.0));
        let predicate = condition_builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::LessThan),
                Vec::new(),
                vec![condition_state, threshold],
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let state = body_builder.add_input(scalar_f64);
        let squared = body_builder.add_instruction(MulOperation, Vec::new(), vec![state, state]).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![squared], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let while_operation = WhileOperation::new();

        let domain = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let result = array_linearize(
            &domain,
            move |inputs| {
                inputs[0].context().stage_operation(
                    ArrayOperation::While(while_operation),
                    vec![condition.clone(), body.clone()],
                    &[&inputs[0]],
                )
            },
            vec![TestArray::scalar(2.0)],
        );
        match result {
            Err(crate::differentiation::DifferentiationError::Program(ProgramError::UnsupportedOperation {
                ..
            })) => {}
            Err(other) => panic!("expected an UnsupportedOperation error for an unbounded while but got {other:?}"),
            Ok(_) => panic!("expected an UnsupportedOperation error for an unbounded while but it linearized"),
        }
    }

    /// Builds the single-input primal program `x -> sin(x)` over a scalar, the primal half of the deliberately wrong
    /// custom-JVP oracle shared with the [`custom_derivatives`](crate::tracing_v2::operations::custom_derivatives)
    /// tests.
    fn custom_jvp_sin_primal() -> Program<TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>> {
        use crate::operations::math::SinOperation;
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;
        use crate::types::DataType;

        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(SinOperation, Vec::new(), vec![input]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds the deliberately wrong custom-JVP rule program `(x, dx) -> (sin(x), 2 * cos(x) * dx)`, detectably
    /// different from the true `cos(x) * dx`, so a passing equivalence proves the spliced rule (and not the primal
    /// body) governs both forward and reverse mode. Shared in shape with the `custom_derivatives` tests.
    fn custom_jvp_sin_doubled_rule() -> Program<TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>> {
        use crate::operations::math::{CosOperation, MulOperation, SinOperation};
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;
        use crate::types::DataType;

        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let dx = builder.add_input(ArrayType::scalar(DataType::F64));
        let y = builder.add_instruction(SinOperation, Vec::new(), vec![x]).unwrap()[0];
        let cosine = builder.add_instruction(CosOperation, Vec::new(), vec![x]).unwrap()[0];
        let two = builder.add_constant(TestArray::scalar(2.0));
        let scaled = builder.add_instruction(MulOperation, Vec::new(), vec![cosine, two]).unwrap()[0];
        let tangent = builder.add_instruction(MulOperation, Vec::new(), vec![scaled, dx]).unwrap()[0];
        builder
            .build(vec![y, tangent], vec![Placeholder, Placeholder], vec![Placeholder, Placeholder])
            .unwrap()
    }

    /// Stages `custom_jvp(sin, doubled_rule)(x)` over the single closure input `[x]`, the shared body of the custom-JVP
    /// equivalence closures. Differentiation replays the user-supplied (doubled) rule instead of the primal `sin` body.
    fn custom_jvp_function<V>(inputs: Vec<V>) -> Result<Vec<V>, ProgramError>
    where
        V: Value<Type = ArrayType>,
        V::DispatchDomain: Context<Type = ArrayType, Constant = TestArray, Operation = ArrayOperation<TestArray>>,
    {
        let operation = crate::tracing_v2::operations::custom_derivatives::CustomJvpOperation::new();
        inputs[0].dispatch_domain().bind(
            ArrayOperation::CustomJvp(operation),
            vec![custom_jvp_sin_primal(), custom_jvp_sin_doubled_rule()],
            &[inputs[0].clone()],
        )
    }

    /// Binds the same `custom_jvp(sin, doubled_rule)` call through the forward-mode dual context, the `jvp` twin of
    /// [`custom_jvp_function`].
    fn custom_jvp_jvp_function(inputs: Vec<ArrayJvpTracer>) -> Result<Vec<ArrayJvpTracer>, ProgramError> {
        let operation = crate::tracing_v2::operations::custom_derivatives::CustomJvpOperation::new();
        inputs[0].context().bind(
            ArrayOperation::CustomJvp(operation),
            vec![custom_jvp_sin_primal(), custom_jvp_sin_doubled_rule()],
            &[inputs[0].clone()],
        )
    }

    #[test]
    fn test_array_forward_equivalent_to_jvp_for_custom_jvp() {
        // f(x) = custom_jvp(sin, doubled_rule)(x). The spliced JVP program maps `(x, dx) -> (sin(x), 2*cos(x)*dx)`, so
        // the tangent is the deliberately doubled `2*cos(x)*dx` rather than the primal body's `cos(x)*dx`. The
        // single operand tangent is live, so the reassembling forward harness applies directly.
        assert_array_forward_equivalent(
            custom_jvp_jvp_function,
            custom_jvp_function,
            vec![TestArray::scalar(0.7)],
            vec![TestArray::scalar(1.5)],
        );
    }

    #[test]
    fn test_array_reverse_equivalent_to_vjp_for_custom_jvp() {
        // Reverse mode transposes the spliced (straight-line) JVP program, so the doubled derivative carries over: the
        // pullback of a cotangent is `2*cos(x)*cotangent`, matching `vjp` of the same custom-JVP call.
        assert_array_reverse_equivalent(
            custom_jvp_function,
            custom_jvp_function,
            vec![TestArray::scalar(0.7)],
            vec![TestArray::scalar(2.5)],
        );
    }

    /// Computes `f(x) = u * sin(u)` with `u = x · x` (inner product), whose linearization residuals span a dot, a sine,
    /// and the sine rule's `cos(u)` factor — the body shared with the
    /// [`rematerialization`](crate::tracing_v2::rematerialization) tests. Reused here as a rematerialized region; it is
    /// generic over the value type so it serves both as the un-rematerialized reference closure (over [`ArrayTracer`])
    /// and as the rematerialized body (over a [`DomainTracer`](crate::tracing::DomainTracer)).
    fn remat_dot_sine_body<V>(input: V) -> V
    where
        V: Clone + Sin + Dot + std::ops::Mul<Output = V>,
    {
        let u = input.dot(&input, &DotDimensionNumbers::inner_product());
        u.clone() * u.sin().unwrap()
    }

    /// Derives the [`RematerializeOperation`](crate::tracing_v2::RematerializeOperation) for
    /// `rematerialize(remat_dot_sine_body)` at a length-2 vector input under the default `NothingSaveable` policy, then
    /// stages it directly so the equivalence closures exercise the concrete operation (mirroring how the scan and
    /// condition tests stage their operations directly rather than through the wrapper). The derivation is run
    /// once through a `TracingContext::trace` and the staged operation is extracted from the resulting
    /// single-instruction program.
    fn rematerialize_function(inputs: Vec<ArrayTracer>) -> Result<Vec<ArrayTracer>, ProgramError> {
        use crate::tracing::{DomainTracer, Trace};
        use crate::tracing_v2::operations::ArrayOperation;
        use crate::tracing_v2::rematerialize;
        use crate::types::{DataType, Shape, Size};

        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]));
        let function = rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(
            |x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| Ok(remat_dot_sine_body(x)),
        );
        let (_, program) =
            EagerContext::<TestArray, ArrayOperation<TestArray>>::trace(|x| function.call(x), vector_type).unwrap();
        let instruction = &program.instructions()[0];
        let ArrayOperation::Rematerialize(operation) = instruction.operation() else {
            panic!("rematerialize should stage a rematerialize call");
        };
        let operation_regions = instruction
            .regions()
            .iter()
            .map(|region| program.region_ref(*region).map(|region| region.to_program()))
            .collect::<Result<Vec<_>, _>>()?;
        inputs[0]
            .context()
            .stage_operation(ArrayOperation::Rematerialize(*operation), operation_regions, &[&inputs[0]])
    }

    #[test]
    fn test_array_forward_equivalent_to_jvp_for_rematerialize() {
        // f(x) = rematerialize(u * sin(u)) with u = x · x, under the default `NothingSaveable` policy. The rule
        // splices the derived forward program (recovering the region inputs as the residual tail) and the derived
        // tangent program (which recomputes the interior residuals from that tail), so the forward reproduces
        // `jvp` of the un-rematerialized body. The single operand tangent is live, so the reassembling harness applies.
        assert_array_forward_equivalent(
            |inputs| Ok(vec![remat_dot_sine_body(inputs[0].clone())]),
            rematerialize_function,
            vec![TestArray::vector(vec![0.5, 1.5])],
            vec![TestArray::vector(vec![1.0, 0.0])],
        );
    }

    #[test]
    fn test_array_reverse_equivalent_to_vjp_for_rematerialize() {
        // Reverse mode transposes the spliced recompute-and-pushforward tangent program, so the pullback matches
        // `vjp` of the un-rematerialized body one-to-one (the rematerialization boundary is a forward-pass memory
        // tradeoff with no effect on the differentiated result).
        assert_array_reverse_equivalent(
            |inputs| Ok(vec![remat_dot_sine_body(inputs[0].clone())]),
            rematerialize_function,
            vec![TestArray::vector(vec![0.5, 1.5])],
            vec![TestArray::scalar(1.0)],
        );
    }

    /// Builds the single-input primal program `x -> sin(x)` over a scalar, the primal half of the deliberately wrong
    /// custom-VJP oracle shared with the [`custom_derivatives`](crate::tracing_v2::operations::custom_derivatives)
    /// tests.
    fn custom_vjp_sin_primal() -> Program<TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>> {
        use crate::operations::math::SinOperation;
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;
        use crate::types::DataType;

        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(SinOperation, Vec::new(), vec![input]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds the forward program `x -> (sin(x), cos(x))`, exposing `cos(x)` as the single residual consumed by the
    /// backward program.
    fn custom_vjp_sin_forward() -> Program<TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>> {
        use crate::operations::math::{CosOperation, SinOperation};
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;
        use crate::types::DataType;

        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let y = builder.add_instruction(SinOperation, Vec::new(), vec![x]).unwrap()[0];
        let residual = builder.add_instruction(CosOperation, Vec::new(), vec![x]).unwrap()[0];
        builder.build(vec![y, residual], vec![Placeholder], vec![Placeholder, Placeholder]).unwrap()
    }

    /// Builds the deliberately wrong backward program `(residual, cotangent) -> 3 * residual * cotangent`, detectably
    /// different from the true `cos(x) * cotangent`, so a passing equivalence proves the carrier actually replays the
    /// user backward program (a folded zero or the primal derivative would give a different, wrong answer). Shared in
    /// shape with the `custom_derivatives` tests.
    fn custom_vjp_sin_tripled_backward() -> Program<TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>>
    {
        use crate::operations::math::MulOperation;
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;
        use crate::types::DataType;

        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let residual = builder.add_input(ArrayType::scalar(DataType::F64));
        let cotangent = builder.add_input(ArrayType::scalar(DataType::F64));
        let three = builder.add_constant(TestArray::scalar(3.0));
        let scaled = builder.add_instruction(MulOperation, Vec::new(), vec![residual, three]).unwrap()[0];
        let gradient = builder.add_instruction(MulOperation, Vec::new(), vec![scaled, cotangent]).unwrap()[0];
        builder.build(vec![gradient], vec![Placeholder, Placeholder], vec![Placeholder]).unwrap()
    }

    /// Stages `custom_vjp(sin, forward, tripled_backward)(x)` over the single closure input `[x]`, the shared body of
    /// the custom-VJP equivalence closures. Reverse mode replays the user-supplied (tripled) backward rule on the
    /// forward program's residuals.
    fn custom_vjp_function<V>(inputs: Vec<V>) -> Result<Vec<V>, ProgramError>
    where
        V: Value<Type = ArrayType>,
        V::DispatchDomain: Context<Type = ArrayType, Constant = TestArray, Operation = ArrayOperation<TestArray>>,
    {
        let operation = crate::tracing_v2::operations::custom_derivatives::CustomVjpOperation::new();
        inputs[0].dispatch_domain().bind(
            ArrayOperation::CustomVjp(operation),
            vec![custom_vjp_sin_primal(), custom_vjp_sin_forward(), custom_vjp_sin_tripled_backward()],
            &[inputs[0].clone()],
        )
    }

    #[test]
    fn test_array_direct_transpose_equivalent_to_vjp_for_custom_vjp() {
        // f(x) = custom_vjp(sin, forward, tripled_backward)(x). The forward splices the forward program and
        // stages one opaque `CustomVjpTangent` carrier over `[dx, residual=cos(x)]`; the direct transpose replays the
        // tripled backward program on `[residual, cotangent]`, yielding `3*cos(x)*cotangent`. This must equal `vjp` of
        // the same custom-VJP call, proving the carrier actually runs `backward` (a folded zero tangent would silently
        // give a wrong zero gradient, and the tripled oracle would also diverge from the primal `cos(x)*cotangent`).
        //
        // A re-key reverse path is intentionally not exercised here: it would need to re-key the
        // primal-enum `CustomVjpTangent` carrier into the linear operation family, which has no such variant. The
        // carrier exists precisely so the direct-transpose path can keep the tangent program in the primal enum.
        let domain = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let context = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let primals = vec![TestArray::scalar(0.7)];
        let output_cotangents = vec![TestArray::scalar(2.5)];

        let (_, reference_pullback) = domain.vjp(custom_vjp_function, primals.clone()).unwrap();
        let (reference_pullback, reference_residuals) = reference_pullback.into_parts();
        let mut reference_inputs = output_cotangents.clone();
        reference_inputs.extend(reference_residuals);
        let reference_cotangents = reference_pullback.interpret_in_context(&context, reference_inputs).unwrap();

        let (_, direct_pullback, residuals) = vjp_direct(&domain, custom_vjp_function, primals).unwrap();
        let mut direct_inputs = output_cotangents;
        direct_inputs.extend(residuals);
        let direct_cotangents = direct_pullback.interpret_in_context(&context, direct_inputs).unwrap();

        assert_arrays_close(&direct_cotangents, &reference_cotangents, "direct-transpose vs vjp cotangent");
        // The tripled oracle gives `3 * cos(0.7) * 2.5`; assert the concrete value so a silently-zero or primal-rule
        // gradient is caught even if `vjp` itself regressed.
        assert_close(&direct_cotangents[0].values, &[3.0 * 0.7f64.cos() * 2.5], "direct-transpose tripled gradient");
    }

    #[test]
    fn test_array_forward_through_custom_vjp_is_rejected() {
        // `custom_vjp` is reverse-mode-only. The forward builds a tangent program containing the opaque
        // `CustomVjpTangent` carrier, but interpreting that tangent program (which forward mode does) replays the
        // carrier, whose interpretation rejects forward mode with the canonical reverse-only error rather than
        // silently producing a wrong tangent.
        use crate::programs::types::TypeError;

        let domain = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        match domain.jvp(
            |inputs: Vec<ArrayJvpTracer>| {
                let operation = crate::tracing_v2::operations::custom_derivatives::CustomVjpOperation::new();
                inputs[0].context().bind(
                    ArrayOperation::CustomVjp(operation),
                    vec![custom_vjp_sin_primal(), custom_vjp_sin_forward(), custom_vjp_sin_tripled_backward()],
                    &[inputs[0].clone()],
                )
            },
            vec![TestArray::scalar(0.7)],
            vec![TestArray::scalar(1.5)],
        ) {
            Err(crate::differentiation::DifferentiationError::Program(ProgramError::Type(TypeError { message })))
                if message.starts_with("custom_vjp does not support forward-mode differentiation") => {}
            Err(other) => panic!("expected the reverse-only TypeError from the forward but got {other:?}"),
            Ok(_) => panic!("expected the forward through custom_vjp to be rejected but it succeeded"),
        }
    }

    /// Builds the bounded `while (x < threshold) { x = x + x }` doubling loop over a scalar `f64` state with the given
    /// semantic iteration bound, the per-item body of the vmapped masked-while test. The predicate is per batch
    /// item, so vmapping this loop stages a masked bounded `while` whose augmented state carries a Boolean validity
    /// mask whose tangent is structurally zero.
    fn bounded_doubling_while_operation(
        threshold: f64,
        bound: usize,
    ) -> (WhileOperation, Vec<Program<TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>>>) {
        use crate::operations::math::AddOperation;
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;
        use crate::types::DataType;

        let scalar_f64 = ArrayType::scalar(DataType::F64);

        let mut condition_builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let condition_state = condition_builder.add_input(scalar_f64.clone());
        let threshold = condition_builder.add_constant(TestArray::scalar(threshold));
        let predicate = condition_builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::LessThan),
                Vec::new(),
                vec![condition_state, threshold],
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let state = body_builder.add_input(scalar_f64);
        let doubled = body_builder.add_instruction(AddOperation, Vec::new(), vec![state, state]).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();

        (WhileOperation::new().with_iteration_bound(bound).unwrap(), vec![condition, body])
    }

    /// Stages the vmapped per-item bounded doubling `while` over the single closure input `[x]`, the shared body of the
    /// masked-while equivalence test. Batching a per-item predicate stages one masked bounded `while` over the
    /// augmented state `[f64[N] value, bool[N] mask]`, so fuse-linearizing it drives the Gap B path: the Boolean mask
    /// item's tangent is structurally zero, so partial evaluation prunes that tangent input and the linearization
    /// must restore the canonical `[carry_tangents..., residuals...]` arity the bounded-`while` rule assumes.
    fn batched_bounded_while_function(inputs: Vec<ArrayTracer>) -> Result<Vec<ArrayTracer>, ProgramError> {
        use crate::batching::{Batch, BatchAxis};

        let context = inputs[0].context().clone();
        let mapped = Batch::batch(
            &context,
            |item| {
                let (while_operation, while_regions) = bounded_doubling_while_operation(8.0, 5);
                let mut outputs =
                    item.context().bind(ArrayOperation::While(while_operation), while_regions, &[item.clone()])?;
                Ok(outputs.remove(0))
            },
            inputs[0].clone(),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )?;
        Ok(vec![mapped])
    }

    #[test]
    fn test_array_masked_bounded_while_restores_pruned_mask_tangent() {
        // Gap B: a vmapped per-item bounded `while` stages a masked bounded loop whose augmented state carries a
        // `bool[3]` validity mask. The mask item's tangent is structurally zero, so partial evaluation prunes that
        // tangent input from the unknown sub-program; the linearization restores the canonical
        // `[carry_tangents..., residuals...]` arity the bounded-while rule assumes, so the staged augmented while
        // plus masked tangent scan composes instead of tripping the rule's input-count check. Batch items [1, 5, 9]
        // double 3, 1, and 0 times under the threshold `x < 8`, so the primal is [8, 10, 9] and the per-item tangent
        // scale is `2^iterations = [8, 2, 1]`.
        let domain = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let primals = vec![TestArray::vector(vec![1.0, 5.0, 9.0])];
        let tangents = vec![TestArray::vector(vec![1.0, 1.0, 1.0])];

        let linearization = array_linearize(&domain, batched_bounded_while_function, primals.clone()).unwrap();
        let context = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();

        // The known side computes the primal outputs followed by the residuals; its primal half is [8, 10, 9].
        let mut known_outputs = linearization.primal().interpret_in_context(&context, primals).unwrap();
        let residuals = known_outputs.split_off(known_outputs.len() - linearization.residual_count());
        assert_eq!(known_outputs.len(), 1, "expected one primal output");
        assert_close(&known_outputs[0].values, &[8.0, 10.0, 9.0], "masked-while primal");

        // The unknown (tangent) side now presents the canonical arity, so the masked tangent scan interprets to the
        // per-item doubling scales [8, 2, 1].
        let mut tangent_inputs = tangents;
        tangent_inputs.extend(residuals);
        let tangent_outputs = linearization.tangent().interpret_in_context(&context, tangent_inputs).unwrap();
        assert_eq!(tangent_outputs.len(), 1, "expected one tangent output");
        assert_close(&tangent_outputs[0].values, &[8.0, 2.0, 1.0], "masked-while tangent");

        // Reverse mode exercises the same restored arity on the pullback side. The direct-transpose path transposes the
        // tangent sub-program in the primal operation enum through `transpose_with_respect_to`, which carries the
        // structurally zero `bool[3]` mask state as a linear-zero carry without the re-key path's leading-linear
        // operand heuristic; it yields per-item gradients equal to the doubling scales.
        let reverse_primals = vec![TestArray::vector(vec![1.0, 5.0, 9.0])];
        let (primal_outputs, pullback, residuals) =
            vjp_direct(&domain, batched_bounded_while_function, reverse_primals).unwrap();
        assert_close(&primal_outputs[0].values, &[8.0, 10.0, 9.0], "masked-while vjp primal");
        let mut pullback_inputs = vec![TestArray::vector(vec![1.0, 1.0, 1.0])];
        pullback_inputs.extend(residuals);
        let cotangents = pullback.interpret_in_context(&context, pullback_inputs).unwrap();
        assert_close(&cotangents[0].values, &[8.0, 2.0, 1.0], "masked-while vjp cotangent");
    }

    #[test]
    fn test_array_vjp_stages_into_an_enclosing_trace() {
        use crate::parameters::Placeholder;
        use crate::tracing::DomainTracingContext;

        // Reverse-mode-under-tracing duality for array programs: running `vjp` against an
        // enclosing array `TracingContext` (whose values are tracers, while its constants stay concrete `TestArray`s)
        // must produce a tracer-valued pullback that splices into the enclosing trace, recovering the residuals as
        // enclosing-trace tracers. We trace the elementwise nonlinear `f(x) = x * sin(x)` over a vector under an outer
        // trace, interpret the tracer-valued pullback at an outer cotangent tracer to stage the backward pass into the
        // trace, then interpret the staged program eagerly and assert the input cotangent equals the established `vjp`
        // pullback at the same point.
        let domain = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let vector_type = TestArray::vector(vec![0.0; 3]).r#type;
        let outer_context = DomainTracingContext::<EagerContext<TestArray, ArrayOperation<TestArray>>>::new();
        let outer_builder = outer_context.builder().clone();

        // The outer trace's input is the primal `x` followed by the output cotangent, so interpreting the staged
        // program at `(x, cotangent)` mirrors seeding the established `vjp` pullback with the same cotangent.
        let primal_x = outer_context.input(vector_type.clone());
        let cotangent = outer_context.input(vector_type.clone());

        let (primal_outputs, pullback) = outer_context
            .vjp(
                |inputs: Vec<
                    LinearizationTracer<DomainTracingContext<EagerContext<TestArray, ArrayOperation<TestArray>>>>,
                >| { Ok(vec![inputs[0].clone() * inputs[0].sin()?]) },
                vec![primal_x],
            )
            .unwrap();
        let (pullback, residuals) = pullback.into_parts();
        assert_eq!(primal_outputs.len(), 1);

        // The pullback is genuinely tracer-valued, so interpreting it through the enclosing `TracingContext` splices
        // the backward pass into the outer trace. It consumes the residuals as ordinary inputs (themselves outer
        // tracers recovered from the primal replay), so it is interpreted at `[cotangent, residuals]`.
        let mut pullback_inputs = vec![cotangent];
        pullback_inputs.extend(residuals);
        let input_cotangents = pullback.interpret_in_context(&outer_context, pullback_inputs).unwrap();
        assert_eq!(input_cotangents.len(), 1);

        // Build the staged outer program over its two inputs `(x, cotangent)`, producing the staged input cotangent,
        // then interpret it eagerly at a sample point.
        let output_atoms = vec![input_cotangents[0].atom_id().unwrap()];
        let staged = outer_builder
            .borrow()
            .clone()
            .build::<Vec<TestArray>, Vec<TestArray>>(output_atoms, vec![Placeholder; 2], vec![Placeholder; 1])
            .unwrap();

        let context = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let sample_x = TestArray::vector(vec![0.7, -1.2, 2.0]);
        let sample_cotangent = TestArray::vector(vec![2.5, 1.0, -0.5]);
        let staged_cotangents =
            staged.interpret_in_context(&context, vec![sample_x.clone(), sample_cotangent.clone()]).unwrap();

        // Reference: the established `vjp` pullback at the sample `x`, seeded with the same cotangent. For the
        // elementwise `f(x) = x * sin(x)` the per-element gradient is `sin(x) + x * cos(x)` scaled by the cotangent.
        let (_, reference_pullback) =
            domain.vjp(|inputs| Ok(vec![inputs[0].clone() * inputs[0].sin()?]), vec![sample_x]).unwrap();
        let (reference_pullback, reference_residuals) = reference_pullback.into_parts();
        let mut reference_inputs = vec![sample_cotangent];
        reference_inputs.extend(reference_residuals);
        let reference_cotangents = reference_pullback.interpret_in_context(&context, reference_inputs).unwrap();
        assert_arrays_close(&staged_cotangents, &reference_cotangents, "array reverse-under-tracing cotangent");
    }
}

#[cfg(test)]
mod batching_tests {
    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::axes::AxisIndex;
    use crate::batching::{
        ArrayBatch, Batch, BatchAxis, BatchAxisSpecification, BatchableOperation, BatchingContext, BatchingError,
        BatchingTracer,
    };
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::LinearizationTracer;
    use crate::operations::constants::OneLike;
    use crate::operations::control_flow::ConditionOperation;
    use crate::operations::manipulation::Transpose;
    use crate::operations::math::{AddOperation, NegOperation, Sin};
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;
    use crate::programs::operations::Operation;
    use crate::tracing::DomainTracingContext;
    use crate::tracing_v2::operations::primitive::ArrayOperation;
    use crate::tracing_v2::operations::{Collective, CollectiveKind};
    use crate::tracing_v2::test_util::scalar_scale_branch;
    use crate::tracing_v2::{ForwardModeDifferentiate, ReverseModeDifferentiate};
    use crate::types::{DataType, Shape};

    use super::*;

    #[test]
    fn test_batching_error_conversions_normalize_round_trips() {
        // A batching error that crossed into the kernel as a custom payload converts back to itself, and a
        // `BatchingError::Program` converts back to the program error it carries, so round trips never nest.
        let batching = BatchingError::MismatchedBatchSizes { expected: 4, actual: 5 };
        let program = ProgramError::from(batching.clone());
        assert!(matches!(
            program.downcast_custom::<BatchingError>(),
            Some(BatchingError::MismatchedBatchSizes { expected: 4, actual: 5 }),
        ));
        assert_eq!(BatchingError::from(program), batching);

        let program = ProgramError::EscapedProgramBuilder;
        let batching = BatchingError::from(program.clone());
        assert_eq!(batching, BatchingError::Program(ProgramError::EscapedProgramBuilder));
        assert_eq!(ProgramError::from(batching), program);
    }

    #[test]
    fn test_array_batch_derives_unbatched_type_from_batch_axis() {
        let batch = {
            let value = TestArray::vector(vec![1.0, 2.0, 3.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();

        assert_eq!(batch.batch_size(), Ok(Some(3)));
        assert_eq!(batch.unbatched_type(), Ok(ArrayType::scalar(DataType::F64)));
    }

    #[test]
    fn test_batch_uses_one_packed_array_value() {
        let output: TestArray = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(
                |x| Ok(x.clone() * x.clone() + x.sin()?),
                TestArray::vector(vec![0.0, 1.0, 2.0]),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();

        assert_eq!(output.r#type, ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)])),);
        for (actual, expected) in output.values.iter().zip([0.0, 1.0 + 1.0f64.sin(), 4.0 + 2.0f64.sin()]) {
            assert_abs_diff_eq!(*actual, expected, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_batch_broadcasts_scalar_constants_inside_packed_operations() {
        let output: TestArray = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(
                |x| Ok(x.clone() + x.one_like()),
                TestArray::vector(vec![2.0, 4.0, 6.0]),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();

        assert_eq!(output.values, vec![3.0, 5.0, 7.0]);
    }

    #[test]
    fn test_batch_maps_structured_packed_inputs_and_outputs() {
        let output: (TestArray, TestArray) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(
                |(left, right)| Ok((left.clone() + right.clone(), left * right)),
                (TestArray::vector(vec![1.0, 3.0]), TestArray::vector(vec![2.0, 4.0])),
                (BatchAxis::new(0), BatchAxis::new(0)),
                (BatchAxis::new(0), BatchAxis::new(0)),
                None,
            )
            .unwrap();

        assert_eq!(output.0.values, vec![3.0, 7.0]);
        assert_eq!(output.1.values, vec![2.0, 12.0]);
    }

    #[test]
    fn test_batch_named_axis_psum_reduces_over_batch() {
        let output: TestArray = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(
                |x| x.collective("i", CollectiveKind::PSum),
                TestArray::vector(vec![1.0, 2.0, 3.0]),
                BatchAxis::new(0),
                BatchAxis::replicated(),
                BatchAxisSpecification::named("i"),
            )
            .unwrap();

        assert_eq!(output.r#type, ArrayType::scalar(DataType::F64));
        assert_eq!(output.values, vec![6.0]);
    }

    #[test]
    fn test_batch_axis_index_produces_per_item_indices() {
        // `axis_index("i")` gives each batch item its own position along the mapped axis `"i"` (size 3), so the
        // batched result is the `u64` index vector `[0, 1, 2]` regardless of the operand values.
        let output: TestArray = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(
                |item| item.context().axis_index("i"),
                TestArray::vector(vec![10.0, 20.0, 30.0]),
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxisSpecification::named("i"),
            )
            .unwrap();

        assert_eq!(output.r#type, ArrayType::new(DataType::U64, Shape::new(vec![Size::Static(3)])));
        assert_eq!(output.values, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_nested_batch_axis_index_forwards_outer_axis_through_inner_level() {
        // Outer `batch` over axis 0 (size 2, named "o") of a [2, 3] matrix; inner `batch` over axis 0 (size 3, named
        // "i") of each row. The inner body asks for `axis_index("o")`, which the inner level does not bind, so it is
        // forwarded to the outer level and re-wrapped as replicated across the inner axis (the outer index does not
        // vary over inner items). The inner output is therefore declared replicated (`out_axes = replicated`), and the
        // outer level stacks the per-row outer index, giving the `u64` vector `[0, 1]`.
        let x = TestArray::matrix(2, 3, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        let output: TestArray = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(
                |row| {
                    let context = row.context().clone();
                    Ok(Batch::batch(
                        &context,
                        |scalar| scalar.context().axis_index("o"),
                        row,
                        BatchAxis::new(0),
                        BatchAxis::replicated(),
                        BatchAxisSpecification::named("i"),
                    )?)
                },
                x,
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxisSpecification::named("o"),
            )
            .unwrap();

        assert_eq!(output.r#type, ArrayType::new(DataType::U64, Shape::new(vec![Size::Static(2)])));
        assert_eq!(output.values, vec![0.0, 1.0]);
    }

    #[test]
    fn test_batch_axis_index_rejects_unbound_axis() {
        // `axis_index` over a name no enclosing batch binds fails fast, mirroring the collective readers.
        let result: Result<TestArray, BatchingError> = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(
                |item| item.context().axis_index("j"),
                TestArray::vector(vec![10.0, 20.0, 30.0]),
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxisSpecification::named("i"),
            );
        assert_eq!(
            result.unwrap_err(),
            BatchingError::Axis(crate::axes::AxisError::UnboundAxisName { name: "j".to_string() }),
        );
    }

    #[test]
    fn test_nested_batch_named_axes_route_collectives_to_matching_level() {
        // The inner `psum` targets the *outer* named axis, so each inner batch item must reduce over the
        // outer batch items: column sums of [[1, 2], [3, 4]].
        let x = TestArray::new(
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2)])),
            vec![1.0, 2.0, 3.0, 4.0],
        );
        let output: TestArray = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(
                |row| {
                    let context = row.context().clone();
                    Ok(Batch::batch(
                        &context,
                        |scalar| scalar.collective("outer", CollectiveKind::PSum),
                        row,
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        BatchAxisSpecification::named("inner"),
                    )?)
                },
                x,
                BatchAxis::new(0),
                BatchAxis::replicated(),
                BatchAxisSpecification::named("outer"),
            )
            .unwrap();

        assert_eq!(output.r#type, ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)])),);
        assert_eq!(output.values, vec![4.0, 6.0]);
    }

    #[test]
    fn test_value_and_grad_flows_through_batch_staged_broadcast() {
        use crate::tracing_v2::operations::reduce::{Reduce, ReductionKind};

        // The scalar input is replicated inside the batch, so the elementwise batching rule
        // stages a `Broadcast` on the differentiated value; the gradient must flow back
        // through the broadcast's transpose rule (a sum-reduction over the batch axis).
        let (value, gradient) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .value_and_gradient(
                |x| {
                    let context = x.context().clone();
                    let y = context.lift(TestArray::vector(vec![1.0, 2.0, 3.0, 4.0])).unwrap();
                    let mapped: LinearizationTracer<EagerContext<TestArray, ArrayOperation<TestArray>>> = Batch::batch(
                        &context,
                        |(item, shift)| Ok(item * shift),
                        (y, x),
                        (BatchAxis::new(0), BatchAxis::replicated()),
                        BatchAxis::new(0),
                        None,
                    )
                    .unwrap();
                    mapped.reduce(&[0], ReductionKind::Sum)
                },
                TestArray::scalar(2.0),
            )
            .unwrap();
        assert_abs_diff_eq!(value.values[0], 20.0, epsilon = 1e-9);
        assert_eq!(gradient.values, vec![10.0]);
    }

    #[test]
    fn test_batch_composes_with_context_jvp() {
        let output: (TestArray, TestArray) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(
                |x| {
                    let context = x.context().clone();
                    ForwardModeDifferentiate::jvp(&context, |y| Ok(y.clone() * y), x.clone(), x.one_like())
                        .map_err(ProgramError::from)
                },
                TestArray::vector(vec![2.0, 3.0]),
                BatchAxis::new(0),
                (BatchAxis::new(0), BatchAxis::new(0)),
                None,
            )
            .unwrap();

        assert_eq!(output.0.values, vec![4.0, 9.0]);
        assert_eq!(output.1.values, vec![4.0, 6.0]);
    }

    #[test]
    fn test_batch_composes_with_context_value_and_grad() {
        let output: (TestArray, TestArray) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(
                |x| {
                    let context = x.context().clone();
                    Ok(context
                        .value_and_gradient(|y| y.clone() * y, x)
                        .expect("scalar value_and_gradient should succeed"))
                },
                TestArray::vector(vec![2.0, 3.0]),
                BatchAxis::new(0),
                (BatchAxis::new(0), BatchAxis::new(0)),
                None,
            )
            .unwrap();

        assert_eq!(output.0.values, vec![4.0, 9.0]);
        assert_eq!(output.1.values, vec![4.0, 6.0]);
    }

    #[test]
    fn test_context_batch_composes_inside_jvp() {
        let (primal, tangent): (TestArray, TestArray) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jvp(
                |x| {
                    let context = x.context().clone();
                    Ok(Batch::batch(
                        &context,
                        |item| Ok(item.clone() * item),
                        x,
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        None,
                    )?)
                },
                TestArray::vector(vec![2.0, 3.0]),
                TestArray::vector(vec![1.0, 1.0]),
            )
            .unwrap();

        assert_eq!(primal.values, vec![4.0, 9.0]);
        assert_eq!(tangent.values, vec![4.0, 6.0]);
    }

    #[test]
    fn test_context_batch_composes_inside_value_and_grad() {
        use crate::tracing_v2::operations::reduce::{Reduce, ReductionKind};

        let (value, gradient): (TestArray, TestArray) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .value_and_gradient(
                |x| {
                    let context = x.context().clone();
                    let mapped: LinearizationTracer<EagerContext<TestArray, ArrayOperation<TestArray>>> = Batch::batch(
                        &context,
                        |item| Ok(item.clone() * item),
                        x,
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        None,
                    )
                    .unwrap();
                    mapped.reduce(&[0], ReductionKind::Sum)
                },
                TestArray::vector(vec![2.0, 3.0]),
            )
            .unwrap();

        assert_eq!(value.values, vec![13.0]);
        assert_eq!(gradient.values, vec![4.0, 6.0]);
    }

    #[test]
    fn test_batching_rule_auto_aligns_unaligned_batch_axes() {
        // Both square so the batch sizes agree (4), but they sit on different batch axes.
        // The elementwise batching rule realigns the second operand to match the first batched
        // input's canonical axis (JAX's matchaxis policy), then computes elementwise add.
        //
        // Left is identity-like along axis 0; right is transposed (axis 1). Using row 0 of each
        // batch item: left[item=k, j] == 1.0; right[item=k, j] == 1.0 (since right is symmetric here),
        // so the sum is `2.0` for every element after realignment.
        let left = {
            let value = TestArray::matrix(4, 4, vec![1.0; 16]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let right = {
            let value = TestArray::matrix(4, 4, vec![1.0; 16]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(1))
        }
        .unwrap();
        let context = BatchingContext::new(EagerContext::<TestArray, ArrayOperation<TestArray>>::new(), 4, None);
        let outputs = ArrayOperation::<TestArray>::Add(AddOperation)
            .batch(&context, &crate::EmptyRegionDriver, &[left, right])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert!(outputs[0].value().values().iter().all(|value| (value - 2.0).abs() < 1e-12));
    }

    #[test]
    fn test_elementwise_batch_unary_op() {
        // A unary elementwise op over a single batched operand preserves elementwise semantics and reports the
        // operand's batch axis on its single output. `NegOperation` is elementwise, so its batching rule is the
        // blanket elementwise `BatchableOperation` impl.
        let value = TestArray::vector(vec![1.0, 2.0, 3.0]);
        let batched = ArrayBatch::new(value.r#type().into_owned(), value, Some(0)).unwrap();
        let context = BatchingContext::new(EagerContext::<TestArray, ArrayOperation<TestArray>>::new(), 3, None);
        let outputs = NegOperation.batch(&context, &crate::EmptyRegionDriver, &[batched]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().values(), &[-1.0, -2.0, -3.0]);
    }

    #[test]
    fn test_elementwise_batch_broadcasts_replicated_input() {
        // A batched operand (axis 0, size 3) added to a replicated scalar broadcasts the replicated operand to the
        // batched physical shape, so the single output carries the common batch axis and adds the scalar per item.
        let batched_value = TestArray::vector(vec![1.0, 2.0, 3.0]);
        let batched = ArrayBatch::new(batched_value.r#type().into_owned(), batched_value, Some(0)).unwrap();
        let replicated = ArrayBatch::replicated(TestArray::scalar(10.0));
        let context = BatchingContext::new(EagerContext::<TestArray, ArrayOperation<TestArray>>::new(), 3, None);
        let outputs = AddOperation.batch(&context, &crate::EmptyRegionDriver, &[batched, replicated]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().values(), &[11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_nested_batch_squares_every_element() {
        // x has shape [3, 4]; outer batch maps axis 0 (size 3), inner batch maps axis 0 of the
        // per-outer-item shape [4]. Each element should be squared.
        let x_data: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let x = TestArray::matrix(3, 4, x_data.clone());

        let output: TestArray = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(
                |row| {
                    let context = row.context().clone();
                    Ok(Batch::batch(
                        &context,
                        |scalar| Ok(scalar.clone() * scalar),
                        row,
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        None,
                    )?)
                },
                x,
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();

        assert_eq!(output.r#type, ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(4)])),);
        let expected: Vec<f64> = x_data.iter().map(|value| value * value).collect();
        for (actual, expected) in output.values.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_nested_batch_over_dot_lifts_dimension_numbers() {
        use crate::tracing_v2::operations::dot::{Dot, DotDimensionNumbers};

        // x has shape [3, 4]; outer batch over axis 0 produces per-item rank-1 vectors. Inside,
        // we want every per-item vector dotted with itself, giving a per-item scalar; batch
        // over the leading axis then yields a length-3 vector of dot products.
        let x_data: Vec<f64> = (1..=12).map(|value| value as f64).collect();
        let x = TestArray::matrix(3, 4, x_data);

        let output: TestArray = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(
                |row| Ok(row.dot(&row, &DotDimensionNumbers::inner_product())),
                x,
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();

        assert_eq!(output.r#type, ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)])),);
        // Batch item 0: [1,2,3,4]·[1,2,3,4] = 30. Batch item 1: [5,6,7,8]·[5,6,7,8] = 174. Batch item 2: 446.
        for (actual, expected) in output.values.iter().zip([30.0_f64, 174.0, 446.0].iter()) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_nested_batch_over_transpose_lifts_permutation() {
        // x has shape [2, 3, 4]; outer batch over axis 0 yields per-item rank-2 matrices,
        // which we transpose. The combined effect is to permute axes 1 and 2 of the original
        // tensor, leaving the batch axis (originally axis 0) in place.
        let x_data: Vec<f64> = (0..24).map(|value| value as f64).collect();
        let x = TestArray {
            r#type: ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3), Size::Static(4)])),
            values: x_data,
        };

        let output: TestArray = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(|row| row.transpose(vec![1, 0]), x, BatchAxis::new(0), BatchAxis::new(0), None)
            .unwrap();

        assert_eq!(
            output.r#type,
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(4), Size::Static(3)])),
        );
        // Spot-check: original [0, 0, 0] = 0 → output[0, 0, 0] = 0. Original [0, 0, 1] = 1 → output[0, 1, 0] = 1.
        assert_eq!(output.values[0], 0.0);
        assert_eq!(output.values[1 * 3], 1.0);
    }

    #[test]
    fn test_batch_broadcasts_replicated_input_with_in_axes_none() {
        // x is a [4]-vector mapped on axis 0 (batch items), y is a replicated scalar that should be
        // added to every batch item. The output should be element-wise `x + y` over the 4 batch items.
        let x = TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]);
        let y = TestArray::scalar(10.0);
        let output: TestArray = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(
                |(left, right)| Ok(left + right),
                (x, y),
                (BatchAxis::new(0), BatchAxis::replicated()),
                BatchAxis::new(0),
                None,
            )
            .unwrap();
        assert_eq!(output.values, vec![11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_batch_with_axis_size_validates_mapped_batch_size() {
        // With explicit axis_size = Some(4), the batch size is pinned. A mapped input of size 4
        // must agree, and the batch size flows through to subsequent operations.
        let x = TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]);
        let output: TestArray = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(|x| Ok(x.clone() + x), x, BatchAxis::new(0), BatchAxis::new(0), Some(4))
            .unwrap();
        assert_eq!(output.values, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_batch_with_out_axes_none_rejects_mapped_output() {
        // Function produces a per-item output (mapped on axis 0), but `out_axes = None` declares
        // the output as replicated — matching JAX's semantics. The batch rejects because the
        // computed output is genuinely per-item; users wanting to collapse the batch axis must
        // apply an explicit reduction inside the function.
        let x = TestArray::vector(vec![1.0, 2.0, 3.0]);
        let result: Result<TestArray, BatchingError> = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(|x| Ok(x.clone() + x), x, BatchAxis::new(0), BatchAxis::replicated(), None);
        assert!(matches!(
            result,
            Err(BatchingError::MismatchedOutputAxes { expected, actual })
                if expected == BatchAxis::replicated() && actual == BatchAxis::new(0),
        ));
    }

    #[test]
    fn test_batch_with_out_axes_position_broadcasts_replicated_output() {
        // No input is mapped, so the natural output is replicated. A mapped output declaration materializes the
        // requested axis with a broadcast, matching JAX `vmap` output instantiation.
        let x = TestArray::vector(vec![1.0, 2.0, 3.0]);
        let output: TestArray = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(|x| Ok(x.clone() + x), x, BatchAxis::replicated(), BatchAxis::new(0), Some(3))
            .unwrap();
        assert_eq!(output, TestArray::matrix(3, 3, vec![2.0, 4.0, 6.0, 2.0, 4.0, 6.0, 2.0, 4.0, 6.0]));
    }

    #[test]
    fn test_batch_rejects_dynamic_batch_axis() {
        // A mapped input whose batch dimension is `Size::Dynamic` cannot be batched: batch has no
        // way to determine the batch size.
        let dynamic_input = TestArray {
            r#type: ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None)])),
            values: vec![1.0, 2.0, 3.0],
        };
        let result: Result<TestArray, BatchingError> = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(|x| Ok(x.clone() + x), dynamic_input, BatchAxis::new(0), BatchAxis::new(0), None);
        assert!(matches!(result, Err(BatchingError::DynamicBatchAxis { axis: 0, .. })));
    }

    #[test]
    fn test_batch_with_mismatched_axis_size_rejects_mapped_input() {
        // axis_size=Some(5) conflicts with the mapped input of length 4; this should be detected.
        let x = TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]);
        let result: Result<TestArray, BatchingError> = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(|x| Ok(x.clone() + x), x, BatchAxis::new(0), BatchAxis::new(0), Some(5));
        assert!(matches!(result, Err(BatchingError::MismatchedBatchSizes { expected: 5, actual: 4 })));
    }

    #[test]
    fn test_batch_repositions_output_with_out_axes() {
        // Outer batch over axis 0 of a [3, 4] matrix: each batch item returns its row unchanged.
        // out_axes=Some(1) requests that the batch axis end up at position 1 of the rank-2
        // output, which forces a transpose to swap the axes.
        let x_data: Vec<f64> = (0..12).map(|value| value as f64).collect();
        let x = TestArray::matrix(3, 4, x_data.clone());
        let output: TestArray = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(|row| Ok(row), x, BatchAxis::new(0), BatchAxis::new(1), None)
            .unwrap();
        assert_eq!(output.r#type, ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(4), Size::Static(3)])),);
        // Transpose of [3, 4]: output[i, j] = x[j, i]. Row-major flat indexing:
        // x[j, i] = x_data[j*4 + i]; output[i, j] = output_values[i*3 + j].
        for j in 0..3 {
            for i in 0..4 {
                assert_eq!(output.values[i * 3 + j], x_data[j * 4 + i]);
            }
        }
    }

    #[test]
    fn test_nested_batch_with_mixed_in_axes_propagates_broadcast() {
        // Outer batch over axis 0 of `x: [3, 4]` exposes a rank-1 row to the closure; inside, a
        // second inner batch maps that row's batch axis 0 while broadcasting a captured `bias`
        // scalar to every inner batch item. The combined output is x + bias broadcasted.
        let x_data: Vec<f64> = (0..12).map(|value| value as f64).collect();
        let x = TestArray::matrix(3, 4, x_data.clone());
        let bias = TestArray::scalar(0.5);

        let output: TestArray = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(
                |(row, bias_inner)| {
                    let context = row.context().clone();
                    Ok(Batch::batch(
                        &context,
                        |(scalar, bias_inner)| Ok(scalar + bias_inner),
                        (row, bias_inner),
                        (BatchAxis::new(0), BatchAxis::replicated()),
                        BatchAxis::new(0),
                        None,
                    )?)
                },
                (x, bias),
                (BatchAxis::new(0), BatchAxis::replicated()),
                BatchAxis::new(0),
                None,
            )
            .unwrap();

        assert_eq!(output.r#type, ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(4)])),);
        let expected: Vec<f64> = x_data.iter().map(|value| value + 0.5).collect();
        for (actual, expected) in output.values.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_nested_batch_over_reshape_lifts_input_and_output_shapes() {
        use crate::operations::manipulation::Reshape;

        // x has shape [2, 6]; outer batch over axis 0 yields per-item rank-1 vectors of size 6,
        // which we reshape to per-item [2, 3]. The combined effect should be a [2, 2, 3] tensor
        // whose leading axis is the original batch dimension.
        let x_data: Vec<f64> = (0..12).map(|value| value as f64).collect();
        let x = TestArray::matrix(2, 6, x_data.clone());

        let output: TestArray = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(
                |row| row.reshape(Shape::new(vec![Size::Static(2), Size::Static(3)])),
                x,
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();

        assert_eq!(
            output.r#type,
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2), Size::Static(3)])),
        );
        // Row-major reshape preserves payload ordering; the lifted op only repositions strides.
        assert_eq!(output.values, x_data);
    }

    #[test]
    fn test_batch_stages_replicated_condition_predicates() {
        // A replicated *abstract* condition predicate under trace-time batching cannot be concretized to pick one
        // branch (previously this surfaced a `Concretization` error), so the staged batching rule batches both
        // branch programs at the operand batch axes and stages exactly one `condition` operation over them, with the
        // unbatched predicate passed through. Interpreting the staged batched program with both concrete predicate
        // values matches the eager operational path item for item (scale by 2 when true and by 3 when false).
        let parent = DomainTracingContext::<EagerContext<TestArray, ArrayOperation<TestArray>>>::new();
        let builder = parent.builder().clone();
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let operand_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        let predicate_atom = builder.borrow_mut().add_input(predicate_type.clone());
        let operand_atom = builder.borrow_mut().add_input(operand_type);
        let predicate_tracer = parent.tracer(predicate_atom, None);
        let operand_tracer = parent.tracer(operand_atom, None);
        let output = Batch::batch(
            &parent,
            |(predicate, x)| {
                let condition_regions = vec![scalar_scale_branch(2.0), scalar_scale_branch(3.0)];
                let condition = ConditionOperation::new();
                let op = ArrayOperation::Condition(condition);
                let outputs = x.context().bind(op, condition_regions, &[predicate.clone(), x.clone()])?;
                Ok(outputs.into_iter().next().unwrap())
            },
            (predicate_tracer, operand_tracer),
            (BatchAxis::replicated(), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        let output_atom = output.atom_id().unwrap();
        let program = builder
            .borrow()
            .clone()
            .build::<(TestArray, TestArray), TestArray>(vec![output_atom], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        let condition_count = program
            .instructions()
            .iter()
            .filter(|instruction| instruction.operation().name() == "condition")
            .count();
        assert_eq!(condition_count, 1, "{program}");
        let truthy = TestArray::new(ArrayType::scalar(DataType::Boolean), vec![1.0]);
        let falsy = TestArray::new(ArrayType::scalar(DataType::Boolean), vec![0.0]);
        let operand = TestArray::vector(vec![1.0, 4.0, 9.0]);
        assert_eq!(program.interpret((truthy, operand.clone())).unwrap().values, vec![2.0, 8.0, 18.0]);
        assert_eq!(program.interpret((falsy, operand)).unwrap().values, vec![3.0, 12.0, 27.0]);
    }

    #[test]
    fn test_batch_normalizes_replicated_condition_branch_output_axes() {
        // The two branches of a staged batched condition may disagree on their natural output batch axes: here the
        // true branch scales the batched operand per batch item (axis 0) while the false branch returns a replicated
        // constant (no batch axis). The staged rule normalizes the false branch by appending a broadcast at its
        // tail, so the staged condition stays well-typed and both predicate values interpret correctly per batch item.
        let mut constant_builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        constant_builder.add_input(ArrayType::scalar(DataType::F64));
        let constant_output = constant_builder.add_constant(TestArray::scalar(7.0));
        let constant_branch = constant_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![constant_output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let parent = DomainTracingContext::<EagerContext<TestArray, ArrayOperation<TestArray>>>::new();
        let builder = parent.builder().clone();
        let predicate_atom = builder.borrow_mut().add_input(ArrayType::scalar(DataType::Boolean));
        let operand_atom =
            builder.borrow_mut().add_input(ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)])));
        let predicate_tracer = parent.tracer(predicate_atom, None);
        let operand_tracer = parent.tracer(operand_atom, None);
        let output = Batch::batch(
            &parent,
            |(predicate, x)| {
                let condition_regions = vec![scalar_scale_branch(2.0), constant_branch];
                let condition = ConditionOperation::new();
                let op = ArrayOperation::Condition(condition);
                let outputs = x.context().bind(op, condition_regions, &[predicate.clone(), x.clone()])?;
                Ok(outputs.into_iter().next().unwrap())
            },
            (predicate_tracer, operand_tracer),
            (BatchAxis::replicated(), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        let output_atom = output.atom_id().unwrap();
        let program = builder
            .borrow()
            .clone()
            .build::<(TestArray, TestArray), TestArray>(vec![output_atom], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        let rendered = program.to_string();
        assert!(rendered.contains("broadcast"), "{rendered}");
        let truthy = TestArray::new(ArrayType::scalar(DataType::Boolean), vec![1.0]);
        let falsy = TestArray::new(ArrayType::scalar(DataType::Boolean), vec![0.0]);
        let operand = TestArray::vector(vec![1.0, 4.0, 9.0]);
        assert_eq!(program.interpret((truthy, operand.clone())).unwrap().values, vec![2.0, 8.0, 18.0]);
        assert_eq!(program.interpret((falsy, operand)).unwrap().values, vec![7.0, 7.0, 7.0]);
    }

    #[test]
    fn test_batch_lifts_batch_varying_condition_via_select() {
        // A runtime-predicate Condition inside batch with a batch-varying predicate: each batch item
        // independently chooses between `on_true` (scale by 2.0) and `on_false` (scale by 3.0).
        // The trace-time `BatchingContext` dispatches the rule's `batch`, whose
        // batch-varying branch evaluates both branches over the operand axes and combines per batch item
        // via `Select`. Multi-op staging emerges automatically through `Tracer`'s value-level traits.
        let predicate = TestArray::new(
            ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(4)])),
            vec![1.0, 0.0, 1.0, 0.0],
        );
        let operand = TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]);

        let output: TestArray = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(
                |(pred, operand)| {
                    let condition = ConditionOperation::new();
                    let op = ArrayOperation::Condition(condition);
                    let outputs = pred.context().bind(
                        op,
                        vec![scalar_scale_branch(2.0), scalar_scale_branch(3.0)],
                        &[pred.clone(), operand.clone()],
                    )?;
                    Ok(outputs.into_iter().next().unwrap())
                },
                (predicate, operand),
                (BatchAxis::new(0), BatchAxis::new(0)),
                BatchAxis::new(0),
                None,
            )
            .unwrap();
        // Expected per-item: [1*2, 2*3, 3*2, 4*3] = [2, 6, 6, 12].
        assert_eq!(output.values, vec![2.0, 6.0, 6.0, 12.0]);
    }

    #[test]
    fn test_batch_over_zero_operation_yields_replicated_output() {
        // End-to-end: a batched function that stages `ZeroOperation` produces a replicated zero
        // value at the per-item scalar type. Verifies that the trace-time stage hook accepts a
        // zero-input operation and that the post-trace replay materializes the same zero for
        // every batch item through the replicated broadcast path.
        let output: TestArray = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(
                |x| {
                    let zero_op = ArrayOperation::<TestArray>::Zero(crate::operations::constants::ZeroOperation::new(
                        ArrayType::scalar(DataType::F64),
                    ));
                    let zero = x.context().bind(zero_op, Vec::new(), &[])?.into_iter().next().unwrap();
                    Ok(x + zero)
                },
                TestArray::vector(vec![1.0, 2.0, 3.0]),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();

        assert_eq!(output.values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_batch() {
        let output: TestArray = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(
                |x: BatchingTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| Ok(x.clone() * x),
                TestArray::vector(vec![1.0, 2.0, 3.0]),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();
        assert_eq!(output.r#type, ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)])));
        assert_eq!(output.values, vec![1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_batch_axis_per_value_batch_index() {
        // The per-value `BatchAxis` metadata: a mapped batch dimension index or replicated.
        assert_eq!(BatchAxis::default(), BatchAxis::replicated());
        assert!(BatchAxis::replicated().is_replicated());
        assert!(!BatchAxis::new(2).is_replicated());
        assert_eq!(BatchAxis::replicated().axis(), None);
        assert_eq!(BatchAxis::new(2).axis(), Some(2));
        assert_eq!(BatchAxis::from(None), BatchAxis::replicated());
        assert_eq!(BatchAxis::from(Some(3)), BatchAxis::new(3));
        assert_eq!(BatchAxis::from(3), BatchAxis::new(3));
        assert_ne!(BatchAxis::new(0), BatchAxis::new(1));
        assert_eq!(format!("{:?}", BatchAxis::new(1)), "BatchAxis(Some(1))");
    }

    #[test]
    fn test_batch_axis_specification_constructors_and_conversions() {
        assert_eq!(BatchAxisSpecification::from(None), BatchAxisSpecification::default());
        assert_eq!(BatchAxisSpecification::from(Some(4)), BatchAxisSpecification::sized(4));
        assert_eq!(BatchAxisSpecification::from(4), BatchAxisSpecification::sized(4));
        assert_eq!(BatchAxisSpecification::new(4, "i"), BatchAxisSpecification::new(4, "i").clone());
        assert_ne!(BatchAxisSpecification::sized(4), BatchAxisSpecification::sized(5));
        assert_ne!(BatchAxisSpecification::named("i"), BatchAxisSpecification::named("j"));
        assert_ne!(BatchAxisSpecification::named("i"), BatchAxisSpecification::default());
        assert_eq!(
            format!("{:?}", BatchAxisSpecification::named("i")),
            "BatchAxisSpecification { size: None, name: Some(\"i\") }",
        );
    }

    #[test]
    fn test_batch_broadcasts_a_single_axis_to_every_leaf() {
        // A single `BatchAxis` for `in_axes`/`out_axes` broadcasts into the whole input/output parameter structure
        // (JAX's `in_axes=0`), so both leaves of the pair are mapped on axis 0 without spelling out the structure.
        let x = TestArray::vector(vec![1.0, 3.0]);
        let y = TestArray::vector(vec![2.0, 4.0]);
        let output: (TestArray, TestArray) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(
                |(left, right)| Ok((left.clone() + right.clone(), left * right)),
                (x, y),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();
        assert_eq!(output.0.values, vec![3.0, 7.0]);
        assert_eq!(output.1.values, vec![2.0, 12.0]);
    }

    #[test]
    fn test_batch_broadcasts_mapped_inputs_with_mixed_per_item_ranks() {
        // x is mapped with per-item shape [3]; y is mapped with a per-item scalar shape. The
        // elementwise rule broadcasts y's per-item scalar across the common per-item shape, so
        // each batch item computes `row + shift` with its own shift.
        let x = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = TestArray::vector(vec![10.0, 20.0]);
        let output: TestArray = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(|(row, shift)| Ok(row + shift), (x, y), BatchAxis::new(0), BatchAxis::new(0), None)
            .unwrap();
        assert_eq!(output.r#type, ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)])),);
        assert_eq!(output.values, vec![11.0, 12.0, 13.0, 24.0, 25.0, 26.0]);
    }

    #[test]
    fn test_batch_broadcasts_scalar_replicated_operands_to_full_shape() {
        // A replicated scalar constant added to a mapped [3, 4] input: the elementwise rule
        // materializes a `BroadcastOperation` to the full common batched shape so the staged
        // add receives shape-congruent operands — required for backends such as XLA whose
        // elementwise lowerings (e.g., `stablehlo.add`) have no implicit broadcasting.
        let parent = DomainTracingContext::<EagerContext<TestArray, ArrayOperation<TestArray>>>::new();
        let builder = parent.builder().clone();
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(4)]));
        let input_atom = builder.borrow_mut().add_input(input_type);
        let input_tracer = parent.tracer(input_atom, None);
        let output = Batch::batch(
            &parent,
            |x| {
                let bias = x.context().lift(TestArray::scalar(1.0))?;
                Ok(x + bias)
            },
            input_tracer,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        let output_atom = output.atom_id().unwrap();
        let program = builder
            .borrow()
            .clone()
            .build::<TestArray, TestArray>(vec![output_atom], Placeholder, Placeholder)
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[3, 4] .
                let %1:f64[] = const
                    %2:f64[3, 4] = broadcast [output_type=f64[3, 4], output_axes=[]] %1
                    %3:f64[3, 4] = add %0 %2
                in (%3)
            "}
            .trim_end(),
        );
        let input = TestArray::matrix(3, 4, (0..12).map(|value| value as f64).collect());
        let output = program.interpret(input).unwrap();
        assert_eq!(output.values, (0..12).map(|value| value as f64 + 1.0).collect::<Vec<_>>());
    }
}
