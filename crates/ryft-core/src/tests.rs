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
        // `TestArray` stores `f64` elements, so any fill value that promotes to `f64` is representable. Using the
        // promotion-checked conversion prevents a complex imaginary part from being discarded.
        let Scalar::F64(value) = value.promote_element_type(DataType::F64)? else {
            unreachable!("promotion to f64 yields an f64 scalar")
        };
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
