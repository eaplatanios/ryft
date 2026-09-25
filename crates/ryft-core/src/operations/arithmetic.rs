//! Operations that compute elementwise arithmetic on numeric values. Each operation is defined by an [`Operation`]
//! type (e.g., [`AddOperation`]) together with a value capability trait (e.g., [`Add`]) whose functions apply it to
//! eager [`Array`]s and traced values alike, so the same code executes immediately or records into a program depending
//! on the value it runs on. The operations fall into three groups:
//!
//!   - **Binary Arithmetic:** [`Add`], [`Sub`], [`Mul`], and [`Div`] compute sums, differences, products, and
//!     quotients, and [`Rem`] computes remainders that take the sign of the dividend (i.e., truncated division, like
//!     Rust's `%`).
//!   - **Sign and Magnitude:** [`Neg`] negates a value, [`Abs`] computes its absolute value (i.e., the real magnitude
//!     `|z|` for a complex value), and [`Sign`] maps it to `-1`, `0`, or `1` (i.e., `z / |z|` for a nonzero complex
//!     value).
//!   - **Powers and Roots:** [`Pow`] raises one value to the power of another (i.e., the principal value
//!     `exp(y · log(x))` for complex values), and [`Sqrt`] and [`Rsqrt`] compute square roots and reciprocal square
//!     roots.
//!
//! Binary operations promote their element types and broadcast their shapes, as for StableHLO's
//! [`add`](https://openxla.org/stablehlo/spec#add), [`subtract`](https://openxla.org/stablehlo/spec#subtract),
//! [`multiply`](https://openxla.org/stablehlo/spec#multiply), [`divide`](https://openxla.org/stablehlo/spec#divide),
//! [`remainder`](https://openxla.org/stablehlo/spec#remainder), and
//! [`power`](https://openxla.org/stablehlo/spec#power).
//! Array operands that carry partial sums over unreduced mesh axes are accepted only where the result remains a valid
//! partial sum: negation preserves them, sums and differences require both operands to be unreduced over the same
//! axes, and products permit one unreduced operand when the other is reduced over those same axes. Every other
//! operation rejects such operands. Refer to the documentation of each operation for the element types it supports.
//! [`Add`], [`Sub`], [`Mul`], [`Div`], [`Rem`], and [`Neg`] are the fallible counterparts of the corresponding
//! [`std::ops`] operators, which values additionally implement as panicking sugar. Eager integer arrays wrap at their
//! element width, whereas the host integer primitives report overflow as an error.
//!
//! Negation, addition, and subtraction are linear. Multiplication is linear in either operand when the other one is
//! known, and division is linear in its numerator when its denominator is known, so all of them can be transposed.
//! [`Sign`] is piecewise constant and has zero derivatives, and the remaining operations are nonlinear, so reverse-mode
//! differentiation transposes their linearizations instead.
//!
//! # Example
//!
//! ```rust
//! # use ryft_core::{Add, Array, ProgramError, Sqrt};
//! # fn main() -> Result<(), ProgramError> {
//! let input = Array::vector(vec![0.0f64, 3.0, 8.0])?;
//! let output = input.add(&Array::scalar(1.0f64)?)?.sqrt()?;
//! assert_eq!(output.elements::<f64>()?, vec![1.0, 2.0, 3.0]);
//! # Ok(())
//! # }
//! ```

use std::collections::BTreeSet;
use std::ops;
use std::sync::Arc;

use crate::arrays::{
    Array, ArrayAddressing, ArrayElement, ArrayType, DataType, FloatingPointArrayElement, NumericArrayElement,
    RealArrayElement,
};
use crate::contexts::StagingContext;
use crate::differentiation::{DifferentiableType, DifferentiationDual, ElementwiseDerivativeAlignment};
use crate::macros::{
    check_count, check_types, define_elementwise_capability, define_elementwise_operation, define_tracer_operator,
    dispatch_on_array_element_type, impl_array_elementwise_operation, impl_differentiable_elementwise_operation,
    impl_differentiable_operation,
};
use crate::operations::ElementwiseOperation;
use crate::operations::comparisons::{Compare, ComparisonDirection};
use crate::operations::complex::{Complex, Conjugate, Imaginary, Real};
use crate::operations::constants::one_like::OneLike;
use crate::operations::constants::zero_like::ZeroLike;
use crate::operations::control_flow::select::Select;
use crate::operations::exponential::Log;
use crate::programs::{MaybeZero, Operation, ProgramError, Type, TypeError, Typed};
use crate::tracing::{Tracer, TracingContext};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`AddOperation`].
pub const ADD_OPERATION_NAME: &str = "add";

define_elementwise_operation!(
    @binary
    /// [`Operation`] that adds two numeric values elementwise, promoting their element [`DataType`]s and broadcasting
    /// their [`Shape`](crate::arrays::Shape)s. Array operands that carry partial sums must both be unreduced over
    /// exactly the same mesh axes. Mixing an unreduced operand with an already reduced operand would duplicate the
    /// reduced contribution when the result is subsequently reduced. Their reduced-axis markers must likewise agree.
    AddOperation,
    ADD_OPERATION_NAME,
    Add,
    add,
    check_data_types = [@numeric],
    check_array_types = [@same_unreduced_axes, @same_reduced_axes],
);

impl_differentiable_elementwise_operation! {
    @linear
    AddOperation,
    rule = [@positive, @positive],
}

define_elementwise_capability!(
    @binary
    /// Value capability for elementwise addition.
    ///
    /// Eager values compute directly; contextual values bind [`AddOperation`]. Refer to that operation for
    /// supported input types, broadcasting, and reduction-state requirements. Failures are returned as
    /// [`ProgramError`].
    Add,
    /// Adds `rhs` to this value.
    add(rhs),
    AddOperation,
);

impl_array_elementwise_operation!(
    @binary
    Add, add,
    operation = "add",
    inputs = @numeric,
    checks = [@same_unreduced_axes, @same_reduced_axes],
    |lhs, rhs| NumericArrayElement::add(lhs, rhs),
);

impl std::ops::Add for Array {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Add::add(&self, &rhs).unwrap_or_else(|error| panic!("{error}"))
    }
}
define_tracer_operator!(@binary std::ops::Add, add, capability = Add, method = add);

/// Implements [`Add`] for one host primitive type.
macro_rules! impl_add_for_primitive {
    // Integer primitives use checked arithmetic so that host bookkeeping (e.g., dimension-extent math) reports
    // arithmetic failures as errors instead of wrapping like the XLA-mirroring reference backends do on devices.
    (@integer $type:ty) => {
        impl Add for $type {
            fn add(&self, rhs: &Self) -> Result<Self, ProgramError> {
                self.checked_add(*rhs).ok_or_else(|| ProgramError::InvalidArgument {
                    message: format!("`{}` output does not fit in `{}`", ADD_OPERATION_NAME, stringify!($type)),
                })
            }
        }
    };

    // Floating-point primitives use ordinary IEEE 754 arithmetic, which cannot fail.
    (@float $type:ty) => {
        impl Add for $type {
            fn add(&self, rhs: &Self) -> Result<Self, ProgramError> {
                Ok(*self + *rhs)
            }
        }
    };
}

impl_add_for_primitive!(@integer i8);
impl_add_for_primitive!(@integer i16);
impl_add_for_primitive!(@integer i32);
impl_add_for_primitive!(@integer i64);
impl_add_for_primitive!(@integer i128);
impl_add_for_primitive!(@integer isize);
impl_add_for_primitive!(@integer u8);
impl_add_for_primitive!(@integer u16);
impl_add_for_primitive!(@integer u32);
impl_add_for_primitive!(@integer u64);
impl_add_for_primitive!(@integer u128);
impl_add_for_primitive!(@integer usize);
impl_add_for_primitive!(@float f32);
impl_add_for_primitive!(@float f64);

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`SubOperation`].
pub const SUB_OPERATION_NAME: &str = "sub";

define_elementwise_operation!(
    @binary
    /// [`Operation`] that subtracts two numeric values elementwise, promoting their element types and
    /// broadcasting their shapes. Array operands that carry partial sums must both be unreduced over exactly the same
    /// mesh axes (subtraction is linear, so the difference of two partial sums over the same axes is another valid
    /// partial sum); mixing an unreduced operand with an already reduced operand would duplicate the reduced
    /// contribution when the result is subsequently reduced. Their reduced-axis markers must likewise agree.
    SubOperation, SUB_OPERATION_NAME,
    Sub, sub,
    check_data_types = [@numeric],
    check_array_types = [@same_unreduced_axes, @same_reduced_axes],
);

impl_differentiable_elementwise_operation! {
    @linear
    SubOperation,
    rule = [@positive, @negative],
}

define_elementwise_capability!(
    @binary
    /// Value capability for elementwise subtraction.
    ///
    /// Eager values compute directly; contextual values bind [`SubOperation`]. Refer to that operation for
    /// supported input types, broadcasting, and reduction-state requirements. Failures are returned as
    /// [`ProgramError`].
    Sub,
    /// Subtracts `right` from this value.
    sub(right),
    SubOperation,
);

impl_array_elementwise_operation!(
    @binary
    Sub, sub,
    operation = "sub",
    inputs = @numeric,
    checks = [@same_unreduced_axes, @same_reduced_axes],
    |lhs, rhs| NumericArrayElement::sub(lhs, rhs),
);

impl std::ops::Sub for Array {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Sub::sub(&self, &rhs).unwrap_or_else(|error| panic!("{error}"))
    }
}
define_tracer_operator!(@binary std::ops::Sub, sub, capability = Sub, method = sub);

/// Implements [`Sub`] for one host primitive type.
macro_rules! impl_sub_for_primitive {
    // Integer primitives use checked arithmetic so that host bookkeeping (e.g., dimension-extent math) reports
    // arithmetic failures as errors instead of wrapping like the XLA-mirroring reference backends do on devices.
    (@integer $type:ty) => {
        impl Sub for $type {
            fn sub(&self, right: &Self) -> Result<Self, ProgramError> {
                self.checked_sub(*right).ok_or_else(|| ProgramError::InvalidArgument {
                    message: format!("`{}` output does not fit in `{}`", SUB_OPERATION_NAME, stringify!($type)),
                })
            }
        }
    };

    // Floating-point primitives use ordinary IEEE 754 arithmetic, which cannot fail.
    (@float $type:ty) => {
        impl Sub for $type {
            fn sub(&self, right: &Self) -> Result<Self, ProgramError> {
                Ok(*self - *right)
            }
        }
    };
}

impl_sub_for_primitive!(@integer i8);
impl_sub_for_primitive!(@integer i16);
impl_sub_for_primitive!(@integer i32);
impl_sub_for_primitive!(@integer i64);
impl_sub_for_primitive!(@integer i128);
impl_sub_for_primitive!(@integer isize);
impl_sub_for_primitive!(@integer u8);
impl_sub_for_primitive!(@integer u16);
impl_sub_for_primitive!(@integer u32);
impl_sub_for_primitive!(@integer u64);
impl_sub_for_primitive!(@integer u128);
impl_sub_for_primitive!(@integer usize);
impl_sub_for_primitive!(@float f32);
impl_sub_for_primitive!(@float f64);

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`MulOperation`].
pub const MUL_OPERATION_NAME: &str = "mul";

define_elementwise_operation!(
    @binary
    /// [`Operation`] that multiplies two numeric values elementwise, promoting their element types and broadcasting
    /// their shapes. Its bilinear reduction-state rule permits one unreduced operand only when the other operand is
    /// reduced over exactly the same mesh axes.
    MulOperation, MUL_OPERATION_NAME,
    Mul, mul,
    infer_array_types = infer_mul_output_array_types,
    check_data_types = [@numeric],
);

// Transposition accepts exactly one linear operand and scales its output cotangent by the other, known operand. The
// contribution is unbroadcast to the linear operand's exact cotangent type, while the known operand receives a
// structural zero.
impl_differentiable_elementwise_operation! {
    @binary
    MulOperation,
    jvp<C> where C::Value: ops::Mul<Output = C::Value> {
        |(_, left_tangent), (right, _)| right * left_tangent;
        |(left, _), (_, right_tangent)| left * right_tangent;
    },
    transpose<V, O>
    where
        V::Type: DifferentiableType,
        O: From<MulOperation<V::Type>>,
        Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<V::Type>,
    {
        |_operation, context, _driver, inputs, outputs, accumulators| {
            check_count!("input", inputs, 2, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 2, DifferentiationError);
            let (linear, known) = match (inputs[0].is_unknown(), inputs[1].is_unknown()) {
                (true, false) => (0, 1),
                (false, true) => (1, 0),
                (left_linear, right_linear) => return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "operation `mul` does not support transposition for input pattern [left = {}, right = {}]",
                        if left_linear { "linear" } else { "known" },
                        if right_linear { "linear" } else { "known" },
                    ),
                }.into()),
            };
            let target = inputs[linear].r#type().cotangent()?;
            let contribution = match &outputs[0] {
                MaybeZero::Zero(_) => MaybeZero::Zero(target),
                MaybeZero::Value(cotangent) => {
                    if target.is_zero_space() {
                        return Err(ProgramError::UnsupportedOperation {
                            message: format!(
                                "linear input `{}` of operation `mul` has no cotangent space",
                                if linear == 0 { "left" } else { "right" },
                            ),
                        }.into());
                    }
                    // The coefficient is a primal value: keep its reduction state when multiplying the dual
                    // cotangent. Broadcasting and promotion happen in multiplication's own inference.
                    let coefficient = inputs[known].as_known().unwrap();
                    let mut contribution = context.stage_operation(
                        MulOperation::new(),
                        Vec::new(),
                        &[coefficient.clone(), cotangent.clone()],
                    )?;
                    check_count!("output", contribution, 1, ProgramError);
                    MaybeZero::Value(contribution.remove(0).unalign_cotangent(&target)?)
                }
            };
            accumulators[linear].accumulate(context, contribution)
        }
    },
}

define_elementwise_capability!(
    @binary
    /// Value capability for elementwise multiplication.
    ///
    /// Eager values compute directly; contextual values bind [`MulOperation`]. Refer to that operation for
    /// supported input types, broadcasting, and reduction-state requirements. Failures are returned as
    /// [`ProgramError`].
    Mul,
    /// Multiplies `self` by `right`.
    mul(right),
    MulOperation,
);

impl Mul for Array {
    fn mul(&self, rhs: &Self) -> Result<Self, ProgramError> {
        // Multiplication combines reduction states bilinearly rather than requiring congruent operand metadata.
        // Use the operation's inference before evaluating elements so empty inputs obey the same contract.
        let mut output_types = Operation::infer_output_types(
            &MulOperation::<ArrayType>::new(),
            &[self.r#type().into_owned(), rhs.r#type().into_owned()],
            &[],
        )?;
        let output_type = output_types.remove(0);
        if Self::element_count(&output_type) == 0 {
            let addressing = ArrayAddressing::new(output_type.clone())?;
            return Ok(Self::new_unchecked(output_type, Arc::new(vec![0; addressing.storage_byte_len()])));
        }
        let data_type = output_type.data_type();
        let lhs = self.promoted_to(data_type)?;
        let rhs = rhs.promoted_to(data_type)?;
        dispatch_on_array_element_type!(@numeric data_type, |Element| {
            lhs.map_element_pairs::<Element, Element>(&rhs, output_type, NumericArrayElement::mul)
        })
    }
}

impl std::ops::Mul for Array {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Mul::mul(&self, &rhs).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl std::ops::Mul<f64> for Array {
    type Output = Self;

    /// Scales every element by `rhs`, converting `rhs` into this array's element data type first so that scaling
    /// preserves the array's type (e.g., scaling an `f32` array does not promote it to `f64`).
    fn mul(self, rhs: f64) -> Self::Output {
        let data_type = self.r#type().data_type();
        let factor = dispatch_on_array_element_type!(data_type, |Element| {
            Self::scalar(Element::from_real(rhs).unwrap_or_else(|error| panic!("{error}"))).unwrap()
        });
        Mul::mul(&self, &factor).unwrap_or_else(|error| panic!("{error}"))
    }
}
define_tracer_operator!(@binary std::ops::Mul, mul, capability = Mul, method = mul);

/// Implements [`Mul`] for one host primitive type.
macro_rules! impl_mul_for_primitive {
    // Integer primitives use checked arithmetic so that host bookkeeping (e.g., dimension-extent math) reports
    // arithmetic failures as errors instead of wrapping like the XLA-mirroring reference backends do on devices.
    (@integer $type:ty) => {
        impl Mul for $type {
            fn mul(&self, right: &Self) -> Result<Self, ProgramError> {
                self.checked_mul(*right).ok_or_else(|| ProgramError::InvalidArgument {
                    message: format!("`{}` output does not fit in `{}`", MUL_OPERATION_NAME, stringify!($type)),
                })
            }
        }
    };

    // Floating-point primitives use ordinary IEEE 754 arithmetic, which cannot fail.
    (@float $type:ty) => {
        impl Mul for $type {
            fn mul(&self, right: &Self) -> Result<Self, ProgramError> {
                Ok(*self * *right)
            }
        }
    };
}

impl_mul_for_primitive!(@integer i8);
impl_mul_for_primitive!(@integer i16);
impl_mul_for_primitive!(@integer i32);
impl_mul_for_primitive!(@integer i64);
impl_mul_for_primitive!(@integer i128);
impl_mul_for_primitive!(@integer isize);
impl_mul_for_primitive!(@integer u8);
impl_mul_for_primitive!(@integer u16);
impl_mul_for_primitive!(@integer u32);
impl_mul_for_primitive!(@integer u64);
impl_mul_for_primitive!(@integer u128);
impl_mul_for_primitive!(@integer usize);
impl_mul_for_primitive!(@float f32);
impl_mul_for_primitive!(@float f64);

/// Infers multiplication output array types using its bilinear reduction-state rule.
fn infer_mul_output_array_types(input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
    // Multiplication is bilinear, so its output sharding combines the operands' unreduced/reduced state by the
    // bilinear rule rather than the congruent rule used by generic elementwise broadcasting. The reduction state
    // is combined independently of per-dimension placement, so the placement is broadcast with that state stripped
    // and the recomputed state is reattached afterward.
    let stripped = [input_types[0].without_reduction_axes(), input_types[1].without_reduction_axes()];
    let output = MulOperation::<ArrayType>::new().infer_elementwise_broadcast_type(&stripped)?;
    let left_unreduced = input_types[0].unreduced_axes();
    let left_reduced = input_types[0].reduced_axes();
    let right_unreduced = input_types[1].unreduced_axes();
    let right_reduced = input_types[1].reduced_axes();

    // An operand unreduced over some axes is a partial sum still awaiting an all-reduce over them. The product of
    // two partial sums is not a partial sum, so at most one operand may be unreduced. The other must then be
    // reduced over exactly those axes, and the product stays unreduced over them (its matching reduced marker is
    // consumed when the reduced set is computed below).
    let output_unreduced = match (left_unreduced.is_empty(), right_unreduced.is_empty()) {
        (false, false) => {
            return Err(TypeError::invalid(format!(
                "`{MUL_OPERATION_NAME}` cannot multiply two operands that are both unreduced",
            )));
        }
        (false, true) => {
            if left_unreduced != right_reduced {
                return Err(TypeError::invalid(format!(
                    "`{MUL_OPERATION_NAME}` requires the second operand to be reduced over the axes \
                             the first is unreduced over",
                )));
            }
            left_unreduced.clone()
        }
        (true, false) => {
            if right_unreduced != left_reduced {
                return Err(TypeError::invalid(format!(
                    "`{MUL_OPERATION_NAME}` requires the first operand to be reduced over the axes \
                             the second is unreduced over",
                )));
            }
            right_unreduced.clone()
        }
        (true, true) => BTreeSet::new(),
    };

    // Plain reduced axes must agree. The only one-sided reduced marker that is valid is the marker consumed by the
    // partial-sum-times-reduced case above; a one-sided marker without a matching unreduced operand would
    // incorrectly propagate reduction state from only one input.
    let mut output_reduced = if left_reduced == right_reduced {
        left_reduced.clone()
    } else if left_reduced.is_empty() && right_reduced == &output_unreduced {
        right_reduced.clone()
    } else if right_reduced.is_empty() && left_reduced == &output_unreduced {
        left_reduced.clone()
    } else {
        return Err(TypeError::invalid(format!("`{MUL_OPERATION_NAME}` operands must be reduced over the same axes")));
    };
    output_reduced.retain(|axis| !output_unreduced.contains(axis));

    // A non-empty result reduction state means some operand was sharded, so the broadcast output (already stripped
    // of reduction axes) carries a sharding onto which the recomputed state is reattached; otherwise it is already
    // correct as is.
    if output_unreduced.is_empty() && output_reduced.is_empty() {
        return Ok(vec![output]);
    }
    let sharding = output.sharding().unwrap();
    let rebuilt = sharding
        .clone()
        .with_unreduced_axes(output_unreduced)
        .map_err(|error| TypeError::invalid(error.to_string()))?
        .with_reduced_axes(output_reduced)
        .map_err(|error| TypeError::invalid(error.to_string()))?;
    Ok(vec![output.with_sharding(rebuilt).map_err(|error| TypeError::invalid(error.to_string()))?])
}

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`DivOperation`].
pub const DIV_OPERATION_NAME: &str = "div";

define_elementwise_operation!(
    @binary
    /// [`Operation`] that divides two numeric values elementwise, promoting their element types and
    /// broadcasting their shapes. Array operands that still carry partial sums are rejected, and their reduced-axis
    /// markers must agree.
    DivOperation, DIV_OPERATION_NAME,
    Div, div,
    check_data_types = [@numeric],
    check_array_types = [@no_unreduced, @same_reduced_axes],
);

// Transposition accepts a linear numerator and a known denominator.
impl_differentiable_elementwise_operation! {
    @binary
    DivOperation,
    jvp<C>
    where
        C::Value: ops::Neg<Output = C::Value>
            + ops::Mul<Output = C::Value>
            + ops::Div<Output = C::Value>,
    {
        |(_, left_tangent), (right, _)| left_tangent / right;
        |(left, _), (right, right_tangent)| {
            let coefficient = -(left / (right.clone() * right));
            coefficient * right_tangent
        };
    },
    transpose<V, O>
    where
        V::Type: DifferentiableType,
        O: From<DivOperation<V::Type>>,
        Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<V::Type>,
    {
        [numerator = @linear, denominator = @known] =>
            |output_cotangent| output_cotangent.binary(&denominator, DivOperation::new());
    },
}

define_elementwise_capability!(
    @binary
    /// Value capability for elementwise division.
    ///
    /// Eager values compute directly; contextual values bind [`DivOperation`]. Refer to that operation for
    /// supported input types, broadcasting, and reduction-state requirements. Failures are returned as
    /// [`ProgramError`].
    Div,
    /// Divides this value by `right`.
    div(right),
    DivOperation,
);

impl_array_elementwise_operation!(
    @binary
    Div, div,
    operation = "div",
    inputs = @numeric,
    checks = [@no_unreduced, @same_reduced_axes],
    |lhs, rhs| NumericArrayElement::div(lhs, rhs),
);

impl std::ops::Div for Array {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Div::div(&self, &rhs).unwrap_or_else(|error| panic!("{error}"))
    }
}
define_tracer_operator!(@binary std::ops::Div, div, capability = Div, method = div);

/// Implements [`Div`] for one host primitive type.
macro_rules! impl_div_for_primitive {
    // Integer primitives use checked arithmetic so that host bookkeeping (e.g., dimension-extent math) reports
    // arithmetic failures as errors instead of wrapping like the XLA-mirroring reference backends do on devices.
    (@integer $type:ty) => {
        impl Div for $type {
            fn div(&self, right: &Self) -> Result<Self, ProgramError> {
                self.checked_div(*right).ok_or_else(|| ProgramError::InvalidArgument {
                    message: format!(
                        "`{}` divisor is zero or the output does not fit in `{}`",
                        DIV_OPERATION_NAME,
                        stringify!($type),
                    ),
                })
            }
        }
    };

    // Floating-point primitives use ordinary IEEE 754 arithmetic, which cannot fail.
    (@float $type:ty) => {
        impl Div for $type {
            fn div(&self, right: &Self) -> Result<Self, ProgramError> {
                Ok(*self / *right)
            }
        }
    };
}

impl_div_for_primitive!(@integer i8);
impl_div_for_primitive!(@integer i16);
impl_div_for_primitive!(@integer i32);
impl_div_for_primitive!(@integer i64);
impl_div_for_primitive!(@integer i128);
impl_div_for_primitive!(@integer isize);
impl_div_for_primitive!(@integer u8);
impl_div_for_primitive!(@integer u16);
impl_div_for_primitive!(@integer u32);
impl_div_for_primitive!(@integer u64);
impl_div_for_primitive!(@integer u128);
impl_div_for_primitive!(@integer usize);
impl_div_for_primitive!(@float f32);
impl_div_for_primitive!(@float f64);

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`RemOperation`].
pub const REM_OPERATION_NAME: &str = "rem";

define_elementwise_operation!(
    @binary
    /// [`Operation`] that computes the elementwise remainder of a dividend (its left operand) and a divisor (its
    /// right operand), promoting their element types and broadcasting their shapes. The result takes the sign of the
    /// dividend and has magnitude less than the divisor's (i.e., truncation semantics, matching
    /// [StableHLO's `remainder`](https://openxla.org/stablehlo/spec#remainder) and Rust's `%`). Only integer and
    /// floating-point operands are supported. Array operands that still carry partial sums are rejected, and their
    /// reduced-axis markers must agree.
    RemOperation, REM_OPERATION_NAME,
    Rem, rem,
    check_data_types = [@numeric @real],
    check_array_types = [@no_unreduced, @same_reduced_axes],
);

impl_differentiable_elementwise_operation! {
    @binary
    RemOperation,
    jvp<C>
    where
        C::Value: Rem
            + ops::Div<Output = C::Value>
            + ops::Mul<Output = C::Value>
            + ops::Neg<Output = C::Value>
            + ops::Sub<Output = C::Value>,
    {
        // d(rem(x, y)) = dx - trunc(x / y) · dy away from the discontinuities, with the truncated quotient
        // recovered exactly as (x - rem(x, y)) / y.
        |(_, left_tangent), (_, _)| left_tangent;
        |(left, _), (right, right_tangent)| {
            let truncated_quotient = (left.clone() - left.rem(&right)?) / right;
            -(truncated_quotient * right_tangent)
        };
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @binary
    /// Value capability for elementwise remainders with the sign of the dividend.
    ///
    /// Eager values compute directly; contextual values bind [`RemOperation`]. Refer to that operation for
    /// supported input types, broadcasting, and reduction-state requirements. Failures are returned as
    /// [`ProgramError`].
    Rem,
    /// Computes the remainder of dividing this value (the dividend) by `right` (the divisor), with the result taking
    /// the sign of the dividend.
    rem(right),
    RemOperation,
);

impl_array_elementwise_operation!(
    @binary
    Rem, rem,
    operation = "rem",
    inputs = @numeric @real,
    checks = [@no_unreduced, @same_reduced_axes],
    |lhs, rhs| RealArrayElement::rem(lhs, rhs),
);

define_tracer_operator!(
    @binary std::ops::Rem,
    rem,
    capability = Rem,
    method = rem,
);

/// Implements [`Rem`] for one host primitive type.
macro_rules! impl_rem_for_primitive {
    // Integer primitives use checked arithmetic so that host bookkeeping (e.g., dimension-extent math) reports
    // arithmetic failures as errors instead of wrapping like the XLA-mirroring reference backends do on devices.
    (@integer $type:ty) => {
        impl Rem for $type {
            fn rem(&self, right: &Self) -> Result<Self, ProgramError> {
                self.checked_rem(*right).ok_or_else(|| ProgramError::InvalidArgument {
                    message: format!(
                        "`{}` divisor is zero or the output does not fit in `{}`",
                        REM_OPERATION_NAME,
                        stringify!($type),
                    ),
                })
            }
        }
    };

    // Floating-point primitives use ordinary IEEE 754 arithmetic, which cannot fail.
    (@float $type:ty) => {
        impl Rem for $type {
            fn rem(&self, right: &Self) -> Result<Self, ProgramError> {
                Ok(*self % *right)
            }
        }
    };
}

impl_rem_for_primitive!(@integer i8);
impl_rem_for_primitive!(@integer i16);
impl_rem_for_primitive!(@integer i32);
impl_rem_for_primitive!(@integer i64);
impl_rem_for_primitive!(@integer i128);
impl_rem_for_primitive!(@integer isize);
impl_rem_for_primitive!(@integer u8);
impl_rem_for_primitive!(@integer u16);
impl_rem_for_primitive!(@integer u32);
impl_rem_for_primitive!(@integer u64);
impl_rem_for_primitive!(@integer u128);
impl_rem_for_primitive!(@integer usize);
impl_rem_for_primitive!(@float f32);
impl_rem_for_primitive!(@float f64);

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`NegOperation`].
pub const NEG_OPERATION_NAME: &str = "neg";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that negates one integer, floating-point, or complex value while preserving its array metadata
    /// and reduction state. Boolean, token, structural-zero, and the unsigned-only `f8e8m0fnu` data types are rejected.
    NegOperation, NEG_OPERATION_NAME,
    Neg, neg,
    infer_data_types = |input_types: &[DataType]| {
        check_types!(@numeric, NEG_OPERATION_NAME, input_types);
        let input_type = input_types[0];
        if input_type == DataType::F8E8M0FNU {
            return Err(TypeError::invalid(format!(
                "`{NEG_OPERATION_NAME}` does not support input data type `f8e8m0fnu`",
            )));
        }
        Ok(vec![input_type])
    },
);

impl_differentiable_elementwise_operation! {
    @linear
    NegOperation,
    rule = [@negative],
}

define_elementwise_capability!(
    @unary
    /// Value capability for elementwise negation.
    ///
    /// Eager values compute directly; contextual values bind [`NegOperation`]. Refer to that operation for
    /// supported input types, broadcasting, and reduction-state requirements. Failures are returned as
    /// [`ProgramError`].
    Neg,
    /// Negates `self`.
    neg,
    NegOperation,
);

impl Neg for Array {
    fn neg(&self) -> Result<Self, ProgramError> {
        if Self::element_count(self.r#type().as_ref()) == 0 {
            let addressing = ArrayAddressing::new(self.r#type().into_owned())?;
            return Ok(Self::new_unchecked(
                self.r#type().into_owned(),
                Arc::new(vec![0; addressing.storage_byte_len()]),
            ));
        }
        let data_type = self.r#type().data_type();
        if !data_type.is_numeric() {
            return Err(TypeError::invalid(format!("cannot negate a scalar of data type `{data_type}`")).into());
        }
        dispatch_on_array_element_type!(@numeric data_type, |Element| {
            self.map_elements::<Element, Element>(self.r#type().into_owned(), <Element as ArrayElement>::neg)
        })
    }
}

impl std::ops::Neg for Array {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Neg::neg(&self).unwrap_or_else(|error| panic!("{error}"))
    }
}
define_tracer_operator!(@unary std::ops::Neg, neg, NegOperation, "`neg` operation failed");

/// Implements [`Neg`] for one host primitive type.
macro_rules! impl_neg_for_primitive {
    // Signed integer primitives use checked negation so that the `MIN` overflow reports an error instead of
    // wrapping like the XLA-mirroring reference backends do on devices.
    (@signed $type:ty) => {
        impl Neg for $type {
            fn neg(&self) -> Result<Self, ProgramError> {
                self.checked_neg().ok_or_else(|| ProgramError::InvalidArgument {
                    message: format!("`{}` output does not fit in `{}`", NEG_OPERATION_NAME, stringify!($type)),
                })
            }
        }
    };

    // Floating-point primitives use ordinary IEEE 754 negation, which cannot fail.
    (@float $type:ty) => {
        impl Neg for $type {
            fn neg(&self) -> Result<Self, ProgramError> {
                Ok(-*self)
            }
        }
    };
}

impl_neg_for_primitive!(@signed i8);
impl_neg_for_primitive!(@signed i16);
impl_neg_for_primitive!(@signed i32);
impl_neg_for_primitive!(@signed i64);
impl_neg_for_primitive!(@signed i128);
impl_neg_for_primitive!(@signed isize);
impl_neg_for_primitive!(@float f32);
impl_neg_for_primitive!(@float f64);

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`AbsOperation`].
pub const ABS_OPERATION_NAME: &str = "abs";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that computes the elementwise absolute value of a value (i.e., `x ↦ |x|` and the magnitude `|z|`
    /// for complex operands with a real result) while preserving all other type metadata. Inputs that still represent
    /// partial sums over unreduced mesh axes are rejected because taking an absolute value does not preserve
    /// partial-sum semantics. Matching the operand constraints of
    /// [StableHLO's `abs`](https://openxla.org/stablehlo/spec#abs), signed-integer (including the sub-byte
    /// [`DataType::I2`] and [`DataType::I4`] types, with the minimum value wrapping to itself), floating-point,
    /// and complex inputs are supported, while unsigned-integer, Boolean, token, structural-zero, and single-bit
    /// [`DataType::I1`] inputs (whose only negative value `-1` has no representable absolute value) are rejected.
    AbsOperation,
    ABS_OPERATION_NAME,
    Abs,
    abs,
    infer_data_types = |input_types: &[DataType]| {
        let input_type = input_types[0];
        let output_type = if input_type == DataType::C64 {
            DataType::F32
        } else if input_type == DataType::C128 {
            DataType::F64
        } else if (input_type.is_signed() && input_type != DataType::I1) || input_type.is_floating_point() {
            input_type
        } else {
            return Err(TypeError::invalid(format!(
                "cannot compute the absolute value of a value of data type `{input_type}`",
            )));
        };
        Ok(vec![output_type])
    },
    check_array_types = [@no_unreduced],
);

impl_differentiable_operation! {
    <T> AbsOperation<T>,
    jvp<C>
    where
        T: Type,
        C::Type: DifferentiableType,
        C::Value: Abs
            + Compare<C::Value>
            + Complex
            + Conjugate
            + Imaginary
            + Real
            + Select
            + ZeroLike
            + OneLike
            + std::ops::Neg<Output = C::Value>
            + std::ops::Mul<Output = C::Value>
            + std::ops::Div<Output = C::Value>
            + ElementwiseDerivativeAlignment<C::Type>,
    {
        |_operation, context, _driver, inputs| {
            // Away from zero, the real derivative is `d|x| = sign(x) · dx`, while the complex magnitude is a ℂ → ℝ map
            // with `d|z| = Re(z̄ · dz) / |z|`. At the real origin, choose the right derivative and return `dx`. At the
            // complex origin, replace the zero denominator with one so the zero numerator yields zero. These
            // conventions keep the rule finite and stable under higher-order transforms. A structural zero tangent
            // stays symbolic, retyped to the real output's tangent type.
            check_count!("input", inputs, 1, ProgramError);
            let input = &inputs[0];
            let primal = input.primal().abs()?;
            let primal_tangent_type = primal.r#type().tangent()?;
            let tangent = match input.tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal_tangent_type),
                MaybeZero::Value(_) if primal_tangent_type.is_zero_space() => {
                    return Err(ProgramError::UnsupportedOperation {
                        message: format!(
                            "`{}` output type `{}` has no tangent space",
                            ABS_OPERATION_NAME,
                            primal.r#type(),
                        ),
                    }
                    .into());
                }
                MaybeZero::Value(tangent) => {
                    let primal = context.primal_to_tangent(primal.clone())?;
                    let input_primal = context.primal_to_tangent(input.primal().clone())?;
                    if input.primal().r#type().is_complex() {
                        let denominator = primal.align_tangent(&primal_tangent_type, &primal)?;
                        let zero = denominator.zero_like()?;
                        let one = denominator.one_like()?;
                        let denominator_is_zero = denominator.compare(&zero, ComparisonDirection::Equal)?;
                        let denominator = C::Value::select(&denominator_is_zero, &one, &denominator)?;
                        // Normalize `conj(z) / |z|` before multiplying by `dz`. Computing `conj(z) * dz` first is
                        // algebraically equivalent but can overflow even when the final directional derivative is
                        // finite.
                        let conjugate = input_primal.conjugate()?;
                        let real = conjugate.real()? / denominator.clone();
                        let imaginary = conjugate.imaginary()? / denominator.clone();
                        let coefficient = real.complex(&imaginary)?;
                        let input_tangent_type = input.primal().r#type().tangent()?;
                        let tangent = tangent.align_tangent(&input_tangent_type, &input_primal)?;
                        MaybeZero::Value((tangent * coefficient).real()?.align_tangent(&primal_tangent_type, &primal)?)
                    } else {
                        let input = input_primal.align_tangent(&primal_tangent_type, &primal)?;
                        let tangent = tangent.align_tangent(&primal_tangent_type, &primal)?;
                        let zero = input.zero_like()?;
                        let non_negative = input.compare(&zero, ComparisonDirection::GreaterThanOrEqual)?;
                        MaybeZero::Value(C::Value::select(&non_negative, &tangent, &-tangent.clone())?)
                    }
                }
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose = @nonlinear,
}

// TODO(eaplatanios): Review from here onwards.

define_elementwise_capability!(
    @unary
    /// Value capability for elementwise absolute values, returning real magnitudes for complex inputs.
    ///
    /// Eager values compute directly; contextual values bind [`AbsOperation`]. Refer to that operation for
    /// supported input types, broadcasting, and reduction-state requirements. Failures are returned as
    /// [`ProgramError`].
    Abs,
    /// Computes elementwise absolute values, returning real magnitudes for complex inputs. Unsupported input
    /// types, such as Boolean arrays, return a [`ProgramError`].
    abs,
    AbsOperation,
);

impl Abs for Array {
    fn abs(&self) -> Result<Self, ProgramError> {
        // The absolute value of a complex array is its elementwise magnitude, so the element data type maps to its
        // real part data type, mirroring the `AbsOperation` type-inference contract.
        let data_type = match self.r#type().data_type() {
            DataType::C64 => DataType::F32,
            DataType::C128 => DataType::F64,
            other => other,
        };
        let output_type = self.r#type().into_owned().with_data_type(data_type);
        if Self::element_count(&output_type) == 0 {
            let addressing = ArrayAddressing::new(output_type.clone())?;
            return Ok(Self::new_unchecked(output_type, Arc::new(vec![0; addressing.storage_byte_len()])));
        }
        let input_type = self.r#type().data_type();
        if !((input_type.is_signed() && input_type != DataType::I1)
            || input_type.is_floating_point()
            || input_type.is_complex())
        {
            return Err(TypeError::invalid(format!(
                "cannot compute the absolute value of a scalar of data type `{input_type}`",
            ))
            .into());
        }
        dispatch_on_array_element_type!(@numeric input_type, |Element| {
            self.map_elements::<Element, <Element as NumericArrayElement>::Magnitude>(output_type, |value| {
                <Element as NumericArrayElement>::abs(value)
            })
        })
    }
}

/// Implements [`Abs`] for one host primitive type.
macro_rules! impl_abs_for_primitive {
    // Signed integer primitives use checked absolute values so that the `MIN` overflow reports an error instead of
    // wrapping like the XLA-mirroring reference backends do on devices.
    (@signed $type:ty) => {
        impl Abs for $type {
            fn abs(&self) -> Result<Self, ProgramError> {
                self.checked_abs().ok_or_else(|| ProgramError::InvalidArgument {
                    message: format!("`{}` output does not fit in `{}`", ABS_OPERATION_NAME, stringify!($type)),
                })
            }
        }
    };

    // Unsigned integer primitives are their own absolute values.
    (@unsigned $type:ty) => {
        impl Abs for $type {
            fn abs(&self) -> Result<Self, ProgramError> {
                Ok(*self)
            }
        }
    };

    // Floating-point primitives use ordinary IEEE 754 absolute values, which cannot fail.
    (@float $type:ty) => {
        impl Abs for $type {
            fn abs(&self) -> Result<Self, ProgramError> {
                Ok(<$type>::abs(*self))
            }
        }
    };
}

impl_abs_for_primitive!(@signed i8);
impl_abs_for_primitive!(@signed i16);
impl_abs_for_primitive!(@signed i32);
impl_abs_for_primitive!(@signed i64);
impl_abs_for_primitive!(@signed i128);
impl_abs_for_primitive!(@signed isize);
impl_abs_for_primitive!(@unsigned u8);
impl_abs_for_primitive!(@unsigned u16);
impl_abs_for_primitive!(@unsigned u32);
impl_abs_for_primitive!(@unsigned u64);
impl_abs_for_primitive!(@unsigned u128);
impl_abs_for_primitive!(@unsigned usize);
impl_abs_for_primitive!(@float f32);
impl_abs_for_primitive!(@float f64);

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`SignOperation`].
pub const SIGN_OPERATION_NAME: &str = "sign";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that computes the elementwise sign of one value while preserving its array metadata. Matching
    /// the operand constraints of [StableHLO's `sign`](https://openxla.org/stablehlo/spec#sign), signed-integer,
    /// floating-point, and complex operands are supported, while unsigned-integer, Boolean, token, and
    /// structural-zero operands are rejected (unsigned magnitudes carry no sign to extract). Signed integers map to
    /// `-1`, `0`, or `1`; floating-point values map to `-1.0` or `1.0` away from zero while signed zeros and NaNs
    /// pass through unchanged; and complex values map to `z / |z|`, with `0` mapping to `0`. Operands that still
    /// carry partial sums are rejected because the sign of a partial sum is not the sign of the total.
    SignOperation, SIGN_OPERATION_NAME,
    Sign, sign,
    infer_data_types = |input_types: &[DataType]| {
        let input_type = input_types[0];
        if input_type.is_signed() || input_type.is_floating_point() || input_type.is_complex() {
            Ok(vec![input_type])
        } else {
            Err(TypeError::invalid(format!("cannot compute the sign of a value of data type `{input_type}`")))
        }
    },
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation!(@constant SignOperation);

define_elementwise_capability!(
    @unary
    /// Value capability for elementwise signs, preserving floating-point signed zeros and NaNs.
    ///
    /// Eager values compute directly; contextual values bind [`SignOperation`]. Refer to that operation for
    /// supported input types, broadcasting, and reduction-state requirements. Failures are returned as
    /// [`ProgramError`].
    Sign,
    /// Computes [`SignOperation`] elementwise for this value.
    sign,
    SignOperation,
);

impl Sign for Array {
    fn sign(&self) -> Result<Self, ProgramError> {
        if Self::element_count(self.r#type().as_ref()) == 0 {
            let addressing = ArrayAddressing::new(self.r#type().into_owned())?;
            return Ok(Self::new_unchecked(
                self.r#type().into_owned(),
                Arc::new(vec![0; addressing.storage_byte_len()]),
            ));
        }
        let data_type = self.r#type().data_type();
        if !data_type.is_signed() && !data_type.is_floating_point() && !data_type.is_complex() {
            return Err(TypeError::invalid(
                format!("cannot compute the sign of a value of data type `{}`", data_type,),
            )
            .into());
        }
        if data_type.is_signed() {
            dispatch_on_array_element_type!(@signed data_type, |Element| {
                self.map_elements::<Element, Element>(self.r#type().into_owned(), |value| {
                    <Element as NumericArrayElement>::sign(value)
                })
            })
        } else if data_type.is_complex() {
            dispatch_on_array_element_type!(@complex data_type, |Element| {
                self.map_elements::<Element, Element>(self.r#type().into_owned(), |value| {
                    <Element as NumericArrayElement>::sign(value)
                })
            })
        } else {
            dispatch_on_array_element_type!(@float data_type, |Element| {
                self.map_elements::<Element, Element>(self.r#type().into_owned(), |value| {
                    <Element as NumericArrayElement>::sign(value)
                })
            })
        }
    }
}

/// Implements [`Sign`] for one host primitive type.
macro_rules! impl_sign_for_primitive {
    // Signed integer primitives use the ordinary integer signum, which cannot fail.
    (@signed $type:ty) => {
        impl Sign for $type {
            fn sign(&self) -> Result<Self, ProgramError> {
                Ok(<$type>::signum(*self))
            }
        }
    };

    // Floating-point primitives mirror the reference backends. Signed zeros and NaNs are preserved, and every other
    // value maps to `1.0` or `-1.0`.
    (@float $type:ty) => {
        impl Sign for $type {
            fn sign(&self) -> Result<Self, ProgramError> {
                Ok(if self.is_nan() || *self == 0.0 { *self } else { <$type>::signum(*self) })
            }
        }
    };
}

impl_sign_for_primitive!(@signed i8);
impl_sign_for_primitive!(@signed i16);
impl_sign_for_primitive!(@signed i32);
impl_sign_for_primitive!(@signed i64);
impl_sign_for_primitive!(@signed i128);
impl_sign_for_primitive!(@signed isize);
impl_sign_for_primitive!(@float f32);
impl_sign_for_primitive!(@float f64);

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`PowOperation`].
pub const POW_OPERATION_NAME: &str = "pow";

define_elementwise_operation!(
    @binary
    /// [`Operation`] that raises one value to the power of another elementwise (i.e., `(x, y) ↦ x^y`, with the
    /// complex power defined as the principal value `exp(y · log(x))`), promoting their element types and
    /// broadcasting their shapes. Matching the operand constraints of
    /// [StableHLO's `power`](https://openxla.org/stablehlo/spec#power) for those types, only floating-point and
    /// complex operands are supported (Ryft restricts the integer forms to keep the operation differentiable).
    /// Array operands that still carry partial sums are rejected, and their reduced-axis markers must agree.
    PowOperation, POW_OPERATION_NAME,
    Pow, pow,
    check_data_types = [@float],
    check_array_types = [@no_unreduced, @same_reduced_axes],
);

impl_differentiable_elementwise_operation! {
    @binary
    PowOperation,
    jvp<C>
    where
        C::Value: Pow
            + Log
            + Compare<C::Value>
            + Select
            + OneLike
            + ZeroLike
            + ops::Mul<Output = C::Value>
            + ops::Sub<Output = C::Value>,
    {
        // d(x^y) = y · x^{y-1} · dx + x^y · log(x) · dy, with log(x) evaluated at a base of one when x = 0 so
        // that the exponent contribution vanishes instead of producing log(0) = -∞.
        |(left, left_tangent), (right, _)| {
            let exponent = right.clone() - right.one_like()?;
            right * left.pow(&exponent)? * left_tangent
        };
        |(left, _), (right, right_tangent)| {
            let base_is_zero = left.compare(&left.zero_like()?, ComparisonDirection::Equal)?;
            let safe_base = C::Value::select(&base_is_zero, &left.one_like()?, &left)?;
            left.pow(&right)? * safe_base.log()? * right_tangent
        };
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @binary
    /// Value capability for elementwise powers on floating-point and complex inputs.
    ///
    /// Eager values compute directly; contextual values bind [`PowOperation`]. Refer to that operation for
    /// supported input types, broadcasting, and reduction-state requirements. Failures are returned as
    /// [`ProgramError`].
    Pow,
    /// Raises this value to the power `exponent` elementwise, promoting both operands to a common floating-point or
    /// complex element type.
    pow(exponent),
    PowOperation,
);

impl_array_elementwise_operation!(
    @binary
    Pow, pow,
    operation = "pow",
    inputs = @float,
    checks = [@no_unreduced, @same_reduced_axes],
    |lhs, rhs| FloatingPointArrayElement::pow(lhs, rhs),
);

/// Implements [`Pow`] for one host primitive type. Only floating-point primitives are supported, matching the
/// reference backends' float-only power operation.
macro_rules! impl_pow_for_primitive {
    // Implements the capability using the corresponding floating-point primitive function.
    ($type:ty) => {
        impl Pow for $type {
            fn pow(&self, exponent: &Self) -> Result<Self, ProgramError> {
                Ok(<$type>::powf(*self, *exponent))
            }
        }
    };
}

impl_pow_for_primitive!(f32);
impl_pow_for_primitive!(f64);

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`SqrtOperation`].
pub const SQRT_OPERATION_NAME: &str = "sqrt";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that computes the elementwise square root of one value (i.e., `x ↦ √x`, the
    /// principal branch `√z` on complex operands) while preserving its array metadata. Only floating-point and
    /// complex operands are supported, and operands that still carry partial sums are rejected.
    SqrtOperation, SQRT_OPERATION_NAME,
    Sqrt, sqrt,
    check_data_types = [@float],
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation! {
    @unary
    SqrtOperation,
    jvp<C> where C::Value: ops::Add<Output = C::Value> + ops::Div<Output = C::Value> {
        |(_, input_tangent) -> output| input_tangent / (output.clone() + output)
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @unary
    /// Value capability for elementwise principal square roots on floating-point and complex inputs.
    ///
    /// Eager values compute directly; contextual values bind [`SqrtOperation`]. Refer to that operation for
    /// supported input types, broadcasting, and reduction-state requirements. Failures are returned as
    /// [`ProgramError`].
    Sqrt,
    /// Computes [`SqrtOperation`] elementwise for this value.
    sqrt,
    SqrtOperation,
);

impl_array_elementwise_operation!(
    @unary
    Sqrt, sqrt,
    operation = "sqrt",
    inputs = @float,
    checks = [@no_unreduced],
    |input| FloatingPointArrayElement::sqrt(input),
);

/// Implements [`Sqrt`] for one host primitive type.
macro_rules! impl_sqrt_for_primitive {
    // Implements the capability using the corresponding floating-point primitive function.
    ($type:ty) => {
        impl Sqrt for $type {
            fn sqrt(&self) -> Result<Self, ProgramError> {
                Ok(<$type>::sqrt(*self))
            }
        }
    };
}

impl_sqrt_for_primitive!(f32);
impl_sqrt_for_primitive!(f64);

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`RsqrtOperation`].
pub const RSQRT_OPERATION_NAME: &str = "rsqrt";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that computes the elementwise reciprocal square root of one value (i.e., `x ↦ 1/√x`, the
    /// principal branch `1/√z` on complex operands) while preserving its array metadata. Only floating-point and
    /// complex operands are supported, and operands that still carry partial sums are rejected.
    RsqrtOperation, RSQRT_OPERATION_NAME,
    Rsqrt, rsqrt,
    check_data_types = [@float],
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation! {
    @unary
    RsqrtOperation,
    jvp<C>
    where
        C::Value: ops::Add<Output = C::Value>
            + ops::Div<Output = C::Value>
            + ops::Mul<Output = C::Value>
            + ops::Neg<Output = C::Value>,
    {
        // d(rsqrt(x)) = -x^{-3/2} / 2 · dx = -(rsqrt(x) / (x + x)) · dx, reusing the primal output evaluated at
        // the tangent type.
        |(input, input_tangent) -> output| -(output / (input.clone() + input)) * input_tangent
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @unary
    /// Value capability for elementwise reciprocal principal square roots on floating-point and complex inputs.
    ///
    /// Eager values compute directly; contextual values bind [`RsqrtOperation`]. Refer to that operation for
    /// supported input types, broadcasting, and reduction-state requirements. Failures are returned as
    /// [`ProgramError`].
    Rsqrt,
    /// Computes [`RsqrtOperation`] elementwise for this value.
    rsqrt,
    RsqrtOperation,
);

impl_array_elementwise_operation!(
    @unary
    Rsqrt, rsqrt,
    operation = "rsqrt",
    inputs = @float,
    checks = [@no_unreduced],
    |input| FloatingPointArrayElement::rsqrt(input),
);

/// Implements [`Rsqrt`] for one host primitive type.
macro_rules! impl_rsqrt_for_primitive {
    // Implements the capability using the corresponding floating-point primitive function.
    ($type:ty) => {
        impl Rsqrt for $type {
            fn rsqrt(&self) -> Result<Self, ProgramError> {
                Ok(<$type>::sqrt(*self).recip())
            }
        }
    };
}

impl_rsqrt_for_primitive!(f32);
impl_rsqrt_for_primitive!(f64);

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use approx::assert_abs_diff_eq;
    use half::{bf16, f16};
    use indoc::indoc;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayOperation, ArrayType, DataType, Dimension, Layout, LogicalMesh, MeshAxis, MeshAxisType, Shape,
        Sharding, ShardingDimension, StridedLayout, f8e4m3fn, f8e8m0fnu, i2, i4,
    };
    use crate::contexts::EagerContext;
    use crate::differentiation::{
        DifferentiableOperation, DifferentiationContext, DifferentiationDual, differentiate_at,
    };
    use crate::interpretation::InterpretableOperation;
    use crate::macros::{
        check_gradient, check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::operations::constants::one_like::OneLike;
    use crate::operations::manipulation::conversions::ConvertElementType;
    use crate::operations::reductions::{Reduce, ReductionKind};
    use crate::parameters::Placeholder;
    use crate::programs::{EmptyRegionDriver, MaybeZero, Operation, ProgramBuilder, TypeError, Typed};

    use super::*;

    #[test]
    fn test_add_type_inference() {
        check_operation_type_inference!(
            @elementwise @binary,
            operation = AddOperation,
            cases = [
                {
                    input_data_types = [DataType::F32, DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::F8E3M4, DataType::F32],
                    error = format!("`{ADD_OPERATION_NAME}` input types are not broadcast-compatible"),
                },
            ],
        );

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let plain = || {
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
                .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()]).unwrap())
                .unwrap()
        };
        let unreduced = || {
            plain()
                .with_sharding(plain().sharding().unwrap().clone().with_unreduced_axes(["x"]).unwrap())
                .unwrap()
        };

        check_operation_type_inference!(
            operation = AddOperation::<ArrayType>::new(),
            cases = [
                {
                    input_types = [unreduced(), unreduced()],
                    output_types = [unreduced()],
                },
                {
                    input_types = [unreduced(), plain()],
                    error = "`add` operands must be unreduced over the same axes",
                },
                {
                    input_types = [plain(), unreduced()],
                    error = "`add` operands must be unreduced over the same axes",
                },
            ],
        );

        check_operation_type_inference!(
            @reject @mismatched_reduced,
            operation = AddOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_add_interpretation() {
        assert_eq!(
            AddOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.0f32).unwrap(), Array::scalar(3.5f64).unwrap()],
            ),
            Ok(vec![Array::scalar(5.5f64).unwrap()])
        );
        assert_eq!(
            AddOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.0).unwrap(), Array::vector(vec![3.5, -1.0]).unwrap()],
            ),
            Ok(vec![Array::vector(vec![5.5, 1.0]).unwrap()]),
        );
        assert_eq!(
            AddOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[
                    Array::scalar(ComplexNumber::new(1.0f64, 2.0)).unwrap(),
                    Array::scalar(ComplexNumber::new(0.5f64, -1.0)).unwrap()
                ],
            ),
            Ok(vec![Array::scalar(ComplexNumber::new(1.5f64, 1.0)).unwrap()]),
        );
    }

    #[test]
    fn test_add_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = AddOperation::new(),
            inputs = [Array::scalar(2.0).unwrap(), Array::scalar(3.5).unwrap()],
            expected = Array::scalar(5.5).unwrap(),
        );
    }

    #[test]
    fn test_add_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = AddOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::vector(vec![1.0, -2.0]).unwrap()),
                    (@replicated, Array::scalar(3.0).unwrap()),
                ],
                outputs = [(@mapped(axis = 0), Array::vector(vec![4.0, 1.0]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_add_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = AddOperation::new(),
            cases = [{
                primals = [Array::scalar(2.0).unwrap(), Array::scalar(5.0).unwrap()],
                tangents = [Array::scalar(3.0).unwrap(), Array::scalar(-1.0).unwrap()],
                primal_outputs = [Array::scalar(7.0).unwrap()],
                tangent_outputs = [Array::scalar(2.0).unwrap()],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                    let %4:f64[] = add %0 %1
                        %5:f64[] = add %2 %3
                    in (%4, %5)
                "},
            }],
        );
    }

    #[test]
    fn test_add_differentiation_low_precision() {
        // Rank-zero arrays support both half-precision variants through the ordinary array operations.
        assert_eq!(
            differentiate_at(Array::scalar(bf16::from_f32(3.0)).unwrap())
                .jvp(Array::scalar(bf16::ONE).unwrap(), |x| Ok(x.clone() + x)),
            Ok((Array::scalar(bf16::from_f32(6.0)).unwrap(), Array::scalar(bf16::from_f32(2.0)).unwrap())),
        );
        assert_eq!(
            differentiate_at(Array::scalar(f16::from_f32(3.0)).unwrap())
                .jvp(Array::scalar(f16::ONE).unwrap(), |x| Ok(x.clone() + x)),
            Ok((Array::scalar(f16::from_f32(6.0)).unwrap(), Array::scalar(f16::from_f32(2.0)).unwrap())),
        );
    }

    #[test]
    fn test_add_transposition() {
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        check_operation_transposition!(
            @exact,
            operation = AddOperation::new(),
            cases = [
                {
                    inputs = [
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                    ],
                    output_cotangents = [Array::scalar(3.0).unwrap()],
                    input_cotangents = [Array::scalar(3.0).unwrap(), Array::scalar(3.0).unwrap()],
                    pullback = indoc! {"
                        lambda %0:f64[] .
                        in (%0, %0)
                    "},
                },
                {
                    inputs = [
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                        (@linear(type = vector_type.clone())),
                    ],
                    output_cotangents = [Array::from_elements::<f64>(vector_type.clone(), &[2.0, 3.0, 4.0]).unwrap()],
                    input_cotangents = [
                        Array::scalar(9.0).unwrap(),
                        Array::from_elements::<f64>(vector_type, &[2.0, 3.0, 4.0]).unwrap(),
                    ],
                    pullback = indoc! {"
                        lambda %0:f64[3] .
                        let %1:f64[] = reduce_sum [axes=[0]] %0
                        in (%1, %0)
                    "},
                },
            ],
        );
    }

    #[test]
    fn test_array_add() {
        let vector = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(vector.add(&Array::scalar(1.0).unwrap()).unwrap(), Array::vector(vec![2.0, 3.0, 4.0]).unwrap());
        // Mixed-precision operands promote to the common element data type.
        let promoted =
            Array::vector(vec![1.0f32, 2.0]).unwrap().add(&Array::vector(vec![0.5f64, 0.5]).unwrap()).unwrap();
        assert_eq!(promoted, Array::vector(vec![1.5f64, 2.5]).unwrap());
        // General broadcasting traverses arbitrary input layouts while mixed element types normalize through the
        // canonical conversion kernel.
        let left_type =
            ArrayType::new_static(DataType::F32, [2, 1]).with_layout(Layout::Strided(StridedLayout::new(vec![-8, 4])));
        let left = Array::from_elements(left_type, &[1.0f32, 2.0]).unwrap();
        let right_type =
            ArrayType::new_static(DataType::F64, [1, 3]).with_layout(Layout::Strided(StridedLayout::new(vec![24, -8])));
        let right = Array::from_elements(right_type, &[0.5f64, 1.0, 1.5]).unwrap();
        let sum = left.add(&right).unwrap();
        assert_eq!(sum.r#type().into_owned(), ArrayType::new_static(DataType::F64, [2, 3]));
        assert_eq!(sum.elements::<f64>(), Ok(vec![1.5, 2.0, 2.5, 2.5, 3.0, 3.5]));
        // The `std::ops` sugar delegates to the fallible capability.
        assert_eq!(vector.clone() + Array::scalar(1.0).unwrap(), Array::vector(vec![2.0, 3.0, 4.0]).unwrap());
        // Integer arithmetic wraps deterministically, matching the scalar reference backend.
        let wrapped = Array::vector(vec![255u8]).unwrap().add(&Array::vector(vec![1u8]).unwrap()).unwrap();
        assert_eq!(wrapped.elements::<u8>(), Ok(vec![0]));
    }

    #[test]
    fn test_array_add_manual_variation() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["m"]).unwrap())
            .unwrap();
        let varying = Array::from_elements(varying_type.clone(), &[2f32]).unwrap();
        let invariant = Array::scalar(1f32).unwrap();
        let expected = Err(TypeError::invalid(
            "`add` inputs must have matching varying manual axes; insert `parallel_vary` on the inputs that lack an \
             axis, as `align_manual_variation` does",
        )
        .into());
        assert_eq!(invariant.add(&varying), expected);
        assert_eq!(varying.add(&invariant), expected);
        assert_eq!(varying.add(&varying), Array::from_elements(varying_type, &[4f32]));

        let empty_type = ArrayType::new_static(DataType::F32, [0]);
        let empty = Array::from_elements::<f32>(empty_type.clone(), &[]).unwrap();
        let varying_empty = Array::from_elements::<f32>(
            empty_type
                .with_sharding(Sharding::replicated(mesh, 1).with_varying_manual_axes(["m"]).unwrap())
                .unwrap(),
            &[],
        )
        .unwrap();
        assert_eq!(empty.add(&varying_empty), expected);
    }

    #[test]
    fn test_array_add_low_precision() {
        // Low-precision arithmetic computes through decoded values and re-encodes the nearest representable result.
        let left = Array::from_elements::<f8e4m3fn>(
            ArrayType::new_static(DataType::F8E4M3FN, [2]),
            &[1.0, 2.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        let right = Array::from_elements::<f8e4m3fn>(
            ArrayType::new_static(DataType::F8E4M3FN, [2]),
            &[0.5, 0.25].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        let sum = left.add(&right).unwrap();
        assert_eq!(sum.r#type().into_owned(), ArrayType::new_static(DataType::F8E4M3FN, [2]));
        assert_eq!(sum.to_f64s(), vec![1.5, 2.25]);
    }

    #[test]
    fn test_array_add_complex() {
        // Elementwise complex math decodes and encodes the complex element types directly.
        let left = Array::vector(vec![ComplexNumber::new(1.0f64, 2.0), ComplexNumber::new(0.5f64, -1.0)]).unwrap();
        let right = Array::vector(vec![ComplexNumber::new(0.5f64, -1.0), ComplexNumber::new(2.0f64, 0.5)]).unwrap();
        let left_values = [ComplexNumber::new(1.0f64, 2.0), ComplexNumber::new(0.5f64, -1.0)];
        let right_values = [ComplexNumber::new(0.5f64, -1.0), ComplexNumber::new(2.0f64, 0.5)];
        assert_eq!(
            left.add(&right).unwrap(),
            Array::vector(vec![left_values[0] + right_values[0], left_values[1] + right_values[1]]).unwrap(),
        );
    }

    #[test]
    fn test_array_add_integers() {
        // Sub-byte arithmetic wraps using the declared bit width.
        let narrow = Array::vector(vec![i4::new(7).unwrap(), i4::new(-8).unwrap()]).unwrap();
        assert_eq!(
            narrow.add(&Array::scalar(i4::new(1).unwrap()).unwrap()).unwrap().elements::<i4>(),
            Ok(vec![i4::MIN, i4::new(-7).unwrap()]),
        );
    }

    #[test]
    fn test_add_primitives() {
        assert_eq!(Add::add(&2usize, &3), Ok(5));
        assert_eq!(Add::add(&-2i32, &3), Ok(1));
        assert_eq!(
            Add::add(&i8::MAX, &1),
            Err(ProgramError::InvalidArgument { message: "`add` output does not fit in `i8`".to_string() }),
        );
        assert_eq!(Add::add(&2.5f64, &0.5), Ok(3.0));
    }

    #[test]
    fn test_sub_type_inference() {
        check_operation_type_inference!(
            @elementwise @binary,
            operation = SubOperation,
            cases = [
                {
                    input_data_types = [DataType::F32, DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::F8E3M4, DataType::F32],
                    error = format!("`{SUB_OPERATION_NAME}` input types are not broadcast-compatible"),
                },
            ],
        );

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let plain = || {
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
                .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()]).unwrap())
                .unwrap()
        };
        let unreduced = || {
            plain()
                .with_sharding(plain().sharding().unwrap().clone().with_unreduced_axes(["x"]).unwrap())
                .unwrap()
        };

        check_operation_type_inference!(
            operation = SubOperation::<ArrayType>::new(),
            cases = [
                {
                    input_types = [unreduced(), unreduced()],
                    output_types = [unreduced()],
                },
                {
                    input_types = [unreduced(), plain()],
                    error = "`sub` operands must be unreduced over the same axes",
                },
                {
                    input_types = [plain(), unreduced()],
                    error = "`sub` operands must be unreduced over the same axes",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @mismatched_reduced,
            operation = SubOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_sub_interpretation() {
        let operation = SubOperation::<ArrayType>::new();

        assert_eq!(
            operation.interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.0f32).unwrap(), Array::scalar(3.5f64).unwrap()],
            ),
            Ok(vec![Array::scalar(-1.5f64).unwrap()])
        );
        assert_eq!(
            SubOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.0).unwrap(), Array::scalar(3.5).unwrap()],
            ),
            Ok(vec![Array::scalar(-1.5).unwrap()]),
        );
        assert_eq!(
            operation.interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[
                    Array::scalar(ComplexNumber::new(1.0f64, 2.0)).unwrap(),
                    Array::scalar(ComplexNumber::new(0.5f64, -1.0)).unwrap()
                ],
            ),
            Ok(vec![Array::scalar(ComplexNumber::new(0.5f64, 3.0)).unwrap()]),
        );
    }

    #[test]
    fn test_sub_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = SubOperation::new(),
            inputs = [Array::scalar(2.0).unwrap(), Array::scalar(3.5).unwrap()],
            expected = Array::scalar(-1.5).unwrap(),
        );
    }

    #[test]
    fn test_sub_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = SubOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::vector(vec![1.0, -2.0]).unwrap()),
                    (@replicated, Array::scalar(3.0).unwrap()),
                ],
                outputs = [(@mapped(axis = 0), Array::vector(vec![-2.0, -5.0]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_sub_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = SubOperation::new(),
            cases = [{
                primals = [Array::scalar(5.0).unwrap(), Array::scalar(2.0).unwrap()],
                tangents = [Array::scalar(3.0).unwrap(), Array::scalar(1.0).unwrap()],
                primal_outputs = [Array::scalar(3.0).unwrap()],
                tangent_outputs = [Array::scalar(2.0).unwrap()],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                    let %4:f64[] = sub %0 %1
                        %5:f64[] = sub %2 %3
                    in (%4, %5)
                "},
            }],
        );
    }

    #[test]
    fn test_sub_transposition() {
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        check_operation_transposition!(
            @exact,
            operation = SubOperation::new(),
            cases = [
                {
                    inputs = [
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                    ],
                    output_cotangents = [Array::scalar(3.0).unwrap()],
                    input_cotangents = [Array::scalar(3.0).unwrap(), Array::scalar(-3.0).unwrap()],
                    pullback = indoc! {"
                        lambda %0:f64[] .
                        let %1:f64[] = neg %0
                        in (%0, %1)
                    "},
                },
                {
                    inputs = [
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                        (@linear(type = vector_type.clone())),
                    ],
                    output_cotangents = [Array::from_elements::<f64>(vector_type.clone(), &[2.0, 3.0, 4.0]).unwrap()],
                    input_cotangents = [
                        Array::scalar(9.0).unwrap(),
                        Array::from_elements::<f64>(vector_type, &[-2.0, -3.0, -4.0]).unwrap(),
                    ],
                    pullback = indoc! {"
                        lambda %0:f64[3] .
                        let %1:f64[] = reduce_sum [axes=[0]] %0
                            %2:f64[3] = neg %0
                        in (%1, %2)
                    "},
                },
            ],
        );
    }

    #[test]
    fn test_array_sub() {
        let vector = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(
            vector.sub(&Array::vector(vec![0.5, 1.0, 1.5]).unwrap()).unwrap(),
            Array::vector(vec![0.5, 1.0, 1.5]).unwrap()
        );
    }

    #[test]
    fn test_array_sub_low_precision() {
        // Low-precision arithmetic computes through decoded values and re-encodes the nearest representable result.
        let left = Array::from_elements::<f8e4m3fn>(
            ArrayType::new_static(DataType::F8E4M3FN, [2]),
            &[1.0, 2.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        let right = Array::from_elements::<f8e4m3fn>(
            ArrayType::new_static(DataType::F8E4M3FN, [2]),
            &[0.5, 0.25].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        assert_eq!(left.sub(&right).unwrap().to_f64s(), vec![0.5, 1.75]);
    }

    #[test]
    fn test_array_sub_complex() {
        // Elementwise complex math decodes and encodes the complex element types directly.
        let left = Array::vector(vec![ComplexNumber::new(1.0f64, 2.0), ComplexNumber::new(0.5f64, -1.0)]).unwrap();
        let right = Array::vector(vec![ComplexNumber::new(0.5f64, -1.0), ComplexNumber::new(2.0f64, 0.5)]).unwrap();
        let left_values = [ComplexNumber::new(1.0f64, 2.0), ComplexNumber::new(0.5f64, -1.0)];
        let right_values = [ComplexNumber::new(0.5f64, -1.0), ComplexNumber::new(2.0f64, 0.5)];
        assert_eq!(
            left.sub(&right).unwrap(),
            Array::vector(vec![left_values[0] - right_values[0], left_values[1] - right_values[1]]).unwrap(),
        );
    }

    #[test]
    fn test_array_sub_integers() {
        // Sub-byte arithmetic uses the declared bit width for every wrapping operation.
        let narrow = Array::vector(vec![i4::new(7).unwrap(), i4::new(-8).unwrap()]).unwrap();
        assert_eq!(
            narrow.sub(&Array::scalar(i4::new(1).unwrap()).unwrap()).unwrap().elements::<i4>(),
            Ok(vec![i4::new(6).unwrap(), i4::new(7).unwrap()]),
        );
    }

    #[test]
    fn test_sub_primitives() {
        assert_eq!(Sub::sub(&5usize, &3), Ok(2));
        assert_eq!(
            Sub::sub(&0usize, &1),
            Err(ProgramError::InvalidArgument { message: "`sub` output does not fit in `usize`".to_string() }),
        );
        assert_eq!(Sub::sub(&2.5f32, &0.5), Ok(2.0));
    }

    #[test]
    fn test_mul_type_inference() {
        check_operation_type_inference!(
            @elementwise @binary,
            operation = MulOperation,
            cases = [
                {
                    input_data_types = [DataType::F32, DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::F8E3M4, DataType::F32],
                    error = format!("`{MUL_OPERATION_NAME}` input types are not broadcast-compatible"),
                },
            ],
        );

        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let vector_type = || ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));
        let unreduced = |axis: &str| {
            vector_type()
                .with_sharding(
                    Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
                        .unwrap()
                        .with_unreduced_axes([axis])
                        .unwrap(),
                )
                .unwrap()
        };
        let reduced = |axis: &str| {
            vector_type()
                .with_sharding(
                    Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
                        .unwrap()
                        .with_reduced_axes([axis])
                        .unwrap(),
                )
                .unwrap()
        };

        // Unreduced over `x` times reduced over `x` is the partial-sum-times-replicated case: the product stays
        // unreduced over `x`, and the reduced marker is cleared.
        let output = <MulOperation<ArrayType> as Operation>::infer_output_types(
            &MulOperation::new(),
            &[unreduced("x"), reduced("x")],
            &[],
        )
        .unwrap();
        assert_eq!(output[0].sharding().unwrap().unreduced_axes(), &BTreeSet::from(["x".to_string()]));
        assert_eq!(output[0].sharding().unwrap().reduced_axes(), &BTreeSet::new());

        // Two operands both unreduced cannot be multiplied (the product of two partial sums is not a partial sum).
        check_operation_type_inference!(
            operation = MulOperation::<ArrayType>::new(),
            cases = [{
                input_types = [unreduced("x"), unreduced("x")],
                error = format!("`{MUL_OPERATION_NAME}` cannot multiply two operands that are both unreduced"),
            }],
        );

        // Unreduced over `x` requires the other operand to be reduced over exactly `x`, not a different axis.
        check_operation_type_inference!(
            operation = MulOperation::<ArrayType>::new(),
            cases = [{
                input_types = [unreduced("x"), reduced("y")],
                error = format!(
                    "`{MUL_OPERATION_NAME}` requires the second operand to be reduced over the axes the first is \
                     unreduced over",
                ),
            }],
        );

        // Two operands reduced over the same axis multiply to a value reduced over that axis.
        let output = <MulOperation<ArrayType> as Operation>::infer_output_types(
            &MulOperation::new(),
            &[reduced("x"), reduced("x")],
            &[],
        )
        .unwrap();
        assert_eq!(output[0].sharding().unwrap().reduced_axes(), &BTreeSet::from(["x".to_string()]));
        assert_eq!(output[0].sharding().unwrap().unreduced_axes(), &BTreeSet::new());

        // A reduced operand cannot be multiplied by an otherwise replicated operand because the result would inherit
        // a reduction marker that does not describe both inputs.
        check_operation_type_inference!(
            operation = MulOperation::<ArrayType>::new(),
            cases = [{
                input_types = [reduced("x"), vector_type()],
                error = format!("`{MUL_OPERATION_NAME}` operands must be reduced over the same axes"),
            }],
        );
    }

    #[test]
    fn test_mul_interpretation() {
        let operation = MulOperation::<ArrayType>::new();

        assert_eq!(
            operation.interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.0f32).unwrap(), Array::scalar(3.5f64).unwrap()],
            ),
            Ok(vec![Array::scalar(7.0f64).unwrap()]),
        );
        assert_eq!(
            MulOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.0).unwrap(), Array::scalar(3.5).unwrap()],
            ),
            Ok(vec![Array::scalar(7.0).unwrap()]),
        );
        assert_eq!(
            operation.interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[
                    Array::scalar(ComplexNumber::new(1.0f64, 2.0)).unwrap(),
                    Array::scalar(ComplexNumber::new(0.5f64, -1.0)).unwrap()
                ],
            ),
            Ok(vec![Array::scalar(ComplexNumber::new(1.0f64, 2.0) * ComplexNumber::new(0.5f64, -1.0)).unwrap()]),
        );
        assert_eq!(Mul::mul(&3usize, &4), Ok(12));
        assert_eq!(
            Mul::mul(&usize::MAX, &2),
            Err(ProgramError::InvalidArgument { message: "`mul` output does not fit in `usize`".to_string() }),
        );
    }

    #[test]
    fn test_mul_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = MulOperation::new(),
            inputs = [Array::scalar(2.0).unwrap(), Array::scalar(3.5).unwrap()],
            expected = Array::scalar(7.0).unwrap(),
        );
    }

    #[test]
    fn test_mul_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = MulOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::vector(vec![1.0, -2.0]).unwrap()),
                    (@replicated, Array::scalar(3.0).unwrap()),
                ],
                outputs = [(@mapped(axis = 0), Array::vector(vec![3.0, -6.0]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_mul_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = MulOperation::new(),
            cases = [{
                primals = [Array::scalar(2.0).unwrap(), Array::scalar(5.0).unwrap()],
                tangents = [Array::scalar(3.0).unwrap(), Array::scalar(-1.0).unwrap()],
                primal_outputs = [Array::scalar(10.0).unwrap()],
                tangent_outputs = [Array::scalar(13.0).unwrap()],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                    let %4:f64[] = mul %0 %1
                        %5:f64[] = mul %1 %2
                        %6:f64[] = mul %0 %3
                        %7:f64[] = add %5 %6
                    in (%4, %7)
                "},
            }],
        );
    }

    #[test]
    fn test_mul_differentiation_complex() {
        // Complex arrays differentiate through the same rule: the eager JVP computes `l·dr + dl·r` elementwise over
        // `c128` payloads, and the reverse-mode pullback applies the bilinear (conjugation-free) transpose pairing.
        let left = ComplexNumber::new(1.0f64, 2.0);
        let right = ComplexNumber::new(0.5f64, -1.0);
        let left_tangent = ComplexNumber::new(-0.5f64, 0.25);
        let right_tangent = ComplexNumber::new(2.0f64, 1.0);
        let (primal, tangent) =
            differentiate_at((Array::vector(vec![left]).unwrap(), Array::vector(vec![right]).unwrap()))
                .jvp(
                    (Array::vector(vec![left_tangent]).unwrap(), Array::vector(vec![right_tangent]).unwrap()),
                    |(left, right)| Ok(left * right),
                )
                .unwrap();
        assert_eq!(primal, Array::vector(vec![left * right]).unwrap());
        assert_eq!(tangent, Array::vector(vec![left_tangent * right + left * right_tangent]).unwrap());
        let (_, pullback) = differentiate_at((Array::vector(vec![left]).unwrap(), Array::vector(vec![right]).unwrap()))
            .vjp(|(left, right)| Ok(left * right))
            .unwrap();
        let cotangent = ComplexNumber::new(0.5f64, 3.0);
        let (left_cotangent, right_cotangent) = pullback.apply(Array::vector(vec![cotangent]).unwrap()).unwrap();
        assert_eq!(left_cotangent, Array::vector(vec![cotangent * right]).unwrap());
        assert_eq!(right_cotangent, Array::vector(vec![cotangent * left]).unwrap());
    }

    #[test]
    fn test_mul_transposition() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        check_operation_transposition!(
            @exact,
            operation = MulOperation::new(),
            cases = [
                {
                    inputs = [
                        (@known, Array::scalar(4.0).unwrap()),
                        (@linear(type = scalar_type.clone())),
                    ],
                    output_cotangents = [Array::scalar(1.0).unwrap()],
                    input_cotangents = [Array::scalar(4.0).unwrap()],
                    pullback = indoc! {"
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = mul %1 %0
                        in (%2)
                    "},
                },
                {
                    inputs = [
                        (@known, Array::from_elements::<f64>(vector_type.clone(), &[1.0, 2.0, 3.0]).unwrap()),
                        (@linear(type = scalar_type)),
                    ],
                    output_cotangents = [Array::from_elements::<f64>(vector_type.clone(), &[2.0, 3.0, 4.0]).unwrap()],
                    input_cotangents = [Array::scalar(20.0).unwrap()],
                    pullback = indoc! {"
                        lambda %0:f64[3], %1:f64[3] .
                        let %2:f64[3] = mul %1 %0
                            %3:f64[] = reduce_sum [axes=[0]] %2
                        in (%3)
                    "},
                },
            ],
        );

        // A reduced primal coefficient stays reduced when scaling an unreduced output cotangent.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let reduced_type = ArrayType::new_static(DataType::F64, [2])
            .with_sharding(Sharding::replicated(mesh, 1).with_reduced_axes(["x"]).unwrap())
            .unwrap();
        let cotangent_type = reduced_type.cotangent().unwrap();
        let coefficient = Array::from_elements(reduced_type.clone(), &[2.0f64, 3.0]).unwrap();
        let cotangent = Array::from_elements(cotangent_type.clone(), &[4.0f64, 5.0]).unwrap();
        let expected = Array::from_elements(cotangent_type, &[8.0f64, 15.0]).unwrap();
        check_operation_transposition!(
            @exact,
            operation = MulOperation::new(),
            cases = [
                {
                    inputs = [(@known, coefficient.clone()), (@linear(type = reduced_type.clone()))],
                    output_cotangents = [cotangent.clone()],
                    input_cotangents = [expected.clone()],
                },
                {
                    inputs = [(@linear(type = reduced_type)), (@known, coefficient)],
                    output_cotangents = [cotangent],
                    input_cotangents = [expected],
                },
            ],
        );
    }

    #[test]
    fn test_array_mul() {
        let vector = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(Mul::mul(&vector, &vector).unwrap(), Array::vector(vec![1.0, 4.0, 9.0]).unwrap());
        // Scaling by an `f64` preserves the array's element data type.
        let scaled = Array::vector(vec![1.0f32, 2.0]).unwrap() * 2.0;
        assert_eq!(scaled, Array::vector(vec![2.0f32, 4.0]).unwrap());
    }

    #[test]
    fn test_array_mul_reduction_state() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh, vec![ShardingDimension::replicated()]).unwrap();
        let partial_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(sharding.clone().with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let reduced_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(sharding.with_reduced_axes(["x"]).unwrap())
            .unwrap();
        let lhs = Array::from_elements(partial_type.clone(), &[2.0f32, 3.0]).unwrap();
        let rhs = Array::from_elements(reduced_type, &[4.0f32, 5.0]).unwrap();
        let expected = Array::from_elements(partial_type, &[8.0f32, 15.0]).unwrap();

        // A partial sum times an operand reduced over the same mesh axes remains a partial sum in either order.
        assert_eq!(Mul::mul(&lhs, &rhs), Ok(expected.clone()));
        assert_eq!(Mul::mul(&rhs, &lhs), Ok(expected));

        // Multiplying two partial sums is invalid, including when no scalar evaluation would occur.
        assert!(matches!(
            Mul::mul(&lhs, &lhs),
            Err(ProgramError::Type(TypeError::Invalid { message, .. }))
                if message == "`mul` cannot multiply two operands that are both unreduced",
        ));
        let empty_type = lhs.r#type().into_owned().with_shape(Shape::new(vec![Dimension::Static(0)]));
        let empty = Array::from_elements(empty_type, &[] as &[f32]).unwrap();
        assert!(matches!(
            Mul::mul(&empty, &empty),
            Err(ProgramError::Type(TypeError::Invalid { message, .. }))
                if message == "`mul` cannot multiply two operands that are both unreduced",
        ));
    }

    #[test]
    fn test_array_mul_low_precision() {
        // Low-precision arithmetic computes through decoded values and re-encodes the nearest representable result.
        let left = Array::from_elements::<f8e4m3fn>(
            ArrayType::new_static(DataType::F8E4M3FN, [2]),
            &[1.0, 2.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        let right = Array::from_elements::<f8e4m3fn>(
            ArrayType::new_static(DataType::F8E4M3FN, [2]),
            &[0.5, 0.25].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        assert_eq!(Mul::mul(&left, &right).unwrap().to_f64s(), vec![0.5, 0.5]);
    }

    #[test]
    fn test_array_mul_complex() {
        // Elementwise complex math decodes and encodes the complex element types directly.
        let left = Array::vector(vec![ComplexNumber::new(1.0f64, 2.0), ComplexNumber::new(0.5f64, -1.0)]).unwrap();
        let right = Array::vector(vec![ComplexNumber::new(0.5f64, -1.0), ComplexNumber::new(2.0f64, 0.5)]).unwrap();
        let left_values = [ComplexNumber::new(1.0f64, 2.0), ComplexNumber::new(0.5f64, -1.0)];
        let right_values = [ComplexNumber::new(0.5f64, -1.0), ComplexNumber::new(2.0f64, 0.5)];
        assert_eq!(
            Mul::mul(&left, &right).unwrap(),
            Array::vector(vec![left_values[0] * right_values[0], left_values[1] * right_values[1]]).unwrap(),
        );
    }

    #[test]
    fn test_mul_primitives() {
        assert_eq!(Mul::mul(&3usize, &4), Ok(12));
        assert_eq!(
            Mul::mul(&i8::MAX, &2),
            Err(ProgramError::InvalidArgument { message: "`mul` output does not fit in `i8`".to_string() }),
        );
        assert_eq!(Mul::mul(&2.5f64, &4.0), Ok(10.0));
    }

    #[test]
    fn test_div_type_inference() {
        check_operation_type_inference!(
            @elementwise @binary,
            operation = DivOperation,
            cases = [
                {
                    input_data_types = [DataType::F32, DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::F8E3M4, DataType::F32],
                    error = format!("`{DIV_OPERATION_NAME}` input types are not broadcast-compatible"),
                },
            ],
        );

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let plain = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));
        let unreduced = plain
            .clone()
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_unreduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        let reduced = plain
            .clone()
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_reduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        let unreduced_error = || "`div` does not support unreduced operands";
        check_operation_type_inference!(
            operation = DivOperation::<ArrayType>::new(),
            cases = [
                {
                    input_types = [unreduced.clone(), plain.clone()],
                    error = unreduced_error(),
                },
                {
                    input_types = [plain, unreduced.clone()],
                    error = unreduced_error(),
                },
                {
                    input_types = [unreduced, reduced],
                    error = unreduced_error(),
                },
            ],
        );
        check_operation_type_inference!(
            @reject @mismatched_reduced,
            operation = DivOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_div_interpretation() {
        let operation = DivOperation::<ArrayType>::new();

        assert_eq!(
            operation.interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(7.0f32).unwrap(), Array::scalar(2.0f64).unwrap()],
            ),
            Ok(vec![Array::scalar(3.5f64).unwrap()]),
        );
        assert_eq!(
            DivOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(7.0).unwrap(), Array::scalar(2.0).unwrap()],
            ),
            Ok(vec![Array::scalar(3.5).unwrap()]),
        );
        assert_abs_diff_eq!(
            match operation.interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[
                    Array::scalar(ComplexNumber::new(1.0f64, 2.0)).unwrap(),
                    Array::scalar(ComplexNumber::new(0.5f64, -1.0)).unwrap()
                ],
            ) {
                Ok(outputs) => outputs[0].clone(),
                Err(error) => panic!("expected a complex quotient but got {error}"),
            },
            Array::scalar(ComplexNumber::new(1.0f64, 2.0) / ComplexNumber::new(0.5f64, -1.0)).unwrap(),
            epsilon = 1e-12,
        );
    }

    #[test]
    fn test_div_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = DivOperation::new(),
            inputs = [Array::scalar(7.0).unwrap(), Array::scalar(2.0).unwrap()],
            expected = Array::scalar(3.5).unwrap(),
        );
    }

    #[test]
    fn test_div_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = DivOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::vector(vec![3.0, -6.0]).unwrap()),
                    (@replicated, Array::scalar(3.0).unwrap()),
                ],
                outputs = [(@mapped(axis = 0), Array::vector(vec![1.0, -2.0]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_div_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = DivOperation::new(),
            cases = [{
                primals = [Array::scalar(6.0).unwrap(), Array::scalar(2.0).unwrap()],
                tangents = [Array::scalar(3.0).unwrap(), Array::scalar(4.0).unwrap()],
                primal_outputs = [Array::scalar(3.0).unwrap()],
                tangent_outputs = [Array::scalar(-4.5).unwrap()],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                    let %4:f64[] = div %0 %1
                        %5:f64[] = div %2 %1
                        %6:f64[] = mul %1 %1
                        %7:f64[] = div %0 %6
                        %8:f64[] = neg %7
                        %9:f64[] = mul %8 %3
                        %10:f64[] = add %5 %9
                    in (%4, %10)
                "},
            }],
        );
    }

    #[test]
    fn test_div_differentiation_preserves_small_tangents() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let smallest_positive = f64::from_bits(1);
        let outputs = DivOperation::<ArrayType>::new()
            .jvp(
                &DifferentiationContext::fused(context.clone()),
                &EmptyRegionDriver,
                &[
                    DifferentiationDual::new(Array::scalar(0.0).unwrap(), Array::scalar(smallest_positive).unwrap())
                        .unwrap(),
                    DifferentiationDual::new_with_zero_tangent(Array::scalar(smallest_positive).unwrap()).unwrap(),
                ],
            )
            .unwrap();
        match outputs[0].tangent() {
            MaybeZero::Value(tangent) => assert_eq!(tangent, &Array::scalar(1.0).unwrap()),
            MaybeZero::Zero(_) => panic!("expected a live division tangent"),
        }
    }

    #[test]
    fn test_div_differentiation_low_precision_uses_widened_tangents() {
        let left = Array::scalar(4.0f32).unwrap().convert_element_type(DataType::F8E8M0FNU).unwrap();
        let right = Array::scalar(2.0f32).unwrap().convert_element_type(DataType::F8E8M0FNU).unwrap();
        let (primal, tangent) = differentiate_at((left, right))
            .jvp((Array::scalar(1.0f32).unwrap(), Array::scalar(1.0f32).unwrap()), |(left, right)| Ok(left / right))
            .unwrap();
        assert_eq!(primal, Array::scalar(2.0f32).unwrap().convert_element_type(DataType::F8E8M0FNU).unwrap());
        assert_eq!(tangent, Array::scalar(-0.5f32).unwrap());
    }

    #[test]
    fn test_div_transposition() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        check_operation_transposition!(
            @approx(epsilon = 1e-12),
            operation = DivOperation::new(),
            cases = [
                {
                    inputs = [
                        (@linear(type = scalar_type.clone())),
                        (@known, Array::scalar(3.0).unwrap()),
                    ],
                    output_cotangents = [Array::scalar(2.0).unwrap()],
                    input_cotangents = [Array::scalar(2.0 / 3.0).unwrap()],
                    pullback = indoc! {"
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = div %0 %1
                        in (%2)
                    "},
                },
                {
                    inputs = [
                        (@linear(type = scalar_type)),
                        (@known, Array::from_elements::<f64>(vector_type.clone(), &[2.0, 4.0, 5.0]).unwrap()),
                    ],
                    output_cotangents = [Array::from_elements::<f64>(vector_type.clone(), &[2.0, 4.0, 10.0]).unwrap()],
                    input_cotangents = [Array::scalar(4.0).unwrap()],
                    pullback = indoc! {"
                        lambda %0:f64[3], %1:f64[3] .
                        let %2:f64[3] = div %0 %1
                            %3:f64[] = reduce_sum [axes=[0]] %2
                        in (%3)
                    "},
                },
            ],
        );
    }

    #[test]
    fn test_array_div() {
        let vector = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(
            Div::div(&vector, &Array::scalar(2.0).unwrap()).unwrap(),
            Array::vector(vec![0.5, 1.0, 1.5]).unwrap()
        );
    }

    #[test]
    fn test_array_div_low_precision() {
        // Low-precision arithmetic computes through decoded values and re-encodes the nearest representable result.
        let left = Array::from_elements::<f8e4m3fn>(
            ArrayType::new_static(DataType::F8E4M3FN, [2]),
            &[1.0, 2.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        let right = Array::from_elements::<f8e4m3fn>(
            ArrayType::new_static(DataType::F8E4M3FN, [2]),
            &[0.5, 0.25].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        assert_eq!(Div::div(&left, &right).unwrap().to_f64s(), vec![2.0, 8.0]);
    }

    #[test]
    fn test_array_div_complex() {
        // Elementwise complex math decodes and encodes the complex element types directly.
        let left = Array::vector(vec![ComplexNumber::new(1.0f64, 2.0), ComplexNumber::new(0.5f64, -1.0)]).unwrap();
        let right = Array::vector(vec![ComplexNumber::new(0.5f64, -1.0), ComplexNumber::new(2.0f64, 0.5)]).unwrap();
        let left_values = [ComplexNumber::new(1.0f64, 2.0), ComplexNumber::new(0.5f64, -1.0)];
        let right_values = [ComplexNumber::new(0.5f64, -1.0), ComplexNumber::new(2.0f64, 0.5)];
        assert_abs_diff_eq!(
            Div::div(&left, &right).unwrap(),
            Array::vector(vec![left_values[0] / right_values[0], left_values[1] / right_values[1]]).unwrap(),
            epsilon = 1e-12,
        );
        // Ratio-based division can still overflow when both denominator components are near the largest value.
        let large = Array::scalar(ComplexNumber::new(1e308f64, 1e308)).unwrap();
        let quotient = Div::div(&large, &large).unwrap().elements::<ComplexNumber<f64>>().unwrap()[0];
        assert!(quotient.re.is_nan());
        assert_eq!(quotient.im.to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn test_array_div_integers() {
        // Exceptional integer division inputs return the same structured errors as native-width array
        // arithmetic rather than panicking.
        assert!(matches!(
            Div::div(&Array::vector(vec![1i32]).unwrap(), &Array::vector(vec![0i32]).unwrap()),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot divide an integer scalar of data type `i32` by zero",
        ));
        assert!(matches!(
            Div::div(&Array::vector(vec![i8::MIN]).unwrap(), &Array::vector(vec![-1i8]).unwrap()),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot divide the minimum integer scalar of data type `i8` by -1",
        ));
    }

    #[test]
    fn test_div_primitives() {
        assert_eq!(Div::div(&7usize, &2), Ok(3));
        assert_eq!(
            Div::div(&7usize, &0),
            Err(ProgramError::InvalidArgument {
                message: "`div` divisor is zero or the output does not fit in `usize`".to_string(),
            }),
        );
        assert_eq!(
            Div::div(&i8::MIN, &-1),
            Err(ProgramError::InvalidArgument {
                message: "`div` divisor is zero or the output does not fit in `i8`".to_string(),
            }),
        );
        assert_eq!(Div::div(&1.0f64, &0.0), Ok(f64::INFINITY));
    }

    #[test]
    fn test_rem_type_inference() {
        check_operation_type_inference!(
            @elementwise @binary,
            operation = RemOperation,
            cases = [
                {
                    input_data_types = [DataType::I32, DataType::I32],
                    output_data_types = [DataType::I32],
                },
                {
                    input_data_types = [DataType::C64, DataType::C64],
                    error = "`rem` does not support input data type `c64`",
                },
                {
                    input_data_types = [DataType::Boolean, DataType::Boolean],
                    error = "`rem` does not support input data type `bool`",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = RemOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_rem_interpretation() {
        assert_eq!(
            RemOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(7.5f64).unwrap(), Array::scalar(2.0f64).unwrap()],
            ),
            Ok(vec![Array::scalar(1.5f64).unwrap()]),
        );
    }

    #[test]
    fn test_rem_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = RemOperation::new(),
            inputs = [Array::scalar(7.5).unwrap(), Array::scalar(2.0).unwrap()],
            expected = Array::scalar(1.5).unwrap(),
        );
    }

    #[test]
    fn test_rem_batching() {
        check_operation_batching!(
            @exact,
            operation = RemOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::vector(vec![7.5, -7.5]).unwrap()),
                    (@mapped(axis = 0), Array::vector(vec![2.0, 2.0]).unwrap()),
                ],
                outputs = [(@mapped(axis = 0), Array::vector(vec![1.5, -1.5]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_rem_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = RemOperation::new(),
            cases = [{
                primals = [Array::scalar(7.5).unwrap(), Array::scalar(2.0).unwrap()],
                tangents = [Array::scalar(3.0).unwrap(), Array::scalar(5.0).unwrap()],
                primal_outputs = [Array::scalar(1.5).unwrap()],
                // 3.0 - trunc(7.5 / 2.0) · 5.0 = 3.0 - 3.0 · 5.0 = -12.0.
                tangent_outputs = [Array::scalar(-12.0).unwrap()],
            }],
        );
    }

    #[test]
    fn test_rem_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = RemOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_array_rem() {
        assert_eq!(
            Array::scalar(7i32).unwrap().rem(&Array::scalar(3i32).unwrap()).unwrap(),
            Array::scalar(1i32).unwrap()
        );
        // The result takes the sign of the dividend.
        assert_eq!(
            Array::scalar(-7i32).unwrap().rem(&Array::scalar(3i32).unwrap()).unwrap(),
            Array::scalar(-1i32).unwrap()
        );
        assert_eq!(
            Array::scalar(7i64).unwrap().rem(&Array::scalar(-3i64).unwrap()).unwrap(),
            Array::scalar(1i64).unwrap()
        );
        assert_eq!(
            Array::scalar(7u32).unwrap().rem(&Array::scalar(3u32).unwrap()).unwrap(),
            Array::scalar(1u32).unwrap()
        );
        assert_eq!(
            Array::scalar(7.5f64).unwrap().rem(&Array::scalar(2.0f64).unwrap()).unwrap(),
            Array::scalar(1.5f64).unwrap()
        );
        assert_eq!(
            Array::scalar(-7.5f32).unwrap().rem(&Array::scalar(2.0f32).unwrap()).unwrap(),
            Array::scalar(-1.5f32).unwrap()
        );
        assert_eq!(
            Array::scalar(bf16::from_f32(7.5))
                .unwrap()
                .rem(&Array::scalar(bf16::from_f32(2.0)).unwrap())
                .unwrap(),
            Array::scalar(bf16::from_f32(7.5f32 % 2.0f32)).unwrap(),
        );
        assert_eq!(
            Array::scalar(f16::from_f32(7.5)).unwrap().rem(&Array::scalar(f16::from_f32(2.0)).unwrap()).unwrap(),
            Array::scalar(f16::from_f32(7.5f32 % 2.0f32)).unwrap(),
        );
        // Division by an integer zero reports an error instead of panicking.
        assert_eq!(
            Array::scalar(7i32).unwrap().rem(&Array::scalar(0i32).unwrap()),
            Err(TypeError::invalid(
                "cannot compute the remainder of an integer scalar of data type `i32` with a zero divisor",
            )
            .into()),
        );

        assert_eq!(
            Array::vector(vec![7.5, -7.5]).unwrap().rem(&Array::vector(vec![2.0, 2.0]).unwrap()).unwrap(),
            Array::vector(vec![1.5, -1.5]).unwrap(),
        );
    }

    #[test]
    fn test_array_rem_low_precision() {
        // Low-precision arithmetic computes through decoded values and re-encodes the nearest representable result.
        let left = Array::from_elements::<f8e4m3fn>(
            ArrayType::new_static(DataType::F8E4M3FN, [2]),
            &[1.0, 2.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        let right = Array::from_elements::<f8e4m3fn>(
            ArrayType::new_static(DataType::F8E4M3FN, [2]),
            &[0.5, 0.25].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        assert_eq!(left.rem(&right).unwrap().to_f64s(), vec![0.0, 0.0]);
    }

    #[test]
    fn test_array_rem_zero_divisor() {
        // Remainder by zero returns a structured error.
        assert!(matches!(
            Array::vector(vec![1u8]).unwrap().rem(&Array::vector(vec![0u8]).unwrap()),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message
                    == "cannot compute the remainder of an integer scalar of data type `u8` with a zero divisor",
        ));
    }

    #[test]
    fn test_rem_primitives() {
        assert_eq!(Rem::rem(&7usize, &4), Ok(3));
        assert_eq!(
            Rem::rem(&7usize, &0),
            Err(ProgramError::InvalidArgument {
                message: "`rem` divisor is zero or the output does not fit in `usize`".to_string(),
            }),
        );
        assert!(Rem::rem(&1.0f64, &0.0).unwrap().is_nan());
    }

    #[test]
    fn test_neg_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = NegOperation,
            cases = [{
                input_data_types = [DataType::F64],
                output_data_types = [DataType::F64],
            }],
        );
        for input_type in [DataType::Token, DataType::Zero, DataType::Boolean, DataType::F8E8M0FNU] {
            let message = format!("`{NEG_OPERATION_NAME}` does not support input data type `{input_type}`");
            check_operation_type_inference!(
                @elementwise @unary,
                operation = NegOperation,
                cases = [{
                    input_data_types = [input_type],
                    error = message,
                }],
            );
        }

        // Negation is linear, so partial-sum and reduced markers pass through unchanged.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let unreduced = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_unreduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        check_operation_type_inference!(
            operation = NegOperation::<ArrayType>::new(),
            cases = [{
                input_types = [unreduced.clone()],
                output_types = [unreduced],
            }],
        );
        let reduced = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_reduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        check_operation_type_inference!(
            operation = NegOperation::<ArrayType>::new(),
            cases = [{
                input_types = [reduced.clone()],
                output_types = [reduced],
            }],
        );
    }

    #[test]
    fn test_neg_interpretation() {
        let operation = NegOperation::<ArrayType>::new();

        assert_eq!(
            operation.interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, &[Array::scalar(2.0).unwrap()],),
            Ok(vec![Array::scalar(-2.0).unwrap()]),
        );
        assert_eq!(
            operation.interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, &[Array::scalar(1u8).unwrap()],),
            Ok(vec![Array::scalar(u8::MAX).unwrap()]),
        );
        assert_eq!(
            operation.interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(ComplexNumber::new(1.0f64, -2.0)).unwrap()],
            ),
            Ok(vec![Array::scalar(ComplexNumber::new(-1.0f64, 2.0)).unwrap()]),
        );
    }

    #[test]
    fn test_neg_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = NegOperation::new(),
            inputs = [Array::scalar(2.0).unwrap()],
            expected = Array::scalar(-2.0).unwrap(),
        );
    }

    #[test]
    fn test_neg_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = NegOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![1.0, -2.0]).unwrap())],
                outputs = [(@mapped(axis = 0), Array::vector(vec![-1.0, 2.0]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_neg_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = NegOperation::new(),
            cases = [{
                primals = [Array::scalar(2.0).unwrap()],
                tangents = [Array::scalar(3.0).unwrap()],
                primal_outputs = [Array::scalar(-2.0).unwrap()],
                tangent_outputs = [Array::scalar(-3.0).unwrap()],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = neg %0
                        %3:f64[] = neg %1
                    in (%2, %3)
                "},
            }],
        );
    }

    #[test]
    fn test_neg_transposition() {
        check_operation_transposition!(
            @exact,
            operation = NegOperation::new(),
            cases = [{
                inputs = [(@linear(type = ArrayType::scalar(DataType::F64)))],
                output_cotangents = [Array::scalar(3.0).unwrap()],
                input_cotangents = [Array::scalar(-3.0).unwrap()],
                pullback = indoc! {"
                    lambda %0:f64[] .
                    let %1:f64[] = neg %0
                    in (%1)
                "},
            }],
        );
    }

    #[test]
    fn test_array_neg() {
        let vector = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(vector.neg().unwrap(), Array::vector(vec![-1.0, -2.0, -3.0]).unwrap());
        // The `std::ops` sugar delegates to the fallible capability.
        assert_eq!(-vector.clone(), Array::vector(vec![-1.0, -2.0, -3.0]).unwrap());
    }

    #[test]
    fn test_array_neg_low_precision() {
        // Low-precision arithmetic computes through decoded values and re-encodes the nearest representable result.
        let left = Array::from_elements::<f8e4m3fn>(
            ArrayType::new_static(DataType::F8E4M3FN, [2]),
            &[1.0, 2.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        assert_eq!(left.neg().unwrap().to_f64s(), vec![-1.0, -2.0]);
    }

    #[test]
    fn test_array_neg_complex() {
        // Elementwise complex math decodes and encodes the complex element types directly.
        let left = Array::vector(vec![ComplexNumber::new(1.0f64, 2.0), ComplexNumber::new(0.5f64, -1.0)]).unwrap();
        let left_values = [ComplexNumber::new(1.0f64, 2.0), ComplexNumber::new(0.5f64, -1.0)];
        assert_eq!(left.neg().unwrap(), Array::vector(vec![-left_values[0], -left_values[1]]).unwrap());
    }

    #[test]
    fn test_array_neg_integers() {
        // Negation wraps deterministically for unsigned and two's-complement signed elements, matching the scalar
        // reference backend (and StableHLO's integer semantics), rather than panicking or saturating.
        let unsigned = Array::vector(vec![0u8, 1, 255]).unwrap();
        assert_eq!(unsigned.neg().unwrap().elements::<u8>(), Ok(vec![0, 255, 1]));
        let minimum = Array::vector(vec![i8::MIN, -5]).unwrap();
        assert_eq!(minimum.neg().unwrap().elements::<i8>(), Ok(vec![i8::MIN, 5]));
        let narrow = Array::vector(vec![i4::new(7).unwrap(), i4::new(-8).unwrap()]).unwrap();
        assert_eq!(narrow.neg().unwrap().elements::<i4>(), Ok(vec![i4::new(-7).unwrap(), i4::MIN]));
    }

    #[test]
    fn test_neg_primitives() {
        assert_eq!(Neg::neg(&5i32), Ok(-5));
        assert_eq!(
            Neg::neg(&i8::MIN),
            Err(ProgramError::InvalidArgument { message: "`neg` output does not fit in `i8`".to_string() }),
        );
        assert_eq!(Neg::neg(&2.5f64), Ok(-2.5));
    }

    #[test]
    fn test_abs_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = AbsOperation,
            cases = [
                {
                    input_data_types = [DataType::I32],
                    output_data_types = [DataType::I32],
                },
                {
                    input_data_types = [DataType::I2],
                    output_data_types = [DataType::I2],
                },
                {
                    input_data_types = [DataType::I4],
                    output_data_types = [DataType::I4],
                },
                {
                    input_data_types = [DataType::C64],
                    output_data_types = [DataType::F32],
                },
                {
                    input_data_types = [DataType::C128],
                    output_data_types = [DataType::F64],
                },
            ],
        );

        for input_type in [DataType::Token, DataType::Zero, DataType::Boolean, DataType::I1, DataType::U32] {
            let message = format!("cannot compute the absolute value of a value of data type `{input_type}`");
            check_operation_type_inference!(
                @elementwise @unary,
                operation = AbsOperation,
                cases = [{
                    input_data_types = [input_type],
                    error = message,
                }],
            );
        }

        check_operation_type_inference!(
            @reject @unreduced,
            operation = AbsOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F32)],
        );
    }

    #[test]
    fn test_abs_interpretation() {
        let operation = AbsOperation::new();

        assert_eq!(
            operation.interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, &[Array::scalar(-2.0).unwrap()],),
            Ok(vec![Array::scalar(2.0).unwrap()]),
        );
        assert_eq!(
            operation.interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(ComplexNumber::new(3.0f64, -4.0f64)).unwrap()],
            ),
            Ok(vec![Array::scalar(5.0).unwrap()]),
        );
    }

    #[test]
    fn test_abs_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = AbsOperation::new(),
            inputs = [Array::scalar(-2.0).unwrap()],
            expected = Array::scalar(2.0).unwrap(),
        );
    }

    #[test]
    fn test_abs_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = AbsOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, -2.0]).unwrap())],
                outputs = [(@mapped(axis = 0), Array::vector(vec![0.5, 2.0]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_abs_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = AbsOperation::new(),
            cases = [
                {
                    primals = [Array::scalar(0.7).unwrap()],
                    tangents = [Array::scalar(3.0).unwrap()],
                    primal_outputs = [Array::scalar(0.7).unwrap()],
                    tangent_outputs = [Array::scalar(3.0).unwrap()],
                    jvp = indoc! {"
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = abs %0
                            %3:f64[] = zero_like %0
                            %4:bool[] = compare [direction=GreaterThanOrEqual] %0 %3
                            %5:f64[] = neg %1
                            %6:f64[] = select %4 %1 %5
                        in (%2, %6)
                    "},
                },
                {
                    primals = [Array::scalar(-2.5).unwrap()],
                    tangents = [Array::scalar(2.0).unwrap()],
                    primal_outputs = [Array::scalar(2.5).unwrap()],
                    tangent_outputs = [Array::scalar(-2.0).unwrap()],
                },
            ],
        );
    }

    #[test]
    fn test_abs_differentiation_at_zero() {
        // The real rule chooses the right derivative at zero and remains constant under another derivative.
        assert_abs_diff_eq!(
            differentiate_at(Array::scalar(0.0f64).unwrap()).gradient(|x| x.abs().unwrap()).unwrap(),
            Array::scalar(1.0).unwrap(),
            epsilon = 1e-9,
        );
        assert_abs_diff_eq!(
            differentiate_at(Array::scalar(0.0f64).unwrap())
                .gradient(|x| { differentiate_at(x).gradient(|x| x.abs().unwrap()).unwrap() })
                .unwrap(),
            Array::scalar(0.0).unwrap(),
            epsilon = 1e-9,
        );
    }

    #[test]
    fn test_abs_differentiation_complex() {
        // |z| is a ℂ → ℝ function and so it flows through the plain gradient entry point. With
        // d|z| = Re(z̄ · dz) / |z|, the bilinear-pairing gradient is z̄ / |z| (the unit-magnitude conjugate direction):
        // the reverse-mode counterpart of ∇|z|² = 2z̄ after the chain rule through the square root.
        let z = ComplexNumber::new(0.7f64, -0.3f64);
        let (value, gradient_value) =
            differentiate_at(Array::scalar(z).unwrap()).value_and_gradient(|z| z.abs().unwrap()).unwrap();
        assert_eq!(value, Array::scalar(z.norm()).unwrap());
        let expected = z.conj() / z.norm();
        assert_abs_diff_eq!(gradient_value, Array::scalar(expected).unwrap(), epsilon = 1e-12);

        // The array universe agrees: summing the elementwise magnitudes of a complex vector is again ℂⁿ → ℝ, and the
        // finite-difference oracle perturbs each element's real and imaginary parts independently.
        check_gradient!(
            |z| z.abs().map(|magnitudes| magnitudes.reduce(&[0], ReductionKind::Sum).unwrap()),
            at = Array::vector(vec![ComplexNumber::new(0.7f64, -0.3), ComplexNumber::new(-1.2f64, 0.8)]).unwrap(),
            step = 1e-6,
            tolerance = 1e-6,
        );

        // The complex rule replaces a zero magnitude denominator with one, so the zero numerator produces a finite
        // zero tangent and gradient at the origin.
        assert_eq!(
            differentiate_at(Array::scalar(ComplexNumber::new(0.0f64, 0.0f64)).unwrap())
                .jvp(Array::scalar(ComplexNumber::new(1.0f64, 2.0f64)).unwrap(), |z| z.abs()),
            Ok((Array::scalar(0.0f64).unwrap(), Array::scalar(0.0f64).unwrap())),
        );
        assert_eq!(
            differentiate_at(Array::scalar(ComplexNumber::new(0.0f64, 0.0f64)).unwrap()).gradient(|z| z.abs().unwrap()),
            Ok(Array::scalar(ComplexNumber::new(0.0f64, 0.0f64)).unwrap()),
        );
    }

    #[test]
    fn test_abs_differentiation_complex_avoids_overflow() {
        // Normalizing the complex coefficient before applying the tangent avoids overflowing the otherwise finite
        // directional derivative `Re((conj(z) / |z|) * dz)`.
        assert_eq!(
            differentiate_at(Array::scalar(ComplexNumber::new(1e308f64, 0.0)).unwrap())
                .jvp(Array::scalar(ComplexNumber::new(2.0f64, 0.0)).unwrap(), |z| z.abs()),
            Ok((Array::scalar(1e308f64).unwrap(), Array::scalar(2.0f64).unwrap())),
        );
    }

    #[test]
    fn test_abs_differentiation_low_precision_uses_widened_tangents() {
        // The coefficient and tangent are computed in the widened differential representation.
        let primal = Array::from_elements::<f8e8m0fnu>(
            ArrayType::scalar(DataType::F8E8M0FNU),
            &[2.0].map(|value| f8e8m0fnu::from_f64(value).unwrap()),
        )
        .unwrap();
        let input_tangent = Array::from_elements::<f32>(ArrayType::scalar(DataType::F32), &[3.0]).unwrap();
        let (_, tangent) = differentiate_at(primal).jvp(input_tangent, |input| input.abs()).unwrap();
        assert_eq!(tangent.r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        assert_eq!(tangent.to_f64s(), vec![3.0]);
    }

    #[test]
    fn test_abs_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = AbsOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_array_abs() {
        assert_eq!(Array::vector(vec![-1.5, 2.5]).unwrap().abs().unwrap(), Array::vector(vec![1.5, 2.5]).unwrap());
        // The absolute value of a complex array is its elementwise magnitude with a real element data type.
        let complex = Array::vector(vec![3.0]).unwrap().complex(&Array::vector(vec![4.0]).unwrap()).unwrap();
        let magnitude = complex.abs().unwrap();
        assert_eq!(magnitude.r#type().into_owned(), ArrayType::new_static(DataType::F64, [1]));
        assert_abs_diff_eq!(magnitude, Array::vector(vec![5.0]).unwrap(), epsilon = 1e-12);
    }

    #[test]
    fn test_array_abs_low_precision() {
        let left = Array::from_elements::<f8e4m3fn>(
            ArrayType::new_static(DataType::F8E4M3FN, [2]),
            &[1.0, 2.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        let negative = Array::from_elements::<f8e4m3fn>(
            ArrayType::new_static(DataType::F8E4M3FN, [2]),
            &[-1.0, -2.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        assert_eq!(negative.abs().unwrap(), left);
    }

    #[test]
    fn test_array_abs_complex() {
        let left = Array::vector(vec![ComplexNumber::new(1.0f64, 2.0), ComplexNumber::new(0.5f64, -1.0)]).unwrap();
        let left_values = [ComplexNumber::new(1.0f64, 2.0), ComplexNumber::new(0.5f64, -1.0)];
        // The absolute value is the elementwise magnitude with a real element data type.
        let magnitude = left.abs().unwrap();
        assert_eq!(magnitude.r#type().into_owned(), ArrayType::new_static(DataType::F64, [2]));
        assert_abs_diff_eq!(
            magnitude,
            Array::vector(vec![left_values[0].norm(), left_values[1].norm()]).unwrap(),
            epsilon = 1e-12,
        );
    }

    #[test]
    fn test_array_abs_integers() {
        // Sub-byte arithmetic uses the declared bit width for every wrapping operation.
        let narrow = Array::vector(vec![i4::new(7).unwrap(), i4::new(-8).unwrap()]).unwrap();
        assert_eq!(narrow.abs().unwrap().elements::<i4>(), Ok(vec![i4::new(7).unwrap(), i4::MIN]));
    }

    #[test]
    fn test_abs_primitives() {
        assert_eq!(Abs::abs(&-5i32), Ok(5));
        assert_eq!(
            Abs::abs(&i8::MIN),
            Err(ProgramError::InvalidArgument { message: "`abs` output does not fit in `i8`".to_string() }),
        );
        assert_eq!(Abs::abs(&5usize), Ok(5));
        assert_eq!(Abs::abs(&-2.5f64), Ok(2.5));
    }

    #[test]
    fn test_sign_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = SignOperation,
            cases = [
                {
                    input_data_types = [DataType::I32],
                    output_data_types = [DataType::I32],
                },
                {
                    input_data_types = [DataType::C64],
                    output_data_types = [DataType::C64],
                },
                {
                    input_data_types = [DataType::U8],
                    error = "cannot compute the sign of a value of data type `u8`",
                },
                {
                    input_data_types = [DataType::Boolean],
                    error = "cannot compute the sign of a value of data type `bool`",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = SignOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_sign_interpretation() {
        assert_eq!(
            SignOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(-3.0f64).unwrap()],
            ),
            Ok(vec![Array::scalar(-1.0f64).unwrap()]),
        );
    }

    #[test]
    fn test_sign_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = SignOperation::new(),
            inputs = [Array::scalar(-0.7).unwrap()],
            expected = Array::scalar(-1.0).unwrap(),
        );
    }

    #[test]
    fn test_sign_batching() {
        check_operation_batching!(
            @exact,
            operation = SignOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, -1.0]).unwrap())],
                outputs = [(@mapped(axis = 0), Array::vector(vec![1.0, -1.0]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_sign_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = SignOperation::new(),
            cases = [{
                primals = [Array::scalar(-2.0).unwrap()],
                tangents = [Array::scalar(1.0).unwrap()],
                primal_outputs = [Array::scalar(-1.0).unwrap()],
                tangent_outputs = [Array::scalar(0.0).unwrap()],
            }],
        );
    }

    #[test]
    fn test_sign_transposition() {
        check_operation_transposition!(
            @exact,
            operation = SignOperation::<ArrayType>::new(),
            cases = [{
                inputs = [(@linear(type = ArrayType::scalar(DataType::F64)))],
                output_cotangents = [Array::scalar(3.0).unwrap()],
                input_cotangents = [Array::scalar(0.0).unwrap()],
            }],
        );
    }

    #[test]
    fn test_array_sign() {
        assert_eq!(Array::scalar(-3i32).unwrap().sign().unwrap(), Array::scalar(-1i32).unwrap());
        assert_eq!(Array::scalar(0i32).unwrap().sign().unwrap(), Array::scalar(0i32).unwrap());
        assert_eq!(Array::scalar(5i64).unwrap().sign().unwrap(), Array::scalar(1i64).unwrap());
        assert_eq!(Array::scalar(-2.5f32).unwrap().sign().unwrap(), Array::scalar(-1.0f32).unwrap());
        assert_eq!(Array::scalar(2.5f64).unwrap().sign().unwrap(), Array::scalar(1.0f64).unwrap());
        assert_eq!(
            Array::scalar(bf16::from_f32(-4.0)).unwrap().sign().unwrap(),
            Array::scalar(bf16::from_f32(-1.0)).unwrap(),
        );
        assert_eq!(
            Array::scalar(f16::from_f32(4.0)).unwrap().sign().unwrap(),
            Array::scalar(f16::from_f32(1.0)).unwrap(),
        );
        // Signed zeros and NaNs pass through unchanged.
        assert_eq!(Array::scalar(0.0f64).unwrap().sign().unwrap(), Array::scalar(0.0f64).unwrap());
        assert!(Array::scalar(-0.0f64).unwrap().sign().unwrap().to_f64s()[0].is_sign_negative());
        assert!(Array::scalar(f64::NAN).unwrap().sign().unwrap().to_f64s()[0].is_nan());
        // Complex signs normalize to `z / |z|` and map the origin to itself.
        let input = ComplexNumber::new(3.0f64, -4.0f64);
        assert_abs_diff_eq!(
            Array::scalar(input).unwrap().sign().unwrap(),
            Array::scalar(ComplexNumber::new(0.6, -0.8)).unwrap(),
            epsilon = 1e-12,
        );
        assert_eq!(
            Array::scalar(ComplexNumber::new(0.0f64, 0.0f64)).unwrap().sign().unwrap(),
            Array::scalar(ComplexNumber::new(0.0f64, 0.0f64)).unwrap(),
        );

        assert_eq!(
            Array::vector(vec![-0.7, 0.0, 2.0]).unwrap().sign().unwrap(),
            Array::vector(vec![-1.0, 0.0, 1.0]).unwrap(),
        );
    }

    #[test]
    fn test_array_sign_typed_storage() {
        // Sign preserves IEEE signed zero and NaN behavior and also covers signed sub-byte integers.
        let signs = Array::from_elements(
            ArrayType::new_static(DataType::F64, [4]),
            &[-2.0f64, -0.0, 0.0, f64::from_bits(0x7ff8_0000_0000_1234)],
        )
        .unwrap()
        .sign()
        .unwrap()
        .elements::<f64>()
        .unwrap();
        assert_eq!(signs[0], -1.0);
        assert_eq!(signs[1].to_bits(), (-0.0f64).to_bits());
        assert_eq!(signs[2].to_bits(), 0.0f64.to_bits());
        assert_eq!(signs[3].to_bits(), 0x7ff8_0000_0000_1234);
        let narrow =
            Array::from_elements(ArrayType::new_static(DataType::I2, [3]), &[i2::MIN, i2::new(0).unwrap(), i2::MAX])
                .unwrap();
        assert_eq!(
            narrow.sign().unwrap().elements::<i2>(),
            Ok(vec![i2::new(-1).unwrap(), i2::new(0).unwrap(), i2::new(1).unwrap()]),
        );
        assert!(matches!(
            Array::scalar(1u8).unwrap().sign(),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot compute the sign of a value of data type `u8`",
        ));
    }

    #[test]
    fn test_sign_primitives() {
        assert_eq!(Sign::sign(&-5i32), Ok(-1));
        assert_eq!(Sign::sign(&0i32), Ok(0));
        assert_eq!(Sign::sign(&-2.5f64), Ok(-1.0));
        assert_eq!(Sign::sign(&-0.0f64).unwrap().to_bits(), (-0.0f64).to_bits());
        assert!(Sign::sign(&f64::NAN).unwrap().is_nan());
    }

    #[test]
    fn test_pow_type_inference() {
        check_operation_type_inference!(
            @elementwise @binary,
            operation = PowOperation,
            cases = [
                {
                    input_data_types = [DataType::F32, DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::I32, DataType::I32],
                    error = "`pow` does not support input data type `i32`",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = PowOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_pow_interpretation() {
        assert_eq!(
            PowOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.0f64).unwrap(), Array::scalar(3.0f64).unwrap()],
            ),
            Ok(vec![Array::scalar(8.0f64).unwrap()]),
        );
    }

    #[test]
    fn test_pow_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = PowOperation::new(),
            inputs = [Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap()],
            expected = Array::scalar(8.0).unwrap(),
        );
    }

    #[test]
    fn test_pow_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = PowOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::vector(vec![2.0, 3.0]).unwrap()),
                    (@mapped(axis = 0), Array::vector(vec![3.0, 2.0]).unwrap()),
                ],
                outputs = [(@mapped(axis = 0), Array::vector(vec![8.0, 9.0]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_pow_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = PowOperation::new(),
            cases = [
                {
                    primals = [Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap()],
                    tangents = [Array::scalar(3.0).unwrap(), Array::scalar(5.0).unwrap()],
                    primal_outputs = [Array::scalar(8.0).unwrap()],
                    // d(x^y) = y · x^{y-1} · dx + x^y · ln(x) · dy = 3.0 · (3.0 · 4.0) + 5.0 · (8.0 · ln(2)).
                    tangent_outputs = [Array::scalar(3.0 * (3.0 * 4.0) + 5.0 * (8.0 * 2.0f64.ln())).unwrap()],
                },
            ],
        );
    }

    #[test]
    fn test_pow_differentiation_at_zero_base() {
        // A zero base exercises the guarded log factor: the exponent contribution is exactly zero instead of
        // `log(0) = -∞` turning the tangent into a NaN. The finite-difference oracle cannot check this boundary
        // point (perturbing the base below zero is undefined), so the guard is asserted on the staged jvp program
        // directly.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let base = builder.add_input(ArrayType::scalar(DataType::F64));
        let exponent = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(PowOperation::new(), Vec::new(), vec![base, exponent], None).unwrap()[0];
        let jvp_program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap()
            .jvp()
            .unwrap();
        let outputs = jvp_program
            .interpret(vec![
                Array::scalar(0.0).unwrap(),
                Array::scalar(2.0).unwrap(),
                Array::scalar(1.0).unwrap(),
                Array::scalar(1.0).unwrap(),
            ])
            .unwrap();
        assert_eq!(outputs, vec![Array::scalar(0.0).unwrap(), Array::scalar(0.0).unwrap()]);
    }

    #[test]
    fn test_pow_differentiation_complex() {
        // The holomorphic gradient of z² is 2z.
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        assert_abs_diff_eq!(
            differentiate_at(Array::scalar(input).unwrap())
                .holomorphic()
                .gradient(|input| {
                    let exponent = input.one_like().unwrap() + input.one_like().unwrap();
                    input.pow(&exponent).unwrap()
                })
                .unwrap(),
            Array::scalar(input * 2.0).unwrap(),
            epsilon = 1e-12,
        );
    }

    #[test]
    fn test_pow_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = PowOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_array_pow() {
        assert_eq!(
            Array::scalar(2.0f32).unwrap().pow(&Array::scalar(3.0f32).unwrap()).unwrap(),
            Array::scalar(2.0f32.powf(3.0)).unwrap(),
        );
        assert_eq!(
            Array::scalar(2.0f64).unwrap().pow(&Array::scalar(3.0f64).unwrap()).unwrap(),
            Array::scalar(2.0f64.powf(3.0)).unwrap(),
        );
        assert_eq!(
            Array::scalar(bf16::from_f32(2.0))
                .unwrap()
                .pow(&Array::scalar(bf16::from_f32(3.0)).unwrap())
                .unwrap(),
            Array::scalar(bf16::from_f32(2.0f32.powf(3.0))).unwrap(),
        );
        assert_eq!(
            Array::scalar(f16::from_f32(2.0)).unwrap().pow(&Array::scalar(f16::from_f32(3.0)).unwrap()).unwrap(),
            Array::scalar(f16::from_f32(2.0f32.powf(3.0))).unwrap(),
        );
        // The complex power is the principal value `exp(y · log(x))`.
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        let exponent = ComplexNumber::new(2.0f64, 0.0f64);
        assert_abs_diff_eq!(
            Array::scalar(input).unwrap().pow(&Array::scalar(exponent).unwrap()).unwrap(),
            Array::scalar(input.powc(exponent)).unwrap(),
            epsilon = 1e-12,
        );
        assert_eq!(Array::scalar(2.0).unwrap().pow(&Array::scalar(3.0).unwrap()), Array::scalar(8.0));
    }

    #[test]
    fn test_array_pow_typed_storage() {
        // Both inputs promote before broadcasting their independent axes.
        let bases = Array::matrix(2, 1, vec![2.0f32, 3.0]).unwrap();
        let exponents = Array::matrix(1, 3, vec![1.0f64, 2.0, 3.0]).unwrap();
        assert_eq!(bases.pow(&exponents), Array::matrix(2, 3, vec![2.0f64, 4.0, 8.0, 3.0, 9.0, 27.0]),);
        assert_eq!(
            Array::scalar(2.0f64).unwrap().pow(&Array::scalar(3i32).unwrap()),
            Err(TypeError::invalid("`pow` does not support input data type `i32`").into()),
        );
    }

    #[test]
    fn test_pow_primitives() {
        assert_eq!(Pow::pow(&2.0f64, &3.0), Ok(8.0));
    }

    #[test]
    fn test_sqrt_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = SqrtOperation,
            cases = [
                {
                    input_data_types = [DataType::C64],
                    output_data_types = [DataType::C64],
                },
                {
                    input_data_types = [DataType::I32],
                    error = "`sqrt` does not support input data type `i32`",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = SqrtOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_sqrt_interpretation() {
        assert_eq!(
            SqrtOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(4.0f64).unwrap()],
            ),
            Ok(vec![Array::scalar(2.0f64).unwrap()]),
        );
    }

    #[test]
    fn test_sqrt_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = SqrtOperation::new(),
            inputs = [Array::scalar(4.0).unwrap()],
            expected = Array::scalar(2.0).unwrap(),
        );
    }

    #[test]
    fn test_sqrt_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = SqrtOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, 2.0]).unwrap())],
                outputs = [(@mapped(axis = 0), Array::vector(vec![0.5f64.sqrt(), 2.0f64.sqrt()]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_sqrt_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = SqrtOperation::new(),
            cases = [{
                primals = [Array::scalar(2.0).unwrap()],
                tangents = [Array::scalar(3.0).unwrap()],
                primal_outputs = [Array::scalar(2.0f64.sqrt()).unwrap()],
                tangent_outputs = [Array::scalar(3.0 / (2.0 * 2.0f64.sqrt())).unwrap()],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = sqrt %0
                        %3:f64[] = add %2 %2
                        %4:f64[] = div %1 %3
                    in (%2, %4)
                "},
            }],
        );
    }

    #[test]
    fn test_sqrt_differentiation_complex() {
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        assert_eq!(
            differentiate_at(Array::scalar(input).unwrap())
                .holomorphic()
                .gradient(|input| input.sqrt().unwrap()),
            Ok(Array::scalar(ComplexNumber::new(1.0, 0.0) / (input.sqrt() + input.sqrt())).unwrap()),
        );
    }

    #[test]
    fn test_sqrt_differentiation_low_precision_uses_widened_tangents() {
        let primal = Array::from_elements::<f8e8m0fnu>(
            ArrayType::scalar(DataType::F8E8M0FNU),
            &[2.0].map(|value| f8e8m0fnu::from_f64(value).unwrap()),
        )
        .unwrap();
        let input_tangent = Array::from_elements::<f32>(ArrayType::scalar(DataType::F32), &[3.0]).unwrap();
        let (_, tangent) = differentiate_at(primal).jvp(input_tangent, |input| input.sqrt()).unwrap();
        assert_eq!(tangent.r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        // The tangent payload is honestly `f32`-encoded, so the comparison happens at `f32` precision.
        assert_abs_diff_eq!(tangent.to_f64s()[0], 3.0 / (2.0 * 2.0f64.sqrt()), epsilon = 1e-6);

        // The widened staged tangent program recomputes the denominator in the widened differential representation
        // instead of converting the narrower primal output.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F8E8M0FNU));
        let output = builder.add_instruction(SqrtOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
            .jvp()
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f8e8m0fnu[], %1:f32[] .
                let %2:f8e8m0fnu[] = sqrt %0
                    %3:f32[] = convert_element_type [data_type=f32] %0
                    %4:f32[] = sqrt %3
                    %5:f32[] = add %4 %4
                    %6:f32[] = div %1 %5
                in (%2, %6)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_sqrt_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = SqrtOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_array_sqrt() {
        assert_eq!(Array::scalar(0.25f32).unwrap().sqrt().unwrap(), Array::scalar(0.5f32).unwrap());
        assert_eq!(Array::scalar(0.25f64).unwrap().sqrt().unwrap(), Array::scalar(0.5f64).unwrap());
        assert_eq!(
            Array::scalar(bf16::from_f32(0.25)).unwrap().sqrt().unwrap(),
            Array::scalar(bf16::from_f32(0.5)).unwrap()
        );
        assert_eq!(
            Array::scalar(f16::from_f32(0.25)).unwrap().sqrt().unwrap(),
            Array::scalar(f16::from_f32(0.5)).unwrap()
        );
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        assert_abs_diff_eq!(
            Array::scalar(input).unwrap().sqrt().unwrap(),
            Array::scalar(input.sqrt()).unwrap(),
            epsilon = 1e-12
        );
        // The principal branch maps the negative real axis to the positive imaginary axis.
        assert_abs_diff_eq!(
            Array::scalar(ComplexNumber::new(-4.0f64, 0.0)).unwrap().sqrt().unwrap(),
            Array::scalar(ComplexNumber::new(0.0f64, 2.0)).unwrap(),
            epsilon = 1e-12,
        );

        assert_eq!(Array::scalar(4.0).unwrap().sqrt().unwrap(), Array::scalar(2.0).unwrap(),);
    }

    #[test]
    fn test_array_sqrt_typed_storage() {
        assert_eq!(Array::vector(vec![1.0f64, 4.0]).unwrap().sqrt(), Array::vector(vec![1.0f64, 2.0]),);
        // Complex square roots use the principal branch for each typed element.
        let values = [ComplexNumber::new(1.0f64, 2.0), ComplexNumber::new(0.5f64, -1.0)];
        let input = Array::vector(values.to_vec()).unwrap();
        assert_abs_diff_eq!(
            input.sqrt().unwrap(),
            Array::vector(vec![values[0].sqrt(), values[1].sqrt()]).unwrap(),
            epsilon = 1e-12,
        );
    }

    #[test]
    fn test_sqrt_primitives() {
        assert_eq!(Sqrt::sqrt(&4.0f64), Ok(2.0));
    }

    #[test]
    fn test_rsqrt_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = RsqrtOperation,
            cases = [
                {
                    input_data_types = [DataType::C64],
                    output_data_types = [DataType::C64],
                },
                {
                    input_data_types = [DataType::I32],
                    error = "`rsqrt` does not support input data type `i32`",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = RsqrtOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_rsqrt_interpretation() {
        assert_eq!(
            RsqrtOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(4.0f64).unwrap()],
            ),
            Ok(vec![Array::scalar(0.5f64).unwrap()]),
        );
    }

    #[test]
    fn test_rsqrt_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = RsqrtOperation::new(),
            inputs = [Array::scalar(4.0).unwrap()],
            expected = Array::scalar(0.5).unwrap(),
        );
    }

    #[test]
    fn test_rsqrt_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = RsqrtOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, 4.0]).unwrap())],
                outputs = [(@mapped(axis = 0), Array::vector(vec![1.0 / 0.5f64.sqrt(), 0.5]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_rsqrt_differentiation() {
        let expected_tangent = 3.0 * -(1.0 / (2.0 * 4.0f64.powf(1.5)));
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = RsqrtOperation::new(),
            cases = [{
                primals = [Array::scalar(4.0).unwrap()],
                tangents = [Array::scalar(3.0).unwrap()],
                primal_outputs = [Array::scalar(0.5).unwrap()],
                tangent_outputs = [Array::scalar(expected_tangent).unwrap()],
            }],
        );
    }

    #[test]
    fn test_rsqrt_differentiation_complex() {
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        // d(1/√z)/dz = -z^{-3/2} / 2 on the principal branch.
        let expected = input.powc(ComplexNumber::new(-1.5, 0.0)) * ComplexNumber::new(-0.5, 0.0);
        assert_abs_diff_eq!(
            Array::scalar(expected).unwrap(),
            differentiate_at(Array::scalar(input).unwrap())
                .holomorphic()
                .gradient(|input| { input.rsqrt().unwrap() })
                .unwrap(),
            epsilon = 1e-12,
        );
    }

    #[test]
    fn test_rsqrt_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = RsqrtOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_array_rsqrt() {
        assert_eq!(Array::scalar(0.5f32).unwrap().rsqrt().unwrap(), Array::scalar(1.0 / 0.5f32.sqrt()).unwrap());
        assert_eq!(Array::scalar(0.5f64).unwrap().rsqrt().unwrap(), Array::scalar(1.0 / 0.5f64.sqrt()).unwrap());
        assert_eq!(
            Array::scalar(bf16::from_f32(0.5)).unwrap().rsqrt().unwrap(),
            Array::scalar(bf16::from_f32(1.0 / 0.5f32.sqrt())).unwrap(),
        );
        assert_eq!(
            Array::scalar(f16::from_f32(0.5)).unwrap().rsqrt().unwrap(),
            Array::scalar(f16::from_f32(1.0 / 0.5f32.sqrt())).unwrap(),
        );
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        let expected = ComplexNumber::new(1.0, 0.0) / input.sqrt();
        assert_abs_diff_eq!(
            Array::scalar(input).unwrap().rsqrt().unwrap(),
            Array::scalar(expected).unwrap(),
            epsilon = 1e-12
        );

        assert_eq!(Array::scalar(4.0).unwrap().rsqrt().unwrap(), Array::scalar(0.5).unwrap(),);
    }

    #[test]
    fn test_array_rsqrt_typed_storage() {
        assert_eq!(Array::vector(vec![1.0f64, 4.0]).unwrap().rsqrt(), Array::vector(vec![1.0f64, 0.5]),);
    }

    #[test]
    fn test_rsqrt_primitives() {
        assert_eq!(Rsqrt::rsqrt(&4.0f64), Ok(0.5));
    }
}
