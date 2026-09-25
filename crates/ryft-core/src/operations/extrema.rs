//! Operations that select elementwise minima and maxima and clamp values into intervals. Each operation is defined by
//! an [`Operation`] type (e.g., [`MinOperation`]) together with a value capability trait (e.g., [`Min`]) whose
//! functions apply it to eager [`Array`]s and traced values alike, so the same code executes immediately or records
//! into a program depending on the value it runs on:
//!
//!   - [`Min`] and [`Max`] select the smaller and the larger of two values (i.e., `(a, b) ↦ min(a, b)` and
//!     `(a, b) ↦ max(a, b)`).
//!   - [`Clamp`] restricts a value to the interval delimited by a lower and an upper bound (i.e.,
//!     `(lower, x, upper) ↦ min(max(x, lower), upper)`), so crossed bounds (i.e., `lower > upper`) produce `upper`.
//!
//! Inputs may have any Boolean or numeric element type. Unlike StableHLO's
//! [`minimum`](https://openxla.org/stablehlo/spec#minimum), [`maximum`](https://openxla.org/stablehlo/spec#maximum),
//! and [`clamp`](https://openxla.org/stablehlo/spec#clamp), which require matching element types and shapes (up to
//! scalar bounds for `clamp`), these operations promote their inputs to a common element type and broadcast their
//! shapes. Booleans order `false` below `true`, so their minima and maxima are conjunctions and disjunctions. Real
//! floating-point values propagate NaNs and order negative zero below positive zero. Complex values compare their real
//! parts first and their imaginary parts second, and select the right input on exact ties and whenever a NaN makes the
//! deciding comparison unordered, so a NaN in the left input does not propagate. Array inputs that carry partial sums
//! over unreduced mesh axes are rejected, and the reduced-axis markers of the inputs must agree.
//!
//! Differentiation support follows JAX. The tangent of an extremum weighs each input's tangent by `1` where that input
//! is strictly selected, by `0.5` where the inputs tie, and by `0` where the other input is selected or either input is
//! NaN, and complex inputs apply the same weights under their lexicographic ordering. The tangent of a clamp is the
//! input tangent strictly inside the interval, the lower-bound tangent where the input lies below a lower bound that is
//! smaller than the upper bound, the upper-bound tangent where the input lies above the upper bound, and zero at the
//! bounds themselves. None of the operations is linear, so reverse-mode differentiation transposes their linearizations
//! instead.
//!
//! # Example
//!
//! ```rust
//! # use ryft_core::{Array, Clamp, ProgramError};
//! # fn main() -> Result<(), ProgramError> {
//! let input = Array::vector(vec![-2.0f64, 0.5, 3.0])?;
//! let output = input.clamp(&Array::scalar(-1.0)?, &Array::scalar(1.0)?)?;
//! assert_eq!(output, Array::vector(vec![-1.0, 0.5, 1.0])?);
//! # Ok(())
//! # }
//! ```

use std::marker::PhantomData;

use crate::arrays::{Array, ArrayElement, ArrayType, Broadcastable, DataType};
use crate::contexts::Context;
use crate::differentiation::{DifferentiableType, DifferentiationDual, ElementwiseDerivativeAlignment};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{
    check_count, check_types, define_elementwise_capability, define_elementwise_operation,
    impl_array_elementwise_operation, impl_differentiable_elementwise_operation, impl_differentiable_operation,
};
use crate::operations::ElementwiseOperation;
use crate::operations::arithmetic::{Add, Mul};
use crate::operations::collectives::parallel_vary::ManualVariationAlignment;
use crate::operations::comparisons::{Compare, ComparisonDirection};
use crate::operations::complex::{Imaginary, Real};
use crate::operations::constants::fill::Fill;
use crate::operations::constants::one_like::OneLike;
use crate::operations::constants::zero_like::ZeroLike;
use crate::operations::control_flow::select::Select;
use crate::operations::logical::And;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    MaybeZero, Operation, OperationProvider, ProgramError, RegionInterface, Type, TypeError, Typed, Value,
};

/// Canonical operation name for [`MinOperation`].
pub const MIN_OPERATION_NAME: &str = "min";

define_elementwise_operation!(
    @binary
    /// [`Operation`] that computes the elementwise minimum of two Boolean or numeric values, promoting their element
    /// types and broadcasting their shapes. Boolean minima are conjunctions. Real floating-point inputs propagate NaNs
    /// and order negative zero below positive zero. Complex inputs compare real components first, then imaginary
    /// components when the real components are equal, and select one whole input, namely the right input on exact
    /// ties and whenever a NaN makes the deciding comparison unordered. Array inputs that still carry partial sums
    /// are rejected, and their reduced-axis markers must agree.
    ///
    /// The tangent weighs each input tangent by `1` where that input is strictly smaller, by `0.5` where the inputs
    /// tie, and by `0` where the other input is smaller or either input is NaN, applying the lexicographic ordering to
    /// complex inputs.
    MinOperation,
    MIN_OPERATION_NAME,
    Min,
    min,
    check_data_types = [@boolean_or_numeric],
    check_array_types = [@no_unreduced, @same_reduced_axes],
);

impl_differentiable_elementwise_operation! {
    @binary
    MinOperation,
    jvp<C>
    where
        C::Value: ZeroLike + OneLike + Mul + Real + Imaginary + And + Compare<C::Value> + Select,
        <C::Value as Value>::DispatchDomain: Fill<f64, C::Value>,
    {
        |(left, left_tangent), (right, _)| {
            balanced_extremum_weight(&left, &right, ComparisonDirection::LessThan)?.mul(&left_tangent)?
        };
        |(left, _), (right, right_tangent)| {
            balanced_extremum_weight(&right, &left, ComparisonDirection::LessThan)?.mul(&right_tangent)?
        };
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @binary
    /// Represents the ability to select elementwise minima. Concrete arrays compute immediately while context-carrying
    /// values apply [`MinOperation`] through their context.
    Min,
    /// Returns the elementwise minimum of this value and `right`, promoting and broadcasting the inputs. Returns an
    /// error if the input types or metadata are unsupported.
    min(right),
    MinOperation,
);

impl_array_elementwise_operation!(
    @binary
    Min,
    min,
    operation = "min",
    inputs = @boolean_or_numeric,
    checks = [@no_unreduced, @same_reduced_axes],
    |left, right| Ok(ArrayElement::min(&left, &right)),
);

/// Implements [`Min`] for one host primitive type.
macro_rules! impl_min_for_primitive {
    // Boolean primitives compute their minimum as a conjunction.
    (@boolean $type:ty) => {
        impl Min for $type {
            #[inline]
            fn min(&self, right: &Self) -> Result<Self, ProgramError> {
                Ok(*self && *right)
            }
        }
    };

    // Integer primitives use ordinary total-order comparison, which cannot fail.
    (@integer $type:ty) => {
        impl Min for $type {
            #[inline]
            fn min(&self, right: &Self) -> Result<Self, ProgramError> {
                Ok(::std::cmp::Ord::min(*self, *right))
            }
        }
    };

    // Floating-point primitives mirror the reference backends: NaN inputs propagate, and signed zeros order through
    // the IEEE 754 total order (so that `-0.0` sorts below `+0.0`).
    (@float $type:ty) => {
        impl Min for $type {
            #[inline]
            fn min(&self, right: &Self) -> Result<Self, ProgramError> {
                Ok(if self.is_nan() {
                    *self
                } else if right.is_nan() {
                    *right
                } else if matches!(self.total_cmp(right), ::std::cmp::Ordering::Greater) {
                    *right
                } else {
                    *self
                })
            }
        }
    };
}

impl_min_for_primitive!(@boolean bool);
impl_min_for_primitive!(@integer i8);
impl_min_for_primitive!(@integer i16);
impl_min_for_primitive!(@integer i32);
impl_min_for_primitive!(@integer i64);
impl_min_for_primitive!(@integer i128);
impl_min_for_primitive!(@integer isize);
impl_min_for_primitive!(@integer u8);
impl_min_for_primitive!(@integer u16);
impl_min_for_primitive!(@integer u32);
impl_min_for_primitive!(@integer u64);
impl_min_for_primitive!(@integer u128);
impl_min_for_primitive!(@integer usize);
impl_min_for_primitive!(@float f32);
impl_min_for_primitive!(@float f64);

/// Canonical operation name for [`MaxOperation`].
pub const MAX_OPERATION_NAME: &str = "max";

define_elementwise_operation!(
    @binary
    /// [`Operation`] that computes the elementwise maximum of two Boolean or numeric values, promoting their element
    /// types and broadcasting their shapes. Boolean maxima are disjunctions. Real floating-point inputs propagate NaNs
    /// and order negative zero below positive zero. Complex inputs compare real components first, then imaginary
    /// components when the real components are equal, and select one whole input, namely the right input on exact
    /// ties and whenever a NaN makes the deciding comparison unordered. Array inputs that still carry partial sums are
    /// rejected, and their reduced-axis markers must agree.
    ///
    /// The tangent weighs each input tangent by `1` where that input is strictly larger, by `0.5` where the inputs
    /// tie, and by `0` where the other input is larger or either input is NaN, applying the lexicographic ordering to
    /// complex inputs.
    MaxOperation,
    MAX_OPERATION_NAME,
    Max,
    max,
    check_data_types = [@boolean_or_numeric],
    check_array_types = [@no_unreduced, @same_reduced_axes],
);

impl_differentiable_elementwise_operation! {
    @binary
    MaxOperation,
    jvp<C>
    where
        C::Value: ZeroLike + OneLike + Mul + Real + Imaginary + And + Compare<C::Value> + Select,
        <C::Value as Value>::DispatchDomain: Fill<f64, C::Value>,
    {
        |(left, left_tangent), (right, _)| {
            balanced_extremum_weight(&left, &right, ComparisonDirection::GreaterThan)?.mul(&left_tangent)?
        };
        |(left, _), (right, right_tangent)| {
            balanced_extremum_weight(&right, &left, ComparisonDirection::GreaterThan)?.mul(&right_tangent)?
        };
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @binary
    /// Represents the ability to select elementwise maxima. Concrete arrays compute immediately while context-carrying
    /// values apply [`MaxOperation`] through their context.
    Max,
    /// Returns the elementwise maximum of this value and `right`, promoting and broadcasting the inputs. Returns an
    /// error if the input types or metadata are unsupported.
    max(right),
    MaxOperation,
);

impl_array_elementwise_operation!(
    @binary
    Max,
    max,
    operation = "max",
    inputs = @boolean_or_numeric,
    checks = [@no_unreduced, @same_reduced_axes],
    |left, right| Ok(ArrayElement::max(&left, &right)),
);

/// Implements [`Max`] for one host primitive type.
macro_rules! impl_max_for_primitive {
    // Boolean primitives compute their maximum as a disjunction.
    (@boolean $type:ty) => {
        impl Max for $type {
            #[inline]
            fn max(&self, right: &Self) -> Result<Self, ProgramError> {
                Ok(*self || *right)
            }
        }
    };

    // Integer primitives use ordinary total-order comparison, which cannot fail.
    (@integer $type:ty) => {
        impl Max for $type {
            #[inline]
            fn max(&self, right: &Self) -> Result<Self, ProgramError> {
                Ok(::std::cmp::Ord::max(*self, *right))
            }
        }
    };

    // Floating-point primitives mirror the reference backends: NaN inputs propagate, and signed zeros order through
    // the IEEE 754 total order (so that `-0.0` sorts below `+0.0`).
    (@float $type:ty) => {
        impl Max for $type {
            #[inline]
            fn max(&self, right: &Self) -> Result<Self, ProgramError> {
                Ok(if self.is_nan() {
                    *self
                } else if right.is_nan() {
                    *right
                } else if matches!(self.total_cmp(right), ::std::cmp::Ordering::Less) {
                    *right
                } else {
                    *self
                })
            }
        }
    };
}

impl_max_for_primitive!(@boolean bool);
impl_max_for_primitive!(@integer i8);
impl_max_for_primitive!(@integer i16);
impl_max_for_primitive!(@integer i32);
impl_max_for_primitive!(@integer i64);
impl_max_for_primitive!(@integer i128);
impl_max_for_primitive!(@integer isize);
impl_max_for_primitive!(@integer u8);
impl_max_for_primitive!(@integer u16);
impl_max_for_primitive!(@integer u32);
impl_max_for_primitive!(@integer u64);
impl_max_for_primitive!(@integer u128);
impl_max_for_primitive!(@integer usize);
impl_max_for_primitive!(@float f32);
impl_max_for_primitive!(@float f64);

/// Canonical operation name for [`ClampOperation`].
pub const CLAMP_OPERATION_NAME: &str = "clamp";

/// [`Operation`] that clamps its operand elementwise into the interval delimited by a lower and an upper bound
/// (i.e., `(lower, x, upper) ↦ min(max(x, lower), upper)`, same as for StableHLO's
/// [`clamp`](https://openxla.org/stablehlo/spec#clamp)). The inputs are ordered as `[lower, input, upper]`, like the
/// operands of StableHLO's `clamp`, and may have any Boolean or numeric element types, which are promoted to a common
/// element type while their shapes are broadcast together. Crossed bounds (i.e., `lower > upper`) produce `upper`,
/// and the NaN, signed-zero, and complex ordering semantics are those of [`MinOperation`] and [`MaxOperation`].
/// Array inputs that still carry partial sums are rejected, and their reduced-axis markers must agree.
///
/// The tangent follows [JAX's `clamp`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.clamp.html). It is the
/// input tangent where `lower < x < upper`, the lower-bound tangent where `x < lower < upper`, the upper-bound tangent
/// where `upper < x`, and zero everywhere else, including where `x` equals either bound. Complex inputs apply these
/// comparisons under the lexicographic ordering of the extrema.
#[derive(Clone)]
pub struct ClampOperation<T: Type>(PhantomData<fn() -> T>);

impl<T: Type> ClampOperation<T> {
    /// Creates a new [`ClampOperation`].
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Type> Default for ClampOperation<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Type> std::fmt::Debug for ClampOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("ClampOperation")
    }
}

impl<T: Type> std::fmt::Display for ClampOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(CLAMP_OPERATION_NAME)
    }
}

impl Operation for ClampOperation<DataType> {
    type Type = DataType;

    #[inline]
    fn name(&self) -> &'static str {
        CLAMP_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[DataType],
        _region_interfaces: &[RegionInterface<DataType>],
    ) -> Result<Vec<DataType>, TypeError> {
        check_count!("input", input_types, 3, TypeError);
        check_types!(@boolean_or_numeric, CLAMP_OPERATION_NAME, input_types);
        input_types[0]
            .broadcast(&input_types[1])
            .and_then(|output_type| output_type.broadcast(&input_types[2]))
            .map(|output_type| vec![output_type])
            .map_err(|_| {
                TypeError::invalid(format!("`{CLAMP_OPERATION_NAME}` input types are not broadcast-compatible"))
            })
    }
}

impl Operation for ClampOperation<ArrayType> {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        CLAMP_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        ElementwiseOperation::infer_output_types(self, input_types)
    }
}

// Clamping is a ternary broadcasting elementwise operation, so its type inference validates the element domain and
// the reduction state of every input before broadcasting them, and the elementwise blanket supplies its batching rule.
impl ElementwiseOperation for ClampOperation<ArrayType> {
    #[inline]
    fn input_count(&self) -> usize {
        3
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 3, TypeError);
        check_types!(
            @boolean_or_numeric,
            CLAMP_OPERATION_NAME,
            [input_types[0].data_type(), input_types[1].data_type(), input_types[2].data_type()],
        );
        check_types!(@no_unreduced, CLAMP_OPERATION_NAME, input_types);
        check_types!(@same_reduced_axes, CLAMP_OPERATION_NAME, input_types[..2]);
        check_types!(@same_reduced_axes, CLAMP_OPERATION_NAME, input_types[1..]);
        Ok(vec![self.infer_elementwise_broadcast_type(input_types)?])
    }
}

impl<C: crate::contexts::Domain<Value: Clamp>> InterpretableOperation<C> for ClampOperation<C::Type>
where
    ClampOperation<C::Type>: Operation<Type = C::Type>,
{
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 3, ProgramError);
        Ok(vec![inputs[1].clamp(&inputs[0], &inputs[2])?])
    }
}

impl<C: Context<Operation: From<ClampOperation<C::Type>>>> PartiallyEvaluatableOperation<C> for ClampOperation<C::Type> where
    ClampOperation<C::Type>: Operation<Type = C::Type>
{
}

impl_differentiable_operation! {
    <T> ClampOperation<T>,
    jvp<C>
    where
        T: Type,
        C::Type: DifferentiableType,
        C::Value: ZeroLike
            + Add
            + Clamp
            + Real
            + Imaginary
            + And
            + Compare<C::Value>
            + Select
            + ElementwiseDerivativeAlignment<C::Type>,
    {
        |_operation, context, _driver, inputs| {
            // Each input contributes its tangent exactly where it determines the output, using strict comparisons
        // so that no input contributes where `x` equals either bound.
            check_count!("input", inputs, 3, ProgramError);
            let lower = &inputs[0];
            let input = &inputs[1];
            let upper = &inputs[2];
            let output_primal = input.primal().clamp(lower.primal(), upper.primal())?;
            let target = output_primal.r#type().tangent()?;
            if inputs.iter().all(|input| input.tangent().as_value().is_none()) {
                return Ok(vec![DifferentiationDual::new(output_primal, MaybeZero::Zero(target))?]);
            }
            if target.is_zero_space() {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "`{}` output type `{}` has no tangent space",
                        CLAMP_OPERATION_NAME,
                        output_primal.r#type(),
                    ),
                }
                .into());
            }

            // Compare the primals in the differential representation of the output, like every other elementwise
            // derivative, so that the selection masks and the tangents share one aligned geometry.
            let primal = context.primal_to_tangent(output_primal.clone())?;
            let lower_primal = context.primal_to_tangent(lower.primal().clone())?.align_tangent(&target, &primal)?;
            let input_primal = context.primal_to_tangent(input.primal().clone())?.align_tangent(&target, &primal)?;
            let upper_primal = context.primal_to_tangent(upper.primal().clone())?.align_tangent(&target, &primal)?;
            let mut tangent = None::<C::Value>;
            for (dual, conditions) in [
                // The lower bound contributes where it lies above the input and below the upper bound.
                (
                    lower,
                    [
                        (&lower_primal, &input_primal, ComparisonDirection::GreaterThan),
                        (&lower_primal, &upper_primal, ComparisonDirection::LessThan),
                    ]
                    .as_slice(),
                ),
                // The input contributes strictly inside the interval.
                (
                    input,
                    [
                        (&input_primal, &lower_primal, ComparisonDirection::GreaterThan),
                        (&input_primal, &upper_primal, ComparisonDirection::LessThan),
                    ]
                    .as_slice(),
                ),
                // The upper bound contributes wherever it lies below the input, including for crossed bounds.
                (upper, [(&upper_primal, &input_primal, ComparisonDirection::LessThan)].as_slice()),
            ] {
                let Some(dual_tangent) = dual.tangent().as_value() else {
                    continue;
                };
                let mut mask = None::<C::Value>;
                for (left, right, direction) in conditions {
                    let condition = lexicographic_comparison(*left, *right, *direction)?;
                    mask = Some(match mask {
                        Some(mask) => mask.and(&condition)?,
                        None => condition,
                    });
                }
                let mask = mask.unwrap();
                let dual_tangent = dual_tangent.align_tangent(&target, &primal)?;
                let contribution = C::Value::select(&mask, &dual_tangent, &dual_tangent.zero_like()?)?;
                tangent = Some(match tangent {
                    Some(tangent) => tangent.add(&contribution)?,
                    None => contribution,
                });
            }
            let tangent = tangent.map_or_else(|| MaybeZero::Zero(target), MaybeZero::Value);
            Ok(vec![DifferentiationDual::new(output_primal, tangent)?])
        }
    },
    transpose = @nonlinear,
}

/// Represents the ability to clamp values elementwise into the interval delimited by `lower` and `upper`. Concrete
/// arrays compute immediately while context-carrying values apply [`ClampOperation`] through their context.
pub trait Clamp: Sized {
    /// Clamps this value elementwise to the inclusive `[lower, upper]` interval (i.e., computes
    /// `min(max(self, lower), upper)`), promoting and broadcasting its inputs as needed. Returns
    /// an error if the input types or metadata are unsupported.
    ///
    /// # Parameters
    ///
    ///   - `lower`: Inclusive elementwise lower bound.
    ///   - `upper`: Inclusive elementwise upper bound, which takes precedence over `lower` where the two are crossed.
    fn clamp(&self, lower: &Self, upper: &Self) -> Result<Self, ProgramError>;
}

impl<
    T: Type,
    V: Value<
            Type = T,
            DispatchDomain: Context<
                Type = T,
                Value = V,
                Operation: From<<ClampOperation<T> as OperationProvider<T>>::Operation>,
            >,
        > + ManualVariationAlignment<T>,
> Clamp for V
where
    ClampOperation<T>: OperationProvider<T>,
{
    #[inline]
    fn clamp(&self, lower: &Self, upper: &Self) -> Result<Self, ProgramError> {
        let inputs = [lower.clone(), self.clone(), upper.clone()];
        let inputs = V::align_manual_variation(&inputs)?;
        let input_types = inputs.iter().map(Typed::r#type).collect::<Vec<_>>();
        let operation = <ClampOperation<T> as OperationProvider<T>>::provide(
            (),
            &input_types.iter().map(|input_type| input_type.as_ref()).collect::<Vec<_>>(),
        )?;
        Ok(self.dispatch_domain().bind(operation, Vec::new(), &inputs)?.remove(0))
    }
}

impl Clamp for Array {
    fn clamp(&self, lower: &Self, upper: &Self) -> Result<Self, ProgramError> {
        // Validate the complete signature first so that diagnostics name `clamp` rather than one of the extrema that
        // compute it, and then evaluate the StableHLO composition with the eager extremum kernels.
        Operation::infer_output_types(
            &ClampOperation::<ArrayType>::new(),
            &[lower.r#type().into_owned(), self.r#type().into_owned(), upper.r#type().into_owned()],
            &[],
        )?;
        self.max(lower)?.min(upper)
    }
}

/// Implements [`Clamp`] for one host primitive type as the StableHLO composition of its [`Max`] and [`Min`].
macro_rules! impl_clamp_for_primitive {
    ($type:ty) => {
        impl Clamp for $type {
            #[inline]
            fn clamp(&self, lower: &Self, upper: &Self) -> Result<Self, ProgramError> {
                Min::min(&Max::max(self, lower)?, upper)
            }
        }
    };
}

impl_clamp_for_primitive!(bool);
impl_clamp_for_primitive!(i8);
impl_clamp_for_primitive!(i16);
impl_clamp_for_primitive!(i32);
impl_clamp_for_primitive!(i64);
impl_clamp_for_primitive!(i128);
impl_clamp_for_primitive!(isize);
impl_clamp_for_primitive!(u8);
impl_clamp_for_primitive!(u16);
impl_clamp_for_primitive!(u32);
impl_clamp_for_primitive!(u64);
impl_clamp_for_primitive!(u128);
impl_clamp_for_primitive!(usize);
impl_clamp_for_primitive!(f32);
impl_clamp_for_primitive!(f64);

/// Returns the weight with which `candidate` receives the tangent of an extremum of `candidate` and `other`, following
/// JAX's balanced comparison: `1` where `candidate` wins under `direction` (i.e., [`ComparisonDirection::LessThan`]
/// for minima and [`ComparisonDirection::GreaterThan`] for maxima), `0.5` where the two tie, and `0` where `other`
/// wins or either input is NaN. The weight has the element type of `candidate` and the broadcast shape of both inputs.
fn balanced_extremum_weight<
    V: Value<DispatchDomain: Fill<f64, V>> + OneLike + ZeroLike + Real + Imaginary + And + Compare<V> + Select,
>(
    candidate: &V,
    other: &V,
    direction: ComparisonDirection,
) -> Result<V, ProgramError> {
    let wins = lexicographic_comparison(candidate, other, direction)?;
    let ties = lexicographic_comparison(candidate, other, ComparisonDirection::Equal)?;
    let half = candidate.dispatch_domain().fill(candidate.r#type().as_ref(), 0.5)?;
    V::select(&wins, &candidate.one_like()?, &V::select(&ties, &half, &candidate.zero_like()?)?)
}

/// Returns the Boolean mask of `left` compared with `right` under `direction`, using the ordering of the extrema. Real
/// values compare directly. Complex values, and real values paired with complex ones, compare their real parts first
/// and their imaginary parts second (i.e., the imaginary comparison decides only where the real parts are equal),
/// so a NaN in a deciding component makes the comparison false for every ordered direction.
fn lexicographic_comparison<V: Value + ZeroLike + Real + Imaginary + And + Compare<V> + Select>(
    left: &V,
    right: &V,
    direction: ComparisonDirection,
) -> Result<V, ProgramError> {
    let left_is_complex = left.r#type().is_complex();
    let right_is_complex = right.r#type().is_complex();
    if !left_is_complex && !right_is_complex {
        return left.compare(right, direction);
    }
    let left_real = if left_is_complex { left.real()? } else { left.clone() };
    let right_real = if right_is_complex { right.real()? } else { right.clone() };
    let left_imaginary = if left_is_complex { left.imaginary()? } else { left.zero_like()? };
    let right_imaginary = if right_is_complex { right.imaginary()? } else { right.zero_like()? };
    let same_real = left_real.compare(&right_real, ComparisonDirection::Equal)?;
    let imaginary = left_imaginary.compare(&right_imaginary, direction)?;
    if direction == ComparisonDirection::Equal {
        return same_real.and(&imaginary);
    }
    V::select(&same_real, &imaginary, &left_real.compare(&right_real, direction)?)
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::indoc;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayOperation, ArrayType, DataType};
    use crate::contexts::EagerContext;
    use crate::differentiation::{DifferentiableOperation, DifferentiationContext, differentiate_at};
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::programs::EmptyRegionDriver;
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_min_type_inference() {
        check_operation_type_inference!(
            @elementwise @binary,
            operation = MinOperation,
            cases = [
                {
                    input_data_types = [DataType::F32, DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::I32, DataType::I32],
                    output_data_types = [DataType::I32],
                },
                {
                    input_data_types = [DataType::C64, DataType::C64],
                    output_data_types = [DataType::C64],
                },
                {
                    input_data_types = [DataType::Boolean, DataType::Boolean],
                    output_data_types = [DataType::Boolean],
                },
                {
                    input_data_types = [DataType::Token, DataType::Token],
                    error = "`min` does not support input data type `token`",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = MinOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
        check_operation_type_inference!(
            @reject @mismatched_reduced,
            operation = MinOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_min_interpretation() {
        assert_eq!(
            MinOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::vector(vec![1.0f64, 3.0]).unwrap(), Array::scalar(2.0f64).unwrap()],
            ),
            Ok(vec![Array::vector(vec![1.0f64, 2.0]).unwrap()]),
        );
    }

    #[test]
    fn test_min_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = MinOperation::new(),
            inputs = [Array::scalar(0.7).unwrap(), Array::scalar(0.3).unwrap()],
            expected = Array::scalar(0.3).unwrap(),
        );
    }

    #[test]
    fn test_min_batching() {
        check_operation_batching!(
            @exact,
            operation = MinOperation::new(),
            axis_size = 2,
            cases = [
                {
                    inputs = [
                        (@mapped(axis = 0), Array::vector(vec![0.5, -1.0]).unwrap()),
                        (@mapped(axis = 0), Array::vector(vec![0.3, 2.0]).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![0.3, -1.0]).unwrap())],
                },
                {
                    inputs = [
                        (@replicated, Array::scalar(0.0).unwrap()),
                        (@mapped(axis = 0), Array::vector(vec![1.0, -2.0]).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![0.0, -2.0]).unwrap())],
                },
            ],
        );
    }

    #[test]
    fn test_min_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = MinOperation::new(),
            cases = [
                {
                    primals = [Array::scalar(2.0).unwrap(), Array::scalar(1.0).unwrap()],
                    tangents = [Array::scalar(3.0).unwrap(), Array::scalar(5.0).unwrap()],
                    primal_outputs = [Array::scalar(1.0).unwrap()],
                    tangent_outputs = [Array::scalar(5.0).unwrap()],
                    jvp = indoc! {"
                        lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                        let %4:f64[] = min %0 %1
                            %5:bool[] = compare [direction=LessThan] %0 %1
                            %6:bool[] = compare [direction=Equal] %0 %1
                            %7:f64[] = constant [value=0.5]
                            %8:f64[] = one_like %0
                            %9:f64[] = zero_like %0
                            %10:f64[] = select %6 %7 %9
                            %11:f64[] = select %5 %8 %10
                            %12:f64[] = mul %11 %2
                            %13:bool[] = compare [direction=LessThan] %1 %0
                            %14:bool[] = compare [direction=Equal] %1 %0
                            %15:f64[] = constant [value=0.5]
                            %16:f64[] = one_like %1
                            %17:f64[] = zero_like %1
                            %18:f64[] = select %14 %15 %17
                            %19:f64[] = select %13 %16 %18
                            %20:f64[] = mul %19 %3
                            %21:f64[] = add %12 %20
                        in (%4, %21)
                    "},
                },
                {
                    primals = [Array::scalar(1.0).unwrap(), Array::scalar(2.0).unwrap()],
                    tangents = [Array::scalar(3.0).unwrap(), Array::scalar(5.0).unwrap()],
                    primal_outputs = [Array::scalar(1.0).unwrap()],
                    tangent_outputs = [Array::scalar(3.0).unwrap()],
                },
            ],
        );
    }

    #[test]
    fn test_min_differentiation_ties() {
        // Tied inputs split the tangent evenly, which the finite-difference oracle cannot check at the kink.
        assert_eq!(
            differentiate_at((Array::scalar(2.0).unwrap(), Array::scalar(2.0).unwrap()))
                .jvp((Array::scalar(3.0).unwrap(), Array::scalar(5.0).unwrap()), |(left, right)| left.min(&right)),
            Ok((Array::scalar(2.0).unwrap(), Array::scalar(4.0).unwrap())),
        );
    }

    #[test]
    fn test_min_differentiation_nan() {
        // A NaN input makes every comparison false, so neither input receives any tangent.
        let (primal, tangent) = differentiate_at((Array::scalar(f64::NAN).unwrap(), Array::scalar(1.0).unwrap()))
            .jvp((Array::scalar(3.0).unwrap(), Array::scalar(5.0).unwrap()), |(left, right)| left.min(&right))
            .unwrap();
        assert!(primal.elements::<f64>().unwrap()[0].is_nan());
        assert_eq!(tangent, Array::scalar(0.0).unwrap());
    }

    #[test]
    fn test_min_differentiation_complex() {
        let tangents = (
            Array::scalar(ComplexNumber::new(3.0f64, 4.0)).unwrap(),
            Array::scalar(ComplexNumber::new(5.0f64, 6.0)).unwrap(),
        );

        // The imaginary parts decide where the real parts are equal, so the smaller left input takes the tangent.
        assert_eq!(
            differentiate_at((
                Array::scalar(ComplexNumber::new(1.0f64, 1.0)).unwrap(),
                Array::scalar(ComplexNumber::new(1.0f64, 2.0)).unwrap(),
            ))
            .jvp(tangents.clone(), |(left, right)| left.min(&right)),
            Ok((Array::scalar(ComplexNumber::new(1.0f64, 1.0)).unwrap(), tangents.0.clone())),
        );

        // Exactly tied complex inputs split the tangent evenly.
        assert_eq!(
            differentiate_at((
                Array::scalar(ComplexNumber::new(1.0f64, 2.0)).unwrap(),
                Array::scalar(ComplexNumber::new(1.0f64, 2.0)).unwrap(),
            ))
            .jvp(tangents.clone(), |(left, right)| left.min(&right)),
            Ok((
                Array::scalar(ComplexNumber::new(1.0f64, 2.0)).unwrap(),
                Array::scalar(ComplexNumber::new(4.0f64, 5.0)).unwrap(),
            )),
        );

        // A NaN that makes the deciding comparison unordered selects the right input but gives no input a tangent.
        assert_eq!(
            differentiate_at((
                Array::scalar(ComplexNumber::new(f64::NAN, 1.0)).unwrap(),
                Array::scalar(ComplexNumber::new(2.0f64, -1.0)).unwrap(),
            ))
            .jvp(tangents, |(left, right)| left.min(&right)),
            Ok((
                Array::scalar(ComplexNumber::new(2.0f64, -1.0)).unwrap(),
                Array::scalar(ComplexNumber::new(0.0f64, 0.0)).unwrap(),
            )),
        );
    }

    #[test]
    fn test_min_differentiation_boolean() {
        // Boolean inputs have no tangent space, so the output tangent is a structural zero.
        let outputs = MinOperation::<ArrayType>::new()
            .jvp(
                &DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new()),
                &EmptyRegionDriver,
                &[
                    DifferentiationDual::new_with_zero_tangent(Array::scalar(true).unwrap()).unwrap(),
                    DifferentiationDual::new_with_zero_tangent(Array::scalar(false).unwrap()).unwrap(),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &Array::scalar(false).unwrap());
        assert!(
            matches!(outputs[0].tangent(), MaybeZero::Zero(r#type) if r#type == &ArrayType::scalar(DataType::Zero))
        );
    }

    #[test]
    fn test_min_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = MinOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_array_min() {
        // Integer, floating-point, and half-precision inputs retain their element types.
        assert_eq!(Array::scalar(2i32).unwrap().min(&Array::scalar(5i32).unwrap()), Ok(Array::scalar(2i32).unwrap()));
        assert_eq!(
            Array::scalar(-2i64).unwrap().min(&Array::scalar(-5i64).unwrap()),
            Ok(Array::scalar(-5i64).unwrap()),
        );
        assert_eq!(Array::scalar(3u32).unwrap().min(&Array::scalar(7u32).unwrap()), Ok(Array::scalar(3u32).unwrap()));
        assert_eq!(
            Array::scalar(2.5f32).unwrap().min(&Array::scalar(1.5f32).unwrap()),
            Ok(Array::scalar(1.5f32).unwrap()),
        );
        assert_eq!(
            Array::scalar(bf16::from_f32(2.0)).unwrap().min(&Array::scalar(bf16::from_f32(3.0)).unwrap()),
            Ok(Array::scalar(bf16::from_f32(2.0)).unwrap()),
        );
        assert_eq!(
            Array::scalar(f16::from_f32(2.0)).unwrap().min(&Array::scalar(f16::from_f32(3.0)).unwrap()),
            Ok(Array::scalar(f16::from_f32(2.0)).unwrap()),
        );

        // Mixed-precision inputs promote before comparing, and vectors compare elementwise.
        assert_eq!(
            Array::scalar(2.5f32).unwrap().min(&Array::scalar(3.5f64).unwrap()),
            Ok(Array::scalar(2.5f64).unwrap()),
        );
        assert_eq!(
            Array::vector(vec![0.7, -1.0]).unwrap().min(&Array::vector(vec![0.3, 2.0]).unwrap()),
            Ok(Array::vector(vec![0.3, -1.0]).unwrap()),
        );

        // NaNs propagate from either input.
        let output = Array::scalar(f64::NAN).unwrap().min(&Array::scalar(1.0f64).unwrap()).unwrap();
        assert!(output.elements::<f64>().unwrap()[0].is_nan());
        let output = Array::scalar(1.0f64).unwrap().min(&Array::scalar(f64::NAN).unwrap()).unwrap();
        assert!(output.elements::<f64>().unwrap()[0].is_nan());
    }

    #[test]
    fn test_array_min_boolean() {
        // Boolean minima are conjunctions.
        assert_eq!(
            Array::vector(vec![false, false, true, true])
                .unwrap()
                .min(&Array::vector(vec![false, true, false, true]).unwrap()),
            Ok(Array::vector(vec![false, false, false, true]).unwrap()),
        );
    }

    #[test]
    fn test_array_min_complex() {
        // The real component takes precedence, and mixed real and complex inputs promote before selection.
        let left = Array::scalar(ComplexNumber::new(1.0f32, 100.0)).unwrap();
        let right = Array::scalar(ComplexNumber::new(2.0f32, -100.0)).unwrap();
        assert_eq!(left.min(&right), Ok(Array::scalar(ComplexNumber::new(1.0f32, 100.0)).unwrap()));
        assert_eq!(
            left.min(&Array::scalar(2.0f32).unwrap()),
            Ok(Array::scalar(ComplexNumber::new(1.0f32, 100.0)).unwrap()),
        );

        // Equal real components compare imaginary components, with scalar broadcasting across a vector.
        let values = Array::vector(vec![ComplexNumber::new(1.0f64, 1.0), ComplexNumber::new(1.0, 3.0)]).unwrap();
        assert_eq!(
            values.min(&Array::scalar(ComplexNumber::new(1.0f64, 2.0)).unwrap()),
            Ok(Array::vector(vec![ComplexNumber::new(1.0f64, 1.0), ComplexNumber::new(1.0, 2.0)]).unwrap()),
        );

        // Unordered deciding comparisons select the right input, so a NaN in the left input does not propagate,
        // and signed-zero ties select the right input as well.
        let unordered = Array::scalar(ComplexNumber::new(f32::NAN, 1.0)).unwrap();
        assert_eq!(unordered.min(&right), Ok(right.clone()));
        let output = right.min(&unordered).unwrap();
        assert!(output.elements::<ComplexNumber<f32>>().unwrap()[0].re.is_nan());
        let zero = Array::scalar(ComplexNumber::new(1.0f32, -0.0)).unwrap();
        let output = zero.min(&Array::scalar(ComplexNumber::new(1.0f32, 0.0)).unwrap()).unwrap();
        assert_eq!(output.elements::<ComplexNumber<f32>>().unwrap()[0].im.to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn test_array_min_encodings() {
        // Minima retain the selected input's NaN payload and IEEE signed-zero encoding.
        let nan = f32::from_bits(0x7fc0_1234);
        let output = Array::scalar(nan).unwrap().min(&Array::scalar(1.0f32).unwrap()).unwrap();
        assert_eq!(output.elements::<f32>().unwrap()[0].to_bits(), nan.to_bits());
        let output = Array::scalar(-0.0f32).unwrap().min(&Array::scalar(0.0f32).unwrap()).unwrap();
        assert_eq!(output.elements::<f32>().unwrap()[0].to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn test_min_primitives() {
        assert_eq!(Min::min(&3usize, &4), Ok(3));
        assert_eq!(Min::min(&true, &false), Ok(false));
        assert!(Min::min(&1.0f64, &f64::NAN).unwrap().is_nan());
        assert_eq!(Min::min(&0.0f64, &-0.0).unwrap().to_bits(), (-0.0f64).to_bits());
    }

    #[test]
    fn test_max_type_inference() {
        check_operation_type_inference!(
            @elementwise @binary,
            operation = MaxOperation,
            cases = [
                {
                    input_data_types = [DataType::F32, DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::I32, DataType::I32],
                    output_data_types = [DataType::I32],
                },
                {
                    input_data_types = [DataType::C64, DataType::C64],
                    output_data_types = [DataType::C64],
                },
                {
                    input_data_types = [DataType::Boolean, DataType::Boolean],
                    output_data_types = [DataType::Boolean],
                },
                {
                    input_data_types = [DataType::Token, DataType::Token],
                    error = "`max` does not support input data type `token`",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = MaxOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
        check_operation_type_inference!(
            @reject @mismatched_reduced,
            operation = MaxOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_max_interpretation() {
        assert_eq!(
            MaxOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::vector(vec![1.0f64, 3.0]).unwrap(), Array::scalar(2.0f64).unwrap()],
            ),
            Ok(vec![Array::vector(vec![2.0f64, 3.0]).unwrap()]),
        );
    }

    #[test]
    fn test_max_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = MaxOperation::new(),
            inputs = [Array::scalar(0.7).unwrap(), Array::scalar(0.3).unwrap()],
            expected = Array::scalar(0.7).unwrap(),
        );
    }

    #[test]
    fn test_max_batching() {
        check_operation_batching!(
            @exact,
            operation = MaxOperation::new(),
            axis_size = 2,
            cases = [
                {
                    inputs = [
                        (@mapped(axis = 0), Array::vector(vec![0.5, -1.0]).unwrap()),
                        (@mapped(axis = 0), Array::vector(vec![0.3, 2.0]).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![0.5, 2.0]).unwrap())],
                },
                {
                    inputs = [
                        (@replicated, Array::scalar(0.0).unwrap()),
                        (@mapped(axis = 0), Array::vector(vec![1.0, -2.0]).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![1.0, 0.0]).unwrap())],
                },
            ],
        );
    }

    #[test]
    fn test_max_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = MaxOperation::new(),
            cases = [
                {
                    primals = [Array::scalar(2.0).unwrap(), Array::scalar(1.0).unwrap()],
                    tangents = [Array::scalar(3.0).unwrap(), Array::scalar(5.0).unwrap()],
                    primal_outputs = [Array::scalar(2.0).unwrap()],
                    tangent_outputs = [Array::scalar(3.0).unwrap()],
                    jvp = indoc! {"
                        lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                        let %4:f64[] = max %0 %1
                            %5:bool[] = compare [direction=GreaterThan] %0 %1
                            %6:bool[] = compare [direction=Equal] %0 %1
                            %7:f64[] = constant [value=0.5]
                            %8:f64[] = one_like %0
                            %9:f64[] = zero_like %0
                            %10:f64[] = select %6 %7 %9
                            %11:f64[] = select %5 %8 %10
                            %12:f64[] = mul %11 %2
                            %13:bool[] = compare [direction=GreaterThan] %1 %0
                            %14:bool[] = compare [direction=Equal] %1 %0
                            %15:f64[] = constant [value=0.5]
                            %16:f64[] = one_like %1
                            %17:f64[] = zero_like %1
                            %18:f64[] = select %14 %15 %17
                            %19:f64[] = select %13 %16 %18
                            %20:f64[] = mul %19 %3
                            %21:f64[] = add %12 %20
                        in (%4, %21)
                    "},
                },
                {
                    primals = [Array::scalar(1.0).unwrap(), Array::scalar(2.0).unwrap()],
                    tangents = [Array::scalar(3.0).unwrap(), Array::scalar(5.0).unwrap()],
                    primal_outputs = [Array::scalar(2.0).unwrap()],
                    tangent_outputs = [Array::scalar(5.0).unwrap()],
                },
            ],
        );
    }

    #[test]
    fn test_max_differentiation_ties() {
        // Tied inputs split the tangent evenly, which the finite-difference oracle cannot check at the kink.
        assert_eq!(
            differentiate_at((Array::scalar(2.0).unwrap(), Array::scalar(2.0).unwrap()))
                .jvp((Array::scalar(3.0).unwrap(), Array::scalar(5.0).unwrap()), |(left, right)| left.max(&right)),
            Ok((Array::scalar(2.0).unwrap(), Array::scalar(4.0).unwrap())),
        );
    }

    #[test]
    fn test_max_differentiation_nan() {
        // A NaN input makes every comparison false, so neither input receives any tangent.
        let (primal, tangent) = differentiate_at((Array::scalar(1.0).unwrap(), Array::scalar(f64::NAN).unwrap()))
            .jvp((Array::scalar(3.0).unwrap(), Array::scalar(5.0).unwrap()), |(left, right)| left.max(&right))
            .unwrap();
        assert!(primal.elements::<f64>().unwrap()[0].is_nan());
        assert_eq!(tangent, Array::scalar(0.0).unwrap());
    }

    #[test]
    fn test_max_differentiation_complex() {
        let tangents = (
            Array::scalar(ComplexNumber::new(3.0f64, 4.0)).unwrap(),
            Array::scalar(ComplexNumber::new(5.0f64, 6.0)).unwrap(),
        );

        // The imaginary parts decide where the real parts are equal, so the larger right input takes the tangent.
        assert_eq!(
            differentiate_at((
                Array::scalar(ComplexNumber::new(1.0f64, 1.0)).unwrap(),
                Array::scalar(ComplexNumber::new(1.0f64, 2.0)).unwrap(),
            ))
            .jvp(tangents.clone(), |(left, right)| left.max(&right)),
            Ok((Array::scalar(ComplexNumber::new(1.0f64, 2.0)).unwrap(), tangents.1.clone())),
        );

        // Exactly tied complex inputs split the tangent evenly.
        assert_eq!(
            differentiate_at((
                Array::scalar(ComplexNumber::new(1.0f64, 2.0)).unwrap(),
                Array::scalar(ComplexNumber::new(1.0f64, 2.0)).unwrap(),
            ))
            .jvp(tangents, |(left, right)| left.max(&right)),
            Ok((
                Array::scalar(ComplexNumber::new(1.0f64, 2.0)).unwrap(),
                Array::scalar(ComplexNumber::new(4.0f64, 5.0)).unwrap(),
            )),
        );
    }

    #[test]
    fn test_max_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = MaxOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_array_max() {
        // Integer, floating-point, and half-precision inputs retain their element types.
        assert_eq!(Array::scalar(2i32).unwrap().max(&Array::scalar(5i32).unwrap()), Ok(Array::scalar(5i32).unwrap()));
        assert_eq!(
            Array::scalar(-2i64).unwrap().max(&Array::scalar(-5i64).unwrap()),
            Ok(Array::scalar(-2i64).unwrap()),
        );
        assert_eq!(Array::scalar(3u32).unwrap().max(&Array::scalar(7u32).unwrap()), Ok(Array::scalar(7u32).unwrap()));
        assert_eq!(
            Array::scalar(2.5f32).unwrap().max(&Array::scalar(1.5f32).unwrap()),
            Ok(Array::scalar(2.5f32).unwrap()),
        );
        assert_eq!(
            Array::scalar(bf16::from_f32(2.0)).unwrap().max(&Array::scalar(bf16::from_f32(3.0)).unwrap()),
            Ok(Array::scalar(bf16::from_f32(3.0)).unwrap()),
        );
        assert_eq!(
            Array::scalar(f16::from_f32(2.0)).unwrap().max(&Array::scalar(f16::from_f32(3.0)).unwrap()),
            Ok(Array::scalar(f16::from_f32(3.0)).unwrap()),
        );

        // Mixed-precision inputs promote before comparing, and vectors compare elementwise.
        assert_eq!(
            Array::scalar(2.5f32).unwrap().max(&Array::scalar(3.5f64).unwrap()),
            Ok(Array::scalar(3.5f64).unwrap()),
        );
        assert_eq!(
            Array::vector(vec![0.7, -1.0]).unwrap().max(&Array::vector(vec![0.3, 2.0]).unwrap()),
            Ok(Array::vector(vec![0.7, 2.0]).unwrap()),
        );

        // NaNs propagate from either input.
        let output = Array::scalar(f64::NAN).unwrap().max(&Array::scalar(1.0f64).unwrap()).unwrap();
        assert!(output.elements::<f64>().unwrap()[0].is_nan());
        let output = Array::scalar(1.0f64).unwrap().max(&Array::scalar(f64::NAN).unwrap()).unwrap();
        assert!(output.elements::<f64>().unwrap()[0].is_nan());
    }

    #[test]
    fn test_array_max_boolean() {
        // Boolean maxima are disjunctions.
        assert_eq!(
            Array::vector(vec![false, false, true, true])
                .unwrap()
                .max(&Array::vector(vec![false, true, false, true]).unwrap()),
            Ok(Array::vector(vec![false, true, true, true]).unwrap()),
        );
    }

    #[test]
    fn test_array_max_complex() {
        // The real component takes precedence, and mixed real and complex inputs promote before selection.
        let left = Array::scalar(ComplexNumber::new(1.0f32, 100.0)).unwrap();
        let right = Array::scalar(ComplexNumber::new(2.0f32, -100.0)).unwrap();
        assert_eq!(left.max(&right), Ok(Array::scalar(ComplexNumber::new(2.0f32, -100.0)).unwrap()));
        assert_eq!(
            left.max(&Array::scalar(2.0f32).unwrap()),
            Ok(Array::scalar(ComplexNumber::new(2.0f32, 0.0)).unwrap()),
        );

        // Equal real components compare imaginary components, with scalar broadcasting across a vector.
        let values = Array::vector(vec![ComplexNumber::new(1.0f64, 1.0), ComplexNumber::new(1.0, 3.0)]).unwrap();
        assert_eq!(
            values.max(&Array::scalar(ComplexNumber::new(1.0f64, 2.0)).unwrap()),
            Ok(Array::vector(vec![ComplexNumber::new(1.0f64, 2.0), ComplexNumber::new(1.0, 3.0)]).unwrap()),
        );

        // Unordered deciding comparisons select the right input, so a NaN in the left input does not propagate,
        // and signed-zero ties select the right input as well.
        let unordered = Array::scalar(ComplexNumber::new(f32::NAN, 1.0)).unwrap();
        assert_eq!(unordered.max(&right), Ok(right.clone()));
        let output = right.max(&unordered).unwrap();
        assert!(output.elements::<ComplexNumber<f32>>().unwrap()[0].re.is_nan());
        let zero = Array::scalar(ComplexNumber::new(1.0f32, -0.0)).unwrap();
        let output = zero.max(&Array::scalar(ComplexNumber::new(1.0f32, 0.0)).unwrap()).unwrap();
        assert_eq!(output.elements::<ComplexNumber<f32>>().unwrap()[0].im.to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn test_array_max_encodings() {
        // Maxima retain the selected input's NaN payload and IEEE signed-zero encoding.
        let nan = f32::from_bits(0x7fc0_1234);
        let output = Array::scalar(nan).unwrap().max(&Array::scalar(1.0f32).unwrap()).unwrap();
        assert_eq!(output.elements::<f32>().unwrap()[0].to_bits(), nan.to_bits());
        let output = Array::scalar(-0.0f32).unwrap().max(&Array::scalar(0.0f32).unwrap()).unwrap();
        assert_eq!(output.elements::<f32>().unwrap()[0].to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn test_max_primitives() {
        assert_eq!(Max::max(&3usize, &4), Ok(4));
        assert_eq!(Max::max(&true, &false), Ok(true));
        assert!(Max::max(&1.0f64, &f64::NAN).unwrap().is_nan());
        assert_eq!(Max::max(&0.0f64, &-0.0).unwrap().to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn test_clamp() {
        let operation = ClampOperation::<ArrayType>::new();
        assert_eq!(operation.name(), CLAMP_OPERATION_NAME);
        assert_eq!(operation.to_string(), "clamp");
        assert_eq!(format!("{operation:?}"), "ClampOperation");

        // Clamping stages one operation whose inputs are ordered as `[lower, input, upper]`.
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(input, lower, upper): (_, _, _)| input.clamp(&lower, &upper),
            (
                ArrayType::new_static(DataType::F32, [2]),
                ArrayType::scalar(DataType::F32),
                ArrayType::scalar(DataType::F32),
            ),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[], %2:f32[] .
                let %3:f32[2] = clamp %1 %0 %2
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_clamp_type_inference() {
        check_operation_type_inference!(
            operation = ClampOperation::<DataType>::new(),
            cases = [
                {
                    input_types = [DataType::F32, DataType::F64, DataType::F32],
                    output_types = [DataType::F64],
                },
                {
                    input_types = [DataType::Boolean, DataType::Boolean, DataType::Boolean],
                    output_types = [DataType::Boolean],
                },
                {
                    input_types = [DataType::I32, DataType::Token, DataType::I32],
                    error = "`clamp` does not support input data type `token`",
                },
                {
                    input_types = [DataType::F32, DataType::F32],
                    error = "expected 3 inputs but got 2",
                },
            ],
        );

        // Array bounds broadcast against the operand, including scalar bounds.
        check_operation_type_inference!(
            operation = ClampOperation::<ArrayType>::new(),
            cases = [{
                input_types = [
                    ArrayType::scalar(DataType::F32),
                    ArrayType::new_static(DataType::F32, [2, 3]),
                    ArrayType::new_static(DataType::F64, [3]),
                ],
                output_types = [ArrayType::new_static(DataType::F64, [2, 3])],
            }],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = ClampOperation::<ArrayType>::new(),
            input_types = [
                ArrayType::scalar(DataType::F64),
                ArrayType::scalar(DataType::F64),
                ArrayType::scalar(DataType::F64),
            ],
        );
    }

    #[test]
    fn test_clamp_interpretation() {
        // The operand is the second input, between the lower and the upper bound.
        assert_eq!(
            ClampOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[
                    Array::scalar(-1.0f64).unwrap(),
                    Array::vector(vec![-2.0f64, 0.5, 3.0]).unwrap(),
                    Array::scalar(1.0f64).unwrap(),
                ],
            ),
            Ok(vec![Array::vector(vec![-1.0f64, 0.5, 1.0]).unwrap()]),
        );
    }

    #[test]
    fn test_clamp_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = ClampOperation::new(),
            inputs = [Array::scalar(-1.0).unwrap(), Array::scalar(3.0).unwrap(), Array::scalar(1.0).unwrap()],
            expected = Array::scalar(1.0).unwrap(),
        );
    }

    #[test]
    fn test_clamp_batching() {
        check_operation_batching!(
            @exact,
            operation = ClampOperation::new(),
            axis_size = 3,
            cases = [
                {
                    inputs = [
                        (@replicated, Array::scalar(-1.0).unwrap()),
                        (@mapped(axis = 0), Array::vector(vec![-2.0, 0.5, 3.0]).unwrap()),
                        (@replicated, Array::scalar(1.0).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![-1.0, 0.5, 1.0]).unwrap())],
                },
                {
                    inputs = [
                        (@mapped(axis = 0), Array::vector(vec![0.0, 1.0, 2.0]).unwrap()),
                        (@replicated, Array::scalar(1.5).unwrap()),
                        (@replicated, Array::scalar(1.8).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![1.5, 1.5, 1.8]).unwrap())],
                },
            ],
        );
    }

    #[test]
    fn test_clamp_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = ClampOperation::new(),
            cases = [
                // Strictly inside the interval, the tangent follows the input.
                {
                    primals = [Array::scalar(-1.0).unwrap(), Array::scalar(0.5).unwrap(), Array::scalar(1.0).unwrap()],
                    tangents = [Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap(), Array::scalar(5.0).unwrap()],
                    primal_outputs = [Array::scalar(0.5).unwrap()],
                    tangent_outputs = [Array::scalar(3.0).unwrap()],
                    jvp = indoc! {"
                        lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[], %4:f64[], %5:f64[] .
                        let %6:f64[] = clamp %0 %1 %2
                            %7:bool[] = compare [direction=GreaterThan] %0 %1
                            %8:bool[] = compare [direction=LessThan] %0 %2
                            %9:bool[] = and %7 %8
                            %10:f64[] = zero_like %3
                            %11:f64[] = select %9 %3 %10
                            %12:bool[] = compare [direction=GreaterThan] %1 %0
                            %13:bool[] = compare [direction=LessThan] %1 %2
                            %14:bool[] = and %12 %13
                            %15:f64[] = zero_like %4
                            %16:f64[] = select %14 %4 %15
                            %17:f64[] = add %11 %16
                            %18:bool[] = compare [direction=LessThan] %2 %1
                            %19:f64[] = zero_like %5
                            %20:f64[] = select %18 %5 %19
                            %21:f64[] = add %17 %20
                        in (%6, %21)
                    "},
                },
                // Below the interval, the tangent follows the lower bound.
                {
                    primals = [Array::scalar(-1.0).unwrap(), Array::scalar(-2.0).unwrap(), Array::scalar(1.0).unwrap()],
                    tangents = [Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap(), Array::scalar(5.0).unwrap()],
                    primal_outputs = [Array::scalar(-1.0).unwrap()],
                    tangent_outputs = [Array::scalar(2.0).unwrap()],
                },
                // Above the interval, the tangent follows the upper bound.
                {
                    primals = [Array::scalar(-1.0).unwrap(), Array::scalar(3.0).unwrap(), Array::scalar(1.0).unwrap()],
                    tangents = [Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap(), Array::scalar(5.0).unwrap()],
                    primal_outputs = [Array::scalar(1.0).unwrap()],
                    tangent_outputs = [Array::scalar(5.0).unwrap()],
                },
            ],
        );
    }

    #[test]
    fn test_clamp_differentiation_boundaries() {
        // At either bound no input determines the output alone, so the tangent is zero.
        let tangents = (Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap(), Array::scalar(5.0).unwrap());
        for input in [-1.0, 1.0] {
            assert_eq!(
                differentiate_at((
                    Array::scalar(-1.0).unwrap(),
                    Array::scalar(input).unwrap(),
                    Array::scalar(1.0).unwrap()
                ))
                .jvp(tangents.clone(), |(lower, input, upper)| input.clamp(&lower, &upper))
                .map(|(_, tangent)| tangent),
                Ok(Array::scalar(0.0).unwrap()),
            );
        }

        // Crossed bounds produce the upper bound, whose tangent flows only where it lies below the input.
        assert_eq!(
            differentiate_at((Array::scalar(1.0).unwrap(), Array::scalar(2.0).unwrap(), Array::scalar(-1.0).unwrap()))
                .jvp(tangents.clone(), |(lower, input, upper)| input.clamp(&lower, &upper)),
            Ok((Array::scalar(-1.0).unwrap(), Array::scalar(5.0).unwrap())),
        );
        assert_eq!(
            differentiate_at((Array::scalar(1.0).unwrap(), Array::scalar(-2.0).unwrap(), Array::scalar(-1.0).unwrap()))
                .jvp(tangents, |(lower, input, upper)| input.clamp(&lower, &upper)),
            Ok((Array::scalar(-1.0).unwrap(), Array::scalar(0.0).unwrap())),
        );
    }

    #[test]
    fn test_clamp_differentiation_complex() {
        // Complex inputs apply the tangent rule under the lexicographic ordering of the extrema.
        let lower = Array::scalar(ComplexNumber::new(0.0f64, 0.0)).unwrap();
        let upper = Array::scalar(ComplexNumber::new(2.0f64, 0.0)).unwrap();
        let tangent = Array::scalar(ComplexNumber::new(3.0f64, 4.0)).unwrap();
        let zero = Array::scalar(ComplexNumber::new(0.0f64, 0.0)).unwrap();
        assert_eq!(
            differentiate_at((lower, Array::scalar(ComplexNumber::new(1.0f64, 5.0)).unwrap(), upper))
                .jvp((zero.clone(), tangent.clone(), zero), |(lower, input, upper)| input.clamp(&lower, &upper)),
            Ok((Array::scalar(ComplexNumber::new(1.0f64, 5.0)).unwrap(), tangent)),
        );
    }

    #[test]
    fn test_clamp_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = ClampOperation::<ArrayType>::new(),
            input_types = [
                ArrayType::scalar(DataType::F64),
                ArrayType::scalar(DataType::F64),
                ArrayType::scalar(DataType::F64),
            ],
        );
    }

    #[test]
    fn test_array_clamp() {
        // Values below, inside, and above the interval, and integer inputs.
        let lower = Array::scalar(-1.0f64).unwrap();
        let upper = Array::scalar(1.0f64).unwrap();
        assert_eq!(
            Array::vector(vec![-2.5f64, 0.5, 2.5]).unwrap().clamp(&lower, &upper),
            Ok(Array::vector(vec![-1.0f64, 0.5, 1.0]).unwrap()),
        );
        assert_eq!(
            Array::scalar(7i32).unwrap().clamp(&Array::scalar(0i32).unwrap(), &Array::scalar(5i32).unwrap()),
            Ok(Array::scalar(5i32).unwrap()),
        );

        // Bounds broadcast elementwise and promote the result element type.
        assert_eq!(
            Array::vector(vec![0.0f32, 5.0])
                .unwrap()
                .clamp(&Array::vector(vec![1.0f64, 2.0]).unwrap(), &Array::vector(vec![3.0f64, 4.0]).unwrap(),),
            Ok(Array::vector(vec![1.0f64, 4.0]).unwrap()),
        );

        // Crossed bounds produce the upper bound.
        assert_eq!(
            Array::scalar(0.0f64)
                .unwrap()
                .clamp(&Array::scalar(1.0f64).unwrap(), &Array::scalar(-1.0f64).unwrap()),
            Ok(Array::scalar(-1.0f64).unwrap()),
        );

        // NaNs propagate from the input and from either bound.
        let output = Array::scalar(f64::NAN).unwrap().clamp(&lower, &upper).unwrap();
        assert!(output.elements::<f64>().unwrap()[0].is_nan());
        let output = Array::scalar(0.0f64).unwrap().clamp(&Array::scalar(f64::NAN).unwrap(), &upper).unwrap();
        assert!(output.elements::<f64>().unwrap()[0].is_nan());
        let output = Array::scalar(0.0f64).unwrap().clamp(&lower, &Array::scalar(f64::NAN).unwrap()).unwrap();
        assert!(output.elements::<f64>().unwrap()[0].is_nan());

        // A scalar input broadcasts against array bounds.
        assert_eq!(
            Array::scalar(0.0f64).unwrap().clamp(&lower, &Array::vector(vec![1.0f64, 2.0, 3.0]).unwrap()),
            Ok(Array::vector(vec![0.0f64, 0.0, 0.0]).unwrap()),
        );

        // Diagnostics name the clamp rather than the extrema that compute it.
        assert_eq!(
            Array::vector(vec![0.0f64, 1.0])
                .unwrap()
                .clamp(&lower, &Array::vector(vec![1.0f64, 2.0, 3.0]).unwrap()),
            Err(TypeError::invalid("`clamp` input types are not broadcast-compatible").into()),
        );
    }

    #[test]
    fn test_array_clamp_boolean() {
        // Boolean clamps compose the conjunction and disjunction of the extrema.
        assert_eq!(
            Array::vector(vec![false, true])
                .unwrap()
                .clamp(&Array::scalar(true).unwrap(), &Array::scalar(true).unwrap()),
            Ok(Array::vector(vec![true, true]).unwrap()),
        );
        assert_eq!(
            Array::vector(vec![false, true])
                .unwrap()
                .clamp(&Array::scalar(false).unwrap(), &Array::scalar(false).unwrap()),
            Ok(Array::vector(vec![false, false]).unwrap()),
        );
    }

    #[test]
    fn test_array_clamp_complex() {
        // Complex clamps compare lexicographically, so the imaginary part decides between equal real parts.
        let lower = Array::scalar(ComplexNumber::new(1.0f64, 0.0)).unwrap();
        let upper = Array::scalar(ComplexNumber::new(1.0f64, 2.0)).unwrap();
        assert_eq!(
            Array::vector(vec![
                ComplexNumber::new(1.0f64, -1.0),
                ComplexNumber::new(1.0, 1.0),
                ComplexNumber::new(1.0, 3.0)
            ])
            .unwrap()
            .clamp(&lower, &upper),
            Ok(Array::vector(vec![
                ComplexNumber::new(1.0f64, 0.0),
                ComplexNumber::new(1.0, 1.0),
                ComplexNumber::new(1.0, 2.0)
            ])
            .unwrap()),
        );
    }

    #[test]
    fn test_clamp_primitives() {
        assert_eq!(Clamp::clamp(&2.5f64, &-1.0, &1.0), Ok(1.0));
        assert_eq!(Clamp::clamp(&-3i32, &0, &5), Ok(0));
        assert_eq!(Clamp::clamp(&0.0f64, &1.0, &-1.0), Ok(-1.0));
        assert_eq!(Clamp::clamp(&false, &true, &true), Ok(true));
    }
}
