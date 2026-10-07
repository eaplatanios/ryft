//! Contains [`SelectOperation`], which chooses each output element from one of two branch inputs according to the
//! corresponding element of a Boolean condition input, together with the [`Select`] value capability that computes
//! or stages it and its interpretation, partial evaluation, batching, forward mode differentiation, and transposition
//! rules. Unlike [`ConditionOperation`](crate::ConditionOperation), which executes only one of its attached branch
//! [`Region`](crate::Region)s, `select` consumes two ordinary branch values that have both already been computed.
//!
//! `select(condition, on_true, on_false)` takes the `on_true` element wherever `condition` is `true` and the `on_false`
//! element elsewhere. The shapes of all three inputs broadcast together, and the two branch [`DataType`]s promote to
//! the output data type. The condition must be [`DataType::Boolean`] and it does not take part in that promotion
//! because it is a mask rather than a value. This is the three-argument form of JAX's
//! [`jax.numpy.where`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.where.html), which is more permissive
//! than [`jax.lax.select`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.select.html), whose branches must share
//! a shape and data type and whose condition must be a scalar or have that shape. The output keeps the broadcast
//! placement of its inputs and uses the dense row-major layout. Branches that carry pending reductions must carry
//! identical reduction state, which the output inherits, while the condition must neither be unreduced nor vary over
//! any of those reduction axes, because selection commutes with a pending sum only when every shard selects the same
//! way.
//!
//! For a fixed condition, `select` is linear in its two branches. Forward mode differentiation therefore selects the
//! branch tangents under the primal condition and keeps a structural-zero output tangent when both branch tangents are
//! structural zeros. Transposition requires a known condition and routes the output cotangent into the branch that each
//! element selected (i.e., `select(condition, cotangent, 0)` for `on_true` and `select(condition, 0, cotangent)` for
//! `on_false`), reducing broadcast axes and converting element types back to each linear branch's cotangent type. The
//! Boolean condition has no tangent space, and a known branch receives no cotangent. Note that the unselected branch
//! still receives a zero cotangent. A non-finite derivative inside that branch (e.g., of `sqrt` at zero) therefore
//! turns its zero cotangent into `NaN`, so a branch that is only well-defined where it is selected should also be
//! guarded on its own input. Refer to the related JAX
//! [FAQ](https://docs.jax.dev/en/latest/faq.html#gradients-contain-nan-where-using-where) for more information on this
//! topic. Batching follows the standard elementwise broadcasting rule. Backends lower `select` to their elementwise
//! selection construct after broadcasting and converting its inputs (e.g.,
//! [`stablehlo.select`](https://openxla.org/stablehlo/spec#select) in the XLA backend).
//!
//! # Example
//!
//! Selecting between a scalar `f32` branch and an `f64` vector branch stages one instruction whose output broadcasts
//! and promotes both branches:
//!
//! ```rust
//! # use indoc::indoc;
//! # use ryft_core::{Array, ArrayOperation, ArrayType, DataType, ProgramError, Select, TracingContext};
//! # fn main() -> Result<(), ProgramError> {
//! let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
//!     |(condition, on_true, on_false)| Select::select(&condition, &on_true, &on_false),
//!     (
//!         ArrayType::new_static(DataType::Boolean, [3]),
//!         ArrayType::scalar(DataType::F32),
//!         ArrayType::new_static(DataType::F64, [3]),
//!     ),
//! )?;
//!
//! assert_eq!(
//!     program.to_string(),
//!     indoc! {"
//!         lambda %0:bool[3], %1:f32[], %2:f64[3] .
//!         let %3:f64[3] = select %0 %1 %2
//!         in (%3)"},
//! );
//!
//! assert_eq!(
//!     program.interpret((
//!         Array::vector(vec![true, false, true])?,
//!         Array::scalar(7.0f32)?,
//!         Array::vector(vec![4.0f64, 5.0, 6.0])?,
//!     ))?,
//!     Array::vector(vec![7.0f64, 5.0, 7.0])?,
//! );
//! # Ok(())
//! # }
//! ```

use std::fmt::Display;
use std::marker::PhantomData;
use std::sync::Arc;

use ryft_macros::capability;

use crate::arrays::{
    Array, ArrayAddressing, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, Broadcastable,
    DataType, Sharding, ShardingDimension,
};
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{DifferentiableType, DifferentiationDual, ElementwiseDerivativeAlignment};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::collectives::parallel_vary::ManualVariationAlignment;
use crate::operations::constants::zero::{Zero, ZeroOperation};
use crate::operations::constants::zero_like::ZeroLikeOperation;
use crate::operations::{Capability, ElementwiseOperation};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    MaybeZero, Operation, OperationProvider, ProgramError, RegionInterface, Type, TypeError, Typed, Value,
    ValueDomainDispatch, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

/// Canonical operation name for [`SelectOperation`].
pub const SELECT_OPERATION_NAME: &str = "select";

/// [`Operation`] that chooses each output element from one of two branch inputs according to the corresponding element
/// of a Boolean condition input. Its inputs are, in order, the condition, the `on_true` branch, and the `on_false`
/// branch. The `T` parameter fixes the operation's [`Type`] universe, and so `SelectOperation<DataType>` and
/// `SelectOperation<ArrayType>` are distinct zero-sized operations that each implement exactly one [`Operation`]
/// contract. Refer to the documentation of [`Select`] for more information.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SelectOperation<T: Type>(PhantomData<fn() -> T>);

impl<T: Type> SelectOperation<T> {
    /// Creates a new [`SelectOperation`].
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Type> Copy for SelectOperation<T> {}

impl<T: Type> Display for SelectOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(SELECT_OPERATION_NAME)
    }
}

impl Operation for SelectOperation<DataType> {
    type Type = DataType;

    #[inline]
    fn name(&self) -> &'static str {
        SELECT_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[DataType],
        _region_interfaces: &[RegionInterface<DataType>],
    ) -> Result<Vec<DataType>, TypeError> {
        check_count!("input", input_types, 3, TypeError);
        if !input_types[0].is_boolean() {
            return Err(TypeError::invalid(format!(
                "`{}` condition data type `{}` is not `{}`",
                SELECT_OPERATION_NAME,
                input_types[0],
                DataType::Boolean,
            )));
        }

        // The Boolean condition is a mask rather than a value, so only the two branch data types are promoted.
        input_types[1].broadcast(&input_types[2]).map(|output| vec![output]).map_err(|_| {
            TypeError::invalid(format!("`{SELECT_OPERATION_NAME}` input types are not broadcast-compatible"))
        })
    }
}

impl Operation for SelectOperation<ArrayType> {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        SELECT_OPERATION_NAME
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

// `select` broadcasts the shapes and placements of its three inputs like any other elementwise operation, but its type
// inference differs in two ways. First, the Boolean condition is a mask rather than a value, and so only the two branch
// data types are promoted into the output data type. Second, selection commutes with a pending reduction only when
// every shard selects the same way. The branches must therefore carry identical reduction state, which the output
// inherits, while the condition must neither be unreduced nor vary over any of the branches' reduction axes. An
// already-reduced condition is an ordinary Boolean value and is accepted. Implementing `ElementwiseOperation` also
// gives `select` the standard elementwise batching rule through the corresponding blanket `BatchableOperation`
// implementation.
impl ElementwiseOperation for SelectOperation<ArrayType> {
    #[inline]
    fn input_count(&self) -> usize {
        3
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 3, TypeError);
        let (condition, on_true, on_false) = (&input_types[0], &input_types[1], &input_types[2]);
        if !condition.data_type().is_boolean() {
            return Err(TypeError::invalid(format!(
                "`{}` condition data type `{}` is not `{}`",
                SELECT_OPERATION_NAME,
                condition.data_type(),
                DataType::Boolean,
            )));
        }

        let unreduced = on_true.sharding().map(Sharding::unreduced_axes).cloned().unwrap_or_default();
        let reduced = on_true.sharding().map(Sharding::reduced_axes).cloned().unwrap_or_default();
        if on_false.sharding().map(Sharding::unreduced_axes).cloned().unwrap_or_default() != unreduced
            || on_false.sharding().map(Sharding::reduced_axes).cloned().unwrap_or_default() != reduced
        {
            return Err(TypeError::invalid(format!(
                "`{SELECT_OPERATION_NAME}` branches must carry identical reduction state",
            )));
        }

        if let Some(sharding) = condition.sharding() {
            if !sharding.unreduced_axes().is_empty() {
                return Err(TypeError::invalid(format!(
                    "`{SELECT_OPERATION_NAME}` condition must not carry unreduced state",
                )));
            }

            for axis in unreduced.union(&reduced) {
                if sharding.varying_manual_axes().contains(axis)
                    || sharding
                        .dimensions()
                        .iter()
                        .any(|dimension| matches!(dimension, ShardingDimension::Sharded(axes) if axes.contains(axis)))
                {
                    return Err(TypeError::invalid(format!(
                        "`{SELECT_OPERATION_NAME}` condition must be invariant over the branches' reduction axes",
                    )));
                }
            }
        }

        // Broadcast geometry and placement over descriptors without pending reductions, retagging the condition with
        // a branch data type so that it does not take part in promotion, and then restore the shared branch reduction
        // state on the output. These descriptors only drive inference and no runtime value is retagged.
        let condition = condition.without_reduction_axes().with_data_type(on_true.data_type());
        let mut output_type = self.infer_elementwise_broadcast_type(&[
            condition,
            on_true.without_reduction_axes(),
            on_false.without_reduction_axes(),
        ])?;
        if let Some(sharding) = output_type.sharding().cloned() {
            let sharding = sharding
                .with_unreduced_axes(unreduced)
                .and_then(|sharding| sharding.with_reduced_axes(reduced))
                .map_err(|error| TypeError::invalid(error.to_string()))?;
            output_type = output_type.with_sharding(sharding).map_err(|error| TypeError::invalid(error.to_string()))?;
        }
        Ok(vec![output_type])
    }
}

impl<C: Domain<Value: Select>> InterpretableOperation<C> for SelectOperation<C::Type>
where
    Self: Operation<Type = C::Type>,
{
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 3, ProgramError);
        Ok(vec![C::Value::select(&inputs[0], &inputs[1], &inputs[2])?])
    }
}

impl<C: Context<Operation: From<SelectOperation<C::Type>>>> PartiallyEvaluatableOperation<C>
    for SelectOperation<C::Type>
where
    Self: Operation<Type = C::Type>,
{
}

impl_differentiable_operation! {
    <T> SelectOperation<T>,
    jvp<C>
    where
        T: Type,
        C: Zero<C::Value>,
        C::Type: DifferentiableType,
        C::Value: ElementwiseDerivativeAlignment<C::Type>,
        C::Operation: From<SelectOperation<C::Type>> + From<ZeroLikeOperation<C::Type>>,
    {
        |_operation, context, _driver, inputs| {
            // For a fixed condition, `select` is linear in its branches, and so the output tangent selects the branch
            // tangents under the primal condition. Two structural-zero branch tangents yield a structural-zero output
            // tangent. Otherwise, the staged tangent `select` needs both branches, and so a single structural-zero
            // branch tangent is materialized. The tangent `select` broadcasts and promotes its branches like the
            // primal one, and its result is then aligned with the tangent type of the primal output.
            check_count!("input", inputs, 3, ProgramError);
            let (condition, on_true, on_false) = (&inputs[0], &inputs[1], &inputs[2]);
            let mut primal = context.primal().bind(
                SelectOperation::new(),
                Vec::new(),
                &[condition.primal().clone(), on_true.primal().clone(), on_false.primal().clone()],
            )?;
            check_count!("output", primal, 1, ProgramError);
            let primal = primal.remove(0);
            let tangent_type = primal.r#type().tangent()?;
            let tangent = if on_true.tangent().is_zero() && on_false.tangent().is_zero() {
                MaybeZero::Zero(tangent_type)
            } else {
                let condition = context.primal_to_tangent(condition.primal().clone())?;
                let exemplar = context.primal_to_tangent(primal.clone())?;

                // Some primal formats cannot represent zero and use a wider tangent format. Convert the shape
                // exemplar before constructing any zero or using it to broadcast a tangent.
                let exemplar = exemplar.align_tangent(&tangent_type, &exemplar)?;
                let mut tangent_inputs = vec![condition];
                for branch in [on_true, on_false] {
                    // Align live tangents to the output type before selection. Missing tangents use that same type,
                    // which also lets an integer branch contribute zero when promotion gives the output a tangent
                    // space. Dynamic output geometry comes from the primal exemplar rather than a nullary zero.
                    let tangent = match branch.tangent() {
                        MaybeZero::Value(tangent) => tangent.align_tangent(&tangent_type, &exemplar)?,
                        MaybeZero::Zero(_) if tangent_type.identities().next().is_some() => {
                            let mut zero = context.tangent().bind(
                                ZeroLikeOperation::new(),
                                Vec::new(),
                                std::slice::from_ref(&exemplar),
                            )?;
                            check_count!("output", zero, 1, ProgramError);
                            zero.remove(0).align_tangent(&tangent_type, &exemplar)?
                        }
                        MaybeZero::Zero(_) => MaybeZero::Zero(tangent_type.clone()).materialize(context.tangent())?,
                    };
                    tangent_inputs.push(tangent);
                }
                let mut tangent = context.tangent().bind(SelectOperation::new(), Vec::new(), &tangent_inputs)?;
                check_count!("output", tangent, 1, ProgramError);
                MaybeZero::Value(tangent.remove(0).align_tangent(&tangent_type, &exemplar)?)
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        T: Type,
        V::Type: DifferentiableType,
        O: From<ZeroLikeOperation<V::Type>>
            + From<SelectOperation<V::Type>>
            + OperationProvider<V::Type, ZeroOperation<V::Type>, Operation = O>,
        Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<V::Type>,
    {
        |_operation, context, _driver, inputs, outputs, accumulators| {
            // The Boolean condition has no tangent space, and so it must be a known input. Each linear branch then
            // receives the output cotangent at the elements that the condition selected from it and zero elsewhere
            // (i.e., `select(condition, cotangent, 0)` for `on_true` and `select(condition, 0, cotangent)` for
            // `on_false`), reduced over its broadcast axes and converted to its own cotangent type. A known branch,
            // an unrequested cotangent, and a structural-zero output cotangent stage no work.
            check_count!("input", inputs, 3, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 3, DifferentiationError);
            if inputs[0].is_unknown() {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "operation `{}` does not support transposition for input pattern \
                         [condition = linear, on_true = {}, on_false = {}]",
                        SELECT_OPERATION_NAME,
                        if inputs[1].is_unknown() { "linear" } else { "known" },
                        if inputs[2].is_unknown() { "linear" } else { "known" },
                    ),
                }
                .into());
            }

            let MaybeZero::Value(cotangent) = &outputs[0] else { return Ok(()) };
            if !accumulators[1].is_needed() && !accumulators[2].is_needed() {
                return Ok(());
            }

            // The condition was checked to be known above.
            let condition = inputs[0].as_known().unwrap();
            let cotangent_type = cotangent.r#type().into_owned();
            let zero = if cotangent_type.identities().next().is_some() {
                // A type with symbolic extents does not determine its runtime extents, so the live cotangent serves as
                // the shape exemplar. Identity-free types use the canonical nullary zero, whose zero-producing marker
                // keeps higher-order partial evaluation structural.
                let mut zero =
                    context.stage_operation(ZeroLikeOperation::new(), Vec::new(), std::slice::from_ref(cotangent))?;
                check_count!("output", zero, 1, ProgramError);
                zero.remove(0)
            } else {
                MaybeZero::Zero(cotangent_type).materialize(&**context)?
            };

            for branch in [1, 2] {
                if !accumulators[branch].is_needed() {
                    continue;
                }
                let selected_inputs = if branch == 1 {
                    [condition.clone(), cotangent.clone(), zero.clone()]
                } else {
                    [condition.clone(), zero.clone(), cotangent.clone()]
                };
                let mut contribution = context.stage_operation(SelectOperation::new(), Vec::new(), &selected_inputs)?;
                check_count!("output", contribution, 1, ProgramError);
                let contribution = contribution.remove(0).unalign_cotangent(&inputs[branch].r#type().cotangent()?)?;
                accumulators[branch].accumulate(context, MaybeZero::Value(contribution))?;
            }
            Ok(())
        }
    },
}

impl<A: Value<Type = ArrayType>> From<SelectOperation<ArrayIrType>> for ArrayIrOperation<A> {
    #[inline]
    fn from(_: SelectOperation<ArrayIrType>) -> Self {
        // A `select` reads its complete output geometry, including every runtime extent, from its inputs, and so the
        // homogeneous array member already expresses the dynamic case and the composite family needs no mixed encoding
        // for it. This conversion lets composite-typed values stage `select` through the shared `Select` blanket
        // implementation. Projecting the member rejects first-class dimension and reference inputs.
        Self::Array(ArrayOperation::Select(SelectOperation::new()))
    }
}

/// Represents the ability to choose each element from one of two values according to a Boolean condition. [`Select`]
/// supplies [`SelectOperation`]'s interpretation capability. `select(condition, on_true, on_false)` takes the `on_true`
/// element wherever `condition` is `true` and the `on_false` element elsewhere. The condition is represented by the
/// same value type as the branches (e.g., a Boolean [`Array`] for concrete arrays and a Boolean [`Tracer`] for staged
/// values).
///
/// For arrays, the three input shapes broadcast together and the two branch element data types promote to the output
/// data type, so the inputs need not share a shape and the branches need not share a data type. After any required
/// element type conversion, selection copies the chosen element encodings exactly, including NaN payloads and signed
/// zeros. Equal-typed branches retain their original encodings, and the output uses the dense row-major layout. A
/// staged call retains all three inputs, so dynamic extents are read at execution time. Selection is linear in its
/// branches for a fixed condition. Refer to the [module documentation](self) for its sharding, differentiation, and
/// batching semantics.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and composite array IR values implement it for [`ArrayIrType`].
///
/// # Example
///
/// ```rust
/// # use ryft_core::{Array, ProgramError, Select};
/// let condition = Array::vector(vec![true, false, true])?;
/// let on_true = Array::vector(vec![1.0, 2.0, 3.0])?;
/// let on_false = Array::scalar(0.0)?;
/// assert_eq!(Array::select(&condition, &on_true, &on_false)?.to_f64s(), vec![1.0, 0.0, 3.0]);
/// # Ok::<(), ProgramError>(())
/// ```
#[capability]
pub trait Select<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Returns the value whose elements are taken from `on_true` wherever `condition` is `true` and from `on_false`
    /// elsewhere. The inputs broadcast and the branch element types promote as described on [`Select`].
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if `condition` is not Boolean, if the inputs are not broadcast-compatible, if their
    /// reduction state violates the constraints in the [module documentation](self), or if their context fails to
    /// bind the operation.
    fn select(condition: &Self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError>;
}

impl Select for Array {
    fn select(condition: &Self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        let output_type = ElementwiseOperation::infer_output_types(
            &SelectOperation::<ArrayType>::new(),
            &[condition.r#type().into_owned(), on_true.r#type().into_owned(), on_false.r#type().into_owned()],
        )?
        .remove(0);

        // Only a branch whose element data type differs from the output data type is converted, so an equal-typed
        // branch is read from its original storage and layout. Every input is addressed through its own physical
        // layout under broadcasting, and the selected element bytes are copied into a dense output.
        let output_data_type = output_type.data_type();
        let on_true = on_true.promoted_to(output_data_type)?;
        let on_false = on_false.promoted_to(output_data_type)?;

        let output_shape = output_type.static_shape().unwrap();
        let condition_shape = condition.r#type().static_shape().unwrap();
        let on_true_shape = on_true.r#type().static_shape().unwrap();
        let on_false_shape = on_false.r#type().static_shape().unwrap();
        let output_strides = output_shape.row_major_strides();
        let condition_strides = condition_shape.row_major_strides();
        let on_true_strides = on_true_shape.row_major_strides();
        let on_false_strides = on_false_shape.row_major_strides();
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let condition_addressing = ArrayAddressing::new(condition.r#type().into_owned())?;
        let on_true_addressing = ArrayAddressing::new(on_true.r#type().into_owned())?;
        let on_false_addressing = ArrayAddressing::new(on_false.r#type().into_owned())?;
        let mut output_bytes = vec![0; output_addressing.storage_byte_len()];
        for output_index in 0..output_addressing.element_count() {
            let condition_index = Self::broadcast_index(
                output_index,
                &output_shape,
                &output_strides,
                &condition_shape,
                &condition_strides,
            );
            let condition_range = condition_addressing.byte_range_for_flat_index(condition_index);
            let (source, source_range) = if condition.storage_bytes()[condition_range.start] != 0 {
                let source_index = Self::broadcast_index(
                    output_index,
                    &output_shape,
                    &output_strides,
                    &on_true_shape,
                    &on_true_strides,
                );
                (on_true.storage_bytes(), on_true_addressing.byte_range_for_flat_index(source_index))
            } else {
                let source_index = Self::broadcast_index(
                    output_index,
                    &output_shape,
                    &output_strides,
                    &on_false_shape,
                    &on_false_strides,
                );
                (on_false.storage_bytes(), on_false_addressing.byte_range_for_flat_index(source_index))
            };
            let output_range = output_addressing.byte_range_for_flat_index(output_index);
            output_bytes[output_range].copy_from_slice(&source[source_range]);
        }

        Ok(Self::new_unchecked(output_type, Arc::new(output_bytes)))
    }
}

impl<A: Value<Type = ArrayType> + Select> Select for ArrayIrValue<A> {
    fn select(condition: &Self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        let condition = <Self as ValueProjection<ArrayType>>::projected(condition)?;
        let on_true = <Self as ValueProjection<ArrayType>>::projected(on_true)?;
        let on_false = <Self as ValueProjection<ArrayType>>::projected(on_false)?;
        Ok(Self::Array(A::select(condition, on_true, on_false)?))
    }
}

// Context-carrying values (e.g., staged tracers, batching tracers, and differentiation tracers) select by binding a
// `SelectOperation` through their own context after aligning the manual variation of all three inputs. The
// `ValueDomainDispatch` marker keeps this implementation disjoint from the eager value implementations above.
impl<
    T: Type,
    V: Value<Type = T, Dispatch = ValueDomainDispatch, Domain: Context<Operation: From<SelectOperation<T>>>>
        + ManualVariationAlignment<T>,
> Select<T> for V
{
    fn select(condition: &Self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        let inputs = V::align_manual_variation(&[condition.clone(), on_true.clone(), on_false.clone()])?;
        let mut outputs = condition.domain().bind(SelectOperation::new(), Vec::new(), &inputs)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use num_complex::Complex;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Dimension, DimensionBounds, DimensionType, DimensionValue, DimensionVariable, Layout, LogicalMesh, MeshAxis,
        MeshAxisType, Shape, StridedLayout, i4,
    };
    use crate::contexts::EagerContext;
    use crate::differentiation::{
        DifferentiableOperation, DifferentiationContext, DifferentiationError, TransposableOperation,
        TranspositionContext, differentiate_at,
    };
    use crate::macros::{
        check_operation_batching, check_operation_partial_evaluation, check_operation_transposition,
        check_operation_type_inference,
    };
    use crate::operations::comparisons::{Compare, ComparisonDirection};
    use crate::operations::constants::zero_like::ZeroLike;
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::{EmptyRegionDriver, ProgramBuilder};

    use super::*;

    #[test]
    fn test_select() {
        // Verify the operation's identity and rendering in both type universes.
        let operation = SelectOperation::<ArrayType>::new();
        assert_eq!(operation.name(), SELECT_OPERATION_NAME);
        assert_eq!(format!("{operation}"), SELECT_OPERATION_NAME);
        assert_eq!(SelectOperation::<DataType>::new().name(), SELECT_OPERATION_NAME);

        // Verify the operation's textual form when it appears in a program.
        let mut builder = ProgramBuilder::<Array, SelectOperation<ArrayType>>::new();
        let condition = builder.add_input(ArrayType::new_static(DataType::Boolean, [2]));
        let on_true = builder.add_input(ArrayType::scalar(DataType::F32));
        let on_false = builder.add_input(ArrayType::new_static(DataType::F64, [2]));
        let output =
            builder.add_instruction(operation, Vec::new(), vec![condition, on_true, on_false], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[2], %1:f32[], %2:f64[2] .
                let %3:f64[2] = select %0 %1 %2
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_select_type_inference() {
        // Scalar data types require a Boolean condition and promote only the two branches.
        check_operation_type_inference!(
            operation = SelectOperation::<DataType>::new(),
            cases = [
                {
                    input_types = [DataType::Boolean, DataType::F32, DataType::F64],
                    output_types = [DataType::F64],
                },
                {
                    input_types = [DataType::Boolean, DataType::I32, DataType::I32],
                    output_types = [DataType::I32],
                },
                {
                    type = DataType,
                    input_types = [],
                    error = "expected 3 inputs but got 0",
                },
                {
                    input_types = [DataType::F32, DataType::F32, DataType::F32],
                    error = "`select` condition data type `f32` is not `bool`",
                },
                {
                    input_types = [DataType::Boolean, DataType::F8E3M4, DataType::F32],
                    error = "`select` input types are not broadcast-compatible",
                },
            ],
        );

        // Array shapes broadcast across all three inputs, including a condition that is larger than its branches and
        // symbolic extents, while the branch element data types promote.
        let condition = ArrayType::new_static(DataType::Boolean, [3]);
        let branch = ArrayType::new_static(DataType::F64, [3]);
        let length = Dimension::Dynamic(DimensionVariable::new("length", DimensionBounds::unbounded()));
        let dynamic_condition = ArrayType::new(DataType::Boolean, Shape::new(vec![length.clone()]));
        let dynamic_branch = ArrayType::new(DataType::F32, Shape::new(vec![length]));
        check_operation_type_inference!(
            operation = SelectOperation::<ArrayType>::new(),
            cases = [
                {
                    input_types = [condition.clone(), branch.clone(), branch.clone()],
                    output_types = [branch.clone()],
                },
                {
                    input_types = [condition.clone(), ArrayType::scalar(DataType::F32), branch.clone()],
                    output_types = [branch.clone()],
                },
                {
                    input_types = [ArrayType::scalar(DataType::Boolean), branch.clone(), branch.clone()],
                    output_types = [branch.clone()],
                },
                {
                    input_types = [
                        ArrayType::new_static(DataType::Boolean, [2, 3]),
                        ArrayType::scalar(DataType::F32),
                        ArrayType::new_static(DataType::F32, [3]),
                    ],
                    output_types = [ArrayType::new_static(DataType::F32, [2, 3])],
                },
                {
                    input_types = [dynamic_condition, ArrayType::scalar(DataType::F32), dynamic_branch.clone()],
                    output_types = [dynamic_branch],
                },
                {
                    input_types = [branch.clone(), branch.clone(), branch.clone()],
                    error = "`select` condition data type `f64` is not `bool`",
                },
                {
                    input_types = [ArrayType::new_static(DataType::Boolean, [2]), branch.clone(), branch.clone()],
                    error = "`select` input types are not broadcast-compatible",
                },
                {
                    input_types = [
                        condition,
                        ArrayType::new_static(DataType::F8E3M4, [3]),
                        ArrayType::new_static(DataType::F32, [3]),
                    ],
                    error = "`select` input types are not broadcast-compatible",
                },
            ],
        );
    }

    #[test]
    fn test_select_type_inference_reduction_state() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let unreduced = Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let reduced = Sharding::replicated(mesh.clone(), 1).with_reduced_axes(["x"]).unwrap();
        let unreduced_branch = ArrayType::new_static(DataType::F32, [2]).with_sharding(unreduced.clone()).unwrap();
        let reduced_branch = ArrayType::new_static(DataType::F32, [2]).with_sharding(reduced.clone()).unwrap();
        let condition = ArrayType::new_static(DataType::Boolean, [2]);
        check_operation_type_inference!(
            operation = SelectOperation::<ArrayType>::new(),
            cases = [
                // Branches with identical reduction state pass that state on, including under a reduced condition.
                {
                    input_types = [condition.clone(), unreduced_branch.clone(), unreduced_branch.clone()],
                    output_types = [unreduced_branch.clone()],
                },
                {
                    input_types = [condition.clone(), reduced_branch.clone(), reduced_branch.clone()],
                    output_types = [reduced_branch.clone()],
                },
                {
                    input_types = [
                        condition.clone().with_sharding(reduced).unwrap(),
                        unreduced_branch.clone(),
                        unreduced_branch.clone(),
                    ],
                    output_types = [unreduced_branch.clone()],
                },
                // Mismatched branch states and conditions that differ across the reduction axes are rejected.
                {
                    input_types = [
                        condition.clone(),
                        unreduced_branch.clone(),
                        unreduced_branch.without_reduction_axes(),
                    ],
                    error = "`select` branches must carry identical reduction state",
                },
                {
                    input_types = [condition.clone(), unreduced_branch.clone(), reduced_branch],
                    error = "`select` branches must carry identical reduction state",
                },
                {
                    input_types = [
                        condition.clone().with_sharding(unreduced).unwrap(),
                        unreduced_branch.clone(),
                        unreduced_branch.clone(),
                    ],
                    error = "`select` condition must not carry unreduced state",
                },
                {
                    input_types = [
                        condition.with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
                            .unwrap(),
                        unreduced_branch.clone(),
                        unreduced_branch,
                    ],
                    error = "`select` condition must be invariant over the branches' reduction axes",
                },
            ],
        );
    }

    #[test]
    fn test_select_type_inference_mixed() {
        let operation = ArrayIrOperation::<Array>::from(SelectOperation::<ArrayIrType>::new());
        assert!(matches!(operation, ArrayIrOperation::Array(ArrayOperation::Select(_))));
        let size = DimensionVariable::new("size", DimensionBounds::non_negative(Some(8)).unwrap());
        let condition = ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Dynamic(size.clone())]));
        let branch = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(size.clone())]));
        assert_eq!(
            operation
                .infer_output_types(&[condition.clone().into(), branch.clone().into(), branch.clone().into()], &[]),
            Ok(vec![branch.clone().into()]),
        );
        assert_eq!(
            operation.infer_output_types(&[condition.into(), DimensionType::from(size).into(), branch.into()], &[]),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
    }

    #[test]
    fn test_select_interpretation() {
        // Interpretation selects elementwise, broadcasting the scalar branch and promoting the branch data types.
        assert_eq!(
            SelectOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[
                    Array::vector(vec![true, false, true]).unwrap(),
                    Array::scalar(7.0f32).unwrap(),
                    Array::vector(vec![4.0f64, 5.0, 6.0]).unwrap(),
                ],
            ),
            Ok(vec![Array::vector(vec![7.0f64, 5.0, 7.0]).unwrap()]),
        );
        assert_eq!(
            Array::select(
                &Array::scalar(false).unwrap(),
                &Array::vector(vec![1i32, 2]).unwrap(),
                &Array::vector(vec![-1i32, -2]).unwrap(),
            ),
            Ok(Array::vector(vec![-1i32, -2]).unwrap()),
        );
        assert_eq!(
            Array::select(
                &Array::vector(Vec::<bool>::new()).unwrap(),
                &Array::scalar(1.0f32).unwrap(),
                &Array::vector(Vec::<f32>::new()).unwrap(),
            ),
            Ok(Array::vector(Vec::<f32>::new()).unwrap()),
        );
        assert_eq!(
            Array::select(&Array::scalar(1f32).unwrap(), &Array::scalar(2f32).unwrap(), &Array::scalar(3f32).unwrap()),
            Err(ProgramError::Type(TypeError::invalid("`select` condition data type `f32` is not `bool`"))),
        );

        // Selection copies element encodings exactly, including NaN payloads and signed zeros.
        let on_true = Array::vector(vec![f32::from_bits(0x7fc00123), 1.0]).unwrap();
        let on_false = Array::vector(vec![2.0f32, -0.0]).unwrap();
        assert_eq!(
            Array::select(&Array::vector(vec![true, false]).unwrap(), &on_true, &on_false)
                .unwrap()
                .storage_bytes(),
            Array::vector(vec![f32::from_bits(0x7fc00123), -0.0]).unwrap().storage_bytes(),
        );

        // Sub-byte and complex branches preserve their complete element encodings as well.
        assert_eq!(
            Array::select(
                &Array::vector(vec![true, false]).unwrap(),
                &Array::vector(vec![i4::MIN, i4::MAX]).unwrap(),
                &Array::scalar(i4::new(-1).unwrap()).unwrap(),
            ),
            Ok(Array::vector(vec![i4::MIN, i4::new(-1).unwrap()]).unwrap()),
        );
        assert_eq!(
            Array::select(
                &Array::vector(vec![true, false]).unwrap(),
                &Array::scalar(Complex::new(1.0f64, 2.0)).unwrap(),
                &Array::vector(vec![Complex::new(3.0f64, 4.0), Complex::new(5.0, 6.0)]).unwrap(),
            ),
            Ok(Array::vector(vec![Complex::new(1.0f64, 2.0), Complex::new(5.0, 6.0)]).unwrap()),
        );

        // Every input is read through its own physical layout under general broadcasting, and the output is dense.
        let condition = Array::from_elements(
            ArrayType::new_static(DataType::Boolean, [2, 1])
                .with_layout(Layout::Strided(StridedLayout::new(vec![-3, 1]))),
            &[true, false],
        )
        .unwrap();
        let on_true = Array::from_elements(
            ArrayType::new_static(DataType::U16, [1, 3]).with_layout(Layout::Strided(StridedLayout::new(vec![8, -2]))),
            &[0x1111u16, 0x2222, 0x3333],
        )
        .unwrap();
        let on_false = Array::from_elements(
            ArrayType::new_static(DataType::U16, [2, 1]).with_layout(Layout::Strided(StridedLayout::new(vec![-4, 2]))),
            &[0xaaaau16, 0xbbbb],
        )
        .unwrap();
        let output = Array::select(&condition, &on_true, &on_false).unwrap();
        assert_eq!(output.r#type().as_ref(), &ArrayType::new_static(DataType::U16, [2, 3]));
        assert_eq!(output.elements::<u16>(), Ok(vec![0x1111, 0x2222, 0x3333, 0xbbbb, 0xbbbb, 0xbbbb]));
        assert_eq!(output.storage_bytes(), [0x11, 0x11, 0x22, 0x22, 0x33, 0x33, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb]);

        // The output keeps the branches' shared reduction state.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let branch_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh, 1).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        assert_eq!(
            Array::select(
                &Array::vector(vec![true, false]).unwrap(),
                &Array::from_elements(branch_type.clone(), &[1f32, 2.0]).unwrap(),
                &Array::from_elements(branch_type.clone(), &[3f32, 4.0]).unwrap(),
            ),
            Ok(Array::from_elements(branch_type, &[1f32, 4.0]).unwrap()),
        );
    }

    #[test]
    fn test_select_interpretation_mixed() {
        let condition = ArrayIrValue::Array(Array::vector(vec![true, false]).unwrap());
        let on_true = ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0]).unwrap());
        let on_false = ArrayIrValue::Array(Array::vector(vec![3.0f32, 4.0]).unwrap());
        assert_eq!(
            ArrayIrValue::select(&condition, &on_true, &on_false),
            Ok(ArrayIrValue::Array(Array::vector(vec![1.0f32, 4.0]).unwrap())),
        );
        let dimension = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(2).unwrap());
        assert_eq!(
            ArrayIrValue::select(&condition, &dimension, &on_false),
            Err(ProgramError::Type(TypeError::invalid("expected array type but got dimension type"))),
        );
    }

    #[test]
    fn test_select_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = SelectOperation::<ArrayType>::new(),
            inputs = [
                Array::vector(vec![true, false]).unwrap(),
                Array::scalar(2.0f32).unwrap(),
                Array::vector(vec![3.0f64, 4.0]).unwrap(),
            ],
            expected = Array::vector(vec![2.0f64, 4.0]).unwrap(),
        );
    }

    #[test]
    fn test_select_batching() {
        check_operation_batching!(
            @exact,
            operation = SelectOperation::<ArrayType>::new(),
            axis_size = 2,
            cases = [
                // A mapped condition selects between a replicated branch and a mapped branch.
                {
                    inputs = [
                        (@mapped(axis = 0), Array::vector(vec![true, false]).unwrap()),
                        (@replicated, Array::scalar(2.0).unwrap()),
                        (@mapped(axis = 0), Array::vector(vec![3.0, 4.0]).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![2.0, 4.0]).unwrap())],
                },
                // A mapped per-item scalar condition broadcasts against replicated vector branches.
                {
                    inputs = [
                        (@mapped(axis = 0), Array::vector(vec![true, false]).unwrap()),
                        (@replicated, Array::vector(vec![1.0, 2.0, 3.0]).unwrap()),
                        (@replicated, Array::vector(vec![4.0, 5.0, 6.0]).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 0), Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap())],
                },
                // Branches mapped at different axes are aligned onto the first mapped input's axis.
                {
                    inputs = [
                        (@replicated, Array::vector(vec![true, false, true]).unwrap()),
                        (@mapped(axis = 1), Array::matrix(3, 2, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]).unwrap()),
                        (@mapped(axis = 0), Array::matrix(2, 3, vec![-1.0, -2.0, -3.0, -10.0, -20.0, -30.0]).unwrap()),
                    ],
                    outputs = [
                        (@mapped(axis = 1), Array::matrix(3, 2, vec![1.0, 10.0, -2.0, -20.0, 3.0, 30.0]).unwrap()),
                    ],
                },
                // Replicated inputs produce a replicated output.
                {
                    inputs = [
                        (@replicated, Array::scalar(true).unwrap()),
                        (@replicated, Array::scalar(1.0).unwrap()),
                        (@replicated, Array::scalar(2.0).unwrap()),
                    ],
                    outputs = [(@replicated, Array::scalar(1.0).unwrap())],
                },
            ],
        );
    }

    #[test]
    fn test_select_differentiation() {
        // Differentiation routes tangents and cotangents through the selected branch. These checks stay explicit
        // because the finite-difference oracle of `check_operation_differentiation!` cannot perturb a Boolean input.

        /// Selects a scaled branch under a comparison of the two inputs.
        fn piecewise<V: Clone + Compare<V> + Select + std::ops::Add<Output = V>>(
            x: V,
            y: V,
        ) -> Result<V, ProgramError> {
            let mask = x.compare(&y, ComparisonDirection::GreaterThan)?;
            Select::select(&mask, &(x.clone() + x), &(y.clone() + y.clone() + y))
        }

        assert_eq!(
            differentiate_at((Array::scalar(3.0).unwrap(), Array::scalar(2.0).unwrap()))
                .jvp((Array::scalar(1.0).unwrap(), Array::scalar(1.0).unwrap()), |(x, y)| piecewise(x, y)),
            Ok((Array::scalar(6.0).unwrap(), Array::scalar(2.0).unwrap())),
        );
        assert_eq!(
            differentiate_at((Array::scalar(3.0).unwrap(), Array::scalar(2.0).unwrap()))
                .value_and_gradient(|(x, y)| piecewise(x, y).unwrap()),
            Ok((Array::scalar(6.0).unwrap(), (Array::scalar(2.0).unwrap(), Array::scalar(0.0).unwrap()))),
        );
        assert_eq!(
            differentiate_at((Array::scalar(1.0).unwrap(), Array::scalar(2.0).unwrap()))
                .jvp((Array::scalar(1.0).unwrap(), Array::scalar(1.0).unwrap()), |(x, y)| piecewise(x, y)),
            Ok((Array::scalar(6.0).unwrap(), Array::scalar(3.0).unwrap())),
        );
        assert_eq!(
            differentiate_at((Array::scalar(1.0).unwrap(), Array::scalar(2.0).unwrap()))
                .value_and_gradient(|(x, y)| piecewise(x, y).unwrap()),
            Ok((Array::scalar(6.0).unwrap(), (Array::scalar(0.0).unwrap(), Array::scalar(3.0).unwrap()))),
        );

        // Branch tangents broadcast and promote with their primals.
        assert_eq!(
            differentiate_at((Array::scalar(2.0f32).unwrap(), Array::vector(vec![-1.0f64, 3.0]).unwrap()))
                .jvp((Array::scalar(1.0f32).unwrap(), Array::vector(vec![10.0f64, 20.0]).unwrap()), |(x, y)| {
                    Select::select(&y.compare(&y.zero_like()?, ComparisonDirection::LessThan)?, &x, &y)
                },),
            Ok((Array::vector(vec![2.0f64, 3.0]).unwrap(), Array::vector(vec![1.0f64, 20.0]).unwrap())),
        );

        // Invoke the rule directly so that the driver's zero-tangent shortcut cannot hide its staged form. Two
        // structural-zero branch tangents stage only the primal `select`, while one live branch tangent materializes
        // the other branch's zero tangent for the tangent `select`.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let branch_type = ArrayType::new_static(DataType::F64, [2]);
        let condition = context.input(ArrayType::new_static(DataType::Boolean, [2]));
        let on_true = context.input(branch_type.clone());
        let on_false = context.input(branch_type.clone());
        let on_true_tangent = context.input(branch_type.clone());
        let outputs = SelectOperation::<ArrayType>::new()
            .jvp(
                &DifferentiationContext::fused(context.clone()),
                &EmptyRegionDriver,
                &[
                    DifferentiationDual::new_with_zero_tangent(condition.clone()).unwrap(),
                    DifferentiationDual::new_with_zero_tangent(on_true.clone()).unwrap(),
                    DifferentiationDual::new_with_zero_tangent(on_false.clone()).unwrap(),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].tangent().is_zero());
        assert_eq!(outputs[0].tangent().r#type().as_ref(), &branch_type.tangent().unwrap());
        assert_eq!(context.builder().borrow().instructions().len(), 1);
        let outputs = SelectOperation::<ArrayType>::new()
            .jvp(
                &DifferentiationContext::fused(context.clone()),
                &EmptyRegionDriver,
                &[
                    DifferentiationDual::new_with_zero_tangent(condition).unwrap(),
                    DifferentiationDual::new(on_true, MaybeZero::Value(on_true_tangent)).unwrap(),
                    DifferentiationDual::new_with_zero_tangent(on_false).unwrap(),
                ],
            )
            .unwrap();
        let MaybeZero::Value(tangent) = outputs[0].tangent() else { panic!("expected a live output tangent") };
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(
                vec![outputs[0].primal().atom_id().unwrap(), tangent.atom_id().unwrap()],
                vec![Placeholder; 4],
                vec![Placeholder; 2],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[2], %1:f64[2], %2:f64[2], %3:f64[2] .
                let %4:f64[2] = select %0 %1 %2
                    %5:f64[2] = select %0 %1 %2
                    %6:f64[2] = zero [type=f64[2]]
                    %7:f64[2] = select %0 %3 %6
                in (%5, %7)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_select_differentiation_higher_order() {
        // Differentiating the gradient of a selected square returns the selected branch's second derivative.
        for (input, first, second) in [(3.0f64, 6.0f64, 2.0f64), (-3.0, 0.0, 0.0)] {
            assert_eq!(
                differentiate_at(Array::scalar(input).unwrap()).value_and_gradient(|value| {
                    differentiate_at(value)
                        .gradient(|value| {
                            let zero = value.zero_like()?;
                            let condition = value.compare(&zero, ComparisonDirection::GreaterThan)?;
                            Select::select(&condition, &(value.clone() * value), &zero)
                        })
                        .map_err(Into::into)
                }),
                Ok((Array::scalar(first).unwrap(), Array::scalar(second).unwrap())),
            );
        }
    }

    #[test]
    fn test_select_differentiation_dynamic() {
        // Exercise both positions of a missing tangent with symbolic geometry, including runtime empty arrays.
        for (live_true_branch, true_data_type) in
            [(true, DataType::F64), (false, DataType::F64), (false, DataType::I32)]
        {
            let context = TracingContext::<Array, ArrayOperation<Array>>::new();
            let size = DimensionVariable::new("size", DimensionBounds::non_negative(Some(8)).unwrap());
            let shape = Shape::new(vec![Dimension::Dynamic(size)]);
            let branch_type = ArrayType::new(DataType::F64, shape.clone());
            let condition = context.input(ArrayType::new(DataType::Boolean, shape));
            let on_true = context.input(branch_type.clone().with_data_type(true_data_type));
            let on_false = context.input(branch_type.clone());
            let tangent = context.input(branch_type);
            let true_dual = if live_true_branch {
                DifferentiationDual::new(on_true, MaybeZero::Value(tangent.clone())).unwrap()
            } else {
                DifferentiationDual::new_with_zero_tangent(on_true).unwrap()
            };
            let false_dual = if live_true_branch {
                DifferentiationDual::new_with_zero_tangent(on_false).unwrap()
            } else {
                DifferentiationDual::new(on_false, MaybeZero::Value(tangent)).unwrap()
            };
            let outputs = SelectOperation::<ArrayType>::new()
                .jvp(
                    &DifferentiationContext::fused(context.clone()),
                    &EmptyRegionDriver,
                    &[DifferentiationDual::new_with_zero_tangent(condition).unwrap(), true_dual, false_dual],
                )
                .unwrap();
            let MaybeZero::Value(tangent) = outputs[0].tangent() else { panic!("expected a live tangent") };
            let program = context
                .builder()
                .borrow()
                .clone()
                .build::<Vec<Array>, Vec<Array>>(
                    vec![outputs[0].primal().atom_id().unwrap(), tangent.atom_id().unwrap()],
                    vec![Placeholder; 4],
                    vec![Placeholder; 2],
                )
                .unwrap();
            assert_eq!(
                program.interpret(vec![
                    Array::vector(vec![true, false]).unwrap(),
                    if true_data_type == DataType::I32 {
                        Array::vector(vec![1i32, 2]).unwrap()
                    } else {
                        Array::vector(vec![1.0f64, 2.0]).unwrap()
                    },
                    Array::vector(vec![3.0f64, 4.0]).unwrap(),
                    Array::vector(vec![5.0f64, 7.0]).unwrap(),
                ]),
                Ok(vec![
                    Array::vector(vec![1.0f64, 4.0]).unwrap(),
                    Array::vector(if live_true_branch { vec![5.0f64, 0.0] } else { vec![0.0f64, 7.0] }).unwrap(),
                ]),
            );
            assert_eq!(
                program.interpret(vec![
                    Array::vector(Vec::<bool>::new()).unwrap(),
                    if true_data_type == DataType::I32 {
                        Array::vector(Vec::<i32>::new()).unwrap()
                    } else {
                        Array::vector(Vec::<f64>::new()).unwrap()
                    },
                    Array::vector(Vec::<f64>::new()).unwrap(),
                    Array::vector(Vec::<f64>::new()).unwrap(),
                ]),
                Ok(vec![Array::vector(Vec::<f64>::new()).unwrap(); 2]),
            );
        }
    }

    #[test]
    fn test_select_differentiation_dynamic_widened_tangent() {
        // This primal format cannot represent zero. Its F32 tangent exemplar must be used for the missing tangent.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let size = DimensionVariable::new("size", DimensionBounds::non_negative(Some(8)).unwrap());
        let shape = Shape::new(vec![Dimension::Dynamic(size)]);
        let branch_type = ArrayType::new(DataType::F8E8M0FNU, shape.clone());
        let condition = context.input(ArrayType::new(DataType::Boolean, shape));
        let on_true = context.input(branch_type.clone());
        let on_false = context.input(branch_type.clone());
        let tangent_type = branch_type.tangent().unwrap();
        let tangent = context.input(tangent_type.clone());
        let outputs = SelectOperation::<ArrayType>::new()
            .jvp(
                &DifferentiationContext::fused(context.clone()),
                &EmptyRegionDriver,
                &[
                    DifferentiationDual::new_with_zero_tangent(condition).unwrap(),
                    DifferentiationDual::new(on_true, MaybeZero::Value(tangent)).unwrap(),
                    DifferentiationDual::new_with_zero_tangent(on_false).unwrap(),
                ],
            )
            .unwrap();
        assert_eq!(outputs[0].primal().r#type().as_ref(), &branch_type);
        assert_eq!(outputs[0].tangent().r#type().as_ref(), &tangent_type);
    }

    #[test]
    fn test_select_transposition() {
        let condition = Array::vector(vec![true, false]).unwrap();
        let branch_type = ArrayType::new_static(DataType::F64, [2]);
        check_operation_transposition!(
            @exact,
            operation = SelectOperation::<ArrayType>::new(),
            cases = [
                // Each linear branch receives the cotangent at the elements that the condition selected from it.
                {
                    inputs = [
                        (@known, condition.clone()),
                        (@linear(type = branch_type.clone())),
                        (@linear(type = branch_type.clone())),
                    ],
                    output_cotangents = [Array::vector(vec![5.0, 7.0]).unwrap()],
                    input_cotangents = [Array::vector(vec![5.0, 0.0]).unwrap(), Array::vector(vec![0.0, 7.0]).unwrap()],
                    pullback = indoc! {"
                        lambda %0:f64[2], %1:bool[2] .
                        let %2:f64[2] = zero [type=f64[2]]
                            %3:f64[2] = select %1 %0 %2
                            %4:f64[2] = select %1 %2 %0
                        in (%3, %4)
                    "},
                },
                // A known branch receives no cotangent, and so its cotangent `select` is not staged.
                {
                    inputs = [
                        (@known, condition.clone()),
                        (@linear(type = branch_type.clone())),
                        (@known, Array::vector(vec![1.0, 2.0]).unwrap()),
                    ],
                    output_cotangents = [Array::vector(vec![5.0, 7.0]).unwrap()],
                    input_cotangents = [Array::vector(vec![5.0, 0.0]).unwrap()],
                    pullback = indoc! {"
                        lambda %0:f64[2], %1:bool[2], %2:f64[2] .
                        let %3:f64[2] = zero [type=f64[2]]
                            %4:f64[2] = select %1 %0 %3
                        in (%4)
                    "},
                },
                // A broadcast and promoted branch receives its cotangent reduced and converted to its own type.
                {
                    inputs = [
                        (@known, condition),
                        (@linear(type = ArrayType::scalar(DataType::F32))),
                        (@linear(type = branch_type.clone())),
                    ],
                    output_cotangents = [Array::vector(vec![5.0, 7.0]).unwrap()],
                    input_cotangents = [Array::scalar(5.0f32).unwrap(), Array::vector(vec![0.0, 7.0]).unwrap()],
                },
            ],
        );

        // Structural-zero output cotangents and unrequested cotangents stage no instructions.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let condition = context.input(ArrayType::new_static(DataType::Boolean, [2]));
        let inputs = [
            PartialValue::Known(condition),
            PartialValue::Unknown(branch_type.clone()),
            PartialValue::Unknown(branch_type.clone()),
        ];
        let mut rule_context = TranspositionContext::new(context.clone());
        let accumulators = rule_context.cotangent_accumulators(&inputs, &[]).unwrap();
        SelectOperation::<ArrayType>::new()
            .transpose(
                &mut rule_context,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Zero(branch_type.cotangent().unwrap())],
                &accumulators,
            )
            .unwrap();
        let contributions = rule_context.take_cotangents(&accumulators).unwrap();
        assert_eq!(contributions.len(), 3);
        assert!(contributions.iter().all(MaybeZero::is_zero));
        let accumulators = rule_context.cotangent_accumulators(&inputs, &[true, false, false]).unwrap();
        let cotangent = context.input(branch_type.cotangent().unwrap());
        SelectOperation::<ArrayType>::new()
            .transpose(&mut rule_context, &EmptyRegionDriver, &inputs, &[MaybeZero::Value(cotangent)], &accumulators)
            .unwrap();
        let contributions = rule_context.take_cotangents(&accumulators).unwrap();
        assert_eq!(contributions.len(), 3);
        assert!(contributions.iter().all(MaybeZero::is_zero));
        assert!(context.builder().borrow().instructions().is_empty());

        // A linear condition is rejected because the Boolean condition has no tangent space.
        let inputs = [
            PartialValue::Unknown(ArrayType::new_static(DataType::Boolean, [2])),
            PartialValue::Unknown(branch_type.clone()),
            PartialValue::Unknown(branch_type.clone()),
        ];
        let accumulators = rule_context.cotangent_accumulators(&inputs, &[]).unwrap();
        let cotangent = context.input(branch_type.cotangent().unwrap());
        assert!(matches!(
            SelectOperation::<ArrayType>::new().transpose(
                &mut rule_context,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Value(cotangent)],
                &accumulators,
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation `select` does not support transposition for input pattern \
                               [condition = linear, on_true = linear, on_false = linear]",
        ));
    }

    #[test]
    fn test_select_transposition_dynamic() {
        // Dynamic zeros read the cotangent's geometry, and the scalar branch sums its selected contributions.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let size = DimensionVariable::new("size", DimensionBounds::non_negative(Some(8)).unwrap());
        let shape = Shape::new(vec![Dimension::Dynamic(size)]);
        let branch_type = ArrayType::new(DataType::F64, shape.clone());
        let cotangent = context.input(branch_type.clone());
        let condition = context.input(ArrayType::new(DataType::Boolean, shape));
        let inputs = [
            PartialValue::Known(condition),
            PartialValue::Unknown(ArrayType::scalar(DataType::F32)),
            PartialValue::Unknown(branch_type),
        ];
        let mut rule_context = TranspositionContext::new(context.clone());
        let accumulators = rule_context.cotangent_accumulators(&inputs, &[]).unwrap();
        SelectOperation::<ArrayType>::new()
            .transpose(&mut rule_context, &EmptyRegionDriver, &inputs, &[MaybeZero::Value(cotangent)], &accumulators)
            .unwrap();
        let contributions = rule_context.take_cotangents(&accumulators).unwrap();
        let output_ids = contributions[1..]
            .iter()
            .map(|contribution| {
                let MaybeZero::Value(value) = contribution else { panic!("expected a live cotangent") };
                value.atom_id().unwrap()
            })
            .collect();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(output_ids, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            program.interpret(vec![
                Array::vector(vec![5.0f64, 7.0, 11.0]).unwrap(),
                Array::vector(vec![true, false, true]).unwrap(),
            ]),
            Ok(vec![Array::scalar(16.0f32).unwrap(), Array::vector(vec![0.0f64, 7.0, 0.0]).unwrap()]),
        );
        assert_eq!(
            program
                .interpret(
                    vec![Array::vector(Vec::<f64>::new()).unwrap(), Array::vector(Vec::<bool>::new()).unwrap(),]
                ),
            Ok(vec![Array::scalar(0.0f32).unwrap(), Array::vector(Vec::<f64>::new()).unwrap()]),
        );
    }

    #[test]
    fn test_select_staging_mixed() {
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let size = DimensionVariable::new("size", DimensionBounds::non_negative(Some(8)).unwrap());
        let condition_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Dynamic(size.clone())]));
        let branch_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(size)]));
        let condition = context.input(condition_type.into());
        let on_true = context.input(branch_type.clone().into());
        let on_false = context.input(ArrayType::scalar(DataType::F32).into());
        let output = Select::select(&condition, &on_true, &on_false).unwrap();
        assert_eq!(output.r#type().as_ref(), &ArrayIrType::Array(branch_type));
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder; 3],
                vec![Placeholder],
            )
            .unwrap();
        let [instruction] = program.instructions() else {
            panic!("expected one select instruction");
        };
        assert!(matches!(instruction.operation(), ArrayIrOperation::Array(ArrayOperation::Select(_))));
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![true, false]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0]).unwrap()),
                ArrayIrValue::Array(Array::scalar(0.0f32).unwrap()),
            ]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![1.0f32, 0.0]).unwrap())]),
        );
        // Runtime zero is a valid extent even though the condition's signature is dynamic.
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Array(Array::vector(Vec::<bool>::new()).unwrap()),
                ArrayIrValue::Array(Array::vector(Vec::<f32>::new()).unwrap()),
                ArrayIrValue::Array(Array::scalar(0.0f32).unwrap()),
            ]),
            Ok(vec![ArrayIrValue::Array(Array::vector(Vec::<f32>::new()).unwrap())]),
        );
    }
}
