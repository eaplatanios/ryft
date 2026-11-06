use std::fmt::Display;

use ryft_macros::capability;

use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrOperation, ArrayIrType, ArrayOperation,
    ArrayType, LogicalMesh,
};
use crate::axes::{AxisError, NamedAxes};
use crate::batching::{BatchAxis, BatchableOperation, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{
    CotangentAccumulator, DifferentiableOperation, DifferentiationContext, DifferentiationDriver, DifferentiationDual,
    DifferentiationError, DifferentiationPolicy, TransposableOperation, TranspositionContext, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::Capability;
use crate::operations::arithmetic::AddOperation;
use crate::operations::collectives::parallel_permute::{PARALLEL_PERMUTE_OPERATION_NAME, ParallelPermuteOperation};
use crate::operations::collectives::parallel_reduce::ParallelReduceOperation;
use crate::operations::collectives::parallel_vary::{
    PARALLEL_VARY_OPERATION_NAME, ParallelVary, ParallelVaryOperation,
};
use crate::operations::collectives::{
    CollectiveAxes, LinearCollectiveOperation, resolve_collective_axes, route_axis_slices,
    validate_manual_mesh_axes_input,
};
use crate::operations::constants::zero_like::ZeroLike;
use crate::operations::manipulation::concatenation::Concatenate;
use crate::operations::manipulation::slicing::Slice;
use crate::operations::manipulation::transposition::Transpose;
use crate::operations::reductions::ReductionKind;
use crate::operations::sharding::reshard::Reshard;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, ProgramError, RegionInterface, TypeError, Typed, Value,
    ValueDomainDispatch, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

// TODO(eaplatanios): Review from here onwards.

/// Canonical operation name for [`ParallelBroadcastOperation`].
pub const PARALLEL_BROADCAST_OPERATION_NAME: &str = "parallel_broadcast";

/// [`Operation`] that replicates the input array of one _source_ participant along the named axis, so that every
/// participant receives the source's input. The source is a zero-based coordinate along `axes`, in `0..axis_size`,
/// rather than a global device ID. The output type is the input type. It is equivalent to summing, across the
/// participants, the input of the source and zeros everywhere else, but it moves the source's value instead of
/// adding values.
///
/// For example, inside `shard_map` over a manual mesh axis `x` of size three, source `2` hands the input of the device
/// at mesh coordinate `x = 2` to the devices at `x = 0`, `1`, and `2`. On a multidimensional mesh, this replication is
/// repeated separately for each fixed combination of coordinates along the other axes. Inside [`batch`](crate::batch)
/// over a named axis of size three, source `2` instead gives every batch item the input of batch item two, so the
/// items `A`, `B`, and `C` become `C`, `C`, and `C`.
///
/// This is the Ryft analogue of JAX's `jax.lax.pbroadcast` with a static `source` and of StableHLO's
/// [`collective_broadcast`](https://openxla.org/stablehlo/spec#collective_broadcast). Over several manual mesh axes
/// (i.e., non-single [`CollectiveAxes`]), the source is a position along their combined axis, as for a tuple
/// `axis_name` in JAX.
///
/// A broadcast over a manual mesh axis is created by [`with_mesh`](Self::with_mesh), and
/// [`ParallelBroadcast::parallel_broadcast`] supplies the mesh automatically from the enclosing manual region. As in
/// JAX, its input must vary over the axis (refer to [`ParallelVary`]), and its output keeps the input's variation,
/// although every device along the axis then holds the same value. Pending cross-device sums over that axis are
/// rejected, and sums over unrelated manual axes are preserved. An ordinary broadcast carries no mesh and preserves the
/// input's mesh variation and pending sums, because a `batch` level whose axis name shadows a manual mesh axis may bind
/// it instead.
///
/// The collective is linear. Its tangent rides the same broadcast, and its transpose hands the sum of every
/// participant's cotangent to the source and zeros to every other participant, which is staged as a
/// [`ParallelReduceOperation`] sum followed by a [`ParallelPermuteOperation`] whose only pair sends the source's sum to
/// itself (over a manual mesh axis, with a [`ParallelVaryOperation`] in between that restores the variation of the
/// input cotangent). JAX instead selects the sum with an `axis_index` comparison, which computes the same values but
/// needs a mask and zeros of the cotangent's shape. Outside any binder, the single participant of a degenerate axis
/// keeps its value, and a larger axis has no per-item semantics.
///
/// A matching `batch` level consumes the mapped batch axis of an ordinary broadcast by replacing every batch item with
/// a copy of the source item, wherever the mapped axis sits, and passes a replicated input through unchanged, as JAX's
/// batching rule does. A broadcast over a manual mesh axis or over several axes cannot be consumed by a `batch` level.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParallelBroadcastOperation {
    /// Names of the axes referenced by this collective.
    axes: CollectiveAxes,

    /// Number of participants along the named axis, resolved from the active [`NamedAxes`] environment
    /// when the operation is staged.
    axis_size: usize,

    /// Zero-based coordinate along the named axis of the participant whose input every participant receives.
    source: usize,

    /// Refer to the documentation of [`mesh`](Self::mesh) for more information.
    mesh: Option<LogicalMesh>,
}

impl ParallelBroadcastOperation {
    /// Creates a new [`ParallelBroadcastOperation`] over the axis with the provided name and resolved axis size.
    /// Construction preserves the supplied source, while type inference validates it.
    ///
    /// # Parameters
    ///
    ///   - `axes`: Name of the axis, or names of the axes, along which the source's input is replicated.
    ///   - `axis_size`: Number of participants along `axes`, resolved when the operation is staged.
    ///   - `source`: Coordinate in `0..axis_size` along `axes` of the participant whose input every participant
    ///     receives.
    #[inline]
    pub fn new<A: Into<CollectiveAxes>>(axes: A, axis_size: usize, source: usize) -> Self {
        Self { axes: axes.into(), axis_size, source, mesh: None }
    }

    /// Returns this [`ParallelBroadcastOperation`] configured to broadcast over manual axes of `mesh`. The input must
    /// vary over [`axes`](Self::axes) on that mesh, whose size must equal [`axis_size`](Self::axis_size). Type
    /// inference validates these requirements. [`ParallelBroadcast::parallel_broadcast`] supplies the mesh
    /// automatically from the enclosing manual region.
    #[inline]
    pub fn with_mesh(mut self, mesh: LogicalMesh) -> Self {
        self.mesh = Some(mesh);
        self
    }

    /// Returns the names of the axes referenced by this collective.
    #[inline]
    pub fn axes(&self) -> &CollectiveAxes {
        &self.axes
    }

    /// Returns the number of participants along the named axis.
    #[inline]
    pub fn axis_size(&self) -> usize {
        self.axis_size
    }

    /// Returns the coordinate along the named axis of the participant whose input every participant receives.
    #[inline]
    pub fn source(&self) -> usize {
        self.source
    }

    /// Returns the logical mesh whose manual axis this [`ParallelBroadcastOperation`] broadcasts over, or [`None`] for
    /// an ordinary broadcast, whose named axis may be bound by any enclosing binder. Only a broadcast over a manual
    /// mesh axis requires its input to vary over that axis.
    #[inline]
    pub fn mesh(&self) -> Option<&LogicalMesh> {
        self.mesh.as_ref()
    }
}

impl Display for ParallelBroadcastOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ParallelBroadcastOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        PARALLEL_BROADCAST_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        // The broadcast preserves its single input's type, including dynamic dimensions.
        let input_type = self.validate_input(input_types, region_interfaces)?;
        if self.source >= self.axis_size {
            return Err(TypeError::invalid(format!(
                "`{}` source {} is out of bounds for axis size {}",
                PARALLEL_BROADCAST_OPERATION_NAME, self.source, self.axis_size,
            )));
        }

        // A broadcast over a manual mesh axis follows JAX's typing of collectives, so the input must vary over the axis
        // and the output keeps that variation. An ordinary broadcast never inspects the input's mesh, including its
        // pending sums, because a `batch` level may bind a shadowing axis name.
        if let Some(mesh) = &self.mesh {
            validate_manual_mesh_axes_input(
                PARALLEL_BROADCAST_OPERATION_NAME,
                &self.axes,
                self.axis_size,
                mesh,
                input_type,
            )?;

            // Replicating one participant's value cannot complete a pending sum over the same axes. Sums over unrelated
            // axes commute with this broadcast and retain their pending state.
            let sharding = input_type.sharding().unwrap();
            for axis_name in self.axes.names() {
                if sharding.unreduced_axes().contains(axis_name) {
                    return Err(TypeError::invalid(format!(
                        "`{PARALLEL_BROADCAST_OPERATION_NAME}` does not support unreduced inputs",
                    )));
                }

                if !sharding.varying_manual_axes().contains(axis_name) {
                    return Err(TypeError::invalid(format!(
                        "`{}` input must vary over manual axis `{}`; pass an invariant value \
                         through `{}` first, because the broadcast output keeps the input's variation",
                        PARALLEL_BROADCAST_OPERATION_NAME, axis_name, PARALLEL_VARY_OPERATION_NAME,
                    )));
                }
            }
        }

        Ok(vec![input_type.clone()])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, PARALLEL_BROADCAST_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("axes", format_args!("{:?}", self.axes))?;
            operation.field("axis_size", self.axis_size)?;
            operation.field("source", self.source)?;
            if let Some(value) = &self.mesh {
                operation.field("mesh", value)?;
            }
            Ok(())
        })
    }
}

impl LinearCollectiveOperation for ParallelBroadcastOperation {
    type Adjoint = ParallelBroadcastOperation;

    #[inline]
    fn axes(&self) -> &CollectiveAxes {
        &self.axes
    }

    #[inline]
    fn axis_size(&self) -> usize {
        self.axis_size
    }

    #[inline]
    fn mesh(&self) -> Option<&LogicalMesh> {
        self.mesh.as_ref()
    }

    fn adjoint(&self, _input_type: &ArrayType) -> Result<ParallelBroadcastOperation, ProgramError> {
        // The transpose hands the sum of every participant's cotangent to the source, which is a composition of
        // collectives rather than one adjoint collective, so the `TransposableOperation` implementation stages it
        // itself and never calls this function.
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "`{}` has no single adjoint collective, because its transpose stages a sum followed by a \
                 source-selecting `{}`",
                PARALLEL_BROADCAST_OPERATION_NAME, PARALLEL_PERMUTE_OPERATION_NAME,
            ),
        })
    }

    #[inline]
    fn adapt_to_batch_axis(&self, input_batch_axis: usize) -> (Self, usize) {
        // A broadcast preserves the shape of its input, so it applies to the packed values of a mapped input unchanged
        // and leaves the mapped axis where it is.
        (self.clone(), input_batch_axis)
    }
}

impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for ParallelBroadcastOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[<C as Domain>::Value],
    ) -> Result<Vec<<C as Domain>::Value>, ProgramError> {
        // Eager binding does not infer output types, so interpretation validates the shared input contract and the
        // source before applying the degenerate-axis rule, under which the only participant is its own source.
        check_count!("input", inputs, 1, ProgramError);
        self.validate_degenerate_interpretation()?;
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.infer_output_types(&input_types, &[])?;
        Ok(vec![inputs[0].clone()])
    }
}

impl<C: Context<Type = ArrayType, Operation: From<ParallelBroadcastOperation>>> PartiallyEvaluatableOperation<C>
    for ParallelBroadcastOperation
{
}

impl<C: Context<Type = ArrayType, Operation: From<ParallelBroadcastOperation>>, P: ArrayExtentBatchingPolicy<C>>
    BatchableOperation<C, ArrayBatchingPolicy<P>> for ParallelBroadcastOperation
where
    C::Value: ZeroLike + Concatenate + Slice + Transpose + Reshard,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        // A matching `batch` level consumes the mapped batch axis by giving every position along it a copy of the
        // source item, as JAX's `v.take([source] * axis_size, d)` does. A non-matching level forwards the collective
        // untouched to the parent context.
        if !context.axis_name().is_some_and(|axis_name| self.axes.contains(axis_name)) {
            return context.forward_collective(self, inputs);
        }

        self.reject_mesh_form()?;

        let [input] = inputs else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() }.into());
        };

        // A batching rule runs before any context infers the operation's output type, so the operation is validated
        // here, against the physical (i.e., padded) item type, before its source indexes the batch axis.
        self.infer_output_types(&[input.value().r#type().unbatched(input.batch_axis())?], &[])?;

        let batch_size = P::axis_size(context)?;
        if batch_size != self.axis_size {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{}` over axis `{}` resolved axis size {} but the mapped batch axis has size {}",
                    PARALLEL_BROADCAST_OPERATION_NAME, self.axes, self.axis_size, batch_size,
                ),
            });
        }

        // A replicated input holds the same value for every batch item, which is already the source's value.
        let Some(batch_axis) = input.batch_axis_position() else {
            return Ok(vec![input.clone()]);
        };

        // Every position along the mapped axis, which stays where it is, receives the slice of the source item.
        let sources = vec![Some(self.source); batch_size];
        let broadcast =
            route_axis_slices(PARALLEL_BROADCAST_OPERATION_NAME, input.value(), batch_axis, sources.as_slice())?;
        Ok(vec![ArrayBatch::new(broadcast, BatchAxis::from_position(batch_axis))?])
    }
}

impl<C: Context<Type = ArrayType, Operation: From<ParallelBroadcastOperation>>> DifferentiableOperation<C>
    for ParallelBroadcastOperation
{
    #[inline]
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        self.linear_collective_jvp(context, inputs)
    }
}

impl<
    V: Value<Type = ArrayType>,
    O: Operation<Type = ArrayType>
        + From<AddOperation<ArrayType>>
        + From<ParallelReduceOperation>
        + From<ParallelVaryOperation>
        + From<ParallelPermuteOperation>,
> TransposableOperation<V, O> for ParallelBroadcastOperation
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        // Binds one single-output operation of the transpose on `input` and returns its output.
        fn bind_single_output<V: Value<Type = ArrayType>, O: Operation<Type = ArrayType>>(
            context: &TranspositionContext<V, O>,
            operation: O,
            input: Tracer<TracingContext<V, O>>,
        ) -> Result<Tracer<TracingContext<V, O>>, ProgramError>
        where
            TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
        {
            let mut outputs = context.bind(operation, Vec::new(), &[input])?;
            check_count!("output", outputs, 1, ProgramError);
            Ok(outputs.remove(0))
        }

        // Every participant's output is the source's input, so the input cotangent of the source is the sum of every
        // participant's output cotangent, and every other participant's input cotangent is zero. A known input and a
        // structural zero output cotangent contribute nothing.
        check_count!("input", inputs, 1, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        check_count!("accumulator", accumulators, 1, DifferentiationError);
        let MaybeZero::Value(cotangent) = &outputs[0] else {
            return Ok(());
        };

        if inputs[0].is_known() {
            return Ok(());
        }

        // The single participant of a degenerate axis is its own source, so its broadcast is the identity, and so is
        // its transpose, which then remains interpretable without an enclosing binder.
        let mut contribution = cotangent.clone();
        if self.axis_size > 1 {
            // The sum runs one axis at a time, because a reduction names one axis and the sum over the combined axis
            // is the sum over each of its axes. Over manual mesh axes, the sum is invariant over them, while the input
            // cotangent varies over them as the input does, so the sum is made varying again with real
            // `parallel_vary` transitions, whose transposes are the sums that a nested transposition needs.
            for axis_name in self.axes.names() {
                let mut operation = ParallelReduceOperation::new(ReductionKind::Sum, axis_name.clone());
                if let Some(mesh) = &self.mesh {
                    operation = operation.with_mesh(mesh.clone());
                }
                contribution = bind_single_output(context, operation.into(), contribution)?;
            }
            if self.mesh.is_some() {
                for axis_name in self.axes.names() {
                    let operation = ParallelVaryOperation::new(axis_name.clone());
                    contribution = bind_single_output(context, operation.into(), contribution)?;
                }
            }

            // A permutation whose only pair sends the source's sum to itself keeps that sum at the source and gives
            // every other participant zeros. Unlike a mask of the cotangent's shape, it needs no zeros of the
            // cotangent's shape to select from, and it preserves every placement and variation of its input.
            let mut operation =
                ParallelPermuteOperation::new(self.axes.clone(), self.axis_size, vec![(self.source, self.source)]);
            if let Some(mesh) = &self.mesh {
                operation = operation.with_mesh(mesh.clone());
            }
            contribution = bind_single_output(context, operation.into(), contribution)?;
        }
        accumulators[0].accumulate(context, MaybeZero::Value(contribution))?;
        Ok(())
    }
}

impl<A: Value<Type = ArrayType>> From<ParallelBroadcastOperation> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: ParallelBroadcastOperation) -> Self {
        Self::Array(ArrayOperation::ParallelBroadcast(operation))
    }
}

/// Represents the ability to replicate the value of one participant of a named axis to every participant. Refer to the
/// documentation of [`ParallelBroadcastOperation`] for the semantics of this operation and its transformation rules.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and array IR values implement it for [`ArrayIrType`].
///
/// # Example
///
/// Every batch item receives the row of batch item two:
///
/// ```rust
/// # use ryft_core::{
/// #     Array, ArrayBatchingPolicy, ArrayOperation, BatchAxis, BatchAxisSpecification, BatchingTracer, EagerContext,
/// #     ParallelBroadcast, batch,
/// # };
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let rows = Array::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
/// let broadcast = batch(
///     |row: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>| {
///         row.parallel_broadcast("rows", 2)
///     },
///     rows,
///     BatchAxis::new(0),
///     BatchAxis::new(0),
///     BatchAxisSpecification::named("rows"),
/// )?;
/// assert_eq!(broadcast, Array::matrix(3, 2, vec![5.0, 6.0, 5.0, 6.0, 5.0, 6.0])?);
/// # Ok(())
/// # }
/// ```
#[capability]
pub trait ParallelBroadcast<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Returns the value of participant `source` along the named axes `axes`, which every participant receives. Over a
    /// manual mesh axis, an input that does not vary over the axis is first made varying through [`ParallelVary`],
    /// because the broadcast output keeps its input's variation. JAX's `pbroadcast` rejects such an input instead, so
    /// this is a deliberate extension that mirrors how [`ParallelPermute`](crate::ParallelPermute) stages it.
    ///
    /// # Parameters
    ///
    ///   - `axes`: Name of an axis bound by an enclosing `batch` level or manual region, or names of several
    ///     manual mesh axes along whose combined axis the value is broadcast.
    ///   - `source`: Position along the named axis of the participant whose value every participant receives.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError::Axis`] error wrapping [`AxisError::UnboundAxisName`](crate::axes::AxisError) when no
    /// enclosing binder binds `axes`, and a [`ProgramError`] if `source` is outside the axis or if this value carries
    /// a pending sum over the participating manual mesh axis.
    fn parallel_broadcast<Axes: Into<CollectiveAxes>>(&self, axes: Axes, source: usize) -> Result<Self, ProgramError>;
}

impl ParallelBroadcast<ArrayType> for Array {
    // A concrete `Array` never executes inside an axis binder, because the values under a `batch` level or inside
    // a manual region are tracers, so every axis name is unbound for it.

    #[inline]
    fn parallel_broadcast<Axes: Into<CollectiveAxes>>(&self, axes: Axes, _source: usize) -> Result<Self, ProgramError> {
        let axes = axes.into();
        axes.validate(PARALLEL_BROADCAST_OPERATION_NAME)?;
        Err(AxisError::UnboundAxisName { name: axes.names()[0].clone() }.into())
    }
}

impl<
    V: Value<
            Type = ArrayType,
            Dispatch = ValueDomainDispatch,
            Domain: Context<Operation: From<ParallelBroadcastOperation>> + NamedAxes,
        > + ParallelVary,
> ParallelBroadcast<ArrayType> for V
{
    fn parallel_broadcast<Axes: Into<CollectiveAxes>>(&self, axes: Axes, source: usize) -> Result<Self, ProgramError> {
        // Any context-carrying value broadcasts by resolving the axis size from the active `NamedAxes` environment and
        // binding a `ParallelBroadcastOperation` through its own context. Over manual mesh axes, the operation records
        // the mesh, and an input that is still invariant over an axis is first made varying, as `parallel_permute`
        // does. JAX's `pbroadcast` rejects such an input instead of inserting this transition.
        let axes = axes.into();
        let context = self.domain();
        let (axis_size, mesh) = resolve_collective_axes(&context, PARALLEL_BROADCAST_OPERATION_NAME, &axes)?;
        let mut operation = ParallelBroadcastOperation::new(axes.clone(), axis_size, source);
        let mut input = self.clone();
        if let Some(mesh) = mesh {
            for axis_name in axes.names() {
                if input.r#type().unreduced_axes().contains(axis_name) {
                    return Err(TypeError::invalid(format!(
                        "`{PARALLEL_BROADCAST_OPERATION_NAME}` does not support unreduced inputs",
                    ))
                    .into());
                }
                if !input.r#type().sharding().is_some_and(|sharding| sharding.varying_manual_axes().contains(axis_name))
                {
                    input = input.parallel_vary(axis_name)?;
                }
            }
            operation = operation.with_mesh(mesh);
        }
        let mut outputs = context.bind(operation, Vec::new(), &[input])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<V: Value<Type = ArrayIrType> + ValueProjection<ArrayType>> ParallelBroadcast<ArrayIrType> for V
where
    V::Projected: ParallelBroadcast<ArrayType>,
{
    #[inline]
    fn parallel_broadcast<Axes: Into<CollectiveAxes>>(&self, axes: Axes, source: usize) -> Result<Self, ProgramError> {
        // An array IR value broadcasts through its array view, which owns named-axis resolution and manual-axis
        // variation, so that every broadcast shares one staging path.
        Ok(V::from_projected(
            ValueProjection::<ArrayType>::into_projected(self.clone())?.parallel_broadcast(axes, source)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrValue, DataType, Dimension, DimensionBounds, DimensionValue,
        DimensionVariable, Layout, Memory, MeshAxis, MeshAxisType, Shape, Sharding, ShardingDimension, StridedLayout,
    };
    use crate::axes::NamedAxis;
    use crate::batching::{BatchAxisSpecification, BatchingTracer, batch};
    use crate::contexts::EagerContext;
    use crate::differentiation::differentiate_at;
    use crate::macros::{check_gradient, check_operation_partial_evaluation, check_operation_type_inference};
    use crate::operations::collectives::tests::{batch_collective, collective_program};
    use crate::operations::reductions::Reduce;
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationContext, PartialEvaluationOutput, PartialEvaluationValue};
    use crate::programs::{EmptyRegionDriver, Program};
    use crate::tests::hash_of;

    use super::*;

    /// Creates a manual mesh whose axis `"x"` has two participants and whose axis `"y"` has one.
    fn manual_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 1, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap()
    }

    /// Binds `"x"` as a manual axis of [`manual_mesh`] for [`TracingContext::trace_with_named_axes`], with or without
    /// manual variation checks.
    fn manual_axes(checks_manual_variation: bool) -> Vec<(String, NamedAxis)> {
        vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: manual_mesh(), checks_manual_variation })]
    }

    #[test]
    fn test_parallel_broadcast() {
        let operation = ParallelBroadcastOperation::new("x".to_string(), 3, 2);
        assert_eq!(operation.name(), PARALLEL_BROADCAST_OPERATION_NAME);
        assert_eq!(operation.axes().single(), Some("x"));
        assert_eq!(operation.axis_size(), 3);
        assert_eq!(operation.source(), 2);
        assert_eq!(operation.mesh(), None);
        assert_eq!(operation.to_string(), "parallel_broadcast [axes=\"x\", axis_size=3, source=2]");

        // Several axes render as a tuple in the order that linearizes their combined axis.
        let operation = ParallelBroadcastOperation::new(["y", "x"], 4, 3);
        assert_eq!(operation.axes().names(), &["y".to_string(), "x".to_string()]);
        assert_eq!(operation.to_string(), "parallel_broadcast [axes=(\"y\", \"x\"), axis_size=4, source=3]");
    }

    #[test]
    fn test_parallel_broadcast_with_mesh() {
        // A broadcast over a manual mesh axis records and renders its mesh.
        let operation = ParallelBroadcastOperation::new("x".to_string(), 2, 1).with_mesh(manual_mesh());
        assert_eq!(operation.mesh(), Some(&manual_mesh()));
        assert_eq!(
            operation.to_string(),
            "parallel_broadcast [axes=\"x\", axis_size=2, source=1, mesh=['x'=2:manual, 'y'=1:manual]]",
        );
    }

    #[test]
    fn test_parallel_broadcast_equality_and_hashing() {
        // Independently constructed operations are equal, and hash alike, when their axes, sources, and meshes agree.
        let ordinary = ParallelBroadcastOperation::new("x".to_string(), 3, 2);
        let same_ordinary = ParallelBroadcastOperation::new("x".to_string(), 3, 2);
        let meshed = ParallelBroadcastOperation::new("x".to_string(), 2, 1).with_mesh(manual_mesh());
        let same_meshed = ParallelBroadcastOperation::new("x".to_string(), 2, 1).with_mesh(manual_mesh());
        assert_eq!(ordinary, same_ordinary);
        assert_eq!(meshed, same_meshed);
        assert_eq!(hash_of(&ordinary), hash_of(&same_ordinary));
        assert_eq!(hash_of(&meshed), hash_of(&same_meshed));

        // Every attribute participates in the identity.
        let unmeshed = ParallelBroadcastOperation::new("x".to_string(), 2, 1);
        assert_ne!(ordinary, ParallelBroadcastOperation::new("x".to_string(), 3, 1));
        assert_ne!(ordinary, ParallelBroadcastOperation::new("x".to_string(), 4, 2));
        assert_ne!(ordinary, ParallelBroadcastOperation::new("y".to_string(), 3, 2));
        assert_ne!(meshed, unmeshed);

        // Map lookups find independently constructed equal operations and miss distinct ones.
        let operations = HashMap::from([(ordinary, 0), (meshed, 1)]);
        assert_eq!(operations.get(&same_ordinary), Some(&0));
        assert_eq!(operations.get(&same_meshed), Some(&1));
        assert_eq!(operations.get(&unmeshed), None);
    }

    #[test]
    fn test_parallel_broadcast_type_inference() {
        // A broadcast preserves its input type, including its layout, memory, and dynamic dimensions. An ordinary
        // broadcast also accepts an input that is invariant over a manual mesh axis with the same name, because a
        // `batch` level that shadows that mesh axis may bind it.
        let mesh = manual_mesh();
        let sharding = Sharding::replicated(mesh.clone(), 1);
        let vector = ArrayType::new_static(DataType::F32, [3]);
        let invariant = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding.clone()).unwrap();
        let varying_sharding = sharding.clone().with_varying_manual_axes(["x"]).unwrap();
        let varying = ArrayType::new_static(DataType::F32, [4]).with_sharding(varying_sharding.clone()).unwrap();
        let placed = varying
            .clone()
            .with_layout(Layout::Strided(StridedLayout::new(vec![4])))
            .with_memory(Memory::Host { pinned: true });
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(8)).unwrap());
        let bounded = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(length)]));
        let bounded_varying = bounded.clone().with_sharding(varying_sharding).unwrap();
        check_operation_type_inference!(
            operation = ParallelBroadcastOperation::new("x".to_string(), 2, 1),
            cases = [
                { input_types = [vector.clone()], output_types = [vector.clone()] },
                { input_types = [placed.clone()], output_types = [placed.clone()] },
                { input_types = [invariant.clone()], output_types = [invariant.clone()] },
                { input_types = [bounded.clone()], output_types = [bounded] },
            ],
        );

        // The source must be a participant of a non-empty axis, and at least one axis must be named.
        check_operation_type_inference!(
            operation = ParallelBroadcastOperation::new("x".to_string(), 2, 2),
            cases = [{
                input_types = [vector.clone()],
                error = "`parallel_broadcast` source 2 is out of bounds for axis size 2",
            }],
        );
        check_operation_type_inference!(
            operation = ParallelBroadcastOperation::new("x".to_string(), 0, 0),
            cases = [{
                input_types = [vector.clone()],
                error = "`parallel_broadcast` axis size must be greater than zero",
            }],
        );
        check_operation_type_inference!(
            operation = ParallelBroadcastOperation::new(Vec::<String>::new(), 1, 0),
            cases = [{
                input_types = [vector.clone()],
                error = "`parallel_broadcast` requires at least one axis name",
            }],
        );

        // A broadcast over a manual mesh axis preserves a varying input type, including its dynamic dimensions, as JAX
        // types `pbroadcast`, and requires an input that varies over the axis of the operation's mesh. Replicating one
        // participant cannot complete a pending sum over its own axis, so it rejects one, while an ordinary broadcast
        // preserves every pending sum.
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let other_varying = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(other_mesh, 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let unreduced = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(sharding.with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        check_operation_type_inference!(
            operation = ParallelBroadcastOperation::new("x".to_string(), 2, 1),
            cases = [{ input_types = [unreduced.clone()], output_types = [unreduced.clone()] }],
        );
        check_operation_type_inference!(
            operation = ParallelBroadcastOperation::new("x".to_string(), 2, 1).with_mesh(mesh.clone()),
            cases = [
                { input_types = [varying.clone()], output_types = [varying.clone()] },
                { input_types = [bounded_varying.clone()], output_types = [bounded_varying] },
                {
                    input_types = [invariant],
                    error = "`parallel_broadcast` input must vary over manual axis `x`; pass an invariant value \
                             through `parallel_vary` first, because the broadcast output keeps the input's variation",
                },
                {
                    input_types = [vector],
                    error = "`parallel_broadcast` input must carry a mesh containing manual axis `x`",
                },
                {
                    input_types = [other_varying],
                    error = "`parallel_broadcast` input mesh does not match the operation mesh",
                },
                { input_types = [unreduced], error = "`parallel_broadcast` does not support unreduced inputs" },
            ],
        );

        // The mesh axis must be manual, and its size must equal the axis size of the operation.
        let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        check_operation_type_inference!(
            operation = ParallelBroadcastOperation::new("x".to_string(), 2, 1).with_mesh(explicit_mesh),
            cases = [{ input_types = [varying.clone()], error = "`parallel_broadcast` mesh axis `x` must be manual" }],
        );
        check_operation_type_inference!(
            operation = ParallelBroadcastOperation::new("x".to_string(), 3, 1).with_mesh(mesh),
            cases = [{
                input_types = [varying],
                error = "`parallel_broadcast` axis size 3 does not match the size of manual mesh axis `x`",
            }],
        );
    }

    #[test]
    fn test_parallel_broadcast_type_inference_several_axes() {
        // Over several manual axes, the input must vary over each of them, and a pending sum over an unrelated axis
        // commutes with the broadcast and is preserved.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["z"]).unwrap();
        let varying = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(sharding.clone().with_varying_manual_axes(["x", "y"]).unwrap())
            .unwrap();
        let varying_over_x = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(sharding.with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        check_operation_type_inference!(
            operation = ParallelBroadcastOperation::new(["y", "x"], 4, 3).with_mesh(mesh.clone()),
            cases = [
                { input_types = [varying.clone()], output_types = [varying.clone()] },
                {
                    input_types = [varying_over_x],
                    error = "`parallel_broadcast` input must vary over manual axis `y`; pass an invariant value \
                             through `parallel_vary` first, because the broadcast output keeps the input's variation",
                },
            ],
        );
        check_operation_type_inference!(
            operation = ParallelBroadcastOperation::new(["x", "y"], 2, 1).with_mesh(mesh.clone()),
            cases = [{
                input_types = [varying.clone()],
                error = "`parallel_broadcast` axis size 2 does not match the size 4 of manual mesh axes (x, y)",
            }],
        );
        check_operation_type_inference!(
            operation = ParallelBroadcastOperation::new(["x", "x"], 4, 1).with_mesh(mesh),
            cases = [{ input_types = [varying], error = "`parallel_broadcast` axis names repeat `x`" }],
        );
    }

    #[test]
    fn test_parallel_broadcast_interpretation() {
        // Outside any binder, the single participant of a degenerate axis is its own source and keeps its value, while
        // a larger axis has no per-item semantics.
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
        assert_eq!(
            ParallelBroadcastOperation::new("x".to_string(), 1, 0).interpret(
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&input),
            ),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            ParallelBroadcastOperation::new("x".to_string(), 2, 1).interpret(
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&input),
            ),
            Err(ProgramError::UnsupportedOperation {
                message: "cannot interpret `parallel_broadcast` over axis `x` of size 2 without an enclosing binder"
                    .to_string(),
            }),
        );

        // Even a degenerate broadcast validates its source before interpreting it as the identity.
        assert_eq!(
            context.bind(ParallelBroadcastOperation::new("x".to_string(), 1, 1), Vec::new(), &[input]),
            Err(ProgramError::Type(TypeError::invalid(
                "`parallel_broadcast` source 1 is out of bounds for axis size 1",
            ))),
        );
    }

    #[test]
    fn test_parallel_broadcast_partial_evaluation() {
        // A known input over a degenerate axis folds through interpretation into the input itself.
        check_operation_partial_evaluation!(
            operation = ParallelBroadcastOperation::new("x".to_string(), 1, 0),
            inputs = [Array::vector(vec![1.0f32, 2.0]).unwrap()],
            expected = Array::vector(vec![1.0f32, 2.0]).unwrap(),
        );

        // A known input over a larger axis under an eager parent residualizes the operation, which has no per-item
        // value, so the residual program is the source program itself.
        let operation = ParallelBroadcastOperation::new("x".to_string(), 2, 1);
        let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let program = collective_program(operation.clone(), ArrayType::new_static(DataType::F32, [2]));
        let evaluation = program.partially_evaluate(&[PartialValue::Known(input)]).unwrap();
        assert_eq!(evaluation.program().to_string(), program.to_string());
        assert_eq!(evaluation.outputs(), &[PartialEvaluationOutput::Unknown(0)]);

        // A known input under a staging parent stays known, because the operation is staged into the parent trace.
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = PartialEvaluationValue::known(trace.input(ArrayType::new_static(DataType::F32, [2])));
        let outputs = operation
            .partially_evaluate(&PartialEvaluationContext::new(trace.clone()), &EmptyRegionDriver, &[input])
            .unwrap();
        assert!(outputs[0].is_known());
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(
                vec![outputs[0].as_known().unwrap().atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2] = parallel_broadcast [axes=\"x\", axis_size=2, source=1] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_parallel_broadcast_batching() {
        // A level that binds the axis gives every item a copy of the source item, wherever the mapped axis sits, as
        // JAX's `vmap` of the documented equivalent of `pbroadcast` does, because JAX's own batching rule fails at
        // its reference commit (refer to `.tasks/shard_map_jax_parity_gaps/m8_oracles.py`).
        assert_eq!(
            batch_collective(
                &ParallelBroadcastOperation::new("x".to_string(), 3, 2),
                "x",
                3,
                ArrayBatch::new(Array::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(), BatchAxis::new(0))
                    .unwrap(),
            ),
            Ok(vec![
                ArrayBatch::new(Array::matrix(3, 2, vec![5.0, 6.0, 5.0, 6.0, 5.0, 6.0]).unwrap(), BatchAxis::new(0))
                    .unwrap(),
            ]),
        );
        assert_eq!(
            batch_collective(
                &ParallelBroadcastOperation::new("x".to_string(), 3, 0),
                "x",
                3,
                ArrayBatch::new(Array::matrix(2, 3, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]).unwrap(), BatchAxis::new(1))
                    .unwrap(),
            ),
            Ok(vec![
                ArrayBatch::new(Array::matrix(2, 3, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]).unwrap(), BatchAxis::new(1))
                    .unwrap(),
            ]),
        );

        // A replicated input already holds the source's value for every item, so it passes through unchanged.
        let swap = ParallelBroadcastOperation::new("x".to_string(), 2, 1);
        let input = ArrayBatch::replicated(Array::vector(vec![7.0, 8.0]).unwrap());
        assert_eq!(batch_collective(&swap, "x", 2, input.clone()), Ok(vec![input]));

        // The operation's axis size must equal the size of the level that binds its axis.
        assert_eq!(
            batch_collective(
                &swap,
                "x",
                3,
                ArrayBatch::new(Array::vector(vec![1.0, 2.0, 3.0]).unwrap(), BatchAxis::new(0)).unwrap(),
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_broadcast` over axis `x` resolved axis size 2 but the mapped batch axis has size 3"
                    .to_string(),
            }),
        );

        // A broadcast over a manual mesh axis, or over several axes, cannot be consumed by a level that binds one of
        // its axis names, as JAX's batching rule rejects several axes.
        let vector = ArrayBatch::new(Array::vector(vec![1.0, 2.0]).unwrap(), BatchAxis::new(0)).unwrap();
        assert_eq!(
            batch_collective(&swap.clone().with_mesh(manual_mesh()), "x", 2, vector.clone()),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_broadcast` over a manual mesh axis cannot bind a named batch axis".to_string(),
            }),
        );
        assert_eq!(
            batch_collective(&ParallelBroadcastOperation::new(["x", "y"], 4, 1), "x", 2, vector),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_broadcast` over several axes cannot bind a named batch axis".to_string(),
            }),
        );

        // Copying the source item slices it by its static extents, so a level that binds the axis rejects an input
        // whose items have a dynamic dimension.
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(length)]));
        let input = ArrayBatch::new(trace.input(input_type), BatchAxis::new(0)).unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(trace, 2).with_axis_name("x".to_string());
        assert!(matches!(
            swap.batch(&context, &EmptyRegionDriver, &[input]),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "`parallel_broadcast` batching requires statically shaped inputs",
        ));

        // A level that binds another axis forwards the broadcast to its parent and preserves its mapped axis.
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input =
            ArrayBatch::new(trace.input(ArrayType::new_static(DataType::F32, [2, 3])), BatchAxis::new(1)).unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(trace.clone(), 3).with_axis_name("y".to_string());
        let outputs = swap.batch(&context, &EmptyRegionDriver, std::slice::from_ref(&input)).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(1));
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(
                vec![outputs[0].value().atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 3] .
                let %1:f32[2, 3] = parallel_broadcast [axes=\"x\", axis_size=2, source=1] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_parallel_broadcast_batching_sharding() {
        // Copy the source item of an explicitly sharded mapped axis that is not the leading axis through replicated
        // singleton slices, restoring the original placement on the complete output.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding =
            Sharding::new(mesh, vec![ShardingDimension::Replicated, ShardingDimension::sharded(["devices"])]).unwrap();
        let batch = |values: Vec<f64>| {
            ArrayBatch::new(Array::matrix(3, 2, values).unwrap().reshard(&sharding).unwrap(), BatchAxis::new(1))
                .unwrap()
        };
        assert_eq!(
            batch_collective(
                &ParallelBroadcastOperation::new("x".to_string(), 2, 1),
                "x",
                2,
                batch(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ),
            Ok(vec![batch(vec![2.0, 2.0, 4.0, 4.0, 6.0, 6.0])]),
        );
    }

    #[test]
    fn test_parallel_broadcast_batching_shadows_manual_axis() {
        // The inner named batch binds `x`, so an invariant value on an enclosing manual mesh axis with the same name
        // needs no mesh-axis variation to broadcast its local batch items, and the copies keep its placement.
        let input_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(manual_mesh(), 1))
            .unwrap();
        let (output_type, program): (ArrayType, Program<Array, ArrayOperation<Array>, Array, Array>) =
            TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                |input| {
                    let context = BatchingContext::<_, ArrayBatchingPolicy>::new(input.domain(), 2)
                        .with_axis_name("x".to_string());
                    let input = BatchingTracer::new(context, ArrayBatch::new(input, BatchAxis::new(0))?);
                    Ok(input.parallel_broadcast("x", 1)?.into_batch().into_value())
                },
                input_type.clone(),
                manual_axes(true),
            )
            .unwrap();
        assert_eq!(output_type, input_type);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}]}] .
                let %1:f32[1][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}]}] = \
                        slice [start_indices=[1], limits=[2]] %0
                    %2:f32[2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}]}] = concatenate [axis=0] %1 %1
                in (%2)"
            },
        );
    }

    #[test]
    fn test_parallel_broadcast_differentiation() {
        // The collective is linear, so the tangent rides the same broadcast as the primal.
        let operation = ParallelBroadcastOperation::new("x".to_string(), 2, 1);
        assert_eq!(
            collective_program(operation.clone(), ArrayType::new_static(DataType::F32, [2]))
                .jvp()
                .unwrap()
                .to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[2] .
                let %2:f32[2] = parallel_broadcast [axes=\"x\", axis_size=2, source=1] %0
                    %3:f32[2] = parallel_broadcast [axes=\"x\", axis_size=2, source=1] %1
                in (%2, %3)"
            },
        );

        // A structural zero tangent stays a structural zero, and only the primal broadcast is staged.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = context.input(ArrayType::new_static(DataType::F32, [2]));
        let outputs = operation
            .jvp(
                &DifferentiationContext::fused(context.clone()),
                &EmptyRegionDriver,
                &[DifferentiationDual::new_with_zero_tangent(input).unwrap()],
            )
            .unwrap();
        assert!(outputs[0].tangent().is_zero());
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(
                vec![outputs[0].primal().atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2] = parallel_broadcast [axes=\"x\", axis_size=2, source=1] %0
                in (%1)"
            },
        );

        // Reverse mode through a level that binds the axis hands the cotangents of all three items to the source item,
        // as JAX's gradient of the documented equivalent of `pbroadcast` does.
        assert_eq!(
            differentiate_at(Array::vector(vec![1.0, 2.0, 3.0]).unwrap()).value_and_gradient(|inputs| {
                let broadcast = batch(
                    |item| item.parallel_broadcast("x", 1),
                    inputs,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("x"),
                )?;
                Ok(broadcast.reduce(&[0], ReductionKind::Sum)?)
            }),
            Ok((Array::scalar(6.0).unwrap(), Array::vector(vec![0.0, 3.0, 0.0]).unwrap())),
        );

        // Unequal output weights check against finite differences that only the source item has a derivative.
        check_gradient!(
            |inputs| {
                let broadcast = batch(
                    |item| item.parallel_broadcast("x", 2),
                    inputs,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("x"),
                )?;
                let first = broadcast.slice(&[0], &[1], &[1])?;
                let first = first.clone() + first;
                let last = broadcast.slice(&[2], &[3], &[1])?;
                let weighted = first + last.clone() + last.clone() + last;
                weighted.reduce(&[0], ReductionKind::Sum)
            },
            at = Array::vector(vec![1.0, 2.0, 3.0]).unwrap(),
            step = 1e-6,
            tolerance = 1e-6,
        );
    }

    #[test]
    fn test_parallel_broadcast_transposition() {
        // The transpose sums the cotangents of every participant and keeps the sum at the source through a permutation
        // whose only pair sends it to itself, which gives every other participant zeros.
        let program = collective_program(
            ParallelBroadcastOperation::new("x".to_string(), 3, 1),
            ArrayType::new_static(DataType::F32, [2]),
        );
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2] = parallel_reduce [kind=sum, axis_name=\"x\"] %0
                    %2:f32[2] = parallel_permute [axes=\"x\", axis_size=3, source_target_pairs=[(1, 1)]] %1
                in (%2)"
            },
        );

        // Transposing again yields the broadcast's semantics in composed form, which zeros every participant but the
        // source and then sums, so every participant receives the source's value.
        assert_eq!(
            transposed.transpose_with_respect_to(&[0], &[]).unwrap().to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2] = parallel_permute [axes=\"x\", axis_size=3, source_target_pairs=[(1, 1)]] %0
                    %2:f32[2] = parallel_reduce [kind=sum, axis_name=\"x\"] %1
                in (%2)"
            },
        );

        // The single participant of a degenerate axis is its own source, so the transpose is the identity.
        let program = collective_program(
            ParallelBroadcastOperation::new("x".to_string(), 1, 0),
            ArrayType::new_static(DataType::F32, [2]),
        );
        assert_eq!(
            program.transpose_with_respect_to(&[0], &[]).unwrap().to_string(),
            indoc! {"
                lambda %0:f32[2] .
                in (%0)"
            },
        );

        // Over a manual mesh axis, the sum is invariant over the axis, and a variation transition restores the
        // variation of the input cotangent before the source-selecting permutation over the same mesh.
        let varying = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(manual_mesh(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let program = collective_program(
            ParallelBroadcastOperation::new("x".to_string(), 2, 1).with_mesh(manual_mesh()),
            varying,
        );
        assert_eq!(
            program.transpose_with_respect_to(&[0], &[]).unwrap().to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], varying_manual={'x'}}] .
                let %1:f32[2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}]}] = \
                        parallel_reduce [kind=sum, axis_name=\"x\", mesh=['x'=2:manual, 'y'=1:manual]] %0
                    %2:f32[2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %1
                    %3:f32[2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_permute [
                        axes=\"x\",
                        axis_size=2,
                        source_target_pairs=[(1, 1)],
                        mesh=['x'=2:manual, 'y'=1:manual],
                    ] %2
                in (%3)"
            },
        );

        // Over several manual axes, the sum runs one axis at a time, and the permutation selects the source along
        // their combined axis.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let varying = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["x", "y"]).unwrap())
            .unwrap();
        let program = collective_program(ParallelBroadcastOperation::new(["y", "x"], 4, 2).with_mesh(mesh), varying);
        assert_eq!(
            program.transpose_with_respect_to(&[0], &[]).unwrap().to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [], varying_manual={'x', 'y'}}] .
                let %1:f32[][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [], varying_manual={'x'}}] = \
                        parallel_reduce [kind=sum, axis_name=\"y\", mesh=['x'=2:manual, 'y'=2:manual]] %0
                    %2:f32[][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, []}] = \
                        parallel_reduce [kind=sum, axis_name=\"x\", mesh=['x'=2:manual, 'y'=2:manual]] %1
                    %3:f32[][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [], varying_manual={'y'}}] = \
                        parallel_vary [axis_name=\"y\"] %2
                    %4:f32[][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [], varying_manual={'x', 'y'}}] = \
                        parallel_vary [axis_name=\"x\"] %3
                    %5:f32[][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [], varying_manual={'x', 'y'}}] = \
                        parallel_permute [
                        axes=(\"y\", \"x\"),
                        axis_size=4,
                        source_target_pairs=[(2, 2)],
                        mesh=['x'=2:manual, 'y'=2:manual],
                    ] %4
                in (%5)"
            },
        );
    }

    #[test]
    fn test_parallel_broadcast_transposition_preserves_unrelated_manual_axes() {
        // An input that also varies over the unrelated manual axis `y` and carries a pending sum over the unrelated
        // manual axis `z` has a cotangent that varies over `y` and is reduced over `z`, which keeps both through the
        // sum, variation, and permutation over `x`. Transposing again yields the broadcast's semantics in composed form
        // over the same mesh and restores the input's pending sum.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let input_type = ArrayType::scalar(DataType::F32)
            .with_sharding(
                Sharding::replicated(mesh.clone(), 0)
                    .with_unreduced_axes(["z"])
                    .unwrap()
                    .with_varying_manual_axes(["x", "y"])
                    .unwrap(),
            )
            .unwrap();
        let program =
            collective_program(ParallelBroadcastOperation::new("x".to_string(), 2, 0).with_mesh(mesh), input_type);
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=2:manual, 'y'=2:manual, 'z'=2:manual]>, [], reduced={'z'}, \
                    varying_manual={'x', 'y'}}] .
                let %1:f32[][sharding={mesh<['x'=2:manual, 'y'=2:manual, 'z'=2:manual]>, [], reduced={'z'}, \
                        varying_manual={'y'}}] = \
                        parallel_reduce [kind=sum, axis_name=\"x\", \
                        mesh=['x'=2:manual, 'y'=2:manual, 'z'=2:manual]] %0
                    %2:f32[][sharding={mesh<['x'=2:manual, 'y'=2:manual, 'z'=2:manual]>, [], reduced={'z'}, \
                        varying_manual={'x', 'y'}}] = parallel_vary [axis_name=\"x\"] %1
                    %3:f32[][sharding={mesh<['x'=2:manual, 'y'=2:manual, 'z'=2:manual]>, [], reduced={'z'}, \
                        varying_manual={'x', 'y'}}] = parallel_permute [
                        axes=\"x\",
                        axis_size=2,
                        source_target_pairs=[(0, 0)],
                        mesh=['x'=2:manual, 'y'=2:manual, 'z'=2:manual],
                    ] %2
                in (%3)"
            },
        );
        assert_eq!(
            transposed.transpose_with_respect_to(&[0], &[]).unwrap().to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=2:manual, 'y'=2:manual, 'z'=2:manual]>, [], unreduced={'z'}, \
                    varying_manual={'x', 'y'}}] .
                let %1:f32[][sharding={mesh<['x'=2:manual, 'y'=2:manual, 'z'=2:manual]>, [], unreduced={'z'}, \
                    varying_manual={'x', 'y'}}] = parallel_permute [
                    axes=\"x\",
                    axis_size=2,
                    source_target_pairs=[(0, 0)],
                    mesh=['x'=2:manual, 'y'=2:manual, 'z'=2:manual],
                ] %0
                    %2:f32[][sharding={mesh<['x'=2:manual, 'y'=2:manual, 'z'=2:manual]>, [], unreduced={'z'}, \
                        varying_manual={'y'}}] = \
                        parallel_reduce [kind=sum, axis_name=\"x\", \
                        mesh=['x'=2:manual, 'y'=2:manual, 'z'=2:manual]] %1
                    %3:f32[][sharding={mesh<['x'=2:manual, 'y'=2:manual, 'z'=2:manual]>, [], unreduced={'z'}, \
                        varying_manual={'x', 'y'}}] = parallel_vary [axis_name=\"x\"] %2
                in (%3)"
            },
        );
    }

    #[test]
    fn test_parallel_broadcast_parallel_broadcast() {
        // A name that no enclosing binder binds fails fast instead of silently acting as the identity.
        let inputs = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(
            batch(
                |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>| {
                    item.parallel_broadcast("y", 1)
                },
                inputs.clone(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxisSpecification::named("x"),
            ),
            Err::<Array, _>(BatchingError::Axis(AxisError::UnboundAxisName { name: "y".to_string() })),
        );

        // A name that a `batch` level binds resolves the size of that level and broadcasts its source item, and a
        // source outside that level is rejected.
        assert_eq!(
            batch(
                |item| item.parallel_broadcast("x", 2),
                inputs.clone(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxisSpecification::named("x"),
            ),
            Ok(Array::vector(vec![3.0, 3.0, 3.0]).unwrap()),
        );
        assert_eq!(
            batch(
                |item| item.parallel_broadcast("x", 3),
                inputs,
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxisSpecification::named("x"),
            ),
            Err::<Array, _>(BatchingError::Type(TypeError::invalid(
                "`parallel_broadcast` source 3 is out of bounds for axis size 3",
            ))),
        );

        // An array IR value broadcasts through its array view.
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()),
        )
        .with_axis_name("x".to_string());
        let input = ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let input = BatchingTracer::new(context, ArrayIrBatch::new(input, BatchAxis::new(0)).unwrap());
        assert_eq!(
            input.parallel_broadcast("x", 0).map(|output| output.into_batch().into_value()),
            Ok(ArrayIrValue::Array(Array::vector(vec![1.0f32, 1.0, 1.0]).unwrap())),
        );

        // Over a manual mesh axis, a varying value is broadcast directly, while an invariant value is first made
        // varying, because the broadcast output keeps its input's variation.
        let sharding = Sharding::replicated(manual_mesh(), 0);
        let varying = ArrayType::scalar(DataType::F32)
            .with_sharding(sharding.clone().with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let invariant = ArrayType::scalar(DataType::F32).with_sharding(sharding.clone()).unwrap();
        let (output, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_broadcast("x", 1),
            varying.clone(),
            manual_axes(true),
        )
        .unwrap();
        assert_eq!(output, varying);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [], varying_manual={'x'}}] .
                let %1:f32[][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [], varying_manual={'x'}}] = \
                    parallel_broadcast [axes=\"x\", axis_size=2, source=1, mesh=['x'=2:manual, 'y'=1:manual]] %0
                in (%1)"
            },
        );
        let (output, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_broadcast("x", 1),
            invariant.clone(),
            manual_axes(true),
        )
        .unwrap();
        assert_eq!(output, varying);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, []}] .
                let %1:f32[][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %0
                    %2:f32[][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [], varying_manual={'x'}}] = \
                        parallel_broadcast [axes=\"x\", axis_size=2, source=1, mesh=['x'=2:manual, 'y'=1:manual]] %1
                in (%2)"
            },
        );

        // An unchecked manual binding tracks no variation, so it stages an ordinary broadcast without a mesh.
        let (output, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_broadcast("x", 1),
            invariant,
            manual_axes(false),
        )
        .unwrap();
        assert_eq!(output, ArrayType::scalar(DataType::F32).with_sharding(sharding.clone()).unwrap());
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, []}] .
                let %1:f32[][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, []}] = \
                    parallel_broadcast [axes=\"x\", axis_size=2, source=1] %0
                in (%1)"
            },
        );

        // Over a manual mesh axis, a value with a pending cross-device sum is rejected before it is made varying.
        assert_eq!(
            TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                |input| input.parallel_broadcast("x", 1),
                ArrayType::scalar(DataType::F32)
                    .with_sharding(sharding.with_unreduced_axes(["x"]).unwrap())
                    .unwrap(),
                manual_axes(true),
            )
            .map(|(output, _)| output),
            Err(ProgramError::Type(TypeError::invalid("`parallel_broadcast` does not support unreduced inputs"))),
        );

        // A concrete array is never inside an axis binder, so every axis name is unbound for it, and an empty list of
        // axis names is rejected before any of them is resolved.
        assert_eq!(
            Array::vector(vec![1.0, 2.0]).unwrap().parallel_broadcast("i", 0),
            Err(ProgramError::Axis(AxisError::UnboundAxisName { name: "i".to_string() })),
        );
        assert_eq!(
            Array::vector(vec![1.0, 2.0]).unwrap().parallel_broadcast(Vec::<String>::new(), 0),
            Err(ProgramError::Type(TypeError::invalid("`parallel_broadcast` requires at least one axis name"))),
        );
    }

    #[test]
    fn test_parallel_broadcast_parallel_broadcast_array_ir() {
        // Concrete array IR values broadcast through their array members, which for concrete arrays outside a
        // named-axis environment report the unbound axis.
        let vector = Array::vector(vec![1.0f32, 2.0]).unwrap();
        assert_eq!(
            ArrayIrValue::Array(vector.clone()).parallel_broadcast("x", 0),
            vector.parallel_broadcast("x", 0).map(ArrayIrValue::Array),
        );
    }
}
