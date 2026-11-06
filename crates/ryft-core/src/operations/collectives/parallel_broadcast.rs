use std::borrow::Cow;
use std::fmt::Display;
use std::num::NonZeroUsize;

use ryft_macros::capability;

use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrOperation, ArrayIrType, ArrayOperation,
    ArrayType, DataType, Dimension, LogicalMesh, Shape, ShardingDimension,
};
use crate::axes::{AxisError, NamedAxes};
use crate::batching::{BatchAxis, BatchableOperation, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{
    CotangentAccumulator, DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDriver,
    DifferentiationDual, DifferentiationError, DifferentiationPolicy, TransposableOperation, TranspositionContext,
    TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::Capability;
use crate::operations::arithmetic::{Add, AddOperation, Mul, MulOperation};
use crate::operations::assertions::Assert;
use crate::operations::collectives::axis_index::AxisIndexOperation;
use crate::operations::collectives::parallel_all_gather::ParallelAllGather;
use crate::operations::collectives::parallel_permute::{PARALLEL_PERMUTE_OPERATION_NAME, ParallelPermuteOperation};
use crate::operations::collectives::parallel_reduce::{ParallelReduce, ParallelReduceOperation};
use crate::operations::collectives::parallel_vary::{
    PARALLEL_VARY_OPERATION_NAME, ParallelVary, ParallelVaryOperation,
};
use crate::operations::collectives::{
    CollectiveAxes, LinearCollectiveOperation, resolve_collective_axes, route_axis_slices,
    validate_manual_mesh_axes_input,
};
use crate::operations::comparisons::{Compare, CompareOperation};
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::constants::fill::Fill;
use crate::operations::constants::zero_like::{ZeroLike, ZeroLikeOperation};
use crate::operations::control_flow::select::{Select, SelectOperation};
use crate::operations::manipulation::broadcasting::{Broadcast, BroadcastOperation};
use crate::operations::manipulation::concatenation::Concatenate;
use crate::operations::manipulation::memory::{TransferToMemory, TransferToMemoryOperation};
use crate::operations::manipulation::reshaping::Reshape;
use crate::operations::manipulation::slicing::Slice;
use crate::operations::manipulation::transposition::Transpose;
use crate::operations::reductions::{Reduce, ReductionKind};
use crate::operations::sharding::reshard::Reshard;
use crate::partial::{
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationValue, PartialValue,
    PartiallyEvaluatableOperation,
};
use crate::programs::{
    EffectClass, EffectClasses, Effects, MaybeZero, Operation, OperationFormatter, ProgramError, RegionInterface,
    TypeError, Typed, Value, ValueDomainDispatch, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

/// Selects the source representation of a [`ParallelBroadcastOperation`]. Runtime values are instruction inputs,
/// so this metadata describes their presence without storing a tracer or array inside the operation.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ParallelBroadcastSource {
    /// The operation stores the source's zero-based coordinate along its combined named axis.
    Static(usize),

    /// A second instruction input supplies the source as a non-differentiable `u64` scalar.
    Dynamic,
}

impl From<usize> for ParallelBroadcastSource {
    #[inline]
    fn from(source: usize) -> Self {
        Self::Static(source)
    }
}

impl Display for ParallelBroadcastSource {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Static(source) => source.fmt(formatter),
            Self::Dynamic => formatter.write_str("dynamic"),
        }
    }
}

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
/// This is the Ryft analogue of JAX's
/// [`jax.lax.pbroadcast`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.pbroadcast.html) and of StableHLO's
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
/// With a fixed source the collective is linear in its data. For a static source its tangent rides the same broadcast,
/// and its transpose hands the sum of every participant's cotangent to the source and zeros to every other participant,
/// which is staged as a [`ParallelReduceOperation`] sum followed by a [`ParallelPermuteOperation`] whose only pair
/// sends the source's sum to itself (over a manual mesh axis, with a [`ParallelVaryOperation`] in between that restores
/// the input cotangent's variation). JAX instead selects the sum with an `axis_index` comparison, which computes the
/// same values but needs a mask and zeros of the cotangent's shape. Outside any binder, the single participant of a
/// degenerate axis keeps its value, and a larger axis has no per-item semantics.
///
/// [`ParallelBroadcastSource::Dynamic`] supplies the source through a second input, a `u64` scalar.
/// All participants of a group must supply the same coordinate in `0..axis_size`. The operation carries an
/// [`EffectClass::OrderedAssertion`] so agreement and bounds checks remain observable even when its result is unused.
/// The source cannot carry reduction state and has no tangent or cotangent. Differentiation checks the primal and
/// records the same operation with validation discharged in the tangent, retaining the source as a known residual
/// for its pullback. Only derivative rules can discharge that obligation. Transpose a linearized program to avoid
/// treating the primal's ordered checks as linear work.
///
/// Runtime interpretation and lowering gather participant values and select exact slices, preserving Boolean, wide
/// integer, complex, NaN, and signed-zero payloads. This portable form communicates and stores all participants' values
/// rather than using a native dynamic-root broadcast. A root mapped over a named batch is accepted when participant
/// values agree, and roots may differ across unrelated groups.
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

    /// Static coordinate or declaration of the runtime source input. Refer to [`ParallelBroadcastSource`].
    source: ParallelBroadcastSource,

    /// Whether runtime source validation is still a primal obligation. Constructors require it for dynamic roots.
    /// Only derivative rules clear it after retaining the checked primal, so linear programs remain effect-free.
    requires_source_validation: bool,

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
    ///   - `source`: Static coordinate in `0..axis_size`, or [`ParallelBroadcastSource::Dynamic`] to consume an
    ///     additional `u64` scalar input. A `usize` converts to [`ParallelBroadcastSource::Static`].
    #[inline]
    pub fn new<A: Into<CollectiveAxes>, S: Into<ParallelBroadcastSource>>(
        axes: A,
        axis_size: usize,
        source: S,
    ) -> Self {
        let source = source.into();
        Self {
            axes: axes.into(),
            axis_size,
            source,
            requires_source_validation: source == ParallelBroadcastSource::Dynamic,
            mesh: None,
        }
    }

    /// Returns this [`ParallelBroadcastOperation`] configured to broadcast over manual axes of `mesh`. The input
    /// must vary over [`axes`](Self::axes) on that mesh, whose size must equal [`axis_size`](Self::axis_size).
    /// Type inference validates these requirements. [`ParallelBroadcast::parallel_broadcast`] supplies the mesh
    /// automatically from the enclosing manual region.
    #[inline]
    pub fn with_mesh(mut self, mesh: LogicalMesh) -> Self {
        if self.source == ParallelBroadcastSource::Dynamic && self.mesh.as_ref() != Some(&mesh) {
            self.requires_source_validation = true;
        }
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

    /// Returns the static coordinate or runtime input declaration of this [`ParallelBroadcastOperation`].
    #[inline]
    pub fn source(&self) -> ParallelBroadcastSource {
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
        check_count!("region", region_interfaces, 0, TypeError);
        self.axes.validate(PARALLEL_BROADCAST_OPERATION_NAME)?;
        if self.axis_size == 0 {
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_BROADCAST_OPERATION_NAME}` axis size must be greater than zero"
            )));
        }

        let expected_input_count = if self.source == ParallelBroadcastSource::Dynamic { 2 } else { 1 };
        check_count!("input", input_types, expected_input_count, TypeError);
        let input_type = &input_types[0];
        if let ParallelBroadcastSource::Static(source) = self.source
            && source >= self.axis_size
        {
            return Err(TypeError::invalid(format!(
                "`{}` source {} is out of bounds for axis size {}",
                PARALLEL_BROADCAST_OPERATION_NAME, source, self.axis_size,
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

        if self.source == ParallelBroadcastSource::Dynamic {
            let source = &input_types[1];
            if source.data_type() != DataType::U64 || source.rank() != 0 {
                return Err(TypeError::invalid(format!(
                    "`{PARALLEL_BROADCAST_OPERATION_NAME}` runtime source must be a `u64` scalar but got `{source}`",
                )));
            }

            if !source.unreduced_axes().is_empty() || !source.reduced_axes().is_empty() {
                return Err(TypeError::invalid(format!(
                    "`{PARALLEL_BROADCAST_OPERATION_NAME}` runtime source must not carry reduction state",
                )));
            }

            if source.memory() != input_type.memory() {
                return Err(TypeError::invalid(format!(
                    "`{PARALLEL_BROADCAST_OPERATION_NAME}` runtime source must use the data's memory",
                )));
            }

            // Source-dependent selection cannot introduce variation beyond the result's declared variation.
            if source.sharding().is_some_and(|source| {
                source
                    .varying_manual_axes()
                    .iter()
                    .any(|axis| !input_type.sharding().is_some_and(|data| data.varying_manual_axes().contains(axis)))
            }) {
                return Err(TypeError::invalid(format!(
                    "`{PARALLEL_BROADCAST_OPERATION_NAME}` runtime source variation must be included in \
                     the data's variation",
                )));
            }

            if let (Some(data), Some(source)) = (input_type.sharding(), source.sharding())
                && data.mesh() != source.mesh()
            {
                return Err(TypeError::invalid(format!(
                    "`{PARALLEL_BROADCAST_OPERATION_NAME}` runtime source mesh does not match the data mesh",
                )));
            }
        }

        Ok(vec![input_type.clone()])
    }

    #[inline]
    fn effects(&self) -> Cow<'_, Effects> {
        if self.requires_source_validation {
            Cow::Owned(Effects::explicit(EffectClasses::single(EffectClass::OrderedAssertion)))
        } else {
            Cow::Borrowed(Effects::empty())
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, PARALLEL_BROADCAST_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("axes", format_args!("{:?}", self.axes))?;
            operation.field("axis_size", self.axis_size)?;
            operation.field("source", self.source)?;
            if self.source == ParallelBroadcastSource::Dynamic && !self.requires_source_validation {
                operation.field("source_validated", true)?;
            }
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

    #[inline]
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

impl<C: Domain<Type = ArrayType> + Fill<u64, C::Value>> InterpretableOperation<C> for ParallelBroadcastOperation
where
    C::Value:
        ParallelAllGather + ParallelReduce + Compare + Select + Slice + Reshape + TransferToMemory + Assert + Broadcast,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.infer_output_types(&input_types, &[])?;
        if matches!(self.source, ParallelBroadcastSource::Static(_)) {
            self.validate_degenerate_interpretation()?;
            return Ok(vec![inputs[0].clone()]);
        }
        interpret_dynamic_parallel_broadcast(self, context, inputs, None)
    }
}

impl<C: Context<Type = ArrayType, Operation: From<ParallelBroadcastOperation>>> PartiallyEvaluatableOperation<C>
    for ParallelBroadcastOperation
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        self.infer_output_types(
            &inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
            &driver.region_interfaces(),
        )?;
        let regions = driver.region_programs();
        match context.fold_or_residualize(self.clone(), regions.clone(), inputs) {
            Err(error)
                if self.source == ParallelBroadcastSource::Dynamic
                    && self.axis_size > 1
                    && context.parent().is_eager() =>
            {
                // Axis resolution precedes runtime assertions. An eager specialization without this collective's
                // binder can retain the whole check, while a batched eager parent executes it normally.
                let unbound = match &error {
                    ProgramError::Axis(AxisError::UnboundAxisName { name }) => self.axes.contains(name),
                    _ => matches!(error.downcast_custom::<BatchingError>(),
                        Some(BatchingError::Axis(AxisError::UnboundAxisName { name })) if self.axes.contains(name)),
                };
                if unbound { context.residualize(self.clone(), regions, inputs) } else { Err(error) }
            }
            result => result,
        }
    }
}

impl<
    C: Context<Type = ArrayType, Operation: From<ParallelBroadcastOperation>> + Fill<u64, C::Value>,
    P: ArrayExtentBatchingPolicy<C>,
> BatchableOperation<C, ArrayBatchingPolicy<P>> for ParallelBroadcastOperation
where
    C::Value: ZeroLike
        + Concatenate
        + Slice
        + Transpose
        + Reshard
        + ParallelAllGather
        + ParallelReduce
        + Compare
        + Select
        + Reshape
        + TransferToMemory
        + Assert
        + Reduce
        + Broadcast,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        if self.source == ParallelBroadcastSource::Dynamic {
            check_count!("input", inputs, 2, ProgramError);
            self.infer_output_types(&inputs.iter().map(|input| input.unbatched_type()).collect::<Vec<_>>(), &[])?;
            let (data, source) = (&inputs[0], &inputs[1]);
            if context.axis_name().is_some_and(|axis| self.axes.contains(axis)) {
                self.reject_mesh_form()?;
                let batch_size = P::axis_size(context)?;
                if batch_size != self.axis_size {
                    return Err(BatchingError::UnsupportedOperation {
                        message: format!(
                            "`{}` over axis `{}` resolved axis size {} but the mapped batch axis has size {}",
                            PARALLEL_BROADCAST_OPERATION_NAME, self.axes, self.axis_size, batch_size,
                        ),
                    });
                }

                // This level owns the participants. Reduce a mapped root over its physical batch dimension,
                // checking agreement before replacing the participant dimension by a replicated selected item.
                let minimum = if let Some(axis) = source.batch_axis_position() {
                    source.value().reduce(&[axis], ReductionKind::Min)?
                } else {
                    source.value().clone()
                };

                let maximum = if self.requires_source_validation
                    && let Some(axis) = source.batch_axis_position()
                {
                    source.value().reduce(&[axis], ReductionKind::Max)?
                } else {
                    minimum.clone()
                };

                if self.requires_source_validation {
                    validate_runtime_broadcast_source(context.parent(), &minimum, &maximum, &minimum, self.axis_size)?;
                }

                let Some(axis) = data.batch_axis_position() else {
                    return Ok(vec![data.clone()]);
                };

                let mut gathered = data.value().move_axis(axis, 0)?;
                if let Some(sharding) = gathered.r#type().sharding().cloned()
                    && sharding.dimensions()[0] != ShardingDimension::Replicated
                {
                    // Participant slices can be smaller than their original device placement. Replicate the
                    // routed dimension before slicing, as static routing does, and restore it on materialization.
                    let mut dimensions = sharding.dimensions().to_vec();
                    dimensions[0] = ShardingDimension::Replicated;
                    gathered = gathered.reshard(&sharding.with_dimensions(dimensions).map_err(TypeError::from)?)?;
                }

                let mut selected =
                    select_runtime_broadcast_source(context.parent(), &gathered, &minimum, self.axis_size, None)?;
                let output_type = data.unbatched_type();
                if selected.r#type().as_ref() != &output_type {
                    selected = selected.broadcast(output_type.clone(), &(0..output_type.rank()).collect::<Vec<_>>())?;
                }
                return Ok(vec![ArrayBatch::replicated(selected)]);
            }

            if source.batch_axis_position().is_none() {
                // An unrelated mapped data dimension is simply part of each participant's payload. A replicated
                // scalar root stays scalar, so the same operation can run in the parent without changing its arity.
                let mut outputs =
                    context.parent().bind(self.clone(), Vec::new(), &[data.value().clone(), source.value().clone()])?;
                check_count!("output", outputs, 1, ProgramError);
                return Ok(vec![ArrayBatch::new(outputs.remove(0), data.batch_axis())?]);
            }

            // A root mapped over an unrelated axis selects independently in each group. Canonicalize that batch
            // dimension to the front and interpret the vector of scalar roots over parent-owned packed arrays.
            let data = P::match_axis(context, data, 0.into())?;
            let source = source.value().move_axis(source.batch_axis_position().unwrap(), 0)?;
            let mut outputs =
                interpret_dynamic_parallel_broadcast(self, context.parent(), &[data.into_value(), source], Some(0))?;
            check_count!("output", outputs, 1, ProgramError);
            return Ok(vec![ArrayBatch::new(outputs.remove(0), BatchAxis::new(0))?]);
        }

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
        // here, against the logical (i.e., unbatched) input type, before its source indexes the batch axis.
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
        let ParallelBroadcastSource::Static(source) = self.source else {
            // Dynamic sources return from the branch at the start of this rule, so only `Static` can reach here.
            unreachable!();
        };
        let sources = vec![Some(source); batch_size];
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
        if matches!(self.source, ParallelBroadcastSource::Static(_)) {
            return self.linear_collective_jvp(context, inputs);
        }
        check_count!("input", inputs, 2, ProgramError);
        let mut primals = context.primal().bind(
            self.clone(),
            Vec::new(),
            &[inputs[0].primal().clone(), inputs[1].primal().clone()],
        )?;
        check_count!("output", primals, 1, ProgramError);
        let primal = primals.remove(0);
        let tangent = match inputs[0].tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
            MaybeZero::Value(tangent) => {
                // The checked primal owns source validation. Keep the tangent pure so linearization can
                // retain the root as a known residual and transpose this same operation without replaying checks.
                let source = context.primal_to_tangent(inputs[1].primal().clone())?;
                let mut operation = self.clone();
                operation.requires_source_validation = false;
                let mut tangents = context.tangent().bind(operation, Vec::new(), &[tangent.clone(), source])?;
                check_count!("output", tangents, 1, ProgramError);
                MaybeZero::Value(tangents.remove(0))
            }
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

impl<
    V: Value<Type = ArrayType>,
    O: Operation<Type = ArrayType>
        + From<AddOperation<ArrayType>>
        + From<ParallelReduceOperation>
        + From<ParallelVaryOperation>
        + From<ParallelPermuteOperation>
        + From<AxisIndexOperation>
        + From<ConstantOperation<Array>>
        + From<MulOperation<ArrayType>>
        + From<CompareOperation<ArrayType>>
        + From<SelectOperation<ArrayType>>
        + From<BroadcastOperation>
        + From<TransferToMemoryOperation>
        + From<ZeroLikeOperation<ArrayType>>,
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
        let input_count = if self.source == ParallelBroadcastSource::Dynamic { 2 } else { 1 };
        check_count!("input", inputs, input_count, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        check_count!("accumulator", accumulators, input_count, DifferentiationError);
        self.infer_output_types(&inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(), &[])?;
        let MaybeZero::Value(cotangent) = &outputs[0] else {
            return Ok(());
        };

        if inputs[0].is_known() || !accumulators[0].is_needed() {
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

            match self.source {
                ParallelBroadcastSource::Static(source) => {
                    // A permutation whose only pair sends the source's sum to itself keeps that sum at the source and
                    // gives every other participant zeros. Unlike a mask, it needs no zeros of the
                    // cotangent's shape to select from, and it preserves every placement and variation of its input.
                    let mut operation =
                        ParallelPermuteOperation::new(self.axes.clone(), self.axis_size, vec![(source, source)]);
                    if let Some(mesh) = &self.mesh {
                        operation = operation.with_mesh(mesh.clone());
                    }
                    contribution = bind_single_output(context, operation.into(), contribution)?;
                }
                ParallelBroadcastSource::Dynamic => {
                    let source = inputs[1].as_known().ok_or_else(|| ProgramError::UnsupportedOperation {
                        message: format!(
                            "`{PARALLEL_BROADCAST_OPERATION_NAME}` transpose requires the runtime source \
                             as a known primal",
                        ),
                    })?;
                    let input_type = inputs[0].r#type();
                    let mesh = self.mesh.as_ref().or_else(|| input_type.sharding().map(|sharding| sharding.mesh()));
                    let mut position: Option<Tracer<TracingContext<V, O>>> = None;
                    for axis in self.axes.names() {
                        let mut operation = AxisIndexOperation::new(axis.clone());
                        if let Some(mesh) = &self.mesh {
                            operation = operation.with_mesh(mesh.clone());
                        }
                        let mut indices = context.bind(operation, Vec::new(), &[])?;
                        check_count!("output", indices, 1, ProgramError);
                        let index = indices.remove(0).transfer_to_memory(contribution.r#type().memory())?;
                        position = Some(if let Some(position) = position {
                            let size = mesh.and_then(|mesh| mesh.axis_size(axis)).ok_or_else(|| {
                                ProgramError::UnsupportedOperation {
                                    message: format!(
                                        "`{PARALLEL_BROADCAST_OPERATION_NAME}` transpose requires mesh extents \
                                         for several axes",
                                    ),
                                }
                            })?;

                            // This non-differentiable constant is aligned through the value capabilities below.
                            let size =
                                Array::scalar(size as u64)?.transfer_to_memory(contribution.r#type().memory())?;
                            let mut sizes = context.bind(ConstantOperation::new(size), Vec::new(), &[])?;
                            check_count!("output", sizes, 1, ProgramError);
                            position.mul(&sizes.remove(0))?.add(&index)?
                        } else {
                            index
                        });
                    }
                    let mask = position.unwrap().equal(source)?;
                    contribution = mask.select(&contribution, &contribution.zero_like()?)?;
                }
            }
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
/// In the following example, every batch item receives the row of batch item two:
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
///
/// A source supplied at runtime is an ordinary scalar input:
///
/// ```rust
/// # use ryft_core::{
/// #     Array, ArrayBatchingPolicy, ArrayOperation, BatchAxis, BatchAxisSpecification, BatchingTracer,
/// #     ParallelBroadcast, EagerContext, batch,
/// # };
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let rows = Array::matrix(3, 2, vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
/// let source = Array::scalar(1u64)?;
/// let broadcast = batch(
///     |(row, source): (
///         BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>,
///         BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>,
///     )| row.dynamic_parallel_broadcast("rows", &source),
///     (rows, source),
///     (BatchAxis::new(0), BatchAxis::replicated()),
///     BatchAxis::new(0),
///     BatchAxisSpecification::named("rows"),
/// )?;
/// assert_eq!(broadcast, Array::matrix(3, 2, vec![3f32, 4.0, 3.0, 4.0, 3.0, 4.0])?);
/// # Ok(())
/// # }
/// ```
#[capability]
pub trait ParallelBroadcast<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Returns the value of participant `source` along the named axes `axes`, which every participant receives. Over
    /// a manual mesh axis, an input that does not vary over the axis is first made varying through [`ParallelVary`],
    /// because the broadcast output keeps its input's variation. JAX's `pbroadcast` rejects such an input instead,
    /// so this is a deliberate extension that mirrors how [`ParallelPermute`](crate::ParallelPermute) stages it.
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

    /// Copies participant `source`'s value to every participant along `axes`. The source is a non-differentiable
    /// `u64` scalar, matching [`AxisIndex`](crate::AxisIndex), and may differ between groups formed by unrelated axes.
    /// The output follows the data's shape, memory, reduction state, and the manual variation of the data and source,
    /// varying over the selected manual axes.
    ///
    /// # Parameters
    ///
    ///   - `axes`: One named batch or manual mesh axis, or several manual mesh axes in major-to-minor order.
    ///   - `source`: Runtime scalar giving the source's position along the combined axis.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] for invalid source types or reduction state, invalid or unbound axes, incompatible
    /// meshes, or unsupported delegated operations. Executed assertions reject out-of-range or inconsistent sources.
    fn dynamic_parallel_broadcast<Axes: Into<CollectiveAxes>>(
        &self,
        axes: Axes,
        source: &Self,
    ) -> Result<Self, ProgramError>;
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

    fn dynamic_parallel_broadcast<Axes: Into<CollectiveAxes>>(
        &self,
        axes: Axes,
        _source: &Self,
    ) -> Result<Self, ProgramError> {
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
        > + ParallelVary
        + TransferToMemory,
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

    fn dynamic_parallel_broadcast<Axes: Into<CollectiveAxes>>(
        &self,
        axes: Axes,
        source: &Self,
    ) -> Result<Self, ProgramError> {
        let axes = axes.into();
        let context = self.domain();
        let (axis_size, mesh) = resolve_collective_axes(&context, PARALLEL_BROADCAST_OPERATION_NAME, &axes)?;
        let mut operation = ParallelBroadcastOperation::new(axes.clone(), axis_size, ParallelBroadcastSource::Dynamic);
        let mut value = self.clone();
        if let Some(mesh) = mesh {
            for axis in axes.names() {
                if value.r#type().unreduced_axes().contains(axis) {
                    return Err(TypeError::invalid(format!(
                        "`{PARALLEL_BROADCAST_OPERATION_NAME}` does not support unreduced inputs"
                    ))
                    .into());
                }
                if !value.r#type().sharding().is_some_and(|data| data.varying_manual_axes().contains(axis)) {
                    value = value.parallel_vary(axis)?;
                }
            }
            operation = operation.with_mesh(mesh);
        }

        // Source variation on unrelated axes must also be reflected by the selected data.
        let source_type = source.r#type();
        for axis in source_type.sharding().map(|source| source.varying_manual_axes()).into_iter().flatten() {
            if !value.r#type().sharding().is_some_and(|data| data.varying_manual_axes().contains(axis)) {
                value = value.parallel_vary(axis)?;
            }
        }
        let source = if source.r#type().memory() == value.r#type().memory() {
            source.clone()
        } else {
            source.transfer_to_memory(value.r#type().memory())?
        };
        operation.infer_output_types(&[value.r#type().into_owned(), source.r#type().into_owned()], &[])?;
        let mut outputs = context.bind(operation, Vec::new(), &[value, source])?;
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

    fn dynamic_parallel_broadcast<Axes: Into<CollectiveAxes>>(
        &self,
        axes: Axes,
        source: &Self,
    ) -> Result<Self, ProgramError> {
        Ok(V::from_projected(
            ValueProjection::<ArrayType>::into_projected(self.clone())?
                .dynamic_parallel_broadcast(axes, &ValueProjection::<ArrayType>::into_projected(source.clone())?)?,
        ))
    }
}

// TODO(eaplatanios): Review from here onwards.

/// Interprets a runtime broadcast, including a packed unrelated batch of scalar roots when `source_batch_axis`
/// is present. Primal interpretation validates roots; derived linear instructions reuse selection without effects.
fn interpret_dynamic_parallel_broadcast<C: Domain<Type = ArrayType> + Fill<u64, C::Value>>(
    operation: &ParallelBroadcastOperation,
    context: &C,
    inputs: &[C::Value],
    source_batch_axis: Option<usize>,
) -> Result<Vec<C::Value>, ProgramError>
where
    C::Value:
        ParallelAllGather + ParallelReduce + Compare + Select + Slice + Reshape + TransferToMemory + Assert + Broadcast,
{
    let (value, source) = (&inputs[0], &inputs[1]);
    if operation.requires_source_validation {
        // Integer extrema make every participant check the same bounds and agreement conditions, including
        // unchecked manual bindings whose variation metadata alone cannot prove agreement.
        let mut minimum = source.clone();
        let mut maximum = source.clone();
        for axis in operation.axes.names().iter().filter(|_| operation.axis_size > 1) {
            minimum = minimum.parallel_reduce(ReductionKind::Min, axis)?;
            maximum = maximum.parallel_reduce(ReductionKind::Max, axis)?;
        }
        validate_runtime_broadcast_source(context, &minimum, &maximum, source, operation.axis_size)?;
    }
    if operation.axis_size == 1 {
        return Ok(vec![value.clone()]);
    }
    let gathered = value.parallel_all_gather(operation.axes.clone(), 0)?;
    let mut output =
        select_runtime_broadcast_source(context, &gathered, source, operation.axis_size, source_batch_axis)?;
    let output_type = value.r#type().into_owned();
    if output.r#type().as_ref() != &output_type {
        output = output.broadcast(output_type.clone(), &(0..output_type.rank()).collect::<Vec<_>>())?;
    }
    Ok(vec![output])
}

/// Validates scalar roots, or a rectangular packed batch of scalar roots, without depending on the payload values.
fn validate_runtime_broadcast_source<C: Domain<Type = ArrayType> + Fill<u64, C::Value>>(
    context: &C,
    minimum: &C::Value,
    maximum: &C::Value,
    source: &C::Value,
    axis_size: usize,
) -> Result<(), ProgramError>
where
    C::Value: Compare + Assert,
{
    /// Scalar roots keep the ordinary assertion diagnostic. Packed unrelated batches retain individual failures.
    fn assert<V: Value<Type = ArrayType> + Assert>(
        condition: V,
        message: &str,
        observations: &[(&str, V)],
    ) -> Result<(), ProgramError> {
        if condition.r#type().rank() == 0 {
            condition.assert(message, observations)
        } else {
            condition.assert_with_limit(message, observations, NonZeroUsize::new(1).unwrap())
        }
    }
    assert(
        minimum.equal(maximum)?,
        &format!("`{PARALLEL_BROADCAST_OPERATION_NAME}` runtime sources must agree across participants"),
        &[("minimum", minimum.clone()), ("maximum", maximum.clone())],
    )?;
    let limit =
        context.fill(&ArrayType::scalar(DataType::U64).with_memory(source.r#type().memory()), axis_size as u64)?;
    assert(
        maximum.less_than(&limit)?,
        &format!("`{PARALLEL_BROADCAST_OPERATION_NAME}` runtime source must be smaller than axis size {axis_size}"),
        &[("source", source.clone())],
    )?;
    Ok(())
}

/// Selects exact participant slices from the leading gathered axis. A packed root vector selects independently
/// along `source_batch_axis` of each candidate, so singleton dimensions align it with the remaining item dimensions.
fn select_runtime_broadcast_source<C: Domain<Type = ArrayType> + Fill<u64, C::Value>>(
    context: &C,
    gathered: &C::Value,
    source: &C::Value,
    axis_size: usize,
    source_batch_axis: Option<usize>,
) -> Result<C::Value, ProgramError>
where
    C::Value: Compare + Select + Slice + Reshape,
{
    let mut selected = gathered.slice_axis(0, 0, 1, 1)?.squeeze(&[0])?;
    for index in 1..axis_size {
        let candidate = gathered.slice_axis(0, index, index + 1, 1)?.squeeze(&[0])?;
        let index =
            context.fill(&ArrayType::scalar(DataType::U64).with_memory(source.r#type().memory()), index as u64)?;
        let mut condition = source.equal(&index)?;
        if let Some(axis) = source_batch_axis {
            let mut dimensions = vec![Dimension::Static(1); candidate.r#type().rank()];
            dimensions[axis] = source.r#type().shape().dimensions()[0].clone();
            condition = condition.reshape(Shape::new(dimensions))?;
        }
        selected = condition.select(&candidate, &selected)?;
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrValue, Complex, DataType, Dimension, DimensionBounds,
        DimensionValue, DimensionVariable, Layout, Memory, MeshAxis, MeshAxisType, Shape, Sharding, ShardingDimension,
        StridedLayout,
    };
    use crate::axes::NamedAxis;
    use crate::batching::{BatchAxisSpecification, BatchingTracer, batch};
    use crate::contexts::EagerContext;
    use crate::differentiation::differentiate_at;
    use crate::macros::{check_gradient, check_operation_partial_evaluation, check_operation_type_inference};
    use crate::operations::assertions::AssertionError;
    use crate::operations::collectives::axis_index::AxisIndex;
    use crate::operations::collectives::tests::{batch_collective, collective_program};
    use crate::operations::reductions::Reduce;
    use crate::parameters::Placeholder;
    use crate::partial::{
        PartialEvaluationContext, PartialEvaluationInput, PartialEvaluationOutput, PartialEvaluationValue,
    };
    use crate::programs::{EmptyRegionDriver, Program, ProgramBuilder};
    use crate::tests::hash_of;

    use super::*;

    /// Array-valued item of a named eager batch.
    type TestTracer = BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>;

    /// Broadcasts rows along "x" using either one shared runtime source or one source per participant.
    fn broadcast_runtime_source(rows: Array, source: Array, source_axis: BatchAxis) -> Result<Array, BatchingError> {
        batch(
            |(value, source): (TestTracer, TestTracer)| value.dynamic_parallel_broadcast("x", &source),
            (rows, source),
            (BatchAxis::new(0), source_axis),
            BatchAxis::new(0),
            BatchAxisSpecification::named("x"),
        )
    }

    /// Builds a broadcast whose runtime root remains an ordinary input, without a frontend axis binder.
    fn runtime_broadcast_program(axis_size: usize) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let data = builder.add_input(ArrayType::scalar(DataType::F32));
        let source = builder.add_input(ArrayType::scalar(DataType::U64));
        let outputs = builder
            .add_instruction(
                ParallelBroadcastOperation::new("x", axis_size, ParallelBroadcastSource::Dynamic),
                Vec::new(),
                vec![data, source],
                None,
            )
            .unwrap()
            .to_vec();
        builder.build(outputs, vec![Placeholder; 2], vec![Placeholder]).unwrap()
    }

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
    fn test_parallel_broadcast_source() {
        let source = ParallelBroadcastSource::Static(2);
        assert_eq!(source.to_string(), "2");
        assert_eq!(format!("{source:?}"), "Static(2)");
        assert_eq!(ParallelBroadcastSource::from(2usize), source);
        assert_eq!(source, ParallelBroadcastSource::Static(2));
        assert_eq!(hash_of(&source), hash_of(&ParallelBroadcastSource::Static(2)));
        assert_ne!(source, ParallelBroadcastSource::Static(1));
        assert_ne!(source, ParallelBroadcastSource::Dynamic);
        let sources = HashMap::from([(source, 0), (ParallelBroadcastSource::Dynamic, 1)]);
        assert_eq!(sources.get(&ParallelBroadcastSource::Static(2)), Some(&0));
        assert_eq!(sources.get(&ParallelBroadcastSource::Dynamic), Some(&1));
        assert_eq!(sources.get(&ParallelBroadcastSource::Static(1)), None);
        assert_eq!(ParallelBroadcastSource::Dynamic.to_string(), "dynamic");
        assert_eq!(format!("{:?}", ParallelBroadcastSource::Dynamic), "Dynamic");
    }

    #[test]
    fn test_parallel_broadcast() {
        let operation = ParallelBroadcastOperation::new("x".to_string(), 3, 2);
        assert_eq!(operation.name(), PARALLEL_BROADCAST_OPERATION_NAME);
        assert_eq!(operation.axes().single(), Some("x"));
        assert_eq!(operation.axis_size(), 3);
        assert_eq!(operation.source(), ParallelBroadcastSource::Static(2));
        assert_eq!(operation.mesh(), None);
        assert_eq!(operation.to_string(), "parallel_broadcast [axes=\"x\", axis_size=3, source=2]");

        // Several axes render as a tuple in the order that linearizes their combined axis.
        let operation = ParallelBroadcastOperation::new(["y", "x"], 4, 3);
        assert_eq!(operation.axes().names(), &["y".to_string(), "x".to_string()]);
        assert_eq!(operation.to_string(), "parallel_broadcast [axes=(\"y\", \"x\"), axis_size=4, source=3]");
    }

    #[test]
    fn test_parallel_broadcast_dynamic_source() {
        let operation = ParallelBroadcastOperation::new("x", 2, ParallelBroadcastSource::Dynamic);
        assert_eq!(operation.source(), ParallelBroadcastSource::Dynamic);
        assert_eq!(operation.to_string(), "parallel_broadcast [axes=\"x\", axis_size=2, source=dynamic]");
        assert_ne!(operation, ParallelBroadcastOperation::new("x", 2, 0));
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
                error = format!("`{PARALLEL_BROADCAST_OPERATION_NAME}` source 2 is out of bounds for axis size 2"),
            }],
        );
        check_operation_type_inference!(
            operation = ParallelBroadcastOperation::new("x".to_string(), 0, 0),
            cases = [{
                input_types = [vector.clone()],
                error = format!("`{PARALLEL_BROADCAST_OPERATION_NAME}` axis size must be greater than zero"),
            }],
        );
        check_operation_type_inference!(
            operation = ParallelBroadcastOperation::new(Vec::<String>::new(), 1, 0),
            cases = [{
                input_types = [vector.clone()],
                error = format!("`{PARALLEL_BROADCAST_OPERATION_NAME}` requires at least one axis name"),
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
                    error = format!(
                        "`{PARALLEL_BROADCAST_OPERATION_NAME}` input must vary over manual axis `x`; pass an \
                         invariant value through `{PARALLEL_VARY_OPERATION_NAME}` first, because the broadcast output \
                         keeps the input's variation",
                    ),
                },
                {
                    input_types = [vector],
                    error = format!(
                        "`{PARALLEL_BROADCAST_OPERATION_NAME}` input must carry a mesh containing manual axis `x`",
                    ),
                },
                {
                    input_types = [other_varying],
                    error = format!(
                        "`{PARALLEL_BROADCAST_OPERATION_NAME}` input mesh does not match the operation mesh",
                    ),
                },
                {
                    input_types = [unreduced],
                    error = format!("`{PARALLEL_BROADCAST_OPERATION_NAME}` does not support unreduced inputs"),
                },
            ],
        );

        // The mesh axis must be manual, and its size must equal the axis size of the operation.
        let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        check_operation_type_inference!(
            operation = ParallelBroadcastOperation::new("x".to_string(), 2, 1).with_mesh(explicit_mesh),
            cases = [{
                input_types = [varying.clone()],
                error = format!("`{PARALLEL_BROADCAST_OPERATION_NAME}` mesh axis `x` must be manual"),
            }],
        );
        check_operation_type_inference!(
            operation = ParallelBroadcastOperation::new("x".to_string(), 3, 1).with_mesh(mesh),
            cases = [{
                input_types = [varying],
                error = format!(
                    "`{PARALLEL_BROADCAST_OPERATION_NAME}` axis size 3 does not match the size of manual mesh axis `x`",
                ),
            }],
        );
    }

    #[test]
    fn test_parallel_broadcast_type_inference_dynamic_source() {
        let data = ArrayType::new_static(DataType::F32, [3]);
        let source = ArrayType::scalar(DataType::U64);
        check_operation_type_inference!(
            operation = ParallelBroadcastOperation::new("x", 2, ParallelBroadcastSource::Dynamic),
            cases = [
                { input_types = [data.clone(), source], output_types = [data.clone()] },
                { input_types = [data.clone(), ArrayType::scalar(DataType::I32)], error = format!(
                    "`{PARALLEL_BROADCAST_OPERATION_NAME}` runtime source must be a `u64` scalar but got `i32[]`",
                ) },
                { input_types = [data.clone(), ArrayType::new_static(DataType::U64, [1])], error = format!(
                    "`{PARALLEL_BROADCAST_OPERATION_NAME}` runtime source must be a `u64` scalar but got `u64[1]`",
                ) },
                { input_types = [data, ArrayType::scalar(DataType::U64).with_memory(Memory::Host { pinned: true })],
                    error = format!(
                        "`{PARALLEL_BROADCAST_OPERATION_NAME}` runtime source must use the data's memory",
                    ) },
            ],
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
                    error = format!(
                        "`{PARALLEL_BROADCAST_OPERATION_NAME}` input must vary over manual axis `y`; pass an \
                         invariant value through `{PARALLEL_VARY_OPERATION_NAME}` first, because the broadcast output \
                         keeps the input's variation",
                    ),
                },
            ],
        );
        check_operation_type_inference!(
            operation = ParallelBroadcastOperation::new(["x", "y"], 2, 1).with_mesh(mesh.clone()),
            cases = [{
                input_types = [varying.clone()],
                error = format!(
                    "`{PARALLEL_BROADCAST_OPERATION_NAME}` axis size 2 does not match the size 4 of manual mesh \
                     axes (x, y)",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = ParallelBroadcastOperation::new(["x", "x"], 4, 1).with_mesh(mesh),
            cases = [{
                input_types = [varying],
                error = format!("`{PARALLEL_BROADCAST_OPERATION_NAME}` axis names repeat `x`"),
            }],
        );
    }

    #[test]
    fn test_parallel_broadcast_effects() {
        assert_eq!(ParallelBroadcastOperation::new("x", 2, 1).effects().classes(), EffectClasses::NONE);
        assert_eq!(
            ParallelBroadcastOperation::new("x", 2, ParallelBroadcastSource::Dynamic).effects().classes(),
            EffectClasses::single(EffectClass::OrderedAssertion),
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
                message: format!(
                    "cannot interpret `{PARALLEL_BROADCAST_OPERATION_NAME}` over axis `x` of size 2 without an \
                     enclosing binder"
                ),
            }),
        );

        // Even a degenerate broadcast validates its source before interpreting it as the identity.
        assert_eq!(
            context.bind(ParallelBroadcastOperation::new("x".to_string(), 1, 1), Vec::new(), &[input]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PARALLEL_BROADCAST_OPERATION_NAME}` source 1 is out of bounds for axis size 1"
            ),))),
        );
    }

    #[test]
    fn test_parallel_broadcast_interpretation_dynamic_source() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let operation = ParallelBroadcastOperation::new("x", 1, ParallelBroadcastSource::Dynamic);
        let value = Array::scalar(7f32).unwrap();
        assert_eq!(
            context.bind(operation.clone(), Vec::new(), &[value.clone(), Array::scalar(0u64).unwrap()]),
            Ok(vec![value.clone()]),
        );
        let error = context.bind(operation, Vec::new(), &[value, Array::scalar(1u64).unwrap()]).unwrap_err();
        assert!(matches!(error.downcast_custom::<AssertionError>(),
            Some(AssertionError::Failed { message, .. }) if message == &format!(
                "`{PARALLEL_BROADCAST_OPERATION_NAME}` runtime source must be smaller than axis size 1",
            ),
        ));
    }

    #[test]
    fn test_parallel_broadcast_interpretation_dynamic_source_layout() {
        let data_type =
            ArrayType::new_static(DataType::F32, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |(data, source)| {
                Ok(ParallelBroadcastOperation::new("x", 2, ParallelBroadcastSource::Dynamic)
                    .interpret(&data.domain(), &EmptyRegionDriver, &[data, source])?
                    .remove(0))
            },
            (data_type.clone(), ArrayType::scalar(DataType::U64)),
            vec![("x".to_string(), NamedAxis::Batched { size: Some(2) })],
        )
        .unwrap();
        assert_eq!(output_type, data_type);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2][layout=strided{8}], %1:u64[] .
                let %2:u64[] = parallel_reduce [kind=min, axis_name=\"x\"] %1
                    %3:u64[] = parallel_reduce [kind=max, axis_name=\"x\"] %1
                    %4:bool[] = compare [direction=Equal] %2 %3
                    () = assert [
                        message=\"`parallel_broadcast` runtime sources must agree across participants\",
                        labels=[\"minimum\", \"maximum\"],
                    ] %4 %2 %3
                    %5:u64[] = constant [value=2]
                    %6:bool[] = compare [direction=LessThan] %3 %5
                    () = assert [
                        message=\"`parallel_broadcast` runtime source must be smaller than axis size 2\",
                        labels=[\"source\"],
                    ] %6 %1
                    %7:f32[2, 2] = parallel_all_gather [
                        axes=\"x\",
                        axis_size=2,
                        concatenation_axis=0,
                        options=Untiled,
                        output_variance=Varying,
                    ] %0
                    %8:f32[1, 2] = slice [start_indices=[0, 0], limits=[1, 2]] %7
                    %9:f32[2] = reshape [shape=[2]] %8
                    %10:f32[1, 2] = slice [start_indices=[1, 0], limits=[2, 2]] %7
                    %11:f32[2] = reshape [shape=[2]] %10
                    %12:u64[] = constant [value=1]
                    %13:bool[] = compare [direction=Equal] %1 %12
                    %14:f32[2] = select %13 %11 %9
                    %15:f32[2][layout=strided{8}] = broadcast [output_type=f32[2][layout=strided{8}], \
                        output_axes=[0]] %14
                in (%15)"
            }
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
        let expected = indoc! {"
            lambda %0:f32[2] .
            let %1:f32[2] = parallel_broadcast [axes=\"x\", axis_size=2, source=1] %0
            in (%1)"
        };
        assert_eq!(program.to_string(), expected);
        assert_eq!(evaluation.program().to_string(), expected);
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
    fn test_parallel_broadcast_partial_evaluation_dynamic_source() {
        // Known values without an enclosing collective binder still need the runtime collective and its checks.
        let program = runtime_broadcast_program(2);
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(Array::scalar(7f32).unwrap()),
                PartialValue::Known(Array::scalar(1u64).unwrap()),
            ])
            .unwrap();
        let expected = indoc! {"
            lambda %0:f32[], %1:u64[] .
            let %2:f32[] = parallel_broadcast [axes=\"x\", axis_size=2, source=dynamic] %0 %1
            in (%2)"
        };
        assert_eq!(program.to_string(), expected);
        assert_eq!(evaluation.program().to_string(), expected);
        assert_eq!(evaluation.outputs(), &[PartialEvaluationOutput::Unknown(0)]);
    }

    #[test]
    fn test_parallel_broadcast_partial_evaluation_dynamic_unused_result() {
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |(data, source)| {
                let _ = data.dynamic_parallel_broadcast("x", &source)?;
                Ok(data)
            },
            (ArrayType::scalar(DataType::F32), ArrayType::scalar(DataType::U64)),
            vec![("x".to_string(), NamedAxis::Batched { size: Some(1) })],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
            lambda %0:f32[], %1:u64[] .
            let %2:f32[] = parallel_broadcast [axes=\"x\", axis_size=1, source=dynamic] %0 %1
            in (%0)"
            }
        );
        let evaluation = program
            .to_flat_program()
            .partially_evaluate(&[
                PartialValue::Known(Array::scalar(7f32).unwrap()),
                PartialValue::Unknown(ArrayType::scalar(DataType::U64)),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
            lambda %0:u64[], %1:f32[] .
            let %2:f32[] = parallel_broadcast [axes=\"x\", axis_size=1, source=dynamic] %1 %0
            in ()"
            }
        );
        let error = program.interpret((Array::scalar(7f32).unwrap(), Array::scalar(1u64).unwrap())).unwrap_err();
        assert!(matches!(error.downcast_custom::<AssertionError>(),
            Some(AssertionError::Failed { message, .. }) if message == &format!(
                "`{PARALLEL_BROADCAST_OPERATION_NAME}` runtime source must be smaller than axis size 1",
            ),
        ));
    }

    #[test]
    fn test_parallel_broadcast_batching() {
        // A level that binds the axis gives every item a copy of the source item, wherever the mapped axis sits.
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
                message: format!(
                    "`{PARALLEL_BROADCAST_OPERATION_NAME}` over axis `x` resolved axis size 2 but the mapped \
                     batch axis has size 3"
                ),
            }),
        );

        // A broadcast over a manual mesh axis, or over several axes, cannot be consumed by a level that binds one of
        // its axis names, as JAX's batching rule rejects several axes.
        let vector = ArrayBatch::new(Array::vector(vec![1.0, 2.0]).unwrap(), BatchAxis::new(0)).unwrap();
        assert_eq!(
            batch_collective(&swap.clone().with_mesh(manual_mesh()), "x", 2, vector.clone()),
            Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{PARALLEL_BROADCAST_OPERATION_NAME}` over a manual mesh axis cannot bind a named batch axis"
                ),
            }),
        );
        assert_eq!(
            batch_collective(&ParallelBroadcastOperation::new(["x", "y"], 4, 1), "x", 2, vector),
            Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{PARALLEL_BROADCAST_OPERATION_NAME}` over several axes cannot bind a named batch axis"
                ),
            }),
        );

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
    fn test_parallel_broadcast_batching_dynamic_item_shape() {
        let length = Dimension::Dynamic(DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap()));
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), length]));
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |input| {
                batch(
                    |item| item.parallel_broadcast("x", 1),
                    input,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("x"),
                )
                .map_err(ProgramError::from)
            },
            input_type.clone(),
        )
        .unwrap();
        assert_eq!(output_type, input_type);
        assert_eq!(
            program.to_string(),
            indoc! {"
            lambda %0:f32[2, length] .
            let %1:f32[1, length] = slice [start_indices=[1, 0], limits=[2, length]] %0
                %2:f32[2, length] = concatenate [axis=0] %1 %1
            in (%2)"
            }
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
    fn test_parallel_broadcast_differentiation_dynamic_source() {
        // Replaying a primitive uses its recorded axes, without resolving a frontend binder again.
        let program = runtime_broadcast_program(2);
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
            lambda %0:f32[], %1:u64[], %2:f32[] .
            let %3:f32[] = parallel_broadcast [axes=\"x\", axis_size=2, source=dynamic] %0 %1
                %4:f32[] = parallel_broadcast [axes=\"x\", axis_size=2, source=dynamic, source_validated=true] %2 %1
            in (%3, %4)"
            }
        );
        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
            lambda %0:f32[], %1:u64[] .
            let %2:f32[] = parallel_broadcast [axes=\"x\", axis_size=2, source=dynamic] %0 %1
            in (%2, %1)"
            }
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
            lambda %0:f32[], %1:u64[] .
            let %2:f32[] = parallel_broadcast [axes=\"x\", axis_size=2, source=dynamic, source_validated=true] %0 %1
            in (%2)"
            }
        );
        assert_eq!(linearization.primal().effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
        assert_eq!(linearization.tangent().effects().classes(), EffectClasses::NONE);
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
    fn test_parallel_broadcast_transposition_inactive_inputs() {
        let operation = ParallelBroadcastOperation::new("x", 2, 1);
        let input_type = ArrayType::new_static(DataType::F32, [2]);

        // Ignored destinations must not stage otherwise valid communication.
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let mut context = TranspositionContext::new(trace.clone());
        let inputs = [PartialValue::Unknown(input_type.clone())];
        let accumulators = context.cotangent_accumulators(&inputs, &[false]).unwrap();
        let cotangent = context.input(input_type.clone());
        operation
            .transpose(&mut context, &EmptyRegionDriver, &inputs, &[MaybeZero::Value(cotangent)], &accumulators)
            .unwrap();
        assert_eq!(context.take_cotangents(&accumulators).unwrap(), vec![MaybeZero::Zero(input_type.clone())]);
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2] .
                in ()"
            }
        );

        // A structural zero seed needs no communication even when the input cotangent is requested.
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let mut context = TranspositionContext::new(trace.clone());
        let accumulators = context.cotangent_accumulators(&inputs, &[true]).unwrap();
        operation
            .transpose(&mut context, &EmptyRegionDriver, &inputs, &[MaybeZero::Zero(input_type.clone())], &accumulators)
            .unwrap();
        assert_eq!(context.take_cotangents(&accumulators).unwrap(), vec![MaybeZero::Zero(input_type.clone())]);
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(Vec::new(), Vec::new(), Vec::new())
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                in ()"
            }
        );

        // Known primals contribute no input cotangent even with a live output seed.
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let mut context = TranspositionContext::new(trace.clone());
        let inputs = [PartialValue::Known(context.input(input_type.clone()))];
        let accumulators = context.cotangent_accumulators(&inputs, &[]).unwrap();
        let cotangent = context.input(input_type.clone());
        operation
            .transpose(&mut context, &EmptyRegionDriver, &inputs, &[MaybeZero::Value(cotangent)], &accumulators)
            .unwrap();
        assert_eq!(context.take_cotangents(&accumulators).unwrap(), vec![MaybeZero::Zero(input_type)]);
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(Vec::new(), vec![Placeholder; 2], Vec::new())
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[2] .
                in ()"
            }
        );
    }

    #[test]
    fn test_parallel_broadcast_transposition_invalid_source() {
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let mut context = TranspositionContext::new(trace);
        let input_type = ArrayType::scalar(DataType::F32);
        let inputs = [PartialValue::Unknown(input_type.clone())];
        let accumulators = context.cotangent_accumulators(&inputs, &[false]).unwrap();
        assert!(matches!(
            ParallelBroadcastOperation::new("x", 2, 2).transpose(
                &mut context,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Zero(input_type)],
                &accumulators,
            ),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message, .. })))
                if message == format!(
                    "`{PARALLEL_BROADCAST_OPERATION_NAME}` source 2 is out of bounds for axis size 2",
                ),
        ));
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
        let inputs = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
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
            Err::<Array, _>(BatchingError::Type(TypeError::invalid(format!(
                "`{PARALLEL_BROADCAST_OPERATION_NAME}` source 3 is out of bounds for axis size 3"
            ),))),
        );
    }

    #[test]
    fn test_parallel_broadcast_parallel_broadcast_unbound_axis() {
        // A name that no enclosing binder binds fails fast instead of silently acting as the identity.
        assert_eq!(
            batch(
                |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>| {
                    item.parallel_broadcast("y", 1)
                },
                Array::vector(vec![1.0, 2.0, 3.0]).unwrap(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxisSpecification::named("x"),
            ),
            Err::<Array, _>(BatchingError::Axis(AxisError::UnboundAxisName { name: "y".to_string() })),
        );
        // A concrete array is never inside an axis binder, so every axis name is unbound for it, and an empty list of
        // axis names is rejected before any of them is resolved.
        assert_eq!(
            Array::vector(vec![1.0, 2.0]).unwrap().parallel_broadcast("i", 0),
            Err(ProgramError::Axis(AxisError::UnboundAxisName { name: "i".to_string() })),
        );
        assert_eq!(
            Array::vector(vec![1.0, 2.0]).unwrap().parallel_broadcast(Vec::<String>::new(), 0),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PARALLEL_BROADCAST_OPERATION_NAME}` requires at least one axis name"
            )))),
        );
    }

    #[test]
    fn test_parallel_broadcast_parallel_broadcast_manual_mesh_axis() {
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
    }

    #[test]
    fn test_parallel_broadcast_parallel_broadcast_unchecked_manual_mesh_axis() {
        let sharding = Sharding::replicated(manual_mesh(), 0);
        let invariant = ArrayType::scalar(DataType::F32).with_sharding(sharding.clone()).unwrap();
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
    }

    #[test]
    fn test_parallel_broadcast_parallel_broadcast_unreduced_input() {
        let sharding = Sharding::replicated(manual_mesh(), 0);
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
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PARALLEL_BROADCAST_OPERATION_NAME}` does not support unreduced inputs"
            )))),
        );
    }

    #[test]
    fn test_parallel_broadcast_parallel_broadcast_array_ir() {
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
        let vector = Array::vector(vec![1.0f32, 2.0]).unwrap();
        assert_eq!(
            ArrayIrValue::Array(vector).parallel_broadcast("x", 0),
            Err(ProgramError::Axis(AxisError::UnboundAxisName { name: "x".to_string() })),
        );
    }

    #[test]
    fn test_parallel_broadcast_dynamic_parallel_broadcast() {
        let rows = Array::matrix(3, 2, vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(
            broadcast_runtime_source(rows, Array::scalar(1u64).unwrap(), BatchAxis::replicated()),
            Ok(Array::matrix(3, 2, vec![3f32, 4.0, 3.0, 4.0, 3.0, 4.0]).unwrap()),
        );
    }

    #[test]
    fn test_parallel_broadcast_dynamic_parallel_broadcast_axis_index() {
        // The largest participant coordinate elects one agreed root without an integer conversion.
        let output = batch(
            |value: TestTracer| {
                let source = value.domain().axis_index("x")?.parallel_reduce(ReductionKind::Max, "x")?;
                value.dynamic_parallel_broadcast("x", &source)
            },
            Array::vector(vec![1f32, 2.0]).unwrap(),
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("x"),
        );
        assert_eq!(output, Ok(Array::vector(vec![2f32, 2.0]).unwrap()));
    }

    #[test]
    fn test_parallel_broadcast_dynamic_parallel_broadcast_unbound_axis() {
        let value = Array::scalar(1f32).unwrap();
        let source = Array::scalar(0u64).unwrap();
        assert_eq!(
            value.dynamic_parallel_broadcast("x", &source),
            Err(ProgramError::Axis(AxisError::UnboundAxisName { name: "x".to_string() })),
        );
        assert_eq!(
            value.dynamic_parallel_broadcast(Vec::<String>::new(), &source),
            Err(TypeError::invalid(format!("`{PARALLEL_BROADCAST_OPERATION_NAME}` requires at least one axis name",))
                .into()),
        );
    }

    #[test]
    fn test_parallel_broadcast_dynamic_parallel_broadcast_exact_payloads() {
        let nan = f32::from_bits(0x7fc01234);
        let rows = Array::matrix(2, 2, vec![1f32, 2.0, -0.0, nan]).unwrap();
        let output = broadcast_runtime_source(rows, Array::scalar(1u64).unwrap(), BatchAxis::replicated()).unwrap();
        assert_eq!(output.r#type().as_ref(), &ArrayType::new_static(DataType::F32, [2, 2]));
        assert_eq!(
            output.elements::<f32>().unwrap().into_iter().map(f32::to_bits).collect::<Vec<_>>(),
            vec![(-0.0f32).to_bits(), nan.to_bits(), (-0.0f32).to_bits(), nan.to_bits()],
        );
        let rows = Array::vector(vec![u64::MAX - 1, u64::MAX]).unwrap();
        assert_eq!(
            broadcast_runtime_source(rows, Array::scalar(1u64).unwrap(), BatchAxis::replicated()),
            Ok(Array::vector(vec![u64::MAX; 2]).unwrap()),
        );
        let rows = Array::vector(vec![false, true]).unwrap();
        assert_eq!(
            broadcast_runtime_source(rows, Array::scalar(1u64).unwrap(), BatchAxis::replicated()),
            Ok(Array::vector(vec![true; 2]).unwrap()),
        );
        let first = Complex::new(1f32, 2.0);
        let second = Complex::new(-3.0, 4.0);
        assert_eq!(
            broadcast_runtime_source(
                Array::vector(vec![first, second]).unwrap(),
                Array::scalar(1u64).unwrap(),
                BatchAxis::replicated()
            ),
            Ok(Array::vector(vec![second; 2]).unwrap()),
        );
    }

    #[test]
    fn test_parallel_broadcast_dynamic_parallel_broadcast_degenerate_shapes() {
        assert_eq!(
            broadcast_runtime_source(
                Array::vector(vec![5f32]).unwrap(),
                Array::scalar(0u64).unwrap(),
                BatchAxis::replicated()
            ),
            Ok(Array::vector(vec![5f32]).unwrap()),
        );
        assert_eq!(
            broadcast_runtime_source(
                Array::matrix(2, 0, Vec::<f32>::new()).unwrap(),
                Array::scalar(1u64).unwrap(),
                BatchAxis::replicated()
            ),
            Ok(Array::matrix(2, 0, Vec::<f32>::new()).unwrap()),
        );
        assert_eq!(
            batch(
                |(value, source): (TestTracer, TestTracer)| value.dynamic_parallel_broadcast("x", &source),
                (Array::vector(vec![7f32, 8.0]).unwrap(), Array::scalar(1u64).unwrap()),
                (BatchAxis::replicated(), BatchAxis::replicated()),
                BatchAxis::replicated(),
                BatchAxisSpecification::new(2, "x"),
            ),
            Ok(Array::vector(vec![7f32, 8.0]).unwrap()),
        );
    }

    #[test]
    fn test_parallel_broadcast_dynamic_parallel_broadcast_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding =
            Sharding::new(mesh, vec![ShardingDimension::sharded(["devices"]), ShardingDimension::Replicated]).unwrap();
        let rows = Array::matrix(2, 2, vec![1f32, 2.0, 3.0, 4.0]).unwrap().reshard(&sharding).unwrap();
        assert_eq!(
            broadcast_runtime_source(rows, Array::scalar(1u64).unwrap(), BatchAxis::replicated()),
            Ok(Array::matrix(2, 2, vec![3f32, 4.0, 3.0, 4.0]).unwrap().reshard(&sharding).unwrap()),
        );
    }

    #[test]
    fn test_parallel_broadcast_dynamic_parallel_broadcast_strided_items() {
        let row_major = Array::matrix(2, 2, vec![1f32, 2.0, 3.0, 4.0]).unwrap();
        let column_major = row_major.r#type().into_owned().with_layout(Layout::Strided(StridedLayout::new(vec![4, 8])));
        let rows = Array::from_logical_bytes(column_major, &row_major.logical_bytes()).unwrap();
        let output = broadcast_runtime_source(rows, Array::scalar(1u64).unwrap(), BatchAxis::replicated()).unwrap();
        assert_eq!(output.elements::<f32>().unwrap(), vec![3f32, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn test_parallel_broadcast_dynamic_parallel_broadcast_host_memory() {
        let memory = Memory::Host { pinned: true };
        let rows = Array::vector(vec![1f32, 2.0]).unwrap().transfer_to_memory(memory).unwrap();
        assert_eq!(
            broadcast_runtime_source(rows, Array::scalar(1u64).unwrap(), BatchAxis::replicated()),
            Ok(Array::vector(vec![2f32, 2.0]).unwrap().transfer_to_memory(memory).unwrap()),
        );
    }

    #[test]
    fn test_parallel_broadcast_dynamic_parallel_broadcast_invalid_source() {
        let rows = Array::vector(vec![1f32, 2.0]).unwrap();
        let error = ProgramError::from(
            broadcast_runtime_source(rows.clone(), Array::scalar(1u64 << 32).unwrap(), BatchAxis::replicated())
                .unwrap_err(),
        );
        assert!(matches!(error.downcast_custom::<AssertionError>(),
            Some(AssertionError::Failed { message, .. })
                if message == &format!(
                    "`{PARALLEL_BROADCAST_OPERATION_NAME}` runtime source must be smaller than axis size 2",
                ),
        ));
        let error = ProgramError::from(
            broadcast_runtime_source(rows.clone(), Array::scalar(2u64).unwrap(), BatchAxis::replicated()).unwrap_err(),
        );
        assert!(matches!(error.downcast_custom::<AssertionError>(),
            Some(AssertionError::Failed { message, .. })
                if message == &format!(
                    "`{PARALLEL_BROADCAST_OPERATION_NAME}` runtime source must be smaller than axis size 2",
                ),
        ));
        let error = ProgramError::from(
            broadcast_runtime_source(rows.clone(), Array::scalar(u64::MAX).unwrap(), BatchAxis::replicated())
                .unwrap_err(),
        );
        assert!(matches!(error.downcast_custom::<AssertionError>(),
            Some(AssertionError::Failed { message, .. }) if message == &format!(
                "`{PARALLEL_BROADCAST_OPERATION_NAME}` runtime source must be smaller than axis size 2",
            ),
        ));
        assert_eq!(
            broadcast_runtime_source(rows.clone(), Array::scalar(1i32).unwrap(), BatchAxis::replicated())
                .map_err(ProgramError::from),
            Err(TypeError::invalid(format!(
                "`{PARALLEL_BROADCAST_OPERATION_NAME}` runtime source must be a `u64` scalar but got `i32[]`"
            ))
            .into()),
        );
        assert_eq!(
            broadcast_runtime_source(rows, Array::vector(vec![1u64]).unwrap(), BatchAxis::replicated())
                .map_err(ProgramError::from),
            Err(TypeError::invalid(format!(
                "`{PARALLEL_BROADCAST_OPERATION_NAME}` runtime source must be a `u64` scalar but got `u64[1]`",
            ))
            .into()),
        );
    }

    #[test]
    fn test_parallel_broadcast_dynamic_parallel_broadcast_source_reduction_state() {
        let sharding = Sharding::replicated(manual_mesh(), 0);
        let data_type = ArrayType::scalar(DataType::F32)
            .with_sharding(sharding.clone().with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let source_type = ArrayType::scalar(DataType::U64)
            .with_sharding(sharding.clone().with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        assert_eq!(
            TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                |(value, source)| value.dynamic_parallel_broadcast("x", &source),
                (data_type.clone(), source_type),
                manual_axes(true),
            )
            .map(|(output, _)| output),
            Err(TypeError::invalid(format!(
                "`{PARALLEL_BROADCAST_OPERATION_NAME}` runtime source must not carry reduction state",
            ))
            .into()),
        );
        let source_type =
            ArrayType::scalar(DataType::U64).with_sharding(sharding.with_reduced_axes(["x"]).unwrap()).unwrap();
        assert_eq!(
            TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                |(value, source)| value.dynamic_parallel_broadcast("x", &source),
                (data_type, source_type),
                manual_axes(true),
            )
            .map(|(output, _)| output),
            Err(TypeError::invalid(format!(
                "`{PARALLEL_BROADCAST_OPERATION_NAME}` runtime source must not carry reduction state",
            ))
            .into()),
        );
    }

    #[test]
    fn test_parallel_broadcast_dynamic_parallel_broadcast_participant_sources() {
        let rows = Array::vector(vec![1f32, 2.0]).unwrap();
        assert_eq!(
            broadcast_runtime_source(rows.clone(), Array::vector(vec![1u64, 1]).unwrap(), BatchAxis::new(0)),
            Ok(Array::vector(vec![2f32; 2]).unwrap()),
        );
        let error = ProgramError::from(
            broadcast_runtime_source(rows, Array::vector(vec![0u64, 1]).unwrap(), BatchAxis::new(0)).unwrap_err(),
        );
        assert!(matches!(error.downcast_custom::<AssertionError>(),
            Some(AssertionError::Failed { message, .. })
                if message == &format!(
                    "`{PARALLEL_BROADCAST_OPERATION_NAME}` runtime sources must agree across participants",
                ),
        ));
    }

    #[test]
    fn test_parallel_broadcast_dynamic_parallel_broadcast_unrelated_batch_axis() {
        // Each outer batch item is a separate collective group and can select a different source.
        assert_eq!(
            batch(
                |inputs| {
                    batch(
                        |(value, source)| value.dynamic_parallel_broadcast("x", &source),
                        inputs,
                        (BatchAxis::new(0), BatchAxis::replicated()),
                        BatchAxis::new(0),
                        BatchAxisSpecification::named("x"),
                    )
                    .map_err(ProgramError::from)
                },
                (
                    Array::matrix(2, 3, vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
                    Array::vector(vec![0u64, 2]).unwrap(),
                ),
                (BatchAxis::new(0), BatchAxis::new(0)),
                BatchAxis::new(0),
                BatchAxisSpecification::named("groups"),
            ),
            Ok(Array::matrix(2, 3, vec![1f32, 1.0, 1.0, 6.0, 6.0, 6.0]).unwrap()),
        );
    }

    #[test]
    fn test_parallel_broadcast_dynamic_parallel_broadcast_partial_evaluation_unused_result() {
        let data_type = ArrayType::new_static(DataType::F32, [2]);
        let source_type = ArrayType::scalar(DataType::U64);
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(value, source)| {
                let _ = batch(
                    |(value, source)| value.dynamic_parallel_broadcast("x", &source),
                    (value.clone(), source),
                    (BatchAxis::new(0), BatchAxis::replicated()),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("x"),
                )?;
                Ok(value)
            },
            (data_type, source_type.clone()),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:u64[] .
                let %2:bool[] = compare [direction=Equal] %1 %1
                    () = assert [
                        message=\"`parallel_broadcast` runtime sources must agree across participants\",
                        labels=[\"minimum\", \"maximum\"],
                    ] %2 %1 %1
                    %3:u64[] = constant [value=2]
                    %4:bool[] = compare [direction=LessThan] %1 %3
                    () = assert [
                        message=\"`parallel_broadcast` runtime source must be smaller than axis size 2\",
                        labels=[\"source\"],
                    ] %4 %1
                    %5:f32[1] = slice [start_indices=[0], limits=[1]] %0
                    %6:f32[] = reshape [shape=[]] %5
                    %7:f32[1] = slice [start_indices=[1], limits=[2]] %0
                    %8:f32[] = reshape [shape=[]] %7
                    %9:u64[] = constant [value=1]
                    %10:bool[] = compare [direction=Equal] %1 %9
                    %11:f32[] = select %10 %8 %6
                    %12:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %11
                in (%0)"
            }
        );
        let data = Array::vector(vec![1f32, 2.0]).unwrap();
        let evaluation = program
            .to_flat_program()
            .partially_evaluate(&[PartialValue::Known(data.clone()), PartialValue::Unknown(source_type)])
            .unwrap();
        assert_eq!(evaluation.outputs(), &[PartialEvaluationOutput::Known(data)]);
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:u64[], %1:u64[], %2:u64[], %3:f32[], %4:f32[] .
                let %5:bool[] = compare [direction=Equal] %0 %0
                    () = assert [
                        message=\"`parallel_broadcast` runtime sources must agree across participants\",
                        labels=[\"minimum\", \"maximum\"],
                    ] %5 %0 %0
                    %6:bool[] = compare [direction=LessThan] %0 %1
                    () = assert [
                        message=\"`parallel_broadcast` runtime source must be smaller than axis size 2\",
                        labels=[\"source\"],
                    ] %6 %0
                in ()"
            }
        );
        let inputs = evaluation
            .inputs()
            .iter()
            .map(|input| match input {
                PartialEvaluationInput::Known(value) => value.clone(),
                PartialEvaluationInput::Unknown(index) => {
                    assert_eq!(*index, 1);
                    Array::scalar(2u64).unwrap()
                }
            })
            .collect::<Vec<_>>();
        let error = evaluation.program().interpret(inputs).unwrap_err();
        assert!(matches!(error.downcast_custom::<AssertionError>(),
            Some(AssertionError::Failed { message, .. })
                if message == &format!(
                    "`{PARALLEL_BROADCAST_OPERATION_NAME}` runtime source must be smaller than axis size 2",
                ),
        ));
    }

    #[test]
    fn test_parallel_broadcast_dynamic_parallel_broadcast_differentiation() {
        let rows = Array::vector(vec![1f64, 2.0, 3.0]).unwrap();
        let source = Array::scalar(1u64).unwrap();
        let (value, gradient) = differentiate_at((rows, source))
            .value_and_gradient(|inputs| {
                let output = batch(
                    |(value, source)| value.dynamic_parallel_broadcast("x", &source),
                    inputs,
                    (BatchAxis::new(0), BatchAxis::replicated()),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("x"),
                )?;
                output.reduce(&[0], ReductionKind::Sum)
            })
            .unwrap();
        assert_eq!(value, Array::scalar(6f64).unwrap());
        assert_eq!(gradient.0, Array::vector(vec![0f64, 3.0, 0.0]).unwrap());
        assert_eq!(gradient.1.r#type().data_type(), DataType::Zero);
    }

    #[test]
    fn test_parallel_broadcast_dynamic_parallel_broadcast_batched_differentiation() {
        // Differentiation inside the binder executes known collective work in the eager batching parent.
        let result = batch(
            |inputs| {
                let (value, gradient) = differentiate_at(inputs)
                    .value_and_gradient(|(value, source)| value.dynamic_parallel_broadcast("x", &source))?;
                Ok((value, gradient.0))
            },
            (Array::vector(vec![1f64, 2.0]).unwrap(), Array::scalar(1u64).unwrap()),
            (BatchAxis::new(0), BatchAxis::replicated()),
            (BatchAxis::new(0), BatchAxis::new(0)),
            BatchAxisSpecification::named("x"),
        );
        assert_eq!(result, Ok((Array::vector(vec![2f64, 2.0]).unwrap(), Array::vector(vec![0f64, 2.0]).unwrap(),)));
    }

    #[test]
    fn test_parallel_broadcast_dynamic_parallel_broadcast_array_ir_manual_axis() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let data_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let source_type =
            ArrayType::scalar(DataType::U64).with_sharding(Sharding::replicated(mesh.clone(), 0)).unwrap();
        let (output, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
            |(value, source)| value.dynamic_parallel_broadcast("x", &source),
            (ArrayIrType::from(data_type.clone()), ArrayIrType::from(source_type)),
            vec![("x".to_string(), NamedAxis::Mesh { mesh, axis: 0, size: 2, checks_manual_variation: true })],
        )
        .unwrap();
        assert_eq!(output, ArrayIrType::from(data_type));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}], \
                    %1:u64[][sharding={mesh<['x'=2:manual]>, []}] .
                let %2:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = parallel_broadcast \
                    [axes=\"x\", axis_size=2, source=dynamic, mesh=['x'=2:manual]] %0 %1
                in (%2)"
            }
        );
    }

    #[test]
    fn test_parallel_broadcast_dynamic_parallel_broadcast_array_ir_non_array() {
        let dimension = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(1).unwrap());
        assert_eq!(
            dimension.dynamic_parallel_broadcast("x", &dimension),
            Err(TypeError::invalid("expected array type but got dimension type").into()),
        );
    }

    #[test]
    fn test_parallel_broadcast_dynamic_parallel_broadcast_array_ir_dynamic_items() {
        let length = Dimension::Dynamic(DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap()));
        let data_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), length]));
        let (output_type, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |inputs| {
                batch(
                    |(value, source)| value.dynamic_parallel_broadcast("x", &source),
                    inputs,
                    (BatchAxis::new(0), BatchAxis::replicated()),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("x"),
                )
                .map_err(ProgramError::from)
            },
            (ArrayIrType::from(data_type.clone()), ArrayIrType::from(ArrayType::scalar(DataType::U64))),
        )
        .unwrap();
        assert_eq!(output_type, ArrayIrType::from(data_type));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, length], %1:u64[] .
                let %2:dimension<2> = constant [value=2]
                    %3:bool[] = compare [direction=Equal] %1 %1
                    () = assert [
                        message=\"`parallel_broadcast` runtime sources must agree across participants\",
                        labels=[\"minimum\", \"maximum\"],
                    ] %3 %1 %1
                    %4:u64[] = constant [value=2]
                    %5:bool[] = compare [direction=LessThan] %1 %4
                    () = assert [
                        message=\"`parallel_broadcast` runtime source must be smaller than axis size 2\",
                        labels=[\"source\"],
                    ] %5 %1
                    %6:f32[1, length] = slice [start_indices=[0, 0], limits=[1, length]] %0
                    %7:f32[length] = reshape [shape=[length]] %6
                    %8:f32[1, length] = slice [start_indices=[1, 0], limits=[2, length]] %0
                    %9:f32[length] = reshape [shape=[length]] %8
                    %10:u64[] = constant [value=1]
                    %11:bool[] = compare [direction=Equal] %1 %10
                    %12:f32[length] = select %11 %9 %7
                    %13:dimension<length ∈ [0, 4)> = dimension_size [axis=0] %12
                    %14:f32[2, length] = broadcast [output_axes=[1]] %12 %2 %13
                in (%14)"
            }
        );
        assert_eq!(
            program.interpret((
                ArrayIrValue::Array(Array::matrix(2, 3, vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()),
                ArrayIrValue::Array(Array::scalar(1u64).unwrap()),
            )),
            Ok(ArrayIrValue::Array(Array::matrix(2, 3, vec![4f32, 5.0, 6.0, 4.0, 5.0, 6.0]).unwrap())),
        );
        assert_eq!(
            program.interpret((
                ArrayIrValue::Array(Array::matrix(2, 0, Vec::<f32>::new()).unwrap()),
                ArrayIrValue::Array(Array::scalar(1u64).unwrap()),
            )),
            Ok(ArrayIrValue::Array(Array::matrix(2, 0, Vec::<f32>::new()).unwrap())),
        );
    }
}
