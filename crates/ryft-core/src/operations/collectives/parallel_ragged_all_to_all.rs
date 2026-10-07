use std::fmt::Display;

use ryft_macros::capability;

use crate::arrays::{
    Array, ArrayAddressing, ArrayBatch, ArrayBatchingPolicy, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrType,
    ArrayType, DataType, Dimension, DimensionVariable, DynamicArrayExtentBatchingPolicy, LogicalMesh, Shape, Sharding,
    ShardingDimension,
};
use crate::axes::{AxisError, NamedAxes, NamedAxis};
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    MemberBatchableOperation, batch_projected_operation,
};
use crate::contexts::{Context, Domain, DomainProjection, ProjectedContext, ValueResolution};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDriver, DifferentiationDual,
    DifferentiationError, DifferentiationPolicy, MemberDifferentiableOperation, jvp_projected_operation,
};
use crate::interpretation::{
    InterpretableOperation, InterpretationDriver, MemberInterpretableOperation, interpret_projected_operation,
};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::Capability;
use crate::operations::arithmetic::{Add, AddOperation, Mul, MulOperation, Neg, NegOperation};
use crate::operations::collectives::parallel_all_to_all::ParallelAllToAllOperation;
use crate::operations::collectives::parallel_vary::{
    ManualVariationAlignment, PARALLEL_VARY_OPERATION_NAME, ParallelVary, ParallelVaryOperation,
};
use crate::operations::collectives::{
    CollectiveArrayExtentBatchingPolicy, CollectiveOptions, effective_collective_axis_size, resolve_named_axis_size,
    validate_manual_mesh_input,
};
use crate::operations::comparisons::{Compare, CompareOperation};
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::constants::iota::IotaOperation;
use crate::operations::constants::one::{One, OneOperation};
use crate::operations::constants::zero::{Zero, ZeroOperation};
use crate::operations::control_flow::select::{Select, SelectOperation};
use crate::operations::cumulative::{Cumulative, CumulativeOperation};
use crate::operations::manipulation::broadcasting::{Broadcast, BroadcastOperation};
use crate::operations::manipulation::concatenation::{Concatenate, ConcatenateOperation};
use crate::operations::manipulation::conversions::{ConvertElementType, ConvertElementTypeOperation};
use crate::operations::manipulation::memory::{TransferToMemory, TransferToMemoryOperation};
use crate::operations::manipulation::reshaping::{Reshape, ReshapeOperation};
use crate::operations::manipulation::scattering::{
    Scatter, ScatterDimensionNumbers, ScatterOperation, ScatterOptions, ScatterReductionKind,
};
use crate::operations::manipulation::slicing::{Slice, SliceOperation};
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    MaybeZero, MemberOperation, Operation, OperationFormatter, OperationProjection, OperationProvider, ProgramError,
    ProvenanceScope, RegionInterface, TypeError, TypeIdentityRenaming, Typed, Value, ValueProjection,
    infer_projected_operation_output_types, infer_projected_operation_region_input_types,
};
use crate::tracing::{Tracer, TracingContext};

/// Input representation carried by [`ParallelRaggedAllToAllOperation`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum ParallelRaggedAllToAllRepresentation {
    /// Public per-participant representation with rank-one metadata inputs.
    Logical,

    /// Batching-internal representation with one leading participant axis on every input.
    Physical,
}

/// Update semantics used by the public forward exchange and by the internal adjoint exchange that computes the
/// `operand` cotangent.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum ParallelRaggedAllToAllUpdateKind {
    /// Received segments replace the corresponding output seed regions.
    Overwrite,

    /// Received segments are added into the corresponding output seed regions.
    Add,
}

/// Canonical name of the [`ParallelRaggedAllToAllOperation`].
pub const PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME: &str = "parallel_ragged_all_to_all";

/// [`Operation`] that exchanges variable-length leading-axis segments between participants of a named axis.
///
/// The six inputs, in order, are `operand (N, A, ...)`, `output (M, A, ...)`, and rank-one integer arrays
/// `input_offsets`, `send_sizes`, `output_offsets`, and `receive_sizes`, each of length `K`, which must be positive and
/// divisible by the effective participant-group size. The result has exactly `output`'s type and starts with `output`'s
/// value, so elements outside received regions pass through unchanged. The batching rule uses one internal physical
/// form that prefixes every input with the participant axis, making the data inputs `(P, N, A, ...)` and
/// `(P, M, A, ...)` and the metadata inputs `(P, K)`. This representation is normalized back to the public
/// contract during type inference and eager interpretation.
///
/// `output_offsets` are supplied by each sender but are expressed in the corresponding receiver's coordinate frame.
/// Runtime metadata must satisfy that `receive_sizes` equals the tiled exchange of `send_sizes` within the same
/// participant groups (e.g., `send_sizes.parallel_all_to_all_tiled(axis_name, 0, 0)` for an ungrouped exchange; refer
/// to the documentation of [`ParallelAllToAll`](crate::operations::collectives::ParallelAllToAll) for more
/// information), that every source and destination region is in bounds, and that received regions within one output
/// are disjoint. Send regions may overlap, which intentionally permits resending a source slice. Concrete eager
/// execution validates these conditions with overflow-safe host arithmetic. Staged XLA execution treats them as
/// preconditions, as does JAX's
/// [`jax.lax.ragged_all_to_all`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_all_to_all.html).
///
/// An exchange over a manual mesh axis is created by [`with_mesh`](Self::with_mesh), and
/// [`ParallelRaggedAllToAll::parallel_ragged_all_to_all`] supplies the mesh automatically from the enclosing manual
/// region. Such an exchange can give the receivers different segments, so every input must vary over the axis (refer
/// to [`ParallelVary`]) and the output varies over it too. A pending sum over that axis is rejected. In both forms,
/// every device computes its result from all six inputs, so the inputs follow the standard variation rule of
/// [`ArrayType::check_matching_manual_variation`]. The result inherits the reduction state of `operand` and `output`,
/// which must be identical, and the metadata must be invariant over the corresponding reduction axes so that every
/// shard routes the same way. JAX instead types the result as `output` without inserting any variation, which types
/// the segments that an invariant seed receives as invariant. An ordinary exchange carries no mesh, because a `batch`
/// level whose axis name shadows a manual mesh axis may bind it instead, and a matching `batch` level rejects an
/// exchange over a manual mesh axis.
///
/// [`RaggedAxis`](crate::RaggedAxis) is batching-time metadata and does not participate in this explicitly packed
/// operation contract. Batching rejects inputs that carry it because one per-item logical extent does not determine
/// the per-source/per-destination sizes and two coordinate-frame offset vectors required by this operation. A future
/// adapter must therefore accept an explicit routing descriptor; it cannot infer routing from
/// [`RaggedAxis`](crate::RaggedAxis) alone. The two representations also describe different frames:
/// `ParallelRaggedAllToAllOperation` metadata describes participant chunks, whereas a `RaggedAxis` describes packed
/// batch items and has no carrier outside a batching transform. A batch transform whose named axis matches this
/// operation executes concrete array metadata eagerly. Unresolved non-constant metadata are deliberately gated:
/// dynamic-length copies cannot use [`DynamicSliceOperation`](crate::DynamicSliceOperation), whose slice sizes are
/// static payload fields. A future staged implementation can express the copies with iota-based index arithmetic,
/// gather, and `select` masking at `O(group_size × M)` staged work.
///
/// The transpose stages additional collectives rather than performing a local rewrite, and its metadata inputs are
/// primal residuals: ordinary runtime values retained by linearization, not compile-time constants. Participant groups
/// and the manual mesh are forwarded through both dense metadata exchanges and the adjoint ragged exchange. This
/// deliberately corrects JAX's grouped transpose, which accepts groups on the ragged primitive but omits them from its
/// offset exchanges. The output leading dimension must currently be static so the `M + 1` marker and its final slice
/// can be represented by the existing static scatter and slice operations.
///
/// Batching over an unrelated mapped axis currently requires a static mapped extent and statically shaped data inputs.
/// Offset rebasing stages `N` and `M` as scalar constants, and the shared reshape interface cannot yet recover dynamic
/// trailing extents from both homogeneous and composite carriers. Participant groups are supported in this case,
/// because the merged metadata keep the block of every receiver contiguous for any group size.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParallelRaggedAllToAllOperation {
    /// Axis name referenced by this collective.
    axis_name: String,

    /// Full number of participants along the named axis, resolved when the operation is staged.
    axis_size: usize,

    /// Optional ordered partition of logical participant indices.
    axis_index_groups: Option<Vec<Vec<usize>>>,

    /// Public logical or batching-internal physical input representation.
    representation: ParallelRaggedAllToAllRepresentation,

    /// Public overwrite or transpose-internal additive update semantics.
    update_kind: ParallelRaggedAllToAllUpdateKind,

    /// Refer to the documentation of [`mesh`](Self::mesh) for more information.
    mesh: Option<LogicalMesh>,
}

impl ParallelRaggedAllToAllOperation {
    /// Creates an ungrouped operation over the named axis with the provided resolved axis size.
    #[inline]
    pub fn new(axis_name: String, axis_size: usize) -> Self {
        Self {
            axis_name,
            axis_size,
            axis_index_groups: None,
            representation: ParallelRaggedAllToAllRepresentation::Logical,
            update_kind: ParallelRaggedAllToAllUpdateKind::Overwrite,
            mesh: None,
        }
    }

    /// Creates a grouped operation after validating that `axis_index_groups` is an equal-sized exact partition
    /// of `0..axis_size`.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of the axis over which segments are exchanged.
    ///   - `axis_size`: Full number of participants along `axis_name`, across all groups.
    ///   - `axis_index_groups`: Participant groups, each listing axis indices in `0..axis_size`. Every participant
    ///     exchanges segments only with the members of its own group. The order within a group defines the position
    ///     of each member, and the metadata address peers by that position rather than by axis index: with groups of
    ///     size `G` and metadata of length `K`, the `K / G` entries starting at `p · K / G` describe the segments sent
    ///     to, or received from, the member at position `p`. For example, with groups `[[0, 2], [3, 1]]`, the second
    ///     block of metadata of participant `3` describes its exchange with participant `1`.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] if `axis_size` is zero, if there are no groups, if the groups are empty or differ
    /// in size, or if they repeat, omit, or exceed a participant of `0..axis_size`.
    #[inline]
    pub fn grouped(axis_name: String, axis_size: usize, axis_index_groups: Vec<Vec<usize>>) -> Result<Self, TypeError> {
        effective_collective_axis_size(
            PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME,
            axis_size,
            Some(axis_index_groups.as_slice()),
        )?;
        Ok(Self { axis_index_groups: Some(axis_index_groups), ..Self::new(axis_name, axis_size) })
    }

    /// Returns this [`ParallelRaggedAllToAllOperation`] configured to exchange segments over a manual axis of `mesh`.
    /// Every input must carry that mesh and vary over [`axis_name`](Self::axis_name), whose size on the mesh must equal
    /// [`axis_size`](Self::axis_size), and no input may carry a pending cross-device sum over that axis. Type inference
    /// validates these requirements. [`ParallelRaggedAllToAll::parallel_ragged_all_to_all`] supplies the mesh
    /// automatically from the enclosing manual region.
    #[inline]
    pub fn with_mesh(mut self, mesh: LogicalMesh) -> Self {
        self.mesh = Some(mesh);
        self
    }

    /// Returns a clone that uses the batching-internal physical input representation, in which every input carries
    /// a leading participant axis whose extent is [`axis_size`](Self::axis_size). Row `p` of each input holds the
    /// input of the participant with axis index `p`, so the data inputs are `(P, N, A, ...)` and `(P, M, A, ...)`,
    /// the metadata inputs are `(P, K)`, and the result is `(P, M, A, ...)`. Type inference validates and strips the
    /// leading axis before applying the logical contract, and eager interpretation then executes the exchanges of all
    /// participants in one host-side evaluation without an enclosing binder. The clone keeps the axis name, groups,
    /// and update semantics of this operation, and it renders with `representation=Physical`.
    ///
    /// This function is private because only the batching rule of a level that binds [`axis_name`](Self::axis_name)
    /// can establish that the leading axis enumerates the participants of that axis. That level consumes the axis, so
    /// every other level treats the physical operation as unrelated to its own axis. A physical operation has no
    /// backend lowering (refer to the documentation of [`is_physical`](Self::is_physical)).
    #[inline]
    fn with_physical_representation(&self) -> Self {
        Self { representation: ParallelRaggedAllToAllRepresentation::Physical, ..self.clone() }
    }

    /// Returns a clone whose received segments are added into the output seed instead of overwriting it. Unlike the
    /// overwriting exchange, the additive exchange accepts overlapping received regions and sums every contribution
    /// to a region. The clone keeps the axis name, groups, and input representation of this operation, and it
    /// renders with `update_kind=Add`.
    ///
    /// This function is private because only the transpose rule creates additive exchanges, as the adjoint of the
    /// `operand` input: that adjoint sends the output cotangent back into a zero `operand` seed, and the forward
    /// exchange may send one `operand` region to several destinations, whose cotangents must all be summed. The
    /// additive exchange is linear in both data inputs, and transposing it passes the output cotangent through
    /// unchanged. Backends need a dedicated lowering for it (refer to the documentation of
    /// [`accumulates_updates`](Self::accumulates_updates)).
    #[inline]
    fn with_additive_updates(&self) -> Self {
        Self { update_kind: ParallelRaggedAllToAllUpdateKind::Add, ..self.clone() }
    }

    /// Returns the axis name referenced by this collective.
    #[inline]
    pub fn axis_name(&self) -> &str {
        &self.axis_name
    }

    /// Returns the full number of participants along the named axis.
    #[inline]
    pub fn axis_size(&self) -> usize {
        self.axis_size
    }

    /// Validates the participant partition and returns its common group size, which is the number of participants that
    /// exchange segments with each other: the group size for a grouped exchange and the axis size otherwise.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] if the axis size is zero or if the participant groups are not an equal-sized exact
    /// partition of `0..axis_size`.
    #[inline]
    pub fn effective_axis_size(&self) -> Result<usize, TypeError> {
        effective_collective_axis_size(
            PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME,
            self.axis_size,
            self.axis_index_groups(),
        )
    }

    /// Returns the ordered participant groups, if any.
    #[inline]
    pub fn axis_index_groups(&self) -> Option<&[Vec<usize>]> {
        self.axis_index_groups.as_deref()
    }

    /// Returns the logical mesh whose manual axis this [`ParallelRaggedAllToAllOperation`] exchanges segments over, or
    /// [`None`] for an ordinary exchange, whose named axis may be bound by any enclosing binder. Only an exchange over
    /// a manual mesh axis requires its inputs to vary over that axis.
    #[inline]
    pub fn mesh(&self) -> Option<&LogicalMesh> {
        self.mesh.as_ref()
    }

    /// Returns whether this operation carries the batching-internal physical input representation, in which every input
    /// has one leading axis that enumerates the participants of the named axis. Batching over a named axis produces
    /// this representation so that one host-side evaluation can exchange segments between all participants.
    ///
    /// Backend lowering must check this predicate and reject physical operations, because a device-level
    /// `parallel_ragged_all_to_all` expects the public logical representation, in which each device holds only its own
    /// data inputs and rank-one metadata inputs. Lowering a physical operation as if it were logical would misread its
    /// leading participant axis as data.
    #[inline]
    pub fn is_physical(&self) -> bool {
        self.representation == ParallelRaggedAllToAllRepresentation::Physical
    }

    /// Returns whether received segments add into the output seed instead of overwriting it.
    ///
    /// Operations constructed through the public API always overwrite. Only the transpose rule produces accumulating
    /// operations: its adjoint exchange sends the output cotangent back into a zero seed shaped like the `operand`
    /// array, and `operand` regions that several forward segments read must sum the cotangents of all those segments.
    /// Backend lowering must check this predicate, because a native `ragged_all_to_all` overwrites its output and
    /// would keep only one of those contributions, so accumulating operations need a lowering that adds received
    /// segments explicitly.
    #[inline]
    pub fn accumulates_updates(&self) -> bool {
        self.update_kind == ParallelRaggedAllToAllUpdateKind::Add
    }

    /// Returns the participant groups that exchange segments (i.e., the ordered participant groups of a grouped
    /// exchange, or one group of all `axis_size` participants otherwise).
    #[inline]
    fn participant_groups(&self) -> Vec<Vec<usize>> {
        self.axis_index_groups().map_or_else(|| vec![(0..self.axis_size).collect()], <[Vec<usize>]>::to_vec)
    }

    /// Binds this [`ParallelRaggedAllToAllOperation`] to `inputs`, given in their canonical order, through the dispatch
    /// domain of the first input. Over a manual mesh axis, the operation records the mesh of that axis, and a first
    /// input that does not vary over the axis is made varying first. All inputs are then aligned to the manual axes
    /// that any of them varies over, so that the output type records every manual axis over which the result can
    /// differ.
    fn stage<
        V: Value<
                Type = ArrayType,
                DispatchDomain: Context<Operation: From<ParallelRaggedAllToAllOperation>> + NamedAxes,
            > + ParallelVary,
    >(
        mut self,
        inputs: [&V; 6],
    ) -> Result<V, ProgramError> {
        let context = inputs[0].dispatch_domain();
        let mut inputs = inputs.map(Clone::clone);
        let axis_name = self.axis_name.clone();
        if let Some(NamedAxis::Mesh { mesh, .. }) = context.named_axis(&axis_name) {
            if inputs.iter().any(|input| input.r#type().unreduced_axes().contains(axis_name.as_str())) {
                return Err(TypeError::invalid(format!(
                    "`{PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME}` does not support unreduced inputs",
                ))
                .into());
            }
            if !inputs[0]
                .r#type()
                .sharding()
                .is_some_and(|sharding| sharding.varying_manual_axes().contains(&axis_name))
            {
                inputs[0] = inputs[0].parallel_vary(&axis_name)?;
            }
            self = self.with_mesh(mesh);
        }
        let inputs = V::align_manual_variation(&inputs)?;
        let mut outputs = context.bind(self, Vec::new(), &inputs)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }

    /// Transposes sender-owned offset metadata of this [`ParallelRaggedAllToAllOperation`] in its current logical or
    /// physical representation, staging the transposition in `context`. The result gives every receiver the offsets
    /// that each sender in its participant group addressed to it, so that the adjoint exchange can route cotangents
    /// back along the forward routes.
    fn transpose_offsets<
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType>
            + From<ParallelAllToAllOperation>
            + From<ConcatenateOperation<ArrayType>>
            + From<SliceOperation>
            + OperationProvider<ArrayType, ParallelVaryOperation, Operation = O>
            + OperationProvider<ArrayType, BroadcastOperation, Operation = O>,
    >(
        &self,
        context: &mut TracingContext<V, O>,
        offsets: &Tracer<TracingContext<V, O>>,
    ) -> Result<Tracer<TracingContext<V, O>>, DifferentiationError> {
        if self.is_physical() {
            // Physical batching has already materialized every participant, so no named-axis binder remains in
            // which a dense collective could run. Row `p` of the result instead concatenates, for every sender in the
            // participant group of `p`, the block of that sender's offsets addressed to the position of `p` in that
            // group, using only static slices and concatenations.
            let offset_type = offsets.r#type();
            let metadata_length = offset_type.shape().dimensions()[1].value().unwrap();
            let group_size = self.effective_axis_size()?;
            let slices_per_peer = metadata_length / group_size;
            let groups = self.participant_groups();
            let mut rows = Vec::with_capacity(self.axis_size);
            for participant in 0..self.axis_size {
                let (group, participant_position) = groups
                    .iter()
                    .find_map(|group| {
                        group.iter().position(|candidate| *candidate == participant).map(|position| (group, position))
                    })
                    .unwrap();
                let start = participant_position * slices_per_peer;
                let mut blocks = Vec::with_capacity(group_size);
                for &sender in group {
                    blocks.push(offsets.slice(&[sender, start], &[sender + 1, start + slices_per_peer], &[1, 1])?);
                }
                rows.push(Tracer::concatenate(blocks.iter(), 1)?);
            }
            return Ok(Tracer::concatenate(rows.iter(), 0)?);
        }

        // The logical representation exchanges the offsets through a tiled all-to-all over the same participant groups
        // and, for an exchange over a manual mesh axis, over the same mesh, whose variation contract the offsets
        // already satisfy as inputs of this operation.
        let options = self.axis_index_groups().map_or_else(CollectiveOptions::tiled, |groups| {
            CollectiveOptions::tiled().with_axis_index_groups(groups.to_vec())
        });
        let mut exchange = ParallelAllToAllOperation::new(self.axis_name.clone(), self.axis_size, 0, 0, options);
        if let Some(mesh) = &self.mesh {
            exchange = exchange.with_mesh(mesh.clone());
        }
        let mut outputs = context.bind(exchange, Vec::new(), std::slice::from_ref(offsets))?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl Display for ParallelRaggedAllToAllOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ParallelRaggedAllToAllOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("region", region_interfaces, 0, TypeError);
        check_count!("input", input_types, 6, TypeError);
        let effective_axis_size = self.effective_axis_size()?;
        let result_type = input_types[1].clone();
        let physical = self.is_physical();
        let normalized_input_types = if physical {
            Some(
                input_types
                    .iter()
                    .enumerate()
                    .map(|(index, input_type)| {
                        let Some(participant_extent) =
                            input_type.shape().dimensions().first().and_then(|extent| extent.value())
                        else {
                            return Err(TypeError::invalid(format!(
                                "`{PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME}` physical input {index} must have a \
                                 static leading participant dimension",
                            )));
                        };
                        if participant_extent != self.axis_size {
                            return Err(TypeError::invalid(format!(
                                "`{}` physical input {} leading participant dimension {} must equal axis size {}",
                                PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME, index, participant_extent, self.axis_size,
                            )));
                        }
                        Ok(input_type.without_dimension(0)?.0)
                    })
                    .collect::<Result<Vec<_>, TypeError>>()?,
            )
        } else {
            None
        };

        let input_types = normalized_input_types.as_deref().unwrap_or(input_types);
        let [operand, output, input_offsets, send_sizes, output_offsets, receive_sizes] = input_types else {
            unreachable!();
        };

        if operand.rank() == 0 || output.rank() == 0 {
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME}` data inputs must have rank at least 1 but got \
                 `{operand}` and `{output}`",
            )));
        }

        if operand.data_type() != output.data_type() {
            return Err(TypeError::invalid(format!(
                "`{}` `operand` and `output` data types must match but got \
                 `{}` and `{}`",
                PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME,
                operand.data_type(),
                output.data_type(),
            )));
        }

        if operand.shape().dimensions()[1..] != output.shape().dimensions()[1..] {
            return Err(TypeError::invalid(format!(
                "`{}` `operand` and `output` trailing dimensions must match but got `{}` and `{}`",
                PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME,
                operand.shape(),
                output.shape(),
            )));
        }

        let metadata = [
            ("input_offsets", input_offsets),
            ("send_sizes", send_sizes),
            ("output_offsets", output_offsets),
            ("receive_sizes", receive_sizes),
        ];
        let metadata_data_type = input_offsets.data_type();
        let mut metadata_length = None;
        for (name, r#type) in metadata {
            if r#type.rank() != 1 {
                return Err(TypeError::invalid(format!(
                    "`{}` `{}` must be rank 1 but got `{}`",
                    PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME, name, r#type,
                )));
            }

            if !r#type.data_type().is_integer() {
                return Err(TypeError::invalid(format!(
                    "`{}` `{}` must have an integer data type but got `{}`",
                    PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME,
                    name,
                    r#type.data_type(),
                )));
            }

            if r#type.data_type() != metadata_data_type {
                return Err(TypeError::invalid(format!(
                    "`{}` metadata inputs must share one integer data type but `input_offsets` \
                     has `{}` and `{}` has `{}`",
                    PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME,
                    metadata_data_type,
                    name,
                    r#type.data_type(),
                )));
            }

            let Some(length) = r#type.shape().dimensions()[0].value() else {
                return Err(TypeError::invalid(format!(
                    "`{}` `{}` must have a static length but got `{}`",
                    PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME, name, r#type,
                )));
            };

            match metadata_length {
                Some(expected) if length != expected => {
                    return Err(TypeError::invalid(format!(
                        "`{PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME}` metadata inputs must have equal lengths but \
                         `input_offsets` has length {expected} and `{name}` has length {length}",
                    )));
                }
                None => metadata_length = Some(length),
                _ => {}
            }
        }

        let metadata_length = metadata_length.unwrap();
        if metadata_length == 0 {
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME}` metadata length must be greater than zero",
            )));
        }
        if metadata_length % effective_axis_size != 0 {
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME}` metadata length {metadata_length} is not divisible by \
                 group size {effective_axis_size}",
            )));
        }

        // An exchange over a manual mesh axis can give the receivers different values, so every input must vary over
        // the axis, and exchanging segments cannot complete a pending sum over that same axis.
        let inputs = [operand, output, input_offsets, send_sizes, output_offsets, receive_sizes];
        if let Some(mesh) = &self.mesh {
            let axis_name = self.axis_name.as_str();
            for input in inputs {
                validate_manual_mesh_input(
                    PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME,
                    axis_name,
                    Some(self.axis_size),
                    mesh,
                    input,
                )?;
                let sharding = input.sharding().unwrap();
                if sharding.unreduced_axes().contains(axis_name) {
                    return Err(TypeError::invalid(format!(
                        "`{PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME}` does not support unreduced inputs",
                    )));
                }
                if !sharding.varying_manual_axes().contains(axis_name) {
                    return Err(TypeError::invalid(format!(
                        "`{PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME}` inputs must vary over manual axis \
                         `{axis_name}`; pass invariant values through `{PARALLEL_VARY_OPERATION_NAME}` first so that \
                         the exchanged output is typed as varying",
                    )));
                }
            }
        }

        // Every device computes its result from all six of its inputs, so they follow the standard variation rule of
        // an ordinary operation, and the result, which has the type of `output`, varies over exactly their axes.
        ArrayType::check_matching_manual_variation(PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME, &inputs)?;

        // For fixed metadata, the exchange is jointly linear in `operand` and `output`. It therefore commutes with a
        // pending reduction only when both data inputs carry the same reduction state, which the result inherits, and
        // when every shard routes the same way, i.e., when no metadata input is unreduced, varies, or is sharded over
        // the reduction axes of the data inputs.
        let reduction_state = |r#type: &ArrayType| {
            r#type
                .sharding()
                .map(|sharding| (sharding.unreduced_axes().clone(), sharding.reduced_axes().clone()))
                .unwrap_or_default()
        };
        let (unreduced, reduced) = reduction_state(output);
        if reduction_state(operand) != (unreduced.clone(), reduced.clone()) {
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME}` `operand` and `output` must carry identical reduction \
                 state",
            )));
        }

        for (name, r#type) in metadata {
            let Some(sharding) = r#type.sharding() else {
                continue;
            };

            if !sharding.unreduced_axes().is_empty() {
                return Err(TypeError::invalid(format!(
                    "`{PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME}` `{name}` must not carry unreduced state",
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
                        "`{PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME}` `{name}` must be invariant over the reduction \
                         axes of `operand` and `output`",
                    )));
                }
            }
        }

        Ok(vec![result_type])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME)?.bracketed(
            |operation| {
                operation.field("axis_name", format_args!("{:?}", self.axis_name))?;
                operation.field("axis_size", self.axis_size)?;
                if let Some(axis_index_groups) = &self.axis_index_groups {
                    operation.field("axis_index_groups", format_args!("{axis_index_groups:?}"))?;
                }
                if self.is_physical() {
                    operation.field("representation", "Physical")?;
                }
                if self.accumulates_updates() {
                    operation.field("update_kind", "Add")?;
                }
                if let Some(mesh) = &self.mesh {
                    operation.field("mesh", mesh)?;
                }
                Ok(())
            },
        )
    }
}

impl<C: Domain<Type = ArrayType, Value = Array>> InterpretableOperation<C> for ParallelRaggedAllToAllOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 6, ProgramError);
        let [operand, output, input_offsets, send_sizes, output_offsets, receive_sizes] = inputs else {
            // This is unreachable because of the preceding `check_count!`.
            unreachable!();
        };

        let physical = self.is_physical();
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.infer_output_types(input_types.as_slice(), &[])?;

        // Outside a binder, only an exchange whose participant groups are singletons is degenerate: every participant
        // then exchanges segments only with itself, whatever the size of the full axis.
        let effective_axis_size = self.effective_axis_size()?;
        if !physical && effective_axis_size != 1 {
            let groups = match self.axis_index_groups() {
                Some(_) => format!(" with participant groups of size {effective_axis_size}"),
                None => String::new(),
            };

            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "cannot interpret `{}` over axis `{}` of size {}{} \
                     without an enclosing binder",
                    PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME, self.axis_name, self.axis_size, groups,
                ),
            });
        }

        // Execute the exchange on the host. The public `ParallelRaggedAllToAll` capability cannot reach this point
        // for an `Array`, which is never inside an axis binder, so only the degenerate exchange above and the
        // batching-internal physical representation, in which every input carries one leading participant axis,
        // are evaluated here. Inference has validated the input types, but not the metadata values: they must be
        // non-negative, every source and destination region must be in bounds, every send size must equal the
        // corresponding receive size, and the received regions of an overwriting exchange must not overlap.
        let input_offsets = input_offsets.non_negative_integer_elements("input_offsets")?;
        let send_sizes = send_sizes.non_negative_integer_elements("send_sizes")?;
        let output_offsets = output_offsets.non_negative_integer_elements("output_offsets")?;
        let receive_sizes = receive_sizes.non_negative_integer_elements("receive_sizes")?;
        let participant_count = if physical { self.axis_size } else { 1 };
        let metadata_length = input_offsets.len() / participant_count;
        let input_extent = operand.r#type().shape().dimensions()[usize::from(physical)].value().unwrap();
        let output_extent = output.r#type().shape().dimensions()[usize::from(physical)].value().unwrap();
        let groups = if physical { self.participant_groups() } else { vec![vec![0]] };
        let trailing_start = usize::from(physical) + 1;
        let row_element_count = operand.r#type().shape().dimensions()[trailing_start..]
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(dimension.value().unwrap()))
            .ok_or_else(|| ProgramError::InvalidArgument {
                message: format!(
                    "`{PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME}` trailing row size does not fit in `usize`",
                ),
            })?;
        let row_byte_count = row_element_count
            .checked_mul(ArrayAddressing::new(operand.r#type().into_owned())?.element_byte_width())
            .ok_or_else(|| ProgramError::InvalidArgument {
                message: format!(
                    "`{PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME}` trailing row byte size does not fit in `usize`",
                ),
            })?;

        // Validate the complete exchange before copying anything. `output_offsets` are sender-owned metadata in the
        // receiver coordinate frame, while `receive_sizes` are indexed receiver-first and sender-second.
        let overwrite = !self.accumulates_updates();
        let mut received_regions = overwrite.then(|| vec![Vec::new(); participant_count]);
        let mut transfers = Vec::new();
        for group in &groups {
            let slices_per_peer = metadata_length / group.len();
            for (sender_position, &sender) in group.iter().enumerate() {
                for (receiver_position, &receiver) in group.iter().enumerate() {
                    for slice in 0..slices_per_peer {
                        let send_index = receiver_position * slices_per_peer + slice;
                        let receive_index = sender_position * slices_per_peer + slice;
                        let sender_metadata_index = sender * metadata_length + send_index;
                        let receiver_metadata_index = receiver * metadata_length + receive_index;
                        let input_offset = input_offsets[sender_metadata_index];
                        let send_size = send_sizes[sender_metadata_index];
                        let output_offset = output_offsets[sender_metadata_index];
                        let receive_size = receive_sizes[receiver_metadata_index];

                        let input_end =
                            input_offset.checked_add(send_size).ok_or_else(|| ProgramError::InvalidArgument {
                                message: format!(
                                    "`{PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME}` input region for participant \
                                     {sender} at metadata index {send_index} overflows `usize`",
                                ),
                            })?;
                        if input_end > input_extent {
                            return Err(ProgramError::InvalidArgument {
                                message: format!(
                                    "`{PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME}` input region [{input_offset}, \
                                     {input_end}) for participant {sender} exceeds input extent {input_extent}",
                                ),
                            });
                        }

                        if send_size != receive_size {
                            return Err(ProgramError::InvalidArgument {
                                message: format!(
                                    "`{PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME}` send size {send_size} from \
                                     participant {sender} to participant {receiver} does not match receive size \
                                     {receive_size}",
                                ),
                            });
                        }

                        let output_end =
                            output_offset.checked_add(receive_size).ok_or_else(|| ProgramError::InvalidArgument {
                                message: format!(
                                    "`{PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME}` output region for participant \
                                     {receiver} from participant {sender} overflows `usize`",
                                ),
                            })?;
                        if output_end > output_extent {
                            return Err(ProgramError::InvalidArgument {
                                message: format!(
                                    "`{PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME}` output region [{output_offset}, \
                                     {output_end}) for participant {receiver} exceeds output extent {output_extent}",
                                ),
                            });
                        }

                        if receive_size != 0
                            && let Some(received_regions) = &mut received_regions
                        {
                            received_regions[receiver].push((output_offset, output_end));
                        }

                        if send_size != 0 && row_byte_count != 0 {
                            // The regions were checked against the extents above, so every row lies within the
                            // `participant_count × extent` rows of an existing buffer and no byte offset overflows.
                            let source_start = (sender * input_extent + input_offset) * row_byte_count;
                            let destination_start = (receiver * output_extent + output_offset) * row_byte_count;
                            let byte_count = send_size * row_byte_count;
                            transfers.push((source_start, destination_start, byte_count, send_size));
                        }
                    }
                }
            }
        }

        if let Some(mut received_regions) = received_regions {
            for (receiver, regions) in received_regions.iter_mut().enumerate() {
                regions.sort_unstable();
                for regions in regions.windows(2) {
                    if regions[1].0 < regions[0].1 {
                        return Err(ProgramError::InvalidArgument {
                            message: format!(
                                "`{}` received output regions [{}, {}) and [{}, {}) overlap for participant {}",
                                PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME,
                                regions[0].0,
                                regions[0].1,
                                regions[1].0,
                                regions[1].1,
                                receiver,
                            ),
                        });
                    }
                }
            }
        }

        let operand_bytes = operand.logical_bytes();
        let mut result_bytes = output.logical_bytes();
        for (source_start, destination_start, byte_count, row_count) in transfers {
            let source = &operand_bytes[source_start..source_start + byte_count];
            let destination = &mut result_bytes[destination_start..destination_start + byte_count];
            if overwrite {
                destination.copy_from_slice(source);
            } else {
                let mut dimensions = vec![Dimension::Static(row_count)];
                dimensions.extend(operand.r#type().shape().dimensions()[trailing_start..].iter().cloned());
                let segment_type = ArrayType::new(operand.r#type().data_type(), Shape::new(dimensions));
                let source = Array::from_logical_bytes(segment_type.clone(), source)?;
                let destination_array = Array::from_logical_bytes(segment_type, destination)?;
                destination.copy_from_slice(destination_array.add(&source)?.logical_bytes().as_slice());
            }
        }

        Ok(vec![Array::from_logical_bytes(output.r#type().into_owned(), result_bytes.as_slice())?])
    }
}

impl<C: Context<Type = ArrayType, Operation: From<ParallelRaggedAllToAllOperation>>> PartiallyEvaluatableOperation<C>
    for ParallelRaggedAllToAllOperation
{
}

impl<
    C: Context<
            Type = ArrayType,
            Operation: From<ConstantOperation<Array>>
                           + From<AddOperation<ArrayType>>
                           + From<MulOperation<ArrayType>>
                           + From<BroadcastOperation>
                           + From<ConvertElementTypeOperation<ArrayType>>
                           + From<IotaOperation<ArrayType>>
                           + From<ParallelRaggedAllToAllOperation>,
        >,
    P: CollectiveArrayExtentBatchingPolicy<C>,
> BatchableOperation<C, ArrayBatchingPolicy<P>> for ParallelRaggedAllToAllOperation
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        // A matching named batch axis is the eager reference implementation's participant axis. All inputs are aligned
        // to physical axis zero before one parent bind executes the complete exchange. Unresolved non-constant metadata
        // are gated because no existing slicing primitive carries a dynamic segment length. An exchange over a manual
        // mesh axis belongs to its manual region and requires devices, so a matching level, which can only shadow that
        // mesh axis, rejects it. A non-matching mapped axis merges its batch into the packed leading data and metadata
        // axes with sender/receiver offsets rebased by the mapped item index. The rebasing offsets are created with the
        // placement and manual variation of the metadata that they rebase, so a merge inside a manual region needs no
        // variation transition. An all-replicated application can be forwarded unchanged. The physical form has no
        // backend lowering, so a matching level under a staging parent stages it only for eager interpretation of
        // the staged program.
        check_count!("input", inputs, 6, ProgramError);
        ArrayBatch::reject_ragged_inputs(self, inputs)?;

        // A physical operation has already consumed its named axis at an inner level, which shadows every outer level
        // with the same name, so every level that it reaches is unrelated to it.
        if self.is_physical() || context.axis_name() != Some(self.axis_name()) {
            if inputs.iter().all(|input| input.batch_axis().is_replicated()) {
                let parent_inputs = inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
                let mut outputs = context.parent().bind(self.clone(), Vec::new(), parent_inputs.as_slice())?;
                check_count!("output", outputs, 1, ProgramError);
                return Ok(vec![ArrayBatch::replicated(outputs.remove(0))].into());
            }

            // An unrelated mapped axis is merged into the exchange: the batch items share the participants
            // of the named axis, and the merged exchange routes the segments of every batch item independently.
            let provenance_context = context.parent().clone();
            return provenance_context.invoke_with_provenance_scope(ProvenanceScope::new("ryft"), || {
                provenance_context.invoke_with_provenance_scope(ProvenanceScope::new("batching"), || {
                    provenance_context.invoke_with_provenance_scope(
                        ProvenanceScope::new(PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME),
                        || {
                            // A physical operation keeps its leading participant axis in front of every input.
                            // The mapped axis precedes the leading segment axis of the data inputs and follows the
                            // metadata axis of the metadata inputs, which keeps the per-receiver metadata blocks
                            // contiguous after merging (see below).
                            let participant_axis_count = usize::from(self.is_physical());
                            let input_leading_axis = participant_axis_count;
                            let metadata_batch_axis = participant_axis_count + 1;
                            let output_batch_axis = participant_axis_count;
                            let logical_input_types = inputs.iter().map(ArrayBatch::unbatched_type).collect::<Vec<_>>();
                            self.infer_output_types(logical_input_types.as_slice(), &[])?;
                            let batch_size = P::axis_dimension(context)?.value().ok_or_else(|| {
                                BatchingError::UnsupportedOperation {
                                    message: format!(
                                        "`{PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME}` merged batching \
                                         requires a statically known mapped-axis extent",
                                    ),
                                }
                            })?;

                            if batch_size == 0 {
                                let output = P::match_axis(context, &inputs[1], output_batch_axis.into())?;
                                return Ok(vec![output].into());
                            }

                            let static_extent = |r#type: &ArrayType, axis: usize, name: &str| {
                                r#type.shape().dimensions()[axis].value().ok_or_else(|| {
                                    BatchingError::UnsupportedOperation {
                                        message: format!(
                                            "`{PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME}` merged batching requires \
                                             `{name}` axis {axis} to have a static extent",
                                        ),
                                    }
                                })
                            };

                            let input_extent = static_extent(&logical_input_types[0], input_leading_axis, "operand")?;
                            let output_extent = static_extent(&logical_input_types[1], input_leading_axis, "output")?;
                            let metadata_length =
                                logical_input_types[2].shape().dimensions()[participant_axis_count].value().unwrap();
                            let batch_extent = P::collective_extent_constant(context, batch_size)?;
                            let input_extent_value = P::collective_extent_constant(context, input_extent)?;
                            let output_extent_value = P::collective_extent_constant(context, output_extent)?;
                            let metadata_length_value = P::collective_extent_constant(context, metadata_length)?;
                            let participant_extent = self
                                .is_physical()
                                .then(|| P::collective_extent_constant(context, self.axis_size()))
                                .transpose()?;
                            let trailing_extents = (input_leading_axis + 1..logical_input_types[0].rank())
                                .map(|axis| {
                                    static_extent(&logical_input_types[0], axis, "operand")
                                        .and_then(|extent| P::collective_extent_constant(context, extent))
                                })
                                .collect::<Result<Vec<_>, _>>()?;

                            // Merge the mapped axis into the leading segment axis of both data inputs, so that
                            // `(P?, B, N, ...)` becomes `(P?, B·N, ...)` and the rows of batch item `b` start at row
                            // `b·N` of `operand` and `b·M` of `output`. The merged reshapes drop the output sharding,
                            // which is restored on the result below.
                            let aligned_operand = P::match_axis(context, &inputs[0], output_batch_axis.into())?;
                            let aligned_output = P::match_axis(context, &inputs[1], output_batch_axis.into())?;
                            let restored_output_sharding = aligned_output.r#type().sharding().cloned();
                            let mut operand_extents = Vec::new();
                            let mut output_extents = Vec::new();
                            if let Some(participant_extent) = &participant_extent {
                                operand_extents.push(participant_extent.clone());
                                output_extents.push(participant_extent.clone());
                            }
                            operand_extents.push(batch_extent.mul(&input_extent_value)?);
                            output_extents.push(batch_extent.mul(&output_extent_value)?);
                            operand_extents.extend(trailing_extents.iter().cloned());
                            output_extents.extend(trailing_extents.iter().cloned());
                            let operand = P::reshape_collective(
                                context,
                                aligned_operand.into_value(),
                                operand_extents.as_slice(),
                                None,
                            )?;
                            let output = P::reshape_collective(
                                context,
                                aligned_output.into_value(),
                                output_extents.as_slice(),
                                None,
                            )?;

                            // Every metadata array of shape `(P?, K)` takes its mapped axis right after its metadata
                            // axis, so flattening `(P?, K, B)` into `(P?, K·B)` maps entry `k = r·(K/G) + s` of batch
                            // item `b` to entry `r·(K/G)·B + (s·B + b)`. The block of every receiver `r` therefore
                            // stays contiguous for any participant-group size `G`. The metadata are widened to `u64`
                            // first, so that rebasing offsets by `b·N` and `b·M` cannot overflow their original integer
                            // type.
                            let mut metadata = inputs[2..]
                                .iter()
                                .map(|input| {
                                    let metadata = P::match_axis(context, input, metadata_batch_axis.into())?;
                                    if metadata.r#type().data_type() == DataType::U64 {
                                        return Ok(metadata);
                                    }
                                    let batch_axis = metadata.batch_axis();
                                    let mut values = context.parent().bind(
                                        ConvertElementTypeOperation::new(DataType::U64, false),
                                        Vec::new(),
                                        std::slice::from_ref(metadata.value()),
                                    )?;
                                    check_count!("output", values, 1, ProgramError);
                                    ArrayBatch::new(values.remove(0), batch_axis)
                                })
                                .collect::<Result<Vec<_>, _>>()?;

                            // Rebase the `input_offsets` and `output_offsets` of batch item `b` by `b·N` and `b·M`,
                            // the first rows of that item in the merged data inputs. Sizes need no rebasing. The
                            // rebasing term is an `iota` along the mapped axis of the aligned metadata, scaled by the
                            // leading extent broadcast to the full metadata type, so every operand of the `mul` and
                            // `add` below has the physical type of the metadata, including its sharding and manual
                            // variation.
                            for (metadata_index, leading_extent) in [(0, input_extent), (2, output_extent)] {
                                let metadata_type = metadata[metadata_index].r#type().into_owned();

                                let mut iota = context.parent().bind(
                                    IotaOperation::new(metadata_type.clone(), metadata_batch_axis)?,
                                    Vec::new(),
                                    &[],
                                )?;
                                check_count!("output", iota, 1, ProgramError);

                                // The scale takes part only in the offset arithmetic of each shard, so it is created
                                // in the memory space and with the manual variation of the metadata that it rebases
                                // instead of going through a `parallel_vary` transition. Skipping that transition is
                                // sound only because the scale is a non-differentiable constant: the transpose of the
                                // transition is a cross-device sum, which a value that can carry a tangent would need.
                                let leading_extent =
                                    u64::try_from(leading_extent).map_err(|_| ProgramError::InvalidArgument {
                                        message: format!(
                                            "`{PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME}` extent {leading_extent} \
                                             does not fit in `u64`",
                                        ),
                                    })?;
                                let mut scale_type =
                                    ArrayType::scalar(DataType::U64).with_memory(metadata_type.memory());
                                if let Some(sharding) = metadata_type.sharding() {
                                    let scale_sharding = Sharding::replicated(sharding.mesh().clone(), 0)
                                        .with_varying_manual_axes(sharding.varying_manual_axes().iter().cloned())
                                        .map_err(TypeError::from)?;
                                    scale_type = scale_type.with_sharding(scale_sharding).map_err(TypeError::from)?;
                                }
                                let mut scale = context.parent().bind(
                                    ConstantOperation::new(Array::from_elements(scale_type, &[leading_extent])?),
                                    Vec::new(),
                                    &[],
                                )?;
                                check_count!("output", scale, 1, ProgramError);

                                let mut scale = context.parent().bind(
                                    BroadcastOperation::new(metadata_type, Vec::new()),
                                    Vec::new(),
                                    &[scale.remove(0)],
                                )?;
                                check_count!("output", scale, 1, ProgramError);

                                let mut rebasing = context.parent().bind(
                                    MulOperation::new(),
                                    Vec::new(),
                                    &[iota.remove(0), scale.remove(0)],
                                )?;
                                check_count!("output", rebasing, 1, ProgramError);

                                let mut rebased = context.parent().bind(
                                    AddOperation::new(),
                                    Vec::new(),
                                    &[metadata[metadata_index].value().clone(), rebasing.remove(0)],
                                )?;
                                check_count!("output", rebased, 1, ProgramError);

                                metadata[metadata_index] =
                                    ArrayBatch::new(rebased.remove(0), BatchAxis::from_position(metadata_batch_axis))?;
                            }

                            let mut metadata_extents = Vec::new();
                            if let Some(participant_extent) = &participant_extent {
                                metadata_extents.push(participant_extent.clone());
                            }
                            metadata_extents.push(metadata_length_value.mul(&batch_extent)?);
                            let metadata = metadata
                                .into_iter()
                                .map(|metadata| {
                                    P::reshape_collective(
                                        context,
                                        metadata.into_value(),
                                        metadata_extents.as_slice(),
                                        None,
                                    )
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let merged_inputs = std::iter::once(operand)
                                .chain(std::iter::once(output))
                                .chain(metadata)
                                .collect::<Vec<_>>();
                            let mut outputs =
                                context.parent().bind(self.clone(), Vec::new(), merged_inputs.as_slice())?;
                            check_count!("output", outputs, 1, ProgramError);

                            // Split the merged leading axis of the result back into `(B, M)` and restore the output
                            // sharding that the merged reshape dropped.
                            let mut restored_extents = Vec::new();
                            if let Some(participant_extent) = participant_extent {
                                restored_extents.push(participant_extent);
                            }
                            restored_extents.push(batch_extent);
                            restored_extents.push(output_extent_value);
                            restored_extents.extend(trailing_extents);
                            let output = P::reshape_collective(
                                context,
                                outputs.remove(0),
                                restored_extents.as_slice(),
                                restored_output_sharding,
                            )?;
                            Ok(vec![ArrayBatch::new(output, BatchAxis::from_position(output_batch_axis))?].into())
                        },
                    )
                })
            });
        }

        if self.mesh.is_some() {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME}` over a manual mesh axis \
                     cannot bind a named batch axis",
                ),
            });
        }

        P::collective_axis_extent(
            context,
            PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME,
            self.axis_name(),
            self.axis_size,
        )?;

        let inputs =
            inputs.iter().map(|input| P::match_axis(context, input, 0.into())).collect::<Result<Vec<_>, _>>()?;
        if inputs[2..]
            .iter()
            .any(|input| !matches!(context.parent().resolve(input.value()), ValueResolution::Constant(_)))
        {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME}` cannot materialize a batch-bound collective with \
                     staged metadata",
                ),
            });
        }

        let physical_inputs = inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
        let mut outputs =
            context.parent().bind(self.with_physical_representation(), Vec::new(), physical_inputs.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(vec![ArrayBatch::new(outputs.remove(0), BatchAxis::from_position(0))?].into())
    }
}

impl_differentiable_operation! {
    ParallelRaggedAllToAllOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType> + Zero<C::Value>,
        C::Operation: From<ParallelRaggedAllToAllOperation>,
    {
        |operation, context, _driver, inputs| {
            // The two data inputs are jointly linear. Metadata remain primal values and therefore become ordinary
            // residuals whenever the tangent exchange survives partial evaluation.
            check_count!("input", inputs, 6, ProgramError);
            let [operand, output, input_offsets, send_sizes, output_offsets, receive_sizes] = inputs else {
                unreachable!();
            };
            let primal_inputs = [
                operand.primal().clone(),
                output.primal().clone(),
                input_offsets.primal().clone(),
                send_sizes.primal().clone(),
                output_offsets.primal().clone(),
                receive_sizes.primal().clone(),
            ];
            let mut primal_outputs = context.primal().bind(operation.clone(), Vec::new(), &primal_inputs)?;
            check_count!("output", primal_outputs, 1, ProgramError);
            let primal = primal_outputs.remove(0);
            let tangent = if operand.tangent().is_zero() && output.tangent().is_zero() {
                MaybeZero::Zero(primal.r#type().tangent()?)
            } else {
                // A missing data tangent is a zero of the tangent type rather than of the primal type, so it neither
                // turns the primal data into a residual nor disagrees with a widened tangent data type. Only the
                // metadata primals move to the tangent context.
                let tangent_inputs = [
                    operand.tangent().clone().materialize(context.tangent())?,
                    output.tangent().clone().materialize(context.tangent())?,
                    context.primal_to_tangent(input_offsets.primal().clone())?,
                    context.primal_to_tangent(send_sizes.primal().clone())?,
                    context.primal_to_tangent(output_offsets.primal().clone())?,
                    context.primal_to_tangent(receive_sizes.primal().clone())?,
                ];
                let mut tangent_outputs = context.tangent().bind(operation.clone(), Vec::new(), &tangent_inputs)?;
                check_count!("output", tangent_outputs, 1, ProgramError);
                MaybeZero::Value(tangent_outputs.remove(0))
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: From<ZeroOperation<ArrayType>>
            + From<OneOperation<ArrayType>>
            + From<NegOperation<ArrayType>>
            + From<BroadcastOperation>
            + From<ConcatenateOperation<ArrayType>>
            + From<CompareOperation<ArrayType>>
            + From<SelectOperation<ArrayType>>
            + From<ReshapeOperation>
            + From<SliceOperation>
            + From<ScatterOperation>
            + From<ConvertElementTypeOperation<ArrayType>>
            + From<TransferToMemoryOperation>
            + From<CumulativeOperation>
            + From<ParallelAllToAllOperation>
            + From<ParallelRaggedAllToAllOperation>
            + OperationProvider<ArrayType, ParallelVaryOperation, Operation = O>
            + OperationProvider<ArrayType, BroadcastOperation, Operation = O>,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            let provenance_context = context.clone();
            provenance_context.invoke_with_provenance_scope(ProvenanceScope::new("ryft"), || {
                provenance_context.invoke_with_provenance_scope(ProvenanceScope::new("differentiation"), || {
                    provenance_context.invoke_with_provenance_scope(
                        ProvenanceScope::new(format!("{PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME}_transpose")),
                        || {
                            check_count!("input", inputs, 6, ProgramError);
                            check_count!("output", outputs, 1, ProgramError);
                            check_count!("accumulator", accumulators, 6, DifferentiationError);
                            let [operand, output, input_offsets, send_sizes, output_offsets, receive_sizes] = inputs
                            else {
                                unreachable!();
                            };
                            let MaybeZero::Value(cotangent) = &outputs[0] else {
                                return Ok(());
                            };
                            if !accumulators[0].is_needed() && !accumulators[1].is_needed() {
                                return Ok(());
                            }

                            // The metadata route the cotangents, so they must be known primal residuals.
                            let known_input = |input: &PartialValue<Tracer<TracingContext<V, O>>>, name: &str| {
                                input.as_known().cloned().ok_or_else(|| ProgramError::UnsupportedOperation {
                                    message: format!(
                                        "`{PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME}` transpose requires `{name}` \
                                         to be a known primal residual",
                                    ),
                                })
                            };
                            let input_offsets = known_input(input_offsets, "input_offsets")?;
                            let send_sizes = known_input(send_sizes, "send_sizes")?;
                            let output_offsets = known_input(output_offsets, "output_offsets")?;
                            let receive_sizes = known_input(receive_sizes, "receive_sizes")?;
                            let (operand_cotangent, permuted_output_offsets) = if !accumulators[0].is_needed() {
                                (MaybeZero::Zero(operand.r#type().cotangent()?), None)
                            } else {
                                let permuted_output_offsets = operation.transpose_offsets(context, &output_offsets)?;
                                let permuted_input_offsets = operation.transpose_offsets(context, &input_offsets)?;
                                let zero = context.zero(&operand.r#type().cotangent()?)?;
                                let adjoint_inputs = [
                                    cotangent.clone(),
                                    zero,
                                    permuted_output_offsets.clone(),
                                    receive_sizes.clone(),
                                    permuted_input_offsets,
                                    send_sizes.clone(),
                                ];
                                let mut contributions =
                                    context.bind(operation.with_additive_updates(), Vec::new(), &adjoint_inputs)?;
                                check_count!("output", contributions, 1, ProgramError);
                                (MaybeZero::Value(contributions.remove(0)), Some(permuted_output_offsets))
                            };
                            let output_cotangent = if !accumulators[1].is_needed() {
                                MaybeZero::Zero(output.r#type().cotangent()?)
                            } else if operation.accumulates_updates() {
                                MaybeZero::Value(cotangent.clone())
                            } else {
                                let permuted_output_offsets = match permuted_output_offsets {
                                    Some(permuted_output_offsets) => permuted_output_offsets,
                                    None => operation.transpose_offsets(context, &output_offsets)?,
                                };

                                // Preserve the output seed's cotangent outside the received regions with an interval
                                // mask. Each received region `[o, o + n)` contributes `+1` at `o` and `-1` at `o + n`
                                // to a marker array of extent `M + 1`, whose prefix sums are then nonzero exactly at
                                // received rows.
                                let output_type = cotangent.r#type().into_owned();
                                let leading_axis = usize::from(operation.is_physical());
                                let output_extent =
                                    output_type.shape().dimensions()[leading_axis].value().ok_or_else(|| {
                                        ProgramError::UnsupportedOperation {
                                            message: format!(
                                                "`{PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME}` transpose requires a \
                                                 static output leading dimension",
                                            ),
                                        }
                                    })?;
                                let marker_extent =
                                    output_extent.checked_add(1).ok_or_else(|| ProgramError::InvalidArgument {
                                        message: format!(
                                            "`{PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME}` transpose marker extent \
                                             does not fit in `usize`",
                                        ),
                                    })?;
                                let mut marker_dimensions = output_type.shape().dimensions()[..=leading_axis].to_vec();
                                marker_dimensions[leading_axis] = Dimension::Static(marker_extent);

                                // Marker constants participate in the metadata computation on each shard. Give them the
                                // metadata's placement and variation from creation instead of staging a `parallel_vary`
                                // transition. This is sound only because they are non-differentiable constants: the
                                // transpose of that transition is a cross-device sum, which any value that can carry
                                // a tangent would need. The marker's segment dimension has extent `M + 1`, which is
                                // unrelated to the metadata length, so it is replicated, while the participant axis of
                                // the physical form keeps the offsets' sharding so that it lines up with the batching
                                // dimensions of the scatters below.
                                let marker_sharding = permuted_output_offsets
                                    .r#type()
                                    .sharding()
                                    .map(|sharding| {
                                        let mut dimensions = sharding.dimensions().to_vec();
                                        dimensions[leading_axis] = ShardingDimension::Replicated;
                                        sharding.with_dimensions(dimensions)
                                    })
                                    .transpose()
                                    .map_err(TypeError::from)?;
                                let marker_type = ArrayType::new(DataType::I64, Shape::new(marker_dimensions))
                                    .with_memory(output_type.memory())
                                    .with_sharding(marker_sharding)
                                    .map_err(TypeError::from)?;

                                // Metadata may use any integer width and memory placement. Widen index arithmetic to
                                // `u64` before adding and move it beside the cotangent so scatter's three inputs share
                                // one memory space.
                                let normalize_metadata =
                                    |value: &Tracer<TracingContext<V, O>>| -> Result<_, ProgramError> {
                                        let value = if value.r#type().memory() == output_type.memory() {
                                            value.clone()
                                        } else {
                                            value.transfer_to_memory(output_type.memory())?
                                        };
                                        if value.r#type().data_type() == DataType::U64 {
                                            Ok(value)
                                        } else {
                                            Ok(value.convert_element_type(DataType::U64)?)
                                        }
                                    };
                                let output_offsets = normalize_metadata(&permuted_output_offsets)?;
                                let receive_sizes = normalize_metadata(&receive_sizes)?;
                                let update_type = output_offsets.r#type().into_owned().with_data_type(DataType::I64);
                                let marker = context.zero(&marker_type)?;
                                let ones = context.one(&update_type)?;
                                let negative_ones = ones.neg()?;
                                let end_offsets = output_offsets.add(&receive_sizes)?;
                                let mut index_dimensions = output_offsets.r#type().shape().dimensions().to_vec();
                                index_dimensions.push(Dimension::Static(1));
                                let start_indices = output_offsets.reshape(Shape::new(index_dimensions.clone()))?;
                                let end_indices = end_offsets.reshape(Shape::new(index_dimensions))?;
                                let scatter_dimensions = if operation.is_physical() {
                                    ScatterDimensionNumbers::new(Vec::new(), vec![1], vec![1])
                                        .with_batching_dimensions(vec![0], vec![0])
                                } else {
                                    ScatterDimensionNumbers::new(Vec::new(), vec![0], vec![0])
                                };

                                // Both boundaries are additive. A zero-length region contributes `+1` and `-1` at the
                                // same position, while adjacent regions combine deterministically at their shared
                                // boundary.
                                let options = ScatterOptions::new();
                                let markers = marker
                                    .scatter(
                                        &start_indices,
                                        &ones,
                                        &scatter_dimensions,
                                        ScatterReductionKind::Add,
                                        &options,
                                    )?
                                    .scatter(
                                        &end_indices,
                                        &negative_ones,
                                        &scatter_dimensions,
                                        ScatterReductionKind::Add,
                                        &options,
                                    )?;
                                let markers = markers.cumulative_sum(leading_axis)?;
                                let start_indices = vec![0; markers.r#type().rank()];
                                let mut limit_indices = markers
                                    .r#type()
                                    .shape()
                                    .dimensions()
                                    .iter()
                                    .map(|dimension| dimension.value().unwrap())
                                    .collect::<Vec<_>>();
                                limit_indices[leading_axis] = output_extent;
                                let strides = vec![1; markers.r#type().rank()];
                                let markers = markers.slice(
                                    start_indices.as_slice(),
                                    limit_indices.as_slice(),
                                    strides.as_slice(),
                                )?;
                                let marker_zero = context.zero(markers.r#type().as_ref())?;
                                let received = markers.not_equal(&marker_zero)?;

                                // The receive mask varies with its offset metadata. Expanding it over payload
                                // dimensions changes only geometry; retain those variation facts instead of
                                // replacing them with an unsharded Boolean type.
                                let mut condition_type = received.r#type().into_owned();
                                for (axis, dimension) in
                                    output_type.shape().dimensions().iter().enumerate().skip(leading_axis + 1)
                                {
                                    condition_type = condition_type.with_inserted_dimension(axis, dimension.clone())?;
                                }
                                let received =
                                    received.broadcast(condition_type, &(0..=leading_axis).collect::<Vec<_>>())?;
                                let zero = context.zero(&output_type)?;
                                MaybeZero::Value(Tracer::select(&received, &zero, cotangent)?)
                            };
                            accumulators[0].accumulate(context, operand_cotangent)?;
                            accumulators[1].accumulate(context, output_cotangent)
                        },
                    )
                })
            })
        }
    },
}

impl MemberOperation<ArrayIrType> for ParallelRaggedAllToAllOperation {
    // This direct composite carrier has an array-only boundary, but it cannot be a second projected `ArrayType`
    // member in `ArrayIrOperation`. Keep the projection explicit so its contract remains identical to the homogeneous
    // operation.

    #[inline]
    fn infer_parent_region_input_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<Option<Vec<ArrayIrType>>>, TypeError> {
        infer_projected_operation_region_input_types(self, input_types, region_interfaces)
    }

    #[inline]
    fn infer_parent_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        infer_projected_operation_output_types(self, input_types, region_interfaces)
    }

    #[inline]
    fn rename_parent_type_identities(
        &self,
        _renaming: &TypeIdentityRenaming<DimensionVariable>,
    ) -> Result<Self, TypeError> {
        Ok(self.clone())
    }
}

impl<C: Domain<Type = ArrayIrType, Value: ValueProjection<ArrayType, Projected = Array>>>
    MemberInterpretableOperation<C> for ParallelRaggedAllToAllOperation
{
    #[inline]
    fn interpret_in_parent<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        interpret_projected_operation(context, self, driver, inputs)
    }
}

impl<C: Context<Type = ArrayIrType> + DomainProjection<ArrayType>> MemberBatchableOperation<C, ArrayIrBatchingPolicy>
    for ParallelRaggedAllToAllOperation
where
    ParallelRaggedAllToAllOperation:
        BatchableOperation<ProjectedContext<C, ArrayType>, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>,
{
    #[inline]
    fn batch_in_parent<D: BatchingDriver<C, ArrayIrBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        _driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatchingPolicy>, BatchingError> {
        batch_projected_operation(context, self, inputs)
    }
}

impl<
    C: Context<
            Type = ArrayIrType,
            Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
            Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
            Operation: OperationProjection<
                ArrayType,
                Projected: DifferentiableOperation<ProjectedContext<C, ArrayType>>
                               + From<ParallelRaggedAllToAllOperation>,
            >,
        >,
> MemberDifferentiableOperation<C> for ParallelRaggedAllToAllOperation
{
    #[inline]
    fn jvp_in_parent<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let operation = <C::Operation as OperationProjection<ArrayType>>::Projected::from(self.clone());
        jvp_projected_operation(context, &operation, inputs)
    }
}

/// Represents the ability to exchange variable-length leading-axis segments between participants of a named axis by
/// staging a [`ParallelRaggedAllToAllOperation`]. Refer to that operation for the routing metadata, the runtime
/// preconditions, and the grouping, variation, and transformation semantics. Over a manual mesh axis, the exchange
/// records the mesh of that axis and first makes its inputs vary over the axis, because the receivers generally obtain
/// different segments. Every input is then aligned to the manual axes that any input varies over (refer to
/// [`ManualVariationAlignment`]), because each device computes its result from all six inputs.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and composite array IR values implement it for [`ArrayIrType`].
///
/// # Example
///
/// Each of two participants sends its first row to participant zero and its second row to participant one, and each
/// sender's segment lands at the receiver row that matches the sender's index:
///
/// ```
/// # use ryft_core::{Array, BatchAxis, BatchAxisSpecification, ParallelRaggedAllToAll, batch};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let operand = Array::matrix(2, 2, vec![1.0f32, 2.0, 3.0, 4.0])?;
/// let output = Array::matrix(2, 2, vec![0.0f32; 4])?;
/// let input_offsets = Array::matrix(2, 2, vec![0i32, 1, 0, 1])?;
/// let sizes = Array::matrix(2, 2, vec![1i32; 4])?;
/// let output_offsets = Array::matrix(2, 2, vec![0i32, 0, 1, 1])?;
/// let received: Array = batch(
///     |(operand, output, input_offsets, send_sizes, output_offsets, receive_sizes)| {
///         operand.parallel_ragged_all_to_all(
///             "rows",
///             &output,
///             &input_offsets,
///             &send_sizes,
///             &output_offsets,
///             &receive_sizes,
///         )
///     },
///     (operand, output, input_offsets, sizes.clone(), output_offsets, sizes),
///     BatchAxis::new(0),
///     BatchAxis::new(0),
///     BatchAxisSpecification::named("rows"),
/// )?;
/// assert_eq!(received, Array::matrix(2, 2, vec![1.0f32, 3.0, 2.0, 4.0])?);
/// # Ok(())
/// # }
/// ```
#[capability(projection(ArrayIrType => ArrayType))]
pub trait ParallelRaggedAllToAll<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Exchanges variable-length segments of the leading axis of this value, which is the `operand` of the exchange,
    /// with every participant of `axis_name` (including itself), and returns `output` with the segments that this
    /// participant receives written into it. Segments are ranges of leading-axis rows, so `output` must have the data
    /// type and trailing dimensions of this value, while the two leading extents may differ.
    ///
    /// The four metadata inputs are rank-one arrays of one shared integer data type and of one length `K`, which must
    /// be a positive multiple of the number of participants `P`. Each participant exchanges `K / P` segments with each
    /// participant, some of which may be empty, and the `K / P` entries starting at `p · K / P` describe the segments
    /// that it exchanges with participant `p`. For example, with `P = 2` and `K = 2`, entry `1` of `input_offsets`,
    /// `send_sizes`, and `output_offsets` describes the segment sent to participant `1`, and entry `1` of
    /// `receive_sizes` describes the segment received from participant `1`.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of the batch axis or manual mesh axis whose participants exchange segments.
    ///   - `output`: Value into which the received segments are written. Its rows outside every received region keep
    ///     their values, and the received regions must not overlap.
    ///   - `input_offsets`: Leading-axis offset in this value of the first row of each segment that this participant
    ///     sends.
    ///   - `send_sizes`: Number of rows of each segment that this participant sends.
    ///   - `output_offsets`: Leading-axis offset of the first row of each segment that this participant sends, in the
    ///     `output` of the receiving participant. Senders therefore decide where their segments land.
    ///   - `receive_sizes`: Number of rows of each segment that this participant receives. Every entry must equal the
    ///     `send_sizes` entry of the sending participant for the same segment, which means that `receive_sizes` must
    ///     equal the tiled all-to-all of `send_sizes` (refer to the documentation of
    ///     [`ParallelRaggedAllToAllOperation`] for more information).
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError::Axis`] error wrapping [`AxisError::UnboundAxisName`] when no enclosing binder binds
    /// `axis_name`, and a [`ProgramError`] if an input carries a pending sum over the participating manual mesh axis or
    /// if the inputs violate the type contract of [`ParallelRaggedAllToAllOperation`]. Eager execution also rejects
    /// metadata that violate its runtime preconditions.
    fn parallel_ragged_all_to_all(
        &self,
        axis_name: &str,
        output: &Self,
        input_offsets: &Self,
        send_sizes: &Self,
        output_offsets: &Self,
        receive_sizes: &Self,
    ) -> Result<Self, ProgramError>;

    /// Exchanges variable-length segments of the leading axis of this value like
    /// [`parallel_ragged_all_to_all`](Self::parallel_ragged_all_to_all), except that every participant exchanges
    /// segments only with the members of its own group in `axis_index_groups`. The metadata then address peers by
    /// their position within that group rather than by axis index: with groups of size `G`, the metadata length `K`
    /// must be a positive multiple of `G`, and the `K / G` entries starting at `p · K / G` describe the segments that
    /// a participant exchanges with the member at position `p` of its group. For example, with groups
    /// `[[0, 2], [3, 1]]` and `K = 2`, entry `1` of the `input_offsets`, `send_sizes`, and `output_offsets` of
    /// participant `3` describes the segment that it sends to participant `1`.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of the batch axis or manual mesh axis whose participants exchange segments.
    ///   - `output`: Value into which the received segments are written. Its rows outside every received region keep
    ///     their values, and the received regions must not overlap.
    ///   - `input_offsets`: Leading-axis offset in this value of the first row of each segment that this participant
    ///     sends.
    ///   - `send_sizes`: Number of rows of each segment that this participant sends.
    ///   - `output_offsets`: Leading-axis offset of the first row of each segment that this participant sends, in the
    ///     `output` of the receiving group member. Senders therefore decide where their segments land.
    ///   - `receive_sizes`: Number of rows of each segment that this participant receives. Every entry must equal the
    ///     `send_sizes` entry of the sending group member for the same segment.
    ///   - `axis_index_groups`: Participant groups, each listing axis indices of `axis_name`, that together partition
    ///     all of its indices into groups of equal size. The order within each group defines the positions that the
    ///     metadata use to address peers (refer to the documentation of [`ParallelRaggedAllToAllOperation::grouped`]
    ///     for more information).
    ///
    /// # Errors
    ///
    /// Returns the errors of [`parallel_ragged_all_to_all`](Self::parallel_ragged_all_to_all), and a
    /// [`ProgramError`] if `axis_index_groups` is not an equal-sized exact partition of the full axis.
    fn parallel_ragged_all_to_all_with_axis_index_groups(
        &self,
        axis_name: &str,
        output: &Self,
        input_offsets: &Self,
        send_sizes: &Self,
        output_offsets: &Self,
        receive_sizes: &Self,
        axis_index_groups: Vec<Vec<usize>>,
    ) -> Result<Self, ProgramError>;
}

impl ParallelRaggedAllToAll<ArrayType> for Array {
    #[inline]
    fn parallel_ragged_all_to_all(
        &self,
        axis_name: &str,
        _output: &Self,
        _input_offsets: &Self,
        _send_sizes: &Self,
        _output_offsets: &Self,
        _receive_sizes: &Self,
    ) -> Result<Self, ProgramError> {
        // A concrete `Array` never executes inside an axis binder, because the values under a `batch` level
        // or inside a manual region are tracers, so every axis name is unbound for it.
        Err(AxisError::UnboundAxisName { name: axis_name.to_string() }.into())
    }

    #[inline]
    fn parallel_ragged_all_to_all_with_axis_index_groups(
        &self,
        axis_name: &str,
        _output: &Self,
        _input_offsets: &Self,
        _send_sizes: &Self,
        _output_offsets: &Self,
        _receive_sizes: &Self,
        _axis_index_groups: Vec<Vec<usize>>,
    ) -> Result<Self, ProgramError> {
        // A concrete `Array` never executes inside an axis binder, because the values under a `batch` level
        // or inside a manual region are tracers, so every axis name is unbound for it.
        Err(AxisError::UnboundAxisName { name: axis_name.to_string() }.into())
    }
}

impl<
    V: Value<Type = ArrayType, DispatchDomain: Context<Operation: From<ParallelRaggedAllToAllOperation>> + NamedAxes>
        + ParallelVary,
> ParallelRaggedAllToAll<ArrayType> for V
{
    #[inline]
    fn parallel_ragged_all_to_all(
        &self,
        axis_name: &str,
        output: &Self,
        input_offsets: &Self,
        send_sizes: &Self,
        output_offsets: &Self,
        receive_sizes: &Self,
    ) -> Result<Self, ProgramError> {
        let axis_size = resolve_named_axis_size(&self.dispatch_domain(), axis_name)?;
        ParallelRaggedAllToAllOperation::new(axis_name.to_string(), axis_size).stage([
            self,
            output,
            input_offsets,
            send_sizes,
            output_offsets,
            receive_sizes,
        ])
    }

    #[inline]
    fn parallel_ragged_all_to_all_with_axis_index_groups(
        &self,
        axis_name: &str,
        output: &Self,
        input_offsets: &Self,
        send_sizes: &Self,
        output_offsets: &Self,
        receive_sizes: &Self,
        axis_index_groups: Vec<Vec<usize>>,
    ) -> Result<Self, ProgramError> {
        let axis_size = resolve_named_axis_size(&self.dispatch_domain(), axis_name)?;
        ParallelRaggedAllToAllOperation::grouped(axis_name.to_string(), axis_size, axis_index_groups)?.stage([
            self,
            output,
            input_offsets,
            send_sizes,
            output_offsets,
            receive_sizes,
        ])
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayIrOperation, ArrayIrValue, ArrayOperation, DimensionBounds, DimensionType, MeshAxis, MeshAxisType,
        RaggedAxis,
    };
    use crate::batching::{BatchAxisSpecification, BatchingTracer, batch};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::differentiate_at;
    use crate::macros::{
        check_gradient, check_operation_batching, check_operation_partial_evaluation, check_operation_transposition,
        check_operation_type_inference,
    };
    use crate::operations::collectives::tests::eager_collective_context;
    use crate::operations::manipulation::transposition::Transpose;
    use crate::operations::reductions::{Reduce, ReductionKind};
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationContext, PartialEvaluationOutput, PartialEvaluationValue, PartialTracer};
    use crate::programs::{EmptyRegionDriver, Program, ProgramBuilder, ProgramRenderingMode};

    use super::*;

    /// Composite trace shared by the dynamic mapped-extent batching, composite differentiation, and capability tests.
    type ArrayIrTracingContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    /// Executes a degenerate single-participant ragged exchange of the `operand` `[1, 2, 3]` into the `output` seed
    /// `[9, 9, 9, 9]`, using the provided `i64` metadata.
    fn interpret_single_participant_parallel_ragged_all_to_all(
        input_offsets: Vec<i64>,
        send_sizes: Vec<i64>,
        output_offsets: Vec<i64>,
        receive_sizes: Vec<i64>,
    ) -> Result<Array, ProgramError> {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let mut outputs = context.bind(
            ParallelRaggedAllToAllOperation::new("x".to_string(), 1),
            Vec::new(),
            &[
                Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap(),
                Array::vector(vec![9.0f32, 9.0, 9.0, 9.0]).unwrap(),
                Array::vector(input_offsets).unwrap(),
                Array::vector(send_sizes).unwrap(),
                Array::vector(output_offsets).unwrap(),
                Array::vector(receive_sizes).unwrap(),
            ],
        )?;
        Ok(outputs.remove(0))
    }

    /// Applies an explicit list of logical row transfers without deriving routing from the operation metadata. The
    /// packed `operand` and `output` buffers hold `input_extent` and `output_extent` rows of `row_width` elements per
    /// participant, and each transfer is a `(sender, input_offset, receiver, output_offset, size)` tuple.
    fn reference_ragged_transfers(
        operand: &[i32],
        output: &[i32],
        input_extent: usize,
        output_extent: usize,
        row_width: usize,
        transfers: &[(usize, usize, usize, usize, usize)],
    ) -> Vec<i32> {
        let mut result = output.to_vec();
        for &(sender, input_offset, receiver, output_offset, size) in transfers {
            for row in 0..size {
                let source = (sender * input_extent + input_offset + row) * row_width;
                let destination = (receiver * output_extent + output_offset + row) * row_width;
                result[destination..destination + row_width].copy_from_slice(&operand[source..source + row_width]);
            }
        }
        result
    }

    /// Returns a mesh with two manual axes, so that the unrelated axis `y` can carry variation and pending sums.
    fn manual_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap()
    }

    /// Builds a program over `operand` and `output` inputs of the provided types that applies `operation` to them and
    /// to the provided constant `input_offsets`, `send_sizes`, `output_offsets`, and `receive_sizes`, in that order,
    /// and returns the exchanged result.
    fn constant_metadata_exchange_program(
        operation: ParallelRaggedAllToAllOperation,
        operand_type: ArrayType,
        output_type: ArrayType,
        metadata: [Array; 4],
    ) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Array> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let mut inputs = vec![builder.add_input(operand_type), builder.add_input(output_type)];
        inputs.extend(metadata.map(|metadata| builder.add_constant(metadata)));
        let result = builder.add_instruction(operation, Vec::new(), inputs, None).unwrap()[0];
        builder.build(vec![result], vec![Placeholder; 2], Placeholder).unwrap()
    }

    /// Traces an ungrouped exchange over the named axis `x` of size 2 that an unrelated `y` level of size 2 batches,
    /// over inputs of the provided types. The `y` level maps the leading axis of the data inputs and the trailing axis
    /// of the metadata inputs.
    fn trace_unrelated_axis_exchange(
        input_types: Vec<ArrayType>,
    ) -> Result<(ArrayType, Program<Array, ArrayOperation<Array>, Vec<Array>, Array>), ProgramError> {
        TracingContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<_>| {
                let context = BatchingContext::new(inputs[0].context().clone(), 2).with_axis_name("y".to_string());
                let inputs = inputs
                    .into_iter()
                    .enumerate()
                    .map(|(index, input)| ArrayBatch::new(input, BatchAxis::new(usize::from(index >= 2))))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut outputs = ParallelRaggedAllToAllOperation::new("x".to_string(), 2)
                    .batch(&context, &EmptyRegionDriver, inputs.as_slice())?
                    .into_parts()
                    .0;
                Ok(outputs.remove(0).into_value())
            },
            input_types,
        )
    }

    /// Exchanges the two-participant `operand` rows into the `output` seeds at a matching named `batch` level over the
    /// axis `x` and sums the exchanged result. The fixed metadata route row 0 of participant 0 into row 1 of
    /// participant 1 and row 0 of participant 1 into row 2 of participant 0. The metadata are lifted into the context
    /// of `operand`, so the same expansion serves both concrete evaluation and reverse-mode differentiation. This is a
    /// macro rather than a generic function because each call site binds a different traced value type, and naming the
    /// batching capability bounds of a generic value type overflows the trait solver.
    macro_rules! batched_parallel_ragged_all_to_all_sum {
        // Expands to a `Result` holding the summed exchange of the provided `operand` and `output` values.
        ($operand:expr, $output:expr $(,)?) => {{
            let operand = $operand;
            let output = $output;
            let context = operand.dispatch_domain();
            let input_offsets = context.lift(Array::matrix(2, 2, vec![0i32, 0, 0, 2]).unwrap())?;
            let send_sizes = context.lift(Array::matrix(2, 2, vec![0i32, 1, 1, 0]).unwrap())?;
            let output_offsets = context.lift(Array::matrix(2, 2, vec![1i32, 1, 2, 3]).unwrap())?;
            let receive_sizes = context.lift(Array::matrix(2, 2, vec![0i32, 1, 1, 0]).unwrap())?;
            let exchanged = batch(
                |(operand, output, input_offsets, send_sizes, output_offsets, receive_sizes)| {
                    operand.parallel_ragged_all_to_all(
                        "x",
                        &output,
                        &input_offsets,
                        &send_sizes,
                        &output_offsets,
                        &receive_sizes,
                    )
                },
                (operand, output, input_offsets, send_sizes, output_offsets, receive_sizes),
                (
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                ),
                BatchAxis::new(0),
                BatchAxisSpecification::named("x"),
            )?;
            Ok(exchanged.reduce(&[0, 1], ReductionKind::Sum)?)
        }};
    }

    #[test]
    fn test_parallel_ragged_all_to_all() {
        let operation = ParallelRaggedAllToAllOperation::new("x".to_string(), 4);
        assert_eq!(operation.name(), PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME);
        assert_eq!(operation.axis_name(), "x");
        assert_eq!(operation.axis_size(), 4);
        assert_eq!(operation.axis_index_groups(), None);
        assert_eq!(operation.mesh(), None);
        assert_eq!(operation.effective_axis_size(), Ok(4));
        assert!(!operation.is_physical());
        assert!(!operation.accumulates_updates());
        assert_eq!(operation.to_string(), "parallel_ragged_all_to_all [axis_name=\"x\", axis_size=4]");
        assert_eq!(operation, operation.clone());
        assert_ne!(operation, ParallelRaggedAllToAllOperation::new("y".to_string(), 4));
        assert_ne!(operation, ParallelRaggedAllToAllOperation::new("x".to_string(), 2));
    }

    #[test]
    fn test_parallel_ragged_all_to_all_grouped() {
        // A grouped exchange records an equal-sized exact partition of the full axis, whose common group size is the
        // effective participant count.
        let groups = vec![vec![0, 2], vec![3, 1]];
        let grouped = ParallelRaggedAllToAllOperation::grouped("x".to_string(), 4, groups.clone()).unwrap();
        assert_eq!(grouped.axis_index_groups(), Some(groups.as_slice()));
        assert_eq!(grouped.effective_axis_size(), Ok(2));
        assert_eq!(
            grouped.to_string(),
            "parallel_ragged_all_to_all [axis_name=\"x\", axis_size=4, axis_index_groups=[[0, 2], [3, 1]]]",
        );
        assert_ne!(grouped, ParallelRaggedAllToAllOperation::new("x".to_string(), 4));
        assert_eq!(
            ParallelRaggedAllToAllOperation::grouped("x".to_string(), 4, vec![vec![0, 1], vec![2, 2]]),
            Err(TypeError::invalid(
                "`parallel_ragged_all_to_all` axis index groups contain participant 2 more than once",
            )),
        );
        assert_eq!(
            ParallelRaggedAllToAllOperation::grouped("x".to_string(), 4, vec![vec![0, 1], vec![2]]),
            Err(TypeError::invalid(
                "`parallel_ragged_all_to_all` axis index group 1 has size 1 but every group must have size 2",
            )),
        );
        assert_eq!(
            ParallelRaggedAllToAllOperation::grouped("x".to_string(), 4, vec![vec![0, 1], vec![2, 4]]),
            Err(TypeError::invalid("`parallel_ragged_all_to_all` axis index 4 is out of bounds for axis size 4")),
        );
        assert_eq!(
            ParallelRaggedAllToAllOperation::grouped("x".to_string(), 4, vec![vec![0, 1]]),
            Err(TypeError::invalid("`parallel_ragged_all_to_all` axis index groups do not contain participant 2")),
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_with_mesh() {
        // An exchange over a manual mesh axis records and renders its mesh.
        let mesh_operation = ParallelRaggedAllToAllOperation::new("x".to_string(), 2).with_mesh(manual_mesh());
        assert_eq!(mesh_operation.mesh(), Some(&manual_mesh()));
        assert_eq!(
            mesh_operation.to_string(),
            "parallel_ragged_all_to_all [axis_name=\"x\", axis_size=2, mesh=['x'=2:manual, 'y'=2:manual]]",
        );
        assert_ne!(mesh_operation, ParallelRaggedAllToAllOperation::new("x".to_string(), 2));
    }

    #[test]
    fn test_parallel_ragged_all_to_all_with_physical_representation() {
        // The batching-internal physical representation is rendered explicitly and keeps the overwrite semantics.
        let operation = ParallelRaggedAllToAllOperation::new("x".to_string(), 4);
        let physical = operation.with_physical_representation();
        assert!(physical.is_physical());
        assert!(!physical.accumulates_updates());
        assert_eq!(
            physical.to_string(),
            "parallel_ragged_all_to_all [axis_name=\"x\", axis_size=4, representation=Physical]",
        );
        assert_ne!(physical, operation);
    }

    #[test]
    fn test_parallel_ragged_all_to_all_with_additive_updates() {
        // The transpose-internal additive updates are rendered explicitly and keep the logical representation.
        let operation = ParallelRaggedAllToAllOperation::new("x".to_string(), 4);
        let additive = operation.with_additive_updates();
        assert!(!additive.is_physical());
        assert!(additive.accumulates_updates());
        assert_eq!(additive.to_string(), "parallel_ragged_all_to_all [axis_name=\"x\", axis_size=4, update_kind=Add]");
        assert_ne!(additive, operation);

        // Additive updates compose with the physical representation, which renders first.
        let internal = operation.with_physical_representation().with_additive_updates();
        assert!(internal.is_physical());
        assert!(internal.accumulates_updates());
        assert_eq!(
            internal.to_string(),
            "parallel_ragged_all_to_all [axis_name=\"x\", axis_size=4, representation=Physical, update_kind=Add]",
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_type_inference() {
        // A well-formed exchange returns the type of its `output` seed.
        let data = || ArrayType::new_static(DataType::F32, [3, 2]);
        let output = || ArrayType::new_static(DataType::F32, [4, 2]);
        let metadata = || ArrayType::new_static(DataType::I32, [2]);
        let metadata_length = DimensionVariable::new("metadata_length", DimensionBounds::positive(Some(4)).unwrap());
        let dynamic_metadata = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(metadata_length)]));
        check_operation_type_inference!(
            operation = ParallelRaggedAllToAllOperation::new("x".to_string(), 2),
            cases = [
                {
                    input_types = [data(), output(), metadata(), metadata(), metadata(), metadata()],
                    output_types = [output()],
                },
                // The data inputs must have a leading segment axis, one shared data type, and matching trailing
                // dimensions.
                {
                    input_types = [
                        ArrayType::scalar(DataType::F32),
                        output(),
                        metadata(),
                        metadata(),
                        metadata(),
                        metadata(),
                    ],
                    error = "`parallel_ragged_all_to_all` data inputs must have rank at least 1 but got `f32[]` and \
                             `f32[4, 2]`",
                },
                {
                    input_types = [
                        data(),
                        ArrayType::new_static(DataType::F64, [4, 2]),
                        metadata(),
                        metadata(),
                        metadata(),
                        metadata(),
                    ],
                    error = "`parallel_ragged_all_to_all` `operand` and `output` data types must match but got `f32` \
                             and `f64`",
                },
                {
                    input_types = [
                        data(),
                        ArrayType::new_static(DataType::F32, [4, 3]),
                        metadata(),
                        metadata(),
                        metadata(),
                        metadata(),
                    ],
                    error = "`parallel_ragged_all_to_all` `operand` and `output` trailing dimensions must match but \
                             got `[3, 2]` and `[4, 3]`",
                },
                // The metadata inputs must be rank-one integer arrays of one shared data type. A logical exchange also
                // rejects inputs shaped like the physical representation, whose metadata carry a participant axis.
                {
                    input_types = [
                        data(),
                        output(),
                        ArrayType::new_static(DataType::I32, [2, 1]),
                        metadata(),
                        metadata(),
                        metadata(),
                    ],
                    error = "`parallel_ragged_all_to_all` `input_offsets` must be rank 1 but got `i32[2, 1]`",
                },
                {
                    input_types = [
                        ArrayType::new_static(DataType::F32, [2, 3]),
                        ArrayType::new_static(DataType::F32, [4, 3]),
                        ArrayType::new_static(DataType::I32, [2, 2]),
                        ArrayType::new_static(DataType::I32, [2, 2]),
                        ArrayType::new_static(DataType::I32, [2, 2]),
                        ArrayType::new_static(DataType::I32, [2, 2]),
                    ],
                    error = "`parallel_ragged_all_to_all` `input_offsets` must be rank 1 but got `i32[2, 2]`",
                },
                {
                    input_types = [
                        data(),
                        output(),
                        ArrayType::new_static(DataType::F32, [2]),
                        metadata(),
                        metadata(),
                        metadata(),
                    ],
                    error = "`parallel_ragged_all_to_all` `input_offsets` must have an integer data type but got `f32`",
                },
                {
                    input_types = [
                        data(),
                        output(),
                        metadata(),
                        ArrayType::new_static(DataType::I64, [2]),
                        metadata(),
                        metadata(),
                    ],
                    error = "`parallel_ragged_all_to_all` metadata inputs must share one integer data type but \
                             `input_offsets` has `i32` and `send_sizes` has `i64`",
                },
                // The metadata lengths must be equal, static, positive, and divisible by the participant-group size.
                {
                    input_types = [
                        data(),
                        output(),
                        metadata(),
                        ArrayType::new_static(DataType::I32, [4]),
                        metadata(),
                        metadata(),
                    ],
                    error = "`parallel_ragged_all_to_all` metadata inputs must have equal lengths but `input_offsets` \
                             has length 2 and `send_sizes` has length 4",
                },
                {
                    input_types = [
                        data(),
                        output(),
                        dynamic_metadata.clone(),
                        dynamic_metadata.clone(),
                        dynamic_metadata.clone(),
                        dynamic_metadata.clone(),
                    ],
                    error = format!(
                        "`parallel_ragged_all_to_all` `input_offsets` must have a static length but got \
                         `{dynamic_metadata}`",
                    ),
                },
                {
                    input_types = [
                        data(),
                        output(),
                        ArrayType::new_static(DataType::I32, [0]),
                        ArrayType::new_static(DataType::I32, [0]),
                        ArrayType::new_static(DataType::I32, [0]),
                        ArrayType::new_static(DataType::I32, [0]),
                    ],
                    error = "`parallel_ragged_all_to_all` metadata length must be greater than zero",
                },
                {
                    input_types = [
                        data(),
                        output(),
                        ArrayType::new_static(DataType::I32, [3]),
                        ArrayType::new_static(DataType::I32, [3]),
                        ArrayType::new_static(DataType::I32, [3]),
                        ArrayType::new_static(DataType::I32, [3]),
                    ],
                    error = "`parallel_ragged_all_to_all` metadata length 3 is not divisible by group size 2",
                },
            ],
        );

        // A grouped exchange requires divisibility by its group size rather than by the full axis size.
        check_operation_type_inference!(
            operation =
                ParallelRaggedAllToAllOperation::grouped("x".to_string(), 4, vec![vec![0, 1], vec![2, 3]]).unwrap(),
            cases = [
                {
                    input_types = [data(), output(), metadata(), metadata(), metadata(), metadata()],
                    output_types = [output()],
                },
                {
                    input_types = [
                        data(),
                        output(),
                        ArrayType::new_static(DataType::I32, [3]),
                        ArrayType::new_static(DataType::I32, [3]),
                        ArrayType::new_static(DataType::I32, [3]),
                        ArrayType::new_static(DataType::I32, [3]),
                    ],
                    error = "`parallel_ragged_all_to_all` metadata length 3 is not divisible by group size 2",
                },
            ],
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_type_inference_physical_representation() {
        // The batching-internal representation prefixes every input with a static participant axis whose extent equals
        // the axis size. Type inference validates the public contract without that axis and returns the physical type
        // of `output`.
        let participant_extent = DimensionVariable::new("participants", DimensionBounds::new(0, Some(4)).unwrap());
        let metadata = || ArrayType::new_static(DataType::I32, [2, 2]);
        check_operation_type_inference!(
            operation = ParallelRaggedAllToAllOperation::new("x".to_string(), 2).with_physical_representation(),
            cases = [
                {
                    input_types = [
                        ArrayType::new_static(DataType::F32, [2, 3, 2]),
                        ArrayType::new_static(DataType::F32, [2, 4, 2]),
                        metadata(),
                        metadata(),
                        metadata(),
                        metadata(),
                    ],
                    output_types = [ArrayType::new_static(DataType::F32, [2, 4, 2])],
                },
                {
                    input_types = [
                        ArrayType::new(
                            DataType::F32,
                            Shape::new(vec![
                                Dimension::Dynamic(participant_extent),
                                Dimension::Static(3),
                                Dimension::Static(2),
                            ]),
                        ),
                        ArrayType::new_static(DataType::F32, [2, 4, 2]),
                        metadata(),
                        metadata(),
                        metadata(),
                        metadata(),
                    ],
                    error = "`parallel_ragged_all_to_all` physical input 0 must have a static leading participant \
                             dimension",
                },
                {
                    input_types = [
                        ArrayType::new_static(DataType::F32, [2, 3, 2]),
                        ArrayType::new_static(DataType::F32, [3, 4, 2]),
                        metadata(),
                        metadata(),
                        metadata(),
                        metadata(),
                    ],
                    error = "`parallel_ragged_all_to_all` physical input 1 leading participant dimension 3 must equal \
                             axis size 2",
                },
                {
                    input_types = [
                        ArrayType::new_static(DataType::F32, [2, 3, 2]),
                        ArrayType::new_static(DataType::F32, [2, 4, 2]),
                        ArrayType::new_static(DataType::I32, [2]),
                        metadata(),
                        metadata(),
                        metadata(),
                    ],
                    error = "`parallel_ragged_all_to_all` `input_offsets` must be rank 1 but got `i32[]`",
                },
            ],
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_type_inference_manual_variation() {
        // Every device computes its result from all six inputs, so they must vary over the same manual axes, which the
        // result inherits. An ordinary exchange also accepts inputs that are invariant over a manual mesh axis with
        // the same name, because a `batch` level that shadows that mesh axis may bind it.
        let mesh = manual_mesh();
        let invariant = Sharding::replicated(mesh.clone(), 1);
        let varying = invariant.clone().with_varying_manual_axes(["x"]).unwrap();
        let operand =
            |sharding: &Sharding| ArrayType::new_static(DataType::F32, [3]).with_sharding(sharding.clone()).unwrap();
        let output =
            |sharding: &Sharding| ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding.clone()).unwrap();
        let metadata =
            |sharding: &Sharding| ArrayType::new_static(DataType::I32, [2]).with_sharding(sharding.clone()).unwrap();
        check_operation_type_inference!(
            operation = ParallelRaggedAllToAllOperation::new("x".to_string(), 2),
            cases = [
                {
                    input_types = [
                        operand(&varying),
                        output(&varying),
                        metadata(&varying),
                        metadata(&varying),
                        metadata(&varying),
                        metadata(&varying),
                    ],
                    output_types = [output(&varying)],
                },
                {
                    input_types = [
                        operand(&invariant),
                        output(&invariant),
                        metadata(&invariant),
                        metadata(&invariant),
                        metadata(&invariant),
                        metadata(&invariant),
                    ],
                    output_types = [output(&invariant)],
                },
                {
                    input_types = [
                        operand(&varying),
                        output(&invariant),
                        metadata(&varying),
                        metadata(&varying),
                        metadata(&varying),
                        metadata(&varying),
                    ],
                    error = "`parallel_ragged_all_to_all` inputs must have matching varying manual axes; insert \
                             `parallel_vary` on the inputs that lack an axis, as `align_manual_variation` does",
                },
            ],
        );

        // An exchange over a manual mesh axis can give the receivers different segments, so every input must carry
        // the operation's mesh and vary over its axis, and a pending sum over that same axis is rejected.
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let other_varying = Sharding::replicated(other_mesh, 1).with_varying_manual_axes(["x"]).unwrap();
        let unreduced = invariant.clone().with_unreduced_axes(["x"]).unwrap();
        check_operation_type_inference!(
            operation = ParallelRaggedAllToAllOperation::new("x".to_string(), 2).with_mesh(mesh.clone()),
            cases = [
                {
                    input_types = [
                        operand(&varying),
                        output(&varying),
                        metadata(&varying),
                        metadata(&varying),
                        metadata(&varying),
                        metadata(&varying),
                    ],
                    output_types = [output(&varying)],
                },
                {
                    input_types = [
                        operand(&varying),
                        output(&invariant),
                        metadata(&varying),
                        metadata(&varying),
                        metadata(&varying),
                        metadata(&varying),
                    ],
                    error = "`parallel_ragged_all_to_all` inputs must vary over manual axis `x`; pass invariant values \
                             through `parallel_vary` first so that the exchanged output is typed as varying",
                },
                {
                    input_types = [
                        operand(&varying),
                        output(&varying),
                        ArrayType::new_static(DataType::I32, [2]),
                        metadata(&varying),
                        metadata(&varying),
                        metadata(&varying),
                    ],
                    error = "`parallel_ragged_all_to_all` input must carry a mesh containing manual axis `x`",
                },
                {
                    input_types = [
                        operand(&other_varying),
                        output(&varying),
                        metadata(&varying),
                        metadata(&varying),
                        metadata(&varying),
                        metadata(&varying),
                    ],
                    error = "`parallel_ragged_all_to_all` input mesh does not match the operation mesh",
                },
                {
                    input_types = [
                        operand(&unreduced),
                        output(&unreduced),
                        metadata(&varying),
                        metadata(&varying),
                        metadata(&varying),
                        metadata(&varying),
                    ],
                    error = "`parallel_ragged_all_to_all` does not support unreduced inputs",
                },
            ],
        );

        // The mesh axis must be manual, and its size must equal the axis size of the operation.
        let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        check_operation_type_inference!(
            operation = ParallelRaggedAllToAllOperation::new("x".to_string(), 2).with_mesh(explicit_mesh),
            cases = [{
                input_types = [
                    operand(&varying),
                    output(&varying),
                    metadata(&varying),
                    metadata(&varying),
                    metadata(&varying),
                    metadata(&varying),
                ],
                error = "`parallel_ragged_all_to_all` mesh axis `x` must be manual",
            }],
        );
        check_operation_type_inference!(
            operation = ParallelRaggedAllToAllOperation::new("x".to_string(), 1).with_mesh(mesh),
            cases = [{
                input_types = [
                    operand(&varying),
                    output(&varying),
                    metadata(&varying),
                    metadata(&varying),
                    metadata(&varying),
                    metadata(&varying),
                ],
                error = "`parallel_ragged_all_to_all` axis size 1 does not match the size of manual mesh axis `x`",
            }],
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_type_inference_reduction_state() {
        // For fixed metadata, the exchange is jointly linear in `operand` and `output`, so it preserves a pending sum
        // that both carry, but rejects mismatched reduction state and metadata that are themselves unreduced.
        let mesh = manual_mesh();
        let varying = Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap();
        let pending = varying.clone().with_unreduced_axes(["y"]).unwrap();
        let operand =
            |sharding: &Sharding| ArrayType::new_static(DataType::F32, [3]).with_sharding(sharding.clone()).unwrap();
        let output =
            |sharding: &Sharding| ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding.clone()).unwrap();
        let metadata =
            |sharding: &Sharding| ArrayType::new_static(DataType::I32, [2]).with_sharding(sharding.clone()).unwrap();
        check_operation_type_inference!(
            operation = ParallelRaggedAllToAllOperation::new("x".to_string(), 2).with_mesh(mesh),
            cases = [
                {
                    input_types = [
                        operand(&pending),
                        output(&pending),
                        metadata(&varying),
                        metadata(&varying),
                        metadata(&varying),
                        metadata(&varying),
                    ],
                    output_types = [output(&pending)],
                },
                {
                    input_types = [
                        operand(&pending),
                        output(&varying),
                        metadata(&varying),
                        metadata(&varying),
                        metadata(&varying),
                        metadata(&varying),
                    ],
                    error = "`parallel_ragged_all_to_all` `operand` and `output` must carry identical reduction state",
                },
                {
                    input_types = [
                        operand(&varying),
                        output(&varying),
                        metadata(&varying),
                        metadata(&pending),
                        metadata(&varying),
                        metadata(&varying),
                    ],
                    error = "`parallel_ragged_all_to_all` `send_sizes` must not carry unreduced state",
                },
            ],
        );

        // Every shard must also route the same way, so metadata must not be sharded over a reduction axis of the data
        // inputs.
        let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let unreduced = Sharding::replicated(explicit_mesh.clone(), 1).with_unreduced_axes(["y"]).unwrap();
        let sharded = Sharding::new(explicit_mesh, vec![ShardingDimension::sharded(["y"])]).unwrap();
        check_operation_type_inference!(
            operation = ParallelRaggedAllToAllOperation::new("x".to_string(), 2),
            cases = [{
                input_types = [
                    operand(&unreduced),
                    output(&unreduced),
                    metadata(&sharded),
                    ArrayType::new_static(DataType::I32, [2]),
                    ArrayType::new_static(DataType::I32, [2]),
                    ArrayType::new_static(DataType::I32, [2]),
                ],
                error = "`parallel_ragged_all_to_all` `input_offsets` must be invariant over the reduction axes of \
                         `operand` and `output`",
            }],
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_type_inference_array_ir() {
        // The composite carrier projects its inputs onto the homogeneous array contract and wraps the inferred type.
        let operation = ParallelRaggedAllToAllOperation::new("x".to_string(), 2);
        let metadata = || ArrayIrType::Array(ArrayType::new_static(DataType::I32, [2]));
        let input_types = [
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3, 2])),
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [4, 2])),
            metadata(),
            metadata(),
            metadata(),
            metadata(),
        ];
        assert_eq!(
            operation.infer_parent_output_types(&input_types, &[]),
            Ok(vec![ArrayIrType::Array(ArrayType::new_static(DataType::F32, [4, 2]))]),
        );
        assert_eq!(
            operation.infer_parent_output_types(
                &[
                    ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3, 2])),
                    ArrayIrType::Array(ArrayType::new_static(DataType::F64, [4, 2])),
                    metadata(),
                    metadata(),
                    metadata(),
                    metadata(),
                ],
                &[],
            ),
            Err(TypeError::invalid(
                "`parallel_ragged_all_to_all` `operand` and `output` data types must match but got `f32` and `f64`",
            )),
        );

        // The exchange has no regions, so it requests no region input types from its composite parent, whose inputs
        // must still project to arrays.
        assert_eq!(operation.infer_parent_region_input_types(&input_types, &[]), Ok(Vec::new()));
        let mut invalid_input_types = input_types.to_vec();
        invalid_input_types[0] = ArrayIrType::Dimension(DimensionType::from(DimensionVariable::new(
            "extent",
            DimensionBounds::new(0, Some(4)).unwrap(),
        )));
        assert_eq!(
            operation.infer_parent_region_input_types(&invalid_input_types, &[]),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_type_inference_array_ir_identity_renaming() {
        // The exchange records no type identities of its own, so renaming the identities of its composite parent
        // leaves it unchanged.
        let operation =
            ParallelRaggedAllToAllOperation::grouped("x".to_string(), 4, vec![vec![0, 2], vec![3, 1]]).unwrap();
        let mut renaming = TypeIdentityRenaming::new();
        renaming
            .insert(
                DimensionVariable::new("source", DimensionBounds::unbounded()),
                DimensionVariable::new("target", DimensionBounds::unbounded()),
            )
            .unwrap();
        assert_eq!(operation.rename_parent_type_identities(&renaming), Ok(operation));
    }

    #[test]
    fn test_parallel_ragged_all_to_all_interpretation() {
        // A single participant copies its own segments into the output seed, leaving the remaining rows unchanged.
        assert_eq!(
            interpret_single_participant_parallel_ragged_all_to_all(vec![1], vec![2], vec![0], vec![2]),
            Ok(Array::vector(vec![2.0f32, 3.0, 9.0, 9.0]).unwrap()),
        );

        // Send regions may overlap, which resends the same source row into disjoint received regions.
        assert_eq!(
            interpret_single_participant_parallel_ragged_all_to_all(vec![0, 0], vec![1, 1], vec![0, 1], vec![1, 1]),
            Ok(Array::vector(vec![1.0f32, 1.0, 9.0, 9.0]).unwrap()),
        );

        // The complete exchange is validated before anything is copied: metadata must be non-negative, every region
        // must be in bounds without overflowing, send and receive sizes must agree, and received regions must be
        // disjoint.
        assert_eq!(
            interpret_single_participant_parallel_ragged_all_to_all(vec![-1], vec![1], vec![0], vec![1]),
            Err(ProgramError::InvalidArgument {
                message: "`input_offsets[0]` must be non-negative but got -1".to_string(),
            }),
        );
        assert_eq!(
            interpret_single_participant_parallel_ragged_all_to_all(vec![2], vec![2], vec![0], vec![2]),
            Err(ProgramError::InvalidArgument {
                message: "`parallel_ragged_all_to_all` input region [2, 4) for participant 0 exceeds input extent 3"
                    .to_string(),
            }),
        );
        assert_eq!(
            interpret_single_participant_parallel_ragged_all_to_all(vec![0], vec![2], vec![0], vec![1]),
            Err(ProgramError::InvalidArgument {
                message: "`parallel_ragged_all_to_all` send size 2 from participant 0 to participant 0 does not match \
                          receive size 1"
                    .to_string(),
            }),
        );
        assert_eq!(
            interpret_single_participant_parallel_ragged_all_to_all(vec![0], vec![2], vec![3], vec![2]),
            Err(ProgramError::InvalidArgument {
                message: "`parallel_ragged_all_to_all` output region [3, 5) for participant 0 exceeds output extent 4"
                    .to_string(),
            }),
        );
        assert_eq!(
            interpret_single_participant_parallel_ragged_all_to_all(vec![0, 1], vec![1, 1], vec![0, 0], vec![1, 1]),
            Err(ProgramError::InvalidArgument {
                message:
                    "`parallel_ragged_all_to_all` received output regions [0, 1) and [0, 1) overlap for participant 0"
                        .to_string(),
            }),
        );
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        assert_eq!(
            context.bind(
                ParallelRaggedAllToAllOperation::new("x".to_string(), 1),
                Vec::new(),
                &[
                    Array::vector(vec![1.0f32]).unwrap(),
                    Array::vector(vec![0.0f32]).unwrap(),
                    Array::vector(vec![u64::MAX]).unwrap(),
                    Array::vector(vec![1u64]).unwrap(),
                    Array::vector(vec![0u64]).unwrap(),
                    Array::vector(vec![1u64]).unwrap(),
                ],
            ),
            Err(ProgramError::InvalidArgument {
                message:
                    "`parallel_ragged_all_to_all` input region for participant 0 at metadata index 0 overflows `usize`"
                        .to_string(),
            }),
        );

        // Outside any binder, an exchange is degenerate only when its participant groups are singletons, whatever the
        // size of the full axis.
        let inputs = [
            Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap(),
            Array::vector(vec![9.0f32, 9.0, 9.0, 9.0]).unwrap(),
            Array::vector(vec![1i64, 0]).unwrap(),
            Array::vector(vec![2i64, 0]).unwrap(),
            Array::vector(vec![0i64, 2]).unwrap(),
            Array::vector(vec![2i64, 0]).unwrap(),
        ];
        assert_eq!(
            context.bind(
                ParallelRaggedAllToAllOperation::grouped("x".to_string(), 2, vec![vec![0], vec![1]]).unwrap(),
                Vec::new(),
                &inputs,
            ),
            Ok(vec![Array::vector(vec![2.0f32, 3.0, 9.0, 9.0]).unwrap()]),
        );
        assert_eq!(
            context.bind(ParallelRaggedAllToAllOperation::new("x".to_string(), 2), Vec::new(), &inputs),
            Err(ProgramError::UnsupportedOperation {
                message:
                    "cannot interpret `parallel_ragged_all_to_all` over axis `x` of size 2 without an enclosing binder"
                        .to_string(),
            }),
        );
        assert_eq!(
            context.bind(
                ParallelRaggedAllToAllOperation::grouped("x".to_string(), 4, vec![vec![0, 1], vec![2, 3]]).unwrap(),
                Vec::new(),
                &inputs,
            ),
            Err(ProgramError::UnsupportedOperation {
                message: "cannot interpret `parallel_ragged_all_to_all` over axis `x` of size 4 with participant \
                          groups of size 2 without an enclosing binder"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_interpretation_skips_zero_byte_transfers() {
        // Zero-length segments at maximal offsets stay within `usize::MAX`-row extents of zero-byte rows, so the
        // physical exchange copies nothing and never computes an overflowing byte offset.
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let data_type = ArrayType::new_static(DataType::F32, [3, usize::MAX, 0]);
        let data = || Array::from_elements::<f32>(data_type.clone(), &[]).unwrap();
        let metadata_type = ArrayType::new_static(DataType::U64, [3, 3]);
        let offsets = Array::from_elements(metadata_type.clone(), &[u64::MAX; 9]).unwrap();
        let sizes = Array::from_elements(metadata_type, &[0u64; 9]).unwrap();
        assert_eq!(
            context.bind(
                ParallelRaggedAllToAllOperation::new("x".to_string(), 3).with_physical_representation(),
                Vec::new(),
                &[data(), data(), offsets.clone(), sizes.clone(), offsets, sizes],
            ),
            Ok(vec![data()]),
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_interpretation_additive_updates() {
        // Received segments add into the output seed, so overlapping received regions accumulate instead of being
        // rejected.
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        assert_eq!(
            context.bind(
                ParallelRaggedAllToAllOperation::new("x".to_string(), 1).with_additive_updates(),
                Vec::new(),
                &[
                    Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap(),
                    Array::vector(vec![9.0f32, 9.0, 9.0, 9.0]).unwrap(),
                    Array::vector(vec![0i64, 1]).unwrap(),
                    Array::vector(vec![2i64, 2]).unwrap(),
                    Array::vector(vec![0i64, 1]).unwrap(),
                    Array::vector(vec![2i64, 2]).unwrap(),
                ],
            ),
            Ok(vec![Array::vector(vec![10.0f32, 13.0, 12.0, 9.0]).unwrap()]),
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_interpretation_array_ir() {
        // A composite value interprets the exchange through its array member.
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        assert_eq!(
            context.bind(
                ParallelRaggedAllToAllOperation::new("x".to_string(), 1),
                Vec::new(),
                &[
                    ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()),
                    ArrayIrValue::Array(Array::vector(vec![9.0f32, 9.0, 9.0, 9.0]).unwrap()),
                    ArrayIrValue::Array(Array::vector(vec![1i64]).unwrap()),
                    ArrayIrValue::Array(Array::vector(vec![2i64]).unwrap()),
                    ArrayIrValue::Array(Array::vector(vec![0i64]).unwrap()),
                    ArrayIrValue::Array(Array::vector(vec![2i64]).unwrap()),
                ],
            ),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![2.0f32, 3.0, 9.0, 9.0]).unwrap())]),
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_partial_evaluation() {
        // A degenerate exchange folds known inputs and residualizes whenever any input is unknown.
        check_operation_partial_evaluation!(
            operation = ParallelRaggedAllToAllOperation::new("x".to_string(), 1),
            inputs = [
                Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap(),
                Array::vector(vec![9.0f32, 9.0, 9.0, 9.0]).unwrap(),
                Array::vector(vec![1i64, 0]).unwrap(),
                Array::vector(vec![2i64, 0]).unwrap(),
                Array::vector(vec![0i64, 2]).unwrap(),
                Array::vector(vec![2i64, 0]).unwrap(),
            ],
            expected = Array::vector(vec![2.0f32, 3.0, 9.0, 9.0]).unwrap(),
        );

        // An exchange with other participants has no eager per-participant value and therefore residualizes even when
        // every input is known, keeping its known metadata as ordinary runtime values.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let inputs = vec![
            builder.add_input(ArrayType::new_static(DataType::F32, [3])),
            builder.add_input(ArrayType::new_static(DataType::F32, [4])),
            builder.add_input(ArrayType::new_static(DataType::I64, [2])),
            builder.add_input(ArrayType::new_static(DataType::I64, [2])),
            builder.add_input(ArrayType::new_static(DataType::I64, [2])),
            builder.add_input(ArrayType::new_static(DataType::I64, [2])),
        ];
        let outputs = builder
            .add_instruction(ParallelRaggedAllToAllOperation::new("x".to_string(), 2), Vec::new(), inputs, None)
            .unwrap()
            .to_vec();
        let exchange =
            builder.build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 6], vec![Placeholder]).unwrap();
        let evaluation = exchange
            .partially_evaluate(&[
                PartialValue::Known(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()),
                PartialValue::Known(Array::vector(vec![9.0f32, 9.0, 9.0, 9.0]).unwrap()),
                PartialValue::Known(Array::vector(vec![1i64, 0]).unwrap()),
                PartialValue::Known(Array::vector(vec![2i64, 0]).unwrap()),
                PartialValue::Known(Array::vector(vec![0i64, 2]).unwrap()),
                PartialValue::Known(Array::vector(vec![2i64, 0]).unwrap()),
            ])
            .unwrap();
        assert_eq!(evaluation.outputs(), &[PartialEvaluationOutput::<Array>::Unknown(0)]);
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[3], %1:f32[4], %2:i64[2], %3:i64[2], %4:i64[2], %5:i64[2] .
                let %6:f32[4] = parallel_ragged_all_to_all [axis_name=\"x\", axis_size=2] %0 %1 %2 %3 %4 %5
                in (%6)"
            },
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_batching() {
        // A matching named `batch` level is the participant axis of the eager reference implementation, which executes
        // the complete exchange between its batch items.
        let operand = vec![1i32, 2, 2, 3, 4, 0];
        let output_seed = vec![0i32; 8];
        let output: Array = batch(
            |inputs: Vec<_>| {
                inputs[0].parallel_ragged_all_to_all("x", &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5])
            },
            vec![
                Array::matrix(2, 3, operand.clone()).unwrap(),
                Array::matrix(2, 4, output_seed.clone()).unwrap(),
                Array::matrix(2, 2, vec![0i32, 1, 0, 1]).unwrap(),
                Array::matrix(2, 2, vec![1i32, 2, 1, 1]).unwrap(),
                Array::matrix(2, 2, vec![0i32, 0, 1, 2]).unwrap(),
                Array::matrix(2, 2, vec![1i32, 1, 2, 1]).unwrap(),
            ],
            vec![BatchAxis::new(0); 6],
            BatchAxis::new(0),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        let expected = reference_ragged_transfers(
            operand.as_slice(),
            output_seed.as_slice(),
            3,
            4,
            1,
            &[(0, 0, 0, 0, 1), (0, 1, 1, 0, 2), (1, 0, 0, 1, 1), (1, 1, 1, 2, 1)],
        );
        assert_eq!(output, Array::matrix(2, 4, expected).unwrap());

        // Reversed noncontiguous groups, two slices per peer, and width-two rows exercise every routing index and
        // prove that the byte kernel preserves trailing dimensions.
        let groups = vec![vec![3, 1], vec![2, 0]];
        let operand = (0..4)
            .flat_map(|participant| {
                (0..4).flat_map(move |row| [participant * 100 + row * 10, participant * 100 + row * 10 + 1])
            })
            .collect::<Vec<i32>>();
        let output_seed = vec![-1i32; 4 * 5 * 2];
        let output: Array = batch(
            |inputs: Vec<_>| {
                inputs[0].parallel_ragged_all_to_all_with_axis_index_groups(
                    "x",
                    &inputs[1],
                    &inputs[2],
                    &inputs[3],
                    &inputs[4],
                    &inputs[5],
                    groups.clone(),
                )
            },
            vec![
                Array::from_elements(ArrayType::new_static(DataType::I32, [4, 4, 2]), operand.as_slice()).unwrap(),
                Array::from_elements(ArrayType::new_static(DataType::I32, [4, 5, 2]), output_seed.as_slice()).unwrap(),
                Array::matrix(4, 4, [0i32, 1, 2, 3].repeat(4)).unwrap(),
                Array::matrix(4, 4, vec![1i32; 16]).unwrap(),
                Array::matrix(4, 4, vec![2i32, 3, 2, 3, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1]).unwrap(),
                Array::matrix(4, 4, vec![1i32; 16]).unwrap(),
            ],
            vec![BatchAxis::new(0); 6],
            BatchAxis::new(0),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        let expected = reference_ragged_transfers(
            operand.as_slice(),
            output_seed.as_slice(),
            4,
            5,
            2,
            &[
                (3, 0, 3, 0, 1),
                (3, 1, 3, 1, 1),
                (3, 2, 1, 0, 1),
                (3, 3, 1, 1, 1),
                (1, 0, 3, 2, 1),
                (1, 1, 3, 3, 1),
                (1, 2, 1, 2, 1),
                (1, 3, 1, 3, 1),
                (2, 0, 2, 0, 1),
                (2, 1, 2, 1, 1),
                (2, 2, 0, 0, 1),
                (2, 3, 0, 1, 1),
                (0, 0, 2, 2, 1),
                (0, 1, 2, 3, 1),
                (0, 2, 0, 2, 1),
                (0, 3, 0, 3, 1),
            ],
        );
        assert_eq!(
            output,
            Array::from_elements(ArrayType::new_static(DataType::I32, [4, 5, 2]), expected.as_slice()).unwrap(),
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_batching_rejects_unsupported_inputs() {
        let context = eager_collective_context("x", 2);
        let item = |value: Array| ArrayBatch::new(value, BatchAxis::new(0)).unwrap();
        let metadata = || item(Array::matrix(2, 2, vec![0i32; 4]).unwrap());

        // One per-item logical extent does not determine the per-peer sizes and offsets of an explicitly packed
        // exchange, so batching rejects inputs that carry a `RaggedAxis` before choosing any batching path.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let ragged_operand = item(Array::matrix(2, 3, vec![1.0f32; 6]).unwrap())
            .with_ragged_axes(vec![RaggedAxis::new(1, Array::vector(vec![1i32, 3]).unwrap(), length, vec![0])])
            .unwrap();
        assert_eq!(
            ParallelRaggedAllToAllOperation::new("x".to_string(), 2).batch(
                &context,
                &EmptyRegionDriver,
                &[
                    ragged_operand,
                    item(Array::matrix(2, 4, vec![0.0f32; 8]).unwrap()),
                    metadata(),
                    metadata(),
                    metadata(),
                    metadata(),
                ],
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_ragged_all_to_all` does not support bounded ragged dimension `length` on input 0"
                    .to_string(),
            }),
        );

        // An exchange over the manual mesh axis itself belongs to its manual region and requires devices, so a
        // matching `batch` level, which can only shadow that axis, rejects it.
        assert_eq!(
            ParallelRaggedAllToAllOperation::new("x".to_string(), 2).with_mesh(manual_mesh()).batch(
                &context,
                &EmptyRegionDriver,
                &[
                    item(Array::matrix(2, 3, vec![0.0f32; 6]).unwrap()),
                    item(Array::matrix(2, 4, vec![0.0f32; 8]).unwrap()),
                    metadata(),
                    metadata(),
                    metadata(),
                    metadata(),
                ],
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_ragged_all_to_all` over a manual mesh axis cannot bind a named batch axis"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_batching_requires_constant_metadata() {
        // A matching `batch` level executes the exchange eagerly, so it rejects metadata that do not resolve to
        // constants, whether they are staged in a trace or unknown during partial evaluation, while unknown data
        // inputs with known constant metadata are accepted.
        let input_types = vec![
            ArrayType::new_static(DataType::F32, [2, 3]),
            ArrayType::new_static(DataType::F32, [2, 4]),
            ArrayType::new_static(DataType::I32, [2, 2]),
            ArrayType::new_static(DataType::I32, [2, 2]),
            ArrayType::new_static(DataType::I32, [2, 2]),
            ArrayType::new_static(DataType::I32, [2, 2]),
        ];
        let error = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<_>| {
                let context = BatchingContext::new(inputs[0].context().clone(), 2).with_axis_name("x".to_string());
                let inputs = inputs
                    .into_iter()
                    .map(|input| ArrayBatch::new(input, BatchAxis::new(0)))
                    .collect::<Result<Vec<_>, _>>()?;
                let outputs = ParallelRaggedAllToAllOperation::new("x".to_string(), 2)
                    .batch(&context, &EmptyRegionDriver, inputs.as_slice())?
                    .into_parts()
                    .0;
                Ok(outputs[0].value().clone())
            },
            input_types.clone(),
        )
        .unwrap_err();
        let error = error.downcast_custom::<BatchingError>().unwrap();
        assert_eq!(
            error,
            &BatchingError::UnsupportedOperation {
                message:
                    "`parallel_ragged_all_to_all` cannot materialize a batch-bound collective with staged metadata"
                        .to_string(),
            },
        );

        // Metadata that are unknown during partial evaluation are rejected as well.
        let partial_context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let partial_inputs = input_types
            .into_iter()
            .enumerate()
            .map(|(index, r#type)| {
                let value = partial_context.unknown_input(r#type, index);
                ArrayBatch::new(PartialTracer::new(partial_context.clone(), value), BatchAxis::new(0)).unwrap()
            })
            .collect::<Vec<_>>();
        let batching_context = BatchingContext::new(partial_context, 2).with_axis_name("x".to_string());
        assert_eq!(
            ParallelRaggedAllToAllOperation::new("x".to_string(), 2).batch(
                &batching_context,
                &EmptyRegionDriver,
                partial_inputs.as_slice(),
            ),
            Err(BatchingError::UnsupportedOperation {
                message:
                    "`parallel_ragged_all_to_all` cannot materialize a batch-bound collective with staged metadata"
                        .to_string(),
            }),
        );

        // Unknown data inputs with known constant metadata are accepted, and the exchange of unknown data is itself
        // unknown, so partial evaluation residualizes it with the batch axis in front.
        let partial_context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let operand = PartialTracer::new(
            partial_context.clone(),
            partial_context.unknown_input(ArrayType::new_static(DataType::F32, [2, 3]), 0),
        );
        let output = PartialTracer::new(
            partial_context.clone(),
            partial_context.unknown_input(ArrayType::new_static(DataType::F32, [2, 4]), 1),
        );
        let metadata = || {
            PartialTracer::new(
                partial_context.clone(),
                PartialEvaluationValue::known_constant(Array::matrix(2, 2, vec![0i32; 4]).unwrap()),
            )
        };
        let inputs = [operand, output, metadata(), metadata(), metadata(), metadata()]
            .into_iter()
            .map(|input| ArrayBatch::new(input, BatchAxis::new(0)).unwrap())
            .collect::<Vec<_>>();
        let batching_context = BatchingContext::new(partial_context, 2).with_axis_name("x".to_string());
        let outputs = ParallelRaggedAllToAllOperation::new("x".to_string(), 2)
            .batch(&batching_context, &EmptyRegionDriver, inputs.as_slice())
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().r#type().into_owned(), ArrayType::new_static(DataType::F32, [2, 4]));
        assert!(outputs[0].value().value().unwrap().is_unknown());
    }

    #[test]
    fn test_parallel_ragged_all_to_all_batching_shadows_manual_axis() {
        // A `batch` level that binds `x` inside a manual region whose mesh also has an axis `x` shadows that mesh axis.
        // The capability then stages an ordinary exchange, which the `batch` level executes over its own batch items
        // without requiring any variation over the mesh axis.
        let mesh = manual_mesh();
        let typed = |data_type, dimensions: [usize; 2]| {
            ArrayType::new_static(data_type, dimensions)
                .with_sharding(Sharding::replicated(mesh.clone(), 2))
                .unwrap()
        };
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |inputs: Vec<_>| {
                let domain = inputs[0].dispatch_domain();
                let context =
                    BatchingContext::<_, ArrayBatchingPolicy>::new(domain.clone(), 2).with_axis_name("x".to_string());
                let item = |value| -> Result<_, ProgramError> {
                    Ok(BatchingTracer::new(context.clone(), ArrayBatch::new(value, BatchAxis::new(0))?))
                };
                let metadata = || item(domain.lift(Array::matrix(2, 2, vec![0i32; 4]).unwrap())?);
                let operand = item(inputs[0].clone())?;
                let output = item(inputs[1].clone())?;
                let exchanged = operand.parallel_ragged_all_to_all(
                    "x",
                    &output,
                    &metadata()?,
                    &metadata()?,
                    &metadata()?,
                    &metadata()?,
                )?;
                Ok(exchanged.into_batch().into_value())
            },
            vec![typed(DataType::F32, [2, 3]), typed(DataType::F32, [2, 4])],
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
        )
        .unwrap();
        assert_eq!(output_type, typed(DataType::F32, [2, 4]));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 3][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}]}], %1:f32[2, \
                    4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}]}] .
                let %2:i32[2, 2] = const [[0, 0], [0, 0]]
                    %3:i32[2, 2] = const [[0, 0], [0, 0]]
                    %4:i32[2, 2] = const [[0, 0], [0, 0]]
                    %5:i32[2, 2] = const [[0, 0], [0, 0]]
                    %6:f32[2, 4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}]}] = \
                        parallel_ragged_all_to_all [axis_name=\"x\", axis_size=2, representation=Physical] %0 %1 %2 \
                        %3 %4 %5
                in (%6)"
            },
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_batching_unrelated_axis() {
        // An unrelated level forwards an exchange whose inputs are all replicated unchanged. Otherwise, it merges its
        // mapped axis, broadcasting replicated inputs along it. Each batch item here copies its `operand` rows `1..3`
        // into its `output` rows `0..2`, using shared `u64` metadata that need no widening.
        check_operation_batching!(
            @exact,
            operation = ParallelRaggedAllToAllOperation::new("x".to_string(), 1),
            axis_size = 2,
            cases = [
                {
                    inputs = [
                        (@replicated, Array::vector(vec![10.0f64, 11.0, 12.0]).unwrap()),
                        (@replicated, Array::vector(vec![100.0f64, 101.0, 102.0, 103.0]).unwrap()),
                        (@replicated, Array::vector(vec![1i32]).unwrap()),
                        (@replicated, Array::vector(vec![2i32]).unwrap()),
                        (@replicated, Array::vector(vec![0i32]).unwrap()),
                        (@replicated, Array::vector(vec![2i32]).unwrap()),
                    ],
                    outputs = [(@replicated, Array::vector(vec![11.0f64, 12.0, 102.0, 103.0]).unwrap())],
                },
                {
                    inputs = [
                        (@mapped(axis = 0), Array::matrix(2, 3, vec![10.0f64, 11.0, 12.0, 20.0, 21.0, 22.0]).unwrap()),
                        (
                            @mapped(axis = 0),
                            Array::matrix(2, 4, vec![100.0f64, 101.0, 102.0, 103.0, 200.0, 201.0, 202.0, 203.0])
                                .unwrap(),
                        ),
                        (@replicated, Array::vector(vec![1u64]).unwrap()),
                        (@replicated, Array::vector(vec![2u64]).unwrap()),
                        (@replicated, Array::vector(vec![0u64]).unwrap()),
                        (@replicated, Array::vector(vec![2u64]).unwrap()),
                    ],
                    outputs = [(
                        @mapped(axis = 0),
                        Array::matrix(2, 4, vec![11.0f64, 12.0, 102.0, 103.0, 21.0, 22.0, 202.0, 203.0]).unwrap(),
                    )],
                },
            ],
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_batching_unrelated_axis_staging() {
        // A staged merge widens the metadata to `u64`, rebases the input and output offsets of each batch item by its
        // index times the static `operand` and `output` extents, and stages one exchange over the packed values.
        let input_types = vec![
            ArrayType::new_static(DataType::F32, [2, 3]),
            ArrayType::new_static(DataType::F32, [2, 4]),
            ArrayType::new_static(DataType::I8, [2, 2]),
            ArrayType::new_static(DataType::I8, [2, 2]),
            ArrayType::new_static(DataType::I8, [2, 2]),
            ArrayType::new_static(DataType::I8, [2, 2]),
        ];
        let (output_type, program) = trace_unrelated_axis_exchange(input_types).unwrap();

        assert_eq!(output_type, ArrayType::new_static(DataType::F32, [2, 4]));
        assert_eq!(
            std::fmt::from_fn(|formatter| program.render(formatter, 0, ProgramRenderingMode::WithProvenance))
                .to_string(),
            indoc! {"
                lambda %0:f32[2, 3], %1:f32[2, 4], %2:i8[2, 2], %3:i8[2, 2], %4:i8[2, 2], %5:i8[2, 2] .
                let %6:f32[6] = reshape [shape=[6]] %0 ; provenance=ryft::batching::parallel_ragged_all_to_all
                    %7:f32[8] = reshape [shape=[8]] %1 ; provenance=ryft::batching::parallel_ragged_all_to_all
                    %8:u64[2, 2] = convert_element_type [data_type=u64] %2 ; \
                        provenance=ryft::batching::parallel_ragged_all_to_all
                    %9:u64[2, 2] = convert_element_type [data_type=u64] %3 ; \
                        provenance=ryft::batching::parallel_ragged_all_to_all
                    %10:u64[2, 2] = convert_element_type [data_type=u64] %4 ; \
                        provenance=ryft::batching::parallel_ragged_all_to_all
                    %11:u64[2, 2] = convert_element_type [data_type=u64] %5 ; \
                        provenance=ryft::batching::parallel_ragged_all_to_all
                    %12:u64[2, 2] = iota [type=u64[2, 2], dimension=1] ; \
                        provenance=ryft::batching::parallel_ragged_all_to_all
                    %13:u64[] = constant [value=3] ; provenance=ryft::batching::parallel_ragged_all_to_all
                    %14:u64[2, 2] = broadcast [output_type=u64[2, 2], output_axes=[]] %13 ; \
                        provenance=ryft::batching::parallel_ragged_all_to_all
                    %15:u64[2, 2] = mul %12 %14 ; provenance=ryft::batching::parallel_ragged_all_to_all
                    %16:u64[2, 2] = add %8 %15 ; provenance=ryft::batching::parallel_ragged_all_to_all
                    %17:u64[2, 2] = iota [type=u64[2, 2], dimension=1] ; \
                        provenance=ryft::batching::parallel_ragged_all_to_all
                    %18:u64[] = constant [value=4] ; provenance=ryft::batching::parallel_ragged_all_to_all
                    %19:u64[2, 2] = broadcast [output_type=u64[2, 2], output_axes=[]] %18 ; \
                        provenance=ryft::batching::parallel_ragged_all_to_all
                    %20:u64[2, 2] = mul %17 %19 ; provenance=ryft::batching::parallel_ragged_all_to_all
                    %21:u64[2, 2] = add %10 %20 ; provenance=ryft::batching::parallel_ragged_all_to_all
                    %22:u64[4] = reshape [shape=[4]] %16 ; provenance=ryft::batching::parallel_ragged_all_to_all
                    %23:u64[4] = reshape [shape=[4]] %9 ; provenance=ryft::batching::parallel_ragged_all_to_all
                    %24:u64[4] = reshape [shape=[4]] %21 ; provenance=ryft::batching::parallel_ragged_all_to_all
                    %25:u64[4] = reshape [shape=[4]] %11 ; provenance=ryft::batching::parallel_ragged_all_to_all
                    %26:f32[8] = parallel_ragged_all_to_all [axis_name=\"x\", axis_size=2] %6 %7 %22 %23 %24 %25 ; \
                        provenance=ryft::batching::parallel_ragged_all_to_all
                    %27:f32[2, 4] = reshape [shape=[2, 4]] %26 ; provenance=ryft::batching::parallel_ragged_all_to_all
                in (%27)"
            },
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_batching_unrelated_axis_physical_representation() {
        // An unrelated mapped axis merges into the packed data and metadata axes that follow the leading participant
        // axis of the physical representation, so the eager exchange keeps each batch item's segments separate.
        let metadata_type = ArrayType::new_static(DataType::I8, [2, 2, 2]);
        let metadata = |elements: &[i8]| Array::from_elements(metadata_type.clone(), elements).unwrap();
        check_operation_batching!(
            @exact,
            operation = ParallelRaggedAllToAllOperation::new("x".to_string(), 2).with_physical_representation(),
            axis_size = 2,
            cases = [{
                inputs = [
                    (
                        @mapped(axis = 1),
                        Array::from_elements::<f64>(
                            ArrayType::new_static(DataType::F64, [2, 2, 3]),
                            &[10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0, 40.0, 41.0, 42.0],
                        )
                        .unwrap(),
                    ),
                    (
                        @mapped(axis = 1),
                        Array::from_elements::<f64>(
                            ArrayType::new_static(DataType::F64, [2, 2, 4]),
                            &[
                                100.0, 101.0, 102.0, 103.0, 110.0, 111.0, 112.0, 113.0, 200.0, 201.0, 202.0, 203.0,
                                210.0, 211.0, 212.0, 213.0,
                            ],
                        )
                        .unwrap(),
                    ),
                    (@mapped(axis = 2), metadata(&[0, 2, 1, 0, 0, 1, 2, 2])),
                    (@mapped(axis = 2), metadata(&[1; 8])),
                    (@mapped(axis = 2), metadata(&[0, 1, 1, 0, 2, 3, 3, 2])),
                    (@mapped(axis = 2), metadata(&[1; 8])),
                ],
                outputs = [(
                    @mapped(axis = 1),
                    Array::from_elements::<f64>(
                        ArrayType::new_static(DataType::F64, [2, 2, 4]),
                        &[
                            10.0, 101.0, 30.0, 103.0, 110.0, 22.0, 112.0, 41.0, 200.0, 11.0, 202.0, 32.0, 20.0, 211.0,
                            42.0, 213.0,
                        ],
                    )
                    .unwrap(),
                )],
            }],
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_batching_unrelated_axis_grouped() {
        // Merging an unrelated mapped axis keeps the metadata block of every receiver contiguous, so a grouped exchange
        // inside an unrelated `y` level routes each batch item independently within the participant groups of `x`.
        // Each participant sends its first row to the first participant of its group and its second row to the second,
        // at the receiver row that matches its own group position. Participant `p` of batch item `b` holds the rows
        // `[10·p + 100·b, 10·p + 100·b + 1]`.
        let data_type = ArrayType::new_static(DataType::F32, [4, 2, 2]);
        let metadata_type = ArrayType::new_static(DataType::I32, [4, 2, 2]);
        let inputs = (
            Array::from_elements(
                data_type.clone(),
                &[
                    0.0f32, 1.0, 100.0, 101.0, 10.0, 11.0, 110.0, 111.0, 20.0, 21.0, 120.0, 121.0, 30.0, 31.0, 130.0,
                    131.0,
                ],
            )
            .unwrap(),
            Array::from_elements(data_type.clone(), &[-1.0f32; 16]).unwrap(),
            Array::from_elements(metadata_type.clone(), &[0i32, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]).unwrap(),
            Array::from_elements(metadata_type.clone(), &[1i32; 16]).unwrap(),
            Array::from_elements(metadata_type.clone(), &[0i32, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]).unwrap(),
            Array::from_elements(metadata_type, &[1i32; 16]).unwrap(),
        );
        let exchanged: Array = batch(
            |inputs| {
                Ok(batch(
                    |(operand, output, input_offsets, send_sizes, output_offsets, receive_sizes)| {
                        operand.parallel_ragged_all_to_all_with_axis_index_groups(
                            "x",
                            &output,
                            &input_offsets,
                            &send_sizes,
                            &output_offsets,
                            &receive_sizes,
                            vec![vec![0, 2], vec![3, 1]],
                        )
                    },
                    inputs,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("y"),
                )?)
            },
            inputs,
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        assert_eq!(
            exchanged,
            Array::from_elements(
                data_type,
                &[
                    0.0f32, 20.0, 100.0, 120.0, 31.0, 11.0, 131.0, 111.0, 1.0, 21.0, 101.0, 121.0, 30.0, 10.0, 130.0,
                    110.0
                ],
            )
            .unwrap(),
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_batching_unrelated_axis_manual_variation() {
        // Merging an unrelated mapped axis into an exchange over a manual mesh axis rebases the varying metadata by
        // freshly created offsets, which must share the variation of the metadata so that the rebasing and the merged
        // exchange stay well-typed.
        let mesh = manual_mesh();
        let typed = |data_type, dimensions: [usize; 2]| {
            ArrayType::new_static(data_type, dimensions)
                .with_sharding(Sharding::replicated(mesh.clone(), 2).with_varying_manual_axes(["x"]).unwrap())
                .unwrap()
        };
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |inputs: Vec<_>| {
                let context = BatchingContext::<_, ArrayBatchingPolicy>::new(inputs[0].dispatch_domain(), 2)
                    .with_axis_name("y".to_string());
                let inputs = inputs
                    .into_iter()
                    .map(|input| Ok(BatchingTracer::new(context.clone(), ArrayBatch::new(input, BatchAxis::new(0))?)))
                    .collect::<Result<Vec<_>, ProgramError>>()?;
                let exchanged = inputs[0]
                    .parallel_ragged_all_to_all("x", &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5])?;
                Ok(exchanged.into_batch().into_value())
            },
            vec![
                typed(DataType::F32, [2, 3]),
                typed(DataType::F32, [2, 4]),
                typed(DataType::I32, [2, 2]),
                typed(DataType::I32, [2, 2]),
                typed(DataType::I32, [2, 2]),
                typed(DataType::I32, [2, 2]),
            ],
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
        )
        .unwrap();
        assert_eq!(output_type, typed(DataType::F32, [2, 4]));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda \
                %0:f32[2, 3][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x'}}], \
                %1:f32[2, 4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x'}}], \
                %2:i32[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x'}}], \
                %3:i32[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x'}}], \
                %4:i32[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x'}}], \
                %5:i32[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x'}}] .
                let %6:f32[6][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        reshape [shape=[6]] %0
                    %7:f32[8][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        reshape [shape=[8]] %1
                    %8:i32[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                        transpose [permutation=[1, 0]] %2
                    %9:u64[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                        convert_element_type [data_type=u64] %8
                    %10:i32[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                        transpose [permutation=[1, 0]] %3
                    %11:u64[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                        convert_element_type [data_type=u64] %10
                    %12:i32[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                        transpose [permutation=[1, 0]] %4
                    %13:u64[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                        convert_element_type [data_type=u64] %12
                    %14:i32[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                        transpose [permutation=[1, 0]] %5
                    %15:u64[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                        convert_element_type [data_type=u64] %14
                    %16:u64[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                        iota [
                        type=u64[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x'}}],
                        dimension=1,
                    ]
                    %17:u64[][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [], varying_manual={'x'}}] = \
                        constant [value=3]
                    %18:u64[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                        broadcast [
                        output_type=u64[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], \
                            varying_manual={'x'}}],
                        output_axes=[],
                    ] %17
                    %19:u64[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                        mul %16 %18
                    %20:u64[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                        add %9 %19
                    %21:u64[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                        iota [
                        type=u64[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x'}}],
                        dimension=1,
                    ]
                    %22:u64[][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [], varying_manual={'x'}}] = \
                        constant [value=4]
                    %23:u64[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                        broadcast [
                        output_type=u64[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], \
                            varying_manual={'x'}}],
                        output_axes=[],
                    ] %22
                    %24:u64[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                        mul %21 %23
                    %25:u64[2, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                        add %13 %24
                    %26:u64[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        reshape [shape=[4]] %20
                    %27:u64[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        reshape [shape=[4]] %11
                    %28:u64[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        reshape [shape=[4]] %25
                    %29:u64[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        reshape [shape=[4]] %15
                    %30:f32[8][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_ragged_all_to_all [axis_name=\"x\", axis_size=2, mesh=['x'=2:manual, 'y'=2:manual]] \
                        %6 %7 %26 %27 %28 %29
                    %31:f32[2, 4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                        reshape [
                        shape=[2, 4],
                        output_sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x'}},
                    ] %30
                in (%31)"
            },
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_batching_unrelated_axis_empty_batch() {
        // An empty mapped axis exchanges nothing and returns the aligned output seed, but the logical input types are
        // still validated.
        let context = eager_collective_context("y", 0);
        let empty = |r#type: ArrayType| Array::from_elements::<f32>(r#type, &[]).unwrap();
        let empty_metadata = || Array::from_elements::<i32>(ArrayType::new_static(DataType::I32, [2, 0]), &[]).unwrap();
        let inputs = vec![
            ArrayBatch::new(empty(ArrayType::new_static(DataType::F32, [0, 3])), BatchAxis::new(0)).unwrap(),
            ArrayBatch::new(empty(ArrayType::new_static(DataType::F32, [0, 4])), BatchAxis::new(0)).unwrap(),
            ArrayBatch::new(empty_metadata(), BatchAxis::new(1)).unwrap(),
            ArrayBatch::new(empty_metadata(), BatchAxis::new(1)).unwrap(),
            ArrayBatch::new(empty_metadata(), BatchAxis::new(1)).unwrap(),
            ArrayBatch::new(empty_metadata(), BatchAxis::new(1)).unwrap(),
        ];
        let outputs = ParallelRaggedAllToAllOperation::new("x".to_string(), 2)
            .batch(&context, &EmptyRegionDriver, inputs.as_slice())
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(
            outputs,
            vec![ArrayBatch::new(empty(ArrayType::new_static(DataType::F32, [0, 4])), BatchAxis::new(0)).unwrap()],
        );

        // Mismatched data types are still rejected for an empty mapped axis.
        let mut invalid_inputs = inputs;
        invalid_inputs[1] = ArrayBatch::new(
            Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [0, 4]), &[]).unwrap(),
            BatchAxis::new(0),
        )
        .unwrap();
        assert_eq!(
            ParallelRaggedAllToAllOperation::new("x".to_string(), 2).batch(
                &context,
                &EmptyRegionDriver,
                invalid_inputs.as_slice(),
            ),
            Err(BatchingError::Type(TypeError::invalid(
                "`parallel_ragged_all_to_all` `operand` and `output` data types must match but got `f32` and `f64`",
            ))),
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_batching_unrelated_axis_rejects_dynamic_data_extents() {
        // Offset rebasing stages the leading `operand` and `output` extents as scalar constants, so a merged unrelated
        // axis requires a static leading extent.
        let input_extent = DimensionVariable::new("input_extent", DimensionBounds::new(0, Some(8)).unwrap());
        let error = trace_unrelated_axis_exchange(vec![
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(input_extent)])),
            ArrayType::new_static(DataType::F32, [2, 4]),
            ArrayType::new_static(DataType::I32, [2, 2]),
            ArrayType::new_static(DataType::I32, [2, 2]),
            ArrayType::new_static(DataType::I32, [2, 2]),
            ArrayType::new_static(DataType::I32, [2, 2]),
        ])
        .unwrap_err();
        let error = error.downcast_custom::<BatchingError>().unwrap();
        assert_eq!(
            error,
            &BatchingError::UnsupportedOperation {
                message:
                    "`parallel_ragged_all_to_all` merged batching requires `operand` axis 0 to have a static extent"
                        .to_string(),
            },
        );

        // Reshaping the packed data cannot yet recover dynamic trailing extents, so those must be static as well.
        let trailing_extent = DimensionVariable::new("trailing_extent", DimensionBounds::new(0, Some(8)).unwrap());
        let error = trace_unrelated_axis_exchange(vec![
            ArrayType::new(
                DataType::F32,
                Shape::new(vec![
                    Dimension::Static(2),
                    Dimension::Static(3),
                    Dimension::Dynamic(trailing_extent.clone()),
                ]),
            ),
            ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Static(2), Dimension::Static(4), Dimension::Dynamic(trailing_extent)]),
            ),
            ArrayType::new_static(DataType::I32, [2, 2]),
            ArrayType::new_static(DataType::I32, [2, 2]),
            ArrayType::new_static(DataType::I32, [2, 2]),
            ArrayType::new_static(DataType::I32, [2, 2]),
        ])
        .unwrap_err();
        let error = error.downcast_custom::<BatchingError>().unwrap();
        assert_eq!(
            error,
            &BatchingError::UnsupportedOperation {
                message:
                    "`parallel_ragged_all_to_all` merged batching requires `operand` axis 1 to have a static extent"
                        .to_string(),
            },
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_batching_unrelated_axis_rejects_dynamic_mapped_extent() {
        // Offset rebasing stages the mapped-axis extent as a scalar constant, so a merged unrelated axis requires a
        // static extent.
        let context = ArrayIrTracingContext::new();
        let batch_size = DimensionVariable::new("batch", DimensionBounds::new(1, Some(5)).unwrap());
        let batch_extent = context.input(DimensionType::from(batch_size.clone()).into());
        let batching_context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(context.clone(), batch_extent)
            .with_axis_name("y".to_string());
        let packed_type = |data_type, dimensions: &[usize]| {
            ArrayType::new(
                data_type,
                Shape::new(
                    std::iter::once(Dimension::Dynamic(batch_size.clone()))
                        .chain(dimensions.iter().copied().map(Dimension::Static))
                        .collect(),
                ),
            )
        };
        let metadata_type = packed_type(DataType::I32, &[2, 2]);
        let input_types = [
            packed_type(DataType::F32, &[2, 3]),
            packed_type(DataType::F32, &[2, 4]),
            metadata_type.clone(),
            metadata_type.clone(),
            metadata_type.clone(),
            metadata_type,
        ];
        let inputs = input_types.map(|r#type| {
            BatchingTracer::new(
                batching_context.clone(),
                ArrayIrBatch::new(context.input(r#type.into()), BatchAxis::new(0)).unwrap(),
            )
        });
        let error = batching_context
            .bind(
                ArrayIrOperation::ParallelRaggedAllToAll(
                    ParallelRaggedAllToAllOperation::new("x".to_string(), 2).with_physical_representation(),
                ),
                Vec::new(),
                &inputs,
            )
            .unwrap_err();
        let error = error.downcast_custom::<BatchingError>().unwrap();
        assert_eq!(
            error,
            &BatchingError::UnsupportedOperation {
                message: "`parallel_ragged_all_to_all` merged batching requires a statically known mapped-axis extent"
                    .to_string(),
            },
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_batching_composes_named_axes_in_both_orders() {
        // A matching `x` level and an unrelated `y` level compose in either nesting order and exchange the same
        // segments, so the two results differ only in the physical order of their batch axes. An outer level that
        // reuses the name `x` is shadowed by the inner one and therefore merges like the unrelated `y` level.
        let metadata_type = ArrayType::new_static(DataType::I8, [2, 2, 2]);
        let inputs = (
            Array::from_elements::<f64>(
                ArrayType::new_static(DataType::F64, [2, 2, 3]),
                &[10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0, 40.0, 41.0, 42.0],
            )
            .unwrap(),
            Array::from_elements::<f64>(
                ArrayType::new_static(DataType::F64, [2, 2, 4]),
                &[
                    100.0, 101.0, 102.0, 103.0, 110.0, 111.0, 112.0, 113.0, 200.0, 201.0, 202.0, 203.0, 210.0, 211.0,
                    212.0, 213.0,
                ],
            )
            .unwrap(),
            Array::from_elements(metadata_type.clone(), &[0i8, 1, 2, 0, 0, 2, 1, 2]).unwrap(),
            Array::from_elements(metadata_type.clone(), &[1i8; 8]).unwrap(),
            Array::from_elements(metadata_type.clone(), &[0i8, 1, 1, 0, 2, 3, 3, 2]).unwrap(),
            Array::from_elements(metadata_type, &[1i8; 8]).unwrap(),
        );
        let transpose = |input: &Array| input.transpose([1, 0, 2]).unwrap();
        let transposed_inputs = (
            transpose(&inputs.0),
            transpose(&inputs.1),
            transpose(&inputs.2),
            transpose(&inputs.3),
            transpose(&inputs.4),
            transpose(&inputs.5),
        );
        let x_then_y: Array = batch(
            |inputs| {
                Ok(batch(
                    |(operand, output, input_offsets, send_sizes, output_offsets, receive_sizes)| {
                        operand.parallel_ragged_all_to_all(
                            "x",
                            &output,
                            &input_offsets,
                            &send_sizes,
                            &output_offsets,
                            &receive_sizes,
                        )
                    },
                    inputs,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("y"),
                )?)
            },
            inputs,
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        assert_eq!(
            x_then_y,
            Array::from_elements::<f64>(
                ArrayType::new_static(DataType::F64, [2, 2, 4]),
                &[
                    10.0, 101.0, 30.0, 103.0, 110.0, 22.0, 112.0, 41.0, 200.0, 11.0, 202.0, 32.0, 20.0, 211.0, 42.0,
                    213.0,
                ],
            )
            .unwrap(),
        );

        let y_then_x: Array = batch(
            |inputs| {
                Ok(batch(
                    |(operand, output, input_offsets, send_sizes, output_offsets, receive_sizes)| {
                        operand.parallel_ragged_all_to_all(
                            "x",
                            &output,
                            &input_offsets,
                            &send_sizes,
                            &output_offsets,
                            &receive_sizes,
                        )
                    },
                    inputs,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("x"),
                )?)
            },
            transposed_inputs.clone(),
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("y"),
        )
        .unwrap();
        assert_eq!(
            y_then_x,
            Array::from_elements::<f64>(
                ArrayType::new_static(DataType::F64, [2, 2, 4]),
                &[
                    10.0, 101.0, 30.0, 103.0, 200.0, 11.0, 202.0, 32.0, 110.0, 22.0, 112.0, 41.0, 20.0, 211.0, 42.0,
                    213.0,
                ],
            )
            .unwrap(),
        );

        let x_then_x: Array = batch(
            |inputs| {
                Ok(batch(
                    |(operand, output, input_offsets, send_sizes, output_offsets, receive_sizes)| {
                        operand.parallel_ragged_all_to_all(
                            "x",
                            &output,
                            &input_offsets,
                            &send_sizes,
                            &output_offsets,
                            &receive_sizes,
                        )
                    },
                    inputs,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("x"),
                )?)
            },
            transposed_inputs,
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        assert_eq!(x_then_x, y_then_x);
    }

    #[test]
    fn test_parallel_ragged_all_to_all_batching_array_ir() {
        // Composite values reach the eager exchange of a matching named `batch` level through their array members.
        let operand = vec![1i32, 2, 2, 3, 4, 0];
        let output_seed = vec![0i32; 8];
        let output: ArrayIrValue<Array> = batch(
            |inputs: Vec<
                BatchingTracer<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>, ArrayIrBatchingPolicy>,
            >| {
                inputs[0].parallel_ragged_all_to_all("x", &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5])
            },
            vec![
                ArrayIrValue::Array(Array::matrix(2, 3, operand.clone()).unwrap()),
                ArrayIrValue::Array(Array::matrix(2, 4, output_seed.clone()).unwrap()),
                ArrayIrValue::Array(Array::matrix(2, 2, vec![0i32, 1, 0, 1]).unwrap()),
                ArrayIrValue::Array(Array::matrix(2, 2, vec![1i32, 2, 1, 1]).unwrap()),
                ArrayIrValue::Array(Array::matrix(2, 2, vec![0i32, 0, 1, 2]).unwrap()),
                ArrayIrValue::Array(Array::matrix(2, 2, vec![1i32, 1, 2, 1]).unwrap()),
            ],
            vec![BatchAxis::new(0); 6],
            BatchAxis::new(0),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        let expected = reference_ragged_transfers(
            operand.as_slice(),
            output_seed.as_slice(),
            3,
            4,
            1,
            &[(0, 0, 0, 0, 1), (0, 1, 1, 0, 2), (1, 0, 0, 1, 1), (1, 1, 1, 2, 1)],
        );
        assert_eq!(output, ArrayIrValue::Array(Array::matrix(2, 4, expected).unwrap()));
    }

    #[test]
    fn test_parallel_ragged_all_to_all_differentiation() {
        // The exchange is jointly linear in `operand` and `output`, so a live tangent of either data input is exchanged
        // with the primal metadata, a structural zero contributes zeros, and two structural zeros yield a structural
        // zero.
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let operation = ParallelRaggedAllToAllOperation::new("x".to_string(), 1);
        let operand = Array::vector(vec![10.0f64, 11.0, 12.0]).unwrap();
        let output = Array::vector(vec![100.0f64, 101.0, 102.0, 103.0]).unwrap();
        let metadata = [
            Array::vector(vec![1i32]).unwrap(),
            Array::vector(vec![2i32]).unwrap(),
            Array::vector(vec![0i32]).unwrap(),
            Array::vector(vec![2i32]).unwrap(),
        ]
        .map(|primal| DifferentiationDual::new_with_zero_tangent(primal).unwrap());
        let operand_tangent = Array::vector(vec![1.0f64, 2.0, 3.0]).unwrap();
        let output_tangent = Array::vector(vec![10.0f64, 20.0, 30.0, 40.0]).unwrap();

        // Two structural zeros yield a structural zero of the result type next to the exchanged primal.
        let zero = operation
            .jvp(
                &DifferentiationContext::fused(context.clone()),
                &EmptyRegionDriver,
                &[
                    DifferentiationDual::new_with_zero_tangent(operand.clone()).unwrap(),
                    DifferentiationDual::new_with_zero_tangent(output.clone()).unwrap(),
                    metadata[0].clone(),
                    metadata[1].clone(),
                    metadata[2].clone(),
                    metadata[3].clone(),
                ],
            )
            .unwrap()
            .remove(0);
        assert_eq!(zero.primal(), &Array::vector(vec![11.0f64, 12.0, 102.0, 103.0]).unwrap());
        assert!(matches!(
            zero.tangent(),
            MaybeZero::Zero(r#type) if r#type == &ArrayType::new_static(DataType::F64, [4]),
        ));

        // A live `operand` tangent is exchanged into a zero `output` tangent.
        let operand_only = operation
            .jvp(
                &DifferentiationContext::fused(context.clone()),
                &EmptyRegionDriver,
                &[
                    DifferentiationDual::new(operand.clone(), MaybeZero::Value(operand_tangent.clone())).unwrap(),
                    DifferentiationDual::new_with_zero_tangent(output.clone()).unwrap(),
                    metadata[0].clone(),
                    metadata[1].clone(),
                    metadata[2].clone(),
                    metadata[3].clone(),
                ],
            )
            .unwrap()
            .remove(0);
        assert_eq!(operand_only.tangent().as_value(), Some(&Array::vector(vec![2.0f64, 3.0, 0.0, 0.0]).unwrap()));

        // A live `output` tangent survives outside the received region, which the zero `operand` tangent overwrites.
        let output_only = operation
            .jvp(
                &DifferentiationContext::fused(context.clone()),
                &EmptyRegionDriver,
                &[
                    DifferentiationDual::new_with_zero_tangent(operand.clone()).unwrap(),
                    DifferentiationDual::new(output.clone(), MaybeZero::Value(output_tangent.clone())).unwrap(),
                    metadata[0].clone(),
                    metadata[1].clone(),
                    metadata[2].clone(),
                    metadata[3].clone(),
                ],
            )
            .unwrap()
            .remove(0);
        assert_eq!(output_only.tangent().as_value(), Some(&Array::vector(vec![0.0f64, 0.0, 30.0, 40.0]).unwrap()));

        // Live tangents of both data inputs are exchanged together.
        let joint = operation
            .jvp(
                &DifferentiationContext::fused(context),
                &EmptyRegionDriver,
                &[
                    DifferentiationDual::new(operand, MaybeZero::Value(operand_tangent)).unwrap(),
                    DifferentiationDual::new(output, MaybeZero::Value(output_tangent)).unwrap(),
                    metadata[0].clone(),
                    metadata[1].clone(),
                    metadata[2].clone(),
                    metadata[3].clone(),
                ],
            )
            .unwrap()
            .remove(0);
        assert_eq!(joint.tangent().as_value(), Some(&Array::vector(vec![2.0f64, 3.0, 30.0, 40.0]).unwrap()));
    }

    #[test]
    fn test_parallel_ragged_all_to_all_differentiation_array_ir() {
        // A composite exchange differentiates through the rule of its array member, which applies the same exchange,
        // with the same primal metadata, to the live tangents of both data inputs.
        let context = ArrayIrTracingContext::new();
        let operand = context.input(ArrayType::new_static(DataType::F32, [3]).into());
        let output = context.input(ArrayType::new_static(DataType::F32, [4]).into());
        let metadata =
            (0..4).map(|_| context.input(ArrayType::new_static(DataType::I32, [1]).into())).collect::<Vec<_>>();
        let operand_tangent = context.input(ArrayType::new_static(DataType::F32, [3]).into());
        let output_tangent = context.input(ArrayType::new_static(DataType::F32, [4]).into());
        let mut inputs = vec![
            DifferentiationDual::new(operand, MaybeZero::Value(operand_tangent)).unwrap(),
            DifferentiationDual::new(output, MaybeZero::Value(output_tangent)).unwrap(),
        ];
        inputs
            .extend(metadata.into_iter().map(|metadata| DifferentiationDual::new_with_zero_tangent(metadata).unwrap()));
        let outputs = ParallelRaggedAllToAllOperation::new("x".to_string(), 1)
            .jvp_in_parent(&DifferentiationContext::fused(context.clone()), &EmptyRegionDriver, &inputs)
            .unwrap();
        let output_ids =
            vec![outputs[0].primal().atom_id().unwrap(), outputs[0].tangent().as_value().unwrap().atom_id().unwrap()];
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                output_ids,
                vec![Placeholder; 8],
                vec![Placeholder; 2],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[3], %1:f32[4], %2:i32[1], %3:i32[1], %4:i32[1], %5:i32[1], %6:f32[3], %7:f32[4] .
                let %8:f32[4] = parallel_ragged_all_to_all [axis_name=\"x\", axis_size=1] %0 %1 %2 %3 %4 %5
                    %9:f32[4] = parallel_ragged_all_to_all [axis_name=\"x\", axis_size=1] %6 %7 %2 %3 %4 %5
                in (%8, %9)"
            },
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_transposition() {
        // Overlapping send regions accumulate into the shared `operand` cotangent, zero-length segments contribute
        // nothing, and the `output` cotangent survives only outside received regions.
        check_operation_transposition!(
            @exact,
            operation = ParallelRaggedAllToAllOperation::new("x".to_string(), 1),
            cases = [
                {
                    inputs = [
                        (@linear(type = ArrayType::new_static(DataType::F64, [2]))),
                        (@linear(type = ArrayType::new_static(DataType::F64, [2]))),
                        (@known, Array::vector(vec![0i32, 0]).unwrap()),
                        (@known, Array::vector(vec![1i32, 1]).unwrap()),
                        (@known, Array::vector(vec![0i32, 1]).unwrap()),
                        (@known, Array::vector(vec![1i32, 1]).unwrap()),
                    ],
                    output_cotangents = [Array::vector(vec![3.0f64, 5.0]).unwrap()],
                    input_cotangents = [
                        Array::vector(vec![8.0f64, 0.0]).unwrap(),
                        Array::vector(vec![0.0f64, 0.0]).unwrap(),
                    ],
                },
                {
                    inputs = [
                        (@linear(type = ArrayType::new_static(DataType::F64, [3]))),
                        (@linear(type = ArrayType::new_static(DataType::F64, [3]))),
                        (@known, Array::vector(vec![0i32, 1, 1]).unwrap()),
                        (@known, Array::vector(vec![1i32, 0, 1]).unwrap()),
                        (@known, Array::vector(vec![0i32, 1, 1]).unwrap()),
                        (@known, Array::vector(vec![1i32, 0, 1]).unwrap()),
                    ],
                    output_cotangents = [Array::vector(vec![3.0f64, 5.0, 7.0]).unwrap()],
                    input_cotangents = [
                        Array::vector(vec![3.0f64, 5.0, 0.0]).unwrap(),
                        Array::vector(vec![0.0f64, 0.0, 7.0]).unwrap(),
                    ],
                },
            ],
        );

        // Each data input can also be the only linear input: the `operand` cotangent does not depend on the `output`
        // seed, and the masked `output` cotangent does not depend on the `operand`.
        check_operation_transposition!(
            @exact,
            operation = ParallelRaggedAllToAllOperation::new("x".to_string(), 1),
            cases = [
                {
                    inputs = [
                        (@linear(type = ArrayType::new_static(DataType::F64, [2]))),
                        (@known, Array::vector(vec![10.0f64, 20.0]).unwrap()),
                        (@known, Array::vector(vec![0i32, 0]).unwrap()),
                        (@known, Array::vector(vec![1i32, 1]).unwrap()),
                        (@known, Array::vector(vec![0i32, 1]).unwrap()),
                        (@known, Array::vector(vec![1i32, 1]).unwrap()),
                    ],
                    output_cotangents = [Array::vector(vec![3.0f64, 5.0]).unwrap()],
                    input_cotangents = [Array::vector(vec![8.0f64, 0.0]).unwrap()],
                },
                {
                    inputs = [
                        (@known, Array::vector(vec![10.0f64, 20.0, 30.0]).unwrap()),
                        (@linear(type = ArrayType::new_static(DataType::F64, [3]))),
                        (@known, Array::vector(vec![0i32, 1, 1]).unwrap()),
                        (@known, Array::vector(vec![1i32, 0, 1]).unwrap()),
                        (@known, Array::vector(vec![0i32, 1, 1]).unwrap()),
                        (@known, Array::vector(vec![1i32, 0, 1]).unwrap()),
                    ],
                    output_cotangents = [Array::vector(vec![3.0f64, 5.0, 7.0]).unwrap()],
                    input_cotangents = [Array::vector(vec![0.0f64, 0.0, 7.0]).unwrap()],
                },
            ],
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_transposition_grouped() {
        // The dense offset exchanges and the adjoint ragged exchange stay within the participant groups of the forward
        // exchange, and the output seed's cotangent is masked outside the received regions.
        let groups = vec![vec![0, 2], vec![3, 1]];
        let program = constant_metadata_exchange_program(
            ParallelRaggedAllToAllOperation::grouped("x".to_string(), 4, groups.clone()).unwrap(),
            ArrayType::new_static(DataType::F32, [3]),
            ArrayType::new_static(DataType::F32, [4]),
            [
                Array::vector(vec![0i32, 1, 0, 2]).unwrap(),
                Array::vector(vec![1i32, 1, 0, 1]).unwrap(),
                Array::vector(vec![0i32, 2, 1, 3]).unwrap(),
                Array::vector(vec![1i32, 0, 1, 1]).unwrap(),
            ],
        );
        let pullback = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();

        assert_eq!(
            std::fmt::from_fn(|formatter| pullback.render(formatter, 0, ProgramRenderingMode::WithProvenance))
                .to_string(),
            indoc! {"
                lambda %0:f32[4] .
                let %1:i32[4] = const [0, 1, 0, 2]
                    %2:i32[4] = const [1, 1, 0, 1]
                    %3:i32[4] = const [0, 2, 1, 3]
                    %4:i32[4] = const [1, 0, 1, 1]
                    %5:i32[4] = parallel_all_to_all [
                        axis_name=\"x\",
                        axis_size=4,
                        split_axis=0,
                        concatenation_axis=0,
                        options=Tiled,
                        axis_index_groups=[[0, 2], [3, 1]],
                    ] %3 ; provenance=ryft::differentiation::parallel_ragged_all_to_all_transpose
                    %6:i32[4] = parallel_all_to_all [
                        axis_name=\"x\",
                        axis_size=4,
                        split_axis=0,
                        concatenation_axis=0,
                        options=Tiled,
                        axis_index_groups=[[0, 2], [3, 1]],
                    ] %1 ; provenance=ryft::differentiation::parallel_ragged_all_to_all_transpose
                    %7:f32[3] = zero [type=f32[3]] ; \
                        provenance=ryft::differentiation::parallel_ragged_all_to_all_transpose
                    %8:f32[3] = parallel_ragged_all_to_all [axis_name=\"x\", axis_size=4, axis_index_groups=[[0, 2], \
                        [3, 1]], update_kind=Add] %0 %7 %5 %4 %6 %2 ; \
                        provenance=ryft::differentiation::parallel_ragged_all_to_all_transpose
                    %9:u64[4] = convert_element_type [data_type=u64] %5 ; \
                        provenance=ryft::differentiation::parallel_ragged_all_to_all_transpose
                    %10:u64[4] = convert_element_type [data_type=u64] %4 ; \
                        provenance=ryft::differentiation::parallel_ragged_all_to_all_transpose
                    %11:i64[5] = zero [type=i64[5]] ; \
                        provenance=ryft::differentiation::parallel_ragged_all_to_all_transpose
                    %12:i64[4] = one [type=i64[4]] ; \
                        provenance=ryft::differentiation::parallel_ragged_all_to_all_transpose
                    %13:i64[4] = neg %12 ; provenance=ryft::differentiation::parallel_ragged_all_to_all_transpose
                    %14:u64[4] = add %9 %10 ; provenance=ryft::differentiation::parallel_ragged_all_to_all_transpose
                    %15:u64[4, 1] = reshape [shape=[4, 1]] %9 ; \
                        provenance=ryft::differentiation::parallel_ragged_all_to_all_transpose
                    %16:u64[4, 1] = reshape [shape=[4, 1]] %14 ; \
                        provenance=ryft::differentiation::parallel_ragged_all_to_all_transpose
                    %17:i64[5] = scatter [
                        kind=add,
                        dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                            operand_batching=[], scatter_indices_batching=[]),
                    ] %11 %15 %12 ; provenance=ryft::differentiation::parallel_ragged_all_to_all_transpose
                    %18:i64[5] = scatter [
                        kind=add,
                        dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                            operand_batching=[], scatter_indices_batching=[]),
                    ] %17 %16 %13 ; provenance=ryft::differentiation::parallel_ragged_all_to_all_transpose
                    %19:i64[5] = cumulative [kind=sum, axis=0] %18 ; \
                        provenance=ryft::differentiation::parallel_ragged_all_to_all_transpose
                    %20:i64[4] = slice [start_indices=[0], limits=[4]] %19 ; \
                        provenance=ryft::differentiation::parallel_ragged_all_to_all_transpose
                    %21:i64[4] = zero [type=i64[4]] ; \
                        provenance=ryft::differentiation::parallel_ragged_all_to_all_transpose
                    %22:bool[4] = compare [direction=NotEqual] %20 %21 ; \
                        provenance=ryft::differentiation::parallel_ragged_all_to_all_transpose
                    %23:f32[4] = zero [type=f32[4]] ; \
                        provenance=ryft::differentiation::parallel_ragged_all_to_all_transpose
                    %24:f32[4] = select %22 %23 %0 ; \
                        provenance=ryft::differentiation::parallel_ragged_all_to_all_transpose
                in (%8, %24)"
            },
        );

        // The adjoint exchange is itself transposable. Transposing the pullback exchanges the offsets twice, which is
        // the identity within every participant group, and recovers the forward exchange as an additive exchange of
        // the operand added to the output seed masked outside the received regions.
        let transposed_twice = pullback.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed_twice.to_string(),
            indoc! {"
                lambda %0:f32[3], %1:f32[4] .
                let %2:i64[5] = zero [type=i64[5]]
                    %3:i32[4] = const [0, 2, 1, 3]
                    %4:i32[4] = parallel_all_to_all [
                        axis_name=\"x\",
                        axis_size=4,
                        split_axis=0,
                        concatenation_axis=0,
                        options=Tiled,
                        axis_index_groups=[[0, 2], [3, 1]],
                    ] %3
                    %5:u64[4] = convert_element_type [data_type=u64] %4
                    %6:u64[4, 1] = reshape [shape=[4, 1]] %5
                    %7:i64[4] = one [type=i64[4]]
                    %8:i64[5] = scatter [
                        kind=add,
                        dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                            operand_batching=[], scatter_indices_batching=[]),
                    ] %2 %6 %7
                    %9:i32[4] = const [1, 0, 1, 1]
                    %10:u64[4] = convert_element_type [data_type=u64] %9
                    %11:u64[4] = add %5 %10
                    %12:u64[4, 1] = reshape [shape=[4, 1]] %11
                    %13:i64[4] = neg %7
                    %14:i64[5] = scatter [
                        kind=add,
                        dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                            operand_batching=[], scatter_indices_batching=[]),
                    ] %8 %12 %13
                    %15:i64[5] = cumulative [kind=sum, axis=0] %14
                    %16:i64[4] = slice [start_indices=[0], limits=[4]] %15
                    %17:i64[4] = zero [type=i64[4]]
                    %18:bool[4] = compare [direction=NotEqual] %16 %17
                    %19:f32[4] = zero [type=f32[4]]
                    %20:f32[4] = zero [type=f32[4]]
                    %21:f32[4] = select %18 %20 %1
                    %22:f32[3] = zero [type=f32[3]]
                    %23:i32[4] = const [0, 1, 0, 2]
                    %24:i32[4] = parallel_all_to_all [
                        axis_name=\"x\",
                        axis_size=4,
                        split_axis=0,
                        concatenation_axis=0,
                        options=Tiled,
                        axis_index_groups=[[0, 2], [3, 1]],
                    ] %23
                    %25:i32[4] = const [1, 1, 0, 1]
                    %26:i32[4] = parallel_all_to_all [
                        axis_name=\"x\",
                        axis_size=4,
                        split_axis=0,
                        concatenation_axis=0,
                        options=Tiled,
                        axis_index_groups=[[0, 2], [3, 1]],
                    ] %24
                    %27:i32[4] = parallel_all_to_all [
                        axis_name=\"x\",
                        axis_size=4,
                        split_axis=0,
                        concatenation_axis=0,
                        options=Tiled,
                        axis_index_groups=[[0, 2], [3, 1]],
                    ] %4
                    %28:f32[4] = zero [type=f32[4]]
                    %29:f32[4] = parallel_ragged_all_to_all [axis_name=\"x\", axis_size=4, axis_index_groups=[[0, 2], \
                        [3, 1]], update_kind=Add] %0 %28 %26 %25 %27 %9
                    %30:f32[4] = add %21 %29
                in (%30)"
            },
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_transposition_grouped_physical() {
        // The physical representation has no named-axis binder left in which to exchange offsets, so its transpose
        // permutes the offset blocks of every participant group with static slices and concatenations.
        let operand_type = ArrayType::new_static(DataType::F64, [4, 2]);
        let output_type = ArrayType::new_static(DataType::F64, [4, 2]);
        check_operation_transposition!(
            @exact,
            operation = ParallelRaggedAllToAllOperation::grouped("x".to_string(), 4, vec![vec![0, 2], vec![3, 1]])
                .unwrap()
                .with_physical_representation(),
            cases = [{
                inputs = [
                    (@linear(type = operand_type)),
                    (@linear(type = output_type)),
                    (@known, Array::matrix(4, 2, vec![0i32, 1, 0, 1, 0, 1, 0, 1]).unwrap()),
                    (@known, Array::matrix(4, 2, vec![1i32; 8]).unwrap()),
                    (@known, Array::matrix(4, 2, vec![0i32, 0, 1, 1, 1, 1, 0, 0]).unwrap()),
                    (@known, Array::matrix(4, 2, vec![1i32; 8]).unwrap()),
                ],
                output_cotangents = [Array::matrix(4, 2, vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap()],
                input_cotangents = [
                    Array::matrix(4, 2, vec![1.0f64, 5.0, 8.0, 4.0, 2.0, 6.0, 7.0, 3.0]).unwrap(),
                    Array::matrix(4, 2, vec![0.0f64; 8]).unwrap(),
                ],
            }],
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_transposition_additive_updates() {
        // An accumulating exchange keeps every element of its output seed, so the `output` cotangent passes through
        // unmasked and no metadata exchange is staged.
        let program = constant_metadata_exchange_program(
            ParallelRaggedAllToAllOperation::new("x".to_string(), 1).with_additive_updates(),
            ArrayType::new_static(DataType::F32, [3]),
            ArrayType::new_static(DataType::F32, [4]),
            [
                Array::vector(vec![0i32]).unwrap(),
                Array::vector(vec![1i32]).unwrap(),
                Array::vector(vec![0i32]).unwrap(),
                Array::vector(vec![1i32]).unwrap(),
            ],
        );
        let pullback = program.transpose_with_respect_to(&[1], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[4], %1:f32[3] .
                let %2:i32[1] = const [0]
                    %3:i32[1] = const [1]
                    %4:i32[1] = const [0]
                    %5:i32[1] = const [1]
                in (%0)"
            },
        );

        // The `operand` cotangent of an accumulating exchange sums the cotangents of every segment read from a row, as
        // for an overwriting exchange, while the `output` cotangent is the unmasked output cotangent.
        check_operation_transposition!(
            @exact,
            operation = ParallelRaggedAllToAllOperation::new("x".to_string(), 1).with_additive_updates(),
            cases = [{
                inputs = [
                    (@linear(type = ArrayType::new_static(DataType::F64, [2]))),
                    (@linear(type = ArrayType::new_static(DataType::F64, [2]))),
                    (@known, Array::vector(vec![0i32, 0]).unwrap()),
                    (@known, Array::vector(vec![1i32, 1]).unwrap()),
                    (@known, Array::vector(vec![0i32, 1]).unwrap()),
                    (@known, Array::vector(vec![1i32, 1]).unwrap()),
                ],
                output_cotangents = [Array::vector(vec![3.0f64, 5.0]).unwrap()],
                input_cotangents = [
                    Array::vector(vec![8.0f64, 0.0]).unwrap(),
                    Array::vector(vec![3.0f64, 5.0]).unwrap(),
                ],
            }],
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_transposition_manual_axis() {
        // The transpose of an exchange over a manual mesh axis stays on that mesh: both dense offset exchanges and the
        // adjoint ragged exchange record it, and the cotangents keep the variation of the forward data inputs.
        let mesh = manual_mesh();
        let varying = Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap();
        let typed = |data_type, extent: usize| {
            ArrayType::new_static(data_type, [extent]).with_sharding(varying.clone()).unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let inputs = vec![
            builder.add_input(typed(DataType::F32, 3)),
            builder.add_input(typed(DataType::F32, 4)),
            builder.add_input(typed(DataType::I32, 2)),
            builder.add_input(typed(DataType::I32, 2)),
            builder.add_input(typed(DataType::I32, 2)),
            builder.add_input(typed(DataType::I32, 2)),
        ];
        let result = builder
            .add_instruction(
                ParallelRaggedAllToAllOperation::new("x".to_string(), 2).with_mesh(mesh.clone()),
                Vec::new(),
                inputs,
                None,
            )
            .unwrap()[0];
        let program = builder.build::<Vec<Array>, Array>(vec![result], vec![Placeholder; 6], Placeholder).unwrap();
        let pullback = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}], \
                    %1:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}], \
                    %2:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}], \
                    %3:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}], \
                    %4:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] .
                let %5:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                    parallel_all_to_all [
                    axis_name=\"x\",
                    axis_size=2,
                    split_axis=0,
                    concatenation_axis=0,
                    options=Tiled,
                    mesh=['x'=2:manual, 'y'=2:manual],
                ] %3
                    %6:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_all_to_all [
                        axis_name=\"x\",
                        axis_size=2,
                        split_axis=0,
                        concatenation_axis=0,
                        options=Tiled,
                        mesh=['x'=2:manual, 'y'=2:manual],
                    ] %1
                    %7:f32[3][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = zero [
                        type=f32[3][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}],
                    ]
                    %8:f32[3][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_ragged_all_to_all [axis_name=\"x\", axis_size=2, update_kind=Add, \
                        mesh=['x'=2:manual, 'y'=2:manual]] %0 %7 %5 %4 %6 %2
                    %9:u64[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        convert_element_type [data_type=u64] %5
                    %10:u64[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        convert_element_type [data_type=u64] %4
                    %11:i64[5][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = zero [
                        type=i64[5][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}],
                    ]
                    %12:i64[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = one [
                        type=i64[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}],
                    ]
                    %13:i64[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = neg %12
                    %14:u64[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = add %9 %10
                    %15:u64[2, 1][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                        reshape [shape=[2, 1]] %9
                    %16:u64[2, 1][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                        reshape [shape=[2, 1]] %14
                    %17:i64[5][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = scatter [
                        kind=add,
                        dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                            operand_batching=[], scatter_indices_batching=[]),
                    ] %11 %15 %12
                    %18:i64[5][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = scatter [
                        kind=add,
                        dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                            operand_batching=[], scatter_indices_batching=[]),
                    ] %17 %16 %13
                    %19:i64[5][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        cumulative [kind=sum, axis=0] %18
                    %20:i64[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = slice \
                        [start_indices=[0], limits=[4]] %19
                    %21:i64[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = zero [
                        type=i64[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}],
                    ]
                    %22:bool[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = compare \
                        [direction=NotEqual] %20 %21
                    %23:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = zero [
                        type=f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}],
                    ]
                    %24:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = select \
                        %22 %23 %0
                in (%8, %24)"
            },
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_transposition_preserves_explicit_sharding() {
        // The cotangents keep the explicit sharding of their data inputs, even over the exchanged leading axis, while
        // the metadata computation that masks the `output` cotangent stays unsharded.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("data", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding =
            Sharding::new(mesh, vec![ShardingDimension::sharded(["data"]), ShardingDimension::replicated()]).unwrap();
        let operand_type = ArrayType::new_static(DataType::F32, [3, 2]).with_sharding(sharding.clone()).unwrap();
        let output_type = ArrayType::new_static(DataType::F32, [4, 2]).with_sharding(sharding).unwrap();
        let program = constant_metadata_exchange_program(
            ParallelRaggedAllToAllOperation::new("x".to_string(), 1),
            operand_type,
            output_type,
            [
                Array::vector(vec![0i32]).unwrap(),
                Array::vector(vec![1i32]).unwrap(),
                Array::vector(vec![0i32]).unwrap(),
                Array::vector(vec![1i32]).unwrap(),
            ],
        );
        let pullback = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[4, 2][sharding={mesh<['data'=2:explicit]>, [{'data'}, {}]}] .
                let %1:i32[1] = const [0]
                    %2:i32[1] = const [1]
                    %3:i32[1] = const [0]
                    %4:i32[1] = const [1]
                    %5:i32[1] = parallel_all_to_all [axis_name=\"x\", axis_size=1, split_axis=0, concatenation_axis=0, \
                        options=Tiled] %3
                    %6:i32[1] = parallel_all_to_all [axis_name=\"x\", axis_size=1, split_axis=0, concatenation_axis=0, \
                        options=Tiled] %1
                    %7:f32[3, 2][sharding={mesh<['data'=2:explicit]>, [{'data'}, {}]}] = zero \
                        [type=f32[3, 2][sharding={mesh<['data'=2:explicit]>, [{'data'}, {}]}]]
                    %8:f32[3, 2][sharding={mesh<['data'=2:explicit]>, [{'data'}, {}]}] = parallel_ragged_all_to_all \
                        [axis_name=\"x\", axis_size=1, update_kind=Add] %0 %7 %5 %4 %6 %2
                    %9:u64[1] = convert_element_type [data_type=u64] %5
                    %10:u64[1] = convert_element_type [data_type=u64] %4
                    %11:i64[5] = zero [type=i64[5]]
                    %12:i64[1] = one [type=i64[1]]
                    %13:i64[1] = neg %12
                    %14:u64[1] = add %9 %10
                    %15:u64[1, 1] = reshape [shape=[1, 1]] %9
                    %16:u64[1, 1] = reshape [shape=[1, 1]] %14
                    %17:i64[5] = scatter [
                        kind=add,
                        dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                            operand_batching=[], scatter_indices_batching=[]),
                    ] %11 %15 %12
                    %18:i64[5] = scatter [
                        kind=add,
                        dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                            operand_batching=[], scatter_indices_batching=[]),
                    ] %17 %16 %13
                    %19:i64[5] = cumulative [kind=sum, axis=0] %18
                    %20:i64[4] = slice [start_indices=[0], limits=[4]] %19
                    %21:i64[4] = zero [type=i64[4]]
                    %22:bool[4] = compare [direction=NotEqual] %20 %21
                    %23:bool[4, 2] = broadcast [output_type=bool[4, 2], output_axes=[0]] %22
                    %24:f32[4, 2][sharding={mesh<['data'=2:explicit]>, [{'data'}, {}]}] = zero \
                        [type=f32[4, 2][sharding={mesh<['data'=2:explicit]>, [{'data'}, {}]}]]
                    %25:f32[4, 2][sharding={mesh<['data'=2:explicit]>, [{'data'}, {}]}] = select %23 %24 %0
                in (%8, %25)"
            },
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_transposition_around_named_batch_axis() {
        // Reverse-mode differentiation of a function that contains a matching `batch` level transposes the physical
        // exchange: each `operand` row receives the cotangent of every row that it was sent to, and each `output` row
        // receives its cotangent only when no segment overwrote it.
        let operand = Array::matrix(2, 3, vec![10.0f64, 11.0, 12.0, 20.0, 21.0, 22.0]).unwrap();
        let output = Array::matrix(2, 4, vec![100.0f64, 101.0, 102.0, 103.0, 200.0, 201.0, 202.0, 203.0]).unwrap();
        let (value, gradient) = differentiate_at((operand.clone(), output.clone()))
            .value_and_gradient(|(operand, output)| batched_parallel_ragged_all_to_all_sum!(operand, output))
            .unwrap();
        assert_eq!(value, Array::scalar(939.0f64).unwrap());
        assert_eq!(gradient.0, Array::matrix(2, 3, vec![1.0f64, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap());
        assert_eq!(gradient.1, Array::matrix(2, 4, vec![1.0f64, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap());

        // Both data inputs also agree with central finite differences.
        check_gradient!(
            |operand, output| batched_parallel_ragged_all_to_all_sum!(operand, output),
            at = operand.clone(),
            with = output.clone(),
            step = 1e-6,
            tolerance = 1e-6,
        );
        check_gradient!(
            |output, operand| batched_parallel_ragged_all_to_all_sum!(operand, output),
            at = output,
            with = operand,
            step = 1e-6,
            tolerance = 1e-6,
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_transposition_around_named_batch_axis_grouped() {
        // Reversed noncontiguous groups exchange unequal width-two segments, including empty transfers and repeated
        // source rows. The pullback must accumulate repeated reads and preserve cotangents of untouched seed rows.
        let data_type = ArrayType::new_static(DataType::F64, [4, 4, 2]);
        let operand = Array::from_elements(data_type.clone(), &(1..=32).map(f64::from).collect::<Vec<_>>()).unwrap();
        let output = Array::from_elements(data_type.clone(), &(101..=132).map(f64::from).collect::<Vec<_>>()).unwrap();
        let cotangent = operand.clone();
        let (value, gradient) = differentiate_at((operand.clone(), output.clone()))
            .value_and_gradient(|(operand, output)| {
                let context = operand.dispatch_domain();
                let input_offsets = context.lift(Array::matrix(4, 2, vec![0i32, 0, 1, 4, 4, 1, 0, 0]).unwrap())?;
                let send_sizes = context.lift(Array::matrix(4, 2, vec![2i32, 1, 1, 0, 0, 1, 1, 2]).unwrap())?;
                let output_offsets = context.lift(Array::matrix(4, 2, vec![0i32, 2, 2, 4, 4, 0, 0, 0]).unwrap())?;
                let receive_sizes = context.lift(Array::matrix(4, 2, vec![1i32, 1, 2, 0, 0, 2, 1, 1]).unwrap())?;
                let cotangent = context.lift(cotangent.clone())?;
                let exchanged = batch(
                    |(operand, output, input_offsets, send_sizes, output_offsets, receive_sizes)| {
                        operand.parallel_ragged_all_to_all_with_axis_index_groups(
                            "x",
                            &output,
                            &input_offsets,
                            &send_sizes,
                            &output_offsets,
                            &receive_sizes,
                            vec![vec![3, 1], vec![2, 0]],
                        )
                    },
                    (operand, output, input_offsets, send_sizes, output_offsets, receive_sizes),
                    (
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                    ),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("x"),
                )?;
                Ok(exchanged.mul(&cotangent)?.reduce(&[0, 1, 2], ReductionKind::Sum)?)
            })
            .unwrap();
        assert_eq!(
            gradient.0,
            Array::from_elements(
                data_type.clone(),
                &[
                    22.0f64, 24.0, 19.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 29.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 34.0, 36.0, 11.0, 12.0, 0.0, 0.0, 0.0, 0.0,
                ],
            )
            .unwrap(),
        );
        assert_eq!(
            gradient.1,
            Array::from_elements(
                data_type,
                &[
                    0.0f64, 0.0, 3.0, 4.0, 0.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0, 13.0, 14.0, 15.0, 16.0, 0.0, 0.0,
                    0.0, 0.0, 21.0, 22.0, 23.0, 24.0, 0.0, 0.0, 27.0, 28.0, 0.0, 0.0, 31.0, 32.0,
                ],
            )
            .unwrap(),
        );

        // The weighted primal equals the inner product with the pullback over both independent linear data inputs.
        // All values are exactly representable, so this adjoint identity needs no numerical tolerance.
        assert_eq!(value, Array::scalar(38_676.0f64).unwrap());
        let operand_inner_product = operand.mul(&gradient.0).unwrap().reduce(&[0, 1, 2], ReductionKind::Sum).unwrap();
        let output_inner_product = output.mul(&gradient.1).unwrap().reduce(&[0, 1, 2], ReductionKind::Sum).unwrap();
        assert_eq!(operand_inner_product.add(&output_inner_product), Ok(value));
    }

    #[test]
    fn test_parallel_ragged_all_to_all_transposition_inside_named_batch_axis() {
        // Reverse-mode differentiation inside a matching `batch` level transposes the logical exchange, whose offset
        // metadata are transposed by dense exchanges over the same named axis. The gradients equal those of the
        // transposed physical exchange, which finite differences check in the outer differentiation test.
        let (values, gradients) = batch(
            |(operand, output, input_offsets, send_sizes, output_offsets, receive_sizes)| {
                differentiate_at((operand, output))
                    .with_captures((input_offsets, send_sizes, output_offsets, receive_sizes))
                    .value_and_gradient(
                        |(operand, output), (input_offsets, send_sizes, output_offsets, receive_sizes)| {
                            operand
                                .parallel_ragged_all_to_all(
                                    "x",
                                    &output,
                                    &input_offsets,
                                    &send_sizes,
                                    &output_offsets,
                                    &receive_sizes,
                                )?
                                .reduce(&[0], ReductionKind::Sum)
                        },
                    )
                    .map_err(ProgramError::from)
            },
            (
                Array::matrix(2, 3, vec![10.0f64, 11.0, 12.0, 20.0, 21.0, 22.0]).unwrap(),
                Array::matrix(2, 4, vec![100.0f64, 101.0, 102.0, 103.0, 200.0, 201.0, 202.0, 203.0]).unwrap(),
                Array::matrix(2, 2, vec![0i32, 0, 0, 2]).unwrap(),
                Array::matrix(2, 2, vec![0i32, 1, 1, 0]).unwrap(),
                Array::matrix(2, 2, vec![1i32, 1, 2, 3]).unwrap(),
                Array::matrix(2, 2, vec![0i32, 1, 1, 0]).unwrap(),
            ),
            (
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxis::new(0),
            ),
            (BatchAxis::new(0), (BatchAxis::new(0), BatchAxis::new(0))),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();

        assert_eq!(values, Array::vector(vec![324.0f64, 615.0]).unwrap());
        assert_eq!(gradients.0, Array::matrix(2, 3, vec![1.0f64, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap());
        assert_eq!(gradients.1, Array::matrix(2, 4, vec![1.0f64, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap());
    }

    #[test]
    fn test_parallel_ragged_all_to_all_transposition_rejects_unknown_metadata() {
        // Transposition treats the metadata as primal residuals that route the adjoint exchange, so it rejects metadata
        // that are themselves linear inputs and therefore unknown.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let inputs = [
            builder.add_input(ArrayType::new_static(DataType::F32, [3])),
            builder.add_input(ArrayType::new_static(DataType::F32, [4])),
            builder.add_input(ArrayType::new_static(DataType::I32, [1])),
            builder.add_input(ArrayType::new_static(DataType::I32, [1])),
            builder.add_input(ArrayType::new_static(DataType::I32, [1])),
            builder.add_input(ArrayType::new_static(DataType::I32, [1])),
        ];
        let result = builder
            .add_instruction(
                ParallelRaggedAllToAllOperation::new("x".to_string(), 1),
                Vec::new(),
                inputs.to_vec(),
                None,
            )
            .unwrap()[0];
        let program = builder.build::<Vec<Array>, Array>(vec![result], vec![Placeholder; 6], Placeholder).unwrap();
        assert!(matches!(
            program.transpose_with_respect_to(&[0, 1, 2], &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`parallel_ragged_all_to_all` transpose requires `input_offsets` to be a known primal \
                               residual",
        ));
    }

    #[test]
    fn test_parallel_ragged_all_to_all_transposition_rejects_dynamic_output_extent() {
        // The output mask stages an `M + 1` marker with static scatter and slice operations, so the leading `output`
        // extent `M` must be static.
        let output_extent = DimensionVariable::new("output_extent", DimensionBounds::new(0, Some(8)).unwrap());
        let program = constant_metadata_exchange_program(
            ParallelRaggedAllToAllOperation::new("x".to_string(), 1),
            ArrayType::new_static(DataType::F32, [3]),
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(output_extent)])),
            [
                Array::vector(vec![0i32]).unwrap(),
                Array::vector(vec![1i32]).unwrap(),
                Array::vector(vec![0i32]).unwrap(),
                Array::vector(vec![1i32]).unwrap(),
            ],
        );
        assert!(matches!(
            program.transpose_with_respect_to(&[0, 1], &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`parallel_ragged_all_to_all` transpose requires a static output leading dimension",
        ));
    }

    #[test]
    fn test_parallel_ragged_all_to_all_transposition_rejects_marker_extent_overflow() {
        // The `M + 1` marker extent of the output mask must itself fit in `usize`.
        let program = constant_metadata_exchange_program(
            ParallelRaggedAllToAllOperation::new("x".to_string(), 1),
            ArrayType::new_static(DataType::F32, [1]),
            ArrayType::new_static(DataType::F32, [usize::MAX]),
            [
                Array::vector(vec![0i32]).unwrap(),
                Array::vector(vec![0i32]).unwrap(),
                Array::vector(vec![0i32]).unwrap(),
                Array::vector(vec![0i32]).unwrap(),
            ],
        );
        assert!(matches!(
            program.transpose_with_respect_to(&[0, 1], &[]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "`parallel_ragged_all_to_all` transpose marker extent does not fit in `usize`",
        ));
    }

    #[test]
    fn test_parallel_ragged_all_to_all_parallel_ragged_all_to_all() {
        // A name that no enclosing binder binds fails fast.
        let input_types = vec![
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3])),
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [4])),
            ArrayIrType::Array(ArrayType::new_static(DataType::I32, [2])),
            ArrayIrType::Array(ArrayType::new_static(DataType::I32, [2])),
            ArrayIrType::Array(ArrayType::new_static(DataType::I32, [2])),
            ArrayIrType::Array(ArrayType::new_static(DataType::I32, [2])),
        ];
        assert_eq!(
            ArrayIrTracingContext::trace(
                |inputs: Vec<_>| {
                    inputs[0]
                        .parallel_ragged_all_to_all("x", &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5])
                },
                input_types.clone(),
            )
            .map(|(output_type, _)| output_type),
            Err(ProgramError::Axis(AxisError::UnboundAxisName { name: "x".to_string() })),
        );

        // Over a manual mesh axis, the exchange records the mesh. An input that is invariant over the axis is first
        // made varying, and alignment then makes every input vary over each manual axis that any input varies over,
        // because every device computes its result from all six inputs.
        let mesh = manual_mesh();
        let sharding = Sharding::replicated(mesh.clone(), 1);
        let axes = vec![
            ("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() }),
            ("y".to_string(), NamedAxis::Mesh { axis: 1, size: 2, mesh: mesh.clone() }),
        ];
        let typed = |data_type, extent: usize, varying_manual_axes: &[&str]| {
            ArrayType::new_static(data_type, [extent])
                .with_sharding(sharding.clone().with_varying_manual_axes(varying_manual_axes.iter().copied()).unwrap())
                .unwrap()
        };
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |inputs: Vec<_>| {
                inputs[0].parallel_ragged_all_to_all("x", &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5])
            },
            vec![
                typed(DataType::F32, 3, &[]),
                typed(DataType::F32, 4, &["y"]),
                typed(DataType::I32, 2, &["x"]),
                typed(DataType::I32, 2, &["x"]),
                typed(DataType::I32, 2, &["x"]),
                typed(DataType::I32, 2, &["x"]),
            ],
            axes.clone(),
        )
        .unwrap();
        assert_eq!(output_type, typed(DataType::F32, 4, &["x", "y"]));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda \
                %0:f32[3][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}]}], \
                %1:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'y'}}], \
                %2:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}], \
                %3:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}], \
                %4:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}], \
                %5:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] .
                let %6:f32[3][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %0
                    %7:f32[3][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x', 'y'}}] = \
                        parallel_vary [axis_name=\"y\"] %6
                    %8:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x', 'y'}}] = \
                        parallel_vary [axis_name=\"x\"] %1
                    %9:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x', 'y'}}] = \
                        parallel_vary [axis_name=\"y\"] %2
                    %10:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x', 'y'}}] = \
                        parallel_vary [axis_name=\"y\"] %3
                    %11:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x', 'y'}}] = \
                        parallel_vary [axis_name=\"y\"] %4
                    %12:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x', 'y'}}] = \
                        parallel_vary [axis_name=\"y\"] %5
                    %13:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x', 'y'}}] = \
                        parallel_ragged_all_to_all [axis_name=\"x\", axis_size=2, mesh=['x'=2:manual, 'y'=2:manual]] \
                        %7 %8 %9 %10 %11 %12
                in (%13)"
            },
        );

        // A composite value exchanges through its array view, which converts the exchange back into its direct
        // composite carrier. Inputs without a sharding are placed on the axis's mesh before they are made varying.
        let (output_type, program) = ArrayIrTracingContext::trace_with_named_axes(
            |inputs: Vec<_>| {
                inputs[0].parallel_ragged_all_to_all("x", &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5])
            },
            input_types,
            axes.clone(),
        )
        .unwrap();
        assert_eq!(output_type, ArrayIrType::Array(typed(DataType::F32, 4, &["x"])));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[3], %1:f32[4], %2:i32[2], %3:i32[2], %4:i32[2], %5:i32[2] .
                let %6:f32[3][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}]}] = broadcast [
                    output_type=f32[3][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}]}],
                    output_axes=[0],
                ] %0
                    %7:f32[3][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %6
                    %8:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}]}] = broadcast [
                        output_type=f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}]}],
                        output_axes=[0],
                    ] %1
                    %9:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %8
                    %10:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}]}] = broadcast [
                        output_type=i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}]}],
                        output_axes=[0],
                    ] %2
                    %11:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %10
                    %12:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}]}] = broadcast [
                        output_type=i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}]}],
                        output_axes=[0],
                    ] %3
                    %13:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %12
                    %14:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}]}] = broadcast [
                        output_type=i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}]}],
                        output_axes=[0],
                    ] %4
                    %15:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %14
                    %16:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}]}] = broadcast [
                        output_type=i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}]}],
                        output_axes=[0],
                    ] %5
                    %17:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %16
                    %18:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_ragged_all_to_all [axis_name=\"x\", axis_size=2, mesh=['x'=2:manual, 'y'=2:manual]] \
                        %7 %9 %11 %13 %15 %17
                in (%18)"
            },
        );

        // A pending sum over the participating manual mesh axis is rejected before any input is made varying.
        assert_eq!(
            TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                |inputs: Vec<_>| {
                    inputs[0]
                        .parallel_ragged_all_to_all("x", &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5])
                },
                vec![
                    ArrayType::new_static(DataType::F32, [3])
                        .with_sharding(sharding.clone().with_unreduced_axes(["x"]).unwrap())
                        .unwrap(),
                    typed(DataType::F32, 4, &["x"]),
                    typed(DataType::I32, 2, &["x"]),
                    typed(DataType::I32, 2, &["x"]),
                    typed(DataType::I32, 2, &["x"]),
                    typed(DataType::I32, 2, &["x"]),
                ],
                axes,
            )
            .map(|(output_type, _)| output_type),
            Err(ProgramError::Type(TypeError::invalid(
                "`parallel_ragged_all_to_all` does not support unreduced inputs",
            ))),
        );

        // Concrete arrays, and concrete composite values through their array members, are never inside an axis
        // binder, so every axis name is unbound for them.
        let data = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let metadata = Array::vector(vec![0i32]).unwrap();
        assert_eq!(
            data.parallel_ragged_all_to_all("x", &data, &metadata, &metadata, &metadata, &metadata),
            Err(ProgramError::Axis(AxisError::UnboundAxisName { name: "x".to_string() })),
        );
        let data = ArrayIrValue::Array(data);
        let metadata = ArrayIrValue::Array(metadata);
        assert_eq!(
            data.parallel_ragged_all_to_all("x", &data, &metadata, &metadata, &metadata, &metadata),
            Err(ProgramError::Axis(AxisError::UnboundAxisName { name: "x".to_string() })),
        );
    }

    #[test]
    fn test_parallel_ragged_all_to_all_parallel_ragged_all_to_all_with_axis_index_groups() {
        // A name that no enclosing binder binds fails fast.
        let unplaced_input_types = vec![
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3])),
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [4])),
            ArrayIrType::Array(ArrayType::new_static(DataType::I32, [2])),
            ArrayIrType::Array(ArrayType::new_static(DataType::I32, [2])),
            ArrayIrType::Array(ArrayType::new_static(DataType::I32, [2])),
            ArrayIrType::Array(ArrayType::new_static(DataType::I32, [2])),
        ];
        let grouped_exchange = |inputs: Vec<Tracer<ArrayIrTracingContext>>| {
            inputs[0].parallel_ragged_all_to_all_with_axis_index_groups(
                "x",
                &inputs[1],
                &inputs[2],
                &inputs[3],
                &inputs[4],
                &inputs[5],
                vec![vec![1], vec![0]],
            )
        };
        assert_eq!(
            ArrayIrTracingContext::trace(grouped_exchange, unplaced_input_types.clone())
                .map(|(output_type, _)| output_type),
            Err(ProgramError::Axis(AxisError::UnboundAxisName { name: "x".to_string() })),
        );

        // A grouped exchange over a manual mesh axis records both its groups and the mesh.
        let mesh = manual_mesh();
        let varying = Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap();
        let typed = |data_type, extent: usize| {
            ArrayType::new_static(data_type, [extent]).with_sharding(varying.clone()).unwrap()
        };
        let input_types = vec![
            typed(DataType::F32, 3),
            typed(DataType::F32, 4),
            typed(DataType::I32, 2),
            typed(DataType::I32, 2),
            typed(DataType::I32, 2),
            typed(DataType::I32, 2),
        ];
        let axes = vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })];
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |inputs: Vec<_>| {
                inputs[0].parallel_ragged_all_to_all_with_axis_index_groups(
                    "x",
                    &inputs[1],
                    &inputs[2],
                    &inputs[3],
                    &inputs[4],
                    &inputs[5],
                    vec![vec![1], vec![0]],
                )
            },
            input_types.clone(),
            axes.clone(),
        )
        .unwrap();
        assert_eq!(output_type, typed(DataType::F32, 4));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[3][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}], \
                    %1:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}], \
                    %2:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}], \
                    %3:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}], \
                    %4:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}], \
                    %5:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] .
                let %6:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                    parallel_ragged_all_to_all [
                    axis_name=\"x\",
                    axis_size=2,
                    axis_index_groups=[[1], [0]],
                    mesh=['x'=2:manual, 'y'=2:manual],
                ] %0 %1 %2 %3 %4 %5
                in (%6)"
            },
        );

        // A composite value exchanges through its array view, which places and varies its inputs like an ungrouped
        // exchange and records the groups.
        let (output_type, program) =
            ArrayIrTracingContext::trace_with_named_axes(grouped_exchange, unplaced_input_types, axes.clone()).unwrap();
        assert_eq!(output_type, ArrayIrType::Array(typed(DataType::F32, 4)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[3], %1:f32[4], %2:i32[2], %3:i32[2], %4:i32[2], %5:i32[2] .
                let %6:f32[3][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}]}] = broadcast [
                    output_type=f32[3][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}]}],
                    output_axes=[0],
                ] %0
                    %7:f32[3][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %6
                    %8:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}]}] = broadcast [
                        output_type=f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}]}],
                        output_axes=[0],
                    ] %1
                    %9:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %8
                    %10:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}]}] = broadcast [
                        output_type=i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}]}],
                        output_axes=[0],
                    ] %2
                    %11:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %10
                    %12:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}]}] = broadcast [
                        output_type=i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}]}],
                        output_axes=[0],
                    ] %3
                    %13:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %12
                    %14:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}]}] = broadcast [
                        output_type=i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}]}],
                        output_axes=[0],
                    ] %4
                    %15:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %14
                    %16:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}]}] = broadcast [
                        output_type=i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}]}],
                        output_axes=[0],
                    ] %5
                    %17:i32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %16
                    %18:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_ragged_all_to_all [
                        axis_name=\"x\",
                        axis_size=2,
                        axis_index_groups=[[1], [0]],
                        mesh=['x'=2:manual, 'y'=2:manual],
                    ] %7 %9 %11 %13 %15 %17
                in (%18)"
            },
        );

        // Groups that do not partition the full axis are rejected before anything is staged.
        assert_eq!(
            TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                |inputs: Vec<_>| {
                    inputs[0].parallel_ragged_all_to_all_with_axis_index_groups(
                        "x",
                        &inputs[1],
                        &inputs[2],
                        &inputs[3],
                        &inputs[4],
                        &inputs[5],
                        vec![vec![0, 1], vec![1]],
                    )
                },
                input_types,
                axes,
            )
            .map(|(output_type, _)| output_type),
            Err(ProgramError::Type(TypeError::invalid(
                "`parallel_ragged_all_to_all` axis index group 1 has size 1 but every group must have size 2",
            ))),
        );

        // Concrete arrays, and concrete composite values through their array members, are never inside an axis
        // binder, so every axis name is unbound for them.
        let data = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let metadata = Array::vector(vec![0i32]).unwrap();
        assert_eq!(
            data.parallel_ragged_all_to_all_with_axis_index_groups(
                "x",
                &data,
                &metadata,
                &metadata,
                &metadata,
                &metadata,
                vec![vec![0]],
            ),
            Err(ProgramError::Axis(AxisError::UnboundAxisName { name: "x".to_string() })),
        );
        let data = ArrayIrValue::Array(data);
        let metadata = ArrayIrValue::Array(metadata);
        assert_eq!(
            data.parallel_ragged_all_to_all_with_axis_index_groups(
                "x",
                &data,
                &metadata,
                &metadata,
                &metadata,
                &metadata,
                vec![vec![0]],
            ),
            Err(ProgramError::Axis(AxisError::UnboundAxisName { name: "x".to_string() })),
        );
    }
}
