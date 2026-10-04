//! Contains the named-axis collective operations, which exchange or reduce values across a named axis, together with
//! their interpretation, partial-evaluation, batching, forward-mode differentiation, and transposition rules. These
//! are the analogues of [JAX's parallel operators](https://docs.jax.dev/en/latest/jax.lax.html#parallel-operators).
//!
//! This module owns the vocabulary that every collective shares (i.e., [`CollectiveMode`], [`CollectiveOptions`], and
//! named axis resolution), while each operation family lives in its own submodule: [`parallel_reduce`],
//! [`parallel_vary`], [`parallel_all_gather`], [`parallel_sum_scatter`], [`parallel_permute`], [`parallel_all_to_all`],
//! and [`parallel_ragged_all_to_all`]. The [`axis_index`] submodule holds the one named-axis operation that exchanges
//! nothing; it reads the current batch item's or device shard's position along the axis. This module also owns the
//! shared machinery of the single-input linear collectives ([`ParallelPermuteOperation`],
//! [`ParallelAllGatherOperation`], [`ParallelSumScatterOperation`], and [`ParallelAllToAllOperation`]). Each carries
//! the referenced axis name and the participant count resolved from the active [`NamedAxes`] environment, consumes one
//! array input, and has only degenerate single-participant semantics outside a binder. Its tangent rides the same
//! collective, and its transpose is another collective over the same axis. The private
//! `LinearCollectiveOperation` trait captures the hooks that their transformation rules need (e.g., the adjoint
//! collective and the forwarding of a collective past an unrelated mapped batch axis) and provides those rules, to
//! which each operation's explicit trait implementations delegate.
//!
//! The collectives that resize an array axis ([`ParallelAllGatherOperation`], [`ParallelSumScatterOperation`], and
//! [`ParallelAllToAllOperation`]) share additional machinery. Their output shapes depend
//! on the participant count and whether the named axis is materialized as a new array axis or tiled into an existing
//! one, so they share:
//!
//!   - `CollectiveArrayExtentBatchingPolicy`, the representation boundary that lets one batching kernel per
//!     collective handle both homogeneous arrays with static extents and composite array/dimension programs with
//!     first-class extents ([`ParallelRaggedAllToAllOperation`] reuses it as well),
//!   - the first-class extent arithmetic that computes and validates result extents at staging time, and
//!   - the [`ArrayIrType`] boundary, where the result extents are passed as additional dimension inputs, with
//!     its type inference, interpretation, batching, and forward-mode differentiation rules, which the private
//!     `ShapeChangingCollectiveOperation` trait provides on top of each collective's matching-axis batching kernel.
//!
//! Collectives reference an enclosing named-axis binder by name, validated against the active
//! [`NamedAxes`] environment at staging time. A name bound by an enclosing `batch` level is
//! resolved at trace time by the operations' batching rules, which collapse or materialize the mapped batch axis at
//! the binding level, while a name bound to a device mesh axis by a `shard_map` manual region stays in the staged
//! body and lowers to cross-device collectives over that mesh axis.

// TODO(eaplatanios): Review this module's docstring.

use std::fmt::Debug;

use crate::arrays::batching::DynamicArrayExtentBatchingPolicy;
use crate::arrays::{
    ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrType,
    ArrayType, Dimension, DimensionType, DimensionValue, DimensionVariable, LinearResiduals, LogicalMesh, MeshAxisType,
    Shape, Sharding, StaticArrayExtentBatchingPolicy,
};
use crate::axes::{AxisError, NamedAxes};
use crate::batching::{BatchAxis, BatchedOutputs, BatchingContext, BatchingError, BatchingPolicy, BatchingTracer};
use crate::contexts::{Context, Domain, DomainProjection, ProjectedContext};
use crate::differentiation::{
    CotangentAccumulator, DifferentiableType, DifferentiationContext, DifferentiationDual, DifferentiationError,
    DifferentiationPolicy, DifferentiationTracer, TranspositionContext,
};
use crate::macros::check_count;
use crate::operations::arithmetic::{AddOperation, Div, Mul, Rem};
use crate::operations::assertions::Assert;
use crate::operations::comparisons::{Compare, ComparisonDirection};
use crate::operations::constants::constant::{ConstantOperation, DimensionConstant};
use crate::operations::differentiation::linear_call::LinearCallOperation;
use crate::operations::dimensions::dimension_max::DimensionMax;
use crate::operations::dimensions::dimension_size::{DimensionSize, DimensionSizeOperation};
use crate::operations::manipulation::broadcasting::{Broadcast, DynamicBroadcast, DynamicBroadcastOperation};
use crate::operations::manipulation::reshaping::{DynamicReshapeOperation, Reshape};
use crate::operations::manipulation::transposition::Transpose;
use crate::partial::PartialValue;
use crate::programs::{
    MaybeZero, Operation, OperationProjection, ProgramError, RegionInterface, Type, TypeError, Typed, Value,
    ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

pub mod axis_index;
pub mod parallel_all_gather;
pub mod parallel_all_to_all;
pub mod parallel_permute;
pub mod parallel_ragged_all_to_all;
pub mod parallel_reduce;
pub mod parallel_sum_scatter;
pub mod parallel_vary;

pub use axis_index::{AXIS_INDEX_OPERATION_NAME, AxisIndex, AxisIndexOperation};
pub use parallel_all_gather::{
    PARALLEL_ALL_GATHER_OPERATION_NAME, ParallelAllGather, ParallelAllGatherOperation, ParallelAllGatherOutputVariance,
};
pub use parallel_all_to_all::{
    PARALLEL_ALL_TO_ALL_OPERATION_NAME, ParallelAllToAll, ParallelAllToAllOperation, ParallelSwapAxes,
};
pub use parallel_permute::{PARALLEL_PERMUTE_OPERATION_NAME, ParallelPermute, ParallelPermuteOperation};
pub use parallel_ragged_all_to_all::{
    PARALLEL_RAGGED_ALL_TO_ALL_OPERATION_NAME, ParallelRaggedAllToAll, ParallelRaggedAllToAllOperation,
};
pub use parallel_reduce::{PARALLEL_REDUCE_OPERATION_NAME, ParallelReduce, ParallelReduceOperation};
pub use parallel_sum_scatter::{PARALLEL_SUM_SCATTER_OPERATION_NAME, ParallelSumScatter, ParallelSumScatterOperation};
pub use parallel_vary::{ManualVariationAlignment, PARALLEL_VARY_OPERATION_NAME, ParallelVary, ParallelVaryOperation};

/// Shape semantics of the collectives that resize an array axis (e.g., [`ParallelAllGatherOperation`],
/// [`ParallelSumScatterOperation`], and [`ParallelAllToAllOperation`]), which determine where the `n` participants of
/// the named axis appear in the shape of the result. In [`Untiled`](Self::Untiled) mode, the participants get an array
/// dimension of their own with extent `n`, which an all-gather inserts and a sum-scatter consumes, so their rank
/// changes, while an all-to-all consumes one such dimension and inserts another, so its rank is preserved. In
/// [`Tiled`](Self::Tiled) mode, the participants are instead folded into an existing array dimension, whose
/// extent is multiplied or divided by `n`, so the rank is always preserved. These are the analogues of
/// the `tiled=False` (the default) and `tiled=True` settings of JAX's
/// [`jax.lax.all_gather`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.all_gather.html),
/// [`jax.lax.psum_scatter`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.psum_scatter.html), and
/// [`jax.lax.all_to_all`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.all_to_all.html).
///
/// Both modes compute the same values and differ only in where the participant dimension lives: an untiled result keeps
/// it as a separate dimension, while a tiled result folds it, participant-major, into an existing one. For example,
/// with `n = 4` participants, an all-gather that concatenates along axis 0, a sum-scatter that scatters along axis 0,
/// and an all-to-all that splits axis 0 and concatenates along axis 1, the shapes are:
///
/// ```text
///   Collective             Participant Input   Untiled Output   Tiled Output
///   ------------------------------------------------------------------------
///   parallel_all_gather    f32[3, 5]           f32[4, 3, 5]     f32[12, 5]
///   parallel_sum_scatter   f32[4, 5]           f32[5]           f32[1, 5]
///   parallel_sum_scatter   f32[12, 5]          (invalid)        f32[3, 5]
///   parallel_all_to_all    f32[4, 6]           f32[6, 4]        f32[1, 24]
///   parallel_all_to_all    f32[8, 6]           (invalid)        f32[2, 24]
/// ```
///
/// The untiled all-gather output stacks the participants' inputs, so index `i` along its new axis 0 holds the input of
/// participant `i`, whereas the tiled all-gather output concatenates them along the existing axis 0, so reshaping the
/// untiled `f32[4, 3, 5]` result to `f32[12, 5]` yields the tiled result exactly. The untiled all-to-all, in contrast,
/// inserts its sender dimension at the concatenation axis, after the dimension it concatenates along, so recovering the
/// tiled `f32[1, 24]` result from the untiled `f32[6, 4]` result also requires moving that sender dimension in front of
/// the concatenated dimension first. An untiled sum-scatter and an untiled all-to-all require the selected axis to have
/// extent exactly `n`, while their tiled forms only require it to be divisible by `n`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum CollectiveMode {
    /// Gives the participants an array dimension of their own with extent `n`: an all-gather inserts it at its
    /// concatenation axis, a sum-scatter consumes its scatter axis (whose extent must be exactly `n`), and an
    /// all-to-all consumes its split axis (whose extent must be exactly `n`) and inserts a sender dimension at
    /// its concatenation axis.
    #[default]
    Untiled,

    /// Folds the participants into an existing array dimension, preserving the rank: an all-gather multiplies the
    /// extent of its concatenation axis by `n`, a sum-scatter divides the extent of its scatter axis by `n`, and an
    /// all-to-all divides the extent of its split axis by `n` and multiplies the extent of its concatenation axis by
    /// `n`. Each divided extent must be divisible by `n`.
    Tiled,
}

impl CollectiveMode {
    /// Returns the physical split axis and mapped result axis when forwarding a collective past a mapped batch axis.
    /// An untiled split consumes the selected input axis, shifting the mapped axis when it follows the consumed axis.
    /// A tiled split preserves the rank and mapped axis position.
    ///
    /// # Parameters
    ///
    ///   - `split_axis`: Axis selected in the logical input, which excludes the mapped batch axis.
    ///   - `batch_axis`: Position of the mapped batch axis in the physical input.
    #[inline]
    fn forwarded_split_axes(self, split_axis: usize, batch_axis: usize) -> (usize, usize) {
        let physical_split_axis = split_axis + usize::from(split_axis >= batch_axis);
        let output_batch_axis = match self {
            Self::Tiled => batch_axis,
            Self::Untiled if split_axis < batch_axis => batch_axis - 1,
            Self::Untiled => batch_axis,
        };
        (physical_split_axis, output_batch_axis)
    }

    /// Returns the physical concatenation axis and mapped result axis when forwarding a collective past a mapped batch
    /// axis. An untiled concatenation inserts a participant axis before the mapped axis when both occupy the same
    /// logical boundary. A tiled concatenation preserves the rank and mapped axis position. For an all-to-all, apply
    /// this function after [`forwarded_split_axes`](Self::forwarded_split_axes), using the mapped axis position
    /// returned by that function.
    ///
    /// # Parameters
    ///
    ///   - `concatenation_axis`: Insertion position for an untiled concatenation, or existing axis for a tiled
    ///     concatenation, in the logical array without the mapped batch axis and after any split performed by
    ///     the caller.
    ///   - `batch_axis`: Position of the mapped batch axis in the physical array before the concatenation.
    #[inline]
    fn forwarded_concatenation_axes(self, concatenation_axis: usize, batch_axis: usize) -> (usize, usize) {
        match self {
            Self::Tiled => (concatenation_axis + usize::from(concatenation_axis >= batch_axis), batch_axis),
            Self::Untiled if concatenation_axis <= batch_axis => (concatenation_axis, batch_axis + 1),
            Self::Untiled => (concatenation_axis + 1, batch_axis),
        }
    }
}

/// Shared shape and grouping options for collective operations that resize an array axis (e.g.,
/// [`ParallelAllGatherOperation`], [`ParallelSumScatterOperation`], and [`ParallelAllToAllOperation`]).
#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub struct CollectiveOptions {
    /// [`CollectiveMode`] of the collective.
    mode: CollectiveMode,

    /// Optional ordered partition of logical participant indices.
    axis_index_groups: Option<Vec<Vec<usize>>>,
}

impl CollectiveOptions {
    /// Creates a new [`CollectiveOptions`] instance for `mode` with no participant subgroups.
    #[inline]
    pub fn new(mode: CollectiveMode) -> Self {
        Self { mode, axis_index_groups: None }
    }

    /// Creates a new rank-preserving tiled [`CollectiveOptions`] instance with no participant subgroups.
    #[inline]
    pub fn tiled() -> Self {
        Self::new(CollectiveMode::Tiled)
    }

    /// Returns this [`CollectiveOptions`] instance with the provided ordered participant groups.
    #[inline]
    pub fn with_axis_index_groups(mut self, axis_index_groups: Vec<Vec<usize>>) -> Self {
        self.axis_index_groups = Some(axis_index_groups);
        self
    }

    /// Returns the [`CollectiveMode`] of this [`CollectiveOptions`] instance.
    #[inline]
    pub fn mode(&self) -> CollectiveMode {
        self.mode
    }

    /// Returns the ordered participant groups of this [`CollectiveOptions`] instance, if any.
    #[inline]
    pub fn axis_index_groups(&self) -> Option<&[Vec<usize>]> {
        self.axis_index_groups.as_deref()
    }

    /// Validates these options against the full named-axis size and returns the effective group size used for shape
    /// arithmetic. Refer to the documentation of [`effective_collective_axis_size`] for more information.
    #[inline]
    pub(super) fn effective_axis_size(&self, operation_name: &str, axis_size: usize) -> Result<usize, TypeError> {
        effective_collective_axis_size(operation_name, axis_size, self.axis_index_groups())
    }
}

impl Debug for CollectiveOptions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.axis_index_groups {
            None => Debug::fmt(&self.mode, formatter),
            Some(axis_index_groups) => formatter
                .debug_struct("CollectiveOptions")
                .field("mode", &self.mode)
                .field("axis_index_groups", axis_index_groups)
                .finish(),
        }
    }
}

/// Validates that `axis_name` is a manual axis of `mesh` and, when `axis_size` is provided, that the size the collective
/// recorded at staging time matches the size of that mesh axis. Collectives that carry a mesh use it to validate the
/// manual axis they exchange values over (e.g., [`AxisIndexOperation`], which has no input).
///
/// # Errors
///
/// Returns a [`TypeError`] naming `operation_name` if the axis is not a manual axis of `mesh` or if its size differs.
fn validate_manual_mesh_axis(
    operation_name: &str,
    axis_name: &str,
    axis_size: Option<usize>,
    mesh: &LogicalMesh,
) -> Result<(), TypeError> {
    if mesh.axis_type(axis_name) != Some(MeshAxisType::Manual) {
        return Err(TypeError::invalid(format!("`{operation_name}` mesh axis `{axis_name}` must be manual")));
    }
    if let Some(axis_size) = axis_size
        && mesh.axis_size(axis_name) != Some(axis_size)
    {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` axis size {axis_size} does not match the size of manual mesh axis `{axis_name}`",
        )));
    }
    Ok(())
}

/// Validates the manual mesh axis of a collective that carries a mesh, as [`validate_manual_mesh_axis`] does, and that
/// `input_type` carries sharding over that same mesh. After successful validation, callers can retrieve the input
/// sharding to apply their collective-specific manual variation and pending-sum contracts.
///
/// # Errors
///
/// Returns a [`TypeError`] naming `operation_name` if the mesh axis is invalid, or if the input carries no sharding or
/// a sharding over a different mesh.
fn validate_manual_mesh_input(
    operation_name: &str,
    axis_name: &str,
    axis_size: Option<usize>,
    mesh: &LogicalMesh,
    input_type: &ArrayType,
) -> Result<(), TypeError> {
    validate_manual_mesh_axis(operation_name, axis_name, axis_size, mesh)?;
    let Some(sharding) = input_type.sharding() else {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` input must carry a mesh containing manual axis `{axis_name}`",
        )));
    };
    if sharding.mesh() != mesh {
        return Err(TypeError::invalid(format!("`{operation_name}` input mesh does not match the operation mesh")));
    }
    Ok(())
}

/// Validates the participant grouping of a collective over a named axis of size `axis_size` and returns its _effective
/// axis size_, which is the number of participants that each instance of the collective combines. Without `groups`,
/// every participant along the axis takes part in one collective, so the effective axis size is `axis_size` itself.
/// With `groups`, the axis is split into independent collectives, one per group, and the effective axis size is the
/// common group size. Callers use it wherever shapes or values depend on the participant count (e.g., the gathered
/// extent of a `parallel_all_gather` operation, the chunk extent of a `parallel_sum_scatter` operation, or the divisor
/// of a mean operation).
///
/// For example, an axis of size 4 split into the groups `[[0, 2], [3, 1]]` runs two independent collectives over two
/// participants each, so its effective axis size is 2:
///
/// ```text
///   participant:   0   1   2   3
///   group:         A   B   A   B      (A = [0, 2] and B = [3, 1])
///   collectives:   A combines participants 0 and 2, and B combines participants 3 and 1
/// ```
///
/// The groups must form an equal-sized exact partition of `0..axis_size`:
///
///   - `axis_size` must be positive, and there must be at least one group, whose size is at least one.
///   - Every group must have the same size as the first one.
///   - Every participant in `0..axis_size` must appear in exactly one group, which rules out out-of-bounds, repeated,
///     and missing participants.
///
/// This function only validates the groups and borrows them without copying. The order of the groups and of the
/// participants within each group is part of the collective's semantics and is preserved by its owner (e.g., the
/// XLA backend's lowering emits replica groups in this order), even though this validation does not depend on it.
///
/// # Parameters
///
///   - `operation_name`: Name of the collective, used in diagnostics.
///   - `axis_size`: Full size of the named axis, which every participant index must be smaller than.
///   - `groups`: Optional ordered participant groups.
///
/// # Errors
///
/// Returns a [`TypeError`] that names `operation_name` and describes the first violated requirement, checking the
/// requirements above in order and the groups and their participants in order.
fn effective_collective_axis_size(
    operation_name: &str,
    axis_size: usize,
    groups: Option<&[Vec<usize>]>,
) -> Result<usize, TypeError> {
    if axis_size == 0 {
        return Err(TypeError::invalid(format!("`{operation_name}` axis size must be greater than zero")));
    }

    let Some(groups) = groups else {
        return Ok(axis_size);
    };

    let Some(first_group) = groups.first() else {
        return Err(TypeError::invalid(format!("`{operation_name}` axis index groups must not be empty")));
    };

    if first_group.is_empty() {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` axis index groups must contain at least one participant",
        )));
    }

    let group_size = first_group.len();
    let mut seen = vec![false; axis_size];
    for (group_index, group) in groups.iter().enumerate() {
        if group.len() != group_size {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` axis index group {group_index} has size {} but every group must have size \
                     {group_size}",
                group.len(),
            )));
        }

        for &participant in group {
            let Some(participant_seen) = seen.get_mut(participant) else {
                return Err(TypeError::invalid(format!(
                    "`{operation_name}` axis index {participant} is out of bounds for axis size {axis_size}",
                )));
            };

            if *participant_seen {
                return Err(TypeError::invalid(format!(
                    "`{operation_name}` axis index groups contain participant {participant} more than once",
                )));
            }

            *participant_seen = true;
        }
    }

    if let Some(missing) = seen.iter().position(|seen| !seen) {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` axis index groups do not contain participant {missing}",
        )));
    }

    Ok(group_size)
}

/// Resolves the static, non-zero size of the named axis bound by the active [`NamedAxes`] environment, failing fast
/// with [`AxisError::UnboundAxisName`] when no enclosing binder binds `axis_name`. The collective capabilities bake
/// the resolved size into their operation payloads at staging time, because their output shapes and payload validation
/// depend on it while [`Operation::infer_output_types`] only sees input types.
fn resolve_named_axis_size<C: NamedAxes>(context: &C, axis_name: &str) -> Result<usize, ProgramError> {
    match context
        .named_axis(axis_name)
        .ok_or_else(|| AxisError::UnboundAxisName { name: axis_name.to_string() })?
        .size()
    {
        Some(0) => {
            Err(TypeError::invalid(format!("collective axis `{axis_name}` must contain at least one participant"))
                .into())
        }
        Some(size) => Ok(size),
        None => Err(BatchingError::UnsupportedOperation {
            message: format!("collective axis `{axis_name}` has a dynamic extent that must remain a first-class input"),
        }
        .into()),
    }
}

/// Infers a linear collective's output type from its input and (possibly resized) dimensions, carrying the input
/// sharding through with the same per-dimension placement (the dimension count never changes). An unchanged shape
/// preserves the complete input type. Resizing clears explicit layout information because input strides and tiling
/// do not generally describe storage for the resized shape; element type and memory are preserved.
fn infer_linear_collective_operation_output_type(
    operation_name: &'static str,
    input_type: &ArrayType,
    output_dimensions: Vec<usize>,
) -> Result<ArrayType, TypeError> {
    let output_sizes = output_dimensions.into_iter().map(Dimension::Static).collect::<Vec<_>>();
    let output_shape = Shape::new(output_sizes);
    if &output_shape == input_type.shape() {
        return Ok(input_type.clone());
    }
    let output_sharding = input_type.resized_sharding(output_shape.dimensions(), operation_name)?;
    Ok(ArrayType::new(input_type.data_type(), output_shape)
        .with_sharding(output_sharding)?
        .with_memory(input_type.memory()))
}

/// Infers one canonical mixed collective result from an array input followed by one explicit extent per output axis.
///
/// # Parameters
///
///   - `operation_name`: Name of the collective, used in diagnostics.
///   - `input_types`: Array input type followed by one explicit extent type per output axis.
///   - `base_output_type`: Output type whose shape is replaced by the explicit extents.
///   - `changed_output_axes`: Output axes whose extents may differ from `base_output_type`. Every other axis must
///     retain the extent already projected into `base_output_type` by the caller.
///   - `validate_exact_extents_fn`: Collective-specific validation of the explicit output extents.
fn infer_array_ir_shape_changing_collective_output_type(
    operation_name: &'static str,
    input_types: &[ArrayIrType],
    base_output_type: ArrayType,
    changed_output_axes: &[usize],
    validate_exact_extents_fn: impl FnOnce(&[Dimension]) -> Result<(), TypeError>,
) -> Result<Vec<ArrayIrType>, TypeError> {
    check_count!("input", input_types, 1 + base_output_type.rank(), TypeError);

    // Only the kind of the first input is checked here. Each collective applies its own pending-sum contract, and
    // the shape-only output type preserves every other piece of the input's mesh state.
    <&ArrayType>::try_from(&input_types[0])?;

    let output_extents = ArrayIrType::extents(&input_types[1..])?;
    for (output_axis, (expected_extent, output_extent)) in
        base_output_type.shape().dimensions().iter().zip(&output_extents).enumerate()
    {
        if !changed_output_axes.contains(&output_axis) && output_extent != expected_extent {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` output axis {output_axis} extent {output_extent} must equal unchanged extent \
                 {expected_extent}",
            )));
        }
    }

    validate_exact_extents_fn(output_extents.as_slice())?;
    Ok(vec![base_output_type.with_shape(Shape::new(output_extents)).into()])
}

// TODO(eaplatanios): Move this to right after `impl Debug for CollectiveOptions`.
/// Single-input linear collective operation over a named axis (i.e., [`ParallelPermuteOperation`],
/// [`ParallelAllGatherOperation`], [`ParallelSumScatterOperation`], or [`ParallelAllToAllOperation`]). Each carries
/// the referenced axis name, the participant count resolved from the active [`NamedAxes`] environment, and, when it
/// exchanges values over a manual mesh axis, that axis's mesh. Its tangent rides the same collective, and its transpose
/// is another collective over the same axis, so this trait captures the hooks that the transformation rules of every
/// such collective need and provides those rules on top of them. The collectives keep explicit trait implementations
/// that delegate to the provided functions, so that every operation module reads the same way.
///
/// The hooks named after a public accessor of the operation (e.g., [`axis_name`](Self::axis_name)) return the same
/// values. They are repeated here because this trait is private, while the accessors are part of the public API.
trait LinearCollectiveOperation: Clone + Operation<Type = ArrayType> {
    /// Collective operation type that transposition stages on the output cotangent.
    type Adjoint: Clone + Operation<Type = ArrayType>;

    /// Returns the name of the axis that this collective exchanges values over.
    fn axis_name(&self) -> &str;

    /// Returns the number of participants along the named axis, resolved when the collective was staged.
    fn axis_size(&self) -> usize;

    /// Returns the mesh whose manual axis this collective exchanges values over, or [`None`] for an ordinary
    /// collective, whose named axis may be bound by any enclosing binder.
    fn mesh(&self) -> Option<&LogicalMesh>;

    /// Returns the number of participants that each instance of this collective combines. Collectives with participant
    /// groups return the common group size while every other collective combines all participants of its axis.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] if the participant groups of this collective are invalid.
    #[inline]
    fn effective_axis_size(&self) -> Result<usize, TypeError> {
        Ok(self.axis_size())
    }

    /// Returns the adjoint collective that transposition stages on the output cotangent of a collective whose array
    /// input has type `input_type`. Most collectives ignore `input_type`, but a sum-scatter that completes a pending
    /// sum over its manual axis needs a reduced, rather than varying, all-gather to restore its input cotangent's
    /// state.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] for configurations that have no adjoint collective (e.g., an invariant all-gather,
    /// whose transpose selects a participant-indexed chunk instead).
    fn adjoint(&self, input_type: &ArrayType) -> Result<Self::Adjoint, ProgramError>;

    /// Returns this collective with its array axes adjusted around the mapped batch axis at `input_batch_axis` in the
    /// physical input of a `batch` level that does not bind its named axis, together with the position of the mapped
    /// batch axis in the physical result.
    fn adapt_to_batch_axis(&self, input_batch_axis: usize) -> (Self, usize);

    /// Validates the input contract that every linear collective shares (i.e., no regions, a non-zero axis size, and
    /// exactly one input) and returns that input's type.
    fn check_input<'o>(
        &self,
        input_types: &'o [ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<&'o ArrayType, TypeError> {
        check_count!("region", region_interfaces, 0, TypeError);

        // A zero-participant collective is rejected before any extent arithmetic divides by its size.
        if self.axis_size() == 0 {
            return Err(TypeError::invalid(format!("`{}` axis size must be greater than zero", self.name())));
        }

        check_count!("input", input_types, 1, TypeError);
        Ok(&input_types[0])
    }

    /// Checks whether this collective can be evaluated locally, without an enclosing binder that supplies its
    /// participants. Each participant group must contain only one participant; a larger group requires values
    /// from other participants that evaluating one array in isolation cannot provide.
    ///
    /// This single-participant case is called degenerate because no exchange between participants is needed. The
    /// function only validates that condition; the caller computes the local result, which may still change the shape
    /// (e.g., an untiled all-gather inserts a size-one array axis).
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if the participant groups are invalid or require more than one participant per group.
    fn check_degenerate_interpretation(&self) -> Result<(), ProgramError> {
        let effective_axis_size = self.effective_axis_size()?;
        if effective_axis_size > 1 {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "cannot interpret `{}` over axis `{}` of size {} without an enclosing binder",
                    self.name(),
                    self.axis_name(),
                    effective_axis_size,
                ),
            });
        }
        Ok(())
    }

    /// Rejects replacing a manual mesh collective with local array operations at a `batch` level that binds its axis
    /// name. A matching batch level normally implements the collective by rearranging or reducing the elements of its
    /// batch dimension. A collective carrying a [`mesh`](Self::mesh) instead describes communication between devices,
    /// which those local batch elements cannot stand in for.
    ///
    /// For example, a local batch axis named `devices` cannot take over an exchange across a manual mesh axis also
    /// named `devices`, even though the batch axis shadows the mesh axis's name.
    ///
    /// Call this function when the batching level would consume the collective. An unrelated batch level can still
    /// forward the mesh collective to its parent context without consuming it.
    ///
    /// # Errors
    ///
    /// Returns a [`BatchingError`] if this collective carries a mesh.
    fn reject_mesh_form(&self) -> Result<(), BatchingError> {
        if self.mesh().is_some() {
            return Err(BatchingError::UnsupportedOperation {
                message: format!("`{}` over a manual mesh axis cannot bind a named batch axis", self.name()),
            });
        }
        Ok(())
    }

    /// Implements [`DifferentiableOperation::jvp`](crate::DifferentiableOperation::jvp) for this collective. The
    /// collective is linear, so its tangent rides the same collective as its primal, while a structural-zero tangent
    /// stays symbolic, retyped to the output tangent type because the collective may change shapes.
    fn linear_collective_jvp<C: Context<Type = ArrayType, Operation: From<Self>>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let mut primals = context.primal().bind(self.clone(), Vec::new(), std::slice::from_ref(inputs[0].primal()))?;
        check_count!("output", primals, 1, ProgramError);
        let primal = primals.remove(0);
        let tangent = match inputs[0].tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
            MaybeZero::Value(tangent) => {
                let mut tangents = context.tangent().bind(self.clone(), Vec::new(), std::slice::from_ref(tangent))?;
                check_count!("output", tangents, 1, ProgramError);
                MaybeZero::Value(tangents.remove(0))
            }
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }

    /// Implements [`TransposableOperation::transpose`](crate::TransposableOperation::transpose) for this collective
    /// by staging its [`adjoint`](Self::adjoint) on the output cotangent. A known input and a structural-zero output
    /// cotangent contribute nothing, which leaves the input cotangent a structural zero.
    fn linear_collective_transpose<
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType> + From<AddOperation<ArrayType>> + From<Self::Adjoint>,
    >(
        &self,
        context: &mut TranspositionContext<V, O>,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        check_count!("accumulator", accumulators, 1, DifferentiationError);

        // The adjoint is resolved first, so that a configuration without one is rejected regardless of the cotangent,
        // and only a live output cotangent of an unknown input then stages it.
        let adjoint = self.adjoint(inputs[0].r#type().as_ref())?;
        let MaybeZero::Value(cotangent) = &outputs[0] else {
            return Ok(());
        };

        if inputs[0].is_known() {
            return Ok(());
        }

        let mut contributions = context.bind(O::from(adjoint), Vec::new(), std::slice::from_ref(cotangent))?;
        check_count!("output", contributions, 1, ProgramError);
        accumulators[0].accumulate(context, MaybeZero::Value(contributions.remove(0)))?;
        Ok(())
    }
}

// TODO(eaplatanios): Review form here onwards.

/// [`LinearCollectiveOperation`] that resizes an array axis (i.e., [`ParallelAllGatherOperation`],
/// [`ParallelSumScatterOperation`], or [`ParallelAllToAllOperation`]). Its output shape depends on the participant
/// count and its [`CollectiveMode`], so in the composite array/dimension family it is staged with one explicit extent
/// input per output axis. This trait captures the hooks that differ between these collectives (i.e., their options and
/// their composite type inference) and provides the composite interpretation and forward-mode differentiation rules,
/// together with the batching rules of both array families, on top of them and of their
/// [`ShapeChangingCollectiveBatching`] implementations.
trait ShapeChangingCollectiveOperation: LinearCollectiveOperation {
    /// Returns the shared rank and participant-group semantics of this collective.
    fn options(&self) -> &CollectiveOptions;

    /// Infers the output type of this collective in the composite array/dimension family, whose array input is followed
    /// by one explicit extent per output axis. Statically known extents are checked here, while dynamic extents are
    /// checked by the runtime assertions that the collective's capability stages.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] if the inputs violate the collective's shape, extent, or mesh contract.
    fn infer_array_ir_output_types(&self, input_types: &[ArrayIrType]) -> Result<Vec<ArrayIrType>, TypeError>;

    /// Returns the error that the provided batching rules raise for a bounded ragged `dimension` on input
    /// `input_index`, which these collectives cannot route because one extent per item does not describe how the
    /// participants partition their live elements.
    fn ragged_input_error(&self, dimension: &DimensionVariable, input_index: usize) -> BatchingError {
        BatchingError::UnsupportedOperation {
            message: format!(
                "`{}` does not support bounded ragged dimension `{}` on input {}",
                self.name(),
                dimension,
                input_index,
            ),
        }
    }

    /// Implements [`interpret_in_parent`](crate::interpretation::MemberInterpretableOperation::interpret_in_parent) for
    /// this collective. Outside any binder, only a collective whose instances each combine a single participant has
    /// defined semantics: a tiled one leaves the array unchanged, and an untiled one only removes or inserts a size-one
    /// axis. The explicit result extents must match the shape that the observed input implies.
    fn shape_changing_collective_interpret<C>(&self, inputs: &[C::Value]) -> Result<Vec<C::Value>, ProgramError>
    where
        C: Domain<
                Type = ArrayIrType,
                Value: ValueProjection<
                    ArrayType,
                    Projected: Value<Type = ArrayType> + DimensionSize<usize> + Reshape,
                > + ValueProjection<DimensionType, Projected = DimensionValue>,
            >,
    {
        // The composite collective consumes one array followed by a dimension value for each result axis.
        let Some((input, output_extents)) = inputs.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 });
        };
        let input = <C::Value as ValueProjection<ArrayType>>::into_projected(input.clone())?;

        // Resolve symbolic input dimensions from the actual array, then reuse homogeneous type inference to validate
        // the collective's geometry and compute the concrete result shape while retaining the input metadata.
        let concrete_input_type = input.r#type().as_ref().clone().with_shape(Shape::new(
            (0..input.r#type().rank())
                .map(|axis| input.dimension_size(axis).map(Dimension::Static))
                .collect::<Result<Vec<_>, _>>()?,
        ));
        let mut output_types = self.infer_output_types(std::slice::from_ref(&concrete_input_type), &[])?;
        check_count!("output", output_types, 1, ProgramError);
        let output_type = output_types.remove(0);
        let expected_extents = output_type.static_shape().ok_or_else(|| {
            TypeError::invalid(format!("`{}` could not resolve its concrete output shape", self.name()))
        })?;

        // Explicit result extents must agree with the shape implied by the observed input and the collective options,
        // because accepting arbitrary extents here would let dynamic shape inputs change the collective's semantics.
        if output_extents.len() != expected_extents.rank() {
            return Err(ProgramError::InvalidInputCount {
                expected: 1 + expected_extents.rank(),
                actual: inputs.len(),
            });
        }

        for (axis, (extent, expected)) in output_extents.iter().zip(expected_extents.dimensions()).enumerate() {
            let actual = ValueProjection::<DimensionType>::into_projected(extent.clone())?.extent();
            if actual != *expected {
                return Err(ProgramError::InvalidArgument {
                    message: format!(
                        "`{}` output axis {axis} extent must equal observed result extent {expected} but got {actual}",
                        self.name(),
                    ),
                });
            }
        }

        // A degenerate tiled collective leaves the array unchanged, and its untiled form only removes or inserts a
        // size-one axis, so reshaping to the validated result shape is sufficient and preserves element order.
        self.check_degenerate_interpretation()?;
        let output = match self.options().mode() {
            CollectiveMode::Tiled => input,
            CollectiveMode::Untiled => input.reshape(Shape::from(expected_extents))?,
        };
        Ok(vec![<C::Value as ValueProjection<ArrayType>>::from_projected(output)])
    }

    /// Implements [`jvp_in_parent`](crate::differentiation::MemberDifferentiableOperation::jvp_in_parent) for this
    /// collective. The explicit output extents and the exact input shape become ordinary residuals of one linear call,
    /// whose transpose applies the [`adjoint`](LinearCollectiveOperation::adjoint) of the primal array input to the
    /// output cotangent.
    fn shape_changing_collective_jvp<C, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError>
    where
        C: Context<
                Type = ArrayIrType,
                Operation: From<Self>
                               + From<Self::Adjoint>
                               + From<DimensionSizeOperation>
                               + From<LinearCallOperation<ArrayIrType>>
                               + From<ConstantOperation<DimensionValue>>,
            >,
    {
        let Some((array, _)) = inputs.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };
        let input_type = array.primal().r#type();
        let adjoint = self.adjoint(<&ArrayType>::try_from(input_type.as_ref())?)?;
        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal = context.primal().bind(self.clone(), Vec::new(), primal_inputs.as_slice())?.remove(0);
        let tangent = match array.tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
            MaybeZero::Value(array_tangent) => {
                let tangent_inputs = context.dual_primal_to_tangent(inputs)?;
                let (array, output_extents) = tangent_inputs.split_first().unwrap();
                let context = context.tangent();
                let mut residuals = LinearResiduals::new();
                let output_extents = residuals.retain_all(output_extents.iter().map(|extent| extent.primal().clone()));
                let input_shape = residuals.retain_shape(context, array.primal())?;
                let forward_operation = self.clone();
                let forward_output_extents = output_extents.clone();
                let tangent = LinearCallOperation::stage(
                    context,
                    residuals.into_values(),
                    vec![array_tangent.clone()],
                    move |residuals, linear_inputs| {
                        let mut collective_inputs = Vec::with_capacity(1 + forward_output_extents.len());
                        collective_inputs.push(linear_inputs[0].clone());
                        collective_inputs.extend(forward_output_extents.iter().map(|index| residuals[*index].clone()));
                        linear_inputs[0].dispatch_domain().bind(
                            forward_operation,
                            Vec::new(),
                            collective_inputs.as_slice(),
                        )
                    },
                    move |residuals, output_cotangents| {
                        let transpose_context = output_cotangents[0].dispatch_domain();
                        let input_dimensions = input_shape.dimensions(&transpose_context, residuals)?;
                        let mut adjoint_inputs = Vec::with_capacity(1 + input_dimensions.len());
                        adjoint_inputs.push(output_cotangents[0].clone());
                        adjoint_inputs.extend(input_dimensions);
                        transpose_context.bind(adjoint, Vec::new(), adjoint_inputs.as_slice())
                    },
                )?
                .remove(0);
                MaybeZero::Value(tangent)
            }
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }

    /// Implements [`BatchableOperation::batch`](crate::batching::BatchableOperation::batch) for this collective in the
    /// homogeneous array family. Bounded ragged inputs are rejected. A `batch` level that does not bind the
    /// collective's axis forwards it to its parent with its array axes moved past the mapped axis, while a level that
    /// binds the axis consumes it through [`batch_matching_axis`](ShapeChangingCollectiveBatching::batch_matching_axis).
    fn shape_changing_collective_batch<C, P: CollectiveArrayExtentBatchingPolicy<C>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError>
    where
        C: Context<Type = ArrayType, Operation: From<Self>>,
        Self: ShapeChangingCollectiveBatching<C>,
    {
        if let Some((index, ragged_axis)) = inputs
            .iter()
            .enumerate()
            .find_map(|(index, input)| input.ragged_axes().first().map(|axis| (index, axis)))
        {
            return Err(self.ragged_input_error(ragged_axis.dimension(), index));
        }

        if context.axis_name() != Some(self.axis_name()) {
            return forward_linear_collective(context, self, inputs);
        }

        self.reject_mesh_form()?;
        let [input] = inputs else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() }.into());
        };

        let (output_type, output_extents) = collective_output_extents(context, self, &input.unbatched_type())?;
        Ok(vec![self.batch_matching_axis(context, input, output_extents, output_type.sharding().cloned())?].into())
    }

    /// Implements [`batch_in_parent`](crate::batching::MemberBatchableOperation::batch_in_parent) for this collective
    /// in the composite array/dimension family, whose explicit result extents remain the only source of dynamic reshape
    /// geometry. Bounded ragged inputs are rejected, and the result extents, which describe the shape shared by every
    /// batch item, must be replicated. A `batch` level that does not bind the collective's axis forwards it to its
    /// parent, while a level that binds the axis consumes it through
    /// [`batch_matching_axis`](ShapeChangingCollectiveBatching::batch_matching_axis) over the array projection of its
    /// parent.
    fn shape_changing_collective_batch_in_parent<C>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatchingPolicy>, BatchingError>
    where
        C: Context<
                Type = ArrayIrType,
                Value: Assert
                           + DimensionSize
                           + DynamicBroadcast
                           + ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>
                           + ValueProjection<
                    DimensionType,
                    Projected: Compare<C::Value> + DimensionMax + Rem + Div + Mul + Value<Type = DimensionType>,
                >,
                Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
                Operation: From<Self>
                               + From<DynamicBroadcastOperation>
                               + From<ConstantOperation<DimensionValue>>
                               + From<DimensionSizeOperation>
                               + From<DynamicReshapeOperation>
                               + OperationProjection<ArrayType>,
            >,
        Self: ShapeChangingCollectiveBatching<ProjectedContext<C, ArrayType>>,
    {
        let Some((array, output_extents)) = inputs.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };
        <&ArrayType>::try_from(&array.unbatched_type())?;
        if let Some((index, ragged_axis)) = inputs
            .iter()
            .enumerate()
            .find_map(|(index, input)| input.ragged_axes().first().map(|axis| (index, axis)))
        {
            return Err(self.ragged_input_error(ragged_axis.dimension(), index));
        }

        for output_extent in output_extents {
            output_extent.validate_replicated_dimension()?;
        }

        // Infer the per-item result type before lifting physical axes. This also supplies the sharding metadata used by
        // the matching-axis kernel.
        let logical_input_types = inputs.iter().map(|input| input.unbatched_type().clone()).collect::<Vec<_>>();
        let mut logical_output_types = self.infer_array_ir_output_types(logical_input_types.as_slice())?;
        let logical_output_type = <&ArrayType>::try_from(&logical_output_types.remove(0))?.clone();

        if context.axis_name() != Some(self.axis_name()) {
            return Ok(context.forward_collective(self, array, output_extents)?.into());
        }

        self.reject_mesh_form()?;

        // Project the composite values onto their array and dimension domains, so that the homogeneous kernel can use
        // the explicit dimension values directly for dynamic reshape geometry.
        let array = ArrayBatch::new(
            <C::Value as ValueProjection<ArrayType>>::into_projected(array.value().clone())?,
            array.batch_axis(),
        )?;
        let output_extents = output_extents
            .iter()
            .map(|extent| <C::Value as ValueProjection<DimensionType>>::into_projected(extent.value().clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let output = self.batch_matching_axis::<DynamicArrayExtentBatchingPolicy>(
            &context.array_projection(),
            &array,
            output_extents,
            logical_output_type.sharding().cloned(),
        )?;

        // Embed the array result back into the composite family without changing the batch axis that the kernel chose.
        let batch_axis = output.batch_axis();
        Ok(ArrayIrBatch::new(<C::Value as ValueProjection<ArrayType>>::from_projected(output.into_value()), batch_axis)
            .map(|output| vec![output])?
            .into())
    }
}

/// Context-specific batching capability of a [`ShapeChangingCollectiveOperation`] for `batch` levels whose parent is
/// `C`. Its function consumes a level that binds the collective's named axis; the shared batching rules use it after
/// validating the inputs and determining the output geometry.
///
/// The context parameter belongs to this trait so that each collective's implementation can require exactly the value
/// capabilities it uses: all-gather and all-to-all require [`Transpose`], while sum-scatter also requires [`Reduce`].
/// Keeping this capability separate from [`ShapeChangingCollectiveOperation`] leaves type inference, interpretation,
/// and differentiation independent of batching contexts. A generic function on that operation trait would give every
/// implementation the same context bounds and prevent sum-scatter from adding its reduction requirement separately.
trait ShapeChangingCollectiveBatching<C: Context<Type = ArrayType>>: ShapeChangingCollectiveOperation {
    /// Consumes the mapped batch axis of a `batch` level that binds this collective's named axis, given the per-item
    /// output extents and sharding in the batching policy's representation, and returns the result together with the
    /// output batch axis that the collective chooses (e.g., replicated for an all-gather, whose items all receive the
    /// same value).
    ///
    /// # Errors
    ///
    /// Returns a [`BatchingError`] if the collective cannot be consumed by the binding level (e.g., because it has
    /// participant groups) or if staging the exchange fails.
    fn batch_matching_axis<P: CollectiveArrayExtentBatchingPolicy<C>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        input: &ArrayBatch<C::Value>,
        output_extents: Vec<P::ShapeExtent>,
        output_sharding: Option<Sharding>,
    ) -> Result<ArrayBatch<C::Value>, BatchingError>;
}

/// Value that stages shape-changing collectives directly through its homogeneous array dispatch domain.
///
/// This marker opts a value into the provided [`ParallelAllGather`], [`ParallelSumScatter`], and [`ParallelAllToAll`]
/// implementations. Each implementation separately requires its operation to be supported by the dispatch domain,
/// named-axis resolution through [`NamedAxes`], and manual variation through [`ParallelVary`]; implementing this trait
/// alone does not require support for every collective. Backend array types implement it to reuse these staging rules
/// without defining their own collective capability implementations.
///
/// Homogeneous [`Tracer`], [`BatchingTracer`], and [`DifferentiationTracer`] values opt in. Projected array values
/// instead delegate through their composite value so that runtime output extents remain explicit inputs; they must
/// not implement this trait. Concrete host arrays retain their own unbound-axis diagnostics and do not opt in either.
pub trait ShapeChangingCollectiveValue: Value<Type = ArrayType> {}

impl<C: Context> ShapeChangingCollectiveValue for Tracer<C> where Self: Value<Type = ArrayType> {}

impl<C: Context, P: BatchingPolicy<C>> ShapeChangingCollectiveValue for BatchingTracer<C, P> where
    Self: Value<Type = ArrayType>
{
}

impl<C: Context, P: DifferentiationPolicy<C>> ShapeChangingCollectiveValue for DifferentiationTracer<C, P> where
    Self: Value<Type = ArrayType>
{
}

/// Representation boundary used only by shape-changing collective batching rules.
///
/// The collective kernels own every formula. This trait exposes only the extent representation and the alignment and
/// reshape encodings that differ between homogeneous arrays and composite array/dimension programs.
trait CollectiveArrayExtentBatchingPolicy<C: Context<Type = ArrayType>>: ArrayExtentBatchingPolicy<C> {
    /// Extent representation consumed by the shared collective kernels.
    type ShapeExtent: Clone + Debug + Div + Mul;

    /// Returns and validates the active mapped-axis extent in the kernel's representation.
    fn collective_axis_extent(
        context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
        operation_name: &str,
        axis_name: &str,
        axis_size: usize,
    ) -> Result<Self::ShapeExtent, BatchingError>;

    /// Materializes a statically known extent in the kernel's representation.
    fn collective_extent_constant(
        context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
        extent: usize,
    ) -> Result<Self::ShapeExtent, BatchingError>;

    /// Materializes a statically known type-level dimension in the kernel's representation.
    fn collective_extent_from_dimension(
        context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
        dimension: &Dimension,
    ) -> Result<Self::ShapeExtent, BatchingError> {
        let extent = dimension.value().ok_or_else(|| BatchingError::UnsupportedOperation {
            message: "shape-changing collective batching requires statically shaped inputs".to_string(),
        })?;
        Self::collective_extent_constant(context, extent)
    }

    /// Enforces exact divisibility and returns a positive divisor safe for subsequent arithmetic.
    fn require_divisible_collective_extents(
        context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
        left: &Self::ShapeExtent,
        right: &Self::ShapeExtent,
    ) -> Result<Self::ShapeExtent, BatchingError>;

    /// Aligns `batch` to the leading mapped axis using its complete logical input extents.
    fn match_collective_axis(
        context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
        batch: &ArrayBatch<C::Value>,
        input_extents: &[Self::ShapeExtent],
    ) -> Result<ArrayBatch<C::Value>, BatchingError>;

    /// Reshapes `value` using a complete extent list in this policy's representation.
    fn reshape_collective(
        context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
        value: C::Value,
        output_extents: &[Self::ShapeExtent],
        output_sharding: Option<Sharding>,
    ) -> Result<C::Value, BatchingError>;
}

impl<C> CollectiveArrayExtentBatchingPolicy<C> for StaticArrayExtentBatchingPolicy
where
    C: Context<Type = ArrayType, Value: Broadcast + Reshape + Transpose>,
{
    type ShapeExtent = usize;

    fn collective_axis_extent(
        context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
        operation_name: &str,
        axis_name: &str,
        axis_size: usize,
    ) -> Result<Self::ShapeExtent, BatchingError> {
        let batch_size = *context.axis_extent();
        if batch_size != axis_size {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{operation_name}` over axis `{axis_name}` resolved axis size {axis_size} but the mapped batch \
                     axis has size {batch_size}",
                ),
            });
        }
        Ok(batch_size)
    }

    fn collective_extent_constant(
        _context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
        extent: usize,
    ) -> Result<Self::ShapeExtent, BatchingError> {
        Ok(extent)
    }

    fn require_divisible_collective_extents(
        _context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
        left: &Self::ShapeExtent,
        right: &Self::ShapeExtent,
    ) -> Result<Self::ShapeExtent, BatchingError> {
        if *right == 0 || left % right != 0 {
            return Err(BatchingError::UnsupportedOperation {
                message: format!("extent {left} must be divisible by extent {right}"),
            });
        }
        Ok(*right)
    }

    fn match_collective_axis(
        context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
        batch: &ArrayBatch<C::Value>,
        _input_extents: &[Self::ShapeExtent],
    ) -> Result<ArrayBatch<C::Value>, BatchingError> {
        Self::match_axis(context, batch, 0.into())
    }

    fn reshape_collective(
        _context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
        value: C::Value,
        output_extents: &[Self::ShapeExtent],
        output_sharding: Option<Sharding>,
    ) -> Result<C::Value, BatchingError> {
        let output_shape = Shape::new(output_extents.iter().copied().map(Dimension::Static).collect());
        if value.r#type().shape() == &output_shape && value.r#type().sharding() == output_sharding.as_ref() {
            return Ok(value);
        }
        Ok(value.reshape_with_output_sharding(output_shape, output_sharding)?)
    }
}

impl<C> CollectiveArrayExtentBatchingPolicy<ProjectedContext<C, ArrayType>> for DynamicArrayExtentBatchingPolicy
where
    C: Context<
            Type = ArrayIrType,
            Operation: From<DynamicBroadcastOperation>
                           + From<ConstantOperation<DimensionValue>>
                           + From<DimensionSizeOperation>
                           + From<DynamicReshapeOperation>
                           + OperationProjection<ArrayType>,
        >,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: Assert
        + DimensionSize
        + DynamicBroadcast
        + ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>
        + ValueProjection<DimensionType>,
    <C::Value as ValueProjection<DimensionType>>::Projected:
        Compare<C::Value> + DimensionMax + Rem + Div + Mul + Value<Type = DimensionType>,
{
    type ShapeExtent = <C::Value as ValueProjection<DimensionType>>::Projected;

    fn collective_axis_extent(
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatchingPolicy<Self>>,
        _operation_name: &str,
        _axis_name: &str,
        axis_size: usize,
    ) -> Result<Self::ShapeExtent, BatchingError> {
        let axis_extent = ValueProjection::<DimensionType>::into_projected(context.axis_extent().clone())?;
        let axis_size = Self::collective_extent_constant(context, axis_size)?;
        axis_extent.compare(&axis_size, ComparisonDirection::Equal)?.assert(
            "collective axis extent must match the participant count",
            &[
                ("extent", ValueProjection::<DimensionType>::from_projected(axis_extent.clone())),
                ("participants", ValueProjection::<DimensionType>::from_projected(axis_size)),
            ],
        )?;
        Ok(axis_extent)
    }

    fn collective_extent_constant(
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatchingPolicy<Self>>,
        extent: usize,
    ) -> Result<Self::ShapeExtent, BatchingError> {
        let value = DimensionValue::constant(extent).map_err(ProgramError::from)?;
        let mut outputs = context.parent().parent().bind(ConstantOperation::new(value), Vec::new(), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(ValueProjection::<DimensionType>::into_projected(outputs.remove(0))?)
    }

    fn require_divisible_collective_extents(
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatchingPolicy<Self>>,
        left: &Self::ShapeExtent,
        right: &Self::ShapeExtent,
    ) -> Result<Self::ShapeExtent, BatchingError> {
        // Only what the extent types cannot prove is checked at runtime: two static extents are checked on the host,
        // and a divisor whose lower bound is positive needs neither a positivity assertion nor a clamp.
        if let (Some(left_extent), Some(right_extent)) = (left.r#type().extent(), right.r#type().extent()) {
            if right_extent == 0 || left_extent % right_extent != 0 {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!("extent {left_extent} must be divisible by extent {right_extent}"),
                });
            }
            return Ok(right.clone());
        }
        let zero = Self::collective_extent_constant(context, 0)?;
        let divisor = if right.r#type().bounds().lower() > 0 {
            right.clone()
        } else {
            let one = Self::collective_extent_constant(context, 1)?;
            right.compare(&zero, ComparisonDirection::GreaterThan)?.assert(
                "collective divisor must be positive",
                &[("divisor", ValueProjection::<DimensionType>::from_projected(right.clone()))],
            )?;
            right.dimension_max(&one)?
        };
        left.rem(&divisor)?.compare(&zero, ComparisonDirection::Equal)?.assert(
            "collective extent must be divisible by the participant count",
            &[
                ("extent", ValueProjection::<DimensionType>::from_projected(left.clone())),
                ("divisor", ValueProjection::<DimensionType>::from_projected(right.clone())),
            ],
        )?;
        Ok(divisor)
    }

    fn match_collective_axis(
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatchingPolicy<Self>>,
        batch: &ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>,
        input_extents: &[Self::ShapeExtent],
    ) -> Result<ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>, BatchingError> {
        if !batch.batch_axis().is_replicated() {
            return batch.move_axis(0);
        }

        let input_type = batch.unbatched_type();
        let input_extent_dimensions =
            input_extents.iter().map(|extent| extent.r#type().to_dimension()).collect::<Vec<_>>();
        let value = if input_type.shape().dimensions() == input_extent_dimensions {
            batch.value().clone()
        } else {
            Self::reshape_collective(context, batch.value().clone(), input_extents, input_type.sharding().cloned())?
        };
        let output_axes = (1..=input_type.rank()).collect::<Vec<_>>();
        let output_sharding = input_type
            .sharding()
            .map(|sharding| sharding.batched(0, context.axis_sharding().clone()))
            .transpose()?;
        let mut output_extents = Vec::with_capacity(input_extents.len() + 1);
        output_extents.push(context.axis_extent().clone());
        output_extents
            .extend(input_extents.iter().cloned().map(<C::Value as ValueProjection<DimensionType>>::from_projected));
        let value = <C::Value as ValueProjection<ArrayType>>::from_projected(value)
            .dynamic_broadcast_with_output_sharding(&output_extents, &output_axes, output_sharding)?;
        ArrayBatch::new(<C::Value as ValueProjection<ArrayType>>::into_projected(value)?, BatchAxis::from_position(0))
    }

    fn reshape_collective(
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatchingPolicy<Self>>,
        value: <C::Value as ValueProjection<ArrayType>>::Projected,
        output_extents: &[Self::ShapeExtent],
        output_sharding: Option<Sharding>,
    ) -> Result<<C::Value as ValueProjection<ArrayType>>::Projected, BatchingError> {
        let operation = DynamicReshapeOperation::new().with_output_sharding(output_sharding);
        let inputs = std::iter::once(<C::Value as ValueProjection<ArrayType>>::from_projected(value))
            .chain(output_extents.iter().cloned().map(<C::Value as ValueProjection<DimensionType>>::from_projected))
            .collect::<Vec<_>>();
        let mut outputs = context.parent().parent().bind(operation, Vec::new(), inputs.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(<C::Value as ValueProjection<ArrayType>>::into_projected(outputs.remove(0))?)
    }
}

/// Forwards a linear collective over an axis that the active batching level does not bind to the parent context. An
/// input without a mapped batch axis forwards the collective unchanged. A mapped input instead forwards the collective
/// that [`LinearCollectiveOperation::adapt_to_batch_axis`] returns for the input's mapped axis position, because the
/// collective's own axes shift around the mapped axis, together with the position of the mapped axis in the forwarded
/// result.
fn forward_linear_collective<C, P, O>(
    context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
    operation: &O,
    inputs: &[ArrayBatch<C::Value>],
) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError>
where
    C: Context<Type = ArrayType>,
    C::Operation: From<O>,
    P: ArrayExtentBatchingPolicy<C>,
    O: LinearCollectiveOperation,
{
    let [input] = inputs else {
        return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() }.into());
    };
    let Some(batch_axis) = input.batch_axis_position() else {
        return Ok(context.forward_to_parent(C::Operation::from(operation.clone()), inputs)?.into());
    };
    let (operation, output_batch_axis) = operation.adapt_to_batch_axis(batch_axis);
    let mut outputs =
        context
            .parent()
            .bind(C::Operation::from(operation), Vec::new(), std::slice::from_ref(input.value()))?;
    check_count!("output", outputs, 1, ProgramError);
    Ok(vec![ArrayBatch::new(outputs.remove(0), BatchAxis::from_position(output_batch_axis))?].into())
}

/// Infers the output type of a shape-changing collective for the logical (i.e., unbatched) `input_type` of a level
/// that binds its axis, and returns that type together with its extents in the batching policy's representation,
/// which the collective's matching-axis kernel consumes.
fn collective_output_extents<C, P, O>(
    context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
    operation: &O,
    input_type: &ArrayType,
) -> Result<(ArrayType, Vec<P::ShapeExtent>), BatchingError>
where
    C: Context<Type = ArrayType>,
    P: CollectiveArrayExtentBatchingPolicy<C>,
    O: Operation<Type = ArrayType>,
{
    let mut output_types = operation.infer_output_types(std::slice::from_ref(input_type), &[])?;
    let output_type = output_types.remove(0);
    let output_extents = output_type
        .shape()
        .dimensions()
        .iter()
        .map(|dimension| P::collective_extent_from_dimension(context, dimension))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((output_type, output_extents))
}

/// Result extent of a shape-changing collective, which the collective capabilities assemble before they stage the
/// collective with one explicit extent per output axis. Statically known extents stay on the host until
/// [`CollectiveExtent::stage`] stages them as constants, so that an input extent that a collective consumes or replaces
/// stages no instruction, and only runtime extents stage dimension arithmetic and runtime assertions. A static extent
/// that does not fit a collective is rejected by the type inference of the staged collective instead, with an
/// operation-specific diagnostic.
#[derive(Clone)]
enum CollectiveExtent<V> {
    /// Statically known extent.
    Static(usize),

    /// Runtime extent, observed as a first-class dimension value.
    Dynamic(V),
}

impl<V> CollectiveExtent<V> {
    /// Returns the result extent of a tiled collective that multiplies this extent by the effective participant count
    /// `effective_axis_size`.
    fn multiplied(&self, context: &V::DispatchDomain, effective_axis_size: usize) -> Result<Self, ProgramError>
    where
        V: Value<Type = ArrayIrType> + ValueProjection<DimensionType>,
        V::DispatchDomain: DimensionConstant<Value = V>,
        <V as ValueProjection<DimensionType>>::Projected: Mul,
    {
        match self {
            Self::Static(extent) => {
                let output_extent = extent.checked_mul(effective_axis_size).ok_or_else(|| {
                    TypeError::invalid(format!(
                        "collective result extent {extent} times {effective_axis_size} does not fit in usize",
                    ))
                })?;
                Ok(Self::Static(output_extent))
            }
            Self::Dynamic(extent) => {
                let extent = <V as ValueProjection<DimensionType>>::into_projected(extent.clone())?;
                let effective_axis_size = context.dimension_constant(effective_axis_size)?;
                let effective_axis_size = <V as ValueProjection<DimensionType>>::into_projected(effective_axis_size)?;
                Ok(Self::Dynamic(<V as ValueProjection<DimensionType>>::from_projected(
                    extent.mul(&effective_axis_size)?,
                )))
            }
        }
    }

    /// Returns the result extent of a tiled collective that divides this extent by the effective participant count
    /// `effective_axis_size`, first asserting at runtime that a runtime extent is exactly divisible by it. The count
    /// is a host value that [`CollectiveOptions::effective_axis_size`] has already validated to be positive, so the
    /// division needs no guard.
    fn divided(&self, context: &V::DispatchDomain, effective_axis_size: usize) -> Result<Self, ProgramError>
    where
        V: Value<Type = ArrayIrType> + Assert + ValueProjection<DimensionType>,
        V::DispatchDomain: DimensionConstant<Value = V>,
        <V as ValueProjection<DimensionType>>::Projected: Value<Type = DimensionType> + Compare<V> + Rem + Div,
    {
        match self {
            Self::Static(extent) => Ok(Self::Static(extent / effective_axis_size)),
            Self::Dynamic(extent) => {
                let extent = <V as ValueProjection<DimensionType>>::into_projected(extent.clone())?;
                let effective_axis_size = context.dimension_constant(effective_axis_size)?;
                let effective_axis_size = <V as ValueProjection<DimensionType>>::into_projected(effective_axis_size)?;
                Self::assert_divisible(context, &extent, &effective_axis_size)?;
                Ok(Self::Dynamic(ValueProjection::<DimensionType>::from_projected(extent.div(&effective_axis_size)?)))
            }
        }
    }

    /// Requires this extent to equal the effective participant count `effective_axis_size` of an untiled collective,
    /// staging a runtime assertion for a runtime extent.
    fn require_equal(&self, context: &V::DispatchDomain, effective_axis_size: usize) -> Result<(), ProgramError>
    where
        V: Value<Type = ArrayIrType> + Assert + ValueProjection<DimensionType>,
        V::DispatchDomain: DimensionConstant<Value = V>,
        <V as ValueProjection<DimensionType>>::Projected: Value<Type = DimensionType> + Compare<V>,
    {
        let Self::Dynamic(extent) = self else {
            return Ok(());
        };
        let extent = <V as ValueProjection<DimensionType>>::into_projected(extent.clone())?;
        let effective_axis_size = context.dimension_constant(effective_axis_size)?;
        let effective_axis_size = <V as ValueProjection<DimensionType>>::into_projected(effective_axis_size)?;
        extent.compare(&effective_axis_size, ComparisonDirection::Equal)?.assert(
            "collective axis extent must match the participant count",
            &[
                ("extent", ValueProjection::<DimensionType>::from_projected(extent)),
                ("participants", ValueProjection::<DimensionType>::from_projected(effective_axis_size)),
            ],
        )
    }

    /// Requires this extent to be exactly divisible by the effective participant count `effective_axis_size`, staging
    /// a runtime assertion for a runtime extent.
    fn require_divisible(&self, context: &V::DispatchDomain, effective_axis_size: usize) -> Result<(), ProgramError>
    where
        V: Value<Type = ArrayIrType> + Assert + ValueProjection<DimensionType>,
        V::DispatchDomain: DimensionConstant<Value = V>,
        <V as ValueProjection<DimensionType>>::Projected: Value<Type = DimensionType> + Compare<V> + Rem,
    {
        let Self::Dynamic(extent) = self else {
            return Ok(());
        };
        let extent = <V as ValueProjection<DimensionType>>::into_projected(extent.clone())?;
        let effective_axis_size = context.dimension_constant(effective_axis_size)?;
        let effective_axis_size = <V as ValueProjection<DimensionType>>::into_projected(effective_axis_size)?;
        Self::assert_divisible(context, &extent, &effective_axis_size)
    }

    /// Returns this extent as a first-class dimension value, staging a constant for a statically known extent.
    fn stage(self, context: &V::DispatchDomain) -> Result<V, ProgramError>
    where
        V: Value<Type = ArrayIrType>,
        V::DispatchDomain: DimensionConstant<Value = V>,
    {
        match self {
            Self::Static(extent) => context.dimension_constant(extent),
            Self::Dynamic(extent) => Ok(extent),
        }
    }

    /// Stages the runtime assertion that `extent` is exactly divisible by the positive `divisor`.
    fn assert_divisible(
        context: &V::DispatchDomain,
        extent: &<V as ValueProjection<DimensionType>>::Projected,
        divisor: &<V as ValueProjection<DimensionType>>::Projected,
    ) -> Result<(), ProgramError>
    where
        V: Value<Type = ArrayIrType> + Assert + ValueProjection<DimensionType>,
        V::DispatchDomain: DimensionConstant<Value = V>,
        <V as ValueProjection<DimensionType>>::Projected: Value<Type = DimensionType> + Compare<V> + Rem,
    {
        let zero = ValueProjection::<DimensionType>::into_projected(context.dimension_constant(0)?)?;
        extent.rem(divisor)?.compare(&zero, ComparisonDirection::Equal)?.assert(
            "collective extent must be divisible by the participant count",
            &[
                ("extent", ValueProjection::<DimensionType>::from_projected(extent.clone())),
                ("divisor", ValueProjection::<DimensionType>::from_projected(divisor.clone())),
            ],
        )
    }
}

/// Returns one [`CollectiveExtent`] for every axis of the array `value`: the static extent of a static axis, and an
/// explicit [`DimensionSize`] observation of a dynamic one.
fn collective_input_extents<V>(value: &V) -> Result<Vec<CollectiveExtent<V>>, ProgramError>
where
    V: Value<Type = ArrayIrType> + DimensionSize<V>,
{
    let r#type = value.r#type();
    let input_type = <&ArrayType>::try_from(r#type.as_ref())?;
    input_type
        .shape()
        .dimensions()
        .iter()
        .enumerate()
        .map(|(axis, dimension)| match dimension {
            Dimension::Static(extent) => Ok(CollectiveExtent::Static(*extent)),
            Dimension::Dynamic(_) => Ok(CollectiveExtent::Dynamic(value.dimension_size(axis)?)),
        })
        .collect()
}

impl<C: Context<Type = ArrayIrType>> BatchingContext<C, ArrayIrBatchingPolicy> {
    /// Binds an array IR linear collective over a non-matching named axis in the parent context. A replicated array
    /// requires no lifting, so the collective is forwarded unchanged and its result stays replicated. A mapped array
    /// instead forwards the collective that [`LinearCollectiveOperation::adapt_to_batch_axis`] returns for the array's
    /// mapped axis position, inserts this context's batch extent into the result extent inputs at the mapped axis
    /// position of the result, and marks the result mapped at that position.
    ///
    /// Unlike homogeneous array forwarding, the input dimension values describe one array result and are not
    /// separate result-producing inputs. The collective may also move the mapped axis when it changes the rank.
    ///
    /// # Parameters
    ///
    ///   - `operation`: Collective over the logical (i.e., unbatched) array.
    ///   - `array`: Array input whose ragged axes have already been rejected by the caller.
    ///   - `output_extents`: Validated replicated dimension inputs describing the per-item result shape.
    ///
    /// # Errors
    ///
    /// Returns a [`BatchingError`] if the parent cannot bind the collective or a result cannot carry its batch axis.
    fn forward_collective<O: LinearCollectiveOperation>(
        &self,
        operation: &O,
        array: &ArrayIrBatch<C::Value>,
        output_extents: &[ArrayIrBatch<C::Value>],
    ) -> Result<Vec<ArrayIrBatch<C::Value>>, BatchingError>
    where
        C::Operation: From<O>,
    {
        let (operation, output_batch_axis) = match array.batch_axis_position() {
            None => (operation.clone(), None),
            Some(batch_axis) => {
                let (operation, output_batch_axis) = operation.adapt_to_batch_axis(batch_axis);
                (operation, Some(output_batch_axis))
            }
        };
        let mut physical_output_extents =
            output_extents.iter().map(|extent| extent.value().clone()).collect::<Vec<_>>();
        if let Some(output_batch_axis) = output_batch_axis {
            physical_output_extents.insert(output_batch_axis, self.axis_extent().clone());
        }
        let physical_inputs = std::iter::once(array.value().clone()).chain(physical_output_extents).collect::<Vec<_>>();
        self.parent()
            .bind(operation, Vec::new(), physical_inputs.as_slice())?
            .into_iter()
            .map(|output| match output_batch_axis {
                Some(output_batch_axis) => ArrayIrBatch::new(output, BatchAxis::from_position(output_batch_axis)),
                None => Ok(ArrayIrBatch::replicated(output)),
            })
            .collect()
    }

    /// Returns this `batch` level re-expressed over the array projection of its parent with the dynamic extent
    /// batching policy, keeping its axis name, extent, and sharding, so that the matching-axis kernels of the
    /// shape-changing collectives, which operate on homogeneous arrays, can consume this level's mapped axis while
    /// reading their reshape geometry from the explicit dimension values of composite programs.
    fn array_projection(
        &self,
    ) -> BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>
    where
        C: DomainProjection<ArrayType>,
    {
        BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
            ProjectedContext::new(self.parent().clone()),
            self.axis_extent().clone(),
        )
        .with_axis_name(self.axis_name().map(str::to_string))
        .with_axis_sharding(self.axis_sharding().clone())
    }
}

/// Group of the value-level collective capabilities [`ParallelReduce`], [`ParallelVary`], [`ParallelAllGather`],
/// [`ParallelSumScatter`], [`ParallelPermute`], [`ParallelAllToAll`], and [`ParallelRaggedAllToAll`]. It is implemented
/// automatically for every type that implements all of its members. The group is parameterized by the [`Type`] universe
/// `T` of its values because several of its members are. The context-side [`AxisIndex`] is implemented by contexts
/// rather than values and is therefore not a member.
pub trait CollectiveOperations<T: Type>:
    ParallelReduce
    + ParallelVary
    + ParallelAllGather<T>
    + ParallelSumScatter<T>
    + ParallelPermute<T>
    + ParallelAllToAll<T>
    + ParallelRaggedAllToAll<T>
{
}

impl<
    T: Type,
    V: ParallelReduce
        + ParallelVary
        + ParallelAllGather<T>
        + ParallelSumScatter<T>
        + ParallelPermute<T>
        + ParallelAllToAll<T>
        + ParallelRaggedAllToAll<T>,
> CollectiveOperations<T> for V
{
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, DataType, Dimension, DimensionBounds, DimensionVariable,
        Layout, Memory, Shape, StridedLayout,
    };
    use crate::batching::{BatchableOperation, BatchingPolicy, BatchingTracer};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::MemberDifferentiableOperation;
    use crate::macros::check_operation_partial_evaluation;
    use crate::operations::collectives::parallel_all_gather::{
        ParallelAllGatherOperation, ParallelAllGatherOutputVariance,
    };
    use crate::operations::collectives::parallel_all_to_all::ParallelAllToAllOperation;
    use crate::parameters::Placeholder;
    use crate::programs::{EmptyRegionDriver, MemberOperation, ProgramBuilder};
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_collective_mode_forwarded_split_axes() {
        for mode in [CollectiveMode::Untiled, CollectiveMode::Tiled] {
            for rank in 1..=4 {
                for batch_axis in 0..=rank {
                    // Original axes retain their labels; `None` identifies the mapped batch axis.
                    let mut input_axes = (0..rank).map(Some).collect::<Vec<_>>();
                    input_axes.insert(batch_axis, None);
                    for split_axis in 0..rank {
                        let physical_split_axis = input_axes.iter().position(|axis| *axis == Some(split_axis)).unwrap();
                        let mut output_axes = input_axes.clone();
                        if mode == CollectiveMode::Untiled {
                            output_axes.remove(physical_split_axis);
                        }
                        let output_batch_axis = output_axes.iter().position(Option::is_none).unwrap();
                        assert_eq!(
                            mode.forwarded_split_axes(split_axis, batch_axis),
                            (physical_split_axis, output_batch_axis),
                            "mode={mode:?}, rank={rank}, split_axis={split_axis}, batch_axis={batch_axis}",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_collective_mode_forwarded_concatenation_axes() {
        // Preserve the original all-gather regressions at and on either side of the mapped axis.
        for (mode, concatenation_axis, batch_axis, expected) in [
            (CollectiveMode::Tiled, 0, 0, (1, 0)),
            (CollectiveMode::Tiled, 0, 1, (0, 1)),
            (CollectiveMode::Untiled, 0, 0, (0, 1)),
            (CollectiveMode::Untiled, 1, 0, (2, 0)),
        ] {
            assert_eq!(mode.forwarded_concatenation_axes(concatenation_axis, batch_axis), expected);
        }

        for mode in [CollectiveMode::Untiled, CollectiveMode::Tiled] {
            for rank in 0..=4 {
                let concatenation_positions = if mode == CollectiveMode::Untiled { rank + 1 } else { rank };
                for batch_axis in 0..=rank {
                    for concatenation_axis in 0..concatenation_positions {
                        // Untiled concatenation introduces a new label before the batch label at an equal boundary.
                        let mut output_axes = Vec::new();
                        for axis in 0..=rank {
                            if mode == CollectiveMode::Untiled && axis == concatenation_axis {
                                output_axes.push(Some(rank));
                            }
                            if axis == batch_axis {
                                output_axes.push(None);
                            }
                            if axis < rank {
                                output_axes.push(Some(axis));
                            }
                        }
                        let concatenation_label =
                            if mode == CollectiveMode::Untiled { rank } else { concatenation_axis };
                        let physical_concatenation_axis =
                            output_axes.iter().position(|axis| *axis == Some(concatenation_label)).unwrap();
                        let output_batch_axis = output_axes.iter().position(Option::is_none).unwrap();
                        assert_eq!(
                            mode.forwarded_concatenation_axes(concatenation_axis, batch_axis),
                            (physical_concatenation_axis, output_batch_axis),
                            "mode={mode:?}, rank={rank}, concatenation_axis={concatenation_axis}, batch_axis={batch_axis}",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_collective_mode_forwarded_split_and_concatenation_axes() {
        // Preserve the original all-to-all regression triples while testing the shared mappings' composition.
        for (mode, split_axis, concatenation_axis, batch_axis, expected) in [
            (CollectiveMode::Tiled, 0, 1, 1, (0, 2, 1)),
            (CollectiveMode::Tiled, 1, 0, 0, (2, 1, 0)),
            (CollectiveMode::Untiled, 0, 0, 2, (0, 0, 2)),
            (CollectiveMode::Untiled, 1, 1, 0, (2, 2, 0)),
        ] {
            let (physical_split_axis, batch_axis) = mode.forwarded_split_axes(split_axis, batch_axis);
            let (physical_concatenation_axis, batch_axis) =
                mode.forwarded_concatenation_axes(concatenation_axis, batch_axis);
            assert_eq!((physical_split_axis, physical_concatenation_axis, batch_axis), expected);
        }

        for mode in [CollectiveMode::Untiled, CollectiveMode::Tiled] {
            for rank in 1..=4 {
                for batch_axis in 0..=rank {
                    let mut input_axes = (0..rank).map(Some).collect::<Vec<_>>();
                    input_axes.insert(batch_axis, None);
                    for split_axis in 0..rank {
                        let physical_split_axis = input_axes.iter().position(|axis| *axis == Some(split_axis)).unwrap();
                        let mut intermediate_axes = input_axes.clone();
                        if mode == CollectiveMode::Untiled {
                            intermediate_axes.remove(physical_split_axis);
                        }
                        let intermediate_batch_axis = intermediate_axes.iter().position(Option::is_none).unwrap();
                        let logical_axes = intermediate_axes.iter().filter_map(|axis| *axis).collect::<Vec<_>>();
                        let concatenation_positions =
                            if mode == CollectiveMode::Untiled { logical_axes.len() + 1 } else { logical_axes.len() };
                        for concatenation_axis in 0..concatenation_positions {
                            let mut output_axes = Vec::new();
                            for axis in 0..=logical_axes.len() {
                                if mode == CollectiveMode::Untiled && axis == concatenation_axis {
                                    output_axes.push(Some(rank));
                                }
                                if axis == intermediate_batch_axis {
                                    output_axes.push(None);
                                }
                                if let Some(label) = logical_axes.get(axis) {
                                    output_axes.push(Some(*label));
                                }
                            }
                            let concatenation_label =
                                if mode == CollectiveMode::Untiled { rank } else { logical_axes[concatenation_axis] };
                            let physical_concatenation_axis =
                                output_axes.iter().position(|axis| *axis == Some(concatenation_label)).unwrap();
                            let output_batch_axis = output_axes.iter().position(Option::is_none).unwrap();
                            let (actual_split_axis, actual_batch_axis) =
                                mode.forwarded_split_axes(split_axis, batch_axis);
                            let (actual_concatenation_axis, actual_batch_axis) =
                                mode.forwarded_concatenation_axes(concatenation_axis, actual_batch_axis);
                            assert_eq!(
                                (actual_split_axis, actual_concatenation_axis, actual_batch_axis),
                                (physical_split_axis, physical_concatenation_axis, output_batch_axis),
                                "mode={mode:?}, rank={rank}, split_axis={split_axis}, \
                                 concatenation_axis={concatenation_axis}, batch_axis={batch_axis}",
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_collective_options_validate_axis_index_groups() {
        let options = CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]);
        assert_eq!(options.mode(), CollectiveMode::Tiled);
        assert_eq!(options.axis_index_groups(), Some([vec![0, 2], vec![3, 1]].as_slice()));
        assert_eq!(options.effective_axis_size("parallel_all_gather", 4), Ok(2));

        assert_eq!(
            CollectiveOptions::default()
                .with_axis_index_groups(Vec::new())
                .effective_axis_size("parallel_all_gather", 4),
            Err(TypeError::invalid("`parallel_all_gather` axis index groups must not be empty")),
        );
        assert_eq!(
            CollectiveOptions::default()
                .with_axis_index_groups(vec![vec![0, 1], vec![2]])
                .effective_axis_size("parallel_all_gather", 3),
            Err(TypeError::invalid(
                "`parallel_all_gather` axis index group 1 has size 1 but every group must have size 2",
            )),
        );
        assert_eq!(
            CollectiveOptions::default()
                .with_axis_index_groups(vec![vec![0, 1], vec![1, 2]])
                .effective_axis_size("parallel_all_gather", 4),
            Err(TypeError::invalid("`parallel_all_gather` axis index groups contain participant 1 more than once",)),
        );
        assert_eq!(
            CollectiveOptions::default()
                .with_axis_index_groups(vec![vec![0, 1], vec![2, 4]])
                .effective_axis_size("parallel_all_gather", 4),
            Err(TypeError::invalid("`parallel_all_gather` axis index 4 is out of bounds for axis size 4")),
        );
        assert_eq!(
            CollectiveOptions::default()
                .with_axis_index_groups(vec![vec![0, 1]])
                .effective_axis_size("parallel_all_gather", 3),
            Err(TypeError::invalid("`parallel_all_gather` axis index groups do not contain participant 2")),
        );
    }

    #[test]
    fn test_grouped_collective_shape_arithmetic_uses_group_size() {
        let grouped = CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]);
        let result_extent = DimensionValue::constant(6).unwrap().r#type().into_owned();
        assert_eq!(
            ParallelAllGatherOperation::new("x".to_string(), 4, 0, grouped, ParallelAllGatherOutputVariance::Varying)
                .infer_array_ir_output_types(&[ArrayType::new_static(DataType::F32, [3]).into(), result_extent.into()]),
            Ok(vec![ArrayType::new_static(DataType::F32, [6]).into()]),
        );
    }

    #[test]
    fn test_infer_linear_collective_operation_output_type() {
        let input_type = ArrayType::new_static(DataType::F32, [2, 3])
            .with_layout(Layout::Strided(StridedLayout::new(vec![12, 4])))
            .with_memory(Memory::Host { pinned: true });
        assert_eq!(
            infer_linear_collective_operation_output_type("parallel_all_gather", &input_type, vec![2, 3]),
            Ok(input_type.clone()),
        );
        assert_eq!(
            infer_linear_collective_operation_output_type("parallel_all_gather", &input_type, vec![4, 3]),
            Ok(ArrayType::new_static(DataType::F32, [4, 3]).with_memory(input_type.memory())),
        );
    }

    #[test]
    fn test_shape_changing_collective_kernel_capabilities() {
        // The batching rules of each shape-changing collective require only the capabilities of its own kernel, so a
        // context whose values can transpose but not reduce still batches all-gathers and all-to-alls. This compiles
        // only if those rules hold under exactly these bounds.
        fn requires_batching<C: Context, P: BatchingPolicy<C>, O: BatchableOperation<C, P>>() {}
        fn rearranging_collectives_batch<
            C: Context<
                    Type = ArrayType,
                    Value: Transpose,
                    Operation: From<ParallelAllGatherOperation> + From<ParallelAllToAllOperation>,
                >,
            P: CollectiveArrayExtentBatchingPolicy<C>,
        >() {
            requires_batching::<C, ArrayBatchingPolicy<P>, ParallelAllGatherOperation>();
            requires_batching::<C, ArrayBatchingPolicy<P>, ParallelAllToAllOperation>();
        }
        rearranging_collectives_batch::<EagerContext<Array, ArrayOperation<Array>>, StaticArrayExtentBatchingPolicy>();
    }

    #[test]
    fn test_linear_collective_operation_degenerate_interpretation() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let input = Array::vector(vec![1.0f32, 2.0]).unwrap();

        // Eager binding validates the shared participant count before it can return an identity value.
        assert_eq!(
            context.bind(
                ParallelAllToAllOperation::new("x".to_string(), 0, 0, 0, CollectiveOptions::tiled()),
                Vec::new(),
                std::slice::from_ref(&input),
            ),
            Err(ProgramError::Type(TypeError::invalid("`parallel_all_to_all` axis size must be greater than zero"))),
        );

        // The tiled identity rule must also honor the operation-specific axis validation.
        assert_eq!(
            context.bind(
                ParallelAllToAllOperation::new("x".to_string(), 1, 1, 0, CollectiveOptions::tiled()),
                Vec::new(),
                std::slice::from_ref(&input),
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "`parallel_all_to_all` split axis 1 or concat axis 0 is out of bounds for rank 1",
            ))),
        );
        assert_eq!(
            context.bind(
                ParallelAllToAllOperation::new("x".to_string(), 1, 0, 0, CollectiveOptions::tiled()),
                Vec::new(),
                std::slice::from_ref(&input),
            ),
            Ok(vec![input.clone()]),
        );

        // Outside any binder, only collectives whose instances each combine a single participant are defined. Groups
        // with one participant each qualify even over a larger axis, in both array families, while ungrouped
        // collectives over that axis do not.
        let singleton_groups = CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0], vec![1]]);
        assert_eq!(
            context.bind(
                ParallelAllToAllOperation::new("x".to_string(), 2, 0, 0, singleton_groups.clone()),
                Vec::new(),
                std::slice::from_ref(&input),
            ),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            context.bind(
                ParallelAllToAllOperation::new("x".to_string(), 2, 0, 0, CollectiveOptions::tiled()),
                Vec::new(),
                std::slice::from_ref(&input),
            ),
            Err(ProgramError::UnsupportedOperation {
                message: "cannot interpret `parallel_all_to_all` over axis `x` of size 2 without an enclosing binder"
                    .to_string(),
            }),
        );
        let composite_inputs =
            [ArrayIrValue::Array(input.clone()), ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap())];
        assert_eq!(
            ParallelSumScatterOperation::new("x".to_string(), 2, 0, singleton_groups)
                .shape_changing_collective_interpret::<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>(
                    &composite_inputs,
                ),
            Ok(vec![ArrayIrValue::Array(input)]),
        );
    }

    #[test]
    fn test_array_ir_shape_changing_collective_member_transforms() -> Result<(), ProgramError> {
        type Context = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        // A live tangent through a dynamically shaped mixed collective stages one residual-aware linear call directly
        // through the payload's member JVP rule.
        let variable = DimensionVariable::new("items", DimensionBounds::new(1, Some(9))?);
        let dimension_type = DimensionType::from(variable.clone());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)]));
        let context = Context::new();
        let primal = context.input(array_type.clone().into());
        let tangent = context.input(array_type.into());
        let extent = context.input(dimension_type.into());
        let extent_tangent_type = extent.r#type().tangent()?;
        let outputs = ParallelAllGatherOperation::new(
            "x".to_string(),
            1,
            0,
            CollectiveOptions::tiled(),
            ParallelAllGatherOutputVariance::Varying,
        )
        .jvp_in_parent(
            &DifferentiationContext::fused(context.clone()),
            &EmptyRegionDriver,
            &[
                DifferentiationDual::new(primal, MaybeZero::Value(tangent))?,
                DifferentiationDual::new(extent, MaybeZero::Zero(extent_tangent_type))?,
            ],
        )?;
        assert!(matches!(outputs[0].tangent(), MaybeZero::Value(_)));
        assert!(
            context
                .builder()
                .borrow()
                .instructions()
                .iter()
                .any(|instruction| matches!(instruction.operation(), ArrayIrOperation::LinearCall(_)))
        );

        Ok(())
    }

    #[test]
    fn test_untiled_collective_type_inference() {
        let shape = |dimensions| ArrayType::new(DataType::F32, Shape::new(dimensions));

        assert_eq!(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                4,
                1,
                CollectiveOptions::default(),
                ParallelAllGatherOutputVariance::Varying,
            )
            .infer_array_ir_output_types(&[
                shape(vec![Dimension::Static(2), Dimension::Static(3)]).into(),
                DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
                DimensionValue::constant(4).unwrap().r#type().into_owned().into(),
                DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
            ]),
            Ok(vec![shape(vec![Dimension::Static(2), Dimension::Static(4), Dimension::Static(3)]).into()]),
        );
        assert_eq!(
            ParallelAllToAllOperation::new("x".to_string(), 4, 1, 0, CollectiveOptions::default())
                .infer_array_ir_output_types(&[
                    shape(vec![Dimension::Static(2), Dimension::Static(4), Dimension::Static(3)]).into(),
                    DimensionValue::constant(4).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                ]),
            Ok(vec![shape(vec![Dimension::Static(4), Dimension::Static(2), Dimension::Static(3)]).into()]),
        );
        assert_eq!(
            ParallelAllToAllOperation::new("x".to_string(), 4, 1, 1, CollectiveOptions::default())
                .infer_array_ir_output_types(&[
                    shape(vec![Dimension::Static(2), Dimension::Static(4), Dimension::Static(3)]).into(),
                    DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(4).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                ]),
            Ok(vec![shape(vec![Dimension::Static(2), Dimension::Static(4), Dimension::Static(3)]).into()]),
        );
    }

    #[test]
    fn test_array_ir_shape_changing_collective_type_inference() {
        let input_axis = DimensionVariable::new("input", DimensionBounds::new(1, Some(17)).unwrap());
        let split_result = DimensionVariable::new("split", DimensionBounds::new(1, Some(9)).unwrap());
        let concat_result = DimensionVariable::new("concat", DimensionBounds::new(2, Some(33)).unwrap());
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(input_axis.clone()), Dimension::Static(3)]),
        );

        assert_eq!(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                2,
                0,
                CollectiveOptions::tiled(),
                ParallelAllGatherOutputVariance::Varying
            )
            .infer_array_ir_output_types(&[
                input_type.clone().into(),
                ArrayIrType::Dimension(DimensionType::from(concat_result.clone())),
                DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
            ]),
            Ok(vec![
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(concat_result.clone()), Dimension::Static(3)]),
                )
                .into()
            ]),
        );
        assert_eq!(
            ParallelAllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled())
                .infer_array_ir_output_types(&[
                    input_type.clone().into(),
                    ArrayIrType::Dimension(DimensionType::from(split_result.clone())),
                    ArrayIrType::Dimension(DimensionType::from(concat_result.clone())),
                ]),
            Ok(vec![
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(split_result), Dimension::Dynamic(concat_result),]),
                )
                .into()
            ]),
        );
        assert_eq!(
            ParallelAllToAllOperation::new("x".to_string(), 2, 0, 0, CollectiveOptions::tiled())
                .infer_array_ir_output_types(&[
                    ArrayIrType::Array(input_type.clone()),
                    ArrayIrType::Dimension(DimensionType::from(input_axis)),
                    DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                ]),
            Ok(vec![input_type.into()]),
        );

        let exact_six = DimensionValue::constant(6).unwrap().r#type().into_owned();
        assert_eq!(
            ParallelAllGatherOperation::new(
                "x".to_string(),
                2,
                0,
                CollectiveOptions::tiled(),
                ParallelAllGatherOutputVariance::Varying
            )
            .infer_array_ir_output_types(&[ArrayType::new_static(DataType::F32, [3]).into(), exact_six.into()]),
            Ok(vec![ArrayType::new_static(DataType::F32, [6]).into()]),
        );
        let exact_five = DimensionValue::constant(5).unwrap().r#type().into_owned();
        assert_eq!(
            ParallelAllGatherOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::tiled(),
                    ParallelAllGatherOutputVariance::Varying
                ).infer_array_ir_output_types(
                &[ArrayType::new_static(DataType::F32, [3]).into(), exact_five.into()]
            ),
            Err(TypeError::invalid(
                "`parallel_all_gather` result extent must equal input axis 0 extent 3 multiplied by axis group size 2; \
                 expected 6 \
                 but got 5"
                    .to_string(),
            )),
        );
    }

    #[test]
    fn test_array_ir_shape_changing_collective_type_inference_untiled() {
        let exact_two = DimensionValue::constant(2).unwrap().r#type().into_owned();
        let exact_three = DimensionValue::constant(3).unwrap().r#type().into_owned();
        let exact_four = DimensionValue::constant(4).unwrap().r#type().into_owned();
        let exact_five = DimensionValue::constant(5).unwrap().r#type().into_owned();

        // Inserting an axis preserves the extents already projected into the base output type on either side.
        let gather = ParallelAllGatherOperation::new(
            "x".to_string(),
            2,
            1,
            CollectiveOptions::default(),
            ParallelAllGatherOutputVariance::Varying,
        );
        assert_eq!(
            gather.infer_array_ir_output_types(&[
                ArrayType::new_static(DataType::F32, [3, 4]).into(),
                exact_three.clone().into(),
                exact_two.clone().into(),
                exact_four.clone().into(),
            ]),
            Ok(vec![ArrayType::new_static(DataType::F32, [3, 2, 4]).into()]),
        );
        assert_eq!(
            gather.infer_array_ir_output_types(&[
                ArrayType::new_static(DataType::F32, [3, 4]).into(),
                exact_three.clone().into(),
                exact_two.clone().into(),
                exact_five.into(),
            ]),
            Err(TypeError::invalid("`parallel_all_gather` output axis 2 extent 5 must equal unchanged extent 4")),
        );

        // Removing an axis or removing then inserting one needs no separate output-to-input axis mapping.
        assert_eq!(
            ParallelSumScatterOperation::new("x".to_string(), 2, 1, CollectiveOptions::default())
                .infer_parent_output_types(
                    &[
                        ArrayType::new_static(DataType::F32, [3, 2, 4]).into(),
                        exact_three.clone().into(),
                        exact_four.clone().into(),
                    ],
                    &[],
                ),
            Ok(vec![ArrayType::new_static(DataType::F32, [3, 4]).into()]),
        );
        assert_eq!(
            ParallelAllToAllOperation::new("x".to_string(), 2, 0, 2, CollectiveOptions::default())
                .infer_array_ir_output_types(&[
                    ArrayType::new_static(DataType::F32, [2, 3, 4]).into(),
                    exact_three.into(),
                    exact_four.into(),
                    exact_two.into(),
                ]),
            Ok(vec![ArrayType::new_static(DataType::F32, [3, 4, 2]).into()]),
        );
    }

    #[test]
    fn test_untiled_collectives_over_batched_axis_materialize_rank_changes() {
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2)
            .with_axis_name("x".to_string());
        let mapped_matrix =
            || ArrayBatch::new(Array::matrix(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap(), Some(0)).unwrap();

        let gathered = ParallelAllGatherOperation::new(
            "x".to_string(),
            2,
            1,
            CollectiveOptions::default(),
            ParallelAllGatherOutputVariance::Varying,
        )
        .batch(&context, &EmptyRegionDriver, &[mapped_matrix()])
        .unwrap()
        .into_parts()
        .0;
        assert_eq!(gathered[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(gathered[0].value(), &Array::matrix(2, 2, vec![1.0_f32, 3.0, 2.0, 4.0]).unwrap(),);

        let exchanged = ParallelAllToAllOperation::new("x".to_string(), 2, 0, 0, CollectiveOptions::default())
            .batch(&context, &EmptyRegionDriver, &[mapped_matrix()])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(exchanged[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(exchanged[0].value(), &Array::matrix(2, 2, vec![1.0_f32, 3.0, 2.0, 4.0]).unwrap(),);
    }

    #[test]
    fn test_shape_changing_collective_transposes_are_involutive() {
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;

        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder
            .add_input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4), Dimension::Static(3)])));
        let output = builder
            .add_instruction(
                ParallelAllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled()),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let transposed_twice =
            program.transpose_with_respect_to(&[0], &[]).unwrap().transpose_with_respect_to(&[0], &[]).unwrap();
        assert!(matches!(transposed_twice.instructions()[0].operation(), ArrayOperation::ParallelAllToAll(_)));
        assert_eq!(transposed_twice.input_types(), program.input_types());
        assert_eq!(transposed_twice.output_types(), program.output_types());
    }

    #[test]
    fn test_array_ir_collective_forwarding() -> Result<(), ProgramError> {
        // A forwarded untiled collective can move the mapped axis, while its replicated form must not acquire one.
        for (operation, output_shape, output_batch_axis) in [
            (
                ArrayIrOperation::<Array>::ParallelAllGather(ParallelAllGatherOperation::new(
                    "inner".to_string(),
                    2,
                    0,
                    CollectiveOptions::default(),
                    ParallelAllGatherOutputVariance::Varying,
                )),
                vec![2, 2, 3],
                2,
            ),
            (
                ArrayIrOperation::ParallelSumScatter(ParallelSumScatterOperation::new(
                    "inner".to_string(),
                    2,
                    0,
                    CollectiveOptions::default(),
                )),
                vec![3],
                0,
            ),
            (
                ArrayIrOperation::ParallelAllToAll(ParallelAllToAllOperation::new(
                    "inner".to_string(),
                    2,
                    0,
                    1,
                    CollectiveOptions::default(),
                )),
                vec![3, 2],
                0,
            ),
        ] {
            for mapped in [false, true] {
                let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
                let batch_extent = trace.input(DimensionValue::constant(5)?.r#type().into_owned().into());
                let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), batch_extent)
                    .with_axis_name("outer".to_string());
                let input_shape = if mapped { vec![2, 5, 3] } else { vec![2, 3] };
                let array = trace.input(ArrayType::new_static(DataType::F32, input_shape).into());
                let batch_axis = if mapped { BatchAxis::new(1) } else { BatchAxis::replicated() };
                let mut inputs = vec![BatchingTracer::new(context.clone(), ArrayIrBatch::new(array, batch_axis)?)];
                for extent in &output_shape {
                    let extent = trace.input(DimensionValue::constant(*extent)?.r#type().into_owned().into());
                    inputs.push(BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(extent)));
                }
                let outputs = context.bind(operation.clone(), Vec::new(), &inputs)?;
                assert_eq!(outputs.len(), 1);
                let output = outputs[0].batch();
                let mut physical_output_shape = output_shape.clone();
                let expected_batch_axis = if mapped {
                    physical_output_shape.insert(output_batch_axis, 5);
                    BatchAxis::new(output_batch_axis)
                } else {
                    BatchAxis::replicated()
                };
                assert_eq!(output.batch_axis(), expected_batch_axis);
                assert_eq!(
                    output.value().r#type().as_ref(),
                    &ArrayIrType::Array(ArrayType::new_static(DataType::F32, physical_output_shape)),
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_array_ir_collective_eager_contracts() {
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap());
        let extent = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());

        assert_eq!(
            context.bind(
                ParallelAllGatherOperation::new(
                    "x".to_string(),
                    1,
                    0,
                    CollectiveOptions::tiled(),
                    ParallelAllGatherOutputVariance::Varying
                ),
                Vec::new(),
                &[input.clone(), extent.clone()],
            ),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            context.bind(
                ParallelAllToAllOperation::new("x".to_string(), 1, 0, 0, CollectiveOptions::tiled()),
                Vec::new(),
                &[input.clone(), extent.clone()],
            ),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            context.bind(
                ParallelAllToAllOperation::new("x".to_string(), 1, 0, 1, CollectiveOptions::tiled()),
                Vec::new(),
                &[
                    ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],).unwrap()),
                    ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
                    ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()),
                ],
            ),
            Ok(vec![ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],).unwrap())]),
        );

        assert_eq!(
            context
                .bind(
                    ParallelAllGatherOperation::new(
                        "x".to_string(),
                        1,
                        0,
                        CollectiveOptions::tiled(),
                        ParallelAllGatherOutputVariance::Varying
                    ),
                    Vec::new(),
                    &[input.clone(), ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap()),],
                )
                .unwrap_err()
                .to_string(),
            "`parallel_all_gather` output axis 0 extent must equal observed result extent 3 but got 4",
        );
        assert_eq!(
            context
                .bind(
                    ParallelAllGatherOperation::new(
                        "x".to_string(),
                        2,
                        0,
                        CollectiveOptions::tiled(),
                        ParallelAllGatherOutputVariance::Varying
                    ),
                    Vec::new(),
                    &[input.clone(), ArrayIrValue::Dimension(DimensionValue::constant(6).unwrap()),],
                )
                .unwrap_err(),
            ProgramError::UnsupportedOperation {
                message: "cannot interpret `parallel_all_gather` over axis `x` of size 2 without an enclosing binder"
                    .to_string(),
            },
        );

        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = ParallelAllGatherOperation::new("x".to_string(), 1, 0, CollectiveOptions::tiled(), ParallelAllGatherOutputVariance::Varying),
            cases = [
                {
                    inputs = [(@known, input.clone()), (@known, extent.clone())],
                    outputs = [(@known, input.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = input.r#type().into_owned(), replay = input.clone())),
                        (@known, extent.clone()),
                    ],
                    outputs = [(@residual, input.clone())],
                    residual_instructions = 1,
                },
            ],
        );

        let variable = DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap());
        let dimension_type = DimensionType::from(variable.clone());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = builder.add_input(array_type.into());
        let result_extent = builder.add_input(dimension_type.clone().into());
        let output = builder
            .add_instruction(
                ParallelAllGatherOperation::new(
                    "x".to_string(),
                    1,
                    0,
                    CollectiveOptions::tiled(),
                    ParallelAllGatherOutputVariance::Varying,
                ),
                Vec::new(),
                vec![array, result_extent],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let primal = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap());
        let tangent = ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]).unwrap());
        let result_extent = ArrayIrValue::Dimension(DimensionValue::new(dimension_type.clone(), 3).unwrap());
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.interpret(vec![primal.clone(), result_extent.clone(), tangent.clone(),]),
            Ok(vec![primal, tangent]),
        );
        assert!(
            jvp.instructions()
                .iter()
                .any(|instruction| { matches!(instruction.operation(), ArrayIrOperation::LinearCall(_)) })
        );
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap()), result_extent])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        let cotangent = ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]).unwrap());
        let mut pullback_inputs = vec![cotangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![cotangent]));
        let zero_extent = ArrayIrValue::Dimension(DimensionValue::new(dimension_type, 0).unwrap());
        let zero_array = || {
            ArrayIrValue::Array(
                Array::from_elements::<f32>(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(0)])), &[])
                    .unwrap(),
            )
        };
        let mut primal_outputs = linearization.primal().interpret(vec![zero_array(), zero_extent]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let zero_cotangent = zero_array();
        let mut pullback_inputs = vec![zero_cotangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![zero_cotangent]));
        assert!(matches!(
            program.transpose_with_respect_to(&[0], &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "direct `parallel_all_gather` transposition with runtime-dependent type metadata requires \
                    linearization so that the relevant primal information can be retained as residuals",
        ));
    }

    #[test]
    fn test_array_ir_shape_changing_collective_linearization() {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = builder.add_input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])).into());
        let extent = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()));
        let output = builder
            .add_instruction(
                ParallelAllToAllOperation::new("x".to_string(), 1, 0, 0, CollectiveOptions::tiled()),
                Vec::new(),
                vec![array, extent],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert!(linearization.tangent().to_string().contains("linear_call"));
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap());
        let mut primal_outputs = linearization.primal().interpret(vec![input]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let cotangent = ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]).unwrap());
        let mut pullback_inputs = vec![cotangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![cotangent]));
    }
}
