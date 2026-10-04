//! Contains the named-axis collective operations, which exchange or reduce values across a named axis, together with
//! their interpretation, partial-evaluation, batching, forward-mode differentiation, and transposition rules. These
//! are the analogues of [JAX's parallel operators](https://docs.jax.dev/en/latest/jax.lax.html#parallel-operators).
//!
//! This module owns the vocabulary that every collective shares (i.e., [`CollectiveMode`], [`CollectiveOptions`], and
//! named axis resolution), while each operation family lives in its own submodule: [`parallel_reduce`],
//! [`parallel_vary`], [`all_gather`], [`parallel_sum_scatter`], [`parallel_permute`], [`all_to_all`], and
//! [`ragged_all_to_all`]. The [`axis_index`] submodule holds the one named-axis operation that exchanges nothing; it
//! reads the current batch item's or device shard's position along the axis. This module also owns the shared machinery
//! of the single-input linear collectives ([`ParallelPermuteOperation`], [`AllGatherOperation`],
//! [`ParallelSumScatterOperation`], and [`AllToAllOperation`]). Each carries the referenced axis name and the
//! participant count resolved from the active [`NamedAxes`] environment, consumes one statically shaped array input,
//! and has only degenerate single-participant semantics outside a binder. Its tangent rides the same collective, and
//! its transpose is another collective over the same axis. The private `define_linear_collective_operation!` macro
//! generates their common operation structure and the private `impl_differentiable_linear_collective_operation!` macro
//! their differentiation rules, while shared functions support the generated code and hand-written rules.
//!
//! The collectives that resize an array axis ([`AllGatherOperation`], [`ParallelSumScatterOperation`], and
//! [`AllToAllOperation`]) share additional machinery. Their output shapes depend
//! on the participant count and whether the named axis is materialized as a new array axis or tiled into an existing
//! one, so they share:
//!
//!   - [`CollectiveArrayExtentBatchingPolicy`], the representation boundary that lets one batching kernel per
//!     collective handle both homogeneous arrays with static extents and composite array/dimension programs with
//!     first-class extents ([`RaggedAllToAllOperation`] reuses it as well),
//!   - the first-class extent arithmetic that computes and validates result extents at staging time, and
//!   - the explicit [`ArrayIrType`] boundary, where the result extents are passed as additional dimension inputs, with
//!     its type inference, interpretation, batching, and forward-mode differentiation rules.
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
    ArrayType, Dimension, DimensionType, DimensionValue, LinearResiduals, Shape, Sharding,
    StaticArrayExtentBatchingPolicy,
};
use crate::axes::{AxisError, NamedAxes};
use crate::batching::{BatchAxis, BatchedOutputs, BatchingContext, BatchingError};
use crate::contexts::{Context, ProjectedContext};
use crate::differentiation::{
    DifferentiableType, DifferentiationContext, DifferentiationDual, DifferentiationError, DifferentiationPolicy,
};
use crate::macros::check_count;
use crate::operations::arithmetic::{Div, Mul, Rem};
use crate::operations::assertions::Assert;
use crate::operations::comparisons::{Compare, ComparisonDirection};
use crate::operations::constants::constant::{ConstantOperation, DimensionConstant};
use crate::operations::differentiation::linear_call::LinearCallOperation;
use crate::operations::dimensions::dimension_max::DimensionMax;
use crate::operations::dimensions::dimension_size::{DimensionSize, DimensionSizeOperation};
use crate::operations::manipulation::broadcasting::{Broadcast, DynamicBroadcast, DynamicBroadcastOperation};
use crate::operations::manipulation::reshaping::{DynamicReshapeOperation, Reshape};
use crate::operations::manipulation::transposition::Transpose;
use crate::programs::{
    MaybeZero, Operation, OperationProjection, ProgramError, TypeError, Typed, Value, ValueProjection,
};

pub mod all_gather;
pub mod all_to_all;
pub mod axis_index;
pub mod parallel_permute;
pub mod parallel_reduce;
pub mod parallel_sum_scatter;
pub mod parallel_vary;
pub mod ragged_all_to_all;

pub use all_gather::{ALL_GATHER_OPERATION_NAME, AllGather, AllGatherOperation, AllGatherOutputVariance};
pub use all_to_all::{ALL_TO_ALL_OPERATION_NAME, AllToAll, AllToAllOperation, ParallelSwapAxes};
pub use axis_index::{AXIS_INDEX_OPERATION_NAME, AxisIndex, AxisIndexOperation};
pub use parallel_permute::{PARALLEL_PERMUTE_OPERATION_NAME, ParallelPermute, ParallelPermuteOperation};
pub use parallel_reduce::{PARALLEL_REDUCE_OPERATION_NAME, ParallelReduce, ParallelReduceOperation};
pub use parallel_sum_scatter::{PARALLEL_SUM_SCATTER_OPERATION_NAME, ParallelSumScatter, ParallelSumScatterOperation};
pub use parallel_vary::{ManualVariationAlignment, PARALLEL_VARY_OPERATION_NAME, ParallelVary, ParallelVaryOperation};
pub use ragged_all_to_all::{RAGGED_ALL_TO_ALL_OPERATION_NAME, RaggedAllToAll, RaggedAllToAllOperation};

/// Shape semantics of the collectives that resize an array axis (e.g., [`AllGatherOperation`],
/// [`ParallelSumScatterOperation`], and [`AllToAllOperation`]), which determine where the `n` participants of the named
/// axis appear in the shape of the result. In [`Untiled`](Self::Untiled) mode, the participants get an array dimension
/// of their own with extent `n`, which an all-gather inserts, a sum-scatter consumes, and an all-to-all moves, so the
/// rank changes. In [`Tiled`](Self::Tiled) mode, the participants are instead folded into an existing array dimension,
/// whose extent is multiplied or divided by `n`, so the rank is preserved. These are the analogues of the `tiled=False`
/// (the default) and `tiled=True` settings of JAX's
/// [`jax.lax.all_gather`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.all_gather.html),
/// [`jax.lax.psum_scatter`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.psum_scatter.html),
/// and [`jax.lax.all_to_all`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.all_to_all.html).
///
/// Both modes compute the same values and differ only in where the participant dimension lives: an untiled result keeps
/// it as a separate dimension, while a tiled result folds it, participant-major, into an existing one. For example,
/// with `n = 4` participants, an all-gather that concatenates along axis 0, a sum-scatter that scatters along axis 0,
/// and an all-to-all that splits axis 0 and concatenates along axis 1, the shapes are:
///
/// ```text
///   Collective             Participant Input   Untiled Output   Tiled Output
///   ------------------------------------------------------------------------
///   all_gather             f32[3, 5]           f32[4, 3, 5]     f32[12, 5]
///   parallel_sum_scatter   f32[4, 5]           f32[5]           f32[1, 5]
///   parallel_sum_scatter   f32[12, 5]          (invalid)        f32[3, 5]
///   all_to_all             f32[4, 6]           f32[6, 4]        f32[1, 24]
///   all_to_all             f32[8, 6]           (invalid)        f32[2, 24]
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
    /// its concatenation axis,.
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

/// Shared shape and grouping options for collective operations that resize an array axis (e.g., [`AllGatherOperation`],
/// [`ParallelSumScatterOperation`], and [`AllToAllOperation`]).
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

/// Validates the participant grouping of a collective over a named axis of size `axis_size` and returns its _effective
/// axis size_, which is the number of participants that each instance of the collective combines. Without `groups`,
/// every participant along the axis takes part in one collective, so the effective axis size is `axis_size` itself.
/// With `groups`, the axis is split into independent collectives, one per group, and the effective axis size is the
/// common group size. Callers use it wherever shapes or values depend on the participant count (e.g., the gathered
/// extent of an `all_gather` operation, the chunk extent of a `parallel_sum_scatter` operation, or the divisor of a
/// mean operation).
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
///   - `accepts_unreduced`: Whether the collective accepts array inputs with unreduced axes (e.g., a sum-scatter,
///     which completes the pending reduction as part of its exchange).
///   - `input_types`: Array input type followed by one explicit extent type per output axis.
///   - `base_output_type`: Output type whose shape is replaced by the explicit extents.
///   - `changed_output_axes`: Output axes whose extents may differ from `base_output_type`. Every other axis must
///     retain the extent already projected into `base_output_type` by the caller.
///   - `validate_exact_extents_fn`: Collective-specific validation of the explicit output extents.
fn infer_explicit_shape_changing_collective_output_type(
    operation_name: &'static str,
    accepts_unreduced: bool,
    input_types: &[ArrayIrType],
    base_output_type: ArrayType,
    changed_output_axes: &[usize],
    validate_exact_extents_fn: impl FnOnce(&[Dimension]) -> Result<(), TypeError>,
) -> Result<Vec<ArrayIrType>, TypeError> {
    check_count!("input", input_types, 1 + base_output_type.rank(), TypeError);

    let input_type = <&ArrayType>::try_from(&input_types[0])?;
    if !accepts_unreduced && !input_type.unreduced_axes().is_empty() {
        return Err(TypeError::invalid(format!("`{operation_name}` does not support unreduced inputs")));
    }

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

// TODO(eaplatanios): Review form here onwards.

/// Defines the structural implementations shared by the single-input linear collectives (e.g., `all_gather` and
/// `parallel_permute`). The generated base includes the operation struct, with its `new` constructor and its
/// `axis_name` and `axis_size` accessors, together with its [`Display`](std::fmt::Display), [`Operation`],
/// [`InterpretableOperation`](crate::InterpretableOperation), and
/// [`PartiallyEvaluatableOperation`](crate::PartiallyEvaluatableOperation) implementations:
///
///   - Type inference validates the shared input contract (a nonzero axis size and exactly one statically shaped input
///     that satisfies the requested array-type checks) and then delegates the payload-dependent output type.
///   - Interpretation outside any binder is defined only over a degenerate single-participant axis, where it is the
///     identity unless the invocation provides its own `interpret` rule. Any larger axis is an error, because the other
///     participants do not exist per item.
///   - Partial evaluation uses the default fold-or-residualize behavior of `Program::partially_evaluate`.
///
/// Batching rules and value-level capabilities are written next to each invocation, because every collective consumes
/// the mapped batch axis, and exposes its named axis to users, differently.
/// [`impl_differentiable_linear_collective_operation!`] generates the differentiation rules.
///
/// # Example
///
/// ```rust,ignore
/// /// Canonical operation name for [`AllToAllOperation`].
/// pub const ALL_TO_ALL_OPERATION_NAME: &str = "all_to_all";
///
/// define_linear_collective_operation!(
///     /// [`Operation`] that exchanges chunks between the participants along the named axis.
///     AllToAllOperation,
///     ALL_TO_ALL_OPERATION_NAME,
///     fields = {
///         /// Axis of the input that is split into one chunk per participant.
///         split_axis: usize,
///     },
///     check_array_types = [@no_unreduced],
///     infer_output_type = |operation, input_type, dimensions| {
///         all_to_all_output_type(operation, input_type, dimensions)
///     },
/// );
/// ```
///
/// # Parameters
///
///   - `$(#[$documentation])*`: Documentation attributes attached to the generated operation struct.
///   - `$operation`: Identifier of the generated operation struct (e.g., `AllToAllOperation`).
///   - `$name`: Identifier of an existing operation-name constant (e.g., `ALL_TO_ALL_OPERATION_NAME`).
///   - `fields = { ... }`: Documented payload fields that follow the shared `axis_name` and `axis_size` fields, in the
///     order in which the generated `new` function takes them and the operation renders them.
///   - `optional_fields = { ... }`: Optional documented payload fields whose declared types are wrapped in [`Option`].
///     The generated `new` function initializes them to [`None`], invocations provide their own builder and accessor
///     functions, and the operation renders each one through its [`Display`](std::fmt::Display) implementation only
///     when it is present (e.g., the manual mesh of a `parallel_permute` over a mesh axis).
///   - `check_array_types = [@selector, ...]`: Optional ordered list of [`check_types!`](crate::check_types) selectors
///     applied to the input type (e.g., `@no_unreduced` for collectives that cannot complete a pending cross-device
///     sum as part of their exchange).
///   - `infer_output_type`: Closure-like rule that returns the output type as a `Result<ArrayType, TypeError>`. It
///     binds the operation, the validated input type, and the input's static dimensions to the provided names. The
///     closure-like syntax only names these values; it does not create a runtime closure.
///   - `interpret<$context> where $bounds { |operation, input| ... }`: Optional closure-like rule that returns the
///     output of a degenerate single-participant collective as a `Result<C::Value, ProgramError>`, for collectives
///     whose single participant does not simply keep its value (e.g., an untargeted `parallel_permute` participant,
///     which receives zeros). Its `where` predicates (e.g., `C::Value: ZeroLike`) bound the generated
///     [`InterpretableOperation`](crate::InterpretableOperation) implementation. When it is omitted, the single
///     participant keeps its value.
macro_rules! define_linear_collective_operation {
    // This branch generates the default interpretation, under which the single participant of a degenerate axis keeps
    // its value, by forwarding an identity rule to the custom interpretation branch.
    (@interpret $operation:ident, $name:ident) => {
        define_linear_collective_operation!(
            @interpret $operation,
            $name,
            C,
            [C::Value: ::std::clone::Clone],
            _operation,
            input,
            { Ok::<_, $crate::ProgramError>(input.clone()) },
        );
    };

    // This branch generates the interpretation of a collective outside any binder from its degenerate-axis rule.
    (
        @interpret $operation:ident,
        $name:ident,
        $context:ident,
        [$($bounded:ty: $bound:path),+],
        $interpret_operation:ident,
        $interpret_input:ident,
        $interpret:block $(,)?
    ) => {
        impl<$context: $crate::Domain<Type = $crate::ArrayType>> $crate::InterpretableOperation<$context> for $operation
        where
            $($bounded: $bound),+
        {
            fn interpret<D: $crate::InterpretationDriver<$context>>(
                &self,
                _context: &$context,
                _driver: &D,
                inputs: &[<$context as $crate::Domain>::Value],
            ) -> Result<Vec<<$context as $crate::Domain>::Value>, $crate::ProgramError> {
                use $crate::{Operation as _, Typed as _};

                // Eager binding does not infer output types, so interpretation validates the shared input contract
                // and the operation payload before applying either degenerate-axis rule.
                $crate::check_count!("input", inputs, 1, ProgramError);
                // Outside any binder, only the degenerate single-participant axis has defined per-item semantics. Any
                // larger axis is an error because the other participants do not exist per item.
                if self.axis_size > 1 {
                    return Err($crate::ProgramError::UnsupportedOperation {
                        message: format!(
                            "cannot interpret `{}` over axis `{}` of size {} without an enclosing binder",
                            $name, self.axis_name, self.axis_size,
                        ),
                    });
                }
                let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
                self.infer_output_types(&input_types, &[])?;
                let $interpret_operation = self;
                let $interpret_input = &inputs[0];
                Ok(vec![$interpret?])
            }
        }
    };

    // This branch accepts the public form and generates the operation struct together with its base implementations.
    (
        $(#[$documentation:meta])*
        $operation:ident,
        $name:ident,
        fields = { $($(#[$field_documentation:meta])* $field:ident: $field_type:ty),* $(,)? },
        $(
            optional_fields = {
                $($(#[$optional_field_documentation:meta])* $optional_field:ident: $optional_field_type:ty),* $(,)?
            },
        )?
        $(check_array_types = [$(@$array_type_check:ident),* $(,)?],)?
        infer_output_type = |$operation_binding:ident, $input_type:ident, $dimensions:ident| $infer:block,
        $(
            interpret<$context:ident> where $($bounded:ty: $bound:path),+ {
                |$interpret_operation:ident, $interpret_input:ident| $interpret:block
            } $(,)?
        )?
    ) => {
        $(#[$documentation])*
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        pub struct $operation {
            /// Axis name referenced by this collective.
            axis_name: String,

            /// Number of participants along the named axis, resolved from the active
            /// [`NamedAxes`](crate::NamedAxes) environment when the operation is staged.
            axis_size: usize,

            $($(#[$field_documentation])* $field: $field_type,)*

            $($($(#[$optional_field_documentation])* $optional_field: Option<$optional_field_type>,)*)?
        }

        impl $operation {
            /// Creates a new operation over the named axis with the provided resolved axis size.
            #[inline]
            pub fn new(axis_name: String, axis_size: usize, $($field: $field_type),*) -> Self {
                Self { axis_name, axis_size, $($field,)* $($($optional_field: None,)*)? }
            }

            /// Returns the axis name referenced by this collective.
            #[inline]
            pub fn axis_name(&self) -> &str {
                &self.axis_name
            }

            /// Returns the number of participants along the named axis.
            #[inline]
            pub fn axis_size(&self) -> usize {
                self.axis_size
            }
        }

        impl ::std::fmt::Display for $operation {
            fn fmt(&self, formatter: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                $crate::Operation::render(self, formatter, 0)
            }
        }

        impl $crate::Operation for $operation {
            type Type = $crate::ArrayType;

            #[inline]
            fn name(&self) -> &'static str {
                $name
            }

            fn infer_output_types(
                &self,
                input_types: &[$crate::ArrayType],
                region_interfaces: &[$crate::RegionInterface<$crate::ArrayType>],
            ) -> Result<Vec<$crate::ArrayType>, $crate::TypeError> {
                $crate::check_count!("region", region_interfaces, 0, TypeError);
                // A zero-participant collective is rejected before any extent arithmetic divides by its size.
                if self.axis_size == 0 {
                    return Err($crate::TypeError::invalid(format!("`{}` axis size must be greater than zero", $name)));
                }
                // Every linear collective has exactly one statically shaped input.
                $crate::check_count!("input", input_types, 1, TypeError);
                $($($crate::check_types!(@$array_type_check, $name, input_types);)*)?
                let Some(shape) = input_types[0].static_shape() else {
                    return Err($crate::TypeError::invalid(format!(
                        "`{}` does not support dynamically shaped inputs",
                        $name,
                    )));
                };
                let $dimensions = shape.dimensions().to_vec();
                let $operation_binding = self;
                let $input_type = &input_types[0];
                Ok(vec![$infer?])
            }

            fn render(&self, formatter: &mut ::std::fmt::Formatter<'_>, indentation: usize) -> ::std::fmt::Result {
                $crate::OperationFormatter::new(formatter, indentation, $name)?.bracketed(|operation| {
                    operation.field("axis_name", format_args!("{:?}", self.axis_name))?;
                    operation.field("axis_size", &self.axis_size)?;
                    $(operation.field(stringify!($field), format_args!("{:?}", &self.$field))?;)*
                    $($(
                        if let Some(value) = &self.$optional_field {
                            operation.field(stringify!($optional_field), value)?;
                        }
                    )*)?
                    Ok(())
                })
            }
        }

        define_linear_collective_operation!(
            @interpret $operation,
            $name
            $(, $context, [$($bounded: $bound),+], $interpret_operation, $interpret_input, $interpret)?
        );

        // Partial evaluation defers to the default fold-or-residualize behavior of `Program::partially_evaluate`.
        impl<C: $crate::Context<Type = $crate::ArrayType>> $crate::PartiallyEvaluatableOperation<C> for $operation where
            C::Operation: From<$operation>
        {
        }
    };
}

/// Implements the forward-mode differentiation (i.e., Jacobian-Vector Product, or JVP) and primitive transposition
/// rules of a collective defined by [`define_linear_collective_operation!`]. Linear collectives need only declare their
/// adjoint collective, and the macro generates the rest:
///
///   - The JVP stages the same collective on the input tangent, because the collective is linear. A structural-zero
///     tangent stays symbolic, retyped to the output tangent type because the collective can change shapes.
///   - Transposition stages the adjoint collective on the output cotangent. A known input and a structural-zero output
///     cotangent contribute nothing, which leaves the input cotangent a structural zero.
///   - A private `adjoint` function returns the adjoint collective, so that other rules can stage it as well (e.g., the
///     explicit-extent forward-mode rules of the shape-changing collectives, which call it inside a linear call).
///
/// Reverse-mode differentiation needs no separate rule because it is derived by linearizing and then transposing the
/// staged tangent program. The closure-like syntax only names the operation and the adjoint type, which the generated
/// transposition bounds require; it does not allocate or dynamically dispatch a runtime closure. The body becomes the
/// body of the generated `adjoint` function, so it may use `?` or return an error early for configurations that have
/// no adjoint collective, and its final expression is the adjoint operation.
///
/// # Examples
///
/// The transpose of a permutation is the permutation with every pair inverted:
///
/// ```rust,ignore
/// impl_differentiable_linear_collective_operation! {
///     ParallelPermuteOperation,
///     transpose = |operation| -> ParallelPermuteOperation {
///         let pairs = operation.source_target_pairs.iter().map(|(source, target)| (*target, *source)).collect();
///         ParallelPermuteOperation::new(operation.axis_name.clone(), operation.axis_size, pairs)
///     },
/// }
/// ```
///
/// # Parameters
///
///   - `$operation`: Linear collective type for which the rules are generated.
///   - `$operation_binding`: Name bound to the operation whose adjoint is being constructed.
///   - `$adjoint`: Type of the adjoint collective that the transposition rule stages.
///   - `$adjoint_body`: Block that evaluates to the adjoint collective, or returns a [`ProgramError`] early.
macro_rules! impl_differentiable_linear_collective_operation {
    // This branch generates the adjoint function together with the JVP and transposition rules of one collective.
    (
        $operation:ident,
        transpose = |$operation_binding:ident| -> $adjoint:ty $adjoint_body:block $(,)?
    ) => {
        impl $operation {
            /// Returns the adjoint collective that transposition stages on the output cotangent.
            fn adjoint(&self) -> Result<$adjoint, $crate::ProgramError> {
                let $operation_binding = self;
                Ok($adjoint_body)
            }
        }

        impl<C: $crate::Context<Type = $crate::ArrayType>> $crate::DifferentiableOperation<C> for $operation
        where
            C::Operation: From<$operation>,
        {
            fn jvp<D: $crate::DifferentiationDriver<C>, P: $crate::DifferentiationPolicy<C>>(
                &self,
                context: &$crate::DifferentiationContext<C, P>,
                _driver: &D,
                inputs: &[$crate::DifferentiationDual<C::Value>],
            ) -> Result<Vec<$crate::DifferentiationDual<C::Value>>, $crate::DifferentiationError> {
                use $crate::{DifferentiableType as _, Typed as _};

                $crate::check_count!("input", inputs, 1, ProgramError);
                let mut primals =
                    context.primal().bind(self.clone(), Vec::new(), std::slice::from_ref(inputs[0].primal()))?;
                $crate::check_count!("output", primals, 1, ProgramError);
                let primal = primals.remove(0);
                let tangent = match inputs[0].tangent() {
                    $crate::MaybeZero::Zero(_) => $crate::MaybeZero::Zero(primal.r#type().tangent()?),
                    $crate::MaybeZero::Value(tangent) => {
                        let mut tangents =
                            context.tangent().bind(self.clone(), Vec::new(), std::slice::from_ref(tangent))?;
                        $crate::check_count!("output", tangents, 1, ProgramError);
                        $crate::MaybeZero::Value(tangents.remove(0))
                    }
                };
                Ok(vec![$crate::DifferentiationDual::new(primal, tangent)?])
            }
        }

        impl<V, O> $crate::TransposableOperation<V, O> for $operation
        where
            V: $crate::Value<Type = $crate::ArrayType>,
            O: $crate::Operation<Type = $crate::ArrayType>
                + From<$crate::AddOperation<$crate::ArrayType>>
                + From<$adjoint>,
        {
            fn transpose<D: $crate::TranspositionDriver<V, O>>(
                &self,
                context: &mut $crate::TranspositionContext<V, O>,
                _driver: &D,
                inputs: &[$crate::PartialValue<$crate::Tracer<$crate::TracingContext<V, O>>>],
                outputs: &[$crate::MaybeZero<$crate::Tracer<$crate::TracingContext<V, O>>>],
                accumulators: &[$crate::CotangentAccumulator],
            ) -> Result<(), $crate::DifferentiationError> {
                use $crate::Context as _;

                $crate::check_count!("input", inputs, 1, ProgramError);
                $crate::check_count!("output", outputs, 1, ProgramError);
                $crate::check_count!("accumulator", accumulators, 1, DifferentiationError);
                // The adjoint is resolved first, so that a configuration without one is rejected regardless of the
                // cotangent, and only a live output cotangent of an unknown input then stages it.
                let adjoint = self.adjoint()?;
                let $crate::MaybeZero::Value(cotangent) = &outputs[0] else {
                    return Ok(());
                };
                if inputs[0].is_known() {
                    return Ok(());
                }
                let mut contributions = context.bind(O::from(adjoint), Vec::new(), std::slice::from_ref(cotangent))?;
                $crate::check_count!("output", contributions, 1, ProgramError);
                accumulators[0].accumulate(context, $crate::MaybeZero::Value(contributions.remove(0)))?;
                Ok(())
            }
        }
    };
}

use {define_linear_collective_operation, impl_differentiable_linear_collective_operation};

/// Representation boundary used only by shape-changing collective batching rules.
///
/// The collective kernels own every formula. This trait exposes only the extent representation and the alignment and
/// reshape encodings that differ between homogeneous arrays and composite array/dimension programs.
pub(crate) trait CollectiveArrayExtentBatchingPolicy<C: Context<Type = ArrayType>>:
    ArrayExtentBatchingPolicy<C>
{
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

/// Forwards a shape-changing collective over an axis that the active batching level does not bind to the parent
/// context. An input without a mapped batch axis forwards the collective unchanged. A mapped input instead forwards the
/// collective that `remap` returns for the input's mapped axis position, because the collective's own axes shift
/// around the mapped axis, and `remap` also returns the position of the mapped axis in the forwarded result.
fn forward_shape_changing_collective<C, P, O>(
    context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
    operation: &O,
    inputs: &[ArrayBatch<C::Value>],
    remap: impl FnOnce(usize) -> (O, usize),
) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError>
where
    C: Context<Type = ArrayType>,
    C::Operation: From<O>,
    P: ArrayExtentBatchingPolicy<C>,
    O: Clone,
{
    let [input] = inputs else {
        return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() }.into());
    };
    let Some(batch_axis) = input.batch_axis_position() else {
        return Ok(context.forward_to_parent(C::Operation::from(operation.clone()), inputs)?.into());
    };
    let (operation, output_batch_axis) = remap(batch_axis);
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

macro_rules! impl_shape_changing_collective_member_operation {
    // Implements the explicit array IR boundary shared by the three shape-changing collective payloads.
    ($operation:ty, $infer_output_types:ident) => {
        impl MemberOperation<ArrayIrType> for $operation {
            fn infer_parent_region_input_types(
                &self,
                _input_types: &[ArrayIrType],
                region_interfaces: &[RegionInterface<ArrayIrType>],
            ) -> Result<Vec<Option<Vec<ArrayIrType>>>, TypeError> {
                Ok(vec![None; region_interfaces.len()])
            }

            fn infer_parent_output_types(
                &self,
                input_types: &[ArrayIrType],
                region_interfaces: &[RegionInterface<ArrayIrType>],
            ) -> Result<Vec<ArrayIrType>, TypeError> {
                check_count!("region", region_interfaces, 0, TypeError);
                $infer_output_types(self, input_types)
            }

            fn rename_parent_type_identities(
                &self,
                renaming: &TypeIdentityRenaming<DimensionVariable>,
            ) -> Result<Self, TypeError> {
                self.rename_type_identities(renaming)
            }
        }

        impl<C> MemberInterpretableOperation<C> for $operation
        where
            C: Domain<Type = ArrayIrType>,
            C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType> + DimensionSize<usize> + Reshape>
                + ValueProjection<DimensionType, Projected = DimensionValue>,
        {
            fn interpret_in_parent<D: InterpretationDriver<C>>(
                &self,
                _context: &C,
                _driver: &D,
                inputs: &[C::Value],
            ) -> Result<Vec<C::Value>, ProgramError> {
                let Some((input, output_extents)) = inputs.split_first() else {
                    return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 });
                };
                let input = <C::Value as ValueProjection<ArrayType>>::into_projected(input.clone())?;
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
                                "`{}` output axis {axis} extent must equal observed result extent {expected} but got \
                                 {actual}",
                                self.name(),
                            ),
                        });
                    }
                }
                let effective_axis_size = self.effective_axis_size()?;
                if effective_axis_size != 1 {
                    return Err(ProgramError::UnsupportedOperation {
                        message: format!(
                            "cannot interpret `{}` over axis `{}` of size {} without an enclosing binder",
                            self.name(),
                            self.axis_name(),
                            effective_axis_size,
                        ),
                    });
                }
                let output = match self.options().mode() {
                    CollectiveMode::Tiled => input,
                    CollectiveMode::Untiled => input.reshape(Shape::from(expected_extents))?,
                };
                Ok(vec![<C::Value as ValueProjection<ArrayType>>::from_projected(output)])
            }
        }
    };
}

use impl_shape_changing_collective_member_operation;

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

/// Applies the mixed array IR JVP shared by shape-changing collectives whose transpose is another collective.
/// Explicit output extents and the exact input shape become ordinary residuals of one linear call.
fn jvp_shape_changing_collective_with_adjoint<C, Forward, Adjoint, P: DifferentiationPolicy<C>>(
    operation: &Forward,
    adjoint: Adjoint,
    context: &DifferentiationContext<C, P>,
    inputs: &[DifferentiationDual<C::Value>],
) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError>
where
    C: Context<Type = ArrayIrType>,
    C::Operation: From<Forward>
        + From<Adjoint>
        + From<DimensionSizeOperation>
        + From<LinearCallOperation<ArrayIrType>>
        + From<ConstantOperation<DimensionValue>>,
    Forward: Clone + Operation<Type = ArrayType>,
    Adjoint: Operation<Type = ArrayType>,
{
    let Some((array, _)) = inputs.split_first() else {
        return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
    };
    let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
    let primal = context.primal().bind(operation.clone(), Vec::new(), primal_inputs.as_slice())?.remove(0);
    let tangent = match array.tangent() {
        MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
        MaybeZero::Value(array_tangent) => {
            let tangent_inputs = context.dual_primal_to_tangent(inputs)?;
            let (array, output_extents) = tangent_inputs.split_first().unwrap();
            let context = context.tangent();
            let mut residuals = LinearResiduals::new();
            let output_extents = residuals.retain_all(output_extents.iter().map(|extent| extent.primal().clone()));
            let input_shape = residuals.retain_shape(context, array.primal())?;
            let forward_operation = operation.clone();
            let forward_output_extents = output_extents.clone();
            let tangent = LinearCallOperation::stage(
                context,
                residuals.into_values(),
                vec![array_tangent.clone()],
                move |residuals, linear_inputs| {
                    let mut collective_inputs = Vec::with_capacity(1 + forward_output_extents.len());
                    collective_inputs.push(linear_inputs[0].clone());
                    collective_inputs.extend(forward_output_extents.iter().map(|index| residuals[*index].clone()));
                    linear_inputs[0].dispatch_domain().bind(forward_operation, Vec::new(), collective_inputs.as_slice())
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

/// Splits a mixed collective's inputs into its validated array input and unchecked explicit result extents.
fn explicit_collective_inputs<'a, V: Value<Type = ArrayIrType>>(
    inputs: &'a [ArrayIrBatch<V>],
) -> Result<(&'a ArrayIrBatch<V>, &'a [ArrayIrBatch<V>]), BatchingError> {
    let Some((array, output_extents)) = inputs.split_first() else {
        return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
    };
    <&ArrayType>::try_from(&array.unbatched_type())?;
    Ok((array, output_extents))
}

/// Validates that every explicit result extent of a mixed collective is replicated.
fn validate_explicit_collective_output_extents<V: Value<Type = ArrayIrType>>(
    output_extents: &[ArrayIrBatch<V>],
) -> Result<(), BatchingError> {
    for output_extent in output_extents {
        output_extent.validate_replicated_dimension()?;
    }
    Ok(())
}

/// Binds a mixed collective over a non-matching named axis after lifting the mapped axis into its explicit result
/// extents. Replicated arrays require no lifting and remain replicated.
fn forward_explicit_collective<C, O>(
    operation: O,
    context: &BatchingContext<C, ArrayIrBatchingPolicy>,
    array: &ArrayIrBatch<C::Value>,
    output_extents: &[ArrayIrBatch<C::Value>],
    output_batch_axis: Option<usize>,
) -> Result<Vec<ArrayIrBatch<C::Value>>, BatchingError>
where
    C: Context<Type = ArrayIrType, Operation: From<O>>,
{
    let mut physical_output_extents = output_extents.iter().map(|extent| extent.value().clone()).collect::<Vec<_>>();
    if let Some(output_batch_axis) = output_batch_axis {
        physical_output_extents.insert(output_batch_axis, context.axis_extent().clone());
    }
    let physical_inputs = std::iter::once(array.value().clone()).chain(physical_output_extents).collect::<Vec<_>>();
    context
        .parent()
        .bind(operation, Vec::new(), physical_inputs.as_slice())?
        .into_iter()
        .map(|output| match output_batch_axis {
            Some(output_batch_axis) => ArrayIrBatch::new(output, BatchAxis::from_position(output_batch_axis)),
            None => Ok(ArrayIrBatch::replicated(output)),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, DataType, Dimension, DimensionBounds, DimensionVariable,
        Layout, Memory, Shape, StridedLayout,
    };
    use crate::batching::BatchableOperation;
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::MemberDifferentiableOperation;
    use crate::macros::check_operation_partial_evaluation;
    use crate::operations::collectives::all_gather::{
        AllGatherOperation, AllGatherOutputVariance, infer_explicit_all_gather_output_types,
    };
    use crate::operations::collectives::all_to_all::{AllToAllOperation, infer_explicit_all_to_all_output_types};
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
        assert_eq!(options.effective_axis_size("all_gather", 4), Ok(2));

        assert_eq!(
            CollectiveOptions::default().with_axis_index_groups(Vec::new()).effective_axis_size("all_gather", 4),
            Err(TypeError::invalid("`all_gather` axis index groups must not be empty")),
        );
        assert_eq!(
            CollectiveOptions::default()
                .with_axis_index_groups(vec![vec![0, 1], vec![2]])
                .effective_axis_size("all_gather", 3),
            Err(TypeError::invalid("`all_gather` axis index group 1 has size 1 but every group must have size 2",)),
        );
        assert_eq!(
            CollectiveOptions::default()
                .with_axis_index_groups(vec![vec![0, 1], vec![1, 2]])
                .effective_axis_size("all_gather", 4),
            Err(TypeError::invalid("`all_gather` axis index groups contain participant 1 more than once",)),
        );
        assert_eq!(
            CollectiveOptions::default()
                .with_axis_index_groups(vec![vec![0, 1], vec![2, 4]])
                .effective_axis_size("all_gather", 4),
            Err(TypeError::invalid("`all_gather` axis index 4 is out of bounds for axis size 4")),
        );
        assert_eq!(
            CollectiveOptions::default()
                .with_axis_index_groups(vec![vec![0, 1]])
                .effective_axis_size("all_gather", 3),
            Err(TypeError::invalid("`all_gather` axis index groups do not contain participant 2")),
        );
    }

    #[test]
    fn test_grouped_collective_shape_arithmetic_uses_group_size() {
        let grouped = CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]);
        let result_extent = DimensionValue::constant(6).unwrap().r#type().into_owned();
        assert_eq!(
            infer_explicit_all_gather_output_types(
                &AllGatherOperation::new("x".to_string(), 4, 0, grouped, AllGatherOutputVariance::Varying,),
                &[ArrayType::new_static(DataType::F32, [3]).into(), result_extent.into(),],
            ),
            Ok(vec![ArrayType::new_static(DataType::F32, [6]).into()]),
        );
    }

    #[test]
    fn test_infer_linear_collective_operation_output_type() {
        let input_type = ArrayType::new_static(DataType::F32, [2, 3])
            .with_layout(Layout::Strided(StridedLayout::new(vec![12, 4])))
            .with_memory(Memory::Host { pinned: true });
        assert_eq!(
            infer_linear_collective_operation_output_type("all_gather", &input_type, vec![2, 3]),
            Ok(input_type.clone()),
        );
        assert_eq!(
            infer_linear_collective_operation_output_type("all_gather", &input_type, vec![4, 3]),
            Ok(ArrayType::new_static(DataType::F32, [4, 3]).with_memory(input_type.memory())),
        );
    }

    #[test]
    fn test_define_linear_collective_operation_interpretation() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let input = Array::vector(vec![1.0f32, 2.0]).unwrap();

        // Eager binding validates the shared participant count before it can return an identity value.
        assert_eq!(
            context.bind(
                AllToAllOperation::new("x".to_string(), 0, 0, 0, CollectiveOptions::tiled()),
                Vec::new(),
                std::slice::from_ref(&input),
            ),
            Err(ProgramError::Type(TypeError::invalid("`all_to_all` axis size must be greater than zero"))),
        );

        // The custom tiled identity rule must also honor its operation-specific axis validation.
        assert_eq!(
            context.bind(
                AllToAllOperation::new("x".to_string(), 1, 1, 0, CollectiveOptions::tiled()),
                Vec::new(),
                std::slice::from_ref(&input),
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "`all_to_all` split axis 1 or concat axis 0 is out of bounds for rank 1",
            ))),
        );
        assert_eq!(
            context.bind(
                AllToAllOperation::new("x".to_string(), 1, 0, 0, CollectiveOptions::tiled()),
                Vec::new(),
                std::slice::from_ref(&input),
            ),
            Ok(vec![input]),
        );
    }

    #[test]
    fn test_explicit_shape_changing_collective_member_transforms() -> Result<(), ProgramError> {
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
        let outputs = AllGatherOperation::new(
            "x".to_string(),
            1,
            0,
            CollectiveOptions::tiled(),
            AllGatherOutputVariance::Varying,
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
            infer_explicit_all_gather_output_types(
                &AllGatherOperation::new(
                    "x".to_string(),
                    4,
                    1,
                    CollectiveOptions::default(),
                    AllGatherOutputVariance::Varying,
                ),
                &[
                    shape(vec![Dimension::Static(2), Dimension::Static(3)]).into(),
                    DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(4).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                ],
            ),
            Ok(vec![shape(vec![Dimension::Static(2), Dimension::Static(4), Dimension::Static(3)]).into()]),
        );
        assert_eq!(
            infer_explicit_all_to_all_output_types(
                &AllToAllOperation::new("x".to_string(), 4, 1, 0, CollectiveOptions::default()),
                &[
                    shape(vec![Dimension::Static(2), Dimension::Static(4), Dimension::Static(3)]).into(),
                    DimensionValue::constant(4).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                ],
            ),
            Ok(vec![shape(vec![Dimension::Static(4), Dimension::Static(2), Dimension::Static(3)]).into()]),
        );
        assert_eq!(
            infer_explicit_all_to_all_output_types(
                &AllToAllOperation::new("x".to_string(), 4, 1, 1, CollectiveOptions::default()),
                &[
                    shape(vec![Dimension::Static(2), Dimension::Static(4), Dimension::Static(3)]).into(),
                    DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(4).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                ],
            ),
            Ok(vec![shape(vec![Dimension::Static(2), Dimension::Static(4), Dimension::Static(3)]).into()]),
        );
    }

    #[test]
    fn test_explicit_shape_changing_collective_type_inference() {
        let input_axis = DimensionVariable::new("input", DimensionBounds::new(1, Some(17)).unwrap());
        let split_result = DimensionVariable::new("split", DimensionBounds::new(1, Some(9)).unwrap());
        let concat_result = DimensionVariable::new("concat", DimensionBounds::new(2, Some(33)).unwrap());
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(input_axis.clone()), Dimension::Static(3)]),
        );

        assert_eq!(
            infer_explicit_all_gather_output_types(
                &AllGatherOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::tiled(),
                    AllGatherOutputVariance::Varying
                ),
                &[
                    input_type.clone().into(),
                    ArrayIrType::Dimension(DimensionType::from(concat_result.clone())),
                    DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                ],
            ),
            Ok(vec![
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(concat_result.clone()), Dimension::Static(3)]),
                )
                .into()
            ]),
        );
        assert_eq!(
            infer_explicit_all_to_all_output_types(
                &AllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled()),
                &[
                    input_type.clone().into(),
                    ArrayIrType::Dimension(DimensionType::from(split_result.clone())),
                    ArrayIrType::Dimension(DimensionType::from(concat_result.clone())),
                ],
            ),
            Ok(vec![
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(split_result), Dimension::Dynamic(concat_result),]),
                )
                .into()
            ]),
        );
        assert_eq!(
            infer_explicit_all_to_all_output_types(
                &AllToAllOperation::new("x".to_string(), 2, 0, 0, CollectiveOptions::tiled()),
                &[
                    ArrayIrType::Array(input_type.clone()),
                    ArrayIrType::Dimension(DimensionType::from(input_axis)),
                    DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                ],
            ),
            Ok(vec![input_type.into()]),
        );

        let exact_six = DimensionValue::constant(6).unwrap().r#type().into_owned();
        assert_eq!(
            infer_explicit_all_gather_output_types(
                &AllGatherOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::tiled(),
                    AllGatherOutputVariance::Varying
                ),
                &[ArrayType::new_static(DataType::F32, [3]).into(), exact_six.into()],
            ),
            Ok(vec![ArrayType::new_static(DataType::F32, [6]).into()]),
        );
        let exact_five = DimensionValue::constant(5).unwrap().r#type().into_owned();
        assert_eq!(
            infer_explicit_all_gather_output_types(
                &AllGatherOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::tiled(),
                    AllGatherOutputVariance::Varying
                ),
                &[ArrayType::new_static(DataType::F32, [3]).into(), exact_five.into()],
            ),
            Err(TypeError::invalid(
                "`all_gather` result extent must equal input axis 0 extent 3 multiplied by axis group size 2; \
                 expected 6 \
                 but got 5"
                    .to_string(),
            )),
        );
    }

    #[test]
    fn test_explicit_shape_changing_collective_type_inference_untiled() {
        let exact_two = DimensionValue::constant(2).unwrap().r#type().into_owned();
        let exact_three = DimensionValue::constant(3).unwrap().r#type().into_owned();
        let exact_four = DimensionValue::constant(4).unwrap().r#type().into_owned();
        let exact_five = DimensionValue::constant(5).unwrap().r#type().into_owned();

        // Inserting an axis preserves the extents already projected into the base output type on either side.
        let gather = AllGatherOperation::new(
            "x".to_string(),
            2,
            1,
            CollectiveOptions::default(),
            AllGatherOutputVariance::Varying,
        );
        assert_eq!(
            infer_explicit_all_gather_output_types(
                &gather,
                &[
                    ArrayType::new_static(DataType::F32, [3, 4]).into(),
                    exact_three.clone().into(),
                    exact_two.clone().into(),
                    exact_four.clone().into(),
                ],
            ),
            Ok(vec![ArrayType::new_static(DataType::F32, [3, 2, 4]).into()]),
        );
        assert_eq!(
            infer_explicit_all_gather_output_types(
                &gather,
                &[
                    ArrayType::new_static(DataType::F32, [3, 4]).into(),
                    exact_three.clone().into(),
                    exact_two.clone().into(),
                    exact_five.into(),
                ],
            ),
            Err(TypeError::invalid("`all_gather` output axis 2 extent 5 must equal unchanged extent 4")),
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
            infer_explicit_all_to_all_output_types(
                &AllToAllOperation::new("x".to_string(), 2, 0, 2, CollectiveOptions::default()),
                &[
                    ArrayType::new_static(DataType::F32, [2, 3, 4]).into(),
                    exact_three.into(),
                    exact_four.into(),
                    exact_two.into(),
                ],
            ),
            Ok(vec![ArrayType::new_static(DataType::F32, [3, 4, 2]).into()]),
        );
    }

    #[test]
    fn test_untiled_collectives_over_batched_axis_materialize_rank_changes() {
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2)
            .with_axis_name("x".to_string());
        let mapped_matrix =
            || ArrayBatch::new(Array::matrix(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap(), Some(0)).unwrap();

        let gathered = AllGatherOperation::new(
            "x".to_string(),
            2,
            1,
            CollectiveOptions::default(),
            AllGatherOutputVariance::Varying,
        )
        .batch(&context, &EmptyRegionDriver, &[mapped_matrix()])
        .unwrap()
        .into_parts()
        .0;
        assert_eq!(gathered[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(gathered[0].value(), &Array::matrix(2, 2, vec![1.0_f32, 3.0, 2.0, 4.0]).unwrap(),);

        let exchanged = AllToAllOperation::new("x".to_string(), 2, 0, 0, CollectiveOptions::default())
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
                AllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled()),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let transposed_twice =
            program.transpose_with_respect_to(&[0], &[]).unwrap().transpose_with_respect_to(&[0], &[]).unwrap();
        assert!(matches!(transposed_twice.instructions()[0].operation(), ArrayOperation::AllToAll(_)));
        assert_eq!(transposed_twice.input_types(), program.input_types());
        assert_eq!(transposed_twice.output_types(), program.output_types());
    }

    #[test]
    fn test_array_ir_explicit_collective_eager_contracts() {
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap());
        let extent = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());

        assert_eq!(
            context.bind(
                AllGatherOperation::new(
                    "x".to_string(),
                    1,
                    0,
                    CollectiveOptions::tiled(),
                    AllGatherOutputVariance::Varying
                ),
                Vec::new(),
                &[input.clone(), extent.clone()],
            ),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            context.bind(
                AllToAllOperation::new("x".to_string(), 1, 0, 0, CollectiveOptions::tiled()),
                Vec::new(),
                &[input.clone(), extent.clone()],
            ),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            context.bind(
                AllToAllOperation::new("x".to_string(), 1, 0, 1, CollectiveOptions::tiled()),
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
                    AllGatherOperation::new(
                        "x".to_string(),
                        1,
                        0,
                        CollectiveOptions::tiled(),
                        AllGatherOutputVariance::Varying
                    ),
                    Vec::new(),
                    &[input.clone(), ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap()),],
                )
                .unwrap_err()
                .to_string(),
            "`all_gather` output axis 0 extent must equal observed result extent 3 but got 4",
        );
        assert_eq!(
            context
                .bind(
                    AllGatherOperation::new(
                        "x".to_string(),
                        2,
                        0,
                        CollectiveOptions::tiled(),
                        AllGatherOutputVariance::Varying
                    ),
                    Vec::new(),
                    &[input.clone(), ArrayIrValue::Dimension(DimensionValue::constant(6).unwrap()),],
                )
                .unwrap_err(),
            ProgramError::UnsupportedOperation {
                message: "cannot interpret `all_gather` over axis `x` of size 2 without an enclosing binder"
                    .to_string(),
            },
        );

        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = AllGatherOperation::new("x".to_string(), 1, 0, CollectiveOptions::tiled(), AllGatherOutputVariance::Varying),
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
                AllGatherOperation::new(
                    "x".to_string(),
                    1,
                    0,
                    CollectiveOptions::tiled(),
                    AllGatherOutputVariance::Varying,
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
                if message == "direct `all_gather` transposition with runtime-dependent type metadata requires \
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
                AllToAllOperation::new("x".to_string(), 1, 0, 0, CollectiveOptions::tiled()),
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
