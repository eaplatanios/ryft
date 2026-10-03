//! Contains the named-axis collective operations, which exchange or reduce values across a named axis, together with
//! their interpretation, partial-evaluation, batching, forward-mode differentiation, and transposition rules. These
//! are the analogues of [JAX's parallel operators](https://docs.jax.dev/en/latest/jax.lax.html#parallel-operators).
//!
//! This module owns the vocabulary that every collective shares (i.e., [`CollectiveMode`], [`CollectiveOptions`], and
//! named axis resolution), while each operation family lives in its own submodule:
//! [`parallel_reduce`], [`parallel_vary`], [`all_gather`], [`parallel_sum_scatter`], [`parallel_permute`],
//! [`all_to_all`], and [`ragged_all_to_all`]. It also owns the shared machinery of the single-input linear collectives
//! ([`ParallelPermuteOperation`], [`AllGatherOperation`], [`ParallelSumScatterOperation`], and [`AllToAllOperation`]).
//! Each carries the referenced axis name and the participant count resolved from the active [`NamedAxes`] environment,
//! consumes one statically shaped array input, and has only degenerate single-participant semantics outside a binder.
//! Its tangent rides the same collective, and its transpose is another collective over the same axis. The
//! [`linear_collective!`] macro generates their common operation structure, while shared functions support the
//! generated code and hand-written rules.
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
use crate::batching::{BatchAxis, BatchingContext, BatchingError};
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
use crate::partial::PartialValue;
use crate::programs::{
    MaybeZero, Operation, OperationProjection, ProgramError, TypeError, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

pub mod all_gather;
pub mod all_to_all;
pub mod parallel_permute;
pub mod parallel_reduce;
pub mod parallel_sum_scatter;
pub mod parallel_vary;
pub mod ragged_all_to_all;

pub use all_gather::{ALL_GATHER_OPERATION_NAME, AllGather, AllGatherOperation, AllGatherOutputVariance};
pub use all_to_all::{ALL_TO_ALL_OPERATION_NAME, AllToAll, AllToAllOperation, ParallelSwapAxes};
pub use parallel_permute::{
    PARALLEL_PERMUTE_OPERATION_NAME, ParallelPermute, ParallelPermuteOperation, ParallelShuffle,
};
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
/// inserts its sender dimension at the concat axis, after the dimension it concatenates along, so recovering the tiled
/// `f32[1, 24]` result from the untiled `f32[6, 4]` result also requires moving that sender dimension in front of the
/// concatenated dimension first. An untiled sum-scatter and an untiled all-to-all require the selected axis to have
/// extent exactly `n`, while their tiled forms only require it to be divisible by `n`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum CollectiveMode {
    /// Gives the participants an array dimension of their own with extent `n`: an all-gather inserts it at its concat
    /// axis, a sum-scatter consumes its scatter axis (whose extent must be exactly `n`), and an all-to-all consumes its
    /// split axis (whose extent must be exactly `n`) and inserts a sender dimension at its concat axis.
    #[default]
    Untiled,

    /// Folds the participants into an existing array dimension, preserving the rank: an all-gather multiplies the
    /// extent of its concat axis by `n`, a sum-scatter divides the extent of its scatter axis by `n`, and an all-to-all
    /// divides the extent of its split axis by `n` and multiplies the extent of its concat axis by `n`. Each divided
    /// extent must be divisible by `n`.
    Tiled,
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
pub(super) fn effective_collective_axis_size(
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
pub(super) fn resolve_named_axis_size<C: NamedAxes>(context: &C, axis_name: &str) -> Result<usize, ProgramError> {
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

// TODO(eaplatanios): Review form here onwards.

/// Validates the shared input contract of the linear collectives (exactly one statically shaped input, which may
/// carry unreduced axes only when the collective accepts them) and returns the input's static dimensions.
///
/// # Parameters
///
///   - `operation_name`: Name of the collective, used in diagnostics.
///   - `accepts_unreduced`: Whether the collective accepts inputs with unreduced axes (e.g., a sum-scatter, which
///     completes the pending reduction as part of its exchange).
///   - `input_types`: Input types of the collective.
fn linear_collective_dimensions(
    operation_name: &str,
    accepts_unreduced: bool,
    input_types: &[ArrayType],
) -> Result<Vec<usize>, TypeError> {
    check_count!("input", input_types, 1, TypeError);
    if !accepts_unreduced && !input_types[0].unreduced_axes().is_empty() {
        return Err(TypeError::invalid(format!("`{operation_name}` does not support unreduced inputs")));
    }
    let Some(shape) = input_types[0].static_shape() else {
        return Err(TypeError::invalid(format!("`{operation_name}` does not support dynamically shaped inputs")));
    };
    Ok(shape.dimensions().to_vec())
}

/// Builds a linear collective's output type from its input and (possibly resized) dimensions, carrying the input
/// sharding through with the same per-dimension placement (the dimension count never changes).
fn linear_collective_output_type(
    operation_name: &'static str,
    input_type: &ArrayType,
    output_dimensions: Vec<usize>,
) -> Result<ArrayType, TypeError> {
    let output_sizes = output_dimensions.into_iter().map(Dimension::Static).collect::<Vec<_>>();
    let sharding = input_type.resized_sharding(output_sizes.as_slice(), operation_name)?;
    let mut output_type =
        ArrayType::new(input_type.data_type(), Shape::new(output_sizes)).with_memory(input_type.memory());
    output_type.sharding = sharding;
    Ok(output_type)
}

/// Interprets a linear collective outside any binder: only the degenerate single-participant axis
/// (`axis_size == 1`) has defined per-item semantics (the identity), and any larger axis reports an error because
/// the other participants do not exist per item.
fn interpret_degenerate_collective<V: Clone>(
    operation_name: &str,
    axis_name: &str,
    axis_size: usize,
    inputs: &[V],
) -> Result<Vec<V>, ProgramError> {
    check_count!("input", inputs, 1, ProgramError);
    if axis_size != 1 {
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "cannot interpret `{operation_name}` over axis `{axis_name}` of size {axis_size} without an \
                 enclosing binder",
            ),
        });
    }
    Ok(vec![inputs[0].clone()])
}

/// Implements the shared structure of the single-input linear collectives: the operation constant and struct with
/// its accessors, the `Display`/`Operation` implementations (with payload-dependent output-shape inference provided as
/// a closure over the input dimensions), degenerate interpretation, default partial evaluation, and the linear
/// forward-mode rule (the tangent rides the same collective). The batching and transposition rules and the
/// value-level staging capabilities are hand-written next to each macro invocation because each collective
/// materializes the mapped batch axis, and exposes its named axis to users, differently.
macro_rules! linear_collective {
    // Public form: generates the operation constant and struct with its accessors, the `Display`/`Operation`
    // implementations, degenerate interpretation, and default partial evaluation. `accepts_unreduced` states whether
    // type inference accepts inputs with unreduced axes.
    (
        $(#[$operation_documentation:meta])*
        operation = $operation:ident,
        name = $operation_name:ident = $name_literal:literal,
        accepts_unreduced = $accepts_unreduced:literal,
        fields = { $($(#[$field_documentation:meta])* $field:ident: $field_type:ty),* $(,)? },
        infer = |$infer_self:ident, $input_type:ident, $dimensions:ident| $infer:block $(,)?
    ) => {
        /// Canonical operation name for the operation.
        pub const $operation_name: &str = $name_literal;

        $(#[$operation_documentation])*
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        pub struct $operation {
            /// Axis name referenced by this collective.
            axis_name: String,

            /// Number of participants along the named axis, resolved from the active [`NamedAxes`] environment when
            /// the operation is staged.
            axis_size: usize,

            $($(#[$field_documentation])* $field: $field_type,)*
        }

        impl $operation {
            /// Creates a new operation over the named axis with the provided resolved axis size.
            #[inline]
            pub fn new(axis_name: String, axis_size: usize, $($field: $field_type),*) -> Self {
                Self { axis_name, axis_size, $($field),* }
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

        impl Display for $operation {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.render(formatter, 0)
            }
        }

        impl Operation for $operation {
            type Type = ArrayType;

            #[inline]
            fn name(&self) -> &'static str {
                $operation_name
            }

            fn infer_output_types(
                &self,
                input_types: &[ArrayType],
                region_interfaces: &[RegionInterface<ArrayType>],
            ) -> Result<Vec<ArrayType>, TypeError> {
                check_count!("region", region_interfaces, 0, TypeError);
                // A zero-participant collective is rejected before any extent arithmetic divides by its size.
                if self.axis_size == 0 {
                    return Err(TypeError::invalid(format!("`{}` axis size must be greater than zero", $name_literal)));
                }
                let $dimensions = linear_collective_dimensions($name_literal, $accepts_unreduced, input_types)?;
                let $infer_self = self;
                let $input_type = &input_types[0];
                Ok(vec![$infer?])
            }

            fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
                OperationFormatter::new(formatter, indentation, $operation_name)?.bracketed(|operation| {
                    operation.field("axis_name", format_args!("{:?}", self.axis_name))?;
                    operation.field("axis_size", &self.axis_size)?;
                    $(operation.field(stringify!($field), format_args!("{:?}", &self.$field))?;)*
                    Ok(())
                })
            }
        }

        impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for $operation {
            fn interpret<D: InterpretationDriver<C>>(
                &self,
                _context: &C,
                _driver: &D,
                inputs: &[C::Value],
            ) -> Result<Vec<C::Value>, ProgramError> {
                interpret_degenerate_collective($name_literal, &self.axis_name, self.axis_size, inputs)
            }
        }

        // Partial evaluation defers to the default fold-or-residualize behavior of
        // `Program::partially_evaluate`.
        impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for $operation where
            C::Operation: From<$operation>
        {
        }
    };

    // Generates the linear forward-mode rule after the operation's batching implementation.
    (@differentiation $operation:ident) => {
        // Forward-mode rule: the collective is linear, so the tangent rides the same collective. Structural-zero
        // tangents stay symbolic, retyped to the output tangent type (the collective changes shapes).
        impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for $operation
        where
            C::Operation: From<$operation>,
        {
            fn jvp<D: DifferentiationDriver<C>, P: $crate::DifferentiationPolicy<C>>(
                &self,
                context: &$crate::DifferentiationContext<C, P>,
                _driver: &D,
                inputs: &[DifferentiationDual<C::Value>],
            ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
                check_count!("input", inputs, 1, ProgramError);
                let mut primal_outputs =
                    context.primal().bind(self.clone(), Vec::new(), std::slice::from_ref(inputs[0].primal()))?;
                check_count!("output", primal_outputs, 1, ProgramError);
                let primal = primal_outputs.remove(0);
                let tangent = match inputs[0].tangent() {
                    MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
                    MaybeZero::Value(tangent) => {
                        let mut tangent_outputs =
                            context.tangent().bind(self.clone(), Vec::new(), std::slice::from_ref(tangent))?;
                        check_count!("output", tangent_outputs, 1, ProgramError);
                        MaybeZero::Value(tangent_outputs.remove(0))
                    }
                };
                Ok(vec![DifferentiationDual::new(primal, tangent)?])
            }
        }
    };
}

use linear_collective;

/// Stages the adjoint collective of a linear collective on the output cotangent: a known input receives a structural
/// zero, a zero output cotangent stays symbolic, and a live cotangent rides the provided adjoint operation.
fn transpose_linear_collective<V, O, A>(
    context: &mut TracingContext<V, O>,
    inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
    outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    adjoint: A,
) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError>
where
    V: Value<Type = ArrayType>,
    O: Operation<Type = ArrayType> + From<A>,
    A: Operation<Type = ArrayType>,
{
    check_count!("input", inputs, 1, ProgramError);
    check_count!("output", outputs, 1, ProgramError);
    if inputs[0].is_known() {
        return Ok(vec![MaybeZero::Zero(inputs[0].r#type().cotangent()?)]);
    }
    match &outputs[0] {
        MaybeZero::Value(cotangent) => {
            let mut contributions = context.bind(O::from(adjoint), Vec::new(), std::slice::from_ref(cotangent))?;
            check_count!("output", contributions, 1, ProgramError);
            Ok(vec![MaybeZero::Value(contributions.remove(0))])
        }
        MaybeZero::Zero(_) => Ok(vec![MaybeZero::Zero(inputs[0].r#type().cotangent()?)]),
    }
}

// TODO(eaplatanios): Review this module.

/// Infers one canonical mixed collective result from an array input followed by one explicit extent per output axis.
///
/// # Parameters
///
///   - `operation_name`: Name of the collective, used in diagnostics.
///   - `accepts_unreduced`: Whether the collective accepts array inputs with unreduced axes (refer to
///     [`linear_collective_dimensions`](linear_collective_dimensions) for more
///     information).
///   - `input_types`: Array input type followed by one explicit extent type per output axis.
///   - `base_output_type`: Output type whose shape is replaced by the explicit extents.
///   - `unchanged_input_axes`: For every output axis, the input axis whose extent it must preserve, if any.
///   - `validate_exact_extents`: Collective-specific validation of the explicit extents against the array input.
fn infer_explicit_shape_changing_collective_output_type(
    operation_name: &'static str,
    accepts_unreduced: bool,
    input_types: &[ArrayIrType],
    base_output_type: ArrayType,
    unchanged_input_axes: &[Option<usize>],
    validate_exact_extents: impl FnOnce(&ArrayType, &[Dimension]) -> Result<(), TypeError>,
) -> Result<Vec<ArrayIrType>, TypeError> {
    let expected = 1 + base_output_type.rank();
    check_count!("input", input_types, expected, TypeError);
    let input_type = <&ArrayType>::try_from(&input_types[0])?;
    if !accepts_unreduced && !input_type.unreduced_axes().is_empty() {
        return Err(TypeError::invalid(format!("`{operation_name}` does not support unreduced inputs")));
    }
    let output_extents = ArrayIrType::extents(&input_types[1..])?;
    if unchanged_input_axes.len() != output_extents.len() {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` internal output-axis mapping has length {} but the result rank is {}",
            unchanged_input_axes.len(),
            output_extents.len(),
        )));
    }
    for (output_axis, (&input_axis, output_extent)) in unchanged_input_axes.iter().zip(&output_extents).enumerate() {
        let Some(input_axis) = input_axis else { continue };
        let input_extent = input_type.shape().dimensions().get(input_axis).ok_or_else(|| {
            TypeError::invalid(format!(
                "`{operation_name}` unchanged output axis {output_axis} references input axis {input_axis}, which is \
                 out of bounds for rank {}",
                input_type.rank(),
            ))
        })?;
        if output_extent != input_extent {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` output axis {output_axis} extent {output_extent} must equal unchanged input axis \
                 {input_axis} extent {input_extent}",
            )));
        }
    }
    validate_exact_extents(input_type, output_extents.as_slice())?;
    Ok(vec![base_output_type.with_shape(Shape::new(output_extents)).into()])
}

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
        let zero = Self::collective_extent_constant(context, 0)?;
        let one = Self::collective_extent_constant(context, 1)?;
        right.compare(&zero, ComparisonDirection::GreaterThan)?.assert(
            "collective divisor must be positive",
            &[("divisor", ValueProjection::<DimensionType>::from_projected(right.clone()))],
        )?;
        let divisor = right.dimension_max(&one)?;
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

/// Forwards one shape-changing collective while updating its mapped result axis.
fn forward_shape_changing_collective<C, P>(
    context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
    operation: C::Operation,
    input: &ArrayBatch<C::Value>,
    output_batch_axis: Option<usize>,
) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError>
where
    C: Context<Type = ArrayType>,
    P: ArrayExtentBatchingPolicy<C>,
{
    let mut outputs = context.parent().bind(operation, Vec::new(), std::slice::from_ref(input.value()))?;
    check_count!("output", outputs, 1, ProgramError);
    let output = outputs.remove(0);
    let output_batch_axis = output_batch_axis.map_or_else(BatchAxis::replicated, BatchAxis::from_position);
    Ok(vec![ArrayBatch::new(output, output_batch_axis)?])
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

/// Returns an exact first-class collective extent constant.
fn collective_extent_constant<V>(context: &V::DispatchDomain, extent: usize) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayIrType>,
    V::DispatchDomain: DimensionConstant<Value = V>,
{
    context.dimension_constant(extent)
}

/// Returns one first-class dimension for every input array axis, using exact constants for static axes and explicit
/// [`DimensionSize`] gateways for dynamic axes.
fn collective_input_extents<V>(context: &V::DispatchDomain, value: &V) -> Result<Vec<V>, ProgramError>
where
    V: Value<Type = ArrayIrType> + DimensionSize<V>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    V::DispatchDomain: DimensionConstant,
{
    let r#type = value.r#type();
    let input_type = <&ArrayType>::try_from(r#type.as_ref())?;
    input_type
        .shape()
        .dimensions()
        .iter()
        .enumerate()
        .map(|(axis, dimension)| match dimension {
            Dimension::Static(extent) => collective_extent_constant(context, *extent),
            Dimension::Dynamic(_) => value.dimension_size(axis),
        })
        .collect()
}

/// Computes one tiled collective result extent by multiplying an input-axis extent by the effective participant count.
fn multiplied_collective_extent<V>(
    context: &V::DispatchDomain,
    input_extent: &V,
    effective_axis_size: usize,
) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayIrType> + ValueProjection<DimensionType>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    V::DispatchDomain: DimensionConstant,
    <V as ValueProjection<DimensionType>>::Projected: Mul,
{
    let input_extent = <V as ValueProjection<DimensionType>>::into_projected(input_extent.clone())?;
    let effective_axis_size = collective_extent_constant(context, effective_axis_size)?;
    let effective_axis_size = <V as ValueProjection<DimensionType>>::into_projected(effective_axis_size)?;
    Ok(<V as ValueProjection<DimensionType>>::from_projected(input_extent.mul(&effective_axis_size)?))
}

/// Computes one tiled collective result extent by requiring exact divisibility and dividing an input-axis extent by
/// the effective participant count.
fn divided_collective_extent<V>(
    context: &V::DispatchDomain,
    input_extent: &V,
    effective_axis_size: usize,
) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayIrType> + Assert + ValueProjection<DimensionType>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    V::DispatchDomain: DimensionConstant,
    <V as ValueProjection<DimensionType>>::Projected:
        Value<Type = DimensionType> + Compare<V> + DimensionMax + Rem + Div,
{
    let input_extent = <V as ValueProjection<DimensionType>>::into_projected(input_extent.clone())?;
    let effective_axis_size = collective_extent_constant(context, effective_axis_size)?;
    let effective_axis_size = <V as ValueProjection<DimensionType>>::into_projected(effective_axis_size)?;
    let zero = ValueProjection::<DimensionType>::into_projected(context.dimension_constant(0)?)?;
    let one = ValueProjection::<DimensionType>::into_projected(context.dimension_constant(1)?)?;
    effective_axis_size.compare(&zero, ComparisonDirection::GreaterThan)?.assert(
        "collective divisor must be positive",
        &[("divisor", ValueProjection::<DimensionType>::from_projected(effective_axis_size.clone()))],
    )?;
    let divisor = effective_axis_size.dimension_max(&one)?;
    input_extent.rem(&divisor)?.compare(&zero, ComparisonDirection::Equal)?.assert(
        "collective extent must be divisible by the participant count",
        &[
            ("extent", ValueProjection::<DimensionType>::from_projected(input_extent.clone())),
            ("divisor", ValueProjection::<DimensionType>::from_projected(effective_axis_size)),
        ],
    )?;
    Ok(ValueProjection::<DimensionType>::from_projected(input_extent.div(&divisor)?))
}

/// Requires an input axis extent to equal the effective participant count used by an untiled collective.
fn require_collective_axis_extent<V>(
    context: &V::DispatchDomain,
    input_extent: &V,
    effective_axis_size: usize,
) -> Result<(), ProgramError>
where
    V: Value<Type = ArrayIrType> + Assert + ValueProjection<DimensionType>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    V::DispatchDomain: DimensionConstant,
    <V as ValueProjection<DimensionType>>::Projected: Value<Type = DimensionType> + Compare<V>,
{
    let input_extent = <V as ValueProjection<DimensionType>>::into_projected(input_extent.clone())?;
    let effective_axis_size = collective_extent_constant(context, effective_axis_size)?;
    let effective_axis_size = <V as ValueProjection<DimensionType>>::into_projected(effective_axis_size)?;
    input_extent.compare(&effective_axis_size, ComparisonDirection::Equal)?.assert(
        "collective axis extent must match the participant count",
        &[
            ("extent", ValueProjection::<DimensionType>::from_projected(input_extent)),
            ("participants", ValueProjection::<DimensionType>::from_projected(effective_axis_size)),
        ],
    )
}

/// Requires an input axis extent to be exactly divisible by the effective participant count.
fn require_collective_axis_divisible<V>(
    context: &V::DispatchDomain,
    input_extent: &V,
    effective_axis_size: usize,
) -> Result<(), ProgramError>
where
    V: Value<Type = ArrayIrType> + Assert + ValueProjection<DimensionType>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    V::DispatchDomain: DimensionConstant,
    <V as ValueProjection<DimensionType>>::Projected: Value<Type = DimensionType> + Compare<V> + DimensionMax + Rem,
{
    let input_extent = <V as ValueProjection<DimensionType>>::into_projected(input_extent.clone())?;
    let effective_axis_size = collective_extent_constant(context, effective_axis_size)?;
    let effective_axis_size = <V as ValueProjection<DimensionType>>::into_projected(effective_axis_size)?;
    let zero = ValueProjection::<DimensionType>::into_projected(context.dimension_constant(0)?)?;
    let one = ValueProjection::<DimensionType>::into_projected(context.dimension_constant(1)?)?;
    effective_axis_size.compare(&zero, ComparisonDirection::GreaterThan)?.assert(
        "collective divisor must be positive",
        &[("divisor", ValueProjection::<DimensionType>::from_projected(effective_axis_size.clone()))],
    )?;
    input_extent
        .rem(&effective_axis_size.dimension_max(&one)?)?
        .compare(&zero, ComparisonDirection::Equal)?
        .assert(
            "collective extent must be divisible by the participant count",
            &[
                ("extent", ValueProjection::<DimensionType>::from_projected(input_extent)),
                ("divisor", ValueProjection::<DimensionType>::from_projected(effective_axis_size)),
            ],
        )
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
        Shape,
    };
    use crate::batching::BatchableOperation;
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{MemberDifferentiableOperation, transpose_mixed_operation};
    use crate::macros::check_operation_partial_evaluation;
    use crate::operations::collectives::all_gather::{
        AllGatherOperation, AllGatherOutputVariance, infer_explicit_all_gather_output_types,
    };
    use crate::operations::collectives::all_to_all::{AllToAllOperation, infer_explicit_all_to_all_output_types};
    use crate::operations::collectives::parallel_sum_scatter::{
        ParallelSumScatterOperation, infer_explicit_parallel_sum_scatter_output_types,
    };
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::{EmptyRegionDriver, ProgramBuilder};
    use crate::tracing::TracingContext;

    use super::*;

    /// Returns the static `f32` vector type of the provided length shared by the collective tests.
    pub(super) fn f32_vector(length: usize) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(length)]))
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
                &AllGatherOperation::new("x".to_string(), 4, 0, grouped.clone(), AllGatherOutputVariance::Varying,),
                &[f32_vector(3).into(), result_extent.into(),],
            ),
            Ok(vec![f32_vector(6).into()]),
        );
        assert_eq!(
            infer_explicit_parallel_sum_scatter_output_types(
                &ParallelSumScatterOperation::new("x".to_string(), 4, 0, grouped),
                &[f32_vector(6).into(), DimensionValue::constant(3).unwrap().r#type().into_owned().into()],
            ),
            Ok(vec![f32_vector(3).into()]),
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

        // Direct mixed transposition delegates the array contribution through the homogeneous projection and gives
        // the explicit extent input a structural-zero cotangent.
        let context = Context::new();
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]));
        let output_cotangent = context.input(array_type.clone().into());
        let extent_type = DimensionValue::constant(3)?.r#type().into_owned();
        let mut context = crate::differentiation::TranspositionContext::new(context);
        let inputs = [PartialValue::Unknown(array_type.into()), PartialValue::Unknown(extent_type.into())];
        let accumulators = context.cotangent_accumulators(&inputs, &[])?;
        transpose_mixed_operation(
            &mut context,
            &ParallelSumScatterOperation::new("x".to_string(), 1, 0, CollectiveOptions::tiled()),
            &inputs,
            &[MaybeZero::Value(output_cotangent)],
            &accumulators,
        )?;
        let cotangents = context.take_cotangents(&accumulators)?;
        assert!(matches!(cotangents.as_slice(), [MaybeZero::Value(_), MaybeZero::Zero(_)]));
        assert!(matches!(
            context.builder().borrow().instructions()[0].operation(),
            ArrayIrOperation::Array(ArrayOperation::AllGather(_)),
        ));

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
            infer_explicit_parallel_sum_scatter_output_types(
                &ParallelSumScatterOperation::new("x".to_string(), 4, 1, CollectiveOptions::default()),
                &[
                    shape(vec![Dimension::Static(2), Dimension::Static(4), Dimension::Static(3)]).into(),
                    DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                ],
            ),
            Ok(vec![shape(vec![Dimension::Static(2), Dimension::Static(3)]).into()]),
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
        assert_eq!(
            infer_explicit_parallel_sum_scatter_output_types(
                &ParallelSumScatterOperation::new("x".to_string(), 4, 1, CollectiveOptions::default()),
                &[
                    shape(vec![Dimension::Static(2), Dimension::Static(5)]).into(),
                    DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
                ],
            ),
            Err(TypeError::invalid("`parallel_sum_scatter` untiled scatter axis 1 size 5 must equal group size 4",)),
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
            infer_explicit_parallel_sum_scatter_output_types(
                &ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled()),
                &[
                    input_type.clone().into(),
                    ArrayIrType::Dimension(DimensionType::from(split_result.clone())),
                    DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                ],
            ),
            Ok(vec![
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(split_result.clone()), Dimension::Static(3)]),
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
                &[f32_vector(3).into(), exact_six.into()],
            ),
            Ok(vec![f32_vector(6).into()]),
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
                &[f32_vector(3).into(), exact_five.into()],
            ),
            Err(TypeError::invalid(
                "`all_gather` result extent must equal input axis 0 extent 3 multiplied by axis group size 2; \
                 expected 6 \
                 but got 5"
                    .to_string(),
            )),
        );
        assert_eq!(
            infer_explicit_parallel_sum_scatter_output_types(
                &ParallelSumScatterOperation::new("empty".to_string(), 0, 0, CollectiveOptions::tiled()),
                &[f32_vector(3).into(), DimensionValue::constant(3).unwrap().r#type().into_owned().into()],
            ),
            Err(TypeError::invalid("`parallel_sum_scatter` axis size must be greater than zero")),
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

        let scattered = ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::default())
            .batch(&context, &EmptyRegionDriver, &[mapped_matrix()])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(scattered[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(scattered[0].value(), &Array::vector(vec![4.0_f32, 6.0]).unwrap());
        assert_eq!(scattered[0].unbatched_type(), ArrayType::scalar(DataType::F32));

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
        let input = builder.add_input(f32_vector(8));
        let output = builder
            .add_instruction(
                ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled()),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let transposed_twice =
            program.transpose_with_respect_to(&[0], &[]).unwrap().transpose_with_respect_to(&[0], &[]).unwrap();
        assert!(matches!(transposed_twice.instructions()[0].operation(), ArrayOperation::ParallelSumScatter(_)));
        assert_eq!(transposed_twice.input_types(), program.input_types());
        assert_eq!(transposed_twice.output_types(), program.output_types());

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
                ParallelSumScatterOperation::new("x".to_string(), 1, 0, CollectiveOptions::tiled()),
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
        assert_eq!(
            context
                .bind(
                    ParallelSumScatterOperation::new("empty".to_string(), 0, 0, CollectiveOptions::tiled()),
                    Vec::new(),
                    &[input.clone(), extent.clone()],
                )
                .unwrap_err(),
            ProgramError::Type(TypeError::invalid("`parallel_sum_scatter` axis size must be greater than zero")),
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
        let variable = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let dimension_type = DimensionType::from(variable.clone());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = builder.add_input(array_type.into());
        let result_extent = builder.add_input(dimension_type.clone().into());
        let output = builder
            .add_instruction(
                ParallelSumScatterOperation::new("x".to_string(), 1, 0, CollectiveOptions::tiled()),
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
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        assert!(linearization.tangent().to_string().contains("linear_call [residual_count=1]"));
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap());
        let extent = ArrayIrValue::Dimension(DimensionValue::new(dimension_type, 3).unwrap());
        let mut primal_outputs = linearization.primal().interpret(vec![input, extent]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let cotangent = ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]).unwrap());
        let mut pullback_inputs = vec![cotangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![cotangent]));

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
