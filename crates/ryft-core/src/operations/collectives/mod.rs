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
use crate::operations::Capability;
use crate::operations::arithmetic::{AddOperation, Div, Mul, Rem};
use crate::operations::assertions::Assert;
use crate::operations::comparisons::{Compare, ComparisonDirection};
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::differentiation::linear_call::LinearCallOperation;
use crate::operations::dimensions::dimension_max::DimensionMax;
use crate::operations::dimensions::dimension_size::{DimensionSize, DimensionSizeOperation};
use crate::operations::manipulation::broadcasting::{Broadcast, DynamicBroadcast, DynamicBroadcastOperation};
use crate::operations::manipulation::reshaping::{DynamicReshapeOperation, Reshape};
use crate::operations::manipulation::transposition::Transpose;
use crate::partial::PartialValue;
use crate::programs::{
    MaybeZero, Operation, OperationProjection, ProgramError, RegionInterface, TypeError, Typed, Value, ValueProjection,
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
pub use parallel_all_to_all::{PARALLEL_ALL_TO_ALL_OPERATION_NAME, ParallelAllToAll, ParallelAllToAllOperation};
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
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
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

/// Value that stages shape-changing collectives directly through its homogeneous array dispatch domain. This marker
/// opts a value into the provided [`ParallelAllGather`], [`ParallelSumScatter`], and [`ParallelAllToAll`]
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
    fn validate_input<'o>(
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
    fn validate_degenerate_interpretation(&self) -> Result<(), ProgramError> {
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

/// [`LinearCollectiveOperation`] that resizes an array axis (e.g., [`ParallelAllGatherOperation`],
/// [`ParallelSumScatterOperation`], and [`ParallelAllToAllOperation`]). Its output shape depends on the participant
/// count and its [`CollectiveMode`], so in the composite array/dimension family it is staged with one explicit extent
/// input per output axis. This trait captures the hooks that differ between these collectives (i.e., their options and
/// their composite type inference) and provides the composite interpretation and forward-mode differentiation rules,
/// together with the batching rules of both array families, on top of them and of their
/// [`ShapeChangingCollectiveBatching`] implementations.
trait ShapeChangingCollectiveOperation: LinearCollectiveOperation {
    /// Returns the [`CollectiveOptions`] of this [`ShapeChangingCollectiveOperation`].
    fn collective_options(&self) -> &CollectiveOptions;

    /// Infers the output type of this collective in the composite array/dimension family, whose array input is followed
    /// by one explicit extent per output axis. Statically known extents are checked here, while dynamic extents are
    /// checked by the runtime assertions that the collective's capability stages.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] if the inputs violate the collective's shape, extent, or mesh contract.
    fn infer_array_ir_output_types(&self, input_types: &[ArrayIrType]) -> Result<Vec<ArrayIrType>, TypeError>;

    /// Returns the error that the provided batching rules raise for a bounded ragged `dimension` on input
    /// `input_index`, which these collectives cannot route because one extent per item does not describe how
    /// the participants partition their live elements.
    fn unsupported_ragged_input_error(&self, dimension: &DimensionVariable, input_index: usize) -> BatchingError {
        BatchingError::UnsupportedOperation {
            message: format!(
                "`{}` does not support bounded ragged dimension `{}` on input {}",
                self.name(),
                dimension,
                input_index,
            ),
        }
    }

    /// Implements [`interpret_in_parent`](crate::MemberInterpretableOperation::interpret_in_parent) for this
    /// collective. Outside any binder, only a collective whose instances each combine a single participant has defined
    /// semantics: a tiled one leaves the array unchanged, and an untiled one only removes or inserts a size-one axis.
    /// The explicit result extents must match the shape that the observed input implies.
    fn shape_changing_collective_interpret_in_parent<
        C: Domain<
                Type = ArrayIrType,
                Value: ValueProjection<
                    ArrayType,
                    Projected: Value<Type = ArrayType> + Reshape + DimensionSize<usize>,
                > + ValueProjection<DimensionType, Projected = DimensionValue>,
            >,
    >(
        &self,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
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
                        "`{}` output axis {} extent must equal observed result extent {} but got {}",
                        self.name(),
                        axis,
                        expected,
                        actual,
                    ),
                });
            }
        }

        // A degenerate tiled collective leaves the array unchanged, and its untiled form only removes or inserts a
        // size-one axis, so reshaping to the validated result shape is sufficient and preserves element order.
        self.validate_degenerate_interpretation()?;
        let output = match self.collective_options().mode() {
            CollectiveMode::Tiled => input,
            CollectiveMode::Untiled => input.reshape(Shape::from(expected_extents))?,
        };

        Ok(vec![<C::Value as ValueProjection<ArrayType>>::from_projected(output)])
    }

    /// Implements [`jvp_in_parent`](crate::MemberDifferentiableOperation::jvp_in_parent) for this collective. The
    /// explicit output extents and the exact input shape become ordinary residuals of one linear call, whose transpose
    /// applies the [`adjoint`](LinearCollectiveOperation::adjoint) of the primal array input to the output cotangent.
    fn shape_changing_collective_jvp_in_parent<
        C: Context<
                Type = ArrayIrType,
                Operation: From<Self>
                               + From<Self::Adjoint>
                               + From<ConstantOperation<DimensionValue>>
                               + From<DimensionSizeOperation>
                               + From<LinearCallOperation<ArrayIrType>>,
            >,
        P: DifferentiationPolicy<C>,
    >(
        &self,
        context: &DifferentiationContext<C, P>,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
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

    /// Implements [`BatchableOperation::batch`](crate::BatchableOperation::batch) for this collective in the
    /// homogeneous array family. Bounded ragged inputs are rejected. A `batch` level that does not bind the
    /// collective's axis forwards it to its parent with its array axes moved past the mapped axis, while a level that
    /// binds the axis consumes it through [`batch_matching_axis`](ShapeChangingCollectiveBatching::batch_matching_axis).
    fn shape_changing_collective_batch<
        C: Context<Type = ArrayType, Operation: From<Self>>,
        P: CollectiveArrayExtentBatchingPolicy<C>,
    >(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError>
    where
        Self: ShapeChangingCollectiveBatching<C>,
    {
        if let Some((index, ragged_axis)) = inputs
            .iter()
            .enumerate()
            .find_map(|(index, input)| input.ragged_axes().first().map(|axis| (index, axis)))
        {
            return Err(self.unsupported_ragged_input_error(ragged_axis.dimension(), index));
        }

        if context.axis_name() != Some(self.axis_name()) {
            return context.forward_collective(self, inputs);
        }

        self.reject_mesh_form()?;
        let [input] = inputs else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() }.into());
        };

        let (output_type, output_extents) =
            context.infer_collective_output_type_and_extents(self, &input.unbatched_type())?;
        Ok(vec![self.batch_matching_axis(context, input, output_extents, output_type.sharding().cloned())?].into())
    }

    /// Implements [`batch_in_parent`](crate::MemberBatchableOperation::batch_in_parent) for this collective in the
    /// composite array/dimension family, whose explicit result extents remain the only source of dynamic reshape
    /// geometry. Bounded ragged inputs are rejected, and the result extents, which describe the shape shared by every
    /// batch item, must be replicated. A `batch` level that does not bind the collective's axis forwards it to its
    /// parent, while a level that binds the axis consumes it through
    /// [`batch_matching_axis`](ShapeChangingCollectiveBatching::batch_matching_axis)
    /// over the array projection of its parent.
    fn shape_changing_collective_batch_in_parent<
        C: Context<
                Type = ArrayIrType,
                Value: Assert
                           + DimensionSize
                           + DynamicBroadcast
                           + ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>
                           + ValueProjection<
                    DimensionType,
                    Projected: Value<Type = DimensionType> + Mul + Div + Rem + DimensionMax + Compare<C::Value>,
                >,
                Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
                Operation: From<Self>
                               + From<ConstantOperation<DimensionValue>>
                               + From<DynamicBroadcastOperation>
                               + From<DynamicReshapeOperation>
                               + From<DimensionSizeOperation>
                               + OperationProjection<ArrayType>,
            >,
    >(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatchingPolicy>, BatchingError>
    where
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
            return Err(self.unsupported_ragged_input_error(ragged_axis.dimension(), index));
        }

        for output_extent in output_extents {
            output_extent.validate_replicated_dimension()?;
        }

        // Infer the per-item result type before lifting physical axes. This also supplies the sharding metadata
        // used by the matching-axis kernel.
        let logical_input_types = inputs.iter().map(|input| input.unbatched_type().clone()).collect::<Vec<_>>();
        let mut logical_output_types = self.infer_array_ir_output_types(logical_input_types.as_slice())?;
        let logical_output_type = <&ArrayType>::try_from(&logical_output_types.remove(0))?.clone();

        if context.axis_name() != Some(self.axis_name()) {
            return Ok(context.forward_collective(self, array, output_extents)?.into());
        }

        self.reject_mesh_form()?;

        // Project the composite values onto their array and dimension domains, so that the homogeneous kernel
        // can use the explicit dimension values directly for dynamic reshape geometry.
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

/// Representation boundary used only by shape-changing collective batching rules. The collective kernels determine the
/// exchange geometry. This trait exposes the extent representation, exact division, and the alignment and reshape
/// encodings that differ between homogeneous arrays and composite array/dimension programs.
trait CollectiveArrayExtentBatchingPolicy<C: Context<Type = ArrayType>>: ArrayExtentBatchingPolicy<C> {
    /// Extent representation consumed by the shared collective kernels.
    type ShapeExtent: Clone + Debug + Mul;

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

    /// Returns the exact quotient of `left` divided by `right`, requiring a positive divisor and no remainder. Dynamic
    /// policies stage assertions for requirements that the extent types cannot prove and guard the divisor before
    /// staging arithmetic that requires it to be positive.
    ///
    /// # Errors
    ///
    /// Returns a [`BatchingError`] if a requirement is statically violated or staging the checked division fails.
    /// Staged assertions reject runtime violations when the resulting program executes.
    fn divide_extents_exactly(
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

impl<C: Context<Type = ArrayType, Value: Broadcast + Reshape + Transpose>> CollectiveArrayExtentBatchingPolicy<C>
    for StaticArrayExtentBatchingPolicy
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

    fn divide_extents_exactly(
        _context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
        left: &Self::ShapeExtent,
        right: &Self::ShapeExtent,
    ) -> Result<Self::ShapeExtent, BatchingError> {
        if *right == 0 || left % right != 0 {
            return Err(BatchingError::UnsupportedOperation {
                message: format!("extent {left} must be divisible by extent {right}"),
            });
        }
        Ok(left / right)
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

impl<
    C: Context<
            Type = ArrayIrType,
            Value: Assert
                       + DimensionSize
                       + DynamicBroadcast
                       + ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>
                       + ValueProjection<
                DimensionType,
                Projected: Value<Type = DimensionType> + Mul + Div + Rem + DimensionMax + Compare<C::Value>,
            >,
            Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
            Operation: From<DynamicBroadcastOperation>
                           + From<ConstantOperation<DimensionValue>>
                           + From<DimensionSizeOperation>
                           + From<DynamicReshapeOperation>
                           + OperationProjection<ArrayType>,
        >,
> CollectiveArrayExtentBatchingPolicy<ProjectedContext<C, ArrayType>> for DynamicArrayExtentBatchingPolicy
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

    fn divide_extents_exactly(
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
            return Ok(left.div(right)?);
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
        Ok(left.div(&divisor)?)
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

impl<C: Context<Type = ArrayType>, P: ArrayExtentBatchingPolicy<C>> BatchingContext<C, ArrayBatchingPolicy<P>> {
    /// Forwards a linear collective over an axis that the active batching level does not bind to the parent context.
    /// An input without a mapped batch axis forwards the collective unchanged. A mapped input instead forwards the
    /// collective that [`LinearCollectiveOperation::adapt_to_batch_axis`] returns for the input's mapped axis position,
    /// because the collective's own axes shift around the mapped axis, together with the position of the mapped axis in
    /// the forwarded result.
    fn forward_collective<O: LinearCollectiveOperation>(
        &self,
        operation: &O,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError>
    where
        C::Operation: From<O>,
    {
        let [input] = inputs else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() }.into());
        };
        let Some(batch_axis) = input.batch_axis_position() else {
            return Ok(self.forward_to_parent(C::Operation::from(operation.clone()), inputs)?.into());
        };
        let (operation, output_batch_axis) = operation.adapt_to_batch_axis(batch_axis);
        let mut outputs =
            self.parent().bind(C::Operation::from(operation), Vec::new(), std::slice::from_ref(input.value()))?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(vec![ArrayBatch::new(outputs.remove(0), BatchAxis::from_position(output_batch_axis))?].into())
    }

    /// Infers the output type of a shape-changing collective for the logical (i.e., unbatched) `input_type` of a level
    /// that binds its axis, and returns that type together with its extents in the batching policy's representation,
    /// which the collective's matching-axis kernel consumes.
    fn infer_collective_output_type_and_extents<O: Operation<Type = ArrayType>>(
        &self,
        operation: &O,
        input_type: &ArrayType,
    ) -> Result<(ArrayType, Vec<P::ShapeExtent>), BatchingError>
    where
        P: CollectiveArrayExtentBatchingPolicy<C>,
    {
        let mut output_types = operation.infer_output_types(std::slice::from_ref(input_type), &[])?;
        let output_type = output_types.remove(0);
        let output_extents = output_type
            .shape()
            .dimensions()
            .iter()
            .map(|dimension| {
                let extent = dimension.value().ok_or_else(|| BatchingError::UnsupportedOperation {
                    message: "shape-changing collective batching requires statically shaped inputs".to_string(),
                })?;
                P::collective_extent_constant(self, extent)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((output_type, output_extents))
    }
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

    /// Returns this `batch` level re-expressed over the array projection of its parent with the dynamic extent batching
    /// policy, keeping its axis name, extent, and sharding, so that the matching-axis kernels of the shape-changing
    /// collectives, which operate on homogeneous arrays, can consume this level's mapped axis while reading their
    /// reshape geometry from the explicit dimension values of composite programs.
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
/// automatically for every type that implements all of its members. The context-side [`AxisIndex`] is implemented by
/// contexts rather than values and is therefore not a member.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor and is passed to every member,
/// so that homogeneous array values implement this bundle for [`ArrayType`] and composite array IR values implement it
/// for [`ArrayIrType`].
pub trait CollectiveOperations<T = <Self as Capability>::Universe>:
    Capability
    + ParallelReduce<T>
    + ParallelVary<T>
    + ParallelAllGather<T>
    + ParallelSumScatter<T>
    + ParallelPermute<T>
    + ParallelAllToAll<T>
    + ParallelRaggedAllToAll<T>
{
}

impl<
    T,
    V: ParallelReduce<T>
        + ParallelVary<T>
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
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, DataType, DimensionBounds, Layout, Memory, MeshAxis,
        ShardingDimension, StridedLayout,
    };
    use crate::batching::BatchableOperation;
    use crate::contexts::{EagerContext, StagingContext};
    use crate::operations::assertions::AssertionError;
    use crate::operations::constants::constant::DimensionConstant;
    use crate::parameters::Placeholder;
    use crate::programs::{EffectClasses, EmptyRegionDriver, Program, ProgramBuilder};

    use super::*;

    /// Creates an eager homogeneous batching level of extent `axis_size` that binds the axis `axis_name`.
    pub(super) fn eager_collective_context(
        axis_name: &str,
        axis_size: usize,
    ) -> BatchingContext<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy> {
        BatchingContext::new(EagerContext::new(), axis_size).with_axis_name(axis_name.to_string())
    }

    /// Builds the single-instruction homogeneous program that applies `operation` to one input of type `input_type`.
    pub(super) fn collective_program<O: Into<ArrayOperation<Array>>>(
        operation: O,
        input_type: ArrayType,
    ) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(input_type);
        let outputs = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap().to_vec();
        builder.build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Applies the batching rule of `operation` to `input` at an [`eager_collective_context`] level of extent
    /// `axis_size` that binds the axis `axis_name`, returning the batched outputs.
    pub(super) fn batch_collective<O: Operation<Type = ArrayType>>(
        operation: &O,
        axis_name: &str,
        axis_size: usize,
        input: ArrayBatch<Array>,
    ) -> Result<Vec<ArrayBatch<Array>>, BatchingError>
    where
        O: BatchableOperation<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>,
    {
        let context = eager_collective_context(axis_name, axis_size);
        Ok(operation.batch(&context, &EmptyRegionDriver, &[input])?.into_parts().0)
    }

    /// Creates an eager composite batching level that binds the axis `"x"` and whose mapped extent is a first-class
    /// dimension value.
    fn dynamic_collective_context(
        extent: DimensionValue,
    ) -> BatchingContext<
        ProjectedContext<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>, ArrayType>,
        ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>,
    > {
        BatchingContext::with_policy(ProjectedContext::new(EagerContext::new()), ArrayIrValue::Dimension(extent))
            .with_axis_name("x".to_string())
    }

    /// Binds `operation` at a traced composite `batch` level of extent 5 that binds the axis `"outer"`, which
    /// `operation` does not reference. The array input has physical shape `input_shape` and is mapped at
    /// `input_batch_axis`, and it is followed by one replicated extent input per entry of `output_extents`. Returns the
    /// batch axis and type of the forwarded result together with the rendering of the staged program.
    fn forward_array_ir_collective(
        operation: ArrayIrOperation<Array>,
        input_shape: Vec<usize>,
        input_batch_axis: BatchAxis,
        output_extents: &[usize],
    ) -> (BatchAxis, ArrayIrType, String) {
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let batch_extent = trace.input(DimensionValue::constant(5).unwrap().r#type().into_owned().into());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), batch_extent)
            .with_axis_name("outer".to_string());
        let array = trace.input(ArrayType::new_static(DataType::F32, input_shape).into());
        let array = ArrayIrBatch::new(array, input_batch_axis).unwrap();
        let mut inputs = vec![BatchingTracer::new(context.clone(), array)];
        for extent in output_extents {
            let extent = trace.input(DimensionValue::constant(*extent).unwrap().r#type().into_owned().into());
            inputs.push(BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(extent)));
        }
        let outputs = context.bind(operation, Vec::new(), &inputs).unwrap();
        assert_eq!(outputs.len(), 1);
        let output = outputs[0].batch();
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.value().atom_id().unwrap()],
                vec![Placeholder; 2 + output_extents.len()],
                vec![Placeholder],
            )
            .unwrap();
        (output.batch_axis(), output.value().r#type().into_owned(), program.to_string())
    }

    #[test]
    fn test_collective_mode_default() {
        assert_eq!(CollectiveMode::default(), CollectiveMode::Untiled);
    }

    #[test]
    fn test_collective_mode_forwarded_split_axes() {
        // Pin the boundary cases with the mapped axis before, at, and after the split axis explicitly. Both modes
        // shift the split axis past a mapped axis at or before it, but only an untiled split consumes the split axis
        // and thus moves a later mapped axis one position to the left.
        for (mode, split_axis, batch_axis, expected) in [
            (CollectiveMode::Tiled, 1, 0, (2, 0)),
            (CollectiveMode::Tiled, 1, 1, (2, 1)),
            (CollectiveMode::Tiled, 1, 2, (1, 2)),
            (CollectiveMode::Untiled, 1, 0, (2, 0)),
            (CollectiveMode::Untiled, 1, 1, (2, 1)),
            (CollectiveMode::Untiled, 1, 2, (1, 1)),
        ] {
            assert_eq!(mode.forwarded_split_axes(split_axis, batch_axis), expected);
        }

        // Check every position against a model that labels each physical axis with its logical axis and marks the
        // mapped batch axis with `None`.
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
        // Pin the boundary cases at and on either side of the mapped axis explicitly.
        for (mode, concatenation_axis, batch_axis, expected) in [
            (CollectiveMode::Tiled, 0, 0, (1, 0)),
            (CollectiveMode::Tiled, 0, 1, (0, 1)),
            (CollectiveMode::Untiled, 0, 0, (0, 1)),
            (CollectiveMode::Untiled, 1, 0, (2, 0)),
        ] {
            assert_eq!(mode.forwarded_concatenation_axes(concatenation_axis, batch_axis), expected);
        }

        // Check every position against a model that labels each physical axis with its logical axis and marks the
        // mapped batch axis with `None`.
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
                            "mode={mode:?}, rank={rank}, concatenation_axis={concatenation_axis}, \
                             batch_axis={batch_axis}",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_collective_mode_forwarded_split_and_concatenation_axes() {
        // Pin representative all-to-all cases, which split and then concatenate, explicitly.
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

        // Check every composition against the same axis-label model, applying the concatenation to the mapped axis
        // position that the split returns.
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
    fn test_collective_options_new() {
        let options = CollectiveOptions::new(CollectiveMode::Untiled);
        assert_eq!(options.mode(), CollectiveMode::Untiled);
        assert_eq!(options.axis_index_groups(), None);
        assert_eq!(options, CollectiveOptions::default());
        assert_eq!(format!("{options:?}"), "CollectiveOptions { mode: Untiled, axis_index_groups: None }");
    }

    #[test]
    fn test_collective_options_tiled() {
        let options = CollectiveOptions::tiled();
        assert_eq!(options, CollectiveOptions::new(CollectiveMode::Tiled));
        assert_eq!(format!("{options:?}"), "CollectiveOptions { mode: Tiled, axis_index_groups: None }");
    }

    #[test]
    fn test_collective_options_with_axis_index_groups() {
        let options = CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]);
        assert_eq!(options.mode(), CollectiveMode::Tiled);
        assert_eq!(options.axis_index_groups(), Some([vec![0, 2], vec![3, 1]].as_slice()));
        assert_eq!(
            format!("{options:?}"),
            "CollectiveOptions { mode: Tiled, axis_index_groups: Some([[0, 2], [3, 1]]) }",
        );
    }

    #[test]
    fn test_collective_options_mode() {
        assert_eq!(CollectiveOptions::default().mode(), CollectiveMode::Untiled);
        assert_eq!(CollectiveOptions::tiled().mode(), CollectiveMode::Tiled);
    }

    #[test]
    fn test_collective_options_axis_index_groups() {
        assert_eq!(CollectiveOptions::default().axis_index_groups(), None);
        assert_eq!(
            CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0], vec![1]]).axis_index_groups(),
            Some([vec![0], vec![1]].as_slice()),
        );
    }

    #[test]
    fn test_collective_options_effective_axis_size() {
        // The options delegate to `effective_collective_axis_size` with their participant groups, whose validation
        // errors propagate unchanged.
        assert_eq!(CollectiveOptions::default().effective_axis_size("parallel_all_gather", 4), Ok(4));
        let options = CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]);
        assert_eq!(options.effective_axis_size("parallel_all_gather", 4), Ok(2));
        assert_eq!(
            CollectiveOptions::default()
                .with_axis_index_groups(vec![vec![0, 1], vec![1, 2]])
                .effective_axis_size("parallel_all_gather", 4),
            Err(TypeError::invalid("`parallel_all_gather` axis index groups contain participant 1 more than once")),
        );
    }

    #[test]
    fn test_shape_changing_collective_value() {
        // Homogeneous staged, batched, and differentiated array values opt into the shared staging rules of the
        // shape-changing collectives. This compiles only if each of them implements the marker trait.
        fn requires_shape_changing_collective_value<V: ShapeChangingCollectiveValue>() {}

        requires_shape_changing_collective_value::<Tracer<TracingContext<Array, ArrayOperation<Array>>>>();
        requires_shape_changing_collective_value::<
            BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>,
        >();
        requires_shape_changing_collective_value::<DifferentiationTracer<EagerContext<Array, ArrayOperation<Array>>>>();
    }

    #[test]
    fn test_linear_collective_operation_validate_input() {
        let operation = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1), (1, 0)]);
        let input_type = ArrayType::new_static(DataType::F32, [4]);
        assert_eq!(operation.validate_input(std::slice::from_ref(&input_type), &[]), Ok(&input_type));

        // Regions are rejected first, then a zero-participant axis, and finally any input count other than one. The
        // region and axis cases use a zero-participant operation without inputs, so they also demonstrate this order.
        let zero_participant_operation = ParallelPermuteOperation::new("x".to_string(), 0, Vec::new());
        assert_eq!(
            zero_participant_operation
                .validate_input(&[], &[RegionInterface::new(Vec::new(), Vec::new(), EffectClasses::NONE)]),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
        assert_eq!(
            zero_participant_operation.validate_input(&[], &[]),
            Err(TypeError::invalid("`parallel_permute` axis size must be greater than zero")),
        );
        assert_eq!(operation.validate_input(&[], &[]), Err(TypeError::invalid("expected 1 input but got 0")));
        assert_eq!(
            operation.validate_input(&[input_type.clone(), input_type], &[]),
            Err(TypeError::invalid("expected 1 input but got 2")),
        );
    }

    #[test]
    fn test_linear_collective_operation_validate_degenerate_interpretation() {
        // A single participant needs no exchange, so the collective can be evaluated locally.
        let operation = ParallelAllToAllOperation::new("x".to_string(), 1, 0, 0, CollectiveOptions::tiled());
        assert_eq!(operation.validate_degenerate_interpretation(), Ok(()));

        // Several participants per collective instance require an enclosing binder.
        let operation = ParallelAllToAllOperation::new("x".to_string(), 2, 0, 0, CollectiveOptions::tiled());
        assert_eq!(
            operation.validate_degenerate_interpretation(),
            Err(ProgramError::UnsupportedOperation {
                message: "cannot interpret `parallel_all_to_all` over axis `x` of size 2 without an enclosing binder"
                    .to_string(),
            }),
        );

        // Singleton participant groups make every collective instance degenerate, even over a larger axis.
        assert_eq!(
            ParallelAllToAllOperation::new(
                "x".to_string(),
                2,
                0,
                0,
                CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0], vec![1]]),
            )
            .validate_degenerate_interpretation(),
            Ok(()),
        );

        // Invalid participant groups are reported as type errors.
        assert_eq!(
            ParallelAllToAllOperation::new(
                "x".to_string(),
                2,
                0,
                0,
                CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 0]]),
            )
            .validate_degenerate_interpretation(),
            Err(ProgramError::Type(TypeError::invalid(
                "`parallel_all_to_all` axis index groups contain participant 0 more than once",
            ))),
        );
    }

    #[test]
    fn test_linear_collective_operation_reject_mesh_form() {
        let operation = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1), (1, 0)]);
        assert_eq!(operation.reject_mesh_form(), Ok(()));

        // A collective over a manual mesh axis describes communication between devices, which the batch items of a
        // level that binds the same name cannot stand in for.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        assert_eq!(
            operation.with_mesh(mesh).reject_mesh_form(),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_permute` over a manual mesh axis cannot bind a named batch axis".to_string(),
            }),
        );
    }

    #[test]
    fn test_shape_changing_collective_operation_unsupported_ragged_input_error() {
        let operation = ParallelAllGatherOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::default(),
            ParallelAllGatherOutputVariance::Varying,
        );
        let dimension = DimensionVariable::new("n", DimensionBounds::new(0, Some(9)).unwrap());
        assert_eq!(
            operation.unsupported_ragged_input_error(&dimension, 1),
            BatchingError::UnsupportedOperation {
                message: "`parallel_all_gather` does not support bounded ragged dimension `n` on input 1".to_string(),
            },
        );
    }

    #[test]
    fn test_shape_changing_collective_batching_capabilities() {
        // The batching rules of each shape-changing collective require only the capabilities of its own kernel, so a
        // context whose values can transpose but not reduce still batches all-gathers and all-to-alls. This compiles
        // only if those rules hold under exactly these bounds.

        /// Requires an operation rule under precisely the supplied context and policy bounds.
        fn requires_batching<C: Context, P: BatchingPolicy<C>, O: BatchableOperation<C, P>>() {}

        /// Checks that rearrangement collectives need no reduction capability.
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
    fn test_static_array_extent_batching_policy_collective_axis_extent() {
        let context = eager_collective_context("x", 2);
        assert_eq!(
            StaticArrayExtentBatchingPolicy::collective_axis_extent(&context, "parallel_all_gather", "x", 2),
            Ok(2),
        );
        assert_eq!(
            StaticArrayExtentBatchingPolicy::collective_axis_extent(&context, "parallel_all_gather", "x", 3),
            Err(BatchingError::UnsupportedOperation {
                message:
                    "`parallel_all_gather` over axis `x` resolved axis size 3 but the mapped batch axis has size 2"
                        .to_string(),
            }),
        );
    }

    #[test]
    fn test_static_array_extent_batching_policy_collective_extent_constant() {
        let context = eager_collective_context("x", 2);
        assert_eq!(StaticArrayExtentBatchingPolicy::collective_extent_constant(&context, 0), Ok(0));
    }

    #[test]
    fn test_static_array_extent_batching_policy_divide_extents_exactly() {
        let context = eager_collective_context("x", 2);
        assert_eq!(StaticArrayExtentBatchingPolicy::divide_extents_exactly(&context, &8, &2), Ok(4));
        assert_eq!(StaticArrayExtentBatchingPolicy::divide_extents_exactly(&context, &0, &2), Ok(0));
        assert_eq!(
            StaticArrayExtentBatchingPolicy::divide_extents_exactly(&context, &8, &0),
            Err(BatchingError::UnsupportedOperation { message: "extent 8 must be divisible by extent 0".to_string() }),
        );
        assert_eq!(
            StaticArrayExtentBatchingPolicy::divide_extents_exactly(&context, &7, &2),
            Err(BatchingError::UnsupportedOperation { message: "extent 7 must be divisible by extent 2".to_string() }),
        );
    }

    #[test]
    fn test_static_array_extent_batching_policy_match_collective_axis() {
        let context = eager_collective_context("x", 2);

        // A replicated batch is broadcast to a leading mapped axis.
        let replicated = ArrayBatch::replicated(Array::vector(vec![1f32, 2.0]).unwrap());
        assert_eq!(
            StaticArrayExtentBatchingPolicy::match_collective_axis(&context, &replicated, &[2]),
            Ok(ArrayBatch::new(Array::matrix(2, 2, vec![1f32, 2.0, 1.0, 2.0]).unwrap(), BatchAxis::new(0)).unwrap()),
        );

        // A mapped batch moves its mapped axis to the front.
        let mapped =
            ArrayBatch::new(Array::matrix(2, 2, vec![1f32, 2.0, 3.0, 4.0]).unwrap(), BatchAxis::new(1)).unwrap();
        assert_eq!(
            StaticArrayExtentBatchingPolicy::match_collective_axis(&context, &mapped, &[2]),
            Ok(ArrayBatch::new(Array::matrix(2, 2, vec![1f32, 3.0, 2.0, 4.0]).unwrap(), BatchAxis::new(0)).unwrap()),
        );
    }

    #[test]
    fn test_static_array_extent_batching_policy_reshape_collective() {
        let context = eager_collective_context("x", 2);
        let input = Array::matrix(2, 3, vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(
            StaticArrayExtentBatchingPolicy::reshape_collective(&context, input, &[3, 2], None),
            Ok(Array::matrix(3, 2, vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()),
        );
    }

    #[test]
    fn test_dynamic_array_extent_batching_policy_collective_axis_extent() {
        let axis_extent = DimensionValue::constant(2).unwrap();
        let context = dynamic_collective_context(axis_extent.clone());
        assert_eq!(
            DynamicArrayExtentBatchingPolicy::collective_axis_extent(&context, "parallel_all_gather", "x", 2),
            Ok(axis_extent),
        );

        // A mismatched participant count fails the staged assertion, which the eager parent evaluates immediately.
        let error = ProgramError::from(
            DynamicArrayExtentBatchingPolicy::collective_axis_extent(&context, "parallel_all_gather", "x", 3)
                .unwrap_err(),
        );
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "collective axis extent must match the participant count".to_string(),
                observations: vec![
                    ("extent".to_string(), "2".to_string()),
                    ("participants".to_string(), "3".to_string()),
                ],
            }),
        );
    }

    #[test]
    fn test_dynamic_array_extent_batching_policy_collective_extent_constant() {
        let context = dynamic_collective_context(DimensionValue::constant(2).unwrap());
        let extent = DynamicArrayExtentBatchingPolicy::collective_extent_constant(&context, 0).unwrap();
        assert_eq!(extent.extent(), 0);
        assert_eq!(extent.r#type().extent(), Some(0));
        assert_eq!(extent.r#type().to_string(), "dimension<0>");
    }

    #[test]
    fn test_dynamic_array_extent_batching_policy_divide_extents_exactly() {
        let context = dynamic_collective_context(DimensionValue::constant(2).unwrap());

        // Exact extent types are checked on the host, and their quotient keeps an exact type.
        let quotient = DynamicArrayExtentBatchingPolicy::divide_extents_exactly(
            &context,
            &DimensionValue::constant(8).unwrap(),
            &DimensionValue::constant(2).unwrap(),
        )
        .unwrap();
        assert_eq!(quotient.extent(), 4);
        assert_eq!(quotient.r#type().extent(), Some(4));
        assert_eq!(
            DynamicArrayExtentBatchingPolicy::divide_extents_exactly(
                &context,
                &DimensionValue::constant(7).unwrap(),
                &DimensionValue::constant(2).unwrap(),
            ),
            Err(BatchingError::UnsupportedOperation { message: "extent 7 must be divisible by extent 2".to_string() }),
        );
        assert_eq!(
            DynamicArrayExtentBatchingPolicy::divide_extents_exactly(
                &context,
                &DimensionValue::constant(8).unwrap(),
                &DimensionValue::constant(0).unwrap(),
            ),
            Err(BatchingError::UnsupportedOperation { message: "extent 8 must be divisible by extent 0".to_string() }),
        );

        // Bounded extent types are checked by staged assertions, which the eager parent evaluates immediately: a
        // divisor whose lower bound is zero must be positive, and the dividend must be divisible by the divisor.
        let left_type = DimensionType::new("left", DimensionBounds::new(0, Some(17)).unwrap());
        let right_type = DimensionType::new("right", DimensionBounds::new(0, Some(9)).unwrap());
        let left = DimensionValue::new(left_type.clone(), 8).unwrap();
        let right = DimensionValue::new(right_type.clone(), 2).unwrap();
        assert_eq!(
            DynamicArrayExtentBatchingPolicy::divide_extents_exactly(&context, &left, &right).unwrap().extent(),
            4,
        );
        let zero = DimensionValue::new(right_type, 0).unwrap();
        let error = ProgramError::from(
            DynamicArrayExtentBatchingPolicy::divide_extents_exactly(&context, &left, &zero).unwrap_err(),
        );
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "collective divisor must be positive".to_string(),
                observations: vec![("divisor".to_string(), "0".to_string())],
            }),
        );
        let left = DimensionValue::new(left_type, 7).unwrap();
        let error = ProgramError::from(
            DynamicArrayExtentBatchingPolicy::divide_extents_exactly(&context, &left, &right).unwrap_err(),
        );
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "collective extent must be divisible by the participant count".to_string(),
                observations: vec![("extent".to_string(), "7".to_string()), ("divisor".to_string(), "2".to_string())],
            }),
        );
    }

    #[test]
    fn test_dynamic_array_extent_batching_policy_divide_extents_exactly_staging() {
        // A divisor whose lower bound is zero is asserted to be positive and clamped to one before the staged
        // divisibility assertion and division, so that the staged arithmetic never divides by zero.
        let left_type = DimensionType::new("extent", DimensionBounds::new(0, Some(17)).unwrap());
        let right_type = DimensionType::new("divisor", DimensionBounds::new(0, Some(9)).unwrap());
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let left = ValueProjection::<DimensionType>::into_projected(trace.input(left_type.clone().into())).unwrap();
        let right = ValueProjection::<DimensionType>::into_projected(trace.input(right_type.clone().into())).unwrap();
        let axis_extent = trace.dimension_constant(2).unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
            ProjectedContext::new(trace.clone()),
            axis_extent,
        );
        let quotient = DynamicArrayExtentBatchingPolicy::divide_extents_exactly(&context, &left, &right).unwrap();
        let quotient: Tracer<_> = ValueProjection::<DimensionType>::from_projected(quotient);
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![quotient.atom_id().unwrap()],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<extent ∈ [0, 17)>, %1:dimension<divisor ∈ [0, 9)> .
                let %2:dimension<2> = constant [value=2]
                    %3:dimension<0> = constant [value=0]
                    %4:dimension<1> = constant [value=1]
                    %5:bool[] = compare [direction=GreaterThan] %1 %3
                    () = assert [message=\"collective divisor must be positive\", labels=[\"divisor\"]] %5 %1
                    %6:dimension<max(divisor, 1) ∈ [1, 9)> = dimension_max %1 %4
                    %7:dimension<extent % max(divisor, 1) ∈ [0, 8)> = dimension_rem %0 %6
                    %8:bool[] = compare [direction=Equal] %7 %3
                    () = assert [
                        message=\"collective extent must be divisible by the participant count\",
                        labels=[\"extent\", \"divisor\"],
                    ] %8 %0 %1
                    %9:dimension<extent / max(divisor, 1) ∈ [0, 17)> = dimension_div %0 %6
                in (%9)"
            },
        );

        // Interpreting the staged program divides exact multiples and rejects indivisible extents and zero divisors.
        let outputs = program
            .interpret(vec![
                ArrayIrValue::Dimension(DimensionValue::new(left_type.clone(), 8).unwrap()),
                ArrayIrValue::Dimension(DimensionValue::new(right_type.clone(), 2).unwrap()),
            ])
            .unwrap();
        let quotient = ValueProjection::<DimensionType>::into_projected(outputs[0].clone()).unwrap();
        assert_eq!(quotient.extent(), 4);
        assert_eq!(quotient.r#type().bounds(), DimensionBounds::new(0, Some(17)).unwrap());
        let error = program
            .interpret(vec![
                ArrayIrValue::Dimension(DimensionValue::new(left_type.clone(), 7).unwrap()),
                ArrayIrValue::Dimension(DimensionValue::new(right_type.clone(), 2).unwrap()),
            ])
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "collective extent must be divisible by the participant count".to_string(),
                observations: vec![("extent".to_string(), "7".to_string()), ("divisor".to_string(), "2".to_string())],
            }),
        );
        let error = program
            .interpret(vec![
                ArrayIrValue::Dimension(DimensionValue::new(left_type, 8).unwrap()),
                ArrayIrValue::Dimension(DimensionValue::new(right_type, 0).unwrap()),
            ])
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "collective divisor must be positive".to_string(),
                observations: vec![("divisor".to_string(), "0".to_string())],
            }),
        );
    }

    #[test]
    fn test_dynamic_array_extent_batching_policy_divide_extents_exactly_staging_positive_divisor() {
        // A divisor whose lower bound is positive needs neither the positivity assertion nor the clamp, so only the
        // divisibility assertion is staged before the division.
        let left_type = DimensionType::new("extent", DimensionBounds::new(0, Some(17)).unwrap());
        let right_type = DimensionType::new("divisor", DimensionBounds::new(1, Some(9)).unwrap());
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let left = ValueProjection::<DimensionType>::into_projected(trace.input(left_type.clone().into())).unwrap();
        let right = ValueProjection::<DimensionType>::into_projected(trace.input(right_type.clone().into())).unwrap();
        let axis_extent = trace.dimension_constant(2).unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
            ProjectedContext::new(trace.clone()),
            axis_extent,
        );
        let quotient = DynamicArrayExtentBatchingPolicy::divide_extents_exactly(&context, &left, &right).unwrap();
        let quotient: Tracer<_> = ValueProjection::<DimensionType>::from_projected(quotient);
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![quotient.atom_id().unwrap()],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<extent ∈ [0, 17)>, %1:dimension<divisor ∈ [1, 9)> .
                let %2:dimension<2> = constant [value=2]
                    %3:dimension<0> = constant [value=0]
                    %4:dimension<extent % divisor ∈ [0, 8)> = dimension_rem %0 %1
                    %5:bool[] = compare [direction=Equal] %4 %3
                    () = assert [
                        message=\"collective extent must be divisible by the participant count\",
                        labels=[\"extent\", \"divisor\"],
                    ] %5 %0 %1
                    %6:dimension<extent / divisor ∈ [0, 17)> = dimension_div %0 %1
                in (%6)"
            },
        );

        // Interpreting the staged program divides exact multiples and rejects indivisible extents.
        let outputs = program
            .interpret(vec![
                ArrayIrValue::Dimension(DimensionValue::new(left_type.clone(), 8).unwrap()),
                ArrayIrValue::Dimension(DimensionValue::new(right_type.clone(), 2).unwrap()),
            ])
            .unwrap();
        let quotient = ValueProjection::<DimensionType>::into_projected(outputs[0].clone()).unwrap();
        assert_eq!(quotient.extent(), 4);
        assert_eq!(quotient.r#type().bounds(), DimensionBounds::new(0, Some(17)).unwrap());
        let error = program
            .interpret(vec![
                ArrayIrValue::Dimension(DimensionValue::new(left_type, 7).unwrap()),
                ArrayIrValue::Dimension(DimensionValue::new(right_type, 2).unwrap()),
            ])
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "collective extent must be divisible by the participant count".to_string(),
                observations: vec![("extent".to_string(), "7".to_string()), ("divisor".to_string(), "2".to_string())],
            }),
        );
    }

    #[test]
    fn test_dynamic_array_extent_batching_policy_match_collective_axis() {
        let context = dynamic_collective_context(DimensionValue::constant(2).unwrap());

        // A replicated batch is broadcast to a leading mapped axis using the explicit input extents.
        let replicated = ArrayBatch::replicated(Array::vector(vec![1f32, 2.0]).unwrap());
        assert_eq!(
            DynamicArrayExtentBatchingPolicy::match_collective_axis(
                &context,
                &replicated,
                &[DimensionValue::constant(2).unwrap()],
            ),
            Ok(ArrayBatch::new(Array::matrix(2, 2, vec![1f32, 2.0, 1.0, 2.0]).unwrap(), BatchAxis::new(0)).unwrap()),
        );

        // A mapped batch moves its mapped axis to the front.
        let mapped =
            ArrayBatch::new(Array::matrix(2, 2, vec![1f32, 2.0, 3.0, 4.0]).unwrap(), BatchAxis::new(1)).unwrap();
        assert_eq!(
            DynamicArrayExtentBatchingPolicy::match_collective_axis(
                &context,
                &mapped,
                &[DimensionValue::constant(2).unwrap()],
            ),
            Ok(ArrayBatch::new(Array::matrix(2, 2, vec![1f32, 3.0, 2.0, 4.0]).unwrap(), BatchAxis::new(0)).unwrap()),
        );
    }

    #[test]
    fn test_dynamic_array_extent_batching_policy_reshape_collective() {
        let context = dynamic_collective_context(DimensionValue::constant(2).unwrap());
        let input = Array::matrix(2, 3, vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(
            DynamicArrayExtentBatchingPolicy::reshape_collective(
                &context,
                input,
                &[DimensionValue::constant(3).unwrap(), DimensionValue::constant(2).unwrap()],
                None,
            ),
            Ok(Array::matrix(3, 2, vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()),
        );
    }

    #[test]
    fn test_validate_manual_mesh_axis() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();

        // The recorded axis size is checked only when the collective provides one.
        assert_eq!(validate_manual_mesh_axis("axis_index", "x", None, &mesh), Ok(()));
        assert_eq!(validate_manual_mesh_axis("axis_index", "x", Some(2), &mesh), Ok(()));
        assert_eq!(
            validate_manual_mesh_axis("axis_index", "x", Some(3), &mesh),
            Err(TypeError::invalid("`axis_index` axis size 3 does not match the size of manual mesh axis `x`")),
        );

        // Non-manual and missing mesh axes are both rejected as non-manual.
        assert_eq!(
            validate_manual_mesh_axis("axis_index", "y", None, &mesh),
            Err(TypeError::invalid("`axis_index` mesh axis `y` must be manual")),
        );
        assert_eq!(
            validate_manual_mesh_axis("axis_index", "z", None, &mesh),
            Err(TypeError::invalid("`axis_index` mesh axis `z` must be manual")),
        );
    }

    #[test]
    fn test_validate_manual_mesh_input() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let unsharded_type = ArrayType::new_static(DataType::F32, [4]);
        let sharded_type = unsharded_type.clone().with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap();
        assert_eq!(validate_manual_mesh_input("parallel_permute", "x", Some(2), &mesh, &sharded_type), Ok(()));

        // The mesh axis is validated before the input.
        assert_eq!(
            validate_manual_mesh_input("parallel_permute", "y", Some(2), &mesh, &unsharded_type),
            Err(TypeError::invalid("`parallel_permute` mesh axis `y` must be manual")),
        );

        // The input must carry sharding over the operation mesh.
        assert_eq!(
            validate_manual_mesh_input("parallel_permute", "x", Some(2), &mesh, &unsharded_type),
            Err(TypeError::invalid("`parallel_permute` input must carry a mesh containing manual axis `x`")),
        );
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let other_mesh_type = unsharded_type.with_sharding(Sharding::replicated(other_mesh, 1)).unwrap();
        assert_eq!(
            validate_manual_mesh_input("parallel_permute", "x", Some(2), &mesh, &other_mesh_type),
            Err(TypeError::invalid("`parallel_permute` input mesh does not match the operation mesh")),
        );
    }

    #[test]
    fn test_effective_collective_axis_size() {
        // Without groups, every participant along the axis takes part in one collective.
        assert_eq!(effective_collective_axis_size("parallel_all_gather", 4, None), Ok(4));

        // With groups, each group runs an independent collective, so the effective axis size is the common group size,
        // regardless of the order of the groups and of the participants within them.
        assert_eq!(
            effective_collective_axis_size("parallel_all_gather", 4, Some([vec![0, 2], vec![3, 1]].as_slice())),
            Ok(2),
        );

        // The requirements are checked in order: a positive axis size, at least one non-empty group, equal group
        // sizes, and an exact partition of the participants.
        assert_eq!(
            effective_collective_axis_size("parallel_all_gather", 0, None),
            Err(TypeError::invalid("`parallel_all_gather` axis size must be greater than zero")),
        );
        assert_eq!(
            effective_collective_axis_size("parallel_all_gather", 4, Some(Vec::<Vec<usize>>::new().as_slice())),
            Err(TypeError::invalid("`parallel_all_gather` axis index groups must not be empty")),
        );
        assert_eq!(
            effective_collective_axis_size("parallel_all_gather", 4, Some([Vec::new()].as_slice())),
            Err(TypeError::invalid("`parallel_all_gather` axis index groups must contain at least one participant")),
        );
        assert_eq!(
            effective_collective_axis_size("parallel_all_gather", 3, Some([vec![0, 1], vec![2]].as_slice())),
            Err(TypeError::invalid(
                "`parallel_all_gather` axis index group 1 has size 1 but every group must have size 2",
            )),
        );
        assert_eq!(
            effective_collective_axis_size("parallel_all_gather", 4, Some([vec![0, 1], vec![2, 4]].as_slice())),
            Err(TypeError::invalid("`parallel_all_gather` axis index 4 is out of bounds for axis size 4")),
        );
        assert_eq!(
            effective_collective_axis_size("parallel_all_gather", 4, Some([vec![0, 1], vec![1, 2]].as_slice())),
            Err(TypeError::invalid("`parallel_all_gather` axis index groups contain participant 1 more than once")),
        );
        assert_eq!(
            effective_collective_axis_size("parallel_all_gather", 3, Some([vec![0, 1]].as_slice())),
            Err(TypeError::invalid("`parallel_all_gather` axis index groups do not contain participant 2")),
        );
    }

    #[test]
    fn test_resolve_named_axis_size() {
        // A batching level resolves the axis that it binds to its static extent, while other names stay unbound.
        let context = eager_collective_context("x", 2);
        assert_eq!(resolve_named_axis_size(&context, "x"), Ok(2));
        assert_eq!(
            resolve_named_axis_size(&context, "y"),
            Err(ProgramError::Axis(AxisError::UnboundAxisName { name: "y".to_string() })),
        );

        // An axis without participants is rejected before any collective divides by its size.
        let context = eager_collective_context("x", 0);
        assert_eq!(
            resolve_named_axis_size(&context, "x"),
            Err(ProgramError::Type(TypeError::invalid("collective axis `x` must contain at least one participant"))),
        );

        // A traced batch extent without exact bounds has no static size that a collective payload could record.
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let axis_extent = trace.input(DimensionType::new("n", DimensionBounds::new(1, Some(9)).unwrap()).into());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::with_policy(trace, axis_extent)
            .with_axis_name("x".to_string());
        let error = resolve_named_axis_size(&context, "x").unwrap_err();
        assert_eq!(
            error.downcast_custom::<BatchingError>(),
            Some(&BatchingError::UnsupportedOperation {
                message: "collective axis `x` has a dynamic extent that must remain a first-class input".to_string(),
            }),
        );
    }

    #[test]
    fn test_infer_linear_collective_operation_output_type() {
        // An unchanged shape preserves the complete input type, while a resized shape drops the explicit layout and
        // preserves the element type and memory.
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

        // A resized dimension keeps its sharding when its new size stays divisible by its explicit mesh axes.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding =
            Sharding::new(mesh, vec![ShardingDimension::sharded(["y"]), ShardingDimension::Replicated]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2, 3]).with_sharding(sharding.clone()).unwrap();
        assert_eq!(
            infer_linear_collective_operation_output_type("parallel_all_gather", &input_type, vec![4, 3]),
            Ok(ArrayType::new_static(DataType::F32, [4, 3]).with_sharding(sharding).unwrap()),
        );
        assert_eq!(
            infer_linear_collective_operation_output_type("parallel_all_gather", &input_type, vec![3, 3]),
            Err(TypeError::invalid(
                "`parallel_all_gather` on a dimension sharded over explicit mesh axes requires the output size (3) at \
                 axis 0 to be divisible by the mesh-axis product (2)",
            )),
        );
    }

    #[test]
    fn test_infer_array_ir_shape_changing_collective_output_type() {
        let array_type = ArrayIrType::from(ArrayType::new_static(DataType::F32, [2, 3]));
        let dynamic_extent_type = DimensionType::new("n", DimensionBounds::new(0, Some(9)).unwrap());
        let extent_two = ArrayIrType::from(DimensionValue::constant(2).unwrap().r#type().into_owned());
        let extent_three = ArrayIrType::from(DimensionValue::constant(3).unwrap().r#type().into_owned());
        let extent_four = ArrayIrType::from(DimensionValue::constant(4).unwrap().r#type().into_owned());
        let base_output_type = ArrayType::new_static(DataType::F32, [4, 3]);

        // The explicit extents replace the shape of the base output type, so a changed axis may become dynamic.
        assert_eq!(
            infer_array_ir_shape_changing_collective_output_type(
                "parallel_all_gather",
                &[array_type.clone(), ArrayIrType::from(dynamic_extent_type.clone()), extent_three.clone()],
                base_output_type.clone(),
                &[0],
                |_| Ok(()),
            ),
            Ok(vec![ArrayIrType::from(ArrayType::new(
                DataType::F32,
                Shape::new(vec![dynamic_extent_type.to_dimension(), Dimension::Static(3)]),
            ))]),
        );

        // The collective-specific validation receives every explicit output extent, and its errors propagate.
        assert_eq!(
            infer_array_ir_shape_changing_collective_output_type(
                "parallel_all_gather",
                &[array_type.clone(), extent_four.clone(), extent_three.clone()],
                base_output_type.clone(),
                &[0],
                |extents| Err(TypeError::invalid(format!("rejected {} extents", extents.len()))),
            ),
            Err(TypeError::invalid("rejected 2 extents")),
        );

        // The inputs must be one array followed by one dimension per output axis.
        assert_eq!(
            infer_array_ir_shape_changing_collective_output_type(
                "parallel_all_gather",
                &[array_type.clone(), extent_four.clone()],
                base_output_type.clone(),
                &[0],
                |_| Ok(()),
            ),
            Err(TypeError::invalid("expected 3 inputs but got 2")),
        );
        assert_eq!(
            infer_array_ir_shape_changing_collective_output_type(
                "parallel_all_gather",
                &[extent_three.clone(), extent_four.clone(), extent_three.clone()],
                base_output_type.clone(),
                &[0],
                |_| Ok(()),
            ),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
        assert_eq!(
            infer_array_ir_shape_changing_collective_output_type(
                "parallel_all_gather",
                &[array_type.clone(), array_type.clone(), extent_three],
                base_output_type.clone(),
                &[0],
                |_| Ok(()),
            ),
            Err(TypeError::invalid("expected dimension type but got array type")),
        );

        // Output axes that the collective does not change must keep the extents of the base output type.
        assert_eq!(
            infer_array_ir_shape_changing_collective_output_type(
                "parallel_all_gather",
                &[array_type, extent_four, extent_two],
                base_output_type,
                &[0],
                |_| Ok(()),
            ),
            Err(TypeError::invalid("`parallel_all_gather` output axis 1 extent 2 must equal unchanged extent 3")),
        );
    }

    #[test]
    fn test_batching_context_forward_collective() {
        // A level that does not bind the collective's axis forwards the collective to its eager parent, which can
        // interpret this single-participant all-gather locally.
        let context = eager_collective_context("outer", 2);
        let operation = ParallelAllGatherOperation::new(
            "inner".to_string(),
            1,
            0,
            CollectiveOptions::default(),
            ParallelAllGatherOutputVariance::Varying,
        );

        // A replicated input forwards the collective unchanged, and its result stays replicated.
        let input = ArrayBatch::replicated(Array::vector(vec![1f32, 2.0]).unwrap());
        assert_eq!(
            context.forward_collective(&operation, &[input]).unwrap().into_parts().0,
            vec![ArrayBatch::replicated(Array::matrix(1, 2, vec![1f32, 2.0]).unwrap())],
        );

        // A mapped input forwards the collective adapted to the mapped axis, so the untiled all-gather inserts its
        // gathered axis in front of the mapped axis, which moves one position to the right.
        let input = ArrayBatch::new(Array::vector(vec![1f32, 2.0]).unwrap(), BatchAxis::new(0)).unwrap();
        assert_eq!(
            context.forward_collective(&operation, &[input]).unwrap().into_parts().0,
            vec![ArrayBatch::new(Array::matrix(1, 2, vec![1f32, 2.0]).unwrap(), BatchAxis::new(1)).unwrap()],
        );
    }

    #[test]
    fn test_batching_context_infer_collective_output_type_and_extents() {
        let context = eager_collective_context("x", 2);
        let operation = ParallelAllGatherOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::default(),
            ParallelAllGatherOutputVariance::Varying,
        );
        assert_eq!(
            context.infer_collective_output_type_and_extents(&operation, &ArrayType::new_static(DataType::F32, [3])),
            Ok((ArrayType::new_static(DataType::F32, [2, 3]), vec![2, 3])),
        );
    }

    #[test]
    fn test_batching_context_forward_collective_array_ir() {
        let all_gather = ArrayIrOperation::<Array>::ParallelAllGather(ParallelAllGatherOperation::new(
            "inner".to_string(),
            2,
            0,
            CollectiveOptions::default(),
            ParallelAllGatherOutputVariance::Varying,
        ));
        let sum_scatter = ArrayIrOperation::<Array>::ParallelSumScatter(ParallelSumScatterOperation::new(
            "inner".to_string(),
            2,
            0,
            CollectiveOptions::default(),
        ));
        let all_to_all = ArrayIrOperation::<Array>::ParallelAllToAll(ParallelAllToAllOperation::new(
            "inner".to_string(),
            2,
            0,
            1,
            CollectiveOptions::default(),
        ));

        // A replicated array forwards the all-gather unchanged, and its result stays replicated.
        assert_eq!(
            forward_array_ir_collective(all_gather.clone(), vec![2, 3], BatchAxis::replicated(), &[2, 2, 3]),
            (
                BatchAxis::replicated(),
                ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2, 2, 3])),
                indoc! {"
                    lambda %0:dimension<5>, %1:f32[2, 3], %2:dimension<2>, %3:dimension<2>, %4:dimension<3> .
                    let %5:f32[2, 2, 3] = parallel_all_gather [
                        axis_name=\"inner\",
                        axis_size=2,
                        concatenation_axis=0,
                        options=Untiled,
                        output_variance=Varying,
                    ] %1 %2 %3 %4
                    in (%5)"
                }
                .to_string(),
            ),
        );

        // A mapped array forwards the all-gather adapted to the mapped axis. Its gathered axis is inserted in front of
        // the mapped axis, which moves one position to the right, and the batch extent joins the result extents at
        // the mapped axis position of the result.
        assert_eq!(
            forward_array_ir_collective(all_gather, vec![2, 5, 3], BatchAxis::new(1), &[2, 2, 3]),
            (
                BatchAxis::new(2),
                ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2, 2, 5, 3])),
                indoc! {"
                    lambda %0:dimension<5>, %1:f32[2, 5, 3], %2:dimension<2>, %3:dimension<2>, %4:dimension<3> .
                    let %5:f32[2, 2, 5, 3] = parallel_all_gather [
                        axis_name=\"inner\",
                        axis_size=2,
                        concatenation_axis=0,
                        options=Untiled,
                        output_variance=Varying,
                    ] %1 %2 %3 %0 %4
                    in (%5)"
                }
                .to_string(),
            ),
        );

        // A replicated array forwards the sum-scatter unchanged, and its result stays replicated.
        assert_eq!(
            forward_array_ir_collective(sum_scatter.clone(), vec![2, 3], BatchAxis::replicated(), &[3]),
            (
                BatchAxis::replicated(),
                ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3])),
                indoc! {"
                    lambda %0:dimension<5>, %1:f32[2, 3], %2:dimension<3> .
                    let %3:f32[3] = parallel_sum_scatter [\
                            axis_name=\"inner\", \
                            axis_size=2, \
                            scatter_axis=0, \
                            options=Untiled\
                        ] %1 %2
                    in (%3)"
                }
                .to_string(),
            ),
        );

        // The forwarded sum-scatter consumes its scatter axis in front of the mapped axis, which moves one position to
        // the left.
        assert_eq!(
            forward_array_ir_collective(sum_scatter, vec![2, 5, 3], BatchAxis::new(1), &[3]),
            (
                BatchAxis::new(0),
                ArrayIrType::Array(ArrayType::new_static(DataType::F32, [5, 3])),
                indoc! {"
                    lambda %0:dimension<5>, %1:f32[2, 5, 3], %2:dimension<3> .
                    let %3:f32[5, 3] = parallel_sum_scatter [\
                            axis_name=\"inner\", \
                            axis_size=2, \
                            scatter_axis=0, \
                            options=Untiled\
                        ] %1 %0 %2
                    in (%3)"
                }
                .to_string(),
            ),
        );

        // A replicated array forwards the all-to-all unchanged, and its result stays replicated.
        assert_eq!(
            forward_array_ir_collective(all_to_all.clone(), vec![2, 3], BatchAxis::replicated(), &[3, 2]),
            (
                BatchAxis::replicated(),
                ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3, 2])),
                indoc! {"
                    lambda %0:dimension<5>, %1:f32[2, 3], %2:dimension<3>, %3:dimension<2> .
                    let %4:f32[3, 2] = parallel_all_to_all [
                        axis_name=\"inner\",
                        axis_size=2,
                        split_axis=0,
                        concatenation_axis=1,
                        options=Untiled,
                    ] %1 %2 %3
                    in (%4)"
                }
                .to_string(),
            ),
        );

        // The forwarded all-to-all consumes its split axis in front of the mapped axis, which moves to the front, and
        // its concatenation axis shifts past the mapped axis.
        assert_eq!(
            forward_array_ir_collective(all_to_all, vec![2, 5, 3], BatchAxis::new(1), &[3, 2]),
            (
                BatchAxis::new(0),
                ArrayIrType::Array(ArrayType::new_static(DataType::F32, [5, 3, 2])),
                indoc! {"
                    lambda %0:dimension<5>, %1:f32[2, 5, 3], %2:dimension<3>, %3:dimension<2> .
                    let %4:f32[5, 3, 2] = parallel_all_to_all [
                        axis_name=\"inner\",
                        axis_size=2,
                        split_axis=0,
                        concatenation_axis=2,
                        options=Untiled,
                    ] %1 %0 %2 %3
                    in (%4)"
                }
                .to_string(),
            ),
        );
    }

    #[test]
    fn test_batching_context_array_projection() {
        // The projection keeps the axis name, extent, and sharding of the composite level.
        let axis_extent = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::with_policy(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            axis_extent.clone(),
        )
        .with_axis_name("x".to_string())
        .with_axis_sharding(ShardingDimension::sharded(["devices"]));
        let projection = context.array_projection();
        assert_eq!(projection.axis_name(), Some("x"));
        assert_eq!(projection.axis_extent(), &axis_extent);
        assert_eq!(projection.axis_sharding(), &ShardingDimension::sharded(["devices"]));
    }
}
