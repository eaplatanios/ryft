use std::collections::BTreeSet;
use std::fmt::Display;

use crate::batching::{
    ArrayBatch, BatchAxis, BatchableOperation, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    TransposableOperation, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::constants::{Zero, ZeroOperation};
use crate::operations::manipulation::{Broadcast, Reshape, Slice, Transpose, UpdateSlice};
use crate::operations::sharding::Reshard;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::ProgramError;
use crate::programs::atoms::MaybeZero;
use crate::programs::identities::TypeIdentityRenaming;
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{TypeError, Typed};
use crate::programs::values::Value;
use crate::sharding::{LogicalMesh, MeshAxisType, Sharding, ShardingDimension};
use crate::tracing::{Tracer, TracingContext};
use crate::tracing_v2::custom_derivatives::CustomVjpResidual;
use crate::types::{ArrayType, Dimension, Shape};

// TODO(eaplatanios): Review this.

use super::scattering::{LinearScatterAddOperation, ScatterDimensionNumbers, ScatterOperation, ScatterReductionKind};
use super::slicing::batch_by_item_expansion;

/// Canonical operation name for [`GatherOperation`].
pub const GATHER_OPERATION_NAME: &str = "gather";

/// Out-of-bounds index handling for [`gather`](Gather) and [`scatter`](super::scattering::Scatter). The mode does not
/// affect the output [`Type`](crate::programs::types::Type)—only how a start index that would read or write outside
/// the operand is treated at execution time. It is shared by both operations; the scatter combiner kind lives in
/// [`super::scattering`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum GatherScatterMode {
    /// The caller promises every index is in bounds; out-of-bounds behavior is undefined (and gradients are wrong if
    /// the promise is violated). This is the default and lowers directly to the bare StableHLO operation.
    #[default]
    PromiseInBounds,

    /// Each start index is clamped so the whole window stays in bounds.
    Clip,

    /// A window that would fall partly out of bounds is dropped: gather fills it with zeros of the operand data type,
    /// while scatter discards the update.
    FillOrDrop,
}

impl GatherScatterMode {
    /// Returns the canonical lowercase name of this mode.
    pub fn name(self) -> &'static str {
        match self {
            Self::PromiseInBounds => "promise_in_bounds",
            Self::Clip => "clip",
            Self::FillOrDrop => "fill_or_drop",
        }
    }
}

impl Display for GatherScatterMode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.name())
    }
}

/// Specification of how the index operand and the sliced windows map onto the operand and output axes of a
/// [`gather`](Gather), following StableHLO's [`gather`](https://openxla.org/stablehlo/spec#gather) dimension numbers.
///
/// The index vector dimension is implicit and always the last axis of the indices operand (JAX's convention): the
/// indices operand has shape `[batch..., index_vector]`, where each length-`index_vector` slice is one start-index
/// vector whose components map onto operand axes through [`start_index_map`](Self::start_index_map). To gather with a
/// scalar index per query, give the indices a trailing size-1 axis.
///
/// The output rank is `offset_dimensions.len() + indices.rank() - 1`. Each output axis named in
/// [`offset_dimensions`](Self::offset_dimensions) carries one sliced window axis (in operand-axis order, skipping the
/// collapsed and batching axes); the remaining output axes carry the indices' batch axes in order.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct GatherDimensionNumbers {
    /// Output axes that hold the sliced window (the "offset" axes), in ascending order. Their count equals the number
    /// of operand axes that are neither collapsed nor batching.
    offset_dimensions: Vec<usize>,

    /// Operand axes whose slice size is `1` and that are removed from the output, in ascending order.
    collapsed_slice_dimensions: Vec<usize>,

    /// For each component of a start-index vector (the last axis of the indices operand), the operand axis it indexes
    /// into. Its length equals the extent of the indices' index vector dimension.
    start_index_map: Vec<usize>,

    /// Operand axes that are batched against [`start_indices_batching_dimensions`](Self::start_indices_batching_dimensions),
    /// aligned 1:1, in ascending order. Each has slice size at most `1`.
    operand_batching_dimensions: Vec<usize>,

    /// Indices axes (other than the index vector dimension) that align 1:1 with
    /// [`operand_batching_dimensions`](Self::operand_batching_dimensions).
    start_indices_batching_dimensions: Vec<usize>,
}

impl GatherDimensionNumbers {
    /// Creates gather dimension numbers from explicit axis lists. The batching axis lists default to empty; use
    /// [`with_batching_dimensions`](Self::with_batching_dimensions) to set them.
    #[inline]
    pub fn new(
        offset_dimensions: Vec<usize>,
        collapsed_slice_dimensions: Vec<usize>,
        start_index_map: Vec<usize>,
    ) -> Self {
        Self {
            offset_dimensions,
            collapsed_slice_dimensions,
            start_index_map,
            operand_batching_dimensions: Vec::new(),
            start_indices_batching_dimensions: Vec::new(),
        }
    }

    /// Attaches the operand/indices batching axis pair (aligned 1:1).
    #[inline]
    pub fn with_batching_dimensions(
        mut self,
        operand_batching_dimensions: Vec<usize>,
        start_indices_batching_dimensions: Vec<usize>,
    ) -> Self {
        self.operand_batching_dimensions = operand_batching_dimensions;
        self.start_indices_batching_dimensions = start_indices_batching_dimensions;
        self
    }

    /// Returns the output offset axes.
    #[inline]
    pub fn offset_dimensions(&self) -> &[usize] {
        &self.offset_dimensions
    }

    /// Returns the collapsed (size-1, removed) operand axes.
    #[inline]
    pub fn collapsed_slice_dimensions(&self) -> &[usize] {
        &self.collapsed_slice_dimensions
    }

    /// Returns the start-index-to-operand-axis map.
    #[inline]
    pub fn start_index_map(&self) -> &[usize] {
        &self.start_index_map
    }

    /// Returns the operand batching axes.
    #[inline]
    pub fn operand_batching_dimensions(&self) -> &[usize] {
        &self.operand_batching_dimensions
    }

    /// Returns the indices batching axes.
    #[inline]
    pub fn start_indices_batching_dimensions(&self) -> &[usize] {
        &self.start_indices_batching_dimensions
    }
}

impl Display for GatherDimensionNumbers {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "(offset={:?}, collapsed_slice={:?}, start_index_map={:?}, operand_batching={:?}, \
             start_indices_batching={:?})",
            self.offset_dimensions,
            self.collapsed_slice_dimensions,
            self.start_index_map,
            self.operand_batching_dimensions,
            self.start_indices_batching_dimensions,
        )
    }
}

/// [`Operation`] that reads slices ("windows") out of an operand at positions named by an integer index operand,
/// assembling them into a new array. Refer to the documentation of [`Gather`] for the full semantics.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GatherOperation {
    /// Dimension numbers mapping the index operand and sliced windows onto the operand and output axes.
    dimensions: GatherDimensionNumbers,

    /// Dimension of the sliced window along each operand axis (length equals the operand rank).
    slice_sizes: Vec<usize>,

    /// Out-of-bounds index handling.
    mode: GatherScatterMode,

    /// Whether the caller guarantees the index vectors are sorted (a lowering hint only).
    indices_are_sorted: bool,

    /// Whether the caller guarantees the gathered windows do not overlap (a lowering hint only).
    unique_indices: bool,

    /// Optional requested output [`Sharding`], used when the inferred placement is ambiguous (see
    /// [`Self::with_output_sharding`]).
    output_sharding: Option<Sharding>,
}

impl GatherOperation {
    /// Creates a new [`GatherOperation`] with the provided dimension numbers and per-operand-axis slice sizes. The
    /// mode defaults to [`GatherScatterMode::PromiseInBounds`] and both index hints default to `false`; use the
    /// chained `with_*` builders to override them.
    #[inline]
    pub fn new(dimensions: GatherDimensionNumbers, slice_sizes: Vec<usize>) -> Self {
        Self {
            dimensions,
            slice_sizes,
            mode: GatherScatterMode::PromiseInBounds,
            indices_are_sorted: false,
            unique_indices: false,
            output_sharding: None,
        }
    }

    /// Sets the out-of-bounds index handling mode.
    #[inline]
    pub fn with_mode(mut self, mode: GatherScatterMode) -> Self {
        self.mode = mode;
        self
    }

    /// Sets the sorted-indices lowering hint.
    #[inline]
    pub fn with_indices_are_sorted(mut self, indices_are_sorted: bool) -> Self {
        self.indices_are_sorted = indices_are_sorted;
        self
    }

    /// Sets the unique-indices lowering hint.
    #[inline]
    pub fn with_unique_indices(mut self, unique_indices: bool) -> Self {
        self.unique_indices = unique_indices;
        self
    }

    /// Requests `output_sharding` for the result. The gather sharding rule replicates the operand axes named by
    /// [`GatherDimensionNumbers::start_index_map`] (and the index vector axis); when that leaves the output placement
    /// ambiguous — for example because a sliced operand axis is sharded over an explicit mesh axis — a requested
    /// output sharding resolves it, bypassing inference. This mirrors `dot`/`reduce`'s `with_output_sharding`.
    #[inline]
    pub fn with_output_sharding(mut self, output_sharding: impl Into<Option<Sharding>>) -> Self {
        self.output_sharding = output_sharding.into();
        self
    }

    /// Returns the dimension numbers.
    #[inline]
    pub fn dimensions(&self) -> &GatherDimensionNumbers {
        &self.dimensions
    }

    /// Returns the per-operand-axis slice sizes.
    #[inline]
    pub fn slice_sizes(&self) -> &[usize] {
        &self.slice_sizes
    }

    /// Returns the out-of-bounds index handling mode.
    #[inline]
    pub fn mode(&self) -> GatherScatterMode {
        self.mode
    }

    /// Returns the sorted-indices hint.
    #[inline]
    pub fn indices_are_sorted(&self) -> bool {
        self.indices_are_sorted
    }

    /// Returns the unique-indices hint.
    #[inline]
    pub fn unique_indices(&self) -> bool {
        self.unique_indices
    }

    /// Returns the requested output sharding, if any.
    #[inline]
    pub fn output_sharding(&self) -> Option<&Sharding> {
        self.output_sharding.as_ref()
    }
}

impl Display for GatherOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation<ArrayType> for GatherOperation {
    #[inline]
    fn name(&self) -> &'static str {
        GATHER_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        match input_types[0].gather(&input_types[1], self) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError::invalid(error.to_string())),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("dimensions", &self.dimensions)?;
            operation.field("slice_sizes", format_args!("{:?}", self.slice_sizes))?;
            if self.mode != GatherScatterMode::PromiseInBounds {
                operation.field("mode", self.mode)?;
            }
            if self.indices_are_sorted {
                operation.field("indices_are_sorted", self.indices_are_sorted)?;
            }
            if self.unique_indices {
                operation.field("unique_indices", self.unique_indices)?;
            }
            if let Some(output_sharding) = &self.output_sharding {
                operation.field("output_sharding", output_sharding)?;
            }
            Ok(())
        })
    }
}

impl<C: Domain<Type = ArrayType, Value: Gather>> InterpretableOperation<C> for GatherOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![inputs[0].gather(&inputs[1], self)?])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for GatherOperation where
    C::Operation: From<GatherOperation>
{
}

/// Forward-mode rule for [`GatherOperation`]: `gather` is linear in the data operand, and the index operand is a
/// non-differentiated primal operand edge, so the tangent gathers the operand tangent at the same primal indices. A
/// zero operand tangent yields a typed zero output tangent.
impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for GatherOperation
where
    C::Operation: From<GatherOperation>,
    C::Value: Gather,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        let indices = inputs[1].primal();
        let primal = inputs[0].primal().gather(indices, self)?;
        let tangent = match inputs[0].tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()),
            MaybeZero::Value(tangent) => MaybeZero::Value(tangent.gather(indices, self)?),
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

/// Batching rule for [`GatherOperation`]. A gather mixes window reads, collapsed axes, and index-driven offsets whose
/// axis bookkeeping does not compose cleanly with an extra mapped axis, so any batched operand, indices, or both is
/// handled by per-item expansion (`batch_by_item_expansion`): each batch item gathers independently and the results
/// restack along a fresh leading batch axis. This stages `O(axis_size)` gathers but is correct for every
/// dimension-number configuration; dimension-number lifting (one lifted gather, no expansion) is a performance
/// optimization left as a follow-up. When no input is mapped the gather applies once, unbatched.
impl<C> BatchableOperation<C> for GatherOperation
where
    C: Context<Type = ArrayType> + Zero<C::Value>,
    C::Value: Broadcast + Transpose + Slice + UpdateSlice + Reshape + Reshard,
    GatherOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        check_count!("input", inputs, 2, ProgramError);
        let Some(axis_size) = ArrayBatch::common_batch_size(inputs)? else {
            return self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()]);
        };
        batch_by_item_expansion(context, GATHER_OPERATION_NAME, self, inputs, axis_size)
    }
}

/// Returns whether `dimension` is sharded over at least one explicit mesh axis of `mesh` (the explicit-axis gate used
/// by the dot/reduce/slice sharding rules). Shared with [`super::scattering`].
pub(crate) fn dimension_has_explicit_axis(mesh: &LogicalMesh, dimension: &ShardingDimension) -> bool {
    matches!(dimension, ShardingDimension::Sharded(axis_names)
        if axis_names.iter().any(|name| mesh.axis_type(name) == Some(MeshAxisType::Explicit)))
}

/// Returns whether `sharding` references any auto mesh axis in a placement or reduction-state set. Shared with
/// [`super::scattering`].
pub(crate) fn references_auto_axis(sharding: &Sharding) -> bool {
    let placement_auto = sharding.dimensions().iter().any(|dimension| {
        matches!(dimension, ShardingDimension::Sharded(axis_names)
            if axis_names.iter().any(|name| sharding.mesh().axis_type(name) == Some(MeshAxisType::Auto)))
    });
    let set_auto =
        |axes: &BTreeSet<String>| axes.iter().any(|name| sharding.mesh().axis_type(name) == Some(MeshAxisType::Auto));
    placement_auto || set_auto(sharding.unreduced_axes()) || set_auto(sharding.reduced_axes())
}

/// Resolves the common mesh of two optional shardings, erroring on a mesh mismatch. Returns `None` when neither side
/// is sharded.
fn resolve_mesh(
    operand_sharding: Option<&Sharding>,
    indices_sharding: Option<&Sharding>,
) -> Result<Option<LogicalMesh>, TypeError> {
    match (operand_sharding, indices_sharding) {
        (None, None) => Ok(None),
        (Some(left), Some(right)) => {
            if left.mesh() != right.mesh() {
                return Err(TypeError::invalid(format!(
                    "'{GATHER_OPERATION_NAME}' operand and indices shardings must use the same mesh"
                )));
            }
            Ok(Some(left.mesh().clone()))
        }
        (Some(left), None) => Ok(Some(left.mesh().clone())),
        (None, Some(right)) => Ok(Some(right.mesh().clone())),
    }
}

/// Validates that `axes` is strictly ascending (sorted and unique) and that every entry is in `0..bound`. Shared with
/// [`super::scattering`].
pub(crate) fn validate_sorted_unique_in_range(
    operation_name: &'static str,
    field: &str,
    axes: &[usize],
    bound: usize,
) -> Result<(), TypeError> {
    for window in axes.windows(2) {
        if window[0] >= window[1] {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' {field} must be sorted and unique but got {axes:?}"
            )));
        }
    }
    if let Some(&axis) = axes.iter().find(|&&axis| axis >= bound) {
        return Err(TypeError::invalid(format!(
            "'{operation_name}' {field} entry {axis} is out of range for bound {bound}"
        )));
    }
    Ok(())
}

/// Validates that every entry of `axes` is unique and in `0..bound` (order not required). Shared with
/// [`super::scattering`].
pub(crate) fn validate_unique_in_range(
    operation_name: &'static str,
    field: &str,
    axes: &[usize],
    bound: usize,
) -> Result<(), TypeError> {
    let mut seen = BTreeSet::new();
    for &axis in axes {
        if axis >= bound {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' {field} entry {axis} is out of range for bound {bound}"
            )));
        }
        if !seen.insert(axis) {
            return Err(TypeError::invalid(format!("'{operation_name}' {field} must be unique but got {axes:?}")));
        }
    }
    Ok(())
}

// TODO(eaplatanios): Should this be renamed to something that's not about "linearity"? This is about captured primals.
/// Captured-index gather linear operation: the linear map `t ↦ gather(t, indices; dimensions)` over the tangent (or
/// cotangent) of the gathered operand.
///
/// It is the captured-index linear map emitted by the JVP of [`GatherOperation`]: the
/// integer index operand is a primal value captured at linearization time as a residual factor (it has no tangent
/// space, so the map is linear in the single tangent operand), and its transpose is the dual scatter-add. The single
/// operation input is the gathered operand's tangent; the captured `indices` factor supplies the gather's index
/// operand during type inference.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearGatherOperation<F> {
    /// Underlying [`GatherOperation`] describing the gather geometry.
    operation: GatherOperation,

    /// Captured integer index operand factor.
    indices: F,
}

impl<F> LinearGatherOperation<F> {
    /// Creates a new [`LinearGatherOperation`] from the underlying gather and the captured index factor.
    #[inline]
    pub fn new(operation: GatherOperation, indices: F) -> Self {
        Self { operation, indices }
    }

    /// Returns the underlying [`GatherOperation`] describing the gather geometry.
    #[inline]
    pub fn operation(&self) -> &GatherOperation {
        &self.operation
    }

    /// Returns the captured integer index operand factor.
    #[inline]
    pub fn indices(&self) -> &F {
        &self.indices
    }
}

impl<F: Value<Type = ArrayType>> Display for LinearGatherOperation<F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<F: Value<Type = ArrayType>> Operation<ArrayType> for LinearGatherOperation<F> {
    #[inline]
    fn name(&self) -> &'static str {
        GATHER_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        self.operation
            .infer_output_types(&[input_types[0].clone(), self.indices.r#type().into_owned()], &[])
    }

    #[inline]
    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<ArrayType as crate::Type>::Identity>,
    ) -> Result<Self, TypeError> {
        Ok(Self { operation: self.operation.clone(), indices: self.indices.rename_type_identities(renaming)? })
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        let _ = indentation;
        formatter.write_str(self.name())
    }
}

impl<F, C> InterpretableOperation<C> for LinearGatherOperation<F>
where
    C: Domain<Type = ArrayType, Value: Gather>,
    F: CustomVjpResidual<C::Value>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].gather(&self.indices().residual_value()?, self.operation())?])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate) for a [`LinearGatherOperation`].
impl<F: Value<Type = ArrayType>, C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C>
    for LinearGatherOperation<F>
where
    C::Operation: From<LinearGatherOperation<F>>,
{
}

/// Transpose rule for the captured-index gather. The forward linear map
/// `t ↦ gather(t, indices)` has, as its adjoint, the dual scatter-add that writes the output cotangent back into a
/// zero operand at the gathered windows: the scatter geometry mirrors the gather axis-for-axis and the captured
/// indices carry over. Symbolic-zero cotangents propagate unchanged.
impl<V: Value<Type = ArrayType>, O, F: Value<Type = ArrayType>> TransposableOperation<V, O> for LinearGatherOperation<F>
where
    O: Operation<ArrayType> + From<ZeroOperation<ArrayType>> + From<LinearScatterAddOperation<F>>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        match &outputs[0] {
            MaybeZero::Zero(_) => Ok(vec![MaybeZero::Zero(inputs[0].r#type().cotangent())]),
            MaybeZero::Value(cotangent) => {
                let zeros = MaybeZero::Zero(inputs[0].r#type().cotangent()).materialize(context)?;
                let dimensions = self.operation().dimensions();
                let scatter_dimensions = ScatterDimensionNumbers::new(
                    dimensions.offset_dimensions().to_vec(),
                    dimensions.collapsed_slice_dimensions().to_vec(),
                    dimensions.start_index_map().to_vec(),
                )
                .with_batching_dimensions(
                    dimensions.operand_batching_dimensions().to_vec(),
                    dimensions.start_indices_batching_dimensions().to_vec(),
                );
                let scatter_operation = ScatterOperation::new(scatter_dimensions, ScatterReductionKind::Add)
                    .with_mode(self.operation().mode())
                    .with_indices_are_sorted(self.operation().indices_are_sorted())
                    .with_unique_indices(self.operation().unique_indices());
                let outputs = context.stage_operation(
                    LinearScatterAddOperation::new(scatter_operation, self.indices().clone()),
                    Vec::new(),
                    &[zeros, cotangent.clone()],
                )?;
                check_count!("output", outputs, 1, ProgramError);
                Ok(vec![MaybeZero::Value(outputs.into_iter().next().unwrap())])
            }
        }
    }
}

/// Partition-aware transpose rule for the primal [`GatherOperation`]. The integer index operand (operand 1) has no
/// tangent space, so in a valid pushforward it is the known operand and the gathered operand (operand 0) is the
/// linear one. The forward map `t ↦ gather(t, indices)` has, as its adjoint, the dual scatter-add that writes the
/// output cotangent back into a zero operand at the gathered windows: the scatter geometry mirrors the gather
/// axis-for-axis. This reproduces the captured-index [`LinearGatherOperation`] transpose rule, reading the indices
/// from the pullback through `operand_values` and staging a primal additive [`ScatterOperation`] instead of folding
/// the indices into a captured factor. The indices receive a structural zero, and a zero output cotangent stays a
/// structural zero.
impl<V: Value<Type = ArrayType>, O> TransposableOperation<V, O> for GatherOperation
where
    O: Operation<ArrayType> + From<ZeroOperation<ArrayType>> + From<ScatterOperation>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        match &outputs[0] {
            MaybeZero::Zero(_) => Ok(vec![
                MaybeZero::Zero(inputs[0].r#type().cotangent()),
                MaybeZero::Zero(inputs[1].r#type().cotangent()),
            ]),
            MaybeZero::Value(cotangent) => {
                // The indices are the known operand; the dispatch guarantees a `Known` operand carries its pullback
                // value, so read the tracer directly.
                let indices = inputs[1]
                    .as_known()
                    .expect("dispatch guarantees a known operand carries its pullback value")
                    .clone();
                let zeros = MaybeZero::Zero(inputs[0].r#type().cotangent()).materialize(context)?;
                let scatter_dimensions = ScatterDimensionNumbers::new(
                    self.dimensions().offset_dimensions().to_vec(),
                    self.dimensions().collapsed_slice_dimensions().to_vec(),
                    self.dimensions().start_index_map().to_vec(),
                )
                .with_batching_dimensions(
                    self.dimensions().operand_batching_dimensions().to_vec(),
                    self.dimensions().start_indices_batching_dimensions().to_vec(),
                );
                let scatter_operation = ScatterOperation::new(scatter_dimensions, ScatterReductionKind::Add)
                    .with_mode(self.mode())
                    .with_indices_are_sorted(self.indices_are_sorted())
                    .with_unique_indices(self.unique_indices());
                let outputs =
                    context.stage_operation(scatter_operation, Vec::new(), &[zeros, indices, cotangent.clone()])?;
                check_count!("output", outputs, 1, ProgramError);
                Ok(vec![
                    MaybeZero::Value(outputs.into_iter().next().unwrap()),
                    MaybeZero::Zero(inputs[1].r#type().cotangent()),
                ])
            }
        }
    }
}

/// Value-level gather capability: the receiver-style entry point for staging or executing [`GatherOperation`].
///
/// The receiver is the operand (the data source); `indices` is a separate integer-typed value whose last axis holds
/// each start-index vector. The output assembles the sliced windows according to `operation`'s
/// [`GatherDimensionNumbers`]; see that type for the shape rule and the implicit index-vector-dimension convention.
/// The operand and indices must reside in the same memory space. The result retains that memory placement and clears
/// explicit physical layout metadata because gathering changes the logical relationship between axes and storage.
///
/// # Example
///
/// ```rust
/// # use ryft_core::operations::manipulation::{Gather, GatherDimensionNumbers, GatherOperation};
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::backends::arrays::Array;
/// # use ryft_core::types::{ArrayType, DataType};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Take rows 0 and 2 of a 3x2 matrix: each query is a scalar row index, so the indices have shape [2, 1]
/// // (two queries, one index component each) and the gathered window is a full row (slice sizes [1, 2]).
/// let operand = Array::matrix(3, 2, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
/// let indices = Array::from_f64s(
///     ArrayType::new(DataType::I32, ryft_core::types::Shape::new(vec![
///         ryft_core::types::Dimension::Static(2),
///         ryft_core::types::Dimension::Static(1),
///     ])),
///     vec![0.0, 2.0],
/// );
/// let dimensions = GatherDimensionNumbers::new(vec![1], vec![0], vec![0]);
/// let operation = GatherOperation::new(dimensions, vec![1, 2]);
/// let rows = operand.gather(&indices, &operation)?;
/// // `rows` has shape [2, 2] holding rows 0 and 2: [[0, 1], [4, 5]].
/// assert_eq!(rows.to_f64s(), vec![0.0, 1.0, 4.0, 5.0]);
/// # Ok(())
/// # }
/// ```
pub trait Gather: Sized {
    /// Gathers windows out of `self` (the operand) at the positions named by `indices`, according to `operation`.
    fn gather(&self, indices: &Self, operation: &GatherOperation) -> Result<Self, ProgramError>;
}

impl Gather for ArrayType {
    /// Type-level gather: validates the dimension numbers and slice sizes against the operand and indices types and
    /// computes the output shape and placement.
    fn gather(&self, indices: &Self, operation: &GatherOperation) -> Result<Self, ProgramError> {
        let operand = self;
        let dimensions = operation.dimensions();
        let slice_sizes = operation.slice_sizes();
        let operand_rank = operand.rank();
        let indices_rank = indices.rank();

        if indices_rank == 0 {
            return Err(TypeError::invalid(format!(
                "'{GATHER_OPERATION_NAME}' indices must have rank at least 1 (the trailing index vector)"
            ))
            .into());
        }
        if !indices.data_type().is_integer() {
            return Err(TypeError::invalid(format!(
                "'{GATHER_OPERATION_NAME}' indices must be integer-typed but have type {indices}"
            ))
            .into());
        }
        if operand.memory() != indices.memory() {
            return Err(TypeError::invalid(format!(
                "'{GATHER_OPERATION_NAME}' operand and indices must share one memory space but reside in {} and {}",
                operand.memory(),
                indices.memory(),
            ))
            .into());
        }
        let index_vector_dimension = indices_rank - 1;
        let Dimension::Static(index_vector_extent) = indices.dimension(index_vector_dimension) else {
            return Err(TypeError::invalid(format!(
                "'{GATHER_OPERATION_NAME}' indices index vector dimension must have a static extent"
            ))
            .into());
        };

        // Output rank, and the constituent operand-axis classification.
        let output_rank = dimensions.offset_dimensions().len() + indices_rank - 1;
        validate_sorted_unique_in_range(
            GATHER_OPERATION_NAME,
            "offset_dimensions",
            dimensions.offset_dimensions(),
            output_rank,
        )?;
        validate_sorted_unique_in_range(
            GATHER_OPERATION_NAME,
            "collapsed_slice_dimensions",
            dimensions.collapsed_slice_dimensions(),
            operand_rank,
        )?;
        validate_sorted_unique_in_range(
            GATHER_OPERATION_NAME,
            "operand_batching_dimensions",
            dimensions.operand_batching_dimensions(),
            operand_rank,
        )?;

        if dimensions.start_index_map().len() != index_vector_extent {
            return Err(TypeError::invalid(format!(
                "'{GATHER_OPERATION_NAME}' start_index_map has length {} but the index vector extent is \
                     {index_vector_extent}",
                dimensions.start_index_map().len(),
            ))
            .into());
        }
        validate_unique_in_range(GATHER_OPERATION_NAME, "start_index_map", dimensions.start_index_map(), operand_rank)?;

        if dimensions.start_indices_batching_dimensions().len() != dimensions.operand_batching_dimensions().len() {
            return Err(TypeError::invalid(format!(
                "'{GATHER_OPERATION_NAME}' operand and start-indices batching dimensions must align 1:1, but got {} \
                     and {}",
                dimensions.operand_batching_dimensions().len(),
                dimensions.start_indices_batching_dimensions().len(),
            ))
            .into());
        }
        for &dimension in dimensions.start_indices_batching_dimensions() {
            if dimension >= indices_rank || dimension == index_vector_dimension {
                return Err(TypeError::invalid(format!(
                    "'{GATHER_OPERATION_NAME}' start_indices_batching_dimensions entry {dimension} is out of range \
                         or names the index vector dimension"
                ))
                .into());
            }
        }

        // The collapsed, batching, and start-index-map axis sets must be mutually disjoint where required.
        let collapsed: BTreeSet<usize> = dimensions.collapsed_slice_dimensions().iter().copied().collect();
        let operand_batching: BTreeSet<usize> = dimensions.operand_batching_dimensions().iter().copied().collect();
        if collapsed.intersection(&operand_batching).next().is_some() {
            return Err(TypeError::invalid(format!(
                "'{GATHER_OPERATION_NAME}' collapsed_slice_dimensions and operand_batching_dimensions must be \
                     disjoint"
            ))
            .into());
        }

        // Slice sizes: one per operand axis; size 1 on collapsed axes; size at most 1 on batching axes; within the
        // operand extent when that extent is static.
        if slice_sizes.len() != operand_rank {
            return Err(TypeError::invalid(format!(
                "'{GATHER_OPERATION_NAME}' slice_sizes has length {} but the operand has rank {operand_rank}",
                slice_sizes.len(),
            ))
            .into());
        }
        for (axis, &size) in slice_sizes.iter().enumerate() {
            if let Dimension::Static(extent) = operand.dimension(axis)
                && size > extent
            {
                return Err(TypeError::invalid(format!(
                    "'{GATHER_OPERATION_NAME}' slice size {size} at axis {axis} exceeds the operand extent {extent}"
                ))
                .into());
            }
            if collapsed.contains(&axis) && size != 1 {
                return Err(TypeError::invalid(format!(
                    "'{GATHER_OPERATION_NAME}' collapsed slice dimension {axis} must have slice size 1 but has {size}"
                ))
                .into());
            }
            if operand_batching.contains(&axis) && size > 1 {
                return Err(TypeError::invalid(format!(
                    "'{GATHER_OPERATION_NAME}' operand batching dimension {axis} must have slice size at most 1 but \
                         has {size}"
                ))
                .into());
            }
        }

        let offset_count = operand_rank - collapsed.len() - operand_batching.len();
        if dimensions.offset_dimensions().len() != offset_count {
            return Err(TypeError::invalid(format!(
                "'{GATHER_OPERATION_NAME}' offset_dimensions has length {} but the operand has {offset_count} \
                     non-collapsed, non-batching axes",
                dimensions.offset_dimensions().len(),
            ))
            .into());
        }

        // Batch-dimension extents must match between operand and indices.
        for (&operand_axis, &indices_axis) in
            dimensions.operand_batching_dimensions().iter().zip(dimensions.start_indices_batching_dimensions())
        {
            if operand.dimension(operand_axis) != indices.dimension(indices_axis) {
                return Err(TypeError::invalid(format!(
                    "'{GATHER_OPERATION_NAME}' batching dimensions must have equal extents, but operand axis \
                         {operand_axis} and indices axis {indices_axis} differ"
                ))
                .into());
            }
        }

        // Output shape: offset positions take the (non-collapsed, non-batching) operand window sizes in operand-axis
        // order; the remaining positions take the indices' batch axes (every axis but the index vector) in order.
        let operand_offset_axes: Vec<usize> = (0..operand_rank)
            .filter(|axis| !collapsed.contains(axis) && !operand_batching.contains(axis))
            .collect();
        let batch_query_sizes: Vec<Dimension> = (0..indices_rank)
            .filter(|axis| *axis != index_vector_dimension)
            .map(|axis| indices.dimension(axis))
            .collect();
        let offset_position: BTreeSet<usize> = dimensions.offset_dimensions().iter().copied().collect();
        let mut offset_iterator = operand_offset_axes.iter();
        let mut batch_iterator = batch_query_sizes.iter();
        let output_dimensions: Vec<Dimension> = (0..output_rank)
            .map(|position| {
                if offset_position.contains(&position) {
                    let &operand_axis = offset_iterator.next().expect("offset axis count was validated");
                    Dimension::Static(slice_sizes[operand_axis])
                } else {
                    batch_iterator.next().expect("batch axis count was validated").clone()
                }
            })
            .collect();

        // Output sharding (JAX's `_gather_sharding_rule`). The operand axes named by `start_index_map`, the collapsed
        // axes, and the indices' index vector axis must be replicated (gated to explicit mesh axes; manual/auto
        // placements pass through). The output then inherits the operand's window-axis placements at the offset
        // positions and the indices' batch-axis placements elsewhere, carrying the operand's reduction state. When the
        // placement is ambiguous (a sliced axis is sharded over an explicit axis), a requested `output_sharding`
        // resolves it, mirroring `dot`/`reduce`.
        let operand_sharding = operand.sharding();
        let indices_sharding = indices.sharding();
        let sharding = if let Some(requested) = operation.output_sharding() {
            if requested.rank() != output_rank {
                return Err(TypeError::invalid(format!(
                    "'{GATHER_OPERATION_NAME}' output sharding rank ({}) does not match the output rank \
                         ({output_rank})",
                    requested.rank(),
                ))
                .into());
            }
            if references_auto_axis(requested) {
                return Err(TypeError::invalid(format!(
                    "'{GATHER_OPERATION_NAME}' output sharding cannot reference auto mesh axes"
                ))
                .into());
            }
            Some(requested.clone())
        } else if let Some(mesh) = resolve_mesh(operand_sharding, indices_sharding)? {
            // The operand axes that drive start indices (start_index_map ∪ collapsed) and the indices' index vector
            // axis must be replicated over explicit mesh axes; otherwise the placement is ambiguous and an output
            // sharding is required.
            let replicated_operand_axes: BTreeSet<usize> = dimensions
                .start_index_map()
                .iter()
                .chain(dimensions.collapsed_slice_dimensions())
                .copied()
                .collect();
            if let Some(sharding) = operand_sharding {
                for &axis in &replicated_operand_axes {
                    if dimension_has_explicit_axis(&mesh, &sharding.dimensions()[axis]) {
                        return Err(TypeError::invalid(format!(
                            "'{GATHER_OPERATION_NAME}' operand axis {axis} is indexed by the start indices and must \
                                 be replicated over explicit mesh axes; request an explicit output sharding to resolve \
                                 placement"
                        ))
                        .into());
                    }
                }
            }
            if let Some(sharding) = indices_sharding
                && dimension_has_explicit_axis(&mesh, &sharding.dimensions()[index_vector_dimension])
            {
                return Err(TypeError::invalid(format!(
                    "'{GATHER_OPERATION_NAME}' indices index vector dimension must be replicated over explicit \
                         mesh axes"
                ))
                .into());
            }

            // Propagate placement: offset positions inherit the operand window axes; the remaining positions inherit
            // the indices' batch axes (every axis but the index vector), in order.
            let indices_batch_axes: Vec<usize> =
                (0..indices.rank()).filter(|axis| *axis != index_vector_dimension).collect();
            let mut offset_iterator = operand_offset_axes.iter();
            let mut batch_iterator = indices_batch_axes.iter();
            let placement: Vec<ShardingDimension> = (0..output_rank)
                .map(|position| {
                    if offset_position.contains(&position) {
                        let &operand_axis = offset_iterator.next().expect("offset axis count was validated");
                        operand_sharding
                            .map(|sharding| sharding.dimensions()[operand_axis].clone())
                            .unwrap_or(ShardingDimension::Replicated)
                    } else {
                        let &indices_axis = batch_iterator.next().expect("batch axis count was validated");
                        indices_sharding
                            .map(|sharding| sharding.dimensions()[indices_axis].clone())
                            .unwrap_or(ShardingDimension::Replicated)
                    }
                })
                .collect();

            // Carry the operand's reduction state: gather selects elements, so a value unreduced/reduced over a mesh
            // axis stays so (selection commutes with a pending cross-device sum), mirroring slice.
            let (unreduced_axes, reduced_axes, varying_manual_axes) = match operand_sharding {
                Some(sharding) => (
                    sharding.unreduced_axes().iter().cloned().collect::<Vec<_>>(),
                    sharding.reduced_axes().iter().cloned().collect::<Vec<_>>(),
                    sharding.varying_manual_axes().iter().cloned().collect::<Vec<_>>(),
                ),
                None => (Vec::new(), Vec::new(), Vec::new()),
            };
            let map_sharding_error = |error| {
                TypeError::invalid(format!("'{GATHER_OPERATION_NAME}' output sharding construction failed: {error}"))
            };
            let sharding = Sharding::new(mesh, placement)
                .map_err(&map_sharding_error)?
                .with_unreduced_axes(unreduced_axes)
                .map_err(&map_sharding_error)?
                .with_reduced_axes(reduced_axes)
                .map_err(&map_sharding_error)?
                .with_varying_manual_axes(varying_manual_axes)
                .map_err(map_sharding_error)?;
            Some(sharding.without_auto_axes())
        } else {
            None
        };
        ArrayType::new(operand.data_type(), Shape::new(output_dimensions))
            .with_memory(operand.memory())
            .with_sharding(sharding)
            .map_err(|error| TypeError::invalid(error.to_string()).into())
    }
}

/// Any context-carrying value gathers by binding a [`GatherOperation`] through its own context. The
/// `From<GatherOperation>` bound makes this disjoint from the eager value types (whose context operation is
/// `ConstantOperation`), so it covers the transform tracers without conflicting with the concrete implementations.
impl<V: Value<Type = ArrayType>> Gather for V
where
    V::DispatchDomain: Context<Type = ArrayType>,
    <V::DispatchDomain as Domain>::Operation: From<GatherOperation>,
{
    fn gather(&self, indices: &Self, operation: &GatherOperation) -> Result<Self, ProgramError> {
        let mut outputs =
            self.dispatch_domain().bind(operation.clone(), Vec::new(), &[self.clone(), indices.clone()])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::contexts::Context;
    use crate::differentiation::jacobian::jacobian_forward;
    use crate::macros::{
        check_operation_batching, check_operation_partial_evaluation, check_operation_transposition,
        check_operation_type_inference,
    };
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::types::{DataType, Memory};

    use super::*;

    fn indices_type(dimensions: Vec<usize>) -> ArrayType {
        ArrayType::new(DataType::I32, Shape::new(dimensions.into_iter().map(Dimension::Static).collect()))
    }

    fn float_type(dimensions: Vec<usize>) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(dimensions.into_iter().map(Dimension::Static).collect()))
    }

    /// Lifts a constant integer index array into the trace or differentiation context that `exemplar` belongs to.
    fn index_array<V>(exemplar: &V, shape: Vec<usize>, values: Vec<f64>) -> V
    where
        V: crate::programs::Value<Type = ArrayType>,
        V::DispatchDomain: crate::contexts::Context<Constant = Array>,
    {
        let r#type = ArrayType::new(DataType::I32, Shape::new(shape.into_iter().map(Dimension::Static).collect()));
        exemplar.dispatch_domain().lift(Array::from_f64s(r#type, values)).unwrap()
    }

    #[test]
    fn test_gather() {
        // Take whole rows of a [3, 2] matrix indexed by a [2, 1] index array: offset axis 1 carries the row (slice
        // sizes [1, 2]); axis 0 (the collapsed row axis) is driven by the start index.
        let dimensions = GatherDimensionNumbers::new(vec![1], vec![0], vec![0]);
        let operation = GatherOperation::new(dimensions, vec![1, 2]);
        assert_eq!(operation.name(), GATHER_OPERATION_NAME);
        assert_eq!(operation.slice_sizes(), &[1, 2]);

        let operand = float_type(vec![3, 2]);
        let indices = indices_type(vec![2, 1]);
        let host_operand = operand.clone().with_memory(Memory::Host { pinned: true });
        let host_indices = indices.clone().with_memory(Memory::Host { pinned: true });
        let host_output = float_type(vec![2, 2]).with_memory(Memory::Host { pinned: true });
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [
                {
                    input_types = [operand.clone(), indices.clone()],
                    output_types = [float_type(vec![2, 2])],
                },
                {
                    input_types = [operand.clone()],
                    error = "expected 2 inputs but got 1",
                },
                {
                    input_types = [operand.clone(), float_type(vec![2, 1])],
                    error = "'gather' indices must be integer-typed but have type f32[2, 1]",
                },
                {
                    input_types = [host_operand.clone(), host_indices],
                    output_types = [host_output],
                },
                {
                    input_types = [host_operand, indices.clone()],
                    error = "'gather' operand and indices must share one memory space but reside in Host[Pinned] and \
                             Device",
                },
            ],
        );

        assert_eq!(
            format!("{operation}"),
            concat!(
                "gather [\n",
                "    dimensions=(offset=[1], collapsed_slice=[0], start_index_map=[0], operand_batching=[], ",
                "start_indices_batching=[]),\n",
                "    slice_sizes=[1, 2],\n",
                "]",
            ),
        );

        // Interpretation handles each out-of-bounds mode explicitly.
        let scalar_dimensions = GatherDimensionNumbers::new(vec![], vec![0], vec![0]);
        let scalar_indices = Array::from_f64s(indices_type(vec![2, 1]), vec![1.0, 5.0]);
        let run = |mode| {
            Array::vector(vec![10.0, 20.0, 30.0, 40.0])
                .gather(&scalar_indices, &GatherOperation::new(scalar_dimensions.clone(), vec![1]).with_mode(mode))
                .unwrap()
                .to_f64s()
        };
        assert_eq!(run(GatherScatterMode::Clip), vec![20.0, 40.0]);
        assert_eq!(run(GatherScatterMode::PromiseInBounds), vec![20.0, 40.0]);
        assert_eq!(run(GatherScatterMode::FillOrDrop), vec![20.0, 0.0]);

        // Partial evaluation folds fully known gathers and residualizes an unknown data operand with known indices.
        let operand_value = Array::matrix(3, 2, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let indices_value = Array::from_f64s(indices_type(vec![2, 1]), vec![0.0, 2.0]);
        let expected = Array::matrix(2, 2, vec![0.0, 1.0, 4.0, 5.0]);
        check_operation_partial_evaluation!(
            backend = (Array, ArrayOperation<Array>),
            operation = operation.clone(),
            cases = [
                {
                    inputs = [(@known, operand_value.clone()), (@known, indices_value.clone())],
                    outputs = [(@known, expected.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = operand_value.r#type().into_owned(), replay = operand_value.clone())),
                        (@known, indices_value.clone()),
                    ],
                    outputs = [(@residual, expected)],
                    residual_instructions = 1,
                },
            ],
        );

        // Batching expands each mapped item independently, including the empty-batch boundary.
        check_operation_batching!(
            @exact,
            operation = operation.clone(),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::from_f64s(
                        ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into(), 2.into()])),
                        (0..12).map(|value| value as f64).collect(),
                    )),
                    (@replicated, indices_value),
                ],
                outputs = [(@mapped(axis = 0), Array::from_f64s(
                    ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 2.into(), 2.into()])),
                    vec![0.0, 1.0, 4.0, 5.0, 6.0, 7.0, 10.0, 11.0],
                ))],
            }],
        );
        check_operation_batching!(
            @exact,
            operation = GatherOperation::new(GatherDimensionNumbers::new(Vec::new(), vec![0], vec![0]), vec![1]),
            axis_size = 0,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::from_f64s(
                        ArrayType::new(DataType::F64, Shape::new(vec![0.into(), 3.into()])),
                        Vec::new(),
                    )),
                    (@mapped(axis = 0), Array::from_f64s(
                        ArrayType::new(DataType::I32, Shape::new(vec![0.into(), 1.into(), 1.into()])),
                        Vec::new(),
                    )),
                ],
                outputs = [(@mapped(axis = 0), Array::from_f64s(
                    ArrayType::new(DataType::F64, Shape::new(vec![0.into(), 1.into()])),
                    Vec::new(),
                ))],
            }],
        );
    }

    #[test]
    fn test_gather_dimension_numbers() {
        let operand = float_type(vec![3, 2]);
        let indices = indices_type(vec![2, 1]);

        // start_index_map length must equal the index vector extent (here 1, not 2).
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0, 1]), vec![1, 2]);
        assert_eq!(
            operation.infer_output_types(&[operand.clone(), indices.clone()], &[]),
            Err(TypeError::invalid(
                "'gather' start_index_map has length 2 but the index vector extent is 1".to_string()
            )),
        );

        // A collapsed axis must have slice size 1.
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![2, 2]);
        assert_eq!(
            operation.infer_output_types(&[operand.clone(), indices.clone()], &[]),
            Err(TypeError::invalid(
                "'gather' collapsed slice dimension 0 must have slice size 1 but has 2".to_string()
            )),
        );

        // offset_dimensions count must equal the non-collapsed, non-batching operand axes (here 1).
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1, 2], vec![0], vec![0]), vec![1, 2]);
        assert_eq!(
            operation.infer_output_types(&[operand, indices], &[]),
            Err(TypeError::invalid(
                "'gather' offset_dimensions has length 2 but the operand has 1 non-collapsed, non-batching \
                          axes"
                    .to_string()
            )),
        );
    }

    #[test]
    fn test_array_type_gather() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        // Operand [4, 2] sharded only on the feature axis (axis 1); axis 0 (indexed by the start index) is replicated.
        let operand = float_type(vec![4, 2])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["y"])])
                    .unwrap(),
            )
            .unwrap();
        let indices = indices_type(vec![3, 1]);
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
        // Output [3, 2]: the query axis (from the indices) is replicated, the feature axis keeps `y`.
        let output = operation.infer_output_types(&[operand, indices], &[]).unwrap();
        assert_eq!(
            output[0].sharding().unwrap().dimensions(),
            &[ShardingDimension::Replicated, ShardingDimension::sharded(["y"])],
        );

        // Sharding the start-indexed operand axis over an explicit mesh axis is ambiguous without an output sharding.
        let operand = float_type(vec![4, 2])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        let indices = indices_type(vec![3, 1]);
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
        assert!(operation.infer_output_types(&[operand, indices], &[]).is_err());
    }

    /// Minimal operation enum hosting the primal [`GatherOperation`] (the forward gather) and the primal
    /// [`ScatterOperation`] (its staged scatter-add adjoint) plus the structural `zero` and `add` operations the
    /// transpose pass needs. The `Constant` variant carries the value parameter `V` so the [`Operation`] derive can
    /// infer the primary type. [`TransposableOperation`] is hand-written rather than derived because the primal
    /// [`ScatterOperation`] adjoint target has no transpose rule (it only ever appears in the pullback, never as a
    /// forward instruction being transposed); the derived all-variant dispatcher would require one.
    #[derive(Clone, Debug, ryft_macros::Operation)]
    enum TestGatherOperation<V: Value<Type = ArrayType>> {
        Zero(ZeroOperation<ArrayType>),
        Constant(crate::operations::constants::ConstantOperation<V>),
        Add(crate::operations::math::AddOperation),
        Gather(GatherOperation),
        Scatter(ScatterOperation),
    }

    impl<V: Value<Type = ArrayType>> TransposableOperation<V, TestGatherOperation<V>> for TestGatherOperation<V> {
        fn transpose<D: TranspositionDriver<V, TestGatherOperation<V>>>(
            &self,
            context: &mut TracingContext<V, TestGatherOperation<V>>,
            driver: &D,
            inputs: &[PartialValue<Tracer<TracingContext<V, TestGatherOperation<V>>>>],
            outputs: &[MaybeZero<Tracer<TracingContext<V, TestGatherOperation<V>>>>],
        ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, TestGatherOperation<V>>>>>, DifferentiationError> {
            match self {
                Self::Gather(operation) => operation.transpose(context, driver, inputs, outputs),
                _ => Err(ProgramError::UnsupportedOperation {
                    message: format!("{} is not transposed in this test enum", self.name()),
                }
                .into()),
            }
        }
    }

    #[test]
    fn test_gather_differentiation() {
        // Take rows 0 and 2 of a [3, 2] operand: the operand is linear and the [2, 1] index array is the known
        // operand. The gathered output and its cotangent have shape [2, 2].
        let dimensions = GatherDimensionNumbers::new(vec![1], vec![0], vec![0]);
        let operation = GatherOperation::new(dimensions, vec![1, 2]);
        let operand = Array::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let indices = Array::from_f64s(indices_type(vec![2, 1]), vec![0.0, 2.0]);
        let cotangent = Array::matrix(2, 2, vec![10.0, 20.0, 30.0, 40.0]);
        check_operation_transposition!(
            @exact,
            backend = (Array, TestGatherOperation<Array>),
            operation = operation,
            cases = [{
                inputs = [
                    (@linear(type = operand.r#type().into_owned())),
                    (@known, indices),
                ],
                output_cotangents = [cotangent],
                input_cotangents = [Array::matrix(3, 2, vec![10.0, 20.0, 0.0, 0.0, 30.0, 40.0])],
            }],
        );

        // Forward mode selects the operand coordinate feeding each gathered output.
        let jacobian = jacobian_forward(
            |operand| {
                let indices = index_array(&operand, vec![2, 1], vec![0.0, 2.0]);
                let operation =
                    GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
                Ok(operand.gather(&indices, &operation).unwrap())
            },
            Array::matrix(3, 2, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        )
        .unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.output_type().static_shape().unwrap().as_slice(), &[2, 2]);
        assert_eq!(block.input_type().static_shape().unwrap().as_slice(), &[3, 2]);
        assert_eq!(
            block.value().values(),
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, //
            ],
        );
    }
}
