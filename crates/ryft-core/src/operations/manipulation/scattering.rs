use std::collections::BTreeSet;
use std::fmt::Display;

use crate::batching::{
    ArrayBatch, BatchAxis, BatchableOperation, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    ElementwiseDerivativeAlignment, TransposableOperation, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::constants::{Zero, ZeroOperation};
use crate::operations::manipulation::{Broadcast, Reshape, Slice, Transpose, UpdateSlice};
use crate::operations::sharding::Reshard;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{TypeError, Typed};
use crate::programs::values::Value;
use crate::programs::{MaybeZero, ProgramError};
use crate::sharding::{LogicalMesh, Sharding};
use crate::tracing::{Tracer, TracingContext};
use crate::tracing_v2::operations::custom_derivatives::CustomVjpResidual;
use crate::types::{ArrayType, Size};

// TODO(eaplatanios): Review this.

use super::gathering::{
    GatherDimensionNumbers, GatherOperation, GatherScatterMode, LinearGatherOperation, dimension_has_explicit_axis,
    references_auto_axis, validate_sorted_unique_in_range, validate_unique_in_range,
};
use super::slicing::batch_by_item_expansion;

/// Canonical operation name for [`ScatterOperation`].
pub const SCATTER_OPERATION_NAME: &str = "scatter";

/// Combiner applied when a [`scatter`](Scatter) writes an update into the operand. Each kind selects the binary
/// reduction used where an update meets the existing operand value and lowers to the corresponding
/// `stablehlo.scatter` combiner region. Only [`Add`](Self::Add) is a linear map and participates in the
/// gather/scatter-add transpose duality; the others require unique indices for a well-defined gradient.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ScatterReductionKind {
    /// The update replaces the operand value (StableHLO's scatter whose combiner returns the update).
    Overwrite,

    /// The update is added to the operand value (`scatter_add`). The only linear kind.
    Add,

    /// The update is multiplied with the operand value (`scatter_mul`).
    Mul,

    /// The operand value is replaced by the minimum of itself and the update (`scatter_min`).
    Min,

    /// The operand value is replaced by the maximum of itself and the update (`scatter_max`).
    Max,
}

impl ScatterReductionKind {
    /// Returns the canonical operation name suffix for this kind.
    pub fn name(self) -> &'static str {
        match self {
            Self::Overwrite => "overwrite",
            Self::Add => "add",
            Self::Mul => "mul",
            Self::Min => "min",
            Self::Max => "max",
        }
    }

    /// Returns `true` when this kind is a linear map in the operand and updates (only [`Add`](Self::Add) is). Linear
    /// kinds participate in the gather/scatter-add transpose duality; the others are differentiable only under unique
    /// indices and through a primal-domain mask.
    pub fn is_linear(self) -> bool {
        matches!(self, Self::Add)
    }
}

impl Display for ScatterReductionKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.name())
    }
}

/// Specification of how the index operand and the update windows map onto the operand axes of a [`scatter`](Scatter),
/// following StableHLO's [`scatter`](https://openxla.org/stablehlo/spec#scatter) dimension numbers. It is the
/// structural dual of [`GatherDimensionNumbers`]:
/// [`update_window_dimensions`](Self::update_window_dimensions) mirrors `offset_dimensions`,
/// [`inserted_window_dimensions`](Self::inserted_window_dimensions) mirrors `collapsed_slice_dimensions`, and
/// [`scatter_dimensions_to_operand_dimensions`](Self::scatter_dimensions_to_operand_dimensions) mirrors
/// `start_index_map`.
///
/// The index vector dimension is implicit and always the last axis of the indices operand. The output has the same
/// shape as the operand.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ScatterDimensionNumbers {
    /// Axes of the updates operand that hold a scattered window, in ascending order. Their count equals the number of
    /// operand axes that are neither inserted nor batching.
    update_window_dimensions: Vec<usize>,

    /// Operand axes whose window size is `1` and that have no corresponding updates axis, in ascending order.
    inserted_window_dimensions: Vec<usize>,

    /// For each component of a start-index vector (the last axis of the indices operand), the operand axis it scatters
    /// into. Its length equals the extent of the indices' index vector dimension.
    scatter_dimensions_to_operand_dimensions: Vec<usize>,

    /// Operand axes batched against [`scatter_indices_batching_dimensions`](Self::scatter_indices_batching_dimensions),
    /// aligned 1:1, in ascending order.
    operand_batching_dimensions: Vec<usize>,

    /// Indices axes (other than the index vector dimension) that align 1:1 with
    /// [`operand_batching_dimensions`](Self::operand_batching_dimensions).
    scatter_indices_batching_dimensions: Vec<usize>,
}

impl ScatterDimensionNumbers {
    /// Creates scatter dimension numbers from explicit axis lists. The batching axis lists default to empty; use
    /// [`with_batching_dimensions`](Self::with_batching_dimensions) to set them.
    #[inline]
    pub fn new(
        update_window_dimensions: Vec<usize>,
        inserted_window_dimensions: Vec<usize>,
        scatter_dimensions_to_operand_dimensions: Vec<usize>,
    ) -> Self {
        Self {
            update_window_dimensions,
            inserted_window_dimensions,
            scatter_dimensions_to_operand_dimensions,
            operand_batching_dimensions: Vec::new(),
            scatter_indices_batching_dimensions: Vec::new(),
        }
    }

    /// Attaches the operand/indices batching axis pair (aligned 1:1).
    #[inline]
    pub fn with_batching_dimensions(
        mut self,
        operand_batching_dimensions: Vec<usize>,
        scatter_indices_batching_dimensions: Vec<usize>,
    ) -> Self {
        self.operand_batching_dimensions = operand_batching_dimensions;
        self.scatter_indices_batching_dimensions = scatter_indices_batching_dimensions;
        self
    }

    /// Returns the updates window axes.
    #[inline]
    pub fn update_window_dimensions(&self) -> &[usize] {
        &self.update_window_dimensions
    }

    /// Returns the inserted (size-1, operand-only) axes.
    #[inline]
    pub fn inserted_window_dimensions(&self) -> &[usize] {
        &self.inserted_window_dimensions
    }

    /// Returns the scatter-index-to-operand-axis map.
    #[inline]
    pub fn scatter_dimensions_to_operand_dimensions(&self) -> &[usize] {
        &self.scatter_dimensions_to_operand_dimensions
    }

    /// Returns the operand batching axes.
    #[inline]
    pub fn operand_batching_dimensions(&self) -> &[usize] {
        &self.operand_batching_dimensions
    }

    /// Returns the indices batching axes.
    #[inline]
    pub fn scatter_indices_batching_dimensions(&self) -> &[usize] {
        &self.scatter_indices_batching_dimensions
    }
}

impl Display for ScatterDimensionNumbers {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "(update_window={:?}, inserted_window={:?}, scatter_to_operand={:?}, operand_batching={:?}, \
             scatter_indices_batching={:?})",
            self.update_window_dimensions,
            self.inserted_window_dimensions,
            self.scatter_dimensions_to_operand_dimensions,
            self.operand_batching_dimensions,
            self.scatter_indices_batching_dimensions,
        )
    }
}

/// [`Operation`] that writes update windows into a copy of an operand at positions named by an integer index operand,
/// combining overlaps with a [`ScatterReductionKind`]. Refer to the documentation of [`Scatter`] for the semantics.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ScatterOperation {
    /// Dimension numbers mapping the index operand and update windows onto the operand axes.
    dimensions: ScatterDimensionNumbers,

    /// Combiner applied where an update meets the existing operand value.
    kind: ScatterReductionKind,

    /// Out-of-bounds index handling.
    mode: GatherScatterMode,

    /// Whether the caller guarantees the index vectors are sorted (a lowering hint only).
    indices_are_sorted: bool,

    /// Whether the caller guarantees the scattered windows do not overlap (a lowering hint; also the boundary for a
    /// well-defined gradient of the non-additive kinds).
    unique_indices: bool,

    /// Optional requested output [`Sharding`], used when the inferred placement is ambiguous (see
    /// [`Self::with_output_sharding`]).
    output_sharding: Option<Sharding>,
}

impl ScatterOperation {
    /// Creates a new [`ScatterOperation`] with the provided dimension numbers and combiner kind. The mode defaults to
    /// [`GatherScatterMode::PromiseInBounds`] and both index hints default to `false`; use the chained `with_*`
    /// builders to override them.
    #[inline]
    pub fn new(dimensions: ScatterDimensionNumbers, kind: ScatterReductionKind) -> Self {
        Self {
            dimensions,
            kind,
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

    /// Sets the unique-indices hint (and the boundary for differentiating the non-additive kinds).
    #[inline]
    pub fn with_unique_indices(mut self, unique_indices: bool) -> Self {
        self.unique_indices = unique_indices;
        self
    }

    /// Requests `output_sharding` for the result. The scatter sharding rule replicates the operand axes named by
    /// [`ScatterDimensionNumbers::scatter_dimensions_to_operand_dimensions`] (and the inserted axes); when that leaves
    /// the placement ambiguous a requested output sharding resolves it, bypassing inference (like `dot`/`reduce`).
    #[inline]
    pub fn with_output_sharding(mut self, output_sharding: impl Into<Option<Sharding>>) -> Self {
        self.output_sharding = output_sharding.into();
        self
    }

    /// Returns the dimension numbers.
    #[inline]
    pub fn dimensions(&self) -> &ScatterDimensionNumbers {
        &self.dimensions
    }

    /// Returns the combiner kind.
    #[inline]
    pub fn kind(&self) -> ScatterReductionKind {
        self.kind
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

impl Display for ScatterOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation<ArrayType> for ScatterOperation {
    #[inline]
    fn name(&self) -> &'static str {
        SCATTER_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 3, TypeError);
        match input_types[0].scatter(&input_types[1], &input_types[2], self) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError { message: error.to_string() }),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("kind", self.kind)?;
            operation.field("dimensions", &self.dimensions)?;
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

impl<C: Domain<Type = ArrayType, Value: Scatter>> InterpretableOperation<C> for ScatterOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 3, ProgramError);
        Ok(vec![inputs[0].scatter(&inputs[1], &inputs[2], self)?])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for ScatterOperation where
    C::Operation: From<ScatterOperation>
{
}

/// Forward-mode rule for [`ScatterOperation`]. For the [`Add`](ScatterReductionKind::Add) combiner the operation
/// is jointly linear in its operand and updates, while the integer index operand is a non-differentiated primal operand
/// edge, so the tangent scatter-adds the operand and update tangents at the same primal indices. A zero operand and
/// update tangent yields a typed zero output tangent. The non-additive combiners are not linear and their tangent
/// is not synthesized, so a non-zero tangent through them is rejected with
/// [`UnsupportedOperation`](ProgramError::UnsupportedOperation).
impl<C: Context<Type = ArrayType> + Zero<C::Value>> DifferentiableOperation<C> for ScatterOperation
where
    C::Operation: From<ScatterOperation>,
    C::Value: Scatter,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 3, ProgramError);
        let operand = &inputs[0];
        let indices = inputs[1].primal();
        let updates = &inputs[2];
        let primal = operand.primal().scatter(indices, updates.primal(), self)?;
        let tangent = if operand.tangent().is_zero() && updates.tangent().is_zero() {
            MaybeZero::Zero(primal.r#type().tangent())
        } else if self.kind() != ScatterReductionKind::Add {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "differentiation of scatter with the {} combiner is not yet implemented (only scatter-add is \
                     linear)",
                    self.kind(),
                ),
            }
            .into());
        } else {
            // One of the two linear tangents may still be a structural zero; scatter-add needs both as real values,
            // so materialize the zero side before staging the tangent scatter.
            let operand_tangent = operand.tangent().clone().materialize(context)?;
            let updates_tangent = updates.tangent().clone().materialize(context)?;
            MaybeZero::Value(operand_tangent.scatter(indices, &updates_tangent, self)?)
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

/// Partition-aware transpose rule for the primal [`ScatterOperation`] with an [`Add`](ScatterReductionKind::Add)
/// combiner. The integer index operand (operand 1) has no tangent space, so in a valid pushforward it is the known
/// operand while the scattered operand (operand 0) and the updates (operand 2) are the linear ones. Scatter-add
/// accumulates into its operand (`output = operand + scattered(updates)`, so the operand Jacobian is the identity), so
/// the operand cotangent is the output cotangent unchanged; the update cotangent gathers the output cotangent at the
/// scattered windows via the dual gather built by mirroring the scatter geometry. This reproduces the captured-index
/// [`LinearScatterAddOperation`] transpose rule, reading
/// the indices from the pullback through `operand_values` and staging a primal [`GatherOperation`] instead of folding
/// the indices into a captured factor. The indices receive a structural zero, and a zero output cotangent stays a
/// structural zero. Non-additive combiners are not linear and are
/// rejected.
impl<V: Value<Type = ArrayType>, O> TransposableOperation<V, O> for ScatterOperation
where
    O: Operation<ArrayType> + From<ZeroOperation<ArrayType>> + From<GatherOperation>,
    Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<ArrayType>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 3, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        if self.kind() != ScatterReductionKind::Add {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "transposition of scatter with the {} combiner is not yet implemented (only scatter-add is linear)",
                    self.kind(),
                ),
            }
            .into());
        }
        match &outputs[0] {
            MaybeZero::Zero(_) => Ok(inputs
                .iter()
                .map(|input| {
                    let input_type = input.r#type();
                    MaybeZero::Zero(input_type.cotangent())
                })
                .collect()),
            MaybeZero::Value(cotangent) => {
                // The indices are the known operand; the dispatch guarantees a `Known` operand carries its pullback
                // value, so read the tracer directly.
                let indices = inputs[1]
                    .as_known()
                    .expect("dispatch guarantees a known operand carries its pullback value")
                    .clone();
                // Build the dual gather by mirroring the scatter geometry: the slice sizes pair each operand window
                // axis with its update window extent, with size 1 at the inserted and batching axes.
                let dimensions = self.dimensions();
                let updates_type = inputs[2].r#type();
                let operand_rank = inputs[0].r#type().rank();
                let update_window_dimensions = dimensions.update_window_dimensions();
                let inserted_window_dimensions = dimensions.inserted_window_dimensions();
                let operand_batching_dimensions = dimensions.operand_batching_dimensions();
                let mut slice_sizes = Vec::with_capacity(operand_rank);
                let mut window_position = 0;
                for operand_axis in 0..operand_rank {
                    if inserted_window_dimensions.contains(&operand_axis)
                        || operand_batching_dimensions.contains(&operand_axis)
                    {
                        slice_sizes.push(1);
                    } else {
                        let update_axis = update_window_dimensions[window_position];
                        let extent = updates_type.dimension(update_axis as isize).value().ok_or_else(|| {
                            ProgramError::from(TypeError {
                                message: format!(
                                    "'{SCATTER_OPERATION_NAME}' transpose requires a static update shape but update axis \
                                     {update_axis} has a dynamic size",
                                ),
                            })
                        })?;
                        slice_sizes.push(extent);
                        window_position += 1;
                    }
                }
                let gather_dimensions = GatherDimensionNumbers::new(
                    update_window_dimensions.to_vec(),
                    inserted_window_dimensions.to_vec(),
                    dimensions.scatter_dimensions_to_operand_dimensions().to_vec(),
                )
                .with_batching_dimensions(
                    operand_batching_dimensions.to_vec(),
                    dimensions.scatter_indices_batching_dimensions().to_vec(),
                );
                let gather_operation = GatherOperation::new(gather_dimensions, slice_sizes)
                    .with_mode(self.mode())
                    .with_indices_are_sorted(self.indices_are_sorted())
                    .with_unique_indices(self.unique_indices())
                    .with_output_sharding(updates_type.sharding().cloned());
                let update_cotangents =
                    context.stage_operation(gather_operation, Vec::new(), &[cotangent.clone(), indices])?;
                check_count!("output", update_cotangents, 1, ProgramError);
                let update_cotangent =
                    update_cotangents.into_iter().next().unwrap().unalign_cotangent(&inputs[2].r#type().cotangent())?;
                Ok(vec![
                    MaybeZero::Value(cotangent.clone()),
                    MaybeZero::Zero(inputs[1].r#type().cotangent()),
                    MaybeZero::Value(update_cotangent),
                ])
            }
        }
    }
}

/// Batching rule for [`ScatterOperation`]. As with gather, a scatter's window/inserted/index axis bookkeeping does not
/// compose cleanly with an extra mapped axis, so any batched operand, indices, or updates is handled by per-item
/// expansion (`batch_by_item_expansion`): each batch item scatters independently and the results restack along a fresh
/// leading batch axis. This stages `O(axis_size)` scatters but is correct for every combiner and dimension-number
/// configuration; dimension-number lifting is a performance optimization left as a follow-up. When no input is mapped
/// the scatter applies once, unbatched.
impl<C> BatchableOperation<C> for ScatterOperation
where
    C: Context<Type = ArrayType> + Zero<C::Value>,
    C::Value: Broadcast + Transpose + Slice + UpdateSlice + Reshape + Reshard,
    ScatterOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        check_count!("input", inputs, 3, ProgramError);
        let Some(axis_size) = ArrayBatch::common_batch_size(inputs)? else {
            return self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()]);
        };
        batch_by_item_expansion(context, SCATTER_OPERATION_NAME, self, inputs, axis_size)
    }
}

/// Errors when `other` is sharded over a different mesh than `mesh`.
fn check_same_mesh(mesh: &LogicalMesh, other: Option<&Sharding>) -> Result<(), TypeError> {
    if let Some(other) = other
        && other.mesh() != mesh
    {
        return Err(TypeError {
            message: format!("'{SCATTER_OPERATION_NAME}' operand, indices, and updates shardings must use one mesh"),
        });
    }
    Ok(())
}

// TODO(eaplatanios): Should this be renamed to something that's not about "linearity"? This is about captured primals.
/// Captured-index scatter-add linear operation: the linear map
/// `(t, u) ↦ scatter_add(t, indices, u; dimensions)` over the tangents (or cotangents) of the scattered operand and
/// the updates.
///
/// It is the captured-index linear map emitted by the JVP of [`ScatterOperation`] with
/// an [`Add`](ScatterReductionKind::Add) combiner: the integer index operand is a primal value captured at
/// linearization time as a residual factor (it has no tangent space, so the map is jointly linear in the two tangent
/// operands), and its transpose is the dual gather. The two operation inputs are the operand and update tangents; the
/// captured `indices` factor is inserted between them as the scatter's index operand during type inference.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearScatterAddOperation<F> {
    /// Underlying [`ScatterOperation`] describing the scatter geometry.
    operation: ScatterOperation,

    /// Captured integer index operand factor.
    indices: F,
}

impl<F> LinearScatterAddOperation<F> {
    /// Creates a new [`LinearScatterAddOperation`] from the underlying scatter and the captured index factor.
    #[inline]
    pub fn new(operation: ScatterOperation, indices: F) -> Self {
        Self { operation, indices }
    }

    /// Returns the underlying [`ScatterOperation`] describing the scatter geometry.
    #[inline]
    pub fn operation(&self) -> &ScatterOperation {
        &self.operation
    }

    /// Returns the captured integer index operand factor.
    #[inline]
    pub fn indices(&self) -> &F {
        &self.indices
    }
}

impl<F: Value<Type = ArrayType>> Display for LinearScatterAddOperation<F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<F: Value<Type = ArrayType>> Operation<ArrayType> for LinearScatterAddOperation<F> {
    #[inline]
    fn name(&self) -> &'static str {
        SCATTER_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        self.operation.infer_output_types(
            &[input_types[0].clone(), self.indices.r#type().into_owned(), input_types[1].clone()],
            &[],
        )
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        let _ = indentation;
        formatter.write_str(self.name())
    }
}

impl<F, C> InterpretableOperation<C> for LinearScatterAddOperation<F>
where
    C: Domain<Type = ArrayType, Value: Scatter>,
    F: CustomVjpResidual<C::Value>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![inputs[0].scatter(&self.indices().residual_value()?, &inputs[1], self.operation())?])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate) for a [`LinearScatterAddOperation`].
impl<F: Value<Type = ArrayType>, C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C>
    for LinearScatterAddOperation<F>
where
    C::Operation: From<LinearScatterAddOperation<F>>,
{
}

/// Transpose rule for the captured-index scatter-add. Because scatter-add accumulates into its operand
/// (`output = operand + scattered(updates)`, so `∂output/∂operand = I`), the operand cotangent is the output cotangent
/// unchanged; the update cotangent gathers the output cotangent at the scattered windows via the dual gather built by
/// mirroring the scatter geometry. That gather needs the operand and update types, so the input count is checked
/// before the dual gather is derived. Symbolic-zero cotangents propagate unchanged.
impl<V: Value<Type = ArrayType>, O, F: Value<Type = ArrayType>> TransposableOperation<V, O>
    for LinearScatterAddOperation<F>
where
    O: Operation<ArrayType> + From<LinearGatherOperation<F>>,
    Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<ArrayType>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        let dimensions = self.operation().dimensions();
        let operand_type = inputs[0].r#type();
        let updates_type = inputs[1].r#type();
        let operand_rank = operand_type.rank();
        let update_window_dimensions = dimensions.update_window_dimensions();
        let inserted_window_dimensions = dimensions.inserted_window_dimensions();
        let operand_batching_dimensions = dimensions.operand_batching_dimensions();
        let mut slice_sizes = Vec::with_capacity(operand_rank);
        let mut window_position = 0;
        for operand_axis in 0..operand_rank {
            if inserted_window_dimensions.contains(&operand_axis) || operand_batching_dimensions.contains(&operand_axis)
            {
                slice_sizes.push(1);
            } else {
                let update_axis = update_window_dimensions[window_position];
                let extent = updates_type.dimension(update_axis as isize).value().ok_or_else(|| {
                    ProgramError::from(TypeError {
                        message: format!(
                            "'{SCATTER_OPERATION_NAME}' transpose requires a static update shape but update axis \
                             {update_axis} has a dynamic size",
                        ),
                    })
                })?;
                slice_sizes.push(extent);
                window_position += 1;
            }
        }
        let gather_dimensions = GatherDimensionNumbers::new(
            update_window_dimensions.to_vec(),
            inserted_window_dimensions.to_vec(),
            dimensions.scatter_dimensions_to_operand_dimensions().to_vec(),
        )
        .with_batching_dimensions(
            operand_batching_dimensions.to_vec(),
            dimensions.scatter_indices_batching_dimensions().to_vec(),
        );
        let gather_operation = GatherOperation::new(gather_dimensions, slice_sizes)
            .with_mode(self.operation().mode())
            .with_indices_are_sorted(self.operation().indices_are_sorted())
            .with_unique_indices(self.operation().unique_indices())
            .with_output_sharding(updates_type.sharding().cloned());
        check_count!("output", outputs, 1, ProgramError);
        match &outputs[0] {
            MaybeZero::Zero(_) => Ok(vec![
                MaybeZero::Zero(inputs[0].r#type().cotangent()),
                MaybeZero::Zero(inputs[1].r#type().cotangent()),
            ]),
            MaybeZero::Value(cotangent) => {
                let update_cotangents = context.stage_operation(
                    LinearGatherOperation::new(gather_operation, self.indices().clone()),
                    Vec::new(),
                    std::slice::from_ref(cotangent),
                )?;
                check_count!("output", update_cotangents, 1, ProgramError);
                let update_cotangent =
                    update_cotangents.into_iter().next().unwrap().unalign_cotangent(&inputs[1].r#type().cotangent())?;
                Ok(vec![MaybeZero::Value(cotangent.clone()), MaybeZero::Value(update_cotangent)])
            }
        }
    }
}

/// Value-level scatter capability: the receiver-style entry point for staging or executing [`ScatterOperation`].
///
/// The receiver is the operand; `indices` is a separate integer-typed value whose last axis holds each start-index
/// vector; `updates` holds the windows to combine into the operand using the operation's [`ScatterReductionKind`].
/// All three values must reside in the same memory space. The output has the same type as the operand, including its
/// layout and memory placement.
///
/// # Example
///
/// ```rust
/// # use ryft_core::operations::manipulation::{Scatter, ScatterDimensionNumbers, ScatterOperation, ScatterReductionKind};
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::backends::arrays::Array;
/// # use ryft_core::types::{ArrayType, DataType, Shape, Size};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Add two row updates into rows 0 and 2 of a 3x2 zero matrix. Each query is a scalar row index, so the indices
/// // have shape [2, 1] and each update window is a full row (update window axis 1).
/// let operand = Array::matrix(3, 2, vec![0.0; 6]);
/// let indices = Array::from_f64s(
///     ArrayType::new(DataType::I32, Shape::new(vec![Size::Static(2), Size::Static(1)])),
///     vec![0.0, 2.0],
/// );
/// let updates = Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
/// let dimensions = ScatterDimensionNumbers::new(vec![1], vec![0], vec![0]);
/// let operation = ScatterOperation::new(dimensions, ScatterReductionKind::Add);
/// let result = operand.scatter(&indices, &updates, &operation)?;
/// // `result` is [[1, 2], [0, 0], [3, 4]].
/// assert_eq!(result.to_f64s(), vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0]);
/// # Ok(())
/// # }
/// ```
pub trait Scatter: Sized {
    /// Scatters `updates` into `self` (the operand) at the positions named by `indices`, according to `operation`.
    fn scatter(&self, indices: &Self, updates: &Self, operation: &ScatterOperation) -> Result<Self, ProgramError>;
}

impl Scatter for ArrayType {
    /// Type-level scatter: validates the dimension numbers, the updates shape, and the data types, and computes the
    /// output type (which equals the operand type) and placement.
    fn scatter(&self, indices: &Self, updates: &Self, operation: &ScatterOperation) -> Result<Self, ProgramError> {
        let operand = self;
        let dimensions = operation.dimensions();
        let operand_rank = operand.rank();
        let indices_rank = indices.rank();
        let updates_rank = updates.rank();

        if indices_rank == 0 {
            return Err(TypeError {
                message: format!(
                    "'{SCATTER_OPERATION_NAME}' indices must have rank at least 1 (the trailing index vector)"
                ),
            }
            .into());
        }
        if !indices.data_type().is_integer() {
            return Err(TypeError {
                message: format!("'{SCATTER_OPERATION_NAME}' indices must be integer-typed but have type {indices}"),
            }
            .into());
        }
        if operand.memory() != indices.memory() || operand.memory() != updates.memory() {
            return Err(TypeError {
                message: format!(
                    "'{SCATTER_OPERATION_NAME}' operand, indices, and updates must share one memory space but reside \
                     in {}, {}, and {}",
                    operand.memory(),
                    indices.memory(),
                    updates.memory(),
                ),
            }
            .into());
        }
        if updates.data_type() != operand.data_type() {
            return Err(TypeError {
                message: format!(
                    "'{SCATTER_OPERATION_NAME}' updates data type {} does not match operand data type {}",
                    updates.data_type(),
                    operand.data_type(),
                ),
            }
            .into());
        }
        let index_vector_dimension = indices_rank - 1;
        let Size::Static(index_vector_extent) = indices.dimension(index_vector_dimension as isize) else {
            return Err(TypeError {
                message: format!("'{SCATTER_OPERATION_NAME}' indices index vector dimension must have a static extent"),
            }
            .into());
        };

        validate_sorted_unique_in_range(
            SCATTER_OPERATION_NAME,
            "update_window_dimensions",
            dimensions.update_window_dimensions(),
            updates_rank,
        )?;
        validate_sorted_unique_in_range(
            SCATTER_OPERATION_NAME,
            "inserted_window_dimensions",
            dimensions.inserted_window_dimensions(),
            operand_rank,
        )?;
        validate_sorted_unique_in_range(
            SCATTER_OPERATION_NAME,
            "operand_batching_dimensions",
            dimensions.operand_batching_dimensions(),
            operand_rank,
        )?;
        if dimensions.scatter_dimensions_to_operand_dimensions().len() != index_vector_extent {
            return Err(TypeError {
                message: format!(
                    "'{SCATTER_OPERATION_NAME}' scatter_dimensions_to_operand_dimensions has length {} but the index \
                     vector extent is {index_vector_extent}",
                    dimensions.scatter_dimensions_to_operand_dimensions().len(),
                ),
            }
            .into());
        }
        validate_unique_in_range(
            SCATTER_OPERATION_NAME,
            "scatter_dimensions_to_operand_dimensions",
            dimensions.scatter_dimensions_to_operand_dimensions(),
            operand_rank,
        )?;
        if dimensions.scatter_indices_batching_dimensions().len() != dimensions.operand_batching_dimensions().len() {
            return Err(TypeError {
                message: format!(
                    "'{SCATTER_OPERATION_NAME}' operand and scatter-indices batching dimensions must align 1:1, but got \
                     {} and {}",
                    dimensions.operand_batching_dimensions().len(),
                    dimensions.scatter_indices_batching_dimensions().len(),
                ),
            }
            .into());
        }
        for &dimension in dimensions.scatter_indices_batching_dimensions() {
            if dimension >= indices_rank || dimension == index_vector_dimension {
                return Err(TypeError {
                    message: format!(
                        "'{SCATTER_OPERATION_NAME}' scatter_indices_batching_dimensions entry {dimension} is out of \
                         range or names the index vector dimension"
                    ),
                }
                .into());
            }
        }

        let inserted: BTreeSet<usize> = dimensions.inserted_window_dimensions().iter().copied().collect();
        let operand_batching: BTreeSet<usize> = dimensions.operand_batching_dimensions().iter().copied().collect();
        if inserted.intersection(&operand_batching).next().is_some() {
            return Err(TypeError {
                message: format!(
                    "'{SCATTER_OPERATION_NAME}' inserted_window_dimensions and operand_batching_dimensions must be \
                     disjoint"
                ),
            }
            .into());
        }

        // Rank decomposition: the operand axes split into window, inserted, and batching axes; the updates axes split
        // into window axes and the scatter/batch axes carried from the indices (every indices axis but the index
        // vector).
        if operand_rank != dimensions.update_window_dimensions().len() + inserted.len() + operand_batching.len() {
            return Err(TypeError {
                message: format!(
                    "'{SCATTER_OPERATION_NAME}' operand rank {operand_rank} must equal update_window + inserted_window + \
                     operand_batching dimension counts"
                ),
            }
            .into());
        }
        if updates_rank != (indices_rank - 1) + dimensions.update_window_dimensions().len() {
            return Err(TypeError {
                message: format!(
                    "'{SCATTER_OPERATION_NAME}' updates rank {updates_rank} must equal (indices rank - 1) + the update \
                     window dimension count"
                ),
            }
            .into());
        }

        // Window-size checks: the operand window axes (operand axes that are neither inserted nor batching, in order)
        // pair 1:1 with the sorted update window axes; each update window extent must fit within the operand window
        // extent.
        let operand_window_axes: Vec<usize> = (0..operand_rank)
            .filter(|axis| !inserted.contains(axis) && !operand_batching.contains(axis))
            .collect();
        for (&operand_axis, &update_axis) in operand_window_axes.iter().zip(dimensions.update_window_dimensions()) {
            if let (Size::Static(update_extent), Size::Static(operand_extent)) =
                (updates.dimension(update_axis as isize), operand.dimension(operand_axis as isize))
                && update_extent > operand_extent
            {
                return Err(TypeError {
                    message: format!(
                        "'{SCATTER_OPERATION_NAME}' update window axis {update_axis} extent {update_extent} exceeds \
                         the operand window axis {operand_axis} extent {operand_extent}"
                    ),
                }
                .into());
            }
        }

        // The updates' scatter/batch axes (every updates axis but the window axes) must match the indices' batch axes
        // (every indices axis but the index vector), in order.
        let update_window: BTreeSet<usize> = dimensions.update_window_dimensions().iter().copied().collect();
        let update_scatter_axes: Vec<usize> = (0..updates_rank).filter(|axis| !update_window.contains(axis)).collect();
        let indices_batch_axes: Vec<usize> = (0..indices_rank).filter(|axis| *axis != index_vector_dimension).collect();
        for (&update_axis, &indices_axis) in update_scatter_axes.iter().zip(&indices_batch_axes) {
            if updates.dimension(update_axis as isize) != indices.dimension(indices_axis as isize) {
                return Err(TypeError {
                    message: format!(
                        "'{SCATTER_OPERATION_NAME}' updates scatter axis {update_axis} must match indices batch axis \
                         {indices_axis} in extent"
                    ),
                }
                .into());
            }
        }

        // Batching extents must match between operand and indices.
        for (&operand_axis, &indices_axis) in dimensions
            .operand_batching_dimensions()
            .iter()
            .zip(dimensions.scatter_indices_batching_dimensions())
        {
            if operand.dimension(operand_axis as isize) != indices.dimension(indices_axis as isize) {
                return Err(TypeError {
                    message: format!(
                        "'{SCATTER_OPERATION_NAME}' batching dimensions must have equal extents, but operand axis \
                         {operand_axis} and indices axis {indices_axis} differ"
                    ),
                }
                .into());
            }
        }

        // Output sharding (JAX's `_scatter_sharding_rule`). The output is distributed exactly like the operand
        // (scatter writes in place), so the operand sharding carries through after validating, gated to explicit mesh
        // axes, that the operand axes the start indices target (`scatter_dimensions_to_operand_dimensions` and the
        // inserted axes) are replicated and that the index vector axis is replicated. An ambiguous placement requires a
        // requested `output_sharding`, mirroring `dot`/`reduce`.
        let sharding = if let Some(requested) = operation.output_sharding() {
            if requested.rank() != operand.rank() {
                return Err(TypeError {
                    message: format!(
                        "'{SCATTER_OPERATION_NAME}' output sharding rank ({}) does not match the operand rank ({})",
                        requested.rank(),
                        operand.rank(),
                    ),
                }
                .into());
            }
            if references_auto_axis(requested) {
                return Err(TypeError {
                    message: format!("'{SCATTER_OPERATION_NAME}' output sharding cannot reference auto mesh axes"),
                }
                .into());
            }
            Some(requested.clone())
        } else if let Some(operand_sharding) = operand.sharding() {
            let mesh = operand_sharding.mesh().clone();
            check_same_mesh(&mesh, indices.sharding())?;
            check_same_mesh(&mesh, updates.sharding())?;
            let replicated_operand_axes: BTreeSet<usize> = dimensions
                .scatter_dimensions_to_operand_dimensions()
                .iter()
                .chain(dimensions.inserted_window_dimensions())
                .copied()
                .collect();
            for &axis in &replicated_operand_axes {
                if dimension_has_explicit_axis(&mesh, &operand_sharding.dimensions()[axis]) {
                    return Err(TypeError {
                        message: format!(
                            "'{SCATTER_OPERATION_NAME}' operand axis {axis} is targeted by the start indices and must be \
                             replicated over explicit mesh axes; request an explicit output sharding to resolve \
                             placement"
                        ),
                    }
                    .into());
                }
            }
            if let Some(indices_sharding) = indices.sharding()
                && dimension_has_explicit_axis(&mesh, &indices_sharding.dimensions()[index_vector_dimension])
            {
                return Err(TypeError {
                    message: format!(
                        "'{SCATTER_OPERATION_NAME}' indices index vector dimension must be replicated over explicit \
                         mesh axes"
                    ),
                }
                .into());
            }
            // The output is the in-place-updated operand, so it keeps the operand's placement and reduction state.
            Some(operand_sharding.clone())
        } else {
            None
        };
        operand
            .clone()
            .with_sharding(sharding)
            .map_err(|error| TypeError { message: error.to_string() }.into())
    }
}

/// Any context-carrying value scatters by binding a [`ScatterOperation`] through its own context. The
/// `From<ScatterOperation>` bound makes this disjoint from the eager value types (whose context operation is
/// `ConstantOperation`), so it covers the transform tracers without conflicting with the concrete implementations.
impl<V: Value<Type = ArrayType>> Scatter for V
where
    V::DispatchDomain: Context<Type = ArrayType>,
    <V::DispatchDomain as Domain>::Operation: From<ScatterOperation>,
{
    fn scatter(&self, indices: &Self, updates: &Self, operation: &ScatterOperation) -> Result<Self, ProgramError> {
        let mut outputs = self.dispatch_domain().bind(
            operation.clone(),
            Vec::new(),
            &[self.clone(), indices.clone(), updates.clone()],
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::contexts::Context;
    use crate::macros::{
        check_operation_batching, check_operation_partial_evaluation, check_operation_transposition,
        check_operation_type_inference,
    };
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tracing_v2::jacfwd;
    use crate::types::{DataType, Layout, Memory, Shape, Size, StridedLayout};

    use super::*;

    fn indices_type(dimensions: Vec<usize>) -> ArrayType {
        ArrayType::new(DataType::I32, Shape::new(dimensions.into_iter().map(Size::Static).collect()))
    }

    fn float_type(dimensions: Vec<usize>) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(dimensions.into_iter().map(Size::Static).collect()))
    }

    /// Lifts a constant integer index array into the trace or differentiation context that `exemplar` belongs to.
    fn index_array<V>(exemplar: &V, shape: Vec<usize>, values: Vec<f64>) -> V
    where
        V: crate::programs::Value<Type = ArrayType>,
        V::DispatchDomain: crate::contexts::Context<Constant = Array>,
    {
        let r#type = ArrayType::new(DataType::I32, Shape::new(shape.into_iter().map(Size::Static).collect()));
        exemplar.dispatch_domain().lift(Array::from_f64s(r#type, values)).unwrap()
    }

    #[test]
    fn test_scatter() {
        // Scatter-add row updates into a [3, 2] operand indexed by a [2, 1] index array: update window axis 1 carries
        // the row, operand axis 0 is inserted (start-index driven).
        let dimensions = ScatterDimensionNumbers::new(vec![1], vec![0], vec![0]);
        let operation = ScatterOperation::new(dimensions, ScatterReductionKind::Add);
        assert_eq!(operation.name(), SCATTER_OPERATION_NAME);
        assert_eq!(operation.kind(), ScatterReductionKind::Add);

        let operand = float_type(vec![3, 2]);
        let indices = indices_type(vec![2, 1]);
        let updates = float_type(vec![2, 2]);
        let host_operand = operand.clone().with_memory(Memory::Host { pinned: true });
        let host_indices = indices.clone().with_memory(Memory::Host { pinned: true });
        let host_updates = updates.clone().with_memory(Memory::Host { pinned: true });
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [
                {
                    input_types = [operand.clone(), indices.clone(), updates.clone()],
                    output_types = [operand.clone()],
                },
                {
                    input_types = [operand.clone(), indices.clone()],
                    error = "expected 3 inputs but got 2",
                },
                {
                    input_types = [operand.clone(), indices.clone(), indices_type(vec![2, 2])],
                    error = "'scatter' updates data type i32 does not match operand data type f32",
                },
                {
                    input_types = [operand.clone(), float_type(vec![2, 1]), updates.clone()],
                    error = "'scatter' indices must be integer-typed but have type f32[2, 1]",
                },
                {
                    input_types = [host_operand.clone(), host_indices, host_updates],
                    output_types = [host_operand.clone()],
                },
                {
                    input_types = [host_operand, indices.clone(), updates.clone()],
                    error = "'scatter' operand, indices, and updates must share one memory space but reside in \
                             Host[Pinned], Device, and Device",
                },
            ],
        );

        assert_eq!(
            format!("{operation}"),
            concat!(
                "scatter [\n",
                "    kind=add,\n",
                "    dimensions=(update_window=[1], inserted_window=[0], scatter_to_operand=[0], operand_batching=[], ",
                "scatter_indices_batching=[]),\n",
                "]",
            ),
        );

        // Interpretation applies each supported combiner and accumulates repeated additive updates.
        let scalar_dimensions = || ScatterDimensionNumbers::new(vec![], vec![0], vec![0]);
        let scalar_indices = Array::from_f64s(indices_type(vec![2, 1]), vec![1.0, 3.0]);
        let run = |kind| {
            Array::vector(vec![1.0, 2.0, 3.0, 4.0])
                .scatter(
                    &scalar_indices,
                    &Array::vector(vec![100.0, 200.0]),
                    &ScatterOperation::new(scalar_dimensions(), kind),
                )
                .unwrap()
                .to_f64s()
        };
        assert_eq!(run(ScatterReductionKind::Add), vec![1.0, 102.0, 3.0, 204.0]);
        assert_eq!(run(ScatterReductionKind::Overwrite), vec![1.0, 100.0, 3.0, 200.0]);
        assert_eq!(run(ScatterReductionKind::Mul), vec![1.0, 200.0, 3.0, 800.0]);
        assert_eq!(run(ScatterReductionKind::Min), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(run(ScatterReductionKind::Max), vec![1.0, 100.0, 3.0, 200.0]);
        let repeated = Array::from_f64s(indices_type(vec![2, 1]), vec![1.0, 1.0]);
        let result = Array::vector(vec![1.0, 2.0, 3.0, 4.0])
            .scatter(
                &repeated,
                &Array::vector(vec![100.0, 200.0]),
                &ScatterOperation::new(scalar_dimensions(), ScatterReductionKind::Add),
            )
            .unwrap();
        assert_eq!(result.to_f64s(), vec![1.0, 302.0, 3.0, 4.0]);

        // Partial evaluation folds fully known scatters and residualizes an unknown data operand.
        let operand_value = Array::matrix(3, 2, vec![0.0; 6]);
        let indices_value = Array::from_f64s(indices_type(vec![2, 1]), vec![0.0, 2.0]);
        let updates_value = Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let expected = Array::matrix(3, 2, vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0]);
        check_operation_partial_evaluation!(
            backend = (Array, ArrayOperation<Array>),
            operation = operation.clone(),
            cases = [
                {
                    inputs = [
                        (@known, operand_value.clone()),
                        (@known, indices_value.clone()),
                        (@known, updates_value.clone()),
                    ],
                    outputs = [(@known, expected.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = operand_value.r#type().into_owned(), replay = operand_value.clone())),
                        (@known, indices_value.clone()),
                        (@known, updates_value.clone()),
                    ],
                    outputs = [(@residual, expected)],
                    residual_instructions = 1,
                },
            ],
        );

        // Batching expands each mapped item independently while preserving replicated indices.
        check_operation_batching!(
            @exact,
            operation = ScatterOperation::new(scalar_dimensions(), ScatterReductionKind::Add),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::matrix(2, 4, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])),
                    (@replicated, scalar_indices),
                    (@mapped(axis = 0), Array::matrix(2, 2, vec![10.0, 20.0, 30.0, 40.0])),
                ],
                outputs = [(@mapped(axis = 0), Array::matrix(
                    2,
                    4,
                    vec![1.0, 12.0, 3.0, 24.0, 5.0, 36.0, 7.0, 48.0],
                ))],
            }],
        );
    }

    #[test]
    fn test_scatter_dimension_numbers() {
        let operand = float_type(vec![3, 2]);
        let indices = indices_type(vec![2, 1]);
        let dimensions = || ScatterDimensionNumbers::new(vec![1], vec![0], vec![0]);

        // updates rank must equal (indices rank - 1) + update window count.
        let operation = ScatterOperation::new(dimensions(), ScatterReductionKind::Add);
        assert_eq!(
            operation.infer_output_types(&[operand.clone(), indices.clone(), float_type(vec![2])], &[]),
            Err(TypeError {
                message: "'scatter' update_window_dimensions entry 1 is out of range for bound 1".to_string(),
            }),
        );
    }

    #[test]
    fn test_array_type_scatter() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let dimensions = || ScatterDimensionNumbers::new(vec![1], vec![0], vec![0]);

        // Operand [4, 2] sharded only on the feature axis (axis 1); the targeted axis 0 is replicated → output keeps
        // the operand sharding.
        let sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["y"])])
                .unwrap();
        let operand = float_type(vec![4, 2]).with_sharding(sharding.clone()).unwrap();
        let indices = indices_type(vec![2, 1]);
        let updates = float_type(vec![2, 2]);
        let operation = ScatterOperation::new(dimensions(), ScatterReductionKind::Add);
        let output = operation.infer_output_types(&[operand, indices.clone(), updates.clone()], &[]).unwrap();
        assert_eq!(output[0].sharding(), Some(&sharding));

        // Sharding the targeted operand axis over an explicit mesh axis is ambiguous without an output sharding.
        let operand = float_type(vec![4, 2])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        let operation = ScatterOperation::new(dimensions(), ScatterReductionKind::Add);
        assert!(operation.infer_output_types(&[operand, indices, updates], &[]).is_err());
    }

    #[test]
    fn test_scatter_differentiation() {
        // The scatter-add pullback leaves the operand cotangent unchanged and gathers the update cotangent.
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Add);
        check_operation_transposition!(
            @exact,
            operation = operation,
            cases = [{
                inputs = [
                    (@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![4.into()])))),
                    (@known, Array::from_f64s(indices_type(vec![2, 1]), vec![1.0, 3.0])),
                    (@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![2.into()])))),
                ],
                output_cotangents = [Array::vector(vec![1.0, 2.0, 3.0, 4.0])],
                input_cotangents = [
                    Array::vector(vec![1.0, 2.0, 3.0, 4.0]),
                    Array::vector(vec![2.0, 4.0]),
                ],
            }],
        );

        // The dual gather restores the update cotangent's complete layout-bearing type.
        let operand_type =
            ArrayType::new(DataType::F64, Shape::new(vec![4.into()])).with_memory(Memory::Host { pinned: true });
        let update_type = ArrayType::new(DataType::F64, Shape::new(vec![2.into()]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![8])))
            .with_memory(Memory::Host { pinned: true });
        let indices =
            Array::from_f64s(indices_type(vec![2, 1]).with_memory(Memory::Host { pinned: true }), vec![1.0, 3.0]);
        check_operation_transposition!(
            @exact,
            operation = ScatterOperation::new(
                ScatterDimensionNumbers::new(vec![], vec![0], vec![0]),
                ScatterReductionKind::Add,
            ),
            cases = [{
                inputs = [
                    (@linear(type = operand_type.clone())),
                    (@known, indices),
                    (@linear(type = update_type.clone())),
                ],
                output_cotangents = [Array::from_f64s(operand_type.clone(), vec![1.0, 2.0, 3.0, 4.0])],
                input_cotangents = [
                    Array::from_f64s(operand_type, vec![1.0, 2.0, 3.0, 4.0]),
                    Array::from_f64s(update_type, vec![2.0, 4.0]),
                ],
            }],
        );

        // Forward mode through `f(x) = scatter_add(x, [[1], [3]], [10, 20])` exercises the captured-index scatter-add
        // under batched basis tangents (the per-item batch rule). Scatter-add is the identity in its operand, so the
        // Jacobian with respect to `x` is the identity matrix.
        let jacobian = jacfwd(
            |x| {
                let indices = index_array(&x, vec![2, 1], vec![1.0, 3.0]);
                let updates = x.context().lift(Array::vector(vec![10.0, 20.0]))?;
                let operation = ScatterOperation::new(
                    ScatterDimensionNumbers::new(vec![], vec![0], vec![0]),
                    ScatterReductionKind::Add,
                );
                Ok(x.scatter(&indices, &updates, &operation).unwrap())
            },
            Array::vector(vec![1.0, 2.0, 3.0, 4.0]),
        )
        .unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.output_type().static_shape().unwrap().as_slice(), &[4]);
        assert_eq!(block.input_type().static_shape().unwrap().as_slice(), &[4]);
        assert_eq!(
            block.value().values(),
            &[
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 1.0, //
            ],
        );
    }
}
