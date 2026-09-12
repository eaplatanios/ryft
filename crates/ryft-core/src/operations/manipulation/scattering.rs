use std::collections::BTreeSet;
use std::fmt::Display;

use crate::arrays::{
    Array, ArrayAddressing, ArrayBatch, ArrayBatchingPolicy, ArrayElement, ArrayExtentBatchingPolicy, ArrayIrType,
    ArrayIrValue, ArrayType, DataType, Dimension, LogicalMesh, NumericArrayElement, Shape, Sharding, i1, i2, i4,
    materialize_array_tangent, u1, u2, u4,
};
use crate::axes::Axis;
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain, EagerContext, ProjectedContext, StagingContext};
use crate::differentiation::{
    CotangentAccumulator, DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDriver,
    DifferentiationDual, DifferentiationError, DifferentiationPolicy, ElementwiseDerivativeAlignment,
    MemberDifferentiableOperation, ResidualZeroProvider, TransposableOperation, TranspositionContext,
    TranspositionDriver, jvp_projected_operation,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, dispatch_on_array_element_type};
use crate::operations::compare::{CompareOperation, ComparisonDirection};
use crate::operations::constants::constant::DimensionConstant;
use crate::operations::constants::iota::IotaOperation;
use crate::operations::constants::one_like::OneLikeOperation;
use crate::operations::constants::zero::{Zero, ZeroOperation};
use crate::operations::constants::zero_like::ZeroLikeOperation;
use crate::operations::control_flow::select::SelectOperation;
use crate::operations::dimensions::dimension_size::DimensionSize;
use crate::operations::manipulation::broadcasting::{Broadcast, BroadcastOperation};
use crate::operations::manipulation::conversions::ConvertElementTypeOperation;
use crate::operations::manipulation::gathering::{
    GatherDimensionNumbers, GatherOperation, GatherScatterMode, dimension_has_explicit_axis,
    dimensions_have_equal_extents, validate_sorted_unique_in_range, validate_unique_in_range,
};
use crate::operations::manipulation::reshaping::{
    DynamicReshape, Reshape, ReshapeOperation, lift_output_sharding_for_leading_batch_axis,
};
use crate::operations::manipulation::transposition::Transpose;
use crate::operations::math::add::AddOperation;
use crate::operations::math::div::DivOperation;
use crate::operations::math::mul::MulOperation;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, OperationProjection, ProgramError, RegionInterface, TypeError, Typed,
    Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

// TODO(eaplatanios): Review this.

/// Combiner applied when a [`scatter`](Scatter) writes an update into the input. Each kind selects the binary
/// reduction used where an update meets the existing input value and lowers to the corresponding
/// `stablehlo.scatter` combiner region. [`Add`](Self::Add) supports differentiation through both the input and
/// updates and participates in the gather/scatter-add transpose duality. Extremal derivatives divide ties equally
/// among matching inputs and updates. Multiplication derivatives with respect to updates require unique indices;
/// repeated overwrite uses a consistent winning update for its primal and tangent. Nonlinear derivatives require
/// static update window sizes, and repeated overwrite additionally requires a static update shape. Overlapping
/// updates may execute in any order; a unique-index hint is a caller promise, not a check.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ScatterReductionKind {
    /// The update replaces the input value (StableHLO's scatter whose combiner returns the update).
    Overwrite,

    /// The update is added to the input value (`scatter_add`). Linear for all index configurations.
    Add,

    /// The update is multiplied with the input value (`scatter_mul`).
    Mul,

    /// The input value is replaced by the minimum of itself and the update (`scatter_min`). Booleans use
    /// conjunction, real numeric values propagate NaNs and order negative zero below positive zero, and complex values
    /// compare lexicographically by `(real, imaginary)`.
    Min,

    /// The input value is replaced by the maximum of itself and the update (`scatter_max`). Booleans use
    /// disjunction, real numeric values propagate NaNs and order negative zero below positive zero, and complex values
    /// compare lexicographically by `(real, imaginary)`.
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

    /// Returns `true` when this kind is a linear map in the input and updates (only [`Add`](Self::Add) is). Linear
    /// kinds participate in the gather/scatter-add transpose duality. Other kinds use coefficients computed from the
    /// primals; overwrite is also linear when its operation promises unique indices.
    pub fn is_linear(self) -> bool {
        matches!(self, Self::Add)
    }
}

impl Display for ScatterReductionKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.name())
    }
}

/// Specification of how the index input and the update windows map onto the input axes of a [`scatter`](Scatter),
/// following StableHLO's [`scatter`](https://openxla.org/stablehlo/spec#scatter) dimension numbers. It is the
/// structural dual of [`GatherDimensionNumbers`]:
/// [`update_window_dimensions`](Self::update_window_dimensions) mirrors `offset_dimensions`,
/// [`inserted_window_dimensions`](Self::inserted_window_dimensions) mirrors `collapsed_slice_dimensions`, and
/// [`scatter_dimensions_to_operand_dimensions`](Self::scatter_dimensions_to_operand_dimensions) mirrors
/// `start_index_map`.
///
/// The index vector dimension is implicit and always the last axis of the indices input. The output has the same
/// shape as the input.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ScatterDimensionNumbers {
    /// Axes of the updates input that hold a scattered window, in ascending order. Their count equals the number of
    /// input axes that are neither inserted nor batching.
    update_window_dimensions: Vec<usize>,

    /// Input axes whose window size is `1` and that have no corresponding updates axis, in ascending order.
    inserted_window_dimensions: Vec<usize>,

    /// For each component of a start-index vector (the last axis of the indices input), the input axis it scatters
    /// into. Its length equals the extent of the indices' index vector dimension.
    scatter_dimensions_to_operand_dimensions: Vec<usize>,

    /// Input axes batched against [`scatter_indices_batching_dimensions`](Self::scatter_indices_batching_dimensions),
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

    /// Returns the updates window axes.
    #[inline]
    pub fn update_window_dimensions(&self) -> &[usize] {
        &self.update_window_dimensions
    }

    /// Returns the inserted (size-1, input-only) axes.
    #[inline]
    pub fn inserted_window_dimensions(&self) -> &[usize] {
        &self.inserted_window_dimensions
    }

    /// Returns the scatter-index-to-input-axis map.
    #[inline]
    pub fn scatter_dimensions_to_operand_dimensions(&self) -> &[usize] {
        &self.scatter_dimensions_to_operand_dimensions
    }

    /// Returns the input batching axes.
    #[inline]
    pub fn operand_batching_dimensions(&self) -> &[usize] {
        &self.operand_batching_dimensions
    }

    /// Returns the indices batching axes.
    #[inline]
    pub fn scatter_indices_batching_dimensions(&self) -> &[usize] {
        &self.scatter_indices_batching_dimensions
    }
    /// Attaches the input/indices batching axis pair (aligned 1:1).
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

/// Canonical operation name for [`ScatterOperation`].
pub const SCATTER_OPERATION_NAME: &str = "scatter";

/// [`Operation`] that writes update windows into a copy of an input at positions named by an integer index input,
/// combining overlaps with a [`ScatterReductionKind`]. Refer to the documentation of [`Scatter`] for the semantics.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ScatterOperation {
    /// Dimension numbers mapping the index input and update windows onto the input axes.
    dimensions: ScatterDimensionNumbers,

    /// Combiner applied where an update meets the existing input value.
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

    /// Promises that update windows do not overlap. Backends may use this promise to simplify execution.
    /// Differentiating multiplicative updates requires this promise when the updates carry nonzero tangents.
    #[inline]
    pub fn with_unique_indices(mut self, unique_indices: bool) -> Self {
        self.unique_indices = unique_indices;
        self
    }

    /// Requests `output_sharding` for the result. Without an explicit request, indexed axes with partial update
    /// windows must be replicated; complete windows preserve their input placement. An explicit request selects the
    /// result placement while preserving its mesh, reduction state, and manual-axis variation.
    #[inline]
    pub fn with_output_sharding(mut self, output_sharding: impl Into<Option<Sharding>>) -> Self {
        self.output_sharding = output_sharding.into();
        self
    }

    /// Shares coefficient construction between homogeneous and projected differentiation. Coefficients depend only
    /// on primals; the staged tangent graph uses ordinary linear gather/scatter and elementwise operations.
    fn linearize_values<C>(
        &self,
        contexts: (&C, &C),
        inputs: [&C::Value; 3],
        primal: C::Value,
        tangents: [(&C::Value, bool); 2],
        primal_to_tangent: impl Fn(C::Value) -> Result<C::Value, ProgramError>,
    ) -> Result<(C::Value, C::Value), ProgramError>
    where
        C: Context<Type = ArrayType>,
        C::Operation: From<IotaOperation<ArrayType>>
            + From<ScatterOperation>
            + From<GatherOperation>
            + From<ZeroLikeOperation<ArrayType>>
            + From<OneLikeOperation<ArrayType>>
            + From<CompareOperation<ArrayType>>
            + From<SelectOperation<ArrayType>>
            + From<ConvertElementTypeOperation<ArrayType>>
            + From<AddOperation<ArrayType>>
            + From<MulOperation<ArrayType>>
            + From<DivOperation<ArrayType>>
            + From<ReshapeOperation>
            + From<BroadcastOperation>,
    {
        let (context, tangent_context) = contexts;
        let [input, indices, updates] = inputs;
        let [(input_tangent, input_is_zero), (updates_tangent, updates_are_zero)] = tangents;
        let bind = |context: &C, operation: C::Operation, inputs: &[C::Value]| {
            let mut outputs = context.bind(operation, Vec::new(), inputs)?;
            check_count!("output", outputs, 1, ProgramError);
            Ok::<_, ProgramError>(outputs.remove(0))
        };
        let zero =
            |context: &C, value: &C::Value| bind(context, ZeroLikeOperation::new().into(), std::slice::from_ref(value));
        let one =
            |context: &C, value: &C::Value| bind(context, OneLikeOperation::new().into(), std::slice::from_ref(value));
        let add = |context: &C, lhs: &C::Value, rhs: &C::Value| {
            bind(context, AddOperation::new().into(), &[lhs.clone(), rhs.clone()])
        };
        let mul = |context: &C, lhs: &C::Value, rhs: &C::Value| {
            bind(context, MulOperation::new().into(), &[lhs.clone(), rhs.clone()])
        };
        let div = |context: &C, lhs: &C::Value, rhs: &C::Value| {
            bind(context, DivOperation::new().into(), &[lhs.clone(), rhs.clone()])
        };
        let equal = |context: &C, lhs: &C::Value, rhs: &C::Value| {
            bind(context, CompareOperation::new(ComparisonDirection::Equal).into(), &[lhs.clone(), rhs.clone()])
        };
        let convert = |context: &C, value: &C::Value, data_type: DataType| {
            bind(context, ConvertElementTypeOperation::new(data_type, false).into(), std::slice::from_ref(value))
        };
        let select = |context: &C, condition: &C::Value, on_true: &C::Value, on_false: &C::Value| {
            bind(context, SelectOperation::new().into(), &[condition.clone(), on_true.clone(), on_false.clone()])
        };
        let scatter = |context: &C, input: &C::Value, indices: &C::Value, updates: &C::Value, operation: &Self| {
            bind(context, operation.clone().into(), &[input.clone(), indices.clone(), updates.clone()])
        };
        let gather_values = |context: &C, input: &C::Value, indices: &C::Value, operation: &GatherOperation| {
            bind(context, operation.clone().into(), &[input.clone(), indices.clone()])
        };
        let align = |context: &C, value: &C::Value, target: &ArrayType| {
            if value.r#type().as_ref() == target {
                Ok(value.clone())
            } else {
                bind(
                    context,
                    BroadcastOperation::new(target.clone(), (0..target.rank()).collect()).into(),
                    std::slice::from_ref(value),
                )
            }
        };
        if self.kind() == ScatterReductionKind::Add
            || (self.kind() == ScatterReductionKind::Overwrite && self.unique_indices())
        {
            return Ok((
                primal,
                scatter(tangent_context, input_tangent, &primal_to_tangent(indices.clone())?, updates_tangent, self)?,
            ));
        }
        let zeros = zero(tangent_context, input_tangent)?;
        let update_zeros = zero(tangent_context, updates_tangent)?;
        let mut additive = self.clone();
        additive.kind = ScatterReductionKind::Add;
        if self.kind() == ScatterReductionKind::Mul {
            if !updates_are_zero && !self.unique_indices() {
                return Err(ProgramError::UnsupportedOperation {
                    message:
                        "`scatter` multiplication derivatives with respect to updates require `unique_indices=true`"
                            .to_string(),
                });
            }
            // Omit structural-zero terms before multiplication so an inactive derivative never becomes `0 * inf`.
            let input_contribution = if input_is_zero {
                None
            } else {
                let coefficient = scatter(context, &one(context, input)?, indices, updates, self)?;
                let input_tangent = align(tangent_context, input_tangent, coefficient.r#type().as_ref())?;
                Some(mul(tangent_context, &input_tangent, &primal_to_tangent(coefficient)?)?)
            };
            let update_contribution = if updates_are_zero {
                None
            } else {
                let updates =
                    scatter(tangent_context, &zeros, &primal_to_tangent(indices.clone())?, updates_tangent, &additive)?;
                let coefficient = align(context, input, updates.r#type().as_ref())?;
                Some(mul(tangent_context, &primal_to_tangent(coefficient)?, &updates)?)
            };
            let tangent = match (input_contribution, update_contribution) {
                (Some(input), Some(updates)) => add(tangent_context, &input, &updates)?,
                (Some(value), None) | (None, Some(value)) => value,
                (None, None) => zeros,
            };
            return Ok((primal, tangent));
        }
        let dimensions = self.dimensions();
        let input_type = input.r#type();
        let updates_type = updates.r#type();
        let mut slice_sizes = Vec::with_capacity(input_type.rank());
        let mut window_position = 0;
        for axis in 0..input_type.rank() {
            if dimensions.inserted_window_dimensions().contains(&axis)
                || dimensions.operand_batching_dimensions().contains(&axis)
            {
                slice_sizes.push(usize::from(input_type.dimension(axis) != Dimension::Static(0)));
            } else {
                let update_axis = dimensions.update_window_dimensions()[window_position];
                slice_sizes.push(updates_type.dimension(update_axis).value().ok_or_else(|| {
                    ProgramError::UnsupportedOperation {
                        message: format!("`scatter` nonlinear differentiation requires a static update window on axis `{update_axis}`"),
                    }
                })?);
                window_position += 1;
            }
        }
        let gather_dimensions = GatherDimensionNumbers::new(
            dimensions.update_window_dimensions().to_vec(),
            dimensions.inserted_window_dimensions().to_vec(),
            dimensions.scatter_dimensions_to_operand_dimensions().to_vec(),
        )
        .with_batching_dimensions(
            dimensions.operand_batching_dimensions().to_vec(),
            dimensions.scatter_indices_batching_dimensions().to_vec(),
        );
        let mut gather = GatherOperation::new(gather_dimensions, slice_sizes)
            .with_mode(self.mode())
            .with_output_sharding(updates_type.sharding().cloned());
        if self.kind() == ScatterReductionKind::Overwrite {
            // Distinct positive IDs select one winning update at each output element. Reconstruct both primal
            // and tangent from those same winners, since duplicate overwrite order is unspecified by the backend.
            let update_shape = updates_type.static_shape().ok_or_else(|| ProgramError::UnsupportedOperation {
                message: "`scatter` overwrite differentiation with repeated indices requires a static update shape"
                    .to_string(),
            })?;
            let count = update_shape
                .as_slice()
                .iter()
                .try_fold(1usize, |count, size| count.checked_mul(*size))
                .ok_or_else(|| TypeError::invalid("`scatter` update ID count overflows `usize`"))?;
            if count == usize::MAX {
                return Err(TypeError::invalid("`scatter` update IDs overflow `u64`").into());
            }
            let id_type = ArrayType::new_static(DataType::U64, [count]).with_memory(updates_type.memory());
            let mut ids = context.bind(IotaOperation::new(id_type, 0)?, Vec::new(), &[])?;
            check_count!("output", ids, 1, ProgramError);
            let ids = bind(context, ReshapeOperation::new(updates_type.shape().clone()).into(), &[ids.remove(0)])?;
            let ids = add(context, &ids, &one(context, &ids)?)?;
            // IDs are discrete selectors, so preserve their array placement while clearing data reduction state.
            let update_ids_type = updates_type.without_reduction_axes().with_data_type(DataType::U64).with_layout(None);
            let ids = bind(
                context,
                BroadcastOperation::new(update_ids_type.clone(), (0..updates_type.rank()).collect()).into(),
                &[ids],
            )?;
            let zero_ids = convert(context, &zero(context, input)?, DataType::U64)?;
            let input_ids_type = input_type.without_reduction_axes().with_data_type(DataType::U64).with_layout(None);
            let zero_ids = bind(
                context,
                BroadcastOperation::new(input_ids_type, (0..input_type.rank()).collect()).into(),
                &[zero_ids],
            )?;
            let id_operation =
                self.clone().with_output_sharding(primal.r#type().without_reduction_axes().sharding().cloned());
            let scattered_ids = scatter(context, &zero_ids, indices, &ids, &id_operation)?;
            gather = gather.with_output_sharding(update_ids_type.sharding().cloned());
            if self.mode() == GatherScatterMode::FillOrDrop {
                gather = gather.with_fill_value(Array::scalar(0u64)?)?;
            }
            let gathered_ids = gather_values(context, &scattered_ids, indices, &gather)?;
            let input_ids = align(context, &scattered_ids, zero_ids.r#type().as_ref())?;
            let input_selected = equal(context, &input_ids, &zero_ids)?;
            let update_selected = equal(context, &ids, &gathered_ids)?;
            let primal_input = select(context, &input_selected, input, &zero(context, input)?)?;
            let primal_updates = select(context, &update_selected, updates, &zero(context, updates)?)?;
            let tangent_input = select(tangent_context, &primal_to_tangent(input_selected)?, input_tangent, &zeros)?;
            let tangent_updates =
                select(tangent_context, &primal_to_tangent(update_selected)?, updates_tangent, &update_zeros)?;
            return Ok((
                scatter(context, &primal_input, indices, &primal_updates, &additive)?,
                scatter(
                    tangent_context,
                    &tangent_input,
                    &primal_to_tangent(indices.clone())?,
                    &tangent_updates,
                    &additive,
                )?,
            ));
        }
        // Each tied extremum receives equal weight, including the original input when it is also retained.
        // Input masks use input placement even when the requested result placement differs.
        let input_primal = align(context, &primal, input.r#type().as_ref())?;
        let selected_input = equal(context, input, &input_primal)?;
        // An untouched input is an identity edge even when it contains NaN and equality is false.
        let references = scatter(context, &zero(context, input)?, indices, &one(context, updates)?, &additive)?;
        let references = align(context, &references, input.r#type().as_ref())?;
        let untouched = equal(context, &references, &zero(context, &references)?)?;
        let selected_input = select(context, &untouched, &equal(context, &references, &references)?, &selected_input)?;
        let targets = gather_values(context, &primal, indices, &gather)?;
        let selected_updates = equal(context, updates, &targets)?;
        let input_count = convert(context, &selected_input, input_tangent.r#type().data_type())?;
        let update_count = convert(context, &selected_updates, updates_tangent.r#type().data_type())?;
        let count = scatter(context, &input_count, indices, &update_count, &additive)?;
        // NaN extrema compare unequal to every source. No source receives a tangent at those locations.
        let count = select(context, &equal(context, &count, &zero(context, &count)?)?, &one(context, &count)?, &count)?;
        let selected_input_tangent =
            select(tangent_context, &primal_to_tangent(selected_input)?, input_tangent, &zeros)?;
        let selected_update_tangent =
            select(tangent_context, &primal_to_tangent(selected_updates)?, updates_tangent, &update_zeros)?;
        let numerator = scatter(
            tangent_context,
            &selected_input_tangent,
            &primal_to_tangent(indices.clone())?,
            &selected_update_tangent,
            &additive,
        )?;
        // Normalize in the primal context so the linear region only multiplies by a constant coefficient.
        // Multiplication also preserves the dual reduction state when this rule is transposed.
        let coefficient = div(context, &one(context, &count)?, &count)?;
        Ok((primal, mul(tangent_context, &numerator, &primal_to_tangent(coefficient)?)?))
    }
}

impl Display for ScatterOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ScatterOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        SCATTER_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 3, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        match input_types[0].scatter(&input_types[1], &input_types[2], self) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError::invalid(error.to_string())),
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

// Partial evaluation defers to the default fold-or-residualize behavior of
// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for ScatterOperation where
    C::Operation: From<ScatterOperation>
{
}

// Lift window and batching dimensions to carry one leading mapped axis. The input and updates acquire that axis;
// mapped indices use an explicit paired batching dimension, while replicated indices keep it in the update window.
impl<C, P: ArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>> for ScatterOperation
where
    C: Context<Type = ArrayType>,
    C::Value: Broadcast + Transpose,
    ScatterOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        check_count!("input", inputs, 3, ProgramError);
        if inputs.iter().any(|input| !input.ragged_axes().is_empty()) {
            return Err(ProgramError::UnsupportedOperation {
                message: "`scatter` does not support bounded ragged array inputs".to_string(),
            }
            .into());
        }
        if inputs.iter().all(|input| input.batch_axis_position().is_none()) {
            return Ok(self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()])?.into());
        }
        let axis_dimension = P::axis_dimension(context)?;
        for input in inputs {
            if let Some(axis) = input.batch_axis_position()
                && input.r#type().dimension(axis) != axis_dimension
            {
                return Err(BatchingError::MisalignedBatchAxes {
                    message: format!(
                        "`scatter` mapped input extent {} does not match batching extent {axis_dimension}",
                        input.r#type().dimension(axis),
                    ),
                });
            }
        }
        // Every mapped item needs an independent input and update. Matching the leading axis broadcasts
        // replicated values without constructing an intermediate zero array, including for empty batches.
        let input = P::match_axis(context, &inputs[0], Axis::from(0))?;
        let updates = P::match_axis(context, &inputs[2], Axis::from(0))?;
        let dimensions = self.dimensions();
        let shift = |axes: &[usize]| axes.iter().map(|axis| axis + 1).collect::<Vec<_>>();
        let (indices, lifted_dimensions) = if inputs[1].batch_axis_position().is_none() {
            let mut update_window_dimensions = shift(dimensions.update_window_dimensions());
            update_window_dimensions.insert(0, 0);
            (
                inputs[1].clone(),
                ScatterDimensionNumbers::new(
                    update_window_dimensions,
                    shift(dimensions.inserted_window_dimensions()),
                    shift(dimensions.scatter_dimensions_to_operand_dimensions()),
                )
                .with_batching_dimensions(
                    shift(dimensions.operand_batching_dimensions()),
                    dimensions.scatter_indices_batching_dimensions().to_vec(),
                ),
            )
        } else {
            let indices = P::match_axis(context, &inputs[1], Axis::from(0))?;
            let mut input_batching_dimensions = shift(dimensions.operand_batching_dimensions());
            input_batching_dimensions.insert(0, 0);
            let mut indices_batching_dimensions = shift(dimensions.scatter_indices_batching_dimensions());
            indices_batching_dimensions.insert(0, 0);
            (
                indices,
                ScatterDimensionNumbers::new(
                    shift(dimensions.update_window_dimensions()),
                    shift(dimensions.inserted_window_dimensions()),
                    shift(dimensions.scatter_dimensions_to_operand_dimensions()),
                )
                .with_batching_dimensions(input_batching_dimensions, indices_batching_dimensions),
            )
        };
        let operation = Self::new(lifted_dimensions, self.kind())
            .with_mode(self.mode())
            .with_indices_are_sorted(self.indices_are_sorted())
            .with_unique_indices(self.unique_indices())
            .with_output_sharding(
                self.output_sharding()
                    .map(|output_sharding| {
                        lift_output_sharding_for_leading_batch_axis(
                            output_sharding,
                            ArrayBatch::sharding_for_inputs(inputs)?,
                        )
                    })
                    .transpose()?,
            );
        Ok(operation
            .interpret_with_batch_axes(context, &[input, indices, updates], &[BatchAxis::from_position(0)])?
            .into())
    }
}

// Coefficients are constructed in the primal context and transferred through the differentiation boundary before
// they multiply tangent values. Structural zeros avoid constructing inactive products with nonfinite coefficients.
impl<C: Context<Type = ArrayType> + Zero<C::Value>> DifferentiableOperation<C> for ScatterOperation
where
    C::Operation: From<IotaOperation<ArrayType>>
        + From<ScatterOperation>
        + From<GatherOperation>
        + From<ZeroLikeOperation<ArrayType>>
        + From<OneLikeOperation<ArrayType>>
        + From<CompareOperation<ArrayType>>
        + From<SelectOperation<ArrayType>>
        + From<ConvertElementTypeOperation<ArrayType>>
        + From<AddOperation<ArrayType>>
        + From<MulOperation<ArrayType>>
        + From<DivOperation<ArrayType>>
        + From<ReshapeOperation>
        + From<BroadcastOperation>,
    C::Value: Scatter,
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 3, ProgramError);
        let input = &inputs[0];
        let indices = inputs[1].primal();
        let updates = &inputs[2];
        let mut primal = input.primal().scatter(indices, updates.primal(), self)?;
        let tangent = if input.tangent().is_zero() && updates.tangent().is_zero() {
            MaybeZero::Zero(primal.r#type().tangent()?)
        } else {
            let input_tangent = input.tangent().clone().materialize(context.tangent())?;
            let updates_tangent = updates.tangent().clone().materialize(context.tangent())?;
            let (linearized_primal, tangent) = self.linearize_values(
                (context.primal(), context.tangent()),
                [input.primal(), indices, updates.primal()],
                primal.clone(),
                [(&input_tangent, input.tangent().is_zero()), (&updates_tangent, updates.tangent().is_zero())],
                |value| context.primal_to_tangent(value).map_err(ProgramError::from),
            )?;
            // The winner-ID rule also defines the primal overwrite choice.
            if self.kind() == ScatterReductionKind::Overwrite && !self.unique_indices() {
                primal = linearized_primal;
            }
            MaybeZero::Value(tangent)
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

// Partition-aware transpose rule for the primal [`ScatterOperation`] with an [`Add`](ScatterReductionKind::Add)
// combiner. The integer index input (input 1) has no tangent space, so in a valid pushforward it is the known
// input while the scattered input (input 0) and the updates (input 2) are the linear ones. Scatter-add
// accumulates into its input (`output = input + scattered(updates)`, so the input Jacobian is the identity), so
// the input cotangent is the output cotangent unchanged; the update cotangent gathers the output cotangent at the
// scattered windows via the dual gather built by mirroring the scatter geometry. The transpose reads the known
// indices from the pullback boundary and stages an ordinary [`GatherOperation`], so linearization retains the
// indices as regular SSA residuals. The indices receive a structural zero, and a zero output cotangent stays a
// structural zero. Unique-index overwrite erases the input cotangent at the written windows; other combiners are rejected.
impl<V: Value<Type = ArrayType>, O> TransposableOperation<V, O> for ScatterOperation
where
    O: Operation<Type = ArrayType>
        + From<AddOperation<ArrayType>>
        + From<ZeroOperation<ArrayType>>
        + From<GatherOperation>
        + From<ScatterOperation>,
    Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<ArrayType>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        check_count!("input", inputs, 3, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        check_count!("accumulator", accumulators, 3, DifferentiationError);
        if self.kind() != ScatterReductionKind::Add
            && !(self.kind() == ScatterReductionKind::Overwrite && self.unique_indices())
        {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "transposition of `{}` with the `{}` combiner requires scatter-add or unique-index overwrite",
                    SCATTER_OPERATION_NAME,
                    self.kind(),
                ),
            }
            .into());
        }
        match &outputs[0] {
            MaybeZero::Zero(_) => Ok(()),
            MaybeZero::Value(cotangent) => {
                if accumulators[0].is_needed() {
                    let contribution = if self.kind() == ScatterReductionKind::Overwrite {
                        // Unique replacement windows erase the input tangent exactly where updates are written.
                        let update_zeros = MaybeZero::Zero(inputs[2].r#type().cotangent()?).materialize(&**context)?;
                        let indices = inputs[1]
                            .as_known()
                            .ok_or_else(|| TypeError::invalid("`scatter` transpose requires known indices"))?
                            .clone();
                        let mut contributions = context.stage_operation(
                            self.clone().with_output_sharding(inputs[0].r#type().cotangent()?.sharding().cloned()),
                            Vec::new(),
                            &[cotangent.clone(), indices, update_zeros],
                        )?;
                        check_count!("output", contributions, 1, ProgramError);
                        contributions.remove(0)
                    } else {
                        cotangent.clone()
                    };
                    let contribution = contribution.unalign_cotangent(&inputs[0].r#type().cotangent()?)?;
                    accumulators[0].accumulate(context, MaybeZero::Value(contribution))?;
                }
                // Only the update input needs a gather; the base input's cotangent is the seed itself.
                if !accumulators[2].is_needed() {
                    return Ok(());
                }
                // The indices are the known input; the dispatch guarantees a `Known` input carries its pullback
                // value, so read the tracer directly.
                let indices =
                    inputs[1].as_known().expect("dispatch guarantees a known input carries its pullback value").clone();
                // Build the dual gather by mirroring the scatter geometry: the slice sizes pair each input window
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
                        let extent = updates_type.dimension(update_axis).value().ok_or_else(|| {
                            ProgramError::from(TypeError::invalid(format!(
                                "`{SCATTER_OPERATION_NAME}` transpose requires a static update shape but update axis \
                                     {update_axis} has a dynamic size",
                            )))
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
                let mut gather_operation = GatherOperation::new(gather_dimensions, slice_sizes)
                    .with_mode(self.mode())
                    .with_indices_are_sorted(self.indices_are_sorted())
                    .with_unique_indices(self.unique_indices())
                    .with_output_sharding(updates_type.cotangent()?.sharding().cloned());
                // Dropped updates have zero derivative, independent of gather's default replacement value.
                if self.mode() == GatherScatterMode::FillOrDrop {
                    gather_operation = gather_operation.with_fill_value(
                        EagerContext::<Array>::new().zero(&ArrayType::scalar(cotangent.r#type().data_type()))?,
                    )?;
                }
                let update_cotangents =
                    context.stage_operation(gather_operation, Vec::new(), &[cotangent.clone(), indices])?;
                check_count!("output", update_cotangents, 1, ProgramError);
                let update_cotangent = update_cotangents
                    .into_iter()
                    .next()
                    .unwrap()
                    .unalign_cotangent(&inputs[2].r#type().cotangent()?)?;
                accumulators[2].accumulate(context, MaybeZero::Value(update_cotangent))
            }
        }
    }
}

// Mixed input extents require missing tangents to be materialized from their primal runtime geometry. Coefficient
// construction remains shared with the homogeneous rule, with ordinary projected operations retaining residuals.
impl<C> MemberDifferentiableOperation<C> for ScatterOperation
where
    C: Context<Type = ArrayIrType> + Zero<C::Value>,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Operation: ResidualZeroProvider<ArrayIrType, Operation = C::Operation> + OperationProjection<ArrayType>,
    <C::Operation as OperationProjection<ArrayType>>::Projected: DifferentiableOperation<ProjectedContext<C, ArrayType>>
        + From<IotaOperation<ArrayType>>
        + From<ScatterOperation>
        + From<GatherOperation>
        + From<ZeroLikeOperation<ArrayType>>
        + From<OneLikeOperation<ArrayType>>
        + From<CompareOperation<ArrayType>>
        + From<SelectOperation<ArrayType>>
        + From<ConvertElementTypeOperation<ArrayType>>
        + From<AddOperation<ArrayType>>
        + From<MulOperation<ArrayType>>
        + From<DivOperation<ArrayType>>
        + From<ReshapeOperation>
        + From<BroadcastOperation>
        + From<ZeroOperation<ArrayType>>,
{
    fn jvp_in_parent<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let destinations = context;
        let context = destinations.primal();
        let [input, _, updates] = inputs else {
            return Err(ProgramError::InvalidInputCount { expected: 3, actual: inputs.len() }.into());
        };
        let operand_type = <&ArrayType>::try_from(input.primal().r#type().as_ref())?.clone();
        let updates_type = <&ArrayType>::try_from(updates.primal().r#type().as_ref())?.clone();
        let is_static = |r#type: &ArrayType| {
            r#type.shape().dimensions().iter().all(|dimension| matches!(dimension, Dimension::Static(_)))
        };
        let operation = <C::Operation as OperationProjection<ArrayType>>::Projected::from(self.clone());
        if is_static(&operand_type) && is_static(&updates_type) {
            return jvp_projected_operation(destinations, &operation, inputs);
        }

        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let mut primal_outputs = context.bind(operation.clone(), Vec::new(), primal_inputs.as_slice())?;
        check_count!("output", primal_outputs, 1, ProgramError);
        let primal = primal_outputs.remove(0);
        let mut output_primal = primal;
        let primal = destinations.primal_to_tangent(output_primal.clone())?;
        let tangent_inputs = destinations.dual_primal_to_tangent(inputs)?;
        let inputs = tangent_inputs.as_slice();
        let input = &inputs[0];
        let updates = &inputs[2];
        let context = destinations.tangent();
        let tangent = if input.tangent().is_zero() && updates.tangent().is_zero() {
            MaybeZero::Zero(primal.r#type().tangent()?)
        } else {
            let projected_context = ProjectedContext::<C, ArrayType>::new(context.clone());
            let input_tangent = materialize_array_tangent(&projected_context, input)?;
            let updates_tangent = materialize_array_tangent(&projected_context, updates)?;
            let primal_context = ProjectedContext::<C, ArrayType>::new(destinations.primal().clone());
            let (linearized_primal, tangent) = self.linearize_values(
                (&primal_context, &projected_context),
                [
                    &<_ as ValueProjection<ArrayType>>::into_projected(primal_inputs[0].clone())?,
                    &<_ as ValueProjection<ArrayType>>::into_projected(primal_inputs[1].clone())?,
                    &<_ as ValueProjection<ArrayType>>::into_projected(primal_inputs[2].clone())?,
                ],
                <_ as ValueProjection<ArrayType>>::into_projected(output_primal.clone())?,
                [(&input_tangent, input.tangent().is_zero()), (&updates_tangent, updates.tangent().is_zero())],
                |value| {
                    let value = destinations
                        .primal_to_tangent(<C::Value as ValueProjection<ArrayType>>::from_projected(value))?;
                    Ok(<_ as ValueProjection<ArrayType>>::into_projected(value)?)
                },
            )?;
            if self.kind() == ScatterReductionKind::Overwrite && !self.unique_indices() {
                output_primal = <C::Value as ValueProjection<ArrayType>>::from_projected(linearized_primal);
            }
            MaybeZero::Value(<C::Value as ValueProjection<ArrayType>>::from_projected(tangent))
        };
        Ok(vec![DifferentiationDual::new(output_primal, tangent)?])
    }
}

/// Value-level scatter capability: the receiver-style entry point for staging or executing [`ScatterOperation`].
///
/// The receiver is the input; `indices` is a separate integer-typed value whose last axis holds each start-index
/// vector; `updates` holds the windows to combine into the input using the operation's [`ScatterReductionKind`].
/// All three values must reside in the same memory space. The output preserves the input shape, element type,
/// layout, and memory placement. Its placement normally follows the input and includes any additional manual-axis
/// variation introduced by indices or updates; an explicit output placement must preserve reduction and manual-axis
/// state and use the same mesh. Input and update reduction states must match; unreduced inputs support additive or
/// overwrite updates with replicated, invariant indices.
///
/// Negative starts are out of bounds and do not wrap from an axis end. Clip mode moves the entire update window
/// inside the input; fill-or-drop mode discards a whole invalid window. Empty inputs remain empty. Repeated updates
/// all participate in reductions, but overwrite conflicts and floating-point reduction order are backend dependent.
/// Use [`Self::scatter_axis`] for complete slices along one axis without constructing dimension numbers.
///
/// # Example
///
/// ```rust
/// use ryft_core::{Array, Scatter, ScatterDimensionNumbers, ScatterOperation, ScatterReductionKind};
///
/// let input = Array::matrix(3, 2, vec![0.0; 6]).unwrap();
/// let indices = Array::matrix(2, 1, vec![0_i32, 2]).unwrap();
/// let updates = Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// // Update axis 1 holds each full row; input axis 0 is supplied by the row index.
/// let dimensions = ScatterDimensionNumbers::new(vec![1], vec![0], vec![0]);
/// let operation = ScatterOperation::new(dimensions, ScatterReductionKind::Add);
/// let output = input.scatter(&indices, &updates, &operation).unwrap();
/// assert_eq!(output, Array::matrix(3, 2, vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0]).unwrap());
/// ```
pub trait Scatter: Sized {
    /// Scatters `updates` into `self` (the input) at the positions named by `indices`, according to `operation`.
    ///
    /// # Parameters
    ///
    ///   - `indices`: integer start-index vectors. The trailing axis contains the coordinates selected by the
    ///     operation's dimension numbers; preceding axes enumerate update windows.
    ///   - `updates`: values to combine with the input. Window and indexing axes are identified by the operation's
    ///     dimension numbers, and the element type must equal the input element type.
    ///   - `operation`: the axis mapping, reduction kind, bounds mode, optional output sharding, and index promises
    ///     governing the update. Sortedness and uniqueness flags are caller promises rather than runtime checks.
    fn scatter(&self, indices: &Self, updates: &Self, operation: &ScatterOperation) -> Result<Self, ProgramError>;

    /// Scatters complete slices along one axis using raw integer indices. The index shape replaces the selected
    /// input axis in the required updates shape; all other input axes retain their full size and order. Negative
    /// indices are out of bounds and are handled directly by `mode`, without wrapping them from the axis end.
    ///
    /// The selected input axis may have a dynamic extent. The index shape must support the homogeneous [`Reshape`]
    /// used to append an index-vector axis; remaining input and update dimensions must satisfy [`ScatterOperation`]'s
    /// window constraints. Use an explicit operation for dynamic queries or partial windows.
    ///
    /// # Parameters
    ///
    ///   - `indices`: Integer indices of any rank, including scalar indices selecting one complete slice.
    ///   - `updates`: Values with the same element type as the input and the shape obtained by replacing `axis`
    ///     with the index array's shape.
    ///   - `axis`: Input axis to update; negative axes count backward from the input rank.
    ///   - `kind`: How each update combines with the value already stored at its target.
    ///   - `mode`: How out-of-bounds indices are clipped or dropped. Promise mode requires valid indices.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ryft_core::{Array, GatherScatterMode, Scatter, ScatterReductionKind};
    ///
    /// let input = Array::vector(vec![10_i32, 20, 30]).unwrap();
    /// let indices = Array::vector(vec![2_i32, 0]).unwrap();
    /// let updates = Array::vector(vec![1_i32, 2]).unwrap();
    /// let output = input.scatter_axis(&indices, &updates, 0, ScatterReductionKind::Add, GatherScatterMode::Clip).unwrap();
    /// assert_eq!(output, Array::vector(vec![12_i32, 20, 31]).unwrap());
    /// ```
    fn scatter_axis<A: Into<Axis>>(
        &self,
        indices: &Self,
        updates: &Self,
        axis: A,
        kind: ScatterReductionKind,
        mode: GatherScatterMode,
    ) -> Result<Self, ProgramError>
    where
        Self: Typed<Type = ArrayType> + Reshape,
    {
        let input_type = self.r#type();
        let axis = axis.into().normalize(input_type.rank()).map_err(|error| TypeError::invalid(error.to_string()))?;
        let indices_type = indices.r#type();
        let mut expected_dimensions = input_type.shape().dimensions()[..axis].to_vec();
        expected_dimensions.extend_from_slice(indices_type.shape().dimensions());
        expected_dimensions.extend_from_slice(&input_type.shape().dimensions()[axis + 1..]);
        let expected_shape = Shape::new(expected_dimensions);
        if updates.r#type().shape() != &expected_shape {
            return Err(TypeError::invalid(format!(
                "`scatter_axis` updates shape must be `{expected_shape}` but got `{}`",
                updates.r#type().shape()
            ))
            .into());
        }
        let mut indices_dimensions = indices_type.shape().dimensions().to_vec();
        indices_dimensions.push(Dimension::Static(1));
        let indices = indices.reshape(crate::arrays::Shape::new(indices_dimensions))?;
        let window_dimensions = (0..input_type.rank())
            .filter(|input_axis| *input_axis != axis)
            .map(|input_axis| if input_axis < axis { input_axis } else { input_axis + indices_type.rank() - 1 })
            .collect();
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(window_dimensions, vec![axis], vec![axis]), kind)
                .with_mode(mode);
        self.scatter(&indices, updates, &operation)
    }
}

impl Scatter for ArrayType {
    // Type-level scatter: validates the dimension numbers, the updates shape, and the data types, and computes the
    // output type (which equals the input type) and placement.
    fn scatter(&self, indices: &Self, updates: &Self, operation: &ScatterOperation) -> Result<Self, ProgramError> {
        let input = self;
        let dimensions = operation.dimensions();
        let operand_rank = input.rank();
        let indices_rank = indices.rank();
        let updates_rank = updates.rank();

        if indices_rank == 0 {
            return Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` indices must have rank at least 1 (the trailing index vector)"
            ))
            .into());
        }
        if !indices.data_type().is_integer() {
            return Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` indices must be integer-typed but have type `{indices}`"
            ))
            .into());
        }
        if input.memory() != indices.memory() || input.memory() != updates.memory() {
            return Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` input, indices, and updates must share one memory space but reside \
                     in {}, {}, and {}",
                input.memory(),
                indices.memory(),
                updates.memory(),
            ))
            .into());
        }
        if updates.data_type() != input.data_type() {
            return Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` updates data type `{}` does not match input data type `{}`",
                updates.data_type(),
                input.data_type(),
            ))
            .into());
        }
        let data_type = input.data_type();
        match operation.kind() {
            ScatterReductionKind::Overwrite => {}
            ScatterReductionKind::Add | ScatterReductionKind::Mul
                if !data_type.is_numeric() && data_type != DataType::Zero =>
            {
                return Err(TypeError::invalid(format!(
                    "`{SCATTER_OPERATION_NAME}` kind `{}` requires numeric input and update elements but got \
                     `{data_type}`",
                    operation.kind(),
                ))
                .into());
            }
            ScatterReductionKind::Min | ScatterReductionKind::Max
                if !data_type.is_boolean() && !data_type.is_numeric() && data_type != DataType::Zero =>
            {
                return Err(TypeError::invalid(format!(
                    "`{SCATTER_OPERATION_NAME}` kind `{}` requires Boolean or numeric input and update elements but got \
                     `{data_type}`",
                    operation.kind(),
                ))
                .into());
            }
            _ => {}
        }
        let index_vector_dimension = indices_rank - 1;
        let Dimension::Static(index_vector_extent) = indices.dimension(index_vector_dimension) else {
            return Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` indices index vector dimension must have a static extent"
            ))
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
            return Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` scatter_dimensions_to_operand_dimensions has length {} but the index \
                     vector extent is {index_vector_extent}",
                dimensions.scatter_dimensions_to_operand_dimensions().len(),
            ))
            .into());
        }
        validate_unique_in_range(
            SCATTER_OPERATION_NAME,
            "scatter_dimensions_to_operand_dimensions",
            dimensions.scatter_dimensions_to_operand_dimensions(),
            operand_rank,
        )?;
        if dimensions.scatter_indices_batching_dimensions().len() != dimensions.operand_batching_dimensions().len() {
            return Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` input and scatter-indices batching dimensions must align 1:1, but got \
                     {} and {}",
                dimensions.operand_batching_dimensions().len(),
                dimensions.scatter_indices_batching_dimensions().len(),
            ))
            .into());
        }
        validate_unique_in_range(
            SCATTER_OPERATION_NAME,
            "scatter_indices_batching_dimensions",
            dimensions.scatter_indices_batching_dimensions(),
            indices_rank,
        )?;
        for &dimension in dimensions.scatter_indices_batching_dimensions() {
            if dimension >= indices_rank || dimension == index_vector_dimension {
                return Err(TypeError::invalid(format!(
                    "`{SCATTER_OPERATION_NAME}` scatter_indices_batching_dimensions entry {dimension} is out of \
                         range or names the index vector dimension"
                ))
                .into());
            }
        }

        let inserted: BTreeSet<usize> = dimensions.inserted_window_dimensions().iter().copied().collect();
        let operand_batching: BTreeSet<usize> = dimensions.operand_batching_dimensions().iter().copied().collect();
        if inserted.intersection(&operand_batching).next().is_some() {
            return Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` inserted_window_dimensions and operand_batching_dimensions must be \
                     disjoint"
            ))
            .into());
        }

        if dimensions
            .scatter_dimensions_to_operand_dimensions()
            .iter()
            .any(|axis| operand_batching.contains(axis))
        {
            return Err(
                TypeError::invalid("`scatter` indexed input axes and batching input axes must be disjoint").into()
            );
        }

        // Rank decomposition: the input axes split into window, inserted, and batching axes; the updates axes split
        // into window axes and the scatter/batch axes carried from the indices (every indices axis but the index
        // vector).
        if operand_rank != dimensions.update_window_dimensions().len() + inserted.len() + operand_batching.len() {
            return Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` input rank {operand_rank} must equal update_window + inserted_window + \
                     operand_batching dimension counts"
            ))
            .into());
        }
        if updates_rank != (indices_rank - 1) + dimensions.update_window_dimensions().len() {
            return Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` updates rank {updates_rank} must equal (indices rank - 1) + the update \
                     window dimension count"
            ))
            .into());
        }

        // Window-size checks: the input window axes (input axes that are neither inserted nor batching, in order)
        // pair 1:1 with the sorted update window axes; each update window extent must fit within the input window
        // extent.
        let operand_window_axes: Vec<usize> = (0..operand_rank)
            .filter(|axis| !inserted.contains(axis) && !operand_batching.contains(axis))
            .collect();
        for (&operand_axis, &update_axis) in operand_window_axes.iter().zip(dimensions.update_window_dimensions()) {
            if let (Dimension::Static(update_extent), Dimension::Static(operand_extent)) =
                (updates.dimension(update_axis), input.dimension(operand_axis))
                && update_extent > operand_extent
            {
                return Err(TypeError::invalid(format!(
                    "`{SCATTER_OPERATION_NAME}` update window axis {update_axis} extent {update_extent} exceeds \
                         the input window axis {operand_axis} extent {operand_extent}"
                ))
                .into());
            }
        }

        // The updates' scatter/batch axes (every updates axis but the window axes) must match the indices' batch axes
        // (every indices axis but the index vector), in order.
        let update_window: BTreeSet<usize> = dimensions.update_window_dimensions().iter().copied().collect();
        let update_scatter_axes: Vec<usize> = (0..updates_rank).filter(|axis| !update_window.contains(axis)).collect();
        let indices_batch_axes: Vec<usize> = (0..indices_rank).filter(|axis| *axis != index_vector_dimension).collect();
        for (&update_axis, &indices_axis) in update_scatter_axes.iter().zip(&indices_batch_axes) {
            if !dimensions_have_equal_extents(&updates.dimension(update_axis), &indices.dimension(indices_axis)) {
                return Err(TypeError::invalid(format!(
                    "`{SCATTER_OPERATION_NAME}` updates scatter axis {update_axis} must match indices batch axis \
                         {indices_axis} in extent"
                ))
                .into());
            }
        }

        // Batching extents must match between input and indices.
        for (&operand_axis, &indices_axis) in dimensions
            .operand_batching_dimensions()
            .iter()
            .zip(dimensions.scatter_indices_batching_dimensions())
        {
            if !dimensions_have_equal_extents(&input.dimension(operand_axis), &indices.dimension(indices_axis)) {
                return Err(TypeError::invalid(format!(
                    "`{SCATTER_OPERATION_NAME}` batching dimensions must have equal extents, but input axis \
                         {operand_axis} and indices axis {indices_axis} differ"
                ))
                .into());
            }
        }

        // Placement validation applies before selecting inferred or explicitly requested output placement.
        let common_mesh = [input.sharding(), indices.sharding(), updates.sharding()]
            .into_iter()
            .flatten()
            .next()
            .map(|sharding| sharding.mesh().clone());
        if let Some(mesh) = &common_mesh {
            check_same_mesh(mesh, input.sharding())?;
            check_same_mesh(mesh, indices.sharding())?;
            check_same_mesh(mesh, updates.sharding())?;
        }
        let unreduced_axes = input.sharding().map(Sharding::unreduced_axes).cloned().unwrap_or_default();
        let reduced_axes = input.sharding().map(Sharding::reduced_axes).cloned().unwrap_or_default();
        let updates_unreduced = updates.sharding().map(Sharding::unreduced_axes).cloned().unwrap_or_default();
        let updates_reduced = updates.sharding().map(Sharding::reduced_axes).cloned().unwrap_or_default();
        if unreduced_axes != updates_unreduced || reduced_axes != updates_reduced {
            return Err(TypeError::invalid("`scatter` input and updates must have matching reduction state").into());
        }
        if !unreduced_axes.is_empty()
            && !matches!(operation.kind(), ScatterReductionKind::Add | ScatterReductionKind::Overwrite)
        {
            return Err(TypeError::invalid("`scatter` nonlinear reductions do not support unreduced inputs").into());
        }
        if indices
            .sharding()
            .is_some_and(|sharding| !sharding.unreduced_axes().is_empty() || !sharding.reduced_axes().is_empty())
        {
            return Err(TypeError::invalid("`scatter` indices cannot carry reduced or unreduced mesh axes").into());
        }
        if !unreduced_axes.is_empty()
            && indices.sharding().is_some_and(|sharding| {
                !sharding.varying_manual_axes().is_empty()
                    || sharding
                        .dimensions()
                        .iter()
                        .any(|dimension| *dimension != crate::arrays::ShardingDimension::Replicated)
            })
        {
            return Err(TypeError::invalid("`scatter` unreduced inputs require replicated, invariant indices").into());
        }
        let mut varying_manual_axes = input.sharding().map(Sharding::varying_manual_axes).cloned().unwrap_or_default();
        for sharding in [indices.sharding(), updates.sharding()].into_iter().flatten() {
            varying_manual_axes.extend(sharding.varying_manual_axes().iter().cloned());
        }
        let sharding = if let Some(requested) = operation.output_sharding() {
            if common_mesh.as_ref().is_some_and(|mesh| requested.mesh() != mesh) {
                return Err(TypeError::invalid("`scatter` requested output sharding uses a different mesh").into());
            }
            if requested.unreduced_axes() != &unreduced_axes
                || requested.reduced_axes() != &reduced_axes
                || requested.varying_manual_axes() != &varying_manual_axes
            {
                return Err(TypeError::invalid(
                    "`scatter` requested output sharding changes reduction or manual-axis state",
                )
                .into());
            }
            if requested.rank() != input.rank() {
                return Err(TypeError::invalid(format!(
                    "`{SCATTER_OPERATION_NAME}` output sharding rank ({}) does not match the input rank ({})",
                    requested.rank(),
                    input.rank(),
                ))
                .into());
            }
            if requested.references_auto_axis() {
                return Err(TypeError::invalid(format!(
                    "`{SCATTER_OPERATION_NAME}` output sharding cannot reference auto mesh axes"
                ))
                .into());
            }
            Some(requested.clone())
        } else if let Some(operand_sharding) = input.sharding() {
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
                let window_extent = if inserted.contains(&axis) {
                    Dimension::Static(1)
                } else {
                    let window = operand_window_axes.iter().position(|window_axis| *window_axis == axis).unwrap();
                    updates.dimension(dimensions.update_window_dimensions()[window])
                };
                if input.dimension(axis) != window_extent
                    && input.dimension(axis) != Dimension::Static(0)
                    && dimension_has_explicit_axis(&mesh, &operand_sharding.dimensions()[axis])
                {
                    return Err(TypeError::invalid(format!(
                        "`{SCATTER_OPERATION_NAME}` input axis {axis} is targeted by the start indices and must be \
                             replicated over explicit mesh axes; request an explicit output sharding to resolve \
                             placement"
                    ))
                    .into());
                }
            }
            if let Some(indices_sharding) = indices.sharding()
                && dimension_has_explicit_axis(&mesh, &indices_sharding.dimensions()[index_vector_dimension])
            {
                return Err(TypeError::invalid(format!(
                    "`{SCATTER_OPERATION_NAME}` indices index vector dimension must be replicated over explicit \
                         mesh axes"
                ))
                .into());
            }
            Some(
                operand_sharding
                    .clone()
                    .with_varying_manual_axes(varying_manual_axes)
                    .map_err(|error| TypeError::invalid(error.to_string()))?,
            )
        } else if let Some(mesh) = common_mesh {
            Some(
                Sharding::new(mesh, vec![crate::arrays::ShardingDimension::Replicated; input.rank()])
                    .and_then(|sharding| sharding.with_varying_manual_axes(varying_manual_axes))
                    .map_err(|error| TypeError::invalid(error.to_string()))?,
            )
        } else {
            None
        };
        input.clone().with_sharding(sharding).map_err(|error| TypeError::invalid(error.to_string()).into())
    }
}

impl Array {
    /// Decodes the logical integer element at `index` without narrowing unsigned values or overflowing when
    /// adding a window offset. The type-level validation
    /// performed by every caller rules out non-integer element types and invalid indices.
    fn index_value(&self, addressing: &ArrayAddressing, index: &[usize]) -> i128 {
        let bytes = &self.storage_bytes()[addressing.byte_range_unchecked(index)];
        match self.r#type().data_type() {
            DataType::I1 => i128::from(i1::decode(bytes).value()),
            DataType::I2 => i128::from(i2::decode(bytes).value()),
            DataType::I4 => i128::from(i4::decode(bytes).value()),
            DataType::I8 => i128::from(i8::decode(bytes)),
            DataType::I16 => i128::from(i16::decode(bytes)),
            DataType::I32 => i128::from(i32::decode(bytes)),
            DataType::I64 => i128::from(i64::decode(bytes)),
            DataType::U1 => i128::from(u1::decode(bytes).value()),
            DataType::U2 => i128::from(u2::decode(bytes).value()),
            DataType::U4 => i128::from(u4::decode(bytes).value()),
            DataType::U8 => i128::from(u8::decode(bytes)),
            DataType::U16 => i128::from(u16::decode(bytes)),
            DataType::U32 => i128::from(u32::decode(bytes)),
            DataType::U64 => i128::from(u64::decode(bytes)),
            data_type => unreachable!("cannot use an array of element data type `{data_type}` as indices"),
        }
    }

    /// Applies one already-validated scatter using a byte-slice combiner, keeping index traversal independent of the
    /// selected element arithmetic. The combiner receives one mutable input encoding and one update encoding.
    fn scatter_with_combiner(
        &self,
        indices: &Self,
        updates: &Self,
        output_type: ArrayType,
        operation: &ScatterOperation,
        combine: impl Fn(&mut [u8], &[u8]) -> Result<(), ProgramError>,
    ) -> Result<Self, ProgramError> {
        let dimensions = operation.dimensions();
        let operand_shape = self.r#type().static_shape().unwrap();
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        // No update can address an element of an empty input, even in clipping mode.
        if output_addressing.element_count() == 0 {
            return Ok(Self::new_unchecked(output_type, self.shared_storage().clone()));
        }
        let indices_shape = indices.r#type().static_shape().unwrap();
        let indices_addressing = ArrayAddressing::new(indices.r#type().into_owned())?;
        let updates_shape = updates.r#type().static_shape().unwrap();
        let updates_addressing = ArrayAddressing::new(updates.r#type().into_owned())?;
        let operand_rank = operand_shape.rank();
        let indices_rank = indices_shape.rank();
        let updates_rank = updates_shape.rank();
        let index_vector_dimension = indices_rank - 1;
        let index_vector_extent = indices_shape[index_vector_dimension];

        let inserted: BTreeSet<usize> = dimensions.inserted_window_dimensions().iter().copied().collect();
        let batching: BTreeSet<usize> = dimensions.operand_batching_dimensions().iter().copied().collect();
        let operand_window_axes: Vec<usize> =
            (0..operand_rank).filter(|axis| !inserted.contains(axis) && !batching.contains(axis)).collect();
        let update_window: BTreeSet<usize> = dimensions.update_window_dimensions().iter().copied().collect();
        let update_scatter_axes: Vec<usize> = (0..updates_rank).filter(|axis| !update_window.contains(axis)).collect();
        let indices_batch_axes: Vec<usize> = (0..indices_rank).filter(|axis| *axis != index_vector_dimension).collect();
        // Window size per input axis (the update extent on window axes, 1 elsewhere), used to clamp the start so the
        // whole window stays in bounds.
        let mut operand_window_size = vec![1usize; operand_rank];
        for (window, &operand_axis) in operand_window_axes.iter().enumerate() {
            operand_window_size[operand_axis] = updates_shape[dimensions.update_window_dimensions()[window]];
        }

        let mut output = Self::new_unchecked(output_type, self.shared_storage().clone());
        let output_bytes = output.storage_bytes_mut();
        let mut update_index = vec![0usize; updates_rank];
        let mut indices_index = vec![0usize; indices_rank];
        let mut starts = vec![0i128; index_vector_extent];
        let mut operand_index = vec![0i128; operand_rank];
        let mut operand_storage_index = vec![0usize; operand_rank];
        for written in 0..updates_addressing.element_count() {
            indices_index.fill(0);
            for (position, &update_axis) in update_scatter_axes.iter().enumerate() {
                indices_index[indices_batch_axes[position]] = update_index[update_axis];
            }
            for (component, start) in starts.iter_mut().enumerate() {
                indices_index[index_vector_dimension] = component;
                *start = indices.index_value(&indices_addressing, &indices_index);
            }
            operand_index.fill(0);
            for (window, &operand_axis) in operand_window_axes.iter().enumerate() {
                operand_index[operand_axis] = update_index[dimensions.update_window_dimensions()[window]] as i128;
            }
            for (batch, &operand_axis) in dimensions.operand_batching_dimensions().iter().enumerate() {
                operand_index[operand_axis] =
                    indices_index[dimensions.scatter_indices_batching_dimensions()[batch]] as i128;
            }
            let mut dropped = false;
            for (component, &operand_axis) in dimensions.scatter_dimensions_to_operand_dimensions().iter().enumerate() {
                let raw = starts[component];
                let maximum = (operand_shape[operand_axis] - operand_window_size[operand_axis]) as i128;
                match operation.mode() {
                    GatherScatterMode::FillOrDrop => {
                        if raw < 0 || raw > maximum {
                            dropped = true;
                        }
                        operand_index[operand_axis] += raw;
                    }
                    GatherScatterMode::PromiseInBounds | GatherScatterMode::Clip => {
                        operand_index[operand_axis] += raw.clamp(0, maximum)
                    }
                }
            }
            if !dropped {
                for axis in 0..operand_rank {
                    operand_storage_index[axis] = operand_index[axis] as usize;
                }
                combine(
                    &mut output_bytes[output_addressing.byte_range_unchecked(&operand_storage_index)],
                    &updates.storage_bytes()[updates_addressing.byte_range_for_flat_index(written)],
                )?;
            }
            updates_addressing.advance_index(&mut update_index);
        }
        Ok(output)
    }
}

impl Scatter for Array {
    fn scatter(&self, indices: &Self, updates: &Self, operation: &ScatterOperation) -> Result<Self, ProgramError> {
        let output_type = self.r#type().scatter(indices.r#type().as_ref(), updates.r#type().as_ref(), operation)?;
        let data_type = output_type.data_type();
        if operation.kind() == ScatterReductionKind::Overwrite || data_type == DataType::Zero {
            return self.scatter_with_combiner(indices, updates, output_type, operation, |current, update| {
                current.copy_from_slice(update);
                Ok(())
            });
        }
        match operation.kind() {
            ScatterReductionKind::Add | ScatterReductionKind::Mul => {
                dispatch_on_array_element_type!(@numeric data_type, |Element| {
                    self.scatter_with_combiner(indices, updates, output_type, operation, |current, update| {
                        let current_value = Element::decode(current);
                        let update_value = Element::decode(update);
                        let result = if operation.kind() == ScatterReductionKind::Add {
                            <Element as NumericArrayElement>::add(current_value, update_value)?
                        } else {
                            <Element as NumericArrayElement>::mul(current_value, update_value)?
                        };
                        result.encode(current);
                        Ok(())
                    })
                })
            }
            ScatterReductionKind::Min | ScatterReductionKind::Max => {
                dispatch_on_array_element_type!(data_type, |Element| {
                    self.scatter_with_combiner(indices, updates, output_type, operation, |current, update| {
                        let current_value = Element::decode(current);
                        let update_value = Element::decode(update);
                        let result = if operation.kind() == ScatterReductionKind::Min {
                            ArrayElement::min(&current_value, &update_value)
                        } else {
                            ArrayElement::max(&current_value, &update_value)
                        };
                        result.encode(current);
                        Ok(())
                    })
                })
            }
            ScatterReductionKind::Overwrite => unreachable!("overwrite scatter returns before typed dispatch"),
        }
    }
}

impl<A: Scatter + Value<Type = ArrayType>> Scatter for ArrayIrValue<A> {
    fn scatter(&self, indices: &Self, updates: &Self, operation: &ScatterOperation) -> Result<Self, ProgramError> {
        let input = <Self as ValueProjection<ArrayType>>::projected(self)?;
        let indices = <Self as ValueProjection<ArrayType>>::projected(indices)?;
        let updates = <Self as ValueProjection<ArrayType>>::projected(updates)?;
        Ok(Self::Array(input.scatter(indices, updates, operation)?))
    }
}

// Any context-carrying value scatters by binding a [`ScatterOperation`] through its own context. The
// `From<ScatterOperation>` bound makes this disjoint from the eager value types (whose context operation is
// `ConstantOperation`), so it covers the transform tracers without conflicting with the concrete implementations.
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

/// Errors when `other` is sharded over a different mesh than `mesh`.
fn check_same_mesh(mesh: &LogicalMesh, other: Option<&Sharding>) -> Result<(), TypeError> {
    if let Some(other) = other
        && other.mesh() != mesh
    {
        return Err(TypeError::invalid(format!(
            "`{SCATTER_OPERATION_NAME}` input, indices, and updates shardings must use one mesh"
        )));
    }
    Ok(())
}

/// Scatters complete slices using a first-class query shape.
///
/// The query shape replaces the selected input axis in the updates shape, just as in [`Scatter::scatter_axis`].
/// Unlike that homogeneous convenience, this capability carries the query extents through an explicit
/// [`DynamicReshape`] before projecting into the existing [`ScatterOperation`]. Both the query shape and the input
/// shape can therefore retain symbolic dimensions. This introduces no separate scatter operation or bounds policy.
///
/// # Examples
///
/// ```rust
/// # use ryft_core::{Array, ArrayIrValue, DynamicScatter, GatherScatterMode, ScatterReductionKind};
/// let input = ArrayIrValue::Array(Array::vector(vec![10_i32, 20, 30]).unwrap());
/// let indices = ArrayIrValue::Array(Array::vector(vec![1_i32, 1]).unwrap());
/// let updates = ArrayIrValue::Array(Array::vector(vec![2_i32, 3]).unwrap());
/// let output = input.dynamic_scatter_axis(
///     &indices, &updates, 0, ScatterReductionKind::Add, GatherScatterMode::Clip,
/// ).unwrap();
/// assert_eq!(output, ArrayIrValue::Array(Array::vector(vec![10_i32, 25, 30]).unwrap()));
/// ```
pub trait DynamicScatter: Value<Type = ArrayIrType> + Sized {
    /// Updates slices along `axis` using raw integer indices. Negative indices are out of bounds rather than
    /// counting backward from the end; `mode` specifies whether to clip, drop, or assume valid indices.
    ///
    /// # Parameters
    ///
    ///   - `indices`: Integer query array of any rank. A scalar selects one complete slice.
    ///   - `updates`: Array with the input element type and the shape obtained by replacing `axis` with the query
    ///     shape. Shared symbolic extents must have the same identities, rather than merely the same bounds.
    ///   - `axis`: Input axis to update; negative axes count from the end of the input rank.
    ///   - `kind`: Reduction combining each update with the existing input value, including overlapping updates.
    ///   - `mode`: Out-of-bounds handling; see [`GatherScatterMode`].
    fn dynamic_scatter_axis<A: Into<Axis>>(
        &self,
        indices: &Self,
        updates: &Self,
        axis: A,
        kind: ScatterReductionKind,
        mode: GatherScatterMode,
    ) -> Result<Self, ProgramError>;
}

impl<V> DynamicScatter for V
where
    V: Value<Type = ArrayIrType> + DimensionSize + DynamicReshape + ValueProjection<ArrayType, Projected: Scatter>,
    V::DispatchDomain: Context<Type = ArrayIrType> + DimensionConstant,
{
    fn dynamic_scatter_axis<A: Into<Axis>>(
        &self,
        indices: &Self,
        updates: &Self,
        axis: A,
        kind: ScatterReductionKind,
        mode: GatherScatterMode,
    ) -> Result<Self, ProgramError> {
        let input_type = self.r#type();
        let input_type = <&ArrayType>::try_from(input_type.as_ref())?;
        let indices_type = indices.r#type();
        let indices_type = <&ArrayType>::try_from(indices_type.as_ref())?;
        let updates_type = updates.r#type();
        let updates_type = <&ArrayType>::try_from(updates_type.as_ref())?;
        let axis = axis.into().normalize(input_type.rank()).map_err(|error| TypeError::invalid(error.to_string()))?;
        let mut expected_dimensions = input_type.shape().dimensions()[..axis].to_vec();
        expected_dimensions.extend_from_slice(indices_type.shape().dimensions());
        expected_dimensions.extend_from_slice(&input_type.shape().dimensions()[axis + 1..]);
        let expected_shape = crate::arrays::Shape::new(expected_dimensions);
        if updates_type.shape() != &expected_shape {
            return Err(TypeError::invalid(format!(
                "`dynamic_scatter_axis` updates shape must be `{expected_shape}` but got `{}`",
                updates_type.shape(),
            ))
            .into());
        }
        // Only the index-vector axis is new. Reading the other extents from the query supplies the dimension
        // definitions needed to specialize a retained `[queries] -> [queries, 1]` reshape.
        let indices = indices.dynamic_expand_dims(-1)?;
        let window_dimensions = (0..input_type.rank())
            .filter(|input_axis| *input_axis != axis)
            .map(|input_axis| if input_axis < axis { input_axis } else { input_axis + indices_type.rank() - 1 })
            .collect();
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(window_dimensions, vec![axis], vec![axis]), kind)
                .with_mode(mode);
        Ok(V::from_projected(self.clone().into_projected()?.scatter(
            &indices.into_projected()?,
            &updates.clone().into_projected()?,
            &operation,
        )?))
    }
}

#[cfg(test)]
mod tests {
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayOperation, DataType, Dimension, DimensionBounds, DimensionType, DimensionValue,
        DimensionVariable, Layout, LogicalMesh, Memory, MeshAxis, MeshAxisType, Shape, Sharding, ShardingDimension,
        StridedLayout,
    };
    use crate::batching::batch;
    use crate::contexts::Context;
    use crate::differentiation::differentiate_at;
    use crate::macros::{
        check_operation_batching, check_operation_partial_evaluation, check_operation_transposition,
        check_operation_type_inference,
    };
    use crate::operations::constants::one::OneOperation;
    use crate::operations::manipulation::reshaping::DynamicReshapeOperation;
    use crate::operations::math::reduce::{Reduce, ReductionKind};
    use crate::parameters::Placeholder;
    use crate::programs::{EmptyRegionDriver, ProgramBuilder};
    use crate::tracing::Trace;

    use super::*;

    /// Constructs an integer index type with the requested static dimensions.
    fn indices_type(dimensions: Vec<usize>) -> ArrayType {
        ArrayType::new(DataType::I32, Shape::new(dimensions.into_iter().map(Dimension::Static).collect()))
    }

    /// Constructs a floating-point type with the requested static dimensions.
    fn float_type(dimensions: Vec<usize>) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(dimensions.into_iter().map(Dimension::Static).collect()))
    }

    /// Lifts a constant integer index array into the trace or differentiation context that `exemplar` belongs to.
    fn index_array<V>(exemplar: &V, shape: Vec<usize>, values: Vec<i32>) -> V
    where
        V: crate::programs::Value<Type = ArrayType>,
        V::DispatchDomain: crate::contexts::Context<Constant = Array>,
    {
        let r#type = ArrayType::new(DataType::I32, Shape::new(shape.into_iter().map(Dimension::Static).collect()));
        exemplar.dispatch_domain().lift(Array::from_elements::<i32>(r#type, &values).unwrap()).unwrap()
    }

    #[test]
    fn test_scatter() {
        // Scatter-add row updates into a [3, 2] input indexed by a [2, 1] index array: update window axis 1 carries
        // the row, input axis 0 is inserted (start-index driven).
        let dimensions = ScatterDimensionNumbers::new(vec![1], vec![0], vec![0]);
        let operation = ScatterOperation::new(dimensions, ScatterReductionKind::Add);
        assert_eq!(operation.name(), SCATTER_OPERATION_NAME);
        assert_eq!(operation.kind(), ScatterReductionKind::Add);

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
    }

    #[test]
    fn test_scatter_type_inference() {
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![1], vec![0], vec![0]), ScatterReductionKind::Add);
        let input = float_type(vec![3, 2]);
        let indices = indices_type(vec![2, 1]);
        let updates = float_type(vec![2, 2]);
        let boolean_operand = ArrayType::new(DataType::Boolean, input.shape().clone());
        let boolean_updates = ArrayType::new(DataType::Boolean, updates.shape().clone());
        let host_operand = input.clone().with_memory(Memory::Host { pinned: true });
        let host_indices = indices.clone().with_memory(Memory::Host { pinned: true });
        let host_updates = updates.clone().with_memory(Memory::Host { pinned: true });
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [
                {
                    input_types = [input.clone(), indices.clone(), updates.clone()],
                    output_types = [input.clone()],
                },
                {
                    input_types = [input.clone(), indices.clone()],
                    error = "expected 3 inputs but got 2",
                },
                {
                    input_types = [input.clone(), indices.clone(), indices_type(vec![2, 2])],
                    error = "`scatter` updates data type `i32` does not match input data type `f32`",
                },
                {
                    input_types = [input.clone(), float_type(vec![2, 1]), updates.clone()],
                    error = "`scatter` indices must be integer-typed but have type `f32[2, 1]`",
                },
                {
                    input_types = [boolean_operand, indices.clone(), boolean_updates],
                    error = "`scatter` kind `add` requires numeric input and update elements but got `bool`",
                },
                {
                    input_types = [host_operand.clone(), host_indices, host_updates],
                    output_types = [host_operand.clone()],
                },
                {
                    input_types = [host_operand, indices.clone(), updates.clone()],
                    error = "`scatter` input, indices, and updates must share one memory space but reside in \
                             Host[Pinned], Device, and Device",
                },
            ],
        );
        let complex_operand = ArrayType::new(DataType::C64, input.shape().clone());
        let complex_updates = ArrayType::new(DataType::C64, updates.shape().clone());
        assert_eq!(
            complex_operand.scatter(
                &indices,
                &complex_updates,
                &ScatterOperation::new(operation.dimensions().clone(), ScatterReductionKind::Max),
            ),
            Ok(complex_operand),
        );
    }

    #[test]
    fn test_scatter_type_inference_invalid_dimension_maps() {
        let input = ArrayType::new_static(DataType::F32, [2, 2, 4]);
        let indices = ArrayType::new_static(DataType::I32, [2, 1]);
        let updates = ArrayType::new_static(DataType::F32, [2]);
        let operation = ScatterOperation::new(
            ScatterDimensionNumbers::new(vec![], vec![2], vec![2]).with_batching_dimensions(vec![0, 1], vec![0, 0]),
            ScatterReductionKind::Add,
        );
        assert_eq!(
            operation.infer_output_types(&[input, indices, updates], &[]),
            Err(TypeError::invalid("`scatter` `scatter_indices_batching_dimensions` must be unique but got [0, 0]"))
        );
        let operation = ScatterOperation::new(
            ScatterDimensionNumbers::new(vec![], vec![1], vec![0]).with_batching_dimensions(vec![0], vec![0]),
            ScatterReductionKind::Add,
        );
        assert_eq!(
            operation.infer_output_types(
                &[
                    ArrayType::new_static(DataType::F32, [2, 4]),
                    ArrayType::new_static(DataType::I32, [2, 1]),
                    ArrayType::new_static(DataType::F32, [2]),
                ],
                &[]
            ),
            Err(TypeError::invalid("`scatter` indexed input axes and batching input axes must be disjoint"))
        );
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Add);
        assert_eq!(
            operation.infer_output_types(
                &[
                    ArrayType::new_static(DataType::F32, [4]),
                    ArrayType::new_static(DataType::I32, [2, 1]),
                    ArrayType::new_static(DataType::F32, [2]),
                ],
                &[RegionInterface::new(vec![], vec![], crate::programs::EffectClasses::NONE)]
            ),
            Err(TypeError::invalid("expected 0 regions but got 1"))
        );
    }

    #[test]
    fn test_scatter_type_inference_sharding_metadata() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::Replicated]).unwrap())
            .unwrap();
        let indices = ArrayType::new_static(DataType::I32, [1, 1]);
        let updates = ArrayType::new_static(DataType::F32, [1]);
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Add);
        assert_eq!(
            operation
                .clone()
                .with_output_sharding(Sharding::new(other_mesh.clone(), vec![ShardingDimension::Replicated]).unwrap())
                .infer_output_types(&[input.clone(), indices.clone(), updates.clone()], &[]),
            Err(TypeError::invalid("`scatter` requested output sharding uses a different mesh"))
        );
        assert_eq!(
            operation.infer_output_types(
                &[
                    ArrayType::new_static(DataType::F32, [4]),
                    indices
                        .clone()
                        .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::Replicated; 2]).unwrap())
                        .unwrap(),
                    updates
                        .clone()
                        .with_sharding(Sharding::new(other_mesh, vec![ShardingDimension::Replicated]).unwrap())
                        .unwrap(),
                ],
                &[]
            ),
            Err(TypeError::invalid("`scatter` input, indices, and updates shardings must use one mesh"))
        );
        let unreduced = Sharding::new(mesh.clone(), vec![ShardingDimension::Replicated])
            .unwrap()
            .with_unreduced_axes(["x"])
            .unwrap();
        let input = input.with_sharding(unreduced.clone()).unwrap();
        let partial_updates = updates.clone().with_sharding(unreduced.clone()).unwrap();
        assert_eq!(
            operation.infer_output_types(&[input.clone(), indices.clone(), partial_updates.clone()], &[]),
            Ok(vec![input.clone()])
        );
        assert_eq!(
            operation.infer_output_types(&[input.clone(), indices.clone(), updates], &[]),
            Err(TypeError::invalid("`scatter` input and updates must have matching reduction state"))
        );
        assert_eq!(
            ScatterOperation::new(operation.dimensions().clone(), ScatterReductionKind::Mul)
                .infer_output_types(&[input.clone(), indices.clone(), partial_updates.clone()], &[]),
            Err(TypeError::invalid("`scatter` nonlinear reductions do not support unreduced inputs"))
        );
        assert_eq!(
            operation
                .with_output_sharding(Sharding::new(mesh, vec![ShardingDimension::Replicated]).unwrap())
                .infer_output_types(&[input, indices, partial_updates], &[]),
            Err(TypeError::invalid("`scatter` requested output sharding changes reduction or manual-axis state"))
        );
    }
    #[test]
    fn test_scatter_interpretation() {
        // Interpretation applies each supported combiner and accumulates repeated additive updates.
        let scalar_dimensions = || ScatterDimensionNumbers::new(vec![], vec![0], vec![0]);
        let scalar_indices = Array::from_elements::<i32>(indices_type(vec![2, 1]), &[1, 3]).unwrap();
        let run = |kind| {
            Array::vector(vec![1.0, 2.0, 3.0, 4.0])
                .unwrap()
                .scatter(
                    &scalar_indices,
                    &Array::vector(vec![100.0, 200.0]).unwrap(),
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
        let repeated = Array::from_elements::<i32>(indices_type(vec![2, 1]), &[1, 1]).unwrap();
        let result = Array::vector(vec![1.0, 2.0, 3.0, 4.0])
            .unwrap()
            .scatter(
                &repeated,
                &Array::vector(vec![100.0, 200.0]).unwrap(),
                &ScatterOperation::new(scalar_dimensions(), ScatterReductionKind::Add),
            )
            .unwrap();
        assert_eq!(result.to_f64s(), vec![1.0, 302.0, 3.0, 4.0]);
    }

    #[test]
    fn test_scatter_partial_evaluation() {
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![1], vec![0], vec![0]), ScatterReductionKind::Add);
        // Partial evaluation folds fully known scatters and residualizes an unknown data input.
        let operand_value = Array::matrix(3, 2, vec![0.0; 6]).unwrap();
        let indices_value = Array::from_elements::<i32>(indices_type(vec![2, 1]), &[0, 2]).unwrap();
        let updates_value = Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let expected = Array::matrix(3, 2, vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0]).unwrap();
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
    }

    #[test]
    fn test_scatter_batching() {
        // A shorter mapped update cannot silently update a prefix of a larger mapped input.
        assert_eq!(
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Add)
                .batch(
                    &BatchingContext::new(EagerContext::<Array>::new(), 2),
                    &EmptyRegionDriver,
                    &[
                        ArrayBatch::new(Array::matrix(3, 2, vec![0.0; 6]).unwrap(), BatchAxis::new(0)).unwrap(),
                        ArrayBatch::replicated(Array::matrix(1, 1, vec![0_i32]).unwrap()),
                        ArrayBatch::new(Array::matrix(2, 1, vec![1.0; 2]).unwrap(), BatchAxis::new(0)).unwrap(),
                    ],
                )
                .unwrap_err(),
            BatchingError::MisalignedBatchAxes {
                message: "`scatter` mapped input extent 3 does not match batching extent 2".to_string(),
            },
        );

        // Mapped indices pair each query with its own input, including when the input and updates are replicated.
        check_operation_batching!(
            @exact,
            operation = ScatterOperation::new(
                ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Add,
            ),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@replicated, Array::vector(vec![1.0, 2.0, 3.0]).unwrap()),
                    (@mapped(axis = 0), Array::from_elements::<i32>(indices_type(vec![2, 1, 1]), &[0, 2]).unwrap()),
                    (@replicated, Array::vector(vec![10.0]).unwrap()),
                ],
                outputs = [(@mapped(axis = 0), Array::matrix(2, 3, vec![11.0, 2.0, 3.0, 1.0, 2.0, 13.0]).unwrap())],
            }],
        );
        // Empty batching never needs a scalar zero representable in the scattered element format.
        check_operation_batching!(
            @exact,
            operation = ScatterOperation::new(
                ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Overwrite,
            ),
            axis_size = 0,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::new(ArrayType::new_static(DataType::F8E8M0FNU, [0, 3]), vec![]).unwrap()),
                    (@mapped(axis = 0), Array::from_elements::<i32>(indices_type(vec![0, 1, 1]), &[]).unwrap()),
                    (@mapped(axis = 0), Array::new(ArrayType::new_static(DataType::F8E8M0FNU, [0, 1]), vec![]).unwrap()),
                ],
                outputs = [(@mapped(axis = 0), Array::new(ArrayType::new_static(DataType::F8E8M0FNU, [0, 3]), vec![]).unwrap())],
            }],
        );

        let scalar_dimensions = || ScatterDimensionNumbers::new(vec![], vec![0], vec![0]);
        let scalar_indices = Array::from_elements::<i32>(indices_type(vec![2, 1]), &[1, 3]).unwrap();
        // Replicated indices retain the mapped input and update axes as one leading window axis.
        check_operation_batching!(
            @exact,
            operation = ScatterOperation::new(scalar_dimensions(), ScatterReductionKind::Add),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::matrix(2, 4, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap()),
                    (@replicated, scalar_indices),
                    (@mapped(axis = 0), Array::matrix(2, 2, vec![10.0, 20.0, 30.0, 40.0]).unwrap()),
                ],
                outputs = [(@mapped(axis = 0), Array::matrix(
                    2,
                    4,
                    vec![1.0, 12.0, 3.0, 24.0, 5.0, 36.0, 7.0, 48.0],
                ).unwrap())],
            }],
        );
    }

    #[test]
    fn test_scatter_differentiation() {
        // Touched NaN extrema assign no derivative, whereas untouched NaNs remain ordinary identity edges.
        for kind in [ScatterReductionKind::Min, ScatterReductionKind::Max] {
            let (value, (input_gradient, updates_gradient)) = differentiate_at((
                Array::vector(vec![f64::NAN, f64::NAN, 4.0]).unwrap(),
                Array::vector(vec![2.0]).unwrap(),
            ))
            .value_and_gradient(|(input, updates)| {
                let indices = index_array(&input, vec![1, 1], vec![0]);
                input
                    .scatter(
                        &indices,
                        &updates,
                        &ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), kind),
                    )
                    .unwrap()
                    .reduce(&[0], ReductionKind::Sum)
            })
            .unwrap();
            assert!(value.to_f64s()[0].is_nan());
            assert_eq!(input_gradient.to_f64s(), vec![0.0, 1.0, 1.0]);
            assert_eq!(updates_gradient.to_f64s(), vec![0.0]);
        }
        // Repeated multiplication still supports an input-only derivative and omits its inactive `inf * 0` term.
        let (value, gradient) = differentiate_at(Array::vector(vec![f64::INFINITY]).unwrap())
            .value_and_gradient(|input| {
                let indices = index_array(&input, vec![2, 1], vec![0, 0]);
                let updates = input.context().lift(Array::vector(vec![2.0, 3.0]).unwrap()).unwrap();
                input
                    .scatter(
                        &indices,
                        &updates,
                        &ScatterOperation::new(
                            ScatterDimensionNumbers::new(vec![], vec![0], vec![0]),
                            ScatterReductionKind::Mul,
                        ),
                    )
                    .unwrap()
                    .reduce(&[0], ReductionKind::Sum)
            })
            .unwrap();
        assert_eq!(value.to_f64s(), vec![f64::INFINITY]);
        assert_eq!(gradient.to_f64s(), vec![6.0]);

        // Extremal ties split gradients among the retained input and every matching update. Duplicate overwrite
        // uses one consistent winning update for both the returned primal and its derivative.
        for (kind, input_values, update_values, expected_value, expected_input, expected_updates) in [
            (
                ScatterReductionKind::Min,
                vec![2.0, 5.0],
                vec![2.0, 2.0, 4.0],
                6.0,
                vec![1.0 / 3.0, 0.0],
                vec![1.0 / 3.0, 1.0 / 3.0, 1.0],
            ),
            (
                ScatterReductionKind::Max,
                vec![2.0, 5.0],
                vec![2.0, 2.0, 4.0],
                7.0,
                vec![1.0 / 3.0, 1.0],
                vec![1.0 / 3.0, 1.0 / 3.0, 0.0],
            ),
            (
                ScatterReductionKind::Overwrite,
                vec![2.0, 5.0],
                vec![7.0, 8.0, 4.0],
                12.0,
                vec![0.0, 0.0],
                vec![0.0, 1.0, 1.0],
            ),
        ] {
            let (value, (input_gradient, updates_gradient)) =
                differentiate_at((Array::vector(input_values).unwrap(), Array::vector(update_values).unwrap()))
                    .value_and_gradient(|(input, updates)| {
                        let indices = index_array(&input, vec![3, 1], vec![0, 0, 1]);
                        input
                            .scatter(
                                &indices,
                                &updates,
                                &ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), kind),
                            )
                            .unwrap()
                            .reduce(&[0], ReductionKind::Sum)
                    })
                    .unwrap();
            assert_eq!(value.to_f64s(), vec![expected_value]);
            assert_eq!(input_gradient.to_f64s(), expected_input);
            assert_eq!(updates_gradient.to_f64s(), expected_updates);
        }
        // A zero multiplier still has a finite derivative with respect to its update; no division by that value
        // is used to construct the product rule.
        let (value, (input_gradient, updates_gradient)) =
            differentiate_at((Array::vector(vec![2.0, 5.0]).unwrap(), Array::vector(vec![0.0, 3.0]).unwrap()))
                .value_and_gradient(|(input, updates)| {
                    let indices = index_array(&input, vec![2, 1], vec![0, 1]);
                    input
                        .scatter(
                            &indices,
                            &updates,
                            &ScatterOperation::new(
                                ScatterDimensionNumbers::new(vec![], vec![0], vec![0]),
                                ScatterReductionKind::Mul,
                            )
                            .with_unique_indices(true),
                        )
                        .unwrap()
                        .reduce(&[0], ReductionKind::Sum)
                })
                .unwrap();
        assert_eq!(value.to_f64s(), vec![15.0]);
        assert_eq!(input_gradient.to_f64s(), vec![0.0, 3.0]);
        assert_eq!(updates_gradient.to_f64s(), vec![2.0, 5.0]);

        // Forward mode through `f(x) = scatter_add(x, [[1], [3]], [10, 20])` exercises the captured-index scatter-add
        // under batched basis tangents (the lifted dimension-number rule). Scatter-add is the identity in its input, so the
        // Jacobian with respect to `x` is the identity matrix.
        let jacobian = differentiate_at(Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap())
            .jacobian_forward(|x| {
                let indices = index_array(&x, vec![2, 1], vec![1, 3]);
                let updates = x.context().lift(Array::vector(vec![10.0, 20.0]).unwrap())?;
                let operation = ScatterOperation::new(
                    ScatterDimensionNumbers::new(vec![], vec![0], vec![0]),
                    ScatterReductionKind::Add,
                );
                Ok(x.scatter(&indices, &updates, &operation).unwrap())
            })
            .unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.output_type().static_shape().unwrap().as_slice(), &[4]);
        assert_eq!(block.input_type().static_shape().unwrap().as_slice(), &[4]);
        assert_eq!(
            block.value().to_f64s(),
            vec![
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 1.0, //
            ],
        );
    }
    #[test]
    fn test_scatter_transposition() {
        // With unique replacement windows, the input cotangent is erased at the written locations.
        check_operation_transposition!(
            @exact,
            operation = ScatterOperation::new(
                ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Overwrite,
            ).with_unique_indices(true).with_mode(GatherScatterMode::FillOrDrop),
            cases = [{
                inputs = [
                    (@linear(type = ArrayType::new_static(DataType::F64, [4]))),
                    (@known, Array::from_elements::<i32>(indices_type(vec![3, 1]), &[-1, 1, 4]).unwrap()),
                    (@linear(type = ArrayType::new_static(DataType::F64, [3]))),
                ],
                output_cotangents = [Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap()],
                input_cotangents = [Array::vector(vec![1.0, 0.0, 3.0, 4.0]).unwrap(), Array::vector(vec![0.0, 2.0, 0.0]).unwrap()],
            }],
        );

        // The scatter-add pullback leaves the input cotangent unchanged and gathers the update cotangent.
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Add);
        check_operation_transposition!(
            @exact,
            operation = operation,
            cases = [{
                inputs = [
                    (@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![4.into()])))),
                    (@known, Array::from_elements::<i32>(indices_type(vec![2, 1]), &[1, 3]).unwrap()),
                    (@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![2.into()])))),
                ],
                output_cotangents = [Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap()],
                input_cotangents = [
                    Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
                    Array::vector(vec![2.0, 4.0]).unwrap(),
                ],
            }],
        );

        // Dropped updates contribute zero even though the dual gather normally fills missing windows with NaN.
        check_operation_transposition!(
            @exact,
            operation = ScatterOperation::new(
                ScatterDimensionNumbers::new(vec![], vec![0], vec![0]),
                ScatterReductionKind::Add,
            ).with_mode(GatherScatterMode::FillOrDrop),
            cases = [{
                inputs = [
                    (@linear(type = ArrayType::new_static(DataType::F64, [4]))),
                    (@known, Array::from_elements::<i32>(indices_type(vec![3, 1]), &[-1, 1, 4]).unwrap()),
                    (@linear(type = ArrayType::new_static(DataType::F64, [3]))),
                ],
                output_cotangents = [Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap()],
                input_cotangents = [
                    Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
                    Array::vector(vec![0.0, 2.0, 0.0]).unwrap(),
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
            Array::from_elements::<i32>(indices_type(vec![2, 1]).with_memory(Memory::Host { pinned: true }), &[1, 3])
                .unwrap();
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
                output_cotangents = [Array::from_elements::<f64>(operand_type.clone(), &[1.0, 2.0, 3.0, 4.0]).unwrap()],
                input_cotangents = [
                    Array::from_elements::<f64>(operand_type, &[1.0, 2.0, 3.0, 4.0]).unwrap(),
                    Array::from_elements::<f64>(update_type, &[2.0, 4.0]).unwrap(),
                ],
            }],
        );
    }

    #[test]
    fn test_scatter_dimension_numbers() {
        let input = float_type(vec![3, 2]);
        let indices = indices_type(vec![2, 1]);
        let dimensions = || ScatterDimensionNumbers::new(vec![1], vec![0], vec![0]);

        // updates rank must equal (indices rank - 1) + update window count.
        let operation = ScatterOperation::new(dimensions(), ScatterReductionKind::Add);
        assert_eq!(
            operation.infer_output_types(&[input.clone(), indices.clone(), float_type(vec![2])], &[]),
            Err(TypeError::invalid(
                "`scatter` `update_window_dimensions` entry 1 is out of range for bound 1".to_string()
            )),
        );
    }

    #[test]
    fn test_array_type_scatter() {
        let exact = Dimension::Dynamic(DimensionVariable::new("batch", DimensionBounds::new(0, Some(1)).unwrap()));
        let operation = ScatterOperation::new(
            ScatterDimensionNumbers::new(vec![], vec![1], vec![1]).with_batching_dimensions(vec![0], vec![0]),
            ScatterReductionKind::Add,
        );
        for input_batch in [Dimension::Static(0), exact.clone()] {
            for query_batch in [Dimension::Static(0), exact.clone()] {
                for update_batch in [Dimension::Static(0), exact.clone()] {
                    let input =
                        ArrayType::new(DataType::F64, Shape::new(vec![input_batch.clone(), Dimension::Static(4)]));
                    let indices = ArrayType::new(
                        DataType::I32,
                        Shape::new(vec![query_batch.clone(), Dimension::Static(2), Dimension::Static(1)]),
                    );
                    let updates = ArrayType::new(DataType::F64, Shape::new(vec![update_batch, Dimension::Static(2)]));
                    assert_eq!(input.scatter(&indices, &updates, &operation).unwrap(), input);
                }
            }
        }
        // Identically named dimensions with equal non-exact bounds remain independent and must not pass either
        // the source/query pairing or the query/update shape equality checks.
        let first = Dimension::Dynamic(DimensionVariable::new("batch", DimensionBounds::new(0, Some(3)).unwrap()));
        let second = Dimension::Dynamic(DimensionVariable::new("batch", DimensionBounds::new(0, Some(3)).unwrap()));
        let input = ArrayType::new(DataType::F64, Shape::new(vec![first.clone(), Dimension::Static(4)]));
        for (query_batch, update_batch) in [(second.clone(), second.clone()), (first, second)] {
            let indices = ArrayType::new(
                DataType::I32,
                Shape::new(vec![query_batch, Dimension::Static(2), Dimension::Static(1)]),
            );
            let updates = ArrayType::new(DataType::F64, Shape::new(vec![update_batch, Dimension::Static(2)]));
            assert!(matches!(input.scatter(&indices, &updates, &operation), Err(ProgramError::Type(_))));
        }

        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let dimensions = || ScatterDimensionNumbers::new(vec![1], vec![0], vec![0]);

        // Input [4, 2] sharded only on the feature axis (axis 1); the targeted axis 0 is replicated → output keeps
        // the input sharding.
        let sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["y"])])
                .unwrap();
        let input = float_type(vec![4, 2]).with_sharding(sharding.clone()).unwrap();
        let indices = indices_type(vec![2, 1]);
        let updates = float_type(vec![2, 2]);
        let operation = ScatterOperation::new(dimensions(), ScatterReductionKind::Add);
        let output = operation.infer_output_types(&[input, indices.clone(), updates.clone()], &[]).unwrap();
        assert_eq!(output[0].sharding(), Some(&sharding));

        // Sharding the targeted input axis over an explicit mesh axis is ambiguous without an output sharding.
        let input = float_type(vec![4, 2])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        let operation = ScatterOperation::new(dimensions(), ScatterReductionKind::Add);
        assert!(operation.infer_output_types(&[input, indices, updates], &[]).is_err());
    }

    #[test]
    fn test_array_ir_scatter_differentiation() {
        // Explicit output placement can differ from the input. Coefficients and masks follow the value edge where
        // they are consumed, and the final pullback restores each original input's placement.
        for kind in [
            ScatterReductionKind::Overwrite,
            ScatterReductionKind::Mul,
            ScatterReductionKind::Min,
            ScatterReductionKind::Max,
        ] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
            let input_type = ArrayType::new_static(DataType::F64, [4])
                .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
                .unwrap();
            let output_sharding = Sharding::replicated(mesh, 1);
            let updates_type =
                ArrayType::new_static(DataType::F64, [2]).with_sharding(output_sharding.clone()).unwrap();
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let input = builder.add_input(input_type.clone().into());
            let updates = builder.add_input(updates_type.clone().into());
            let indices = builder.add_constant(ArrayIrValue::Array(Array::matrix(2, 1, vec![1_i32, 3]).unwrap()));
            let output = builder
                .add_instruction(
                    ArrayIrOperation::Array(ArrayOperation::Scatter(
                        ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), kind)
                            .with_unique_indices(kind == ScatterReductionKind::Mul)
                            .with_output_sharding(output_sharding),
                    )),
                    vec![],
                    vec![input, indices, updates],
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
            assert_eq!(
                linearization.pullback().unwrap().output_types(),
                vec![
                    ArrayIrType::Array(input_type.cotangent().unwrap()),
                    ArrayIrType::Array(updates_type.cotangent().unwrap()),
                ]
            );
        }

        // Winner IDs preserve the mesh without reduction markers; extremal coefficients preserve reduced data state.
        for (kind, unreduced) in [
            (ScatterReductionKind::Overwrite, false),
            (ScatterReductionKind::Overwrite, true),
            (ScatterReductionKind::Min, false),
            (ScatterReductionKind::Max, false),
        ] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
            let sharding = Sharding::replicated(mesh, 1);
            let sharding = if unreduced {
                sharding.with_unreduced_axes(["x"]).unwrap()
            } else {
                sharding.with_reduced_axes(["x"]).unwrap()
            };
            let input_type = ArrayType::new_static(DataType::F64, [4]).with_sharding(sharding.clone()).unwrap();
            let updates_type = ArrayType::new_static(DataType::F64, [2]).with_sharding(sharding).unwrap();
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let input = builder.add_input(input_type.clone().into());
            let updates = builder.add_input(updates_type.clone().into());
            let indices = builder.add_constant(ArrayIrValue::Array(Array::matrix(2, 1, vec![1_i32, 1]).unwrap()));
            let output = builder
                .add_instruction(
                    ArrayIrOperation::Array(ArrayOperation::Scatter(ScatterOperation::new(
                        ScatterDimensionNumbers::new(vec![], vec![0], vec![0]),
                        kind,
                    ))),
                    vec![],
                    vec![input, indices, updates],
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
            assert_eq!(
                linearization.pullback().unwrap().output_types(),
                vec![
                    ArrayIrType::Array(input_type.cotangent().unwrap()),
                    ArrayIrType::Array(updates_type.cotangent().unwrap()),
                ]
            );
        }

        // The same nonlinear coefficient rules work when the input extent is an ordinary symbolic dimension.
        for (kind, index_values, expected_input, expected_updates) in [
            (ScatterReductionKind::Min, vec![1_i32, 3], vec![1.0, 0.5, 1.0, 0.0], vec![0.5, 1.0]),
            (ScatterReductionKind::Mul, vec![1, 3], vec![1.0, 2.0, 1.0, 1.0], vec![2.0, 4.0]),
            (ScatterReductionKind::Overwrite, vec![1, 1], vec![1.0, 0.0, 1.0, 1.0], vec![0.0, 1.0]),
        ] {
            let extent = DimensionVariable::new("extent", DimensionBounds::new(4, Some(7)).unwrap());
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let input =
                builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)])).into());
            let indices_type = ArrayType::new_static(DataType::I32, [2, 1]);
            let indices = builder.add_input(indices_type.clone().into());
            let updates = builder.add_input(ArrayType::new_static(DataType::F64, [2]).into());
            let output = builder
                .add_instruction(
                    ArrayIrOperation::Array(ArrayOperation::Scatter(
                        ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), kind)
                            .with_unique_indices(kind == ScatterReductionKind::Mul),
                    )),
                    vec![],
                    vec![input, indices, updates],
                    None,
                )
                .unwrap()[0];
            let program = builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![output],
                    vec![Placeholder, Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap();
            let linearization = program.linearize().unwrap();
            let mut primals = linearization
                .primal()
                .interpret(vec![
                    ArrayIrValue::Array(Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap()),
                    ArrayIrValue::Array(Array::from_elements(indices_type, &index_values).unwrap()),
                    ArrayIrValue::Array(Array::vector(vec![2.0, 1.0]).unwrap()),
                ])
                .unwrap();
            let mut seeds = vec![ArrayIrValue::Array(Array::vector(vec![1.0; 4]).unwrap())];
            seeds.extend(primals.split_off(1));
            assert_eq!(
                linearization.pullback().unwrap().interpret(seeds).unwrap(),
                vec![
                    ArrayIrValue::Array(Array::vector(expected_input).unwrap()),
                    ArrayIrValue::Array(Array::vector(expected_updates).unwrap()),
                ]
            );
        }

        let extent = DimensionVariable::new("extent", DimensionBounds::new(4, Some(7)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let indices_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let indices = builder.add_input(indices_type.clone().into());
        let updates = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])).into());
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Scatter(ScatterOperation::new(
                    ScatterDimensionNumbers::new(Vec::new(), vec![0], vec![0]),
                    ScatterReductionKind::Add,
                ))),
                Vec::new(),
                vec![input, indices, updates],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();

        assert_eq!(linearization.residual_count(), 1);
        let indices = ArrayIrValue::Array(Array::from_elements::<i32>(indices_type, &[1, 3]).unwrap());
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap()),
                indices,
                ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0]).unwrap()),
            ])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::vector(vec![1.0_f64, 12.0, 3.0, 24.0]).unwrap()));
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![
            ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap()),
            ArrayIrValue::Array(Array::vector(vec![5.0_f64, 6.0]).unwrap()),
        ];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 7.0, 3.0, 10.0]).unwrap())]),
        );
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0]).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![20.0_f64, 40.0]).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_array_ir_dynamic_scatter_disconnected_operand_tangent_uses_runtime_extent_residuals() {
        // Mixed scatter materializes a structurally zero input tangent through the residual protocol, using the
        // input primal as the runtime source for each symbolic extent omitted by its tangent type.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(4, Some(7)).unwrap());
        let extent_type = DimensionType::new(extent);
        let indices_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)]));
        let updates_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let padded_extent = builder.add_input(extent_type.clone().into());
        let indices = builder.add_input(indices_type.clone().into());
        let updates = builder.add_input(updates_type.into());

        // The reshaped input is a static nullary one constant, so its tangent is a structural zero of a static type
        // that no rule needs to materialize. Mixed reshape then carries that zero tangent into a structural zero of its
        // own output type with a symbolic extent, which is exactly the disconnected dynamic input tangent the
        // scattered input receives. Its primal carries the required runtime extent and the update tangent stays
        // live, so the rule must materialize a concrete input tangent through the residual protocol before staging
        // tangent scatter.
        let ones = builder
            .add_instruction(
                ArrayOperation::One(OneOperation::new(ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![Dimension::Static(4)]),
                ))),
                Vec::new(),
                Vec::new(),
                None,
            )
            .unwrap()[0];
        let input = builder
            .add_instruction(DynamicReshapeOperation::new(), Vec::new(), vec![ones, padded_extent], None)
            .unwrap()[0];
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Scatter(ScatterOperation::new(
                    ScatterDimensionNumbers::new(Vec::new(), vec![0], vec![0]),
                    ScatterReductionKind::Add,
                ))),
                Vec::new(),
                vec![input, indices, updates],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.interpret(vec![
                ArrayIrValue::Dimension(DimensionValue::new(extent_type, 4).unwrap()),
                ArrayIrValue::Array(Array::from_elements::<i32>(indices_type, &[1, 3]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0]).unwrap()),
            ]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 11.0, 1.0, 21.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![0.0_f64, 1.0, 0.0, 2.0]).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_array_scatter() {
        // Scatter-add updates 10 and 20 into elements 3 and 0 of a vector.
        let input = Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let indices = Array::from_elements::<i64>(ArrayType::new_static(DataType::I64, [2, 1]), &[3, 0]).unwrap();
        let updates = Array::vector(vec![10.0, 20.0]).unwrap();
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Add);
        let scattered = input.scatter(&indices, &updates, &operation).unwrap();
        assert_eq!(scattered, Array::vector(vec![21.0, 2.0, 3.0, 14.0]).unwrap());

        // Scatter decodes sub-byte indices through their physical layout without materializing a scalar index vector.
        let indices_type =
            ArrayType::new_static(DataType::I4, [2, 1]).with_layout(Layout::Strided(StridedLayout::new(vec![-1, 1])));
        let indices = Array::from_elements(indices_type, &[i4::new(3).unwrap(), i4::new(0).unwrap()]).unwrap();
        assert_eq!(
            input.scatter(&indices, &updates, &operation).unwrap(),
            Array::vector(vec![21.0, 2.0, 3.0, 14.0]).unwrap(),
        );

        // Input and update payloads are decoded and written through their independent physical layouts.
        let operand_type =
            ArrayType::new_static(DataType::U16, [4]).with_layout(Layout::Strided(StridedLayout::new(vec![-2])));
        let input = Array::from_elements(operand_type.clone(), &[1u16, 2, 3, 4]).unwrap();
        let updates_type =
            ArrayType::new_static(DataType::U16, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![-2])));
        let updates = Array::from_elements(updates_type, &[10u16, 20]).unwrap();
        assert_eq!(
            input.scatter(&indices, &updates, &operation),
            Array::from_elements(operand_type, &[21u16, 2, 3, 14]),
        );

        // Sub-byte arithmetic wraps in the declared bit width, including repeated modular addition.
        let input = Array::vector(vec![i4::new(7).unwrap(), i4::new(-8).unwrap()]).unwrap();
        let indices = Array::matrix(2, 1, vec![0i32, 1]).unwrap();
        let updates = Array::vector(vec![i4::new(2).unwrap(), i4::new(-3).unwrap()]).unwrap();
        assert_eq!(
            input.scatter(&indices, &updates, &operation).unwrap().elements::<i4>(),
            Ok(vec![i4::new(-7).unwrap(), i4::new(5).unwrap()]),
        );

        // Overwrite moves encodings without requiring arithmetic identities, including for formats without zero.
        let input = Array::new(ArrayType::new_static(DataType::F8E8M0FNU, [2]), vec![0x7f, 0x80]).unwrap();
        let updates = Array::new(ArrayType::new_static(DataType::F8E8M0FNU, [1]), vec![0x81]).unwrap();
        let indices = Array::matrix(1, 1, vec![0i32]).unwrap();
        let operation = ScatterOperation::new(
            ScatterDimensionNumbers::new(vec![], vec![0], vec![0]),
            ScatterReductionKind::Overwrite,
        );
        assert_eq!(
            input.scatter(&indices, &updates, &operation),
            Array::new(ArrayType::new_static(DataType::F8E8M0FNU, [2]), vec![0x81, 0x80]),
        );

        // Extrema preserve NaNs and signed zero and order complex values lexicographically.
        let indices = Array::matrix(2, 1, vec![0i32, 1]).unwrap();
        let input = Array::vector(vec![f32::NAN, -0.0]).unwrap();
        let updates = Array::vector(vec![1.0f32, 0.0]).unwrap();
        let maximum =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Max);
        let minimum =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Min);
        let maximum_values = input.scatter(&indices, &updates, &maximum).unwrap().elements::<f32>().unwrap();
        assert!(maximum_values[0].is_nan());
        assert_eq!(maximum_values[1].to_bits(), 0.0f32.to_bits());
        let minimum_values = input.scatter(&indices, &updates, &minimum).unwrap().elements::<f32>().unwrap();
        assert!(minimum_values[0].is_nan());
        assert_eq!(minimum_values[1].to_bits(), (-0.0f32).to_bits());

        let input = Array::vector(vec![ComplexNumber::new(1.0f32, 9.0), ComplexNumber::new(2.0, -1.0)]).unwrap();
        let updates = Array::vector(vec![ComplexNumber::new(1.0f32, 10.0), ComplexNumber::new(1.0, 100.0)]).unwrap();
        assert_eq!(
            input.scatter(&indices, &updates, &maximum).unwrap().elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(1.0, 10.0), ComplexNumber::new(2.0, -1.0)]),
        );
        assert_eq!(
            input.scatter(&indices, &updates, &minimum).unwrap().elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(1.0, 9.0), ComplexNumber::new(1.0, 100.0)]),
        );
    }
    #[test]
    fn test_array_scatter_empty_and_extreme_indices() {
        let input = Array::vector(Vec::<i32>::new()).unwrap();
        let indices = Array::matrix(1, 1, vec![u64::MAX]).unwrap();
        let updates = Array::vector(vec![7_i32]).unwrap();
        for mode in [GatherScatterMode::Clip, GatherScatterMode::FillOrDrop, GatherScatterMode::PromiseInBounds] {
            let operation = ScatterOperation::new(
                ScatterDimensionNumbers::new(vec![], vec![0], vec![0]),
                ScatterReductionKind::Overwrite,
            )
            .with_mode(mode);
            assert_eq!(input.scatter(&indices, &updates, &operation), Ok(input.clone()));
        }
        let input = Array::vector(vec![10_i32, 20, 30, 40]).unwrap();
        let updates = Array::matrix(1, 2, vec![1_i32, 2]).unwrap();
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![1], vec![], vec![0]), ScatterReductionKind::Add);
        for indices in [Array::matrix(1, 1, vec![i64::MAX]).unwrap(), Array::matrix(1, 1, vec![u64::MAX]).unwrap()] {
            assert_eq!(
                input.scatter(&indices, &updates, &operation.clone().with_mode(GatherScatterMode::Clip)),
                Array::vector(vec![10_i32, 20, 31, 42])
            );
            assert_eq!(
                input.scatter(&indices, &updates, &operation.clone().with_mode(GatherScatterMode::FillOrDrop)),
                Ok(input.clone())
            );
        }
    }

    #[test]
    fn test_array_ir_scatter() {
        let input = ArrayIrValue::Array(Array::vector(vec![10_i32, 20, 30]).unwrap());
        let indices = ArrayIrValue::Array(Array::matrix(2, 1, vec![2_i32, 0]).unwrap());
        let updates = ArrayIrValue::Array(Array::vector(vec![1_i32, 2]).unwrap());
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Add);
        assert_eq!(
            input.scatter(&indices, &updates, &operation),
            Ok(ArrayIrValue::Array(Array::vector(vec![12_i32, 20, 31]).unwrap()))
        );
        let dimension = ArrayIrValue::Dimension(DimensionValue::constant(1).unwrap());
        assert_eq!(
            input.scatter(&indices, &dimension, &operation),
            Err(TypeError::invalid("expected array type but got dimension type").into())
        );
    }

    #[test]
    fn test_array_scatter_axis() {
        let input = Array::matrix(2, 3, vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
        let indices = Array::vector(vec![2_i32, 0]).unwrap();
        let updates = Array::matrix(2, 2, vec![10_i32, 20, 30, 40]).unwrap();
        assert_eq!(
            input.scatter_axis(&indices, &updates, -1, ScatterReductionKind::Add, GatherScatterMode::Clip),
            Array::matrix(2, 3, vec![21_i32, 2, 13, 44, 5, 36])
        );
        assert_eq!(
            input.scatter_axis(
                &Array::scalar(0_i32).unwrap(),
                &Array::vector(vec![7_i32, 8, 9]).unwrap(),
                0,
                ScatterReductionKind::Overwrite,
                GatherScatterMode::Clip
            ),
            Array::matrix(2, 3, vec![7_i32, 8, 9, 4, 5, 6])
        );
        assert_eq!(
            input.scatter_axis(
                &Array::scalar(-1_i32).unwrap(),
                &Array::vector(vec![7_i32, 8, 9]).unwrap(),
                0,
                ScatterReductionKind::Overwrite,
                GatherScatterMode::FillOrDrop
            ),
            Ok(input.clone())
        );
        assert!(matches!(input.scatter_axis(&indices, &Array::matrix(1, 2, vec![1_i32, 2]).unwrap(), 1,
            ScatterReductionKind::Add, GatherScatterMode::Clip), Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`scatter_axis` updates shape must be `[2, 2]` but got `[1, 2]`"));

        // The update replaces the selected axis, so its window shape does not depend on that input extent.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(6)).unwrap());
        let input_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let (output_type, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |(input, indices, updates)| {
                input.scatter_axis(&indices, &updates, 0, ScatterReductionKind::Add, GatherScatterMode::Clip)
            },
            (input_type.clone(), ArrayType::new_static(DataType::I32, [2]), ArrayType::new_static(DataType::I32, [2])),
        )
        .unwrap();
        assert_eq!(output_type, input_type);
        assert_eq!(
            program.interpret((
                Array::from_elements(ArrayType::new_static(DataType::I32, [3]), &[10_i32, 20, 30]).unwrap(),
                Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[2_i32, 0]).unwrap(),
                Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[1_i32, 2]).unwrap(),
            )),
            Array::from_elements(ArrayType::new_static(DataType::I32, [3]), &[12_i32, 20, 31]),
        );
    }

    #[test]
    fn test_dynamic_scatter_dynamic_scatter_axis() {
        // Mapped queries and updates stay paired with each source row through the mixed query reshape and
        // projected scatter batching rules. Duplicate queries accumulate within their own source item.
        let input = ArrayIrValue::Array(Array::matrix(2, 4, vec![0_f64, 1., 2., 3., 4., 5., 6., 7.]).unwrap());
        let queries = ArrayIrValue::Array(Array::matrix(2, 2, vec![1_i32, 1, 0, 3]).unwrap());
        let updates = ArrayIrValue::Array(Array::matrix(2, 2, vec![10_f64, 20., 30., 40.]).unwrap());
        let (_, batched_program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(input, queries, updates)| {
                batch(
                    |(input, queries, updates)| {
                        input.dynamic_scatter_axis(
                            &queries,
                            &updates,
                            0,
                            ScatterReductionKind::Add,
                            GatherScatterMode::Clip,
                        )
                    },
                    (input, queries, updates),
                    (BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0)),
                    BatchAxis::new(0),
                    None,
                )
                .map_err(ProgramError::from)
            },
            (input.r#type().into_owned(), queries.r#type().into_owned(), updates.r#type().into_owned()),
        )
        .unwrap();
        assert_eq!(
            batched_program.interpret((input, queries, updates)),
            Ok(ArrayIrValue::Array(Array::matrix(2, 4, vec![0_f64, 31., 2., 3., 34., 5., 6., 47.]).unwrap())),
        );

        let query = DimensionVariable::new("queries", DimensionBounds::new(2, Some(4)).unwrap());
        let indices_type = ArrayType::new(DataType::I32, crate::arrays::Shape::new(vec![query.clone().into()]));
        let updates_type = ArrayType::new(DataType::F64, crate::arrays::Shape::new(vec![query.into()]));
        let input_type = ArrayType::new_static(DataType::F64, [4]);
        let (output_type, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(input, indices, updates)| {
                input.dynamic_scatter_axis(&indices, &updates, 0, ScatterReductionKind::Add, GatherScatterMode::Clip)
            },
            (ArrayIrType::from(input_type.clone()), ArrayIrType::from(indices_type), ArrayIrType::from(updates_type)),
        )
        .unwrap();
        assert_eq!(output_type, ArrayIrType::from(input_type));
        // One retained query shape specializes at both lengths. Duplicate indices accumulate all updates.
        for count in [2, 3] {
            let input = ArrayIrValue::Array(Array::vector(vec![10_f64, 20., 30., 40.]).unwrap());
            let indices = ArrayIrValue::Array(Array::vector(vec![1_i32; count]).unwrap());
            let updates = ArrayIrValue::Array(Array::vector(vec![2_f64; count]).unwrap());
            assert_eq!(
                program.interpret((input, indices, updates)),
                Ok(ArrayIrValue::Array(Array::vector(vec![10_f64, 20. + 2. * count as f64, 30., 40.]).unwrap())),
            );
        }
        let input = ArrayIrValue::Array(Array::vector(vec![10_i32, 20, 30]).unwrap());
        let indices = ArrayIrValue::Array(Array::vector(vec![0_i32, 2]).unwrap());
        let updates = ArrayIrValue::Array(Array::vector(vec![1_i32]).unwrap());
        assert_eq!(
            input.dynamic_scatter_axis(&indices, &updates, 0, ScatterReductionKind::Add, GatherScatterMode::Clip),
            Err(TypeError::invalid("`dynamic_scatter_axis` updates shape must be `[2]` but got `[1]`").into()),
        );
    }
}
