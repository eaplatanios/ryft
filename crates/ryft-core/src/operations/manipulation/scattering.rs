use std::collections::BTreeSet;
use std::fmt::Display;

use crate::arrays::{
    Array, ArrayAddressing, ArrayBatch, ArrayBatchingPolicy, ArrayElement, ArrayExtentBatchingPolicy, ArrayIrType,
    ArrayIrValue, ArrayType, DataType, Dimension, NumericArrayElement, Shape, Sharding, ShardingDimension,
    materialize_array_tangent,
};
use crate::axes::Axis;
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain, EagerContext, ProjectedContext, StagingContext};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDriver, DifferentiationDual,
    DifferentiationError, DifferentiationPolicy, ElementwiseDerivativeAlignment, MemberDifferentiableOperation,
    ResidualZeroProvider, jvp_projected_operation,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{
    check_count, dispatch_on_array_element_type, impl_differentiable_operation, impl_reference_dischargeable_operation,
};
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
    GatherDimensionNumbers, GatherMode, GatherOperation, validate_unique_in_range,
};
use crate::operations::manipulation::reshaping::{DynamicReshape, Reshape, ReshapeOperation};
use crate::operations::manipulation::transposition::Transpose;
use crate::operations::math::add::AddOperation;
use crate::operations::math::div::DivOperation;
use crate::operations::math::mul::MulOperation;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, OperationProjection, ProgramError, RegionInterface, TypeError, Typed,
    Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

/// Determines how [`Scatter`] handles windows extending outside its input. Negative indices are out of bounds;
/// they do not count backward from an axis end. The mode does not change the output shape.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum ScatterMode {
    /// The caller promises every update window is in bounds. Violating the promise leaves results and
    /// gradients undefined.
    #[default]
    PromiseInBounds,

    /// Clamps each start so the whole update window stays in bounds.
    Clip,

    /// Discards an update window when any part of it is out of bounds.
    Drop,
}

impl ScatterMode {
    /// Returns the canonical name of this [`ScatterMode`].
    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            Self::PromiseInBounds => "promise_in_bounds",
            Self::Clip => "clip",
            Self::Drop => "drop",
        }
    }
}

impl Display for ScatterMode {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.name())
    }
}

/// Reduction used when a [`Scatter`] writes an update into its input value. Each kind selects the binary reduction used
/// where an update meets the existing input value (and, for the XLA backend, for example, lowers to the corresponding
/// `stablehlo.scatter` combiner region). [`Self::Add`] supports differentiation through both the input and updates and
/// participates in the gather/scatter-add transpose duality (refer to [`ScatterOperation::is_linear`] for more
/// information). Extremal derivatives divide ties equally among matching inputs and updates. [`Self::Mul`] derivatives
/// with respect to updates require unique indices. Repeated [`Self::Overwrite`] uses a consistent winning update for
/// its primal and tangent. Non-linear derivatives require static update window sizes, and repeated overwrite
/// additionally requires a static update shape. Overlapping updates may execute in any order; a unique-index hint
/// is a caller promise and not a check.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ScatterReductionKind {
    /// The update replaces the input value.
    Overwrite,

    /// The update is added to the input value. This is linear for all index configurations.
    Add,

    /// The update is multiplied with the input value.
    Mul,

    /// The input value is replaced by the minimum of itself and the update. Booleans use conjunction, real numeric
    /// values propagate NaNs and order negative zero below positive zero, and complex values compare lexicographically
    /// by `(real, imaginary)`.
    Min,

    /// The input value is replaced by the maximum of itself and the update. Booleans use disjunction, real numeric
    /// values propagate NaNs and order negative zero below positive zero, and complex values compare lexicographically
    /// by `(real, imaginary)`.
    Max,
}

impl ScatterReductionKind {
    /// Returns the canonical name of this [`ScatterReductionKind`].
    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            Self::Overwrite => "overwrite",
            Self::Add => "add",
            Self::Mul => "mul",
            Self::Min => "min",
            Self::Max => "max",
        }
    }
}

impl Display for ScatterReductionKind {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.name())
    }
}

// TODO(eaplatanios): Review from here onwards.

/// Specification of how the index input and the update windows map onto the input axes of a [`scatter`](Scatter),
/// following StableHLO's [`scatter`](https://openxla.org/stablehlo/spec#scatter) dimension numbers. It is the
/// structural dual of [`GatherDimensionNumbers`]:
/// [`Self::update_window_dimensions`] mirrors `offset_dimensions`,
/// [`Self::inserted_window_dimensions`] mirrors `collapsed_slice_dimensions`, and
/// [`Self::scatter_dimensions_to_operand_dimensions`] mirrors
/// `start_index_map`.
///
/// The index vector dimension is implicit and always the last axis of the indices input. The output has the same
/// shape as the input.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ScatterDimensionNumbers {
    /// Refer to the documentation of [`update_window_dimensions`](Self::update_window_dimensions) for more information.
    update_window_dimensions: Vec<usize>,

    /// Refer to the documentation of [`inserted_window_dimensions`](Self::inserted_window_dimensions)
    /// for more information.
    inserted_window_dimensions: Vec<usize>,

    /// Refer to the documentation of
    /// [`scatter_dimensions_to_operand_dimensions`](Self::scatter_dimensions_to_operand_dimensions)
    /// for more information.
    scatter_dimensions_to_operand_dimensions: Vec<usize>,

    /// Refer to the documentation of [`operand_batching_dimensions`](Self::operand_batching_dimensions)
    /// for more information.
    operand_batching_dimensions: Vec<usize>,

    /// Refer to the documentation of [`scatter_indices_batching_dimensions`](Self::scatter_indices_batching_dimensions)
    /// for more information.
    scatter_indices_batching_dimensions: Vec<usize>,
}

impl ScatterDimensionNumbers {
    /// Creates a new [`ScatterDimensionNumbers`] instance from the provided explicit axis lists. The batching axis
    /// lists default to empty; use [`with_batching_dimensions`](Self::with_batching_dimensions) to set them.
    ///
    /// # Parameters
    ///
    ///   - `update_window_dimensions`: Sorted updates axes holding window coordinates, in input-axis order.
    ///   - `inserted_window_dimensions`: Sorted input axes with size-one windows and no corresponding updates axis.
    ///   - `scatter_dimensions_to_operand_dimensions`: Input axis addressed by each component of the trailing
    ///     index vector, in component order.
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

    /// Returns a copy of this [`ScatterDimensionNumbers`] with its input and query batching axes replaced by
    /// `operand_batching_dimensions` and `scatter_indices_batching_dimensions`, respectively. Each update modifies
    /// its corresponding input batch. Paired axes must have equal extents, and input batching axes cannot also be
    /// inserted or indexed by a start vector. These constraints are checked when inferring the scatter result type.
    ///
    /// # Parameters
    ///
    ///   - `operand_batching_dimensions`: Input axes, in ascending order, that select the independent batches.
    ///   - `scatter_indices_batching_dimensions`: Distinct query axes paired with the input axes in the same order.
    ///     The trailing index-vector axis cannot be a batching axis.
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

    /// Returns the axes of the updates input that hold a scattered window, in ascending order. Their count equals the
    /// number of input axes that are neither inserted nor batching.
    #[inline]
    pub fn update_window_dimensions(&self) -> &[usize] {
        &self.update_window_dimensions
    }

    /// Returns the input axes whose window size is `1` and that have no corresponding updates axis, in ascending order.
    #[inline]
    pub fn inserted_window_dimensions(&self) -> &[usize] {
        &self.inserted_window_dimensions
    }

    /// Returns the input axis targeted by each component of a start-index vector (the last axis of the indices input).
    /// The map's length equals the extent of the indices' index vector dimension.
    #[inline]
    pub fn scatter_dimensions_to_operand_dimensions(&self) -> &[usize] {
        &self.scatter_dimensions_to_operand_dimensions
    }

    /// Returns the input axes batched against [`Self::scatter_indices_batching_dimensions`], aligned one-to-one, in
    /// ascending order.
    #[inline]
    pub fn operand_batching_dimensions(&self) -> &[usize] {
        &self.operand_batching_dimensions
    }

    /// Returns the indices axes, excluding the index vector dimension, that align one-to-one with
    /// [`Self::operand_batching_dimensions`].
    #[inline]
    pub fn scatter_indices_batching_dimensions(&self) -> &[usize] {
        &self.scatter_indices_batching_dimensions
    }
}

/// Optional bounds handling, index promises, and output placement for [`Scatter::scatter`].
///
/// The default promises in-bounds indices, makes no sortedness or uniqueness promise, and infers output sharding.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ScatterOptions {
    /// Refer to the documentation of [`mode`](Self::mode) for more information.
    mode: ScatterMode,

    /// Refer to the documentation of [`indices_are_sorted`](Self::indices_are_sorted) for more information.
    indices_are_sorted: bool,

    /// Refer to the documentation of [`unique_indices`](Self::unique_indices) for more information.
    unique_indices: bool,

    /// Refer to the documentation of [`output_sharding`](Self::output_sharding) for more information.
    output_sharding: Option<Sharding>,
}

impl ScatterOptions {
    /// Creates options with [`ScatterMode::PromiseInBounds`], no sortedness or non-overlap promises, and inferred
    /// output [`Sharding`]. Use the consuming `with_*` functions to override these defaults.
    #[inline]
    pub fn new() -> Self {
        Self {
            mode: ScatterMode::PromiseInBounds,
            indices_are_sorted: false,
            unique_indices: false,
            output_sharding: None,
        }
    }

    /// Returns a copy of this [`ScatterOptions`] with its out-of-bounds index handling [`ScatterMode`] replaced by
    /// `mode`.
    #[inline]
    pub fn with_mode(mut self, mode: ScatterMode) -> Self {
        self.mode = mode;
        self
    }

    /// Returns a copy of this [`ScatterOptions`] with its sorted-indices promise set to `indices_are_sorted`. When
    /// `true`, the caller promises that start-index vectors are sorted; `false` makes no such promise. Implementations
    /// and transformations may rely on this property. This function does not sort or validate the indices.
    #[inline]
    pub fn with_indices_are_sorted(mut self, indices_are_sorted: bool) -> Self {
        self.indices_are_sorted = indices_are_sorted;
        self
    }

    /// Returns a copy of this [`ScatterOptions`] with its unique-indices promise set to `unique_indices`. When
    /// `true`, the caller promises that update windows do not overlap; `false` makes no such promise. Implementations
    /// and transformations may rely on this property. This function does not test the windows for overlap.
    /// Differentiating multiplicative updates requires this promise when the updates carry nonzero tangents.
    #[inline]
    pub fn with_unique_indices(mut self, unique_indices: bool) -> Self {
        self.unique_indices = unique_indices;
        self
    }

    /// Returns a copy of this [`ScatterOptions`] with its requested output [`Sharding`] replaced by `output_sharding`.
    /// Without an explicit request, indexed axes with partial update windows must be replicated over explicit mesh
    /// axes; complete windows
    /// preserve their input placement. An explicit request selects the result placement while preserving its mesh,
    /// reduction state, and manual-axis variation.
    ///
    /// The request must have the input rank and cannot reference automatic mesh axes. Passing `None` restores
    /// inferred placement. Validation takes place when inferring the result type.
    #[inline]
    pub fn with_output_sharding<S: Into<Option<Sharding>>>(mut self, output_sharding: S) -> Self {
        self.output_sharding = output_sharding.into();
        self
    }

    /// Returns the out-of-bounds index handling mode.
    #[inline]
    pub fn mode(&self) -> ScatterMode {
        self.mode
    }

    /// Returns whether the caller promises that the index vectors are sorted. This property is not checked.
    /// Implementations and transformations may rely on it; `false` makes no such promise.
    #[inline]
    pub fn indices_are_sorted(&self) -> bool {
        self.indices_are_sorted
    }

    /// Returns whether the caller promises that the scattered windows do not overlap. This property is not checked.
    /// Implementations and transformations may rely on it; `false` makes no such promise. In particular, uniqueness
    /// permits direct linear rules for overwrite scatter and is required for multiplication derivatives with respect
    /// to updates.
    #[inline]
    pub fn unique_indices(&self) -> bool {
        self.unique_indices
    }

    /// Returns the requested output [`Sharding`], if any, used when the inferred placement is ambiguous. Refer to the
    /// documentation of [`with_output_sharding`](Self::with_output_sharding) for more information.
    #[inline]
    pub fn output_sharding(&self) -> Option<&Sharding> {
        self.output_sharding.as_ref()
    }
}

impl Default for ScatterOptions {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Canonical operation name for [`ScatterOperation`].
pub const SCATTER_OPERATION_NAME: &str = "scatter";

/// [`Operation`] that writes update windows into a copy of an input at positions named by an integer index input,
/// combining overlaps with a [`ScatterReductionKind`]. Refer to the documentation of [`Scatter`] for the semantics.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ScatterOperation {
    /// Refer to the documentation of [`dimensions`](Self::dimensions) for more information.
    dimensions: ScatterDimensionNumbers,

    /// Refer to the documentation of [`kind`](Self::kind) for more information.
    kind: ScatterReductionKind,

    /// Refer to the documentation of [`options`](Self::options) for more information.
    options: ScatterOptions,
}

impl ScatterOperation {
    /// Creates a new [`ScatterOperation`] with the provided dimension numbers and combiner kind. The mode defaults to
    /// [`ScatterMode::PromiseInBounds`] and both index promises default to `false`; use the chained `with_*`
    /// builders to override them.
    ///
    /// # Parameters
    ///
    ///   - `dimensions`: Mapping from index components and update window axes to input axes.
    ///   - `kind`: Combiner applied to the existing input value and every update targeting that value.
    #[inline]
    pub fn new(dimensions: ScatterDimensionNumbers, kind: ScatterReductionKind) -> Self {
        Self { dimensions, kind, options: ScatterOptions::new() }
    }

    /// Returns a copy of this [`ScatterOperation`] with its optional behavior and output placement replaced by
    /// `options`.
    #[inline]
    pub fn with_options(mut self, options: ScatterOptions) -> Self {
        self.options = options;
        self
    }

    /// Returns a copy of this [`ScatterOperation`] with its out-of-bounds index handling [`ScatterMode`] replaced by
    /// `mode`.
    #[inline]
    pub fn with_mode(mut self, mode: ScatterMode) -> Self {
        self.options = self.options.with_mode(mode);
        self
    }

    /// Returns a copy of this [`ScatterOperation`] with its sorted-indices promise set to `indices_are_sorted`. When
    /// `true`, the caller promises that start-index vectors are sorted; `false` makes no such promise. Implementations
    /// and transformations may rely on this property. This function does not sort or validate the indices.
    #[inline]
    pub fn with_indices_are_sorted(mut self, indices_are_sorted: bool) -> Self {
        self.options = self.options.with_indices_are_sorted(indices_are_sorted);
        self
    }

    /// Returns a copy of this [`ScatterOperation`] with its unique-indices promise set to `unique_indices`. When
    /// `true`, the caller promises that update windows do not overlap; `false` makes no such promise. Implementations
    /// and transformations may rely on this property. This function does not test the windows for overlap.
    /// Differentiating multiplicative updates requires this promise when the updates carry nonzero tangents.
    #[inline]
    pub fn with_unique_indices(mut self, unique_indices: bool) -> Self {
        self.options = self.options.with_unique_indices(unique_indices);
        self
    }

    /// Returns a copy of this [`ScatterOperation`] with its requested output [`Sharding`] replaced by
    /// `output_sharding`.
    /// Without an explicit request, indexed axes with partial update windows must be replicated over explicit mesh
    /// axes; complete windows
    /// preserve their input placement. An explicit request selects the result placement while preserving its mesh,
    /// reduction state, and manual-axis variation.
    ///
    /// The request must have the input rank and cannot reference automatic mesh axes. Passing `None` restores
    /// inferred placement. Validation takes place when inferring the result type.
    #[inline]
    pub fn with_output_sharding<S: Into<Option<Sharding>>>(mut self, output_sharding: S) -> Self {
        self.options = self.options.with_output_sharding(output_sharding);
        self
    }

    /// Returns the dimension numbers mapping the index input and update windows onto the input axes.
    #[inline]
    pub fn dimensions(&self) -> &ScatterDimensionNumbers {
        &self.dimensions
    }

    /// Returns the combiner applied where an update meets the existing input value.
    #[inline]
    pub fn kind(&self) -> ScatterReductionKind {
        self.kind
    }

    /// Returns the bounds handling, index promises, and output placement for this operation.
    #[inline]
    pub fn options(&self) -> &ScatterOptions {
        &self.options
    }

    /// Returns the out-of-bounds index handling mode.
    #[inline]
    pub fn mode(&self) -> ScatterMode {
        self.options.mode()
    }

    /// Returns whether the caller promises that the index vectors are sorted. This property is not checked.
    /// Implementations and transformations may rely on it; `false` makes no such promise.
    #[inline]
    pub fn indices_are_sorted(&self) -> bool {
        self.options.indices_are_sorted()
    }

    /// Returns whether the caller promises that the scattered windows do not overlap. This property is not checked.
    /// Implementations and transformations may rely on it; `false` makes no such promise. In particular, uniqueness
    /// permits direct linear rules for overwrite scatter and is required for multiplication derivatives with respect
    /// to updates.
    #[inline]
    pub fn unique_indices(&self) -> bool {
        self.options.unique_indices()
    }

    /// Returns the requested output [`Sharding`], if any, used when the inferred placement is ambiguous. Refer to the
    /// documentation of [`with_output_sharding`](Self::with_output_sharding) for more information.
    #[inline]
    pub fn output_sharding(&self) -> Option<&Sharding> {
        self.options.output_sharding()
    }

    /// Returns `true` when this scatter is a linear map in its input and updates with the indices held fixed, which is
    /// the case for the [`Add`](ScatterReductionKind::Add) combiner and for
    /// [`Overwrite`](ScatterReductionKind::Overwrite) with the unique-indices promise (each output element then comes
    /// from exactly one source). These are the scatters with direct linear derivative rules: their tangent is the same
    /// scatter of the tangents, and their transpose is a dual gather (plus, for overwrite, erasing the written windows
    /// of the input cotangent). The other combiners use coefficients computed from the primals and have no transpose.
    #[inline]
    pub fn is_linear(&self) -> bool {
        self.kind == ScatterReductionKind::Add
            || (self.kind == ScatterReductionKind::Overwrite && self.options.unique_indices)
    }

    /// Builds the [`GatherOperation`] that reads the windows targeted by this [`ScatterOperation`]. In the
    /// scatter-add transpose, gathering the output cotangent gives one cotangent per update, including a separate
    /// copy for every repeated index. For example, scalar updates at indices `[2, 0, 2]` receive cotangents
    /// `[c, a, c]` from an output cotangent `[a, b, c]`. Nonlinear derivative rules use the same mapping to read
    /// primal values or winner identifiers at the update locations.
    ///
    /// Update window axes become gather offset axes, inserted input axes become collapsed axes, and the index map
    /// and paired batching axes are retained. Window sizes come from the static update extents; inserted axes use
    /// one, while paired axes use zero if their extent may be empty and one otherwise. A dynamic inserted axis
    /// must have a positive lower bound because the gather collapses it through a size-one window.
    ///
    /// Clipping and in-bounds promises retain their policies. Drop mode becomes fill mode; callers choose the fill
    /// appropriate to the derivative: the transpose and overwrite winner identifiers use a typed zero, while
    /// extremal coefficients retain the default fill (NaN for floating-point inputs). Sortedness and non-overlap
    /// promises also carry over. This function constructs the operation without executing the gather.
    ///
    /// # Parameters
    ///
    ///   - `input_type`: Type of the scattered input, whose axes size the inserted and batching windows.
    ///   - `updates_type`: Type of the updates, whose window axes size the remaining windows.
    ///   - `output_sharding`: Requested placement of the gathered windows.
    fn adjoint_gather_operation(
        &self,
        input_type: &ArrayType,
        updates_type: &ArrayType,
        output_sharding: Option<Sharding>,
    ) -> Result<GatherOperation, ProgramError> {
        let dimensions = &self.dimensions;
        let mut slice_sizes = Vec::with_capacity(input_type.rank());
        let mut window_position = 0;
        for axis in 0..input_type.rank() {
            if dimensions.inserted_window_dimensions().contains(&axis) {
                if let Dimension::Dynamic(variable) = input_type.dimension(axis)
                    && variable.bounds().lower() == 0
                {
                    return Err(ProgramError::UnsupportedOperation {
                        message: format!(
                            "`{SCATTER_OPERATION_NAME}` differentiation requires inserted window axis {axis} to have a \
                             nonzero minimum extent, because its dual gather collapses that axis through a one-element \
                             window"
                        ),
                    });
                }
                slice_sizes.push(1);
            } else if dimensions.operand_batching_dimensions().contains(&axis) {
                // A possibly empty paired axis needs a zero window to satisfy gather's extent bounds. Its
                // output extent still comes from the paired indices dimension, so nonempty batches stay nonempty.
                slice_sizes.push(input_type.dimension(axis).bounds().lower().min(1));
            } else {
                let update_axis = dimensions.update_window_dimensions()[window_position];
                slice_sizes.push(updates_type.dimension(update_axis).value().ok_or_else(|| {
                    ProgramError::UnsupportedOperation {
                        message: format!(
                            "`{SCATTER_OPERATION_NAME}` differentiation requires a static update window on axis \
                             {update_axis} but its extent is `{}`",
                            updates_type.dimension(update_axis),
                        ),
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
            dimensions
                .operand_batching_dimensions()
                .iter()
                .copied()
                .zip(dimensions.scatter_indices_batching_dimensions().iter().copied())
                .collect(),
        );
        Ok(GatherOperation::new(gather_dimensions, slice_sizes)
            .with_mode(match self.options.mode {
                ScatterMode::PromiseInBounds => GatherMode::PromiseInBounds,
                ScatterMode::Clip => GatherMode::Clip,
                ScatterMode::Drop => GatherMode::Fill { value: None },
            })
            .with_indices_are_sorted(self.options.indices_are_sorted)
            .with_unique_indices(self.options.unique_indices)
            .with_output_sharding(output_sharding))
    }

    /// Shares coefficient construction between homogeneous and projected differentiation. Coefficients depend only
    /// on primals; the staged tangent graph uses ordinary linear gather/scatter and elementwise operations.
    ///
    /// # Parameters
    ///
    ///   - `contexts`: The primal context, in which coefficients are computed, and the tangent context, in which the
    ///     linear tangent graph is staged.
    ///   - `inputs`: The primal input, indices, and updates.
    ///   - `primal`: The primal output of this scatter, which repeated overwrite replaces by its winner reconstruction.
    ///   - `tangents`: The materialized input and update tangents, each paired with whether it is a structural zero so
    ///     that inactive terms are omitted rather than multiplied by possibly nonfinite coefficients.
    ///   - `primal_to_tangent`: Transfers a primal-context value across the differentiation boundary.
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
        let gather = |context: &C, input: &C::Value, indices: &C::Value, operation: &GatherOperation| {
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
        // An empty input has no destinations, so every update is inactive for every combiner. Preserve the input
        // tangent and its requested result metadata without constructing a dual gather with an invalid size-one
        // window along an empty axis.
        if input.r#type().element_count().map_err(|error| TypeError::invalid(error.to_string()))? == Some(0) {
            let tangent = align(tangent_context, input_tangent, &primal.r#type().tangent()?)?;
            return Ok((primal, tangent));
        }
        if self.is_linear() {
            return Ok((
                primal,
                scatter(tangent_context, input_tangent, &primal_to_tangent(indices.clone())?, updates_tangent, self)?,
            ));
        }
        let zeros = zero(tangent_context, input_tangent)?;
        let mut additive = self.clone();
        additive.kind = ScatterReductionKind::Add;
        if self.kind() == ScatterReductionKind::Mul {
            if !updates_are_zero && !self.unique_indices() {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "`{SCATTER_OPERATION_NAME}` multiplication derivatives with respect to updates require \
                         `unique_indices=true`"
                    ),
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
        let update_zeros = zero(tangent_context, updates_tangent)?;
        let input_type = input.r#type();
        let updates_type = updates.r#type();
        let mut dual_gather = self.adjoint_gather_operation(
            input_type.as_ref(),
            updates_type.as_ref(),
            updates_type.sharding().cloned(),
        )?;
        if self.kind() == ScatterReductionKind::Overwrite {
            // Distinct positive IDs select one winning update at each output element. Reconstruct both primal
            // and tangent from those same winners, since duplicate overwrite order is unspecified by the backend.
            let update_shape = updates_type.static_shape().ok_or_else(|| ProgramError::UnsupportedOperation {
                message: format!(
                    "`{SCATTER_OPERATION_NAME}` overwrite differentiation with repeated indices requires a static \
                     update shape"
                ),
            })?;
            let count =
                update_shape.as_slice().iter().try_fold(1usize, |count, size| count.checked_mul(*size)).ok_or_else(
                    || TypeError::invalid(format!("`{SCATTER_OPERATION_NAME}` update element count overflows `usize`")),
                )?;
            if count == usize::MAX {
                return Err(TypeError::invalid(format!(
                    "`{SCATTER_OPERATION_NAME}` update IDs (the element count plus one) overflow `u64`"
                ))
                .into());
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
            dual_gather = dual_gather.with_output_sharding(update_ids_type.sharding().cloned());
            if self.mode() == ScatterMode::Drop {
                // A dropped window must not match any positive ID, so pin a zero fill instead of gather's default.
                dual_gather = dual_gather.with_mode(GatherMode::Fill { value: Some(Box::new(Array::scalar(0u64)?)) });
            }
            let gathered_ids = gather(context, &scattered_ids, indices, &dual_gather)?;
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
        let selected_input = select(context, &untouched, &one(context, &selected_input)?, &selected_input)?;
        // The dual gather keeps its default fill: NaN cannot match a floating-point update. Integer extremes can
        // match integer updates, but integer tangents are structural zeros. Moreover, the count and numerator
        // scatters below discard the same out-of-bounds windows, so dropped updates cannot affect the output tangent.
        let targets = gather(context, &primal, indices, &dual_gather)?;
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
    #[inline]
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
        match input_types[0].scatter(&input_types[1], &input_types[2], self.dimensions(), self.kind(), self.options()) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError::invalid(error.to_string())),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("kind", self.kind)?;
            operation.field(
                "dimensions",
                format_args!(
                    "(update_window={:?}, inserted_window={:?}, scatter_to_operand={:?}, operand_batching={:?}, \
                     scatter_indices_batching={:?})",
                    self.dimensions.update_window_dimensions,
                    self.dimensions.inserted_window_dimensions,
                    self.dimensions.scatter_dimensions_to_operand_dimensions,
                    self.dimensions.operand_batching_dimensions,
                    self.dimensions.scatter_indices_batching_dimensions,
                ),
            )?;
            if self.options.mode != ScatterMode::PromiseInBounds {
                operation.field("mode", self.options.mode)?;
            }
            if self.options.indices_are_sorted {
                operation.field("indices_are_sorted", self.options.indices_are_sorted)?;
            }
            if self.options.unique_indices {
                operation.field("unique_indices", self.options.unique_indices)?;
            }
            if let Some(output_sharding) = &self.options.output_sharding {
                operation.field("output_sharding", output_sharding)?;
            }
            Ok(())
        })
    }
}

impl_reference_dischargeable_operation!(@reference_free ScatterOperation);

impl<C: Domain<Type = ArrayType, Value: Scatter>> InterpretableOperation<C> for ScatterOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 3, ProgramError);
        Ok(vec![inputs[0].scatter(&inputs[1], &inputs[2], self.dimensions(), self.kind(), self.options())?])
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
                message: format!("`{SCATTER_OPERATION_NAME}` does not support bounded ragged array inputs"),
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
                        "`{SCATTER_OPERATION_NAME}` mapped input extent {} does not match batching extent \
                         {axis_dimension}",
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
        // The index promises stay valid in both cases: with replicated indices every item writes into its own slice of
        // the new leading window axis, and with mapped indices the paired batching axes keep every item's windows
        // within its own input, so windows that were disjoint stay disjoint.
        let operation = Self::new(lifted_dimensions, self.kind())
            .with_mode(self.mode())
            .with_indices_are_sorted(self.indices_are_sorted())
            .with_unique_indices(self.unique_indices())
            .with_output_sharding(
                self.output_sharding()
                    .map(|output_sharding| {
                        output_sharding.with_leading_batch_axis(ArrayBatch::sharding_for_inputs(inputs)?)
                    })
                    .transpose()?,
            );
        Ok(operation
            .interpret_with_batch_axes(context, &[input, indices, updates], &[BatchAxis::from_position(0)])?
            .into())
    }
}

impl_differentiable_operation! {
    ScatterOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType> + Zero<C::Value>,
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
        |operation, context, _driver, inputs| {
            // Coefficients are constructed in the primal context and transferred through the differentiation boundary
            // before they multiply tangent values. Structural zeros avoid constructing inactive products with nonfinite
            // coefficients.
            check_count!("input", inputs, 3, ProgramError);
            let input = &inputs[0];
            let indices = inputs[1].primal();
            let updates = &inputs[2];
            let mut primal = input.primal().scatter(
                indices,
                updates.primal(),
                operation.dimensions(),
                operation.kind(),
                operation.options(),
            )?;
            let tangent = if input.tangent().is_zero() && updates.tangent().is_zero() {
                MaybeZero::Zero(primal.r#type().tangent()?)
            } else {
                let input_tangent = input.tangent().clone().materialize(context.tangent())?;
                let updates_tangent = updates.tangent().clone().materialize(context.tangent())?;
                let (linearized_primal, tangent) = operation.linearize_values(
                    (context.primal(), context.tangent()),
                    [input.primal(), indices, updates.primal()],
                    primal.clone(),
                    [(&input_tangent, input.tangent().is_zero()), (&updates_tangent, updates.tangent().is_zero())],
                    |value| context.primal_to_tangent(value).map_err(ProgramError::from),
                )?;
                // The winner-ID rule also defines the primal overwrite choice.
                if operation.kind() == ScatterReductionKind::Overwrite && !operation.unique_indices() {
                    primal = linearized_primal;
                }
                MaybeZero::Value(tangent)
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType>
            + From<ZeroOperation<ArrayType>>
            + From<GatherOperation>
            + From<ScatterOperation>,
        Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<ArrayType>,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            // Partition-aware transpose rule for the primal [`ScatterOperation`] with an
            // [`Add`](ScatterReductionKind::Add) combiner. The integer index input (input 1) has no tangent space, so
            // in a valid pushforward it is the known input while the scattered input (input 0) and the updates (input
            // 2) are the linear ones. Scatter-add accumulates into its input (`output = input + scattered(updates)`, so
            // the input Jacobian is the identity), so the input cotangent is the output cotangent unchanged; the update
            // cotangent gathers the output cotangent at the scattered windows via the dual gather built by mirroring
            // the scatter geometry. The transpose reads the known indices from the pullback boundary and stages an
            // ordinary [`GatherOperation`], so linearization retains the indices as regular SSA residuals. The indices
            // receive a structural zero, and a zero output cotangent stays a structural zero. Unique-index overwrite
            // erases the input cotangent at the written windows; other combiners are rejected.
            check_count!("input", inputs, 3, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 3, DifferentiationError);
            // A structural-zero output cotangent contributes nothing for every combiner. Untouched accumulators
            // default to structural zeros when the transposition context collects its cotangents.
            let MaybeZero::Value(cotangent) = &outputs[0] else {
                return Ok(());
            };
            if !operation.is_linear() {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "transposition of `{}` with the `{}` combiner requires scatter-add or unique-index overwrite",
                        SCATTER_OPERATION_NAME,
                        operation.kind(),
                    ),
                }
                .into());
            }
            // Empty inputs have no writable locations, including in clipping mode. The base keeps its
            // identity edge and update cotangents remain structural zeros; no size-one gather is valid here.
            let element_count =
                inputs[0].r#type().element_count().map_err(|error| TypeError::invalid(error.to_string()))?;
            if element_count == Some(0) {
                if accumulators[0].is_needed() {
                    let contribution = cotangent.unalign_cotangent(&inputs[0].r#type().cotangent()?)?;
                    accumulators[0].accumulate(context, MaybeZero::Value(contribution))?;
                }
                return Ok(());
            }
            if accumulators[0].is_needed() {
                let contribution = if operation.kind() == ScatterReductionKind::Overwrite {
                    // Unique replacement windows erase the input tangent exactly where updates are written.
                    let update_zeros = MaybeZero::Zero(inputs[2].r#type().cotangent()?).materialize(&**context)?;
                    let indices = inputs[1]
                        .as_known()
                        .ok_or_else(|| {
                            TypeError::invalid(format!("`{SCATTER_OPERATION_NAME}` transpose requires known indices"))
                        })?
                        .clone();
                    let mut contributions = context.stage_operation(
                        operation.clone().with_output_sharding(inputs[0].r#type().cotangent()?.sharding().cloned()),
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
            // The indices are the known input: an integer type has no tangent space, so a valid pullback never
            // routes them as the linear input.
            let indices = inputs[1]
                .as_known()
                .ok_or_else(|| {
                    TypeError::invalid(format!("`{SCATTER_OPERATION_NAME}` transpose requires known indices"))
                })?
                .clone();
            let updates_type = inputs[2].r#type();
            let mut gather_operation = operation.adjoint_gather_operation(
                inputs[0].r#type().as_ref(),
                updates_type.as_ref(),
                updates_type.cotangent()?.sharding().cloned(),
            )?;
            // Dropped updates have zero derivative, independent of gather's default replacement value.
            if operation.mode() == ScatterMode::Drop {
                gather_operation = gather_operation.with_mode(GatherMode::Fill {
                    value: Some(Box::new(EagerContext::<Array>::new().zero(&ArrayType::scalar(cotangent.r#type().data_type()))?)),
                });
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
    },
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
        let [input, _, updates] = inputs else {
            return Err(ProgramError::InvalidInputCount { expected: 3, actual: inputs.len() }.into());
        };
        let input_type = <&ArrayType>::try_from(input.primal().r#type().as_ref())?.clone();
        let updates_type = <&ArrayType>::try_from(updates.primal().r#type().as_ref())?.clone();
        let is_static = |r#type: &ArrayType| {
            r#type.shape().dimensions().iter().all(|dimension| matches!(dimension, Dimension::Static(_)))
        };
        let operation = <C::Operation as OperationProjection<ArrayType>>::Projected::from(self.clone());
        if is_static(&input_type) && is_static(&updates_type) {
            return jvp_projected_operation(destinations, &operation, inputs);
        }

        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let mut primal_outputs = destinations.primal().bind(operation.clone(), Vec::new(), primal_inputs.as_slice())?;
        check_count!("output", primal_outputs, 1, ProgramError);
        let mut output_primal = primal_outputs.remove(0);
        let tangent_primal = destinations.primal_to_tangent(output_primal.clone())?;
        let tangent_inputs = destinations.dual_primal_to_tangent(inputs)?;
        let input = &tangent_inputs[0];
        let updates = &tangent_inputs[2];
        let tangent_context = destinations.tangent();
        let tangent = if input.tangent().is_zero() && updates.tangent().is_zero() {
            MaybeZero::Zero(tangent_primal.r#type().tangent()?)
        } else {
            let projected_context = ProjectedContext::<C, ArrayType>::new(tangent_context.clone());
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

/// Combines update windows with an array at positions supplied by an integer index array.
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
/// [`Scatter`] fills the same role for [`ScatterOperation`] that [`std::ops::Add`] and [`std::ops::Neg`] fill for their
/// corresponding arithmetic [`Operation`]s. Use [`DynamicScatter::dynamic_scatter_axis`] when the query shape must
/// remain dynamic.
///
/// # Examples
///
/// ```rust
/// use ryft_core::{Array, Scatter, ScatterDimensionNumbers, ScatterOptions, ScatterReductionKind};
///
/// // Shapes: input [3, 2], indices [2, 1], updates [2, 2] -> output [3, 2].
/// let input = Array::matrix(3, 2, vec![0.0; 6]).unwrap();
/// let indices = Array::matrix(2, 1, vec![0_i32, 2]).unwrap();
/// let updates = Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// // Update axis 1 holds each full row; input axis 0 is supplied by the row index.
/// let dimensions = ScatterDimensionNumbers::new(vec![1], vec![0], vec![0]);
/// let output = input.scatter(
///     &indices, &updates, &dimensions, ScatterReductionKind::Add, &ScatterOptions::new(),
/// ).unwrap();
/// assert_eq!(output, Array::matrix(3, 2, vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0]).unwrap());
/// ```
pub trait Scatter: Sized {
    /// Scatters `updates` into `self` at the positions named by `indices`, combining each update with the existing
    /// input value according to `kind`. The dimension numbers describe the windows, and `options` controls bounds
    /// handling, index promises, and output placement.
    ///
    /// # Parameters
    ///
    ///   - `indices`: Integer start-index vectors. The trailing axis contains the coordinates selected by
    ///     `dimensions`; preceding axes enumerate update windows.
    ///   - `updates`: Values to combine with the input. Window and indexing axes are identified by `dimensions`, and
    ///     the element type must equal the input element type.
    ///   - `dimensions`: Mapping from index components and update window axes to input axes.
    ///   - `kind`: How each update combines with the input value at its target.
    ///   - `options`: Bounds mode, index promises, and optional output sharding. Sortedness and uniqueness are
    ///     unchecked caller promises; they do not request sorting or deduplication.
    fn scatter(
        &self,
        indices: &Self,
        updates: &Self,
        dimensions: &ScatterDimensionNumbers,
        kind: ScatterReductionKind,
        options: &ScatterOptions,
    ) -> Result<Self, ProgramError>;

    /// Scatters complete slices along one axis using raw integer indices. The index shape replaces the selected
    /// input axis in the required updates shape; all other input axes retain their full size and order. Negative
    /// indices are out of bounds and are handled directly by `mode`, without wrapping them from the axis end.
    ///
    /// The selected input axis may have a dynamic extent. The index shape must support the homogeneous [`Reshape`]
    /// used to append an index-vector axis; remaining input and update dimensions must satisfy [`ScatterOperation`]'s
    /// window constraints. Use [`Scatter::scatter`] for partial windows or [`DynamicScatter`] for dynamic queries.
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
    /// use ryft_core::{Array, ScatterMode, Scatter, ScatterReductionKind};
    ///
    /// // Shapes: input [3], indices [2], updates [2] -> output [3].
    /// let input = Array::vector(vec![10_i32, 20, 30]).unwrap();
    /// let indices = Array::vector(vec![2_i32, 0]).unwrap();
    /// let updates = Array::vector(vec![1_i32, 2]).unwrap();
    /// let output = input.scatter_axis(
    ///     &indices, &updates, 0, ScatterReductionKind::Add, ScatterMode::Clip,
    /// ).unwrap();
    /// assert_eq!(output, Array::vector(vec![12_i32, 20, 31]).unwrap());
    /// ```
    fn scatter_axis<A: Into<Axis>>(
        &self,
        indices: &Self,
        updates: &Self,
        axis: A,
        kind: ScatterReductionKind,
        mode: ScatterMode,
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
        let indices = indices.reshape(Shape::new(indices_dimensions))?;
        let window_dimensions = (0..input_type.rank())
            .filter(|input_axis| *input_axis != axis)
            .map(|input_axis| if input_axis < axis { input_axis } else { input_axis + indices_type.rank() - 1 })
            .collect();
        self.scatter(
            &indices,
            updates,
            &ScatterDimensionNumbers::new(window_dimensions, vec![axis], vec![axis]),
            kind,
            &ScatterOptions::new().with_mode(mode),
        )
    }
}

impl Scatter for ArrayType {
    // Type-level scatter: validates the dimension numbers, the updates shape, and the data types, and computes the
    // output type (which retains the input shape, element type, and layout) and placement.
    fn scatter(
        &self,
        indices: &Self,
        updates: &Self,
        dimensions: &ScatterDimensionNumbers,
        kind: ScatterReductionKind,
        options: &ScatterOptions,
    ) -> Result<Self, ProgramError> {
        let input = self;
        let input_rank = input.rank();
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
                     in `{}`, `{}`, and `{}`",
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
        match kind {
            ScatterReductionKind::Overwrite => {}
            ScatterReductionKind::Add | ScatterReductionKind::Mul
                if !data_type.is_numeric() && data_type != DataType::Zero =>
            {
                return Err(TypeError::invalid(format!(
                    "`{SCATTER_OPERATION_NAME}` kind `{}` requires numeric input and update elements but got \
                     `{data_type}`",
                    kind,
                ))
                .into());
            }
            ScatterReductionKind::Min | ScatterReductionKind::Max
                if !data_type.is_boolean() && !data_type.is_numeric() && data_type != DataType::Zero =>
            {
                return Err(TypeError::invalid(format!(
                    "`{SCATTER_OPERATION_NAME}` kind `{}` requires Boolean or numeric input and update elements \
                     but got `{data_type}`",
                    kind,
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

        validate_unique_in_range(
            SCATTER_OPERATION_NAME,
            "update_window_dimensions",
            dimensions.update_window_dimensions(),
            updates_rank,
            true,
        )?;
        validate_unique_in_range(
            SCATTER_OPERATION_NAME,
            "inserted_window_dimensions",
            dimensions.inserted_window_dimensions(),
            input_rank,
            true,
        )?;
        validate_unique_in_range(
            SCATTER_OPERATION_NAME,
            "operand_batching_dimensions",
            dimensions.operand_batching_dimensions(),
            input_rank,
            true,
        )?;
        if dimensions.scatter_dimensions_to_operand_dimensions().len() != index_vector_extent {
            return Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` `scatter_dimensions_to_operand_dimensions` has length {} but the index \
                     vector extent is {index_vector_extent}",
                dimensions.scatter_dimensions_to_operand_dimensions().len(),
            ))
            .into());
        }
        validate_unique_in_range(
            SCATTER_OPERATION_NAME,
            "scatter_dimensions_to_operand_dimensions",
            dimensions.scatter_dimensions_to_operand_dimensions(),
            input_rank,
            false,
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
            false,
        )?;
        if dimensions.scatter_indices_batching_dimensions().contains(&index_vector_dimension) {
            return Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` `scatter_indices_batching_dimensions` cannot name the index vector \
                 dimension {index_vector_dimension}"
            ))
            .into());
        }

        let inserted: BTreeSet<usize> = dimensions.inserted_window_dimensions().iter().copied().collect();
        let operand_batching: BTreeSet<usize> = dimensions.operand_batching_dimensions().iter().copied().collect();
        if inserted.intersection(&operand_batching).next().is_some() {
            return Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` `inserted_window_dimensions` and `operand_batching_dimensions` must be \
                     disjoint"
            ))
            .into());
        }

        if dimensions
            .scatter_dimensions_to_operand_dimensions()
            .iter()
            .any(|axis| operand_batching.contains(axis))
        {
            return Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` indexed input axes and batching input axes must be disjoint"
            ))
            .into());
        }

        // Rank decomposition: the input axes split into window, inserted, and batching axes; the updates axes split
        // into window axes and the scatter/batch axes carried from the indices (every indices axis but the index
        // vector).
        if input_rank != dimensions.update_window_dimensions().len() + inserted.len() + operand_batching.len() {
            return Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` input rank {input_rank} must equal update_window + inserted_window + \
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
        let input_window_axes: Vec<usize> = (0..input_rank)
            .filter(|axis| !inserted.contains(axis) && !operand_batching.contains(axis))
            .collect();
        for (&input_axis, &update_axis) in input_window_axes.iter().zip(dimensions.update_window_dimensions()) {
            if let (Dimension::Static(update_extent), Dimension::Static(input_extent)) =
                (updates.dimension(update_axis), input.dimension(input_axis))
                && update_extent > input_extent
            {
                return Err(TypeError::invalid(format!(
                    "`{SCATTER_OPERATION_NAME}` update window axis {update_axis} extent {update_extent} exceeds \
                         the input window axis {input_axis} extent {input_extent}"
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
            if !updates.dimension(update_axis).has_equal_extents(&indices.dimension(indices_axis)) {
                return Err(TypeError::invalid(format!(
                    "`{SCATTER_OPERATION_NAME}` updates scatter axis {update_axis} must match indices batch axis \
                         {indices_axis} in extent"
                ))
                .into());
            }
        }

        // Batching extents must match between input and indices.
        for (&input_axis, &indices_axis) in dimensions
            .operand_batching_dimensions()
            .iter()
            .zip(dimensions.scatter_indices_batching_dimensions())
        {
            if !input.dimension(input_axis).has_equal_extents(&indices.dimension(indices_axis)) {
                return Err(TypeError::invalid(format!(
                    "`{SCATTER_OPERATION_NAME}` batching dimensions must have equal extents, but input axis \
                         {input_axis} and indices axis {indices_axis} differ"
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
        if let Some(mesh) = &common_mesh
            && [input.sharding(), indices.sharding(), updates.sharding()]
                .into_iter()
                .flatten()
                .any(|sharding| sharding.mesh() != mesh)
        {
            return Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` input, indices, and updates shardings must use one mesh",
            ))
            .into());
        }
        let unreduced_axes = input.sharding().map(Sharding::unreduced_axes).cloned().unwrap_or_default();
        let reduced_axes = input.sharding().map(Sharding::reduced_axes).cloned().unwrap_or_default();
        let updates_unreduced = updates.sharding().map(Sharding::unreduced_axes).cloned().unwrap_or_default();
        let updates_reduced = updates.sharding().map(Sharding::reduced_axes).cloned().unwrap_or_default();
        if unreduced_axes != updates_unreduced || reduced_axes != updates_reduced {
            return Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` input and updates must have matching reduction state"
            ))
            .into());
        }
        if !unreduced_axes.is_empty() && !matches!(kind, ScatterReductionKind::Add | ScatterReductionKind::Overwrite) {
            return Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` nonlinear reductions do not support unreduced inputs"
            ))
            .into());
        }
        if indices
            .sharding()
            .is_some_and(|sharding| !sharding.unreduced_axes().is_empty() || !sharding.reduced_axes().is_empty())
        {
            return Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` indices cannot carry reduced or unreduced mesh axes"
            ))
            .into());
        }
        // Reduced axes carry identical values across devices, while unreduced axes carry partial contributions.
        // Both require consistent index routing: varying indices could break replication or route partial sums to
        // different destinations. The dual gather used by the derivative rules enforces the same index contract.
        if (!unreduced_axes.is_empty() || !reduced_axes.is_empty())
            && indices.sharding().is_some_and(|sharding| {
                !sharding.varying_manual_axes().is_empty()
                    || sharding.dimensions().iter().any(|dimension| *dimension != ShardingDimension::Replicated)
            })
        {
            return Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` reduction-state inputs require replicated, invariant indices"
            ))
            .into());
        }
        let mut varying_manual_axes = input.sharding().map(Sharding::varying_manual_axes).cloned().unwrap_or_default();
        for sharding in [indices.sharding(), updates.sharding()].into_iter().flatten() {
            varying_manual_axes.extend(sharding.varying_manual_axes().iter().cloned());
        }
        let sharding = if let Some(requested) = options.output_sharding() {
            if common_mesh.as_ref().is_some_and(|mesh| requested.mesh() != mesh) {
                return Err(TypeError::invalid(format!(
                    "`{SCATTER_OPERATION_NAME}` requested output sharding uses a different mesh"
                ))
                .into());
            }
            if requested.unreduced_axes() != &unreduced_axes
                || requested.reduced_axes() != &reduced_axes
                || requested.varying_manual_axes() != &varying_manual_axes
            {
                return Err(TypeError::invalid(format!(
                    "`{SCATTER_OPERATION_NAME}` requested output sharding changes reduction or manual-axis state"
                ))
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
        } else if let Some(input_sharding) = input.sharding() {
            let mesh = input_sharding.mesh().clone();
            let replicated_input_axes: BTreeSet<usize> = dimensions
                .scatter_dimensions_to_operand_dimensions()
                .iter()
                .chain(dimensions.inserted_window_dimensions())
                .copied()
                .collect();
            for &axis in &replicated_input_axes {
                let window_extent = if inserted.contains(&axis) {
                    Dimension::Static(1)
                } else {
                    // An indexed axis that is not inserted is a window axis, because the disjointness check above
                    // excludes it from the batching axes.
                    let window = input_window_axes.iter().position(|window_axis| *window_axis == axis).unwrap();
                    updates.dimension(dimensions.update_window_dimensions()[window])
                };
                if input.dimension(axis) != window_extent
                    && input.dimension(axis) != Dimension::Static(0)
                    && input_sharding.dimensions()[axis].has_explicit_axis(&mesh)
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
                && indices_sharding.dimensions()[index_vector_dimension].has_explicit_axis(&mesh)
            {
                return Err(TypeError::invalid(format!(
                    "`{SCATTER_OPERATION_NAME}` indices index vector dimension must be replicated over explicit \
                         mesh axes"
                ))
                .into());
            }
            Some(input_sharding.clone().with_varying_manual_axes(varying_manual_axes).map_err(|error| {
                TypeError::invalid(format!("`{SCATTER_OPERATION_NAME}` output sharding is invalid: {error}"))
            })?)
        } else if let Some(mesh) = common_mesh {
            Some(
                Sharding::new(mesh, vec![ShardingDimension::Replicated; input.rank()])
                    .and_then(|sharding| sharding.with_varying_manual_axes(varying_manual_axes))
                    .map_err(|error| {
                        TypeError::invalid(format!("`{SCATTER_OPERATION_NAME}` output sharding is invalid: {error}"))
                    })?,
            )
        } else {
            None
        };
        input.clone().with_sharding(sharding).map_err(|error| {
            TypeError::invalid(format!("`{SCATTER_OPERATION_NAME}` output type is invalid: {error}")).into()
        })
    }
}

impl Array {
    /// Applies one already-validated scatter using a byte-slice combiner, keeping index traversal independent of the
    /// selected element arithmetic. The combiner receives one mutable input encoding and one update encoding.
    fn scatter_with_combiner(
        &self,
        indices: &Self,
        updates: &Self,
        output_type: ArrayType,
        dimensions: &ScatterDimensionNumbers,
        mode: ScatterMode,
        combine: impl Fn(&mut [u8], &[u8]) -> Result<(), ProgramError>,
    ) -> Result<Self, ProgramError> {
        let input_shape = self.r#type().static_shape().unwrap();
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        // No update can address an element of an empty input, even in clipping mode.
        if output_addressing.element_count() == 0 {
            return Ok(Self::new_unchecked(output_type, self.shared_storage().clone()));
        }
        let indices_shape = indices.r#type().static_shape().unwrap();
        let indices_addressing = ArrayAddressing::new(indices.r#type().into_owned())?;
        let updates_shape = updates.r#type().static_shape().unwrap();
        let updates_addressing = ArrayAddressing::new(updates.r#type().into_owned())?;
        // The caller has already validated all inputs and placement. With no updates, retain the input payload
        // before requesting mutable storage, which would otherwise copy the entire shared buffer.
        if updates_addressing.element_count() == 0 {
            return Ok(Self::new_unchecked(output_type, self.shared_storage().clone()));
        }
        let input_rank = input_shape.rank();
        let indices_rank = indices_shape.rank();
        let updates_rank = updates_shape.rank();
        let index_vector_dimension = indices_rank - 1;
        let indices_data_type = indices.r#type().data_type();

        let inserted: BTreeSet<usize> = dimensions.inserted_window_dimensions().iter().copied().collect();
        let batching: BTreeSet<usize> = dimensions.operand_batching_dimensions().iter().copied().collect();
        let input_window_axes: Vec<usize> =
            (0..input_rank).filter(|axis| !inserted.contains(axis) && !batching.contains(axis)).collect();
        let update_window: BTreeSet<usize> = dimensions.update_window_dimensions().iter().copied().collect();
        let update_scatter_axes: Vec<usize> = (0..updates_rank).filter(|axis| !update_window.contains(axis)).collect();
        let indices_batch_axes: Vec<usize> = (0..indices_rank).filter(|axis| *axis != index_vector_dimension).collect();
        // Window size per input axis (the update extent on window axes, 1 elsewhere), used to clamp the start so the
        // whole window stays in bounds.
        let mut input_window_size = vec![1usize; input_rank];
        for (window, &input_axis) in input_window_axes.iter().enumerate() {
            input_window_size[input_axis] = updates_shape[dimensions.update_window_dimensions()[window]];
        }

        let mut output = Self::new_unchecked(output_type, self.shared_storage().clone());
        let output_bytes = output.storage_bytes_mut();
        let mut update_index = vec![0usize; updates_rank];
        let mut indices_index = vec![0usize; indices_rank];
        let mut input_origin = vec![0usize; input_rank];
        let mut input_index = vec![0usize; input_rank];
        let mut dropped = false;
        let drop_out_of_bounds = mode == ScatterMode::Drop;
        for update in 0..updates_addressing.element_count() {
            // Consecutive window elements often use the same query. Cache only that query's origin and bounds
            // decision, retaining the original element traversal order even when query/window axes interleave.
            // The first iteration also initializes queries with an empty index vector or no query axes.
            let mut query_changed = update == 0;
            for (position, &axis) in update_scatter_axes.iter().enumerate() {
                let indices_axis = indices_batch_axes[position];
                let coordinate = update_index[axis];
                query_changed |= indices_index[indices_axis] != coordinate;
                indices_index[indices_axis] = coordinate;
            }
            if query_changed {
                input_origin.fill(0);
                dropped = false;
                for (batch, &input_axis) in dimensions.operand_batching_dimensions().iter().enumerate() {
                    input_origin[input_axis] = indices_index[dimensions.scatter_indices_batching_dimensions()[batch]];
                }
                for (component, &input_axis) in dimensions.scatter_dimensions_to_operand_dimensions().iter().enumerate()
                {
                    indices_index[index_vector_dimension] = component;
                    let index_bytes = &indices.storage_bytes()[indices_addressing.byte_range_unchecked(&indices_index)];
                    let raw = dispatch_on_array_element_type!(@integer indices_data_type, |Element| {
                        let value = Element::decode(index_bytes);
                        if indices_data_type.is_signed() {
                            value.convert_to::<i64>().map(i128::from)
                        } else {
                            value.convert_to::<u64>().map(i128::from)
                        }
                    })?;
                    // Validation guarantees the window fits. Widening before clamping preserves unsigned extremes.
                    let maximum = (input_shape[input_axis] - input_window_size[input_axis]) as i128;
                    dropped |= drop_out_of_bounds && (raw < 0 || raw > maximum);
                    // Dropped origins are never accessed. Promise mode uses defensive clipping without
                    // guaranteeing any particular out-of-bounds result to callers.
                    input_origin[input_axis] = raw.clamp(0, maximum) as usize;
                }
            }
            if !dropped {
                input_index.copy_from_slice(&input_origin);
                for (window, &input_axis) in input_window_axes.iter().enumerate() {
                    input_index[input_axis] += update_index[dimensions.update_window_dimensions()[window]];
                }
                combine(
                    &mut output_bytes[output_addressing.byte_range_unchecked(&input_index)],
                    &updates.storage_bytes()[updates_addressing.byte_range_for_flat_index(update)],
                )?;
            }
            updates_addressing.advance_index(&mut update_index);
        }
        Ok(output)
    }
}

impl Scatter for Array {
    fn scatter(
        &self,
        indices: &Self,
        updates: &Self,
        dimensions: &ScatterDimensionNumbers,
        kind: ScatterReductionKind,
        options: &ScatterOptions,
    ) -> Result<Self, ProgramError> {
        let output_type =
            self.r#type()
                .scatter(indices.r#type().as_ref(), updates.r#type().as_ref(), dimensions, kind, options)?;
        let data_type = output_type.data_type();
        if kind == ScatterReductionKind::Overwrite || data_type == DataType::Zero {
            return self.scatter_with_combiner(
                indices,
                updates,
                output_type,
                dimensions,
                options.mode(),
                |current, update| {
                    current.copy_from_slice(update);
                    Ok(())
                },
            );
        }
        match kind {
            ScatterReductionKind::Add | ScatterReductionKind::Mul => {
                dispatch_on_array_element_type!(@numeric data_type, |Element| {
                    self.scatter_with_combiner(
                        indices,
                        updates,
                        output_type,
                        dimensions,
                        options.mode(),
                        |current, update| {
                            let current_value = Element::decode(current);
                            let update_value = Element::decode(update);
                            let result = if kind == ScatterReductionKind::Add {
                                <Element as NumericArrayElement>::add(current_value, update_value)?
                            } else {
                                <Element as NumericArrayElement>::mul(current_value, update_value)?
                            };
                            result.encode(current);
                            Ok(())
                        },
                    )
                })
            }
            ScatterReductionKind::Min | ScatterReductionKind::Max => {
                dispatch_on_array_element_type!(data_type, |Element| {
                    self.scatter_with_combiner(
                        indices,
                        updates,
                        output_type,
                        dimensions,
                        options.mode(),
                        |current, update| {
                            let current_value = Element::decode(current);
                            let update_value = Element::decode(update);
                            let result = if kind == ScatterReductionKind::Min {
                                ArrayElement::min(&current_value, &update_value)
                            } else {
                                ArrayElement::max(&current_value, &update_value)
                            };
                            result.encode(current);
                            Ok(())
                        },
                    )
                })
            }
            ScatterReductionKind::Overwrite => unreachable!("overwrite scatter returns before typed dispatch"),
        }
    }
}

impl<A: Scatter + Value<Type = ArrayType>> Scatter for ArrayIrValue<A> {
    fn scatter(
        &self,
        indices: &Self,
        updates: &Self,
        dimensions: &ScatterDimensionNumbers,
        kind: ScatterReductionKind,
        options: &ScatterOptions,
    ) -> Result<Self, ProgramError> {
        let input = <Self as ValueProjection<ArrayType>>::projected(self)?;
        let indices = <Self as ValueProjection<ArrayType>>::projected(indices)?;
        let updates = <Self as ValueProjection<ArrayType>>::projected(updates)?;
        Ok(Self::Array(input.scatter(indices, updates, dimensions, kind, options)?))
    }
}

// Any context-carrying value scatters by binding a [`ScatterOperation`] through its own context. The
// `From<ScatterOperation>` bound makes this disjoint from the eager value types (whose context operation is
// `ConstantOperation`), so it covers the transform tracers without conflicting with the concrete implementations.
impl<V: Value<Type = ArrayType>> Scatter for V
where
    V::DispatchDomain: Context<Type = ArrayType, Operation: From<ScatterOperation>>,
{
    fn scatter(
        &self,
        indices: &Self,
        updates: &Self,
        dimensions: &ScatterDimensionNumbers,
        kind: ScatterReductionKind,
        options: &ScatterOptions,
    ) -> Result<Self, ProgramError> {
        let mut outputs = self.dispatch_domain().bind(
            ScatterOperation::new(dimensions.clone(), kind).with_options(options.clone()),
            Vec::new(),
            &[self.clone(), indices.clone(), updates.clone()],
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
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
/// # use ryft_core::{Array, ArrayIrValue, DynamicScatter, ScatterMode, ScatterReductionKind};
/// // Shapes: input [3], indices [2], updates [2] -> output [3].
/// let input = ArrayIrValue::Array(Array::vector(vec![10_i32, 20, 30]).unwrap());
/// let indices = ArrayIrValue::Array(Array::vector(vec![1_i32, 1]).unwrap());
/// let updates = ArrayIrValue::Array(Array::vector(vec![2_i32, 3]).unwrap());
/// let output = input.dynamic_scatter_axis(
///     &indices, &updates, 0, ScatterReductionKind::Add, ScatterMode::Clip,
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
    ///   - `mode`: Out-of-bounds handling; see [`ScatterMode`].
    fn dynamic_scatter_axis<A: Into<Axis>>(
        &self,
        indices: &Self,
        updates: &Self,
        axis: A,
        kind: ScatterReductionKind,
        mode: ScatterMode,
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
        mode: ScatterMode,
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
        let expected_shape = Shape::new(expected_dimensions);
        if updates_type.shape() != &expected_shape {
            return Err(TypeError::invalid(format!(
                "`dynamic_scatter_axis` updates shape must be `{expected_shape}` but got `{}`",
                updates_type.shape(),
            ))
            .into());
        }
        // Only the index-vector axis is new. Reading the other extents from the query supplies the dimension
        // definitions needed to specialize a retained `[queries] -> [queries, 1]` reshape.
        let indices = indices.dynamic_expand_dimensions(-1)?;
        let window_dimensions = (0..input_type.rank())
            .filter(|input_axis| *input_axis != axis)
            .map(|input_axis| if input_axis < axis { input_axis } else { input_axis + indices_type.rank() - 1 })
            .collect();
        Ok(V::from_projected(self.clone().into_projected()?.scatter(
            &indices.into_projected()?,
            &updates.clone().into_projected()?,
            &ScatterDimensionNumbers::new(window_dimensions, vec![axis], vec![axis]),
            kind,
            &ScatterOptions::new().with_mode(mode),
        )?))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use indoc::indoc;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::batching::DynamicArrayExtentBatchingPolicy;
    use crate::arrays::{
        Array, ArrayIrOperation, ArrayOperation, ArrayReferenceDischarge, DataType, Dimension, DimensionBounds,
        DimensionType, DimensionValue, DimensionVariable, Layout, LogicalMesh, Memory, MeshAxis, MeshAxisType,
        RaggedAxis, Shape, Sharding, ShardingDimension, StridedLayout, i4,
    };
    use crate::batching::batch;
    use crate::contexts::Context;
    use crate::differentiation::{Linearization, TransposableOperation, TranspositionContext, differentiate_at};
    use crate::macros::{
        check_gradient, check_operation_batching, check_operation_partial_evaluation, check_operation_transposition,
        check_operation_type_inference,
    };
    use crate::operations::constants::one::OneOperation;
    use crate::operations::manipulation::reshaping::DynamicReshapeOperation;
    use crate::operations::math::reduce::{Reduce, ReductionKind};
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::{
        EffectClasses, EmptyRegionDriver, Program, ProgramBuilder, ReferenceDischargeContext, ReferenceDischargeValue,
        ReferenceDischargeableOperation,
    };
    use crate::tracing::Trace;

    use super::*;

    /// Tracer of the mixed array IR tracing context used by the batching and differentiation edge cases.
    type IrTracer = Tracer<TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>;

    /// Mixed program over an input of `input_type` and updates of `updates_type` that scatters the updates into the
    /// input at the constant `indices` with `operation`.
    type ConstantIndexScatterProgram =
        Program<ArrayIrValue<Array>, ArrayIrOperation<Array>, Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>;

    /// Lifts a constant integer index array into the trace or differentiation context that `exemplar` belongs to.
    fn index_array<V>(exemplar: &V, shape: Vec<usize>, values: Vec<i32>) -> V
    where
        V: Value<Type = ArrayType>,
        V::DispatchDomain: Context<Constant = Array>,
    {
        let r#type = ArrayType::new_static(DataType::I32, shape);
        exemplar.dispatch_domain().lift(Array::from_elements::<i32>(r#type, &values).unwrap()).unwrap()
    }

    /// Builds the mixed program with the input and the updates as its two inputs that scatters the updates into the
    /// input at the constant `indices` with `operation`.
    fn constant_index_scatter_program(
        operation: ScatterOperation,
        input_type: ArrayType,
        indices: Array,
        updates_type: ArrayType,
    ) -> ConstantIndexScatterProgram {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let updates = builder.add_input(updates_type.into());
        let indices = builder.add_constant(ArrayIrValue::Array(indices));
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Scatter(operation)),
                Vec::new(),
                vec![input, indices, updates],
                None,
            )
            .unwrap()[0];
        builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap()
    }

    /// Mixed program with the dynamic `items` extent, a packed `f64` input, packed `i32` indices, and packed `f64`
    /// updates as inputs, staging `operation` (a scatter of one `f64[1]` update into an `f64[3]` input at `i32[1, 1]`
    /// indices) with all three arrays jointly mapped over `items` at `mapped_axis` through the dynamic-extent batching
    /// policy. The output carries the mapped extent at its leading axis.
    fn jointly_mapped_dynamic_scatter_program(
        items: DimensionVariable,
        mapped_axis: usize,
        operation: ScatterOperation,
    ) -> ConstantIndexScatterProgram {
        let with_items = |data_type: DataType, extents: &[usize]| {
            let mut dimensions = extents.iter().map(|extent| Dimension::Static(*extent)).collect::<Vec<_>>();
            dimensions.insert(mapped_axis, Dimension::Dynamic(items.clone()));
            ArrayIrType::Array(ArrayType::new(data_type, Shape::new(dimensions)))
        };
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let extent = trace.input(ArrayIrType::Dimension(DimensionType::new(items.clone())));
        let input = trace.input(with_items(DataType::F64, &[3]));
        let indices = trace.input(with_items(DataType::I32, &[1, 1]));
        let updates = trace.input(with_items(DataType::F64, &[1]));
        let context = BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
            ProjectedContext::new(trace.clone()),
            extent,
        );
        let input = ValueProjection::<ArrayType>::into_projected(input).unwrap();
        let indices = ValueProjection::<ArrayType>::into_projected(indices).unwrap();
        let updates = ValueProjection::<ArrayType>::into_projected(updates).unwrap();
        let (outputs, _) = operation
            .batch(
                &context,
                &EmptyRegionDriver,
                &[
                    ArrayBatch::new(input, BatchAxis::new(mapped_axis)).unwrap(),
                    ArrayBatch::new(indices, BatchAxis::new(mapped_axis)).unwrap(),
                    ArrayBatch::new(updates, BatchAxis::new(mapped_axis)).unwrap(),
                ],
            )
            .unwrap()
            .into_parts();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        let output = <IrTracer as ValueProjection<ArrayType>>::from_projected(outputs[0].value().clone());
        trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder; 4],
                vec![Placeholder],
            )
            .unwrap()
    }

    #[test]
    fn test_scatter_mode() {
        assert_eq!(ScatterMode::default(), ScatterMode::PromiseInBounds);
        for (mode, name, debug) in [
            (ScatterMode::PromiseInBounds, "promise_in_bounds", "PromiseInBounds"),
            (ScatterMode::Clip, "clip", "Clip"),
            (ScatterMode::Drop, "drop", "Drop"),
        ] {
            assert_eq!(mode.name(), name);
            assert_eq!(mode.to_string(), name);
            assert_eq!(format!("{mode:?}"), debug);
        }
    }

    #[test]
    fn test_scatter_reduction_kind() {
        for (kind, name, debug) in [
            (ScatterReductionKind::Overwrite, "overwrite", "Overwrite"),
            (ScatterReductionKind::Add, "add", "Add"),
            (ScatterReductionKind::Mul, "mul", "Mul"),
            (ScatterReductionKind::Min, "min", "Min"),
            (ScatterReductionKind::Max, "max", "Max"),
        ] {
            assert_eq!(kind.name(), name);
            assert_eq!(kind.to_string(), name);
            assert_eq!(format!("{kind:?}"), debug);
        }
    }

    #[test]
    fn test_scatter_dimension_numbers_new() {
        let dimensions = ScatterDimensionNumbers::new(vec![1], vec![0], vec![0]);
        assert_eq!(dimensions.update_window_dimensions(), &[1]);
        assert_eq!(dimensions.inserted_window_dimensions(), &[0]);
        assert_eq!(dimensions.scatter_dimensions_to_operand_dimensions(), &[0]);
        assert_eq!(dimensions.operand_batching_dimensions(), &[] as &[usize]);
        assert_eq!(dimensions.scatter_indices_batching_dimensions(), &[] as &[usize]);
        assert_eq!(
            format!("{dimensions:?}"),
            "ScatterDimensionNumbers { update_window_dimensions: [1], inserted_window_dimensions: [0], \
             scatter_dimensions_to_operand_dimensions: [0], operand_batching_dimensions: [], \
             scatter_indices_batching_dimensions: [] }"
        );
        let lookup = HashMap::from([(dimensions.clone(), 7)]);
        assert_eq!(lookup.get(&dimensions), Some(&7));
        assert_eq!(lookup.get(&ScatterDimensionNumbers::default()), None);
    }

    #[test]
    fn test_scatter_dimension_numbers_with_batching_dimensions() {
        let dimensions =
            ScatterDimensionNumbers::new(vec![2], vec![1], vec![1]).with_batching_dimensions(vec![0], vec![1]);
        assert_eq!(dimensions.update_window_dimensions(), &[2]);
        assert_eq!(dimensions.inserted_window_dimensions(), &[1]);
        assert_eq!(dimensions.scatter_dimensions_to_operand_dimensions(), &[1]);
        assert_eq!(dimensions.operand_batching_dimensions(), &[0]);
        assert_eq!(dimensions.scatter_indices_batching_dimensions(), &[1]);
    }

    #[test]
    fn test_scatter_options_new() {
        let options = ScatterOptions::new();
        assert_eq!(options, ScatterOptions::default());
        assert_eq!(options.mode(), ScatterMode::PromiseInBounds);
        assert!(!options.indices_are_sorted());
        assert!(!options.unique_indices());
        assert_eq!(options.output_sharding(), None);
        let lookup = HashMap::from([(options.clone(), 7)]);
        assert_eq!(lookup.get(&options), Some(&7));
        assert_eq!(lookup.get(&options.with_mode(ScatterMode::Clip)), None);
    }

    #[test]
    fn test_scatter_options_with_mode() {
        let options = ScatterOptions::new().with_mode(ScatterMode::Drop);
        assert_eq!(options.mode(), ScatterMode::Drop);
        assert_eq!(options.with_mode(ScatterMode::PromiseInBounds), ScatterOptions::new());
    }

    #[test]
    fn test_scatter_options_with_indices_are_sorted() {
        let options = ScatterOptions::new().with_indices_are_sorted(true);
        assert!(options.indices_are_sorted());
        assert_eq!(options.with_indices_are_sorted(false), ScatterOptions::new());
    }

    #[test]
    fn test_scatter_options_with_unique_indices() {
        let options = ScatterOptions::new().with_unique_indices(true);
        assert!(options.unique_indices());
        assert_eq!(options.with_unique_indices(false), ScatterOptions::new());
    }

    #[test]
    fn test_scatter_options_with_output_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 2);
        let options = ScatterOptions::new().with_output_sharding(sharding.clone());
        assert_eq!(options.output_sharding(), Some(&sharding));
        assert_eq!(options.with_output_sharding(None), ScatterOptions::new());
    }

    #[test]
    fn test_scatter() {
        // Scatter-add row updates into a [3, 2] input indexed by a [2, 1] index array: update window axis 1 carries
        // the row, input axis 0 is inserted (start-index driven).
        let dimensions = ScatterDimensionNumbers::new(vec![1], vec![0], vec![0]);
        let operation = ScatterOperation::new(dimensions, ScatterReductionKind::Add);
        assert_eq!(operation.name(), SCATTER_OPERATION_NAME);
        assert_eq!(operation.kind(), ScatterReductionKind::Add);
        assert_eq!(operation.dimensions(), &ScatterDimensionNumbers::new(vec![1], vec![0], vec![0]));
        assert_eq!(operation.mode(), ScatterMode::PromiseInBounds);
        assert!(!operation.indices_are_sorted());
        assert!(!operation.unique_indices());
        assert_eq!(operation.output_sharding(), None);
        assert_eq!(operation.effects().classes(), EffectClasses::NONE);
        assert_eq!(
            format!("{operation}"),
            indoc! {"
                scatter [
                    kind=add,
                    dimensions=(update_window=[1], inserted_window=[0], scatter_to_operand=[0], operand_batching=[], \
                        scatter_indices_batching=[]),
                ]
            "}
            .trim_end(),
        );

        // Every builder sets exactly its own field, and only non-default fields render.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let configured = operation
            .clone()
            .with_mode(ScatterMode::Clip)
            .with_indices_are_sorted(true)
            .with_unique_indices(true)
            .with_output_sharding(Sharding::replicated(mesh.clone(), 2));
        assert_eq!(configured.kind(), ScatterReductionKind::Add);
        assert_eq!(configured.dimensions(), operation.dimensions());
        assert_eq!(configured.mode(), ScatterMode::Clip);
        assert!(configured.indices_are_sorted());
        assert!(configured.unique_indices());
        assert_eq!(configured.output_sharding(), Some(&Sharding::replicated(mesh, 2)));
        assert_eq!(
            format!("{configured}"),
            indoc! {"
                scatter [
                    kind=add,
                    dimensions=(update_window=[1], inserted_window=[0], scatter_to_operand=[0], operand_batching=[], \
                        scatter_indices_batching=[]),
                    mode=clip,
                    indices_are_sorted=true,
                    unique_indices=true,
                    output_sharding={mesh<['x'=2:explicit]>, [{}, {}]},
                ]
            "}
            .trim_end(),
        );
        assert_eq!(configured.clone().with_output_sharding(None).output_sharding(), None);

        // Equality and hashing follow the complete payload, including fields that do not affect the result.
        assert_eq!(operation, operation.clone());
        assert_ne!(operation, configured);
        assert_ne!(operation, operation.clone().with_indices_are_sorted(true));
        let lookup = HashMap::from([(operation.clone(), 7)]);
        assert_eq!(lookup.get(&operation), Some(&7));
        assert_eq!(lookup.get(&configured), None);
    }

    #[test]
    fn test_scatter_with_options() {
        let dimensions = ScatterDimensionNumbers::new(vec![], vec![0], vec![0]);
        let operation = ScatterOperation::new(dimensions.clone(), ScatterReductionKind::Add);
        assert_eq!(operation.options(), &ScatterOptions::new());
        let options = ScatterOptions::new()
            .with_mode(ScatterMode::Drop)
            .with_indices_are_sorted(true)
            .with_unique_indices(true);
        let operation = operation.with_options(options.clone());
        assert_eq!(operation.options(), &options);
        assert_eq!(operation.dimensions(), &dimensions);
        assert_eq!(operation.kind(), ScatterReductionKind::Add);
        assert_eq!(operation.mode(), ScatterMode::Drop);
        assert!(operation.indices_are_sorted());
        assert!(operation.unique_indices());
        assert_eq!(
            operation.with_options(ScatterOptions::new()),
            ScatterOperation::new(dimensions, ScatterReductionKind::Add),
        );
    }

    #[test]
    fn test_scatter_is_linear() {
        // Linearity holds the indices fixed: scatter-add is linear for every index configuration, overwrite only when
        // its windows are promised disjoint, and the remaining combiners never are. The sortedness hint is irrelevant.
        let dimensions = ScatterDimensionNumbers::new(vec![], vec![0], vec![0]);
        let add = ScatterOperation::new(dimensions.clone(), ScatterReductionKind::Add);
        assert!(add.is_linear());
        assert!(add.clone().with_unique_indices(true).is_linear());
        assert!(add.with_indices_are_sorted(true).is_linear());
        let overwrite = ScatterOperation::new(dimensions.clone(), ScatterReductionKind::Overwrite);
        assert!(!overwrite.is_linear());
        assert!(!overwrite.clone().with_indices_are_sorted(true).is_linear());
        assert!(overwrite.with_unique_indices(true).is_linear());
        let multiply = ScatterOperation::new(dimensions.clone(), ScatterReductionKind::Mul);
        assert!(!multiply.is_linear());
        assert!(!multiply.with_unique_indices(true).is_linear());
        let minimum = ScatterOperation::new(dimensions.clone(), ScatterReductionKind::Min);
        assert!(!minimum.is_linear());
        assert!(!minimum.with_unique_indices(true).is_linear());
        let maximum = ScatterOperation::new(dimensions, ScatterReductionKind::Max);
        assert!(!maximum.is_linear());
        assert!(!maximum.with_unique_indices(true).with_indices_are_sorted(true).is_linear());
    }

    #[test]
    fn test_scatter_type_inference() {
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![1], vec![0], vec![0]), ScatterReductionKind::Add);
        let input = ArrayType::new_static(DataType::F32, [3, 2]);
        let indices = ArrayType::new_static(DataType::I32, [2, 1]);
        let updates = ArrayType::new_static(DataType::F32, [2, 2]);
        let boolean_input = ArrayType::new(DataType::Boolean, input.shape().clone());
        let boolean_updates = ArrayType::new(DataType::Boolean, updates.shape().clone());
        let zero_input = ArrayType::new(DataType::Zero, input.shape().clone());
        let zero_updates = ArrayType::new(DataType::Zero, updates.shape().clone());
        let host_input = input.clone().with_memory(Memory::Host { pinned: true });
        let host_indices = indices.clone().with_memory(Memory::Host { pinned: true });
        let host_updates = updates.clone().with_memory(Memory::Host { pinned: true });
        let vector = DimensionVariable::new("vector", DimensionBounds::new(1, Some(2)).unwrap());
        let dynamic_vector_indices =
            ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(vector)]));
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [
                {
                    input_types = [input.clone(), indices.clone(), updates.clone()],
                    output_types = [input.clone()],
                },
                {
                    input_types = [zero_input.clone(), indices.clone(), zero_updates],
                    output_types = [zero_input],
                },
                {
                    input_types = [input.clone(), indices.clone()],
                    error = "expected 3 inputs but got 2",
                },
                {
                    input_types = [input.clone(), indices.clone(), ArrayType::new_static(DataType::I32, [2, 2])],
                    error = format!(
                        "`{SCATTER_OPERATION_NAME}` updates data type `i32` does not match input data type `f32`",
                    ),
                },
                {
                    input_types = [input.clone(), ArrayType::new_static(DataType::F32, [2, 1]), updates.clone()],
                    error = format!(
                        "`{SCATTER_OPERATION_NAME}` indices must be integer-typed but have type `f32[2, 1]`",
                    ),
                },
                {
                    input_types = [input.clone(), ArrayType::scalar(DataType::I32), updates.clone()],
                    error = format!(
                        "`{SCATTER_OPERATION_NAME}` indices must have rank at least 1 (the trailing index vector)",
                    ),
                },
                {
                    input_types = [input.clone(), dynamic_vector_indices, updates.clone()],
                    error = format!(
                        "`{SCATTER_OPERATION_NAME}` indices index vector dimension must have a static extent",
                    ),
                },
                {
                    input_types = [boolean_input.clone(), indices.clone(), boolean_updates.clone()],
                    error = format!(
                        "`{SCATTER_OPERATION_NAME}` kind `add` requires numeric input and update elements but got \
                         `bool`",
                    ),
                },
                {
                    input_types = [host_input.clone(), host_indices, host_updates],
                    output_types = [host_input.clone()],
                },
                {
                    input_types = [host_input, indices.clone(), updates.clone()],
                    error = format!(
                        "`{SCATTER_OPERATION_NAME}` input, indices, and updates must share one memory space but \
                         reside in `Host[Pinned]`, `Device`, and `Device`",
                    ),
                },
            ],
        );

        // Extrema accept Boolean and complex elements but not tokens; the element check follows the data-type match.
        let complex_input = ArrayType::new(DataType::C64, input.shape().clone());
        let complex_updates = ArrayType::new(DataType::C64, updates.shape().clone());
        let token_input = ArrayType::new(DataType::Token, input.shape().clone());
        let token_updates = ArrayType::new(DataType::Token, updates.shape().clone());
        check_operation_type_inference!(
            operation = ScatterOperation::new(operation.dimensions().clone(), ScatterReductionKind::Max),
            cases = [
                {
                    input_types = [complex_input.clone(), indices.clone(), complex_updates],
                    output_types = [complex_input],
                },
                {
                    input_types = [boolean_input.clone(), indices.clone(), boolean_updates],
                    output_types = [boolean_input],
                },
                {
                    input_types = [token_input, indices, token_updates],
                    error = format!(
                        "`{SCATTER_OPERATION_NAME}` kind `max` requires Boolean or numeric input and update elements \
                         but got `token`",
                    ),
                },
            ],
        );
    }

    #[test]
    fn test_scatter_type_inference_paired_extents() {
        // Specialization can retain an exact nominal dimension on one side of a paired batch while the other is
        // already static. Both descriptions prove the same extent without equating unrelated symbolic dimensions.
        let exact = Dimension::Dynamic(DimensionVariable::new("batch", DimensionBounds::new(0, Some(1)).unwrap()));
        let operation = ScatterOperation::new(
            ScatterDimensionNumbers::new(vec![], vec![1], vec![1]).with_batching_dimensions(vec![0], vec![0]),
            ScatterReductionKind::Add,
        );
        let static_input = ArrayType::new_static(DataType::F64, [0, 4]);
        let dynamic_input = ArrayType::new(DataType::F64, Shape::new(vec![exact.clone(), 4.into()]));
        let static_indices = ArrayType::new_static(DataType::I32, [0, 2, 1]);
        let dynamic_indices = ArrayType::new(DataType::I32, Shape::new(vec![exact.clone(), 2.into(), 1.into()]));
        let static_updates = ArrayType::new_static(DataType::F64, [0, 2]);
        let dynamic_updates = ArrayType::new(DataType::F64, Shape::new(vec![exact, 2.into()]));
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [
                {
                    input_types = [static_input.clone(), static_indices.clone(), static_updates.clone()],
                    output_types = [static_input.clone()],
                },
                {
                    input_types = [static_input.clone(), static_indices.clone(), dynamic_updates.clone()],
                    output_types = [static_input.clone()],
                },
                {
                    input_types = [static_input.clone(), dynamic_indices.clone(), static_updates.clone()],
                    output_types = [static_input.clone()],
                },
                {
                    input_types = [static_input.clone(), dynamic_indices.clone(), dynamic_updates.clone()],
                    output_types = [static_input.clone()],
                },
                {
                    input_types = [dynamic_input.clone(), static_indices.clone(), static_updates.clone()],
                    output_types = [dynamic_input.clone()],
                },
                {
                    input_types = [dynamic_input.clone(), static_indices.clone(), dynamic_updates.clone()],
                    output_types = [dynamic_input.clone()],
                },
                {
                    input_types = [dynamic_input.clone(), dynamic_indices.clone(), static_updates.clone()],
                    output_types = [dynamic_input.clone()],
                },
                {
                    input_types = [dynamic_input.clone(), dynamic_indices.clone(), dynamic_updates.clone()],
                    output_types = [dynamic_input.clone()],
                },
            ],
        );

        // Identically named dimensions with equal non-exact bounds remain independent and must not pass either the
        // input/indices batching pairing or the indices/updates extent equality checks.
        let first = Dimension::Dynamic(DimensionVariable::new("batch", DimensionBounds::new(0, Some(3)).unwrap()));
        let second = Dimension::Dynamic(DimensionVariable::new("batch", DimensionBounds::new(0, Some(3)).unwrap()));
        let input = ArrayType::new(DataType::F64, Shape::new(vec![first.clone(), Dimension::Static(4)]));
        let first_indices =
            ArrayType::new(DataType::I32, Shape::new(vec![first, Dimension::Static(2), Dimension::Static(1)]));
        let second_indices =
            ArrayType::new(DataType::I32, Shape::new(vec![second.clone(), Dimension::Static(2), Dimension::Static(1)]));
        let second_updates = ArrayType::new(DataType::F64, Shape::new(vec![second, Dimension::Static(2)]));
        assert_eq!(
            input.scatter(
                &second_indices,
                &second_updates,
                operation.dimensions(),
                operation.kind(),
                operation.options()
            ),
            Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` batching dimensions must have equal extents, but input axis 0 and indices \
                 axis 0 differ",
            ))
            .into()),
        );
        assert_eq!(
            input.scatter(
                &first_indices,
                &second_updates,
                operation.dimensions(),
                operation.kind(),
                operation.options()
            ),
            Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` updates scatter axis 0 must match indices batch axis 0 in extent",
            ))
            .into()),
        );
    }

    #[test]
    fn test_scatter_type_inference_invalid_dimension_maps() {
        let input = ArrayType::new_static(DataType::F32, [3, 2]);
        let indices = ArrayType::new_static(DataType::I32, [2, 1]);
        let updates = ArrayType::new_static(DataType::F32, [2, 2]);

        // Each dimension-number list is validated against its own rank bound: update window dimensions against the
        // updates rank, input axis lists against the input rank, and indices axis lists against the indices rank.
        check_operation_type_inference!(
            operation = ScatterOperation::new(
                ScatterDimensionNumbers::new(vec![1, 0], vec![], vec![0]),
                ScatterReductionKind::Add,
            ),
            cases = [{
                input_types = [input.clone(), indices.clone(), updates.clone()],
                error = format!(
                    "`{SCATTER_OPERATION_NAME}` `update_window_dimensions` must be sorted and unique but got [1, 0]",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = ScatterOperation::new(
                ScatterDimensionNumbers::new(vec![2], vec![0], vec![0]),
                ScatterReductionKind::Add,
            ),
            cases = [{
                input_types = [input.clone(), indices.clone(), updates.clone()],
                error = format!(
                    "`{SCATTER_OPERATION_NAME}` `update_window_dimensions` entry 2 is out of range for bound 2",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = ScatterOperation::new(
                ScatterDimensionNumbers::new(vec![1], vec![2], vec![0]),
                ScatterReductionKind::Add,
            ),
            cases = [{
                input_types = [input.clone(), indices.clone(), updates.clone()],
                error = format!(
                    "`{SCATTER_OPERATION_NAME}` `inserted_window_dimensions` entry 2 is out of range for bound 2",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = ScatterOperation::new(
                ScatterDimensionNumbers::new(vec![1], vec![0], vec![0]).with_batching_dimensions(vec![2], vec![0]),
                ScatterReductionKind::Add,
            ),
            cases = [{
                input_types = [input.clone(), indices.clone(), updates.clone()],
                error = format!(
                    "`{SCATTER_OPERATION_NAME}` `operand_batching_dimensions` entry 2 is out of range for bound 2",
                ),
            }],
        );

        // The scatter-to-operand map has one entry per index vector component, each naming a distinct input axis.
        check_operation_type_inference!(
            operation = ScatterOperation::new(
                ScatterDimensionNumbers::new(vec![1], vec![0], vec![0, 1]),
                ScatterReductionKind::Add,
            ),
            cases = [{
                input_types = [input.clone(), indices.clone(), updates.clone()],
                error = format!(
                    "`{SCATTER_OPERATION_NAME}` `scatter_dimensions_to_operand_dimensions` has length 2 but the \
                     index vector extent is 1",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = ScatterOperation::new(
                ScatterDimensionNumbers::new(vec![1], vec![0], vec![2]),
                ScatterReductionKind::Add,
            ),
            cases = [{
                input_types = [input.clone(), indices.clone(), updates.clone()],
                error = format!(
                    "`{SCATTER_OPERATION_NAME}` `scatter_dimensions_to_operand_dimensions` entry 2 is out of range \
                     for bound 2",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = ScatterOperation::new(
                ScatterDimensionNumbers::new(vec![1], vec![0], vec![0, 0]),
                ScatterReductionKind::Add,
            ),
            cases = [{
                input_types = [input.clone(), ArrayType::new_static(DataType::I32, [2, 2]), updates.clone()],
                error = format!(
                    "`{SCATTER_OPERATION_NAME}` `scatter_dimensions_to_operand_dimensions` must be unique but got \
                     [0, 0]",
                ),
            }],
        );

        // Batching axes pair 1:1, name distinct in-range indices axes other than the index vector, and are disjoint
        // from both the scatter-to-operand map and the inserted axes.
        check_operation_type_inference!(
            operation = ScatterOperation::new(
                ScatterDimensionNumbers::new(vec![1], vec![0], vec![0]).with_batching_dimensions(vec![1], vec![]),
                ScatterReductionKind::Add,
            ),
            cases = [{
                input_types = [input.clone(), indices.clone(), updates.clone()],
                error = format!(
                    "`{SCATTER_OPERATION_NAME}` input and scatter-indices batching dimensions must align 1:1, but got \
                     1 and 0",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = ScatterOperation::new(
                ScatterDimensionNumbers::new(vec![1], vec![0], vec![0]).with_batching_dimensions(vec![1], vec![2]),
                ScatterReductionKind::Add,
            ),
            cases = [{
                input_types = [input.clone(), indices.clone(), updates.clone()],
                error = format!(
                    "`{SCATTER_OPERATION_NAME}` `scatter_indices_batching_dimensions` entry 2 is out of range for \
                     bound 2",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = ScatterOperation::new(
                ScatterDimensionNumbers::new(vec![], vec![2], vec![2]).with_batching_dimensions(vec![0, 1], vec![0, 0]),
                ScatterReductionKind::Add,
            ),
            cases = [{
                input_types = [
                    ArrayType::new_static(DataType::F32, [2, 2, 4]),
                    indices.clone(),
                    ArrayType::new_static(DataType::F32, [2]),
                ],
                error = format!(
                    "`{SCATTER_OPERATION_NAME}` `scatter_indices_batching_dimensions` must be unique but got [0, 0]",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = ScatterOperation::new(
                ScatterDimensionNumbers::new(vec![1], vec![], vec![0]).with_batching_dimensions(vec![1], vec![1]),
                ScatterReductionKind::Add,
            ),
            cases = [{
                input_types = [input.clone(), indices.clone(), updates.clone()],
                error = format!(
                    "`{SCATTER_OPERATION_NAME}` `scatter_indices_batching_dimensions` cannot name the index vector \
                     dimension 1",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = ScatterOperation::new(
                ScatterDimensionNumbers::new(vec![], vec![0], vec![1]).with_batching_dimensions(vec![0], vec![0]),
                ScatterReductionKind::Add,
            ),
            cases = [{
                input_types = [input.clone(), indices.clone(), updates.clone()],
                error = format!(
                    "`{SCATTER_OPERATION_NAME}` `inserted_window_dimensions` and `operand_batching_dimensions` must \
                     be disjoint",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = ScatterOperation::new(
                ScatterDimensionNumbers::new(vec![], vec![1], vec![0]).with_batching_dimensions(vec![0], vec![0]),
                ScatterReductionKind::Add,
            ),
            cases = [{
                input_types = [
                    ArrayType::new_static(DataType::F32, [2, 4]),
                    indices.clone(),
                    ArrayType::new_static(DataType::F32, [2]),
                ],
                error = format!(
                    "`{SCATTER_OPERATION_NAME}` indexed input axes and batching input axes must be disjoint",
                ),
            }],
        );

        // The input axes decompose exactly into window, inserted, and batching axes, and the updates axes into the
        // window axes plus one axis per indices batch axis.
        check_operation_type_inference!(
            operation = ScatterOperation::new(
                ScatterDimensionNumbers::new(vec![1], vec![], vec![0]),
                ScatterReductionKind::Add,
            ),
            cases = [{
                input_types = [input.clone(), indices.clone(), updates.clone()],
                error = format!(
                    "`{SCATTER_OPERATION_NAME}` input rank 2 must equal update_window + inserted_window + \
                     operand_batching dimension counts",
                ),
            }],
        );
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![1], vec![0], vec![0]), ScatterReductionKind::Add);
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [
                {
                    input_types = [input.clone(), indices.clone(), ArrayType::new_static(DataType::F32, [2, 2, 1])],
                    error = format!(
                        "`{SCATTER_OPERATION_NAME}` updates rank 3 must equal (indices rank - 1) + the update window \
                         dimension count",
                    ),
                },
                // A rank-1 updates array bounds the update window list before the rank decomposition is reached.
                {
                    input_types = [input.clone(), indices.clone(), ArrayType::new_static(DataType::F32, [2])],
                    error = format!(
                        "`{SCATTER_OPERATION_NAME}` `update_window_dimensions` entry 1 is out of range for bound 1",
                    ),
                },
                // Each update window fits its input window axis, and the updates' scatter axes match the indices'
                // batch axes.
                {
                    input_types = [input.clone(), indices.clone(), ArrayType::new_static(DataType::F32, [2, 3])],
                    error = format!(
                        "`{SCATTER_OPERATION_NAME}` update window axis 1 extent 3 exceeds the input window axis 1 \
                         extent 2",
                    ),
                },
                {
                    input_types = [input.clone(), indices.clone(), ArrayType::new_static(DataType::F32, [3, 2])],
                    error = format!(
                        "`{SCATTER_OPERATION_NAME}` updates scatter axis 0 must match indices batch axis 0 in extent",
                    ),
                },
            ],
        );

        // `scatter` carries no regions.
        assert_eq!(
            operation.infer_output_types(
                &[input, indices, updates],
                &[RegionInterface::new(vec![], vec![], EffectClasses::NONE)],
            ),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_scatter_type_inference_window_sharding() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();

        // Input [4, 2] sharded only on the feature axis (axis 1); the targeted axis 0 is replicated, so the output
        // keeps the input sharding.
        let sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["y"])])
                .unwrap();
        let input = ArrayType::new_static(DataType::F32, [4, 2]).with_sharding(sharding.clone()).unwrap();
        let indices = ArrayType::new_static(DataType::I32, [2, 1]);
        let updates = ArrayType::new_static(DataType::F32, [2, 2]);
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![1], vec![0], vec![0]), ScatterReductionKind::Add);
        assert_eq!(
            operation.infer_output_types(&[input.clone(), indices.clone(), updates.clone()], &[]),
            Ok(vec![input])
        );

        // Sharding the targeted input axis over an explicit mesh axis is ambiguous without an output sharding.
        let input = ArrayType::new_static(DataType::F32, [4, 2])
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap(),
            )
            .unwrap();
        assert_eq!(
            operation.infer_output_types(&[input, indices, updates], &[]),
            Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` input axis 0 is targeted by the start indices and must be replicated over \
                 explicit mesh axes; request an explicit output sharding to resolve placement",
            ))),
        );
    }

    #[test]
    fn test_scatter_type_inference_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let input = ArrayType::new_static(DataType::F32, [4]).with_sharding(replicated.clone()).unwrap();
        let indices = ArrayType::new_static(DataType::I32, [1, 1]);
        let updates = ArrayType::new_static(DataType::F32, [1]);
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Add);

        // The result follows the input placement. Unsharded inputs take a replicated placement on the common mesh,
        // and manual-axis variation contributed by the indices or updates joins the result.
        assert_eq!(
            input.scatter(&indices, &updates, operation.dimensions(), operation.kind(), operation.options()),
            Ok(input.clone())
        );
        let replicated_updates = updates.clone().with_sharding(replicated.clone()).unwrap();
        assert_eq!(
            ArrayType::new_static(DataType::F32, [4]).scatter(
                &indices,
                &replicated_updates,
                operation.dimensions(),
                operation.kind(),
                operation.options()
            ),
            Ok(input.clone()),
        );
        let manual_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let varying = Sharding::replicated(manual_mesh.clone(), 1).with_varying_manual_axes(["m"]).unwrap();
        let varying_updates = updates.clone().with_sharding(varying.clone()).unwrap();
        let expected = ArrayType::new_static(DataType::F32, [4]).with_sharding(varying).unwrap();
        assert_eq!(
            ArrayType::new_static(DataType::F32, [4]).scatter(
                &indices,
                &varying_updates,
                operation.dimensions(),
                operation.kind(),
                operation.options()
            ),
            Ok(expected.clone()),
        );
        let manual_input = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(manual_mesh.clone(), 1))
            .unwrap();
        assert_eq!(
            manual_input.scatter(
                &indices,
                &varying_updates,
                operation.dimensions(),
                operation.kind(),
                operation.options()
            ),
            Ok(expected)
        );

        // All sharded inputs must use one mesh, whichever input establishes it.
        let other_mesh_indices = indices.clone().with_sharding(Sharding::replicated(other_mesh.clone(), 2)).unwrap();
        assert_eq!(
            input.scatter(&other_mesh_indices, &updates, operation.dimensions(), operation.kind(), operation.options()),
            Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` input, indices, and updates shardings must use one mesh"
            ))
            .into()),
        );
        check_operation_type_inference!(
            operation = operation.clone().with_output_sharding(replicated.clone()),
            cases = [{
                input_types = [input.clone(), other_mesh_indices.clone(), updates.clone()],
                error = format!("`{SCATTER_OPERATION_NAME}` input, indices, and updates shardings must use one mesh"),
            }],
        );
        let mesh_indices = indices.clone().with_sharding(Sharding::replicated(mesh.clone(), 2)).unwrap();
        let other_mesh_updates = updates.clone().with_sharding(Sharding::replicated(other_mesh.clone(), 1)).unwrap();
        assert_eq!(
            ArrayType::new_static(DataType::F32, [4]).scatter(
                &mesh_indices,
                &other_mesh_updates,
                operation.dimensions(),
                operation.kind(),
                operation.options()
            ),
            Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` input, indices, and updates shardings must use one mesh"
            ))
            .into()),
        );

        // The index vector axis must be replicated over explicit mesh axes, and indices never carry reduction state.
        let sharded_vector_indices = indices
            .clone()
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::Replicated, ShardingDimension::sharded(["x"])])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            input.scatter(
                &sharded_vector_indices,
                &updates,
                operation.dimensions(),
                operation.kind(),
                operation.options()
            ),
            Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` indices index vector dimension must be replicated over explicit mesh axes"
            ))
            .into()),
        );
        let reduced_indices = indices
            .clone()
            .with_sharding(Sharding::replicated(mesh.clone(), 2).with_reduced_axes(["x"]).unwrap())
            .unwrap();
        assert_eq!(
            input.scatter(&reduced_indices, &updates, operation.dimensions(), operation.kind(), operation.options()),
            Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` indices cannot carry reduced or unreduced mesh axes"
            ))
            .into()),
        );

        // Reduction state must match between the input and the updates, supports only the linear combiners when
        // unreduced, and requires replicated, invariant indices (unreduced here; the reduced case is covered with the
        // derivatives that depend on it).
        let unreduced = replicated.clone().with_unreduced_axes(["x"]).unwrap();
        let unreduced_input = input.clone().with_sharding(unreduced.clone()).unwrap();
        let unreduced_updates = updates.clone().with_sharding(unreduced.clone()).unwrap();
        assert_eq!(
            unreduced_input.scatter(
                &indices,
                &unreduced_updates,
                operation.dimensions(),
                operation.kind(),
                operation.options()
            ),
            Ok(unreduced_input.clone()),
        );
        assert_eq!(
            unreduced_input.scatter(&indices, &updates, operation.dimensions(), operation.kind(), operation.options()),
            Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` input and updates must have matching reduction state"
            ))
            .into()),
        );
        assert_eq!(
            unreduced_input.scatter(
                &indices,
                &unreduced_updates,
                &operation.dimensions().clone(),
                ScatterReductionKind::Mul,
                &ScatterOptions::new()
            ),
            Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` nonlinear reductions do not support unreduced inputs"
            ))
            .into()),
        );
        let distributed_indices = ArrayType::new_static(DataType::I32, [2, 1])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated])
                    .unwrap(),
            )
            .unwrap();
        let unreduced_pair = ArrayType::new_static(DataType::F32, [2]).with_sharding(unreduced).unwrap();
        assert_eq!(
            unreduced_input.scatter(
                &distributed_indices,
                &unreduced_pair,
                operation.dimensions(),
                operation.kind(),
                operation.options()
            ),
            Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` reduction-state inputs require replicated, invariant indices"
            ))
            .into()),
        );

        // A requested output sharding must use the common mesh, preserve reduction and manual-axis state, have the
        // input rank, and avoid automatic mesh axes.
        assert_eq!(
            input.scatter(
                &indices,
                &updates,
                operation.dimensions(),
                operation.kind(),
                &operation.options().clone().with_output_sharding(Sharding::replicated(other_mesh, 1))
            ),
            Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` requested output sharding uses a different mesh"
            ))
            .into()),
        );
        assert_eq!(
            unreduced_input.scatter(
                &indices,
                &unreduced_updates,
                operation.dimensions(),
                operation.kind(),
                &operation.options().clone().with_output_sharding(replicated.clone())
            ),
            Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` requested output sharding changes reduction or manual-axis state"
            ))
            .into()),
        );
        assert_eq!(
            input.scatter(
                &indices,
                &updates,
                operation.dimensions(),
                operation.kind(),
                &operation.options().clone().with_output_sharding(Sharding::replicated(mesh.clone(), 2))
            ),
            Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` output sharding rank (2) does not match the input rank (1)"
            ))
            .into()),
        );
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let requested = Sharding::new(auto_mesh, vec![ShardingDimension::sharded(["a"])]).unwrap();
        assert_eq!(
            ArrayType::new_static(DataType::F32, [4]).scatter(
                &indices,
                &updates,
                operation.dimensions(),
                operation.kind(),
                &operation.options().clone().with_output_sharding(requested)
            ),
            Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` output sharding cannot reference auto mesh axes"
            ))
            .into()),
        );

        // An explicit request resolves a targeted axis that is sharded over an explicit mesh axis.
        let sharded = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        let sharded_input = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded).unwrap();
        assert_eq!(
            sharded_input.scatter(
                &indices,
                &updates,
                operation.dimensions(),
                operation.kind(),
                &operation.options().clone().with_output_sharding(replicated)
            ),
            Ok(input),
        );
    }

    #[test]
    fn test_scatter_reference_discharge() {
        // Reference-free replay preserves the complete scatter payload. Generic replay behavior and reference
        // rejection are covered by the reference-discharge macro tests.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let expected =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![1], vec![0], vec![0]), ScatterReductionKind::Mul)
                .with_mode(ScatterMode::Drop)
                .with_indices_are_sorted(true)
                .with_unique_indices(true)
                .with_output_sharding(Sharding::replicated(mesh.clone(), 2));
        let operation = ArrayIrOperation::Array(ArrayOperation::Scatter(expected.clone()));
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let context = ReferenceDischargeContext::<_, ArrayReferenceDischarge>::new(trace.clone());
        let inputs = [
            ReferenceDischargeValue::Value(trace.input(ArrayType::new_static(DataType::F64, [3, 2]).into())),
            ReferenceDischargeValue::Value(trace.input(ArrayType::new_static(DataType::I32, [2, 1]).into())),
            ReferenceDischargeValue::Value(trace.input(ArrayType::new_static(DataType::F64, [2, 2]).into())),
        ];
        let outputs = operation.discharge_references(&context, &EmptyRegionDriver, &inputs).unwrap();
        assert_eq!(outputs.len(), 1);
        let ReferenceDischargeValue::Value(output) = &outputs[0] else {
            panic!("expected a value carrier but got {}", outputs[0]);
        };
        assert_eq!(
            output.r#type().as_ref(),
            &ArrayIrType::Array(
                ArrayType::new_static(DataType::F64, [3, 2]).with_sharding(Sharding::replicated(mesh, 2)).unwrap(),
            )
        );
        let builder = trace.builder().borrow();
        assert_eq!(builder.instructions().len(), 1);
        let ArrayIrOperation::Array(ArrayOperation::Scatter(staged)) = builder.instructions()[0].operation() else {
            panic!("expected a staged scatter");
        };
        assert_eq!(staged, &expected);
    }

    #[test]
    fn test_scatter_interpretation() {
        // Repeated windows combine every update, including when window axes precede query axes.
        let input = Array::vector(vec![10_i32, 20, 30, 40]).unwrap();
        let indices = Array::matrix(3, 1, vec![0_i32, 0, 2]).unwrap();
        let options = ScatterOptions::new().with_mode(ScatterMode::Drop);
        assert_eq!(
            input.scatter(
                &indices,
                &Array::matrix(3, 2, vec![1_i32, 2, 4, 8, 16, 32]).unwrap(),
                &ScatterDimensionNumbers::new(vec![1], vec![], vec![0]),
                ScatterReductionKind::Add,
                &options,
            ),
            Array::vector(vec![15_i32, 30, 46, 72]),
        );
        assert_eq!(
            input.scatter(
                &indices,
                &Array::matrix(2, 3, vec![1_i32, 4, 16, 2, 8, 32]).unwrap(),
                &ScatterDimensionNumbers::new(vec![0], vec![], vec![0]),
                ScatterReductionKind::Add,
                &options,
            ),
            Array::vector(vec![15_i32, 30, 46, 72]),
        );
        let indices = Array::matrix(3, 1, vec![-1_i32, 1, 5]).unwrap();
        assert_eq!(
            input.scatter(
                &indices,
                &Array::matrix(3, 2, vec![1_i32, 2, 4, 8, 16, 32]).unwrap(),
                &ScatterDimensionNumbers::new(vec![1], vec![], vec![0]),
                ScatterReductionKind::Add,
                &options,
            ),
            Array::vector(vec![10_i32, 24, 38, 40]),
        );

        // Components are [column, row], so the index map must not be treated as sorted input-axis order.
        let reversed_updates =
            Array::from_elements(ArrayType::new_static(DataType::I32, [2, 1, 2]), &[1_i32, 2, 3, 4]).unwrap();
        assert_eq!(
            Array::matrix(3, 4, vec![0_i32; 12]).unwrap().scatter(
                &Array::matrix(2, 2, vec![1_i32, 0, 2, 2]).unwrap(),
                &reversed_updates,
                &ScatterDimensionNumbers::new(vec![1, 2], vec![], vec![1, 0]),
                ScatterReductionKind::Add,
                &options,
            ),
            Array::matrix(3, 4, vec![0_i32, 1, 2, 0, 0, 0, 0, 0, 0, 0, 3, 4]),
        );

        // Empty index vectors select the implicit zero origin. Both query windows still contribute.
        assert_eq!(
            Array::vector(vec![10_i32, 20]).unwrap().scatter(
                &Array::matrix(2, 0, Vec::<i32>::new()).unwrap(),
                &Array::matrix(2, 2, vec![1_i32, 2, 3, 4]).unwrap(),
                &ScatterDimensionNumbers::new(vec![1], vec![], vec![]),
                ScatterReductionKind::Add,
                &options,
            ),
            Array::vector(vec![14_i32, 26]),
        );

        let dimensions = ScatterDimensionNumbers::new(vec![], vec![0], vec![0]);
        let input = Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let indices = Array::matrix(2, 1, vec![1_i32, 3]).unwrap();
        let updates = Array::vector(vec![100.0, 200.0]).unwrap();
        let operation = ScatterOperation::new(dimensions.clone(), ScatterReductionKind::Add);
        let context = EagerContext::<Array>::new();
        assert_eq!(
            operation.interpret(&context, &EmptyRegionDriver, &[input.clone(), indices.clone(), updates.clone()]),
            Ok(vec![Array::vector(vec![1.0, 102.0, 3.0, 204.0]).unwrap()]),
        );
        assert_eq!(
            operation.interpret(&context, &EmptyRegionDriver, &[]),
            Err(ProgramError::InvalidInputCount { expected: 3, actual: 0 }),
        );

        // Each combiner combines one update with the existing element; duplicate additive updates all contribute.
        assert_eq!(
            input.scatter(&indices, &updates, &dimensions, ScatterReductionKind::Overwrite, &ScatterOptions::new()),
            Array::vector(vec![1.0, 100.0, 3.0, 200.0]),
        );
        assert_eq!(
            input.scatter(&indices, &updates, &dimensions, ScatterReductionKind::Mul, &ScatterOptions::new()),
            Array::vector(vec![1.0, 200.0, 3.0, 800.0]),
        );
        assert_eq!(
            input.scatter(&indices, &updates, &dimensions, ScatterReductionKind::Min, &ScatterOptions::new()),
            Ok(input.clone()),
        );
        assert_eq!(
            input.scatter(&indices, &updates, &dimensions, ScatterReductionKind::Max, &ScatterOptions::new()),
            Array::vector(vec![1.0, 100.0, 3.0, 200.0]),
        );
        assert_eq!(
            input.scatter(
                &Array::matrix(2, 1, vec![1_i32, 1]).unwrap(),
                &updates,
                operation.dimensions(),
                operation.kind(),
                operation.options()
            ),
            Array::vector(vec![1.0, 302.0, 3.0, 4.0]),
        );
    }

    #[test]
    fn test_scatter_interpretation_array_ir() {
        let input = ArrayIrValue::Array(Array::vector(vec![10_i32, 20, 30]).unwrap());
        let indices = ArrayIrValue::Array(Array::matrix(2, 1, vec![2_i32, 0]).unwrap());
        let updates = ArrayIrValue::Array(Array::vector(vec![1_i32, 2]).unwrap());
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Add);
        assert_eq!(
            input.scatter(&indices, &updates, operation.dimensions(), operation.kind(), operation.options()),
            Ok(ArrayIrValue::Array(Array::vector(vec![12_i32, 20, 31]).unwrap())),
        );
        let dimension = ArrayIrValue::Dimension(DimensionValue::constant(1).unwrap());
        assert_eq!(
            input.scatter(&indices, &dimension, operation.dimensions(), operation.kind(), operation.options()),
            Err(TypeError::invalid("expected array type but got dimension type").into()),
        );
        assert_eq!(
            dimension.scatter(&indices, &updates, operation.dimensions(), operation.kind(), operation.options()),
            Err(TypeError::invalid("expected array type but got dimension type").into()),
        );

        // Mixed tracers scatter through their array projection.
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let staged_input = context.input(input.r#type().into_owned());
        let staged_indices = context.lift(indices).unwrap();
        let staged_updates = context.input(updates.r#type().into_owned());
        let projected_input = ValueProjection::<ArrayType>::into_projected(staged_input).unwrap();
        let projected_indices = ValueProjection::<ArrayType>::into_projected(staged_indices).unwrap();
        let projected_updates = ValueProjection::<ArrayType>::into_projected(staged_updates).unwrap();
        let output = projected_input
            .scatter(
                &projected_indices,
                &projected_updates,
                operation.dimensions(),
                operation.kind(),
                operation.options(),
            )
            .unwrap();
        assert_eq!(output.r#type().into_owned(), ArrayType::new_static(DataType::I32, [3]));
    }

    #[test]
    fn test_scatter_interpretation_empty_and_extreme_indices() {
        // Empty updates preserve the payload, but must still pass ordinary type validation.
        let input = Array::vector(vec![10_i32, 20, 30]).unwrap();
        let indices = Array::matrix(0, 1, Vec::<i32>::new()).unwrap();
        let updates = Array::vector(Vec::<i32>::new()).unwrap();
        let dimensions = ScatterDimensionNumbers::new(vec![], vec![0], vec![0]);
        let output = input
            .scatter(&indices, &updates, &dimensions, ScatterReductionKind::Add, &ScatterOptions::new())
            .unwrap();
        assert!(std::sync::Arc::ptr_eq(input.shared_storage(), output.shared_storage()));
        assert_eq!(output, input);
        assert_eq!(
            input.scatter(
                &indices,
                &Array::vector(Vec::<f32>::new()).unwrap(),
                &dimensions,
                ScatterReductionKind::Add,
                &ScatterOptions::new(),
            ),
            Err(TypeError::invalid(format!(
                "`{SCATTER_OPERATION_NAME}` updates data type `f32` does not match input data type `i32`",
            ))
            .into()),
        );

        // No update can address an element of an empty input, whichever bounds mode is selected.
        let input = Array::vector(Vec::<i32>::new()).unwrap();
        let indices = Array::matrix(1, 1, vec![u64::MAX]).unwrap();
        let updates = Array::vector(vec![7_i32]).unwrap();
        let dimensions = ScatterDimensionNumbers::new(vec![], vec![0], vec![0]);
        for mode in [ScatterMode::Clip, ScatterMode::Drop, ScatterMode::PromiseInBounds] {
            let operation = ScatterOperation::new(dimensions.clone(), ScatterReductionKind::Overwrite).with_mode(mode);
            assert_eq!(
                input.scatter(&indices, &updates, operation.dimensions(), operation.kind(), operation.options()),
                Ok(input.clone())
            );
        }

        // Fill-or-drop discards a whole invalid window under every combiner, including the integer extrema, while
        // the one in-bounds window still combines.
        let input = Array::vector(vec![10_i32, 20, 30, 40]).unwrap();
        let indices = Array::matrix(3, 1, vec![-1_i32, 1, 4]).unwrap();
        let updates = Array::vector(vec![1_i32, 2, 3]).unwrap();
        let dropping = ScatterOptions::new().with_mode(ScatterMode::Drop);
        assert_eq!(
            input.scatter(&indices, &updates, &dimensions, ScatterReductionKind::Overwrite, &dropping),
            Array::vector(vec![10_i32, 2, 30, 40]),
        );
        assert_eq!(
            input.scatter(&indices, &updates, &dimensions, ScatterReductionKind::Add, &dropping),
            Array::vector(vec![10_i32, 22, 30, 40]),
        );
        assert_eq!(
            input.scatter(&indices, &updates, &dimensions, ScatterReductionKind::Mul, &dropping),
            Array::vector(vec![10_i32, 40, 30, 40]),
        );
        assert_eq!(
            input.scatter(&indices, &updates, &dimensions, ScatterReductionKind::Min, &dropping),
            Array::vector(vec![10_i32, 2, 30, 40]),
        );
        assert_eq!(
            input.scatter(&indices, &updates, &dimensions, ScatterReductionKind::Max, &dropping),
            Ok(input.clone()),
        );

        // Adding a window offset to a maximal start must not overflow: clipping moves the window to the last valid
        // start and fill-or-drop discards it, for signed and unsigned index types alike.
        let updates = Array::matrix(1, 2, vec![1_i32, 2]).unwrap();
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![1], vec![], vec![0]), ScatterReductionKind::Add);
        let signed = Array::matrix(1, 1, vec![i64::MAX]).unwrap();
        let unsigned = Array::matrix(1, 1, vec![u64::MAX]).unwrap();
        let clipping = operation.clone().with_mode(ScatterMode::Clip);
        assert_eq!(
            input.scatter(&signed, &updates, clipping.dimensions(), clipping.kind(), clipping.options()),
            Array::vector(vec![10_i32, 20, 31, 42])
        );
        assert_eq!(
            input.scatter(&unsigned, &updates, clipping.dimensions(), clipping.kind(), clipping.options()),
            Array::vector(vec![10_i32, 20, 31, 42])
        );
        let dropping = operation.with_mode(ScatterMode::Drop);
        assert_eq!(
            input.scatter(&signed, &updates, dropping.dimensions(), dropping.kind(), dropping.options()),
            Ok(input.clone())
        );
        assert_eq!(
            input.scatter(&unsigned, &updates, dropping.dimensions(), dropping.kind(), dropping.options()),
            Ok(input.clone())
        );
        let maximum =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![1], vec![], vec![0]), ScatterReductionKind::Max)
                .with_mode(ScatterMode::Clip);
        assert_eq!(
            input.scatter(
                &unsigned,
                &Array::matrix(1, 2, vec![35_i32, 5]).unwrap(),
                maximum.dimensions(),
                maximum.kind(),
                maximum.options()
            ),
            Array::vector(vec![10_i32, 20, 35, 40]),
        );
    }

    #[test]
    fn test_scatter_interpretation_layouts_and_element_types() {
        // Scatter-add updates 10 and 20 into elements 3 and 0 of a vector.
        let input = Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let indices = Array::from_elements::<i64>(ArrayType::new_static(DataType::I64, [2, 1]), &[3, 0]).unwrap();
        let updates = Array::vector(vec![10.0, 20.0]).unwrap();
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Add);
        assert_eq!(
            input.scatter(&indices, &updates, operation.dimensions(), operation.kind(), operation.options()),
            Array::vector(vec![21.0, 2.0, 3.0, 14.0])
        );

        // Scatter decodes sub-byte indices through their physical layout without materializing a scalar index vector.
        let indices_type =
            ArrayType::new_static(DataType::I4, [2, 1]).with_layout(Layout::Strided(StridedLayout::new(vec![-1, 1])));
        let indices = Array::from_elements(indices_type, &[i4::new(3).unwrap(), i4::new(0).unwrap()]).unwrap();
        assert_eq!(
            input.scatter(&indices, &updates, operation.dimensions(), operation.kind(), operation.options()),
            Array::vector(vec![21.0, 2.0, 3.0, 14.0])
        );

        // Input and update payloads are decoded and written through their independent physical layouts.
        let input_type =
            ArrayType::new_static(DataType::U16, [4]).with_layout(Layout::Strided(StridedLayout::new(vec![-2])));
        let input = Array::from_elements(input_type.clone(), &[1u16, 2, 3, 4]).unwrap();
        let updates_type =
            ArrayType::new_static(DataType::U16, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![-2])));
        let updates = Array::from_elements(updates_type, &[10u16, 20]).unwrap();
        let expected = Array::from_elements(input_type, &[21u16, 2, 3, 14]);
        assert_eq!(
            input.scatter(&indices, &updates, operation.dimensions(), operation.kind(), operation.options()),
            expected
        );

        // Sub-byte arithmetic wraps in the declared bit width, including repeated modular addition.
        let input = Array::vector(vec![i4::new(7).unwrap(), i4::new(-8).unwrap()]).unwrap();
        let indices = Array::matrix(2, 1, vec![0i32, 1]).unwrap();
        let updates = Array::vector(vec![i4::new(2).unwrap(), i4::new(-3).unwrap()]).unwrap();
        assert_eq!(
            input
                .scatter(&indices, &updates, operation.dimensions(), operation.kind(), operation.options())
                .unwrap()
                .elements::<i4>(),
            Ok(vec![i4::new(-7).unwrap(), i4::new(5).unwrap()]),
        );

        // Overwrite moves encodings without requiring arithmetic identities, including for formats without zero.
        let input = Array::new(ArrayType::new_static(DataType::F8E8M0FNU, [2]), vec![0x7f, 0x80]).unwrap();
        let updates = Array::new(ArrayType::new_static(DataType::F8E8M0FNU, [1]), vec![0x81]).unwrap();
        let indices = Array::matrix(1, 1, vec![0i32]).unwrap();
        let overwrite = ScatterOperation::new(
            ScatterDimensionNumbers::new(vec![], vec![0], vec![0]),
            ScatterReductionKind::Overwrite,
        );
        assert_eq!(
            input.scatter(&indices, &updates, overwrite.dimensions(), overwrite.kind(), overwrite.options()),
            Array::new(ArrayType::new_static(DataType::F8E8M0FNU, [2]), vec![0x81, 0x80]),
        );

        // Addition and multiplication combine complex elements with complex arithmetic.
        let indices = Array::matrix(2, 1, vec![0i32, 1]).unwrap();
        let input = Array::vector(vec![ComplexNumber::new(1.0f32, 2.0), ComplexNumber::new(3.0, 4.0)]).unwrap();
        let updates = Array::vector(vec![ComplexNumber::new(1.0f32, 1.0), ComplexNumber::new(2.0, -1.0)]).unwrap();
        assert_eq!(
            input
                .scatter(&indices, &updates, operation.dimensions(), operation.kind(), operation.options())
                .unwrap()
                .elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(2.0, 3.0), ComplexNumber::new(5.0, 3.0)]),
        );
        let multiply =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Mul);
        assert_eq!(
            input
                .scatter(&indices, &updates, multiply.dimensions(), multiply.kind(), multiply.options())
                .unwrap()
                .elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(-1.0, 3.0), ComplexNumber::new(10.0, 5.0)]),
        );

        // Extrema preserve NaNs and signed zero and order complex values lexicographically.
        let input = Array::vector(vec![f32::NAN, -0.0]).unwrap();
        let updates = Array::vector(vec![1.0f32, 0.0]).unwrap();
        let maximum =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Max);
        let minimum =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Min);
        let maximum_values = input
            .scatter(&indices, &updates, maximum.dimensions(), maximum.kind(), maximum.options())
            .unwrap()
            .elements::<f32>()
            .unwrap();
        assert!(maximum_values[0].is_nan());
        assert_eq!(maximum_values[1].to_bits(), 0.0f32.to_bits());
        let minimum_values = input
            .scatter(&indices, &updates, minimum.dimensions(), minimum.kind(), minimum.options())
            .unwrap()
            .elements::<f32>()
            .unwrap();
        assert!(minimum_values[0].is_nan());
        assert_eq!(minimum_values[1].to_bits(), (-0.0f32).to_bits());

        let input = Array::vector(vec![ComplexNumber::new(1.0f32, 9.0), ComplexNumber::new(2.0, -1.0)]).unwrap();
        let updates = Array::vector(vec![ComplexNumber::new(1.0f32, 10.0), ComplexNumber::new(1.0, 100.0)]).unwrap();
        assert_eq!(
            input
                .scatter(&indices, &updates, maximum.dimensions(), maximum.kind(), maximum.options())
                .unwrap()
                .elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(1.0, 10.0), ComplexNumber::new(2.0, -1.0)]),
        );
        assert_eq!(
            input
                .scatter(&indices, &updates, minimum.dimensions(), minimum.kind(), minimum.options())
                .unwrap()
                .elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(1.0, 9.0), ComplexNumber::new(1.0, 100.0)]),
        );
    }

    #[test]
    fn test_scatter_partial_evaluation() {
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![1], vec![0], vec![0]), ScatterReductionKind::Add);
        // Partial evaluation folds fully known scatters and residualizes an unknown data input.
        let input_value = Array::matrix(3, 2, vec![0.0; 6]).unwrap();
        let indices_value = Array::matrix(2, 1, vec![0_i32, 2]).unwrap();
        let updates_value = Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let expected = Array::matrix(3, 2, vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0]).unwrap();
        check_operation_partial_evaluation!(
            backend = (Array, ArrayOperation<Array>),
            operation = operation.clone(),
            cases = [
                {
                    inputs = [
                        (@known, input_value.clone()),
                        (@known, indices_value.clone()),
                        (@known, updates_value.clone()),
                    ],
                    outputs = [(@known, expected.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = input_value.r#type().into_owned(), replay = input_value.clone())),
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
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Add);
        // Unmapped inputs take the fast path and produce a replicated output.
        check_operation_batching!(
            @exact,
            operation = operation.clone(),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@replicated, Array::vector(vec![1.0, 2.0, 3.0]).unwrap()),
                    (@replicated, Array::matrix(1, 1, vec![2_i32]).unwrap()),
                    (@replicated, Array::vector(vec![10.0]).unwrap()),
                ],
                outputs = [(@replicated, Array::vector(vec![1.0, 2.0, 13.0]).unwrap())],
            }],
        );

        // Mapped indices pair each query with its own input, including when the input and updates are replicated.
        check_operation_batching!(
            @exact,
            operation = operation.clone(),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@replicated, Array::vector(vec![1.0, 2.0, 3.0]).unwrap()),
                    (@mapped(axis = 0), Array::from_elements::<i32>(
                        ArrayType::new_static(DataType::I32, [2, 1, 1]),
                        &[0, 2],
                    ).unwrap()),
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
                    (@mapped(axis = 0), Array::new(
                        ArrayType::new_static(DataType::F8E8M0FNU, [0, 3]), vec![],
                    ).unwrap()),
                    (@mapped(axis = 0), Array::from_elements::<i32>(
                        ArrayType::new_static(DataType::I32, [0, 1, 1]),
                        &[],
                    ).unwrap()),
                    (@mapped(axis = 0), Array::new(
                        ArrayType::new_static(DataType::F8E8M0FNU, [0, 1]), vec![],
                    ).unwrap()),
                ],
                outputs = [(@mapped(axis = 0), Array::new(
                    ArrayType::new_static(DataType::F8E8M0FNU, [0, 3]), vec![],
                ).unwrap())],
            }],
        );

        // Replicated indices retain the mapped input and update axes as one leading window axis.
        check_operation_batching!(
            @exact,
            operation = operation.clone(),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::matrix(2, 4, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap()),
                    (@replicated, Array::matrix(2, 1, vec![1_i32, 3]).unwrap()),
                    (@mapped(axis = 0), Array::matrix(2, 2, vec![10.0, 20.0, 30.0, 40.0]).unwrap()),
                ],
                outputs = [(@mapped(axis = 0), Array::matrix(
                    2,
                    4,
                    vec![1.0, 12.0, 3.0, 24.0, 5.0, 36.0, 7.0, 48.0],
                ).unwrap())],
            }],
        );

        // A mapped extent that differs from the batching extent is rejected before any lifting, as is a missing input.
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array>::new(), 2);
        let indices = ArrayBatch::replicated(Array::matrix(1, 1, vec![0_i32]).unwrap());
        assert_eq!(
            operation
                .batch(
                    &context,
                    &EmptyRegionDriver,
                    &[
                        ArrayBatch::new(Array::matrix(3, 2, vec![0.0; 6]).unwrap(), BatchAxis::new(0)).unwrap(),
                        indices.clone(),
                        ArrayBatch::new(Array::matrix(2, 1, vec![1.0; 2]).unwrap(), BatchAxis::new(0)).unwrap(),
                    ],
                )
                .unwrap_err(),
            BatchingError::MisalignedBatchAxes {
                message: format!("`{SCATTER_OPERATION_NAME}` mapped input extent 3 does not match batching extent 2"),
            },
        );
        assert_eq!(
            operation.batch(&context, &EmptyRegionDriver, &[]).unwrap_err(),
            BatchingError::Program(ProgramError::InvalidInputCount { expected: 3, actual: 0 }),
        );

        // Ragged input extents must never be replaced with packed storage extents while selecting windows.
        let variable = DimensionVariable::new("length", DimensionBounds::new(1, Some(4)).unwrap());
        let ragged = ArrayBatch::new(Array::matrix(2, 3, vec![1_f64, 2., 3., 4., 5., 6.]).unwrap(), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, Array::vector(vec![1_i32, 3]).unwrap(), variable, vec![0])])
            .unwrap();
        let updates = ArrayBatch::new(Array::matrix(2, 1, vec![1.0; 2]).unwrap(), BatchAxis::new(0)).unwrap();
        assert_eq!(
            operation.batch(&context, &EmptyRegionDriver, &[ragged, indices, updates]).unwrap_err(),
            BatchingError::Program(ProgramError::UnsupportedOperation {
                message: format!("`{SCATTER_OPERATION_NAME}` does not support bounded ragged array inputs"),
            }),
        );

        // Mapped axes away from position zero are moved to the front first, and a requested output placement gains a
        // leading replicated batch dimension.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let placed = operation.with_output_sharding(Sharding::replicated(mesh.clone(), 1));
        let (output_type, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |(input, indices, updates)| {
                batch(
                    |(input, indices, updates)| {
                        input.scatter(&indices, &updates, placed.dimensions(), placed.kind(), placed.options())
                    },
                    (input, indices, updates),
                    (BatchAxis::new(1), BatchAxis::new(1), BatchAxis::new(1)),
                    BatchAxis::new(0),
                    None,
                )
                .map_err(ProgramError::from)
            },
            (
                ArrayType::new_static(DataType::F32, [3, 2]),
                ArrayType::new_static(DataType::I32, [1, 2, 1]),
                ArrayType::new_static(DataType::F32, [1, 2]),
            ),
        )
        .unwrap();
        assert_eq!(
            output_type,
            ArrayType::new_static(DataType::F32, [2, 3]).with_sharding(Sharding::replicated(mesh, 2)).unwrap(),
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[3, 2], %1:i32[1, 2, 1], %2:f32[1, 2] .
                let %3:f32[2, 3] = transpose [permutation=[1, 0]] %0
                    %4:f32[2, 1] = transpose [permutation=[1, 0]] %2
                    %5:i32[2, 1, 1] = transpose [permutation=[1, 0, 2]] %1
                    %6:f32[2, 3][sharding={mesh<['x'=2:explicit]>, [{}, {}]}] = scatter [
                        kind=add,
                        dimensions=(update_window=[], inserted_window=[1], scatter_to_operand=[1], \
                            operand_batching=[0], scatter_indices_batching=[0]),
                        output_sharding={mesh<['x'=2:explicit]>, [{}, {}]},
                    ] %3 %5 %4
                in (%6)
            "}
            .trim_end(),
        );
        // Item 0 is column 0 of the input updated at row 2, and item 1 is column 1 updated at row 0.
        assert_eq!(
            program.interpret((
                Array::matrix(3, 2, vec![1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap(),
                Array::from_elements(ArrayType::new_static(DataType::I32, [1, 2, 1]), &[2_i32, 0]).unwrap(),
                Array::matrix(1, 2, vec![10.0_f32, 20.0]).unwrap(),
            )),
            Ok(Array::from_elements(output_type, &[1.0_f32, 2.0, 13.0, 24.0, 5.0, 6.0]).unwrap()),
        );
    }

    #[test]
    fn test_scatter_differentiation() {
        // Check both differentiable inputs against independent finite differences. Unique indices give overwrite
        // and multiplication smooth semantics, and leave an untouched input element with an identity derivative.
        // The zero update also checks multiplication's derivative without dividing by the update value.
        for kind in [ScatterReductionKind::Add, ScatterReductionKind::Overwrite, ScatterReductionKind::Mul] {
            check_gradient!(
                |input, updates| {
                    let indices = index_array(&input, vec![2, 1], vec![0, 2]);
                    input
                        .scatter(
                            &indices,
                            &updates,
                            &ScatterDimensionNumbers::new(vec![], vec![0], vec![0]),
                            kind,
                            &ScatterOptions::new().with_unique_indices(true),
                        )
                        .unwrap()
                        .reduce(&[0], ReductionKind::Sum)
                },
                at = Array::vector(vec![2.0, 5.0, 7.0]).unwrap(),
                with = Array::vector(vec![0.0, 3.0]).unwrap(),
                step = 1e-3,
                tolerance = 1e-6,
            );
            check_gradient!(
                |updates, input| {
                    let indices = index_array(&input, vec![2, 1], vec![0, 2]);
                    input
                        .scatter(
                            &indices,
                            &updates,
                            &ScatterDimensionNumbers::new(vec![], vec![0], vec![0]),
                            kind,
                            &ScatterOptions::new().with_unique_indices(true),
                        )
                        .unwrap()
                        .reduce(&[0], ReductionKind::Sum)
                },
                at = Array::vector(vec![0.0, 3.0]).unwrap(),
                with = Array::vector(vec![2.0, 5.0, 7.0]).unwrap(),
                step = 1e-3,
                tolerance = 1e-6,
            );
        }

        // An empty target has no writable locations for any combiner, even with nonempty updates and clipping.
        // Direct JVP invocation keeps a live update tangent so it exercises the empty-target coefficient rule.
        let empty = Array::vector(Vec::<f64>::new()).unwrap();
        let context = DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new());
        for kind in [
            ScatterReductionKind::Overwrite,
            ScatterReductionKind::Add,
            ScatterReductionKind::Mul,
            ScatterReductionKind::Min,
            ScatterReductionKind::Max,
        ] {
            let operation = ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), kind)
                .with_mode(ScatterMode::Clip);
            let outputs = operation
                .jvp(
                    &context,
                    &EmptyRegionDriver,
                    &[
                        DifferentiationDual::new(empty.clone(), MaybeZero::Value(empty.clone())).unwrap(),
                        DifferentiationDual::new_with_zero_tangent(Array::matrix(1, 1, vec![0_i32]).unwrap()).unwrap(),
                        DifferentiationDual::new(
                            Array::vector(vec![7.0]).unwrap(),
                            MaybeZero::Value(Array::vector(vec![1.0]).unwrap()),
                        )
                        .unwrap(),
                    ],
                )
                .unwrap();
            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].primal(), &empty);
            assert_eq!(outputs[0].tangent().as_value(), Some(&empty));
        }

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
                        &ScatterDimensionNumbers::new(vec![], vec![0], vec![0]),
                        kind,
                        &ScatterOptions::new(),
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
                        &ScatterDimensionNumbers::new(vec![], vec![0], vec![0]),
                        ScatterReductionKind::Mul,
                        &ScatterOptions::new(),
                    )
                    .unwrap()
                    .reduce(&[0], ReductionKind::Sum)
            })
            .unwrap();
        assert_eq!(value.to_f64s(), vec![f64::INFINITY]);
        assert_eq!(gradient.to_f64s(), vec![6.0]);

        // Extremal ties split gradients equally among the retained input and every matching update: the minimum at
        // element 0 is shared by the input and two updates, and the untouched element 1 keeps its identity edge.
        let (value, (input_gradient, updates_gradient)) =
            differentiate_at((Array::vector(vec![2.0, 5.0]).unwrap(), Array::vector(vec![2.0, 2.0, 4.0]).unwrap()))
                .value_and_gradient(|(input, updates)| {
                    let indices = index_array(&input, vec![3, 1], vec![0, 0, 1]);
                    input
                        .scatter(
                            &indices,
                            &updates,
                            &ScatterDimensionNumbers::new(vec![], vec![0], vec![0]),
                            ScatterReductionKind::Min,
                            &ScatterOptions::new(),
                        )
                        .unwrap()
                        .reduce(&[0], ReductionKind::Sum)
                })
                .unwrap();
        assert_eq!(value.to_f64s(), vec![6.0]);
        assert_eq!(input_gradient.to_f64s(), vec![1.0 / 3.0, 0.0]);
        assert_eq!(updates_gradient.to_f64s(), vec![1.0 / 3.0, 1.0 / 3.0, 1.0]);
        // The maximum at element 1 stays with the input, while element 0 is again a three-way tie.
        let (value, (input_gradient, updates_gradient)) =
            differentiate_at((Array::vector(vec![2.0, 5.0]).unwrap(), Array::vector(vec![2.0, 2.0, 4.0]).unwrap()))
                .value_and_gradient(|(input, updates)| {
                    let indices = index_array(&input, vec![3, 1], vec![0, 0, 1]);
                    input
                        .scatter(
                            &indices,
                            &updates,
                            &ScatterDimensionNumbers::new(vec![], vec![0], vec![0]),
                            ScatterReductionKind::Max,
                            &ScatterOptions::new(),
                        )
                        .unwrap()
                        .reduce(&[0], ReductionKind::Sum)
                })
                .unwrap();
        assert_eq!(value.to_f64s(), vec![7.0]);
        assert_eq!(input_gradient.to_f64s(), vec![1.0 / 3.0, 1.0]);
        assert_eq!(updates_gradient.to_f64s(), vec![1.0 / 3.0, 1.0 / 3.0, 0.0]);
        // Repeated overwrite uses one consistent winning update for both the returned primal and its derivative.
        let (value, (input_gradient, updates_gradient)) =
            differentiate_at((Array::vector(vec![2.0, 5.0]).unwrap(), Array::vector(vec![7.0, 8.0, 4.0]).unwrap()))
                .value_and_gradient(|(input, updates)| {
                    let indices = index_array(&input, vec![3, 1], vec![0, 0, 1]);
                    input
                        .scatter(
                            &indices,
                            &updates,
                            &ScatterDimensionNumbers::new(vec![], vec![0], vec![0]),
                            ScatterReductionKind::Overwrite,
                            &ScatterOptions::new(),
                        )
                        .unwrap()
                        .reduce(&[0], ReductionKind::Sum)
                })
                .unwrap();
        assert_eq!(value.to_f64s(), vec![12.0]);
        assert_eq!(input_gradient.to_f64s(), vec![0.0, 0.0]);
        assert_eq!(updates_gradient.to_f64s(), vec![0.0, 1.0, 1.0]);

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
                            &ScatterDimensionNumbers::new(vec![], vec![0], vec![0]),
                            ScatterReductionKind::Mul,
                            &ScatterOptions::new().with_unique_indices(true),
                        )
                        .unwrap()
                        .reduce(&[0], ReductionKind::Sum)
                })
                .unwrap();
        assert_eq!(value.to_f64s(), vec![15.0]);
        assert_eq!(input_gradient.to_f64s(), vec![0.0, 3.0]);
        assert_eq!(updates_gradient.to_f64s(), vec![2.0, 5.0]);

        // Forward mode through `f(x) = scatter_add(x, [[1], [3]], [10, 20])` exercises the captured-index scatter-add
        // under batched basis tangents. Scatter-add is the identity in its input, so the Jacobian with respect to the
        // input is the identity matrix.
        let jacobian = differentiate_at(Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap())
            .jacobian_forward(|x| {
                let indices = index_array(&x, vec![2, 1], vec![1, 3]);
                let updates = x.context().lift(Array::vector(vec![10.0, 20.0]).unwrap())?;
                let operation = ScatterOperation::new(
                    ScatterDimensionNumbers::new(vec![], vec![0], vec![0]),
                    ScatterReductionKind::Add,
                );
                x.scatter(&indices, &updates, operation.dimensions(), operation.kind(), operation.options())
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
    fn test_scatter_differentiation_fill_or_drop() {
        // A dropped window contributes nothing to any derivative. The extremal rule's dual gather fills dropped
        // floating-point windows with NaN, and its count and numerator scatters discard those same windows. Even
        // updates equal to existing input values contribute nothing when out of bounds. Update 1 is the only
        // in-bounds window.
        let dimensions = ScatterDimensionNumbers::new(vec![], vec![0], vec![0]);
        let (value, (input_gradient, updates_gradient)) =
            differentiate_at((Array::vector(vec![2.0, 5.0]).unwrap(), Array::vector(vec![2.0, 0.0, 5.0]).unwrap()))
                .value_and_gradient(|(input, updates)| {
                    let indices = index_array(&input, vec![3, 1], vec![-1, 1, 4]);
                    input
                        .scatter(
                            &indices,
                            &updates,
                            &dimensions,
                            ScatterReductionKind::Min,
                            &ScatterOptions::new().with_mode(ScatterMode::Drop),
                        )
                        .unwrap()
                        .reduce(&[0], ReductionKind::Sum)
                })
                .unwrap();
        assert_eq!(value.to_f64s(), vec![2.0]);
        assert_eq!(input_gradient.to_f64s(), vec![1.0, 0.0]);
        assert_eq!(updates_gradient.to_f64s(), vec![0.0, 1.0, 0.0]);
        // Zero-valued updates are dropped the same way, even where the input holds zero as well.
        let (value, (input_gradient, updates_gradient)) =
            differentiate_at((Array::vector(vec![0.0, 5.0]).unwrap(), Array::vector(vec![0.0, 0.0, 0.0]).unwrap()))
                .value_and_gradient(|(input, updates)| {
                    let indices = index_array(&input, vec![3, 1], vec![-1, 1, 4]);
                    input
                        .scatter(
                            &indices,
                            &updates,
                            &dimensions,
                            ScatterReductionKind::Max,
                            &ScatterOptions::new().with_mode(ScatterMode::Drop),
                        )
                        .unwrap()
                        .reduce(&[0], ReductionKind::Sum)
                })
                .unwrap();
        assert_eq!(value.to_f64s(), vec![5.0]);
        assert_eq!(input_gradient.to_f64s(), vec![1.0, 1.0]);
        assert_eq!(updates_gradient.to_f64s(), vec![0.0, 0.0, 0.0]);
        let (value, (input_gradient, updates_gradient)) =
            differentiate_at((Array::vector(vec![0.0, 5.0]).unwrap(), Array::vector(vec![0.0, 0.0, 0.0]).unwrap()))
                .value_and_gradient(|(input, updates)| {
                    let indices = index_array(&input, vec![3, 1], vec![-1, 1, 4]);
                    input
                        .scatter(
                            &indices,
                            &updates,
                            &dimensions,
                            ScatterReductionKind::Min,
                            &ScatterOptions::new().with_mode(ScatterMode::Drop),
                        )
                        .unwrap()
                        .reduce(&[0], ReductionKind::Sum)
                })
                .unwrap();
        assert_eq!(value.to_f64s(), vec![0.0]);
        assert_eq!(input_gradient.to_f64s(), vec![1.0, 0.0]);
        assert_eq!(updates_gradient.to_f64s(), vec![0.0, 1.0, 0.0]);

        // Repeated overwrite pins a zero ID fill so a dropped window never matches a positive winner ID.
        let (value, (input_gradient, updates_gradient)) =
            differentiate_at((Array::vector(vec![2.0, 5.0]).unwrap(), Array::vector(vec![7.0, 8.0, 9.0]).unwrap()))
                .value_and_gradient(|(input, updates)| {
                    let indices = index_array(&input, vec![3, 1], vec![-1, 1, 4]);
                    input
                        .scatter(
                            &indices,
                            &updates,
                            &dimensions,
                            ScatterReductionKind::Overwrite,
                            &ScatterOptions::new().with_mode(ScatterMode::Drop),
                        )
                        .unwrap()
                        .reduce(&[0], ReductionKind::Sum)
                })
                .unwrap();
        assert_eq!(value.to_f64s(), vec![10.0]);
        assert_eq!(input_gradient.to_f64s(), vec![1.0, 0.0]);
        assert_eq!(updates_gradient.to_f64s(), vec![0.0, 1.0, 0.0]);

        // The product rule multiplies by coefficients scattered with the same mode, so dropped multipliers stay one.
        let (value, (input_gradient, updates_gradient)) =
            differentiate_at((Array::vector(vec![2.0, 5.0]).unwrap(), Array::vector(vec![3.0, 4.0, 5.0]).unwrap()))
                .value_and_gradient(|(input, updates)| {
                    let indices = index_array(&input, vec![3, 1], vec![-1, 1, 4]);
                    input
                        .scatter(
                            &indices,
                            &updates,
                            &dimensions,
                            ScatterReductionKind::Mul,
                            &ScatterOptions::new().with_unique_indices(true).with_mode(ScatterMode::Drop),
                        )
                        .unwrap()
                        .reduce(&[0], ReductionKind::Sum)
                })
                .unwrap();
        assert_eq!(value.to_f64s(), vec![22.0]);
        assert_eq!(input_gradient.to_f64s(), vec![1.0, 4.0]);
        assert_eq!(updates_gradient.to_f64s(), vec![0.0, 5.0, 0.0]);
    }

    #[test]
    fn test_scatter_differentiation_zero_tangent() {
        // The shared all-zero fast path lives in the differentiation context's bind, so a direct rule call reaches
        // the body with structural-zero tangents, which stay a typed zero of the output type for every combiner.
        let dimensions = ScatterDimensionNumbers::new(vec![], vec![0], vec![0]);
        let context = DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new());
        let input = Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let indices = Array::matrix(2, 1, vec![1_i32, 3]).unwrap();
        let updates = Array::vector(vec![10.0, 20.0]).unwrap();
        let outputs = ScatterOperation::new(dimensions.clone(), ScatterReductionKind::Min)
            .jvp(
                &context,
                &EmptyRegionDriver,
                &[
                    DifferentiationDual::new_with_zero_tangent(input.clone()).unwrap(),
                    DifferentiationDual::new_with_zero_tangent(indices.clone()).unwrap(),
                    DifferentiationDual::new_with_zero_tangent(updates).unwrap(),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(*outputs[0].primal(), input);
        assert!(matches!(
            outputs[0].tangent(),
            MaybeZero::Zero(tangent_type) if tangent_type == &ArrayType::new_static(DataType::F64, [4]),
        ));

        // Integer elements have a zero-dimensional tangent space, so integer extrema have structural-zero derivatives
        // regardless of which source wins each element.
        let integer_input = Array::vector(vec![1_i32, 5, 3, 4]).unwrap();
        let integer_updates = Array::vector(vec![2_i32, 9]).unwrap();
        let outputs = ScatterOperation::new(dimensions.clone(), ScatterReductionKind::Max)
            .jvp(
                &context,
                &EmptyRegionDriver,
                &[
                    DifferentiationDual::new_with_zero_tangent(integer_input.clone()).unwrap(),
                    DifferentiationDual::new_with_zero_tangent(indices).unwrap(),
                    DifferentiationDual::new_with_zero_tangent(integer_updates.clone()).unwrap(),
                ],
            )
            .unwrap();
        assert_eq!(*outputs[0].primal(), Array::vector(vec![1_i32, 5, 3, 9]).unwrap());
        assert!(matches!(
            outputs[0].tangent(),
            MaybeZero::Zero(tangent_type) if tangent_type == &ArrayType::new_static(DataType::Zero, [4]),
        ));
        let (output, tangent) = differentiate_at(integer_input)
            .jvp(Array::new(ArrayType::new_static(DataType::Zero, [4]), Vec::new()).unwrap(), |input| {
                let indices = index_array(&input, vec![2, 1], vec![1, 3]);
                let updates = input.context().lift(integer_updates.clone())?;
                input.scatter(&indices, &updates, &dimensions, ScatterReductionKind::Min, &ScatterOptions::new())
            })
            .unwrap();
        assert_eq!(output, Array::vector(vec![1_i32, 2, 3, 4]).unwrap());
        assert_eq!(tangent, Array::new(ArrayType::new_static(DataType::Zero, [4]), Vec::new()).unwrap());

        // Arity is validated before any input is inspected.
        assert_eq!(
            ScatterOperation::new(dimensions, ScatterReductionKind::Add)
                .jvp(&context, &EmptyRegionDriver, &[])
                .unwrap_err(),
            DifferentiationError::Program(ProgramError::InvalidInputCount { expected: 3, actual: 0 }),
        );
    }

    #[test]
    fn test_scatter_differentiation_unsupported() {
        // Multiplicative updates with live tangents need disjoint windows: the product rule scatter-adds the update
        // tangents into the multiplier, which is only the derivative when no two updates meet at one element.
        let dimensions = ScatterDimensionNumbers::new(vec![], vec![0], vec![0]);
        let context = DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new());
        assert_eq!(
            ScatterOperation::new(dimensions.clone(), ScatterReductionKind::Mul)
                .jvp(
                    &context,
                    &EmptyRegionDriver,
                    &[
                        DifferentiationDual::new_with_zero_tangent(Array::vector(vec![2.0, 5.0]).unwrap()).unwrap(),
                        DifferentiationDual::new_with_zero_tangent(Array::matrix(2, 1, vec![0_i32, 1]).unwrap())
                            .unwrap(),
                        DifferentiationDual::new(
                            Array::vector(vec![3.0, 4.0]).unwrap(),
                            MaybeZero::Value(Array::vector(vec![1.0, 1.0]).unwrap()),
                        )
                        .unwrap(),
                    ],
                )
                .unwrap_err(),
            DifferentiationError::Program(ProgramError::UnsupportedOperation {
                message: format!(
                    "`{SCATTER_OPERATION_NAME}` multiplication derivatives with respect to updates require \
                     `unique_indices=true`"
                ),
            }),
        );

        // The nonlinear rules read the update windows back through a dual gather, whose window sizes must be static.
        let window = DimensionVariable::new("window", DimensionBounds::new(1, Some(2)).unwrap());
        let dynamic_window_updates =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(window)]));
        let program = constant_index_scatter_program(
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![1], vec![0], vec![0]), ScatterReductionKind::Min),
            ArrayType::new_static(DataType::F64, [3, 2]),
            Array::matrix(2, 1, vec![0_i32, 2]).unwrap(),
            dynamic_window_updates,
        );
        assert_eq!(
            program.linearize().unwrap_err(),
            DifferentiationError::Program(ProgramError::UnsupportedOperation {
                message: format!(
                    "`{SCATTER_OPERATION_NAME}` differentiation requires a static update window on axis 1 but its \
                     extent is `window`"
                ),
            }),
        );

        // Repeated overwrite enumerates one ID per update element, which needs the complete static update shape.
        let queries = DimensionVariable::new("queries", DimensionBounds::new(1, Some(4)).unwrap());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F64, [4]).into());
        let indices = builder.add_input(
            ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(queries.clone()), Dimension::Static(1)]))
                .into(),
        );
        let updates =
            builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(queries)])).into());
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Scatter(ScatterOperation::new(
                    dimensions,
                    ScatterReductionKind::Overwrite,
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
        assert_eq!(
            program.linearize().unwrap_err(),
            DifferentiationError::Program(ProgramError::UnsupportedOperation {
                message: format!(
                    "`{SCATTER_OPERATION_NAME}` overwrite differentiation with repeated indices requires a static \
                     update shape"
                ),
            }),
        );
    }

    #[test]
    fn test_scatter_differentiation_array_ir() {
        // Explicit output placement can differ from the input. Coefficients and masks follow the value edge where
        // they are consumed, and the final pullback restores each original input's placement.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded_input_type = ArrayType::new_static(DataType::F64, [4])
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let replicated_updates_type =
            ArrayType::new_static(DataType::F64, [2]).with_sharding(replicated.clone()).unwrap();
        let indices = Array::matrix(2, 1, vec![1_i32, 3]).unwrap();
        let dimensions = ScatterDimensionNumbers::new(vec![], vec![0], vec![0]);
        let expected_cotangent_types = vec![
            ArrayIrType::Array(sharded_input_type.cotangent().unwrap()),
            ArrayIrType::Array(replicated_updates_type.cotangent().unwrap()),
        ];
        let placed = |kind| {
            ScatterOperation::new(dimensions.clone(), kind)
                .with_unique_indices(kind == ScatterReductionKind::Mul)
                .with_output_sharding(replicated.clone())
        };
        for kind in [
            ScatterReductionKind::Overwrite,
            ScatterReductionKind::Mul,
            ScatterReductionKind::Min,
            ScatterReductionKind::Max,
        ] {
            let program = constant_index_scatter_program(
                placed(kind),
                sharded_input_type.clone(),
                indices.clone(),
                replicated_updates_type.clone(),
            );
            let pullback = program.linearize().unwrap().pullback().unwrap();
            assert_eq!(pullback.output_types(), expected_cotangent_types);
        }

        // Winner IDs preserve the mesh without reduction markers; extremal coefficients preserve reduced data state.
        let repeated = Array::matrix(2, 1, vec![1_i32, 1]).unwrap();
        let unreduced = replicated.clone().with_unreduced_axes(["x"]).unwrap();
        let reduced = replicated.with_reduced_axes(["x"]).unwrap();
        for (kind, sharding) in [
            (ScatterReductionKind::Overwrite, unreduced),
            (ScatterReductionKind::Overwrite, reduced.clone()),
            (ScatterReductionKind::Min, reduced.clone()),
            (ScatterReductionKind::Max, reduced),
        ] {
            let input_type = ArrayType::new_static(DataType::F64, [4]).with_sharding(sharding.clone()).unwrap();
            let updates_type = ArrayType::new_static(DataType::F64, [2]).with_sharding(sharding).unwrap();
            let program = constant_index_scatter_program(
                ScatterOperation::new(dimensions.clone(), kind),
                input_type.clone(),
                repeated.clone(),
                updates_type.clone(),
            );
            let pullback = program.linearize().unwrap().pullback().unwrap();
            assert_eq!(
                pullback.output_types(),
                vec![
                    ArrayIrType::Array(input_type.cotangent().unwrap()),
                    ArrayIrType::Array(updates_type.cotangent().unwrap()),
                ],
            );
        }

        // The same nonlinear coefficient rules work when the input extent is an ordinary symbolic dimension. Each
        // program takes the indices as an input and is pulled back from an all-ones cotangent.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(4, Some(7)).unwrap());
        let dynamic_input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let indices_type = ArrayType::new_static(DataType::I32, [2, 1]);
        let updates_type = ArrayType::new_static(DataType::F64, [2]);
        let dynamic_program = |operation: ScatterOperation| {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let input = builder.add_input(dynamic_input_type.clone().into());
            let indices = builder.add_input(indices_type.clone().into());
            let updates = builder.add_input(updates_type.clone().into());
            let output = builder
                .add_instruction(
                    ArrayIrOperation::Array(ArrayOperation::Scatter(operation)),
                    Vec::new(),
                    vec![input, indices, updates],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![output],
                    vec![Placeholder, Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let pullback_at = |linearization: &Linearization<ArrayIrValue<Array>, ArrayIrOperation<Array>>,
                           index_values: &[i32]| {
            let mut primals = linearization
                .primal()
                .interpret(vec![
                    ArrayIrValue::Array(Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap()),
                    ArrayIrValue::Array(Array::from_elements(indices_type.clone(), index_values).unwrap()),
                    ArrayIrValue::Array(Array::vector(vec![2.0, 1.0]).unwrap()),
                ])
                .unwrap();
            let mut seeds = vec![ArrayIrValue::Array(Array::vector(vec![1.0; 4]).unwrap())];
            seeds.extend(primals.split_off(1));
            linearization.pullback().unwrap().interpret(seeds)
        };
        // Minimum: element 1 ties the input (2) with update 0 (2), element 3 takes update 1 (1 < 4).
        let linearization = dynamic_program(ScatterOperation::new(dimensions.clone(), ScatterReductionKind::Min))
            .linearize()
            .unwrap();
        assert_eq!(
            pullback_at(&linearization, &[1, 3]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0, 0.5, 1.0, 0.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![0.5, 1.0]).unwrap()),
            ]),
        );
        // Multiplication: the input derivative is the scattered multiplier, the update derivative the input value.
        let linearization = dynamic_program(
            ScatterOperation::new(dimensions.clone(), ScatterReductionKind::Mul).with_unique_indices(true),
        )
        .linearize()
        .unwrap();
        assert_eq!(
            pullback_at(&linearization, &[1, 3]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0, 2.0, 1.0, 1.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![2.0, 4.0]).unwrap()),
            ]),
        );
        // Repeated overwrite: one update wins element 1 and erases the input there.
        let linearization = dynamic_program(ScatterOperation::new(dimensions.clone(), ScatterReductionKind::Overwrite))
            .linearize()
            .unwrap();
        assert_eq!(
            pullback_at(&linearization, &[1, 1]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0, 0.0, 1.0, 1.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![0.0, 1.0]).unwrap()),
            ]),
        );

        // Scatter-add over a symbolic input extent retains the indices as the one residual of its linear tangent
        // program; the tangent is the same scatter of the tangents and the pullback gathers the update cotangent.
        let linearization =
            dynamic_program(ScatterOperation::new(dimensions, ScatterReductionKind::Add)).linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[extent], %1:f64[2], %2:i32[2, 1] .
                let %3:f64[extent] = scatter [
                    kind=add,
                    dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], operand_batching=[], \
                        scatter_indices_batching=[]),
                ] %0 %2 %1
                in (%3)
            "}
            .trim_end(),
        );
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
    fn test_scatter_differentiation_array_ir_linear() {
        // The linear combiners stage the same scatter of the tangents and transpose to a dual gather (plus, for unique
        // overwrite, the erasure of the written windows), carrying the index hints onto the gather.
        let dimensions = ScatterDimensionNumbers::new(vec![], vec![0], vec![0]);
        let input_type = ArrayType::new_static(DataType::F64, [4]);
        let updates_type = ArrayType::new_static(DataType::F64, [2]);
        let indices = Array::matrix(2, 1, vec![1_i32, 3]).unwrap();
        let program = constant_index_scatter_program(
            ScatterOperation::new(dimensions.clone(), ScatterReductionKind::Add),
            input_type.clone(),
            indices.clone(),
            updates_type.clone(),
        );
        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[2] .
                let %2:i32[2, 1] = const [[1], [3]]
                    %3:f64[4] = scatter [
                        kind=add,
                        dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                            operand_batching=[], scatter_indices_batching=[]),
                    ] %0 %2 %1
                in (%3)
            "}
            .trim_end(),
        );
        let pullback = linearization.pullback().unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[4] .
                let %1:i32[2, 1] = const [[1], [3]]
                    %2:f64[2] = gather [
                        dimensions=(offset=[], collapsed_slice=[0], start_index_map=[0], batching=[]),
                        slice_sizes=[1],
                    ] %0 %1
                in (%0, %2)
            "}
            .trim_end(),
        );
        assert_eq!(
            pullback.interpret(vec![ArrayIrValue::Array(Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap())]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![2.0, 4.0]).unwrap()),
            ]),
        );

        let program = constant_index_scatter_program(
            ScatterOperation::new(dimensions, ScatterReductionKind::Overwrite)
                .with_indices_are_sorted(true)
                .with_unique_indices(true),
            input_type,
            indices,
            updates_type,
        );
        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[2] .
                let %2:i32[2, 1] = const [[1], [3]]
                    %3:f64[4] = scatter [
                        kind=overwrite,
                        dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                            operand_batching=[], scatter_indices_batching=[]),
                        indices_are_sorted=true,
                        unique_indices=true,
                    ] %0 %2 %1
                in (%3)
            "}
            .trim_end(),
        );
        let pullback = linearization.pullback().unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[4] .
                let %1:i32[2, 1] = const [[1], [3]]
                    %2:f64[2] = zero [type=f64[2]]
                    %3:f64[4] = scatter [
                        kind=overwrite,
                        dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                            operand_batching=[], scatter_indices_batching=[]),
                        indices_are_sorted=true,
                        unique_indices=true,
                    ] %0 %1 %2
                    %4:f64[2] = gather [
                        dimensions=(offset=[], collapsed_slice=[0], start_index_map=[0], batching=[]),
                        slice_sizes=[1],
                        indices_are_sorted=true,
                        unique_indices=true,
                    ] %0 %1
                in (%3, %4)
            "}
            .trim_end(),
        );
        assert_eq!(
            pullback.interpret(vec![ArrayIrValue::Array(Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap())]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0, 0.0, 3.0, 0.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![2.0, 4.0]).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_scatter_differentiation_array_ir_nonlinear() {
        // The nonlinear combiners compute their coefficients in the primal program and stage a linear tangent program
        // over them. Every dual gather these rules build carries the scatter's index hints: the `min` case pins both
        // hints on its target gather and the repeated-overwrite case pins the sortedness hint on its winner-ID gather,
        // while `max` without hints pins a bare gather and `mul` never builds one.
        let dimensions = ScatterDimensionNumbers::new(vec![], vec![0], vec![0]);
        let input_type = ArrayType::new_static(DataType::F64, [4]);
        let updates_type = ArrayType::new_static(DataType::F64, [2]);
        let distinct = Array::matrix(2, 1, vec![1_i32, 3]).unwrap();
        let repeated = Array::matrix(2, 1, vec![1_i32, 1]).unwrap();
        let vector = |values: Vec<f64>| ArrayIrValue::Array(Array::vector(values).unwrap());
        // Every case starts from the input `[1, 2, 3, 4]`, the input tangent `[1, 2, 3, 4]`, the update tangent
        // `[5, 6]`, and the output cotangent `[10, 20, 30, 40]`, and returns the primal output, the tangent output,
        // and the input and update cotangents.
        let evaluate = |linearization: &Linearization<ArrayIrValue<Array>, ArrayIrOperation<Array>>,
                        updates: Vec<f64>| {
            let mut primal_outputs =
                linearization.primal().interpret(vec![vector(vec![1.0, 2.0, 3.0, 4.0]), vector(updates)]).unwrap();
            let residuals = primal_outputs.split_off(1);
            let mut tangent_inputs = vec![vector(vec![1.0, 2.0, 3.0, 4.0]), vector(vec![5.0, 6.0])];
            tangent_inputs.extend(residuals.clone());
            let mut pullback_inputs = vec![vector(vec![10.0, 20.0, 30.0, 40.0])];
            pullback_inputs.extend(residuals);
            (
                primal_outputs.remove(0),
                linearization.tangent().interpret(tangent_inputs).unwrap().remove(0),
                linearization.pullback().unwrap().interpret(pullback_inputs).unwrap(),
            )
        };

        // Repeated overwrite: the last update wins element 1, so the primal, tangent, and cotangent all follow it.
        let program = constant_index_scatter_program(
            ScatterOperation::new(dimensions.clone(), ScatterReductionKind::Overwrite).with_indices_are_sorted(true),
            input_type.clone(),
            repeated.clone(),
            updates_type.clone(),
        );
        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[2] .
                let %2:f64[4] = zero_like %0
                    %3:u64[4] = convert_element_type [data_type=u64] %2
                    %4:u64[4] = broadcast [output_type=u64[4], output_axes=[0]] %3
                    %5:i32[2, 1] = const [[1], [1]]
                    %6:u64[2] = iota [type=u64[2], dimension=0]
                    %7:u64[2] = reshape [shape=[2]] %6
                    %8:u64[2] = one_like %7
                    %9:u64[2] = add %7 %8
                    %10:u64[2] = broadcast [output_type=u64[2], output_axes=[0]] %9
                    %11:u64[4] = scatter [
                        kind=overwrite,
                        dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                            operand_batching=[], scatter_indices_batching=[]),
                        indices_are_sorted=true,
                    ] %4 %5 %10
                    %12:bool[4] = compare [direction=Equal] %11 %4
                    %13:f64[4] = zero_like %0
                    %14:f64[4] = select %12 %0 %13
                    %15:u64[2] = gather [
                        dimensions=(offset=[], collapsed_slice=[0], start_index_map=[0], batching=[]),
                        slice_sizes=[1],
                        indices_are_sorted=true,
                    ] %11 %5
                    %16:bool[2] = compare [direction=Equal] %10 %15
                    %17:f64[2] = zero_like %1
                    %18:f64[2] = select %16 %1 %17
                    %19:f64[4] = scatter [
                        kind=add,
                        dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                            operand_batching=[], scatter_indices_batching=[]),
                        indices_are_sorted=true,
                    ] %14 %5 %18
                in (%19, %12, %16)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[2], %2:bool[4], %3:bool[2] .
                let %4:f64[4] = zero_like %0
                    %5:f64[4] = select %2 %0 %4
                    %6:i32[2, 1] = const [[1], [1]]
                    %7:f64[2] = zero_like %1
                    %8:f64[2] = select %3 %1 %7
                    %9:f64[4] = scatter [
                        kind=add,
                        dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                            operand_batching=[], scatter_indices_batching=[]),
                        indices_are_sorted=true,
                    ] %5 %6 %8
                in (%9)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.pullback().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:bool[4], %2:bool[2] .
                let %3:i32[2, 1] = const [[1], [1]]
                    %4:f64[2] = gather [
                        dimensions=(offset=[], collapsed_slice=[0], start_index_map=[0], batching=[]),
                        slice_sizes=[1],
                        indices_are_sorted=true,
                    ] %0 %3
                    %5:f64[2] = zero [type=f64[2]]
                    %6:f64[2] = select %2 %4 %5
                    %7:f64[2] = select %2 %5 %4
                    %8:f64[4] = zero [type=f64[4]]
                    %9:f64[4] = select %1 %0 %8
                    %10:f64[4] = select %1 %8 %0
                in (%9, %6)
            "}
            .trim_end(),
        );
        assert_eq!(
            evaluate(&linearization, vec![10.0, 20.0]),
            (
                vector(vec![1.0, 20.0, 3.0, 4.0]),
                vector(vec![1.0, 6.0, 3.0, 4.0]),
                vec![vector(vec![10.0, 0.0, 30.0, 40.0]), vector(vec![0.0, 20.0])],
            ),
        );

        // Multiplication: the input coefficient is the scatter of the updates into ones and the update coefficient is
        // the input itself, so no gather is needed until the pullback reads the update cotangents back.
        let program = constant_index_scatter_program(
            ScatterOperation::new(dimensions.clone(), ScatterReductionKind::Mul).with_unique_indices(true),
            input_type.clone(),
            distinct.clone(),
            updates_type.clone(),
        );
        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[2] .
                let %2:i32[2, 1] = const [[1], [3]]
                    %3:f64[4] = scatter [
                        kind=mul,
                        dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                            operand_batching=[], scatter_indices_batching=[]),
                        unique_indices=true,
                    ] %0 %2 %1
                    %4:f64[4] = one_like %0
                    %5:f64[4] = scatter [
                        kind=mul,
                        dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                            operand_batching=[], scatter_indices_batching=[]),
                        unique_indices=true,
                    ] %4 %2 %1
                in (%3, %5, %0)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[2], %2:f64[4], %3:f64[4] .
                let %4:f64[4] = mul %0 %2
                    %5:f64[4] = zero_like %0
                    %6:i32[2, 1] = const [[1], [3]]
                    %7:f64[4] = scatter [
                        kind=add,
                        dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                            operand_batching=[], scatter_indices_batching=[]),
                        unique_indices=true,
                    ] %5 %6 %1
                    %8:f64[4] = mul %3 %7
                    %9:f64[4] = add %4 %8
                in (%9)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.pullback().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4], %2:f64[4] .
                let %3:f64[4] = mul %2 %0
                    %4:i32[2, 1] = const [[1], [3]]
                    %5:f64[2] = gather [
                        dimensions=(offset=[], collapsed_slice=[0], start_index_map=[0], batching=[]),
                        slice_sizes=[1],
                        unique_indices=true,
                    ] %3 %4
                    %6:f64[4] = mul %1 %0
                in (%6, %5)
            "}
            .trim_end(),
        );
        assert_eq!(
            evaluate(&linearization, vec![10.0, 20.0]),
            (
                vector(vec![1.0, 20.0, 3.0, 80.0]),
                vector(vec![1.0, 30.0, 3.0, 104.0]),
                vec![vector(vec![10.0, 200.0, 30.0, 800.0]), vector(vec![40.0, 160.0])],
            ),
        );

        // Minimum with both hints: update 0 wins element 1 and the input keeps element 3.
        let program = constant_index_scatter_program(
            ScatterOperation::new(dimensions.clone(), ScatterReductionKind::Min)
                .with_indices_are_sorted(true)
                .with_unique_indices(true),
            input_type.clone(),
            distinct,
            updates_type.clone(),
        );
        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[2] .
                let %2:i32[2, 1] = const [[1], [3]]
                    %3:f64[4] = scatter [
                        kind=min,
                        dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                            operand_batching=[], scatter_indices_batching=[]),
                        indices_are_sorted=true,
                        unique_indices=true,
                    ] %0 %2 %1
                    %4:f64[4] = zero_like %0
                    %5:f64[2] = one_like %1
                    %6:f64[4] = scatter [
                        kind=add,
                        dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                            operand_batching=[], scatter_indices_batching=[]),
                        indices_are_sorted=true,
                        unique_indices=true,
                    ] %4 %2 %5
                    %7:f64[4] = zero_like %6
                    %8:bool[4] = compare [direction=Equal] %6 %7
                    %9:bool[4] = compare [direction=Equal] %0 %3
                    %10:bool[4] = one_like %9
                    %11:bool[4] = select %8 %10 %9
                    %12:f64[2] = gather [
                        dimensions=(offset=[], collapsed_slice=[0], start_index_map=[0], batching=[]),
                        slice_sizes=[1],
                        indices_are_sorted=true,
                        unique_indices=true,
                    ] %3 %2
                    %13:bool[2] = compare [direction=Equal] %1 %12
                    %14:f64[4] = convert_element_type [data_type=f64] %11
                    %15:f64[2] = convert_element_type [data_type=f64] %13
                    %16:f64[4] = scatter [
                        kind=add,
                        dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                            operand_batching=[], scatter_indices_batching=[]),
                        indices_are_sorted=true,
                        unique_indices=true,
                    ] %14 %2 %15
                    %17:f64[4] = zero_like %16
                    %18:bool[4] = compare [direction=Equal] %16 %17
                    %19:f64[4] = one_like %16
                    %20:f64[4] = select %18 %19 %16
                    %21:f64[4] = one_like %20
                    %22:f64[4] = div %21 %20
                in (%3, %11, %13, %22)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[2], %2:bool[4], %3:bool[2], %4:f64[4] .
                let %5:f64[4] = zero_like %0
                    %6:f64[4] = select %2 %0 %5
                    %7:i32[2, 1] = const [[1], [3]]
                    %8:f64[2] = zero_like %1
                    %9:f64[2] = select %3 %1 %8
                    %10:f64[4] = scatter [
                        kind=add,
                        dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                            operand_batching=[], scatter_indices_batching=[]),
                        indices_are_sorted=true,
                        unique_indices=true,
                    ] %6 %7 %9
                    %11:f64[4] = mul %10 %4
                in (%11)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.pullback().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:bool[4], %2:bool[2], %3:f64[4] .
                let %4:f64[4] = mul %3 %0
                    %5:i32[2, 1] = const [[1], [3]]
                    %6:f64[2] = gather [
                        dimensions=(offset=[], collapsed_slice=[0], start_index_map=[0], batching=[]),
                        slice_sizes=[1],
                        indices_are_sorted=true,
                        unique_indices=true,
                    ] %4 %5
                    %7:f64[2] = zero [type=f64[2]]
                    %8:f64[2] = select %2 %6 %7
                    %9:f64[2] = select %2 %7 %6
                    %10:f64[4] = zero [type=f64[4]]
                    %11:f64[4] = select %1 %4 %10
                    %12:f64[4] = select %1 %10 %4
                in (%11, %8)
            "}
            .trim_end(),
        );
        assert_eq!(
            evaluate(&linearization, vec![0.0, 20.0]),
            (
                vector(vec![1.0, 0.0, 3.0, 4.0]),
                vector(vec![1.0, 5.0, 3.0, 4.0]),
                vec![vector(vec![10.0, 0.0, 30.0, 40.0]), vector(vec![20.0, 0.0])],
            ),
        );

        // Maximum without hints over repeated indices: both updates tie at element 1 and share its derivative.
        let program = constant_index_scatter_program(
            ScatterOperation::new(dimensions, ScatterReductionKind::Max),
            input_type,
            repeated,
            updates_type,
        );
        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[2] .
                let %2:i32[2, 1] = const [[1], [1]]
                    %3:f64[4] = scatter [
                        kind=max,
                        dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                            operand_batching=[], scatter_indices_batching=[]),
                    ] %0 %2 %1
                    %4:f64[4] = zero_like %0
                    %5:f64[2] = one_like %1
                    %6:f64[4] = scatter [
                        kind=add,
                        dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                            operand_batching=[], scatter_indices_batching=[]),
                    ] %4 %2 %5
                    %7:f64[4] = zero_like %6
                    %8:bool[4] = compare [direction=Equal] %6 %7
                    %9:bool[4] = compare [direction=Equal] %0 %3
                    %10:bool[4] = one_like %9
                    %11:bool[4] = select %8 %10 %9
                    %12:f64[2] = gather [
                        dimensions=(offset=[], collapsed_slice=[0], start_index_map=[0], batching=[]),
                        slice_sizes=[1],
                    ] %3 %2
                    %13:bool[2] = compare [direction=Equal] %1 %12
                    %14:f64[4] = convert_element_type [data_type=f64] %11
                    %15:f64[2] = convert_element_type [data_type=f64] %13
                    %16:f64[4] = scatter [
                        kind=add,
                        dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                            operand_batching=[], scatter_indices_batching=[]),
                    ] %14 %2 %15
                    %17:f64[4] = zero_like %16
                    %18:bool[4] = compare [direction=Equal] %16 %17
                    %19:f64[4] = one_like %16
                    %20:f64[4] = select %18 %19 %16
                    %21:f64[4] = one_like %20
                    %22:f64[4] = div %21 %20
                in (%3, %11, %13, %22)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[2], %2:bool[4], %3:bool[2], %4:f64[4] .
                let %5:f64[4] = zero_like %0
                    %6:f64[4] = select %2 %0 %5
                    %7:i32[2, 1] = const [[1], [1]]
                    %8:f64[2] = zero_like %1
                    %9:f64[2] = select %3 %1 %8
                    %10:f64[4] = scatter [
                        kind=add,
                        dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                            operand_batching=[], scatter_indices_batching=[]),
                    ] %6 %7 %9
                    %11:f64[4] = mul %10 %4
                in (%11)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.pullback().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:bool[4], %2:bool[2], %3:f64[4] .
                let %4:f64[4] = mul %3 %0
                    %5:i32[2, 1] = const [[1], [1]]
                    %6:f64[2] = gather [
                        dimensions=(offset=[], collapsed_slice=[0], start_index_map=[0], batching=[]),
                        slice_sizes=[1],
                    ] %4 %5
                    %7:f64[2] = zero [type=f64[2]]
                    %8:f64[2] = select %2 %6 %7
                    %9:f64[2] = select %2 %7 %6
                    %10:f64[4] = zero [type=f64[4]]
                    %11:f64[4] = select %1 %4 %10
                    %12:f64[4] = select %1 %10 %4
                in (%11, %8)
            "}
            .trim_end(),
        );
        assert_eq!(
            evaluate(&linearization, vec![20.0, 20.0]),
            (
                vector(vec![1.0, 20.0, 3.0, 4.0]),
                vector(vec![1.0, 5.5, 3.0, 4.0]),
                vec![vector(vec![10.0, 0.0, 30.0, 40.0]), vector(vec![10.0, 10.0])],
            ),
        );
    }

    #[test]
    fn test_scatter_differentiation_array_ir_zero_tangent() {
        // The member rule stages only the primal scatter when the dynamically shaped inputs carry structural-zero
        // tangents, and it validates its arity before touching any input.
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Min);
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let context = DifferentiationContext::fused(trace.clone());
        let extent = DimensionVariable::new("extent", DimensionBounds::new(4, Some(7)).unwrap());
        let input_type =
            ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)])));
        let indices_type = ArrayIrType::Array(ArrayType::new_static(DataType::I32, [2, 1]));
        let updates_type = ArrayIrType::Array(ArrayType::new_static(DataType::F64, [2]));
        let outputs = operation
            .jvp_in_parent(
                &context,
                &EmptyRegionDriver,
                &[
                    DifferentiationDual::new_with_zero_tangent(trace.input(input_type.clone())).unwrap(),
                    DifferentiationDual::new_with_zero_tangent(trace.input(indices_type)).unwrap(),
                    DifferentiationDual::new_with_zero_tangent(trace.input(updates_type)).unwrap(),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal().r#type().as_ref(), &input_type);
        assert!(matches!(outputs[0].tangent(), MaybeZero::Zero(tangent_type) if tangent_type == &input_type));
        let builder = trace.builder().borrow();
        assert_eq!(builder.instructions().len(), 1);
        assert_eq!(builder.instructions()[0].operation().name(), SCATTER_OPERATION_NAME);
        drop(builder);
        assert_eq!(
            operation.jvp_in_parent(&context, &EmptyRegionDriver, &[]).unwrap_err(),
            DifferentiationError::Program(ProgramError::InvalidInputCount { expected: 3, actual: 0 }),
        );
    }

    #[test]
    fn test_scatter_differentiation_batched_dynamic_extent() {
        // Jointly mapping the input, the indices, and the updates over a dynamic extent pairs the mapped axes as
        // batching axes, so the mapped extent never enters a static window size. The dual gather of the transpose
        // then takes a zero batching window when that extent may be empty at runtime, because a size-one window would
        // exceed the guaranteed minimum extent; the batching axis contributes no window elements, so the zero window
        // does not empty the gathered cotangent.
        let items = DimensionVariable::new("items", DimensionBounds::new(0, Some(9)).unwrap());
        let items_type = DimensionType::new(items.clone());
        let add =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Add);
        let program = jointly_mapped_dynamic_scatter_program(items.clone(), 0, add.clone());
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<items ∈ [0, 9)>, %1:f64[items, 3], %2:i32[items, 1, 1], %3:f64[items, 1] .
                let %4:f64[items, 3] = scatter [
                    kind=add,
                    dimensions=(update_window=[], inserted_window=[1], scatter_to_operand=[1], operand_batching=[0], \
                        scatter_indices_batching=[0]),
                ] %1 %2 %3
                in (%4)
            "}
            .trim_end(),
        );
        let linearization = program.linearize_with_respect_to(&[1, 3]).unwrap();
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[items, 3], %1:f64[items, 1], %2:i32[items, 1, 1] .
                let %3:f64[items, 3] = scatter [
                    kind=add,
                    dimensions=(update_window=[], inserted_window=[1], scatter_to_operand=[1], operand_batching=[0], \
                        scatter_indices_batching=[0]),
                ] %0 %2 %1
                in (%3)
            "}
            .trim_end(),
        );
        let pullback = linearization.pullback().unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[items, 3], %1:i32[items, 1, 1] .
                let %2:f64[items, 1] = gather [
                    dimensions=(offset=[], collapsed_slice=[1], start_index_map=[1], batching=[(0, 0)]),
                    slice_sizes=[0, 1],
                ] %0 %1
                in (%0, %2)
            "}
            .trim_end(),
        );

        // Three items: row 0 adds 10 at column 2, row 1 adds 20 at column 0, and row 2 adds 30 at column 1.
        let matrix = |rows: usize, values: Vec<f64>| {
            ArrayIrValue::Array(Array::from_elements(ArrayType::new_static(DataType::F64, [rows, 3]), &values).unwrap())
        };
        let column = |rows: usize, values: Vec<f64>| {
            ArrayIrValue::Array(Array::from_elements(ArrayType::new_static(DataType::F64, [rows, 1]), &values).unwrap())
        };
        let indices = |rows: usize, values: Vec<i32>| {
            ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::I32, [rows, 1, 1]), &values).unwrap(),
            )
        };
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Dimension(DimensionValue::new(items_type.clone(), 3).unwrap()),
                matrix(3, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                indices(3, vec![2, 0, 1]),
                column(3, vec![10.0, 20.0, 30.0]),
            ])
            .unwrap();
        assert_eq!(primal_outputs[0], matrix(3, vec![0.0, 1.0, 12.0, 23.0, 4.0, 5.0, 6.0, 37.0, 8.0]));
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs =
            vec![matrix(3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]), column(3, vec![10.0, 20.0, 30.0])];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![matrix(3, vec![1.0, 2.0, 13.0, 24.0, 5.0, 6.0, 7.0, 38.0, 9.0])]),
        );
        let mut pullback_inputs = vec![matrix(3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])];
        pullback_inputs.extend(residuals);
        assert_eq!(
            pullback.interpret(pullback_inputs),
            Ok(vec![matrix(3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]), column(3, vec![3.0, 4.0, 8.0])]),
        );

        // No items: the same programs run over empty arrays.
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Dimension(DimensionValue::new(items_type.clone(), 0).unwrap()),
                matrix(0, Vec::new()),
                indices(0, Vec::new()),
                column(0, Vec::new()),
            ])
            .unwrap();
        assert_eq!(primal_outputs[0], matrix(0, Vec::new()));
        let mut pullback_inputs = vec![matrix(0, Vec::new())];
        pullback_inputs.extend(primal_outputs.split_off(1));
        assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![matrix(0, Vec::new()), column(0, Vec::new())]));

        // A positive guaranteed minimum extent admits the ordinary size-one batching window.
        let nonempty = DimensionVariable::new("items", DimensionBounds::new(1, Some(9)).unwrap());
        let linearization = jointly_mapped_dynamic_scatter_program(nonempty, 0, add.clone())
            .linearize_with_respect_to(&[1, 3])
            .unwrap();
        assert_eq!(
            linearization.pullback().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[items, 3], %1:i32[items, 1, 1] .
                let %2:f64[items, 1] = gather [
                    dimensions=(offset=[], collapsed_slice=[1], start_index_map=[1], batching=[(0, 0)]),
                    slice_sizes=[1, 1],
                ] %0 %1
                in (%0, %2)
            "}
            .trim_end(),
        );

        // Mapped axes away from position zero are moved to the front before the same lifting applies. Item 0 is
        // column 0 of the input updated at row 2, and item 1 is column 1 updated at row 0.
        let program = jointly_mapped_dynamic_scatter_program(items, 1, add);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<items ∈ [0, 9)>, %1:f64[3, items], %2:i32[1, items, 1], %3:f64[1, items] .
                let %4:f64[items, 3] = transpose [permutation=[1, 0]] %1
                    %5:f64[items, 1] = transpose [permutation=[1, 0]] %3
                    %6:i32[items, 1, 1] = transpose [permutation=[1, 0, 2]] %2
                    %7:f64[items, 3] = scatter [
                        kind=add,
                        dimensions=(update_window=[], inserted_window=[1], scatter_to_operand=[1], \
                            operand_batching=[0], scatter_indices_batching=[0]),
                    ] %4 %6 %5
                in (%7)
            "}
            .trim_end(),
        );
        let linearization = program.linearize_with_respect_to(&[1, 3]).unwrap();
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Dimension(DimensionValue::new(items_type, 2).unwrap()),
                ArrayIrValue::Array(Array::matrix(3, 2, vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap()),
                ArrayIrValue::Array(
                    Array::from_elements(ArrayType::new_static(DataType::I32, [1, 2, 1]), &[2_i32, 0]).unwrap(),
                ),
                ArrayIrValue::Array(Array::matrix(1, 2, vec![10.0_f64, 20.0]).unwrap()),
            ])
            .unwrap();
        assert_eq!(primal_outputs[0], matrix(2, vec![1.0, 2.0, 13.0, 24.0, 5.0, 6.0]));
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![
            ArrayIrValue::Array(Array::matrix(3, 2, vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap()),
            ArrayIrValue::Array(Array::matrix(1, 2, vec![10.0_f64, 20.0]).unwrap()),
        ];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![matrix(2, vec![1.0, 2.0, 13.0, 24.0, 5.0, 6.0])]),
        );
        // Pullback contributions return to the original non-leading mapped axes, not the normalized batch layout.
        let mut pullback_inputs = vec![matrix(2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![
                ArrayIrValue::Array(Array::matrix(3, 2, vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap()),
                ArrayIrValue::Array(Array::matrix(1, 2, vec![3.0_f64, 4.0]).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_scatter_differentiation_batched_dynamic_extent_nonlinear() {
        // The nonlinear rules compose with the same batched geometry. The extremal rule reads each update's target
        // back through the shared dual gather, which takes the zero batching window over the possibly-empty extent,
        // while the product rule never builds a gather and only exercises the batched scatter itself.
        let items = DimensionVariable::new("items", DimensionBounds::new(0, Some(9)).unwrap());
        let items_type = DimensionType::new(items.clone());
        let dimensions = ScatterDimensionNumbers::new(vec![], vec![0], vec![0]);
        let matrix = |values: Vec<f64>| {
            ArrayIrValue::Array(Array::from_elements(ArrayType::new_static(DataType::F64, [3, 3]), &values).unwrap())
        };
        let column = |values: Vec<f64>| {
            ArrayIrValue::Array(Array::from_elements(ArrayType::new_static(DataType::F64, [3, 1]), &values).unwrap())
        };
        // Evaluates three items whose rows write columns 2, 0, and 1 from the input `[[1, 2, 3], [4, 5, 6],
        // [7, 8, 9]]`, then the tangent at the input tangent `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]` and update tangent
        // `[[10], [20], [30]]`, and the pullback at the cotangent `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]`.
        let evaluate = |linearization: &Linearization<ArrayIrValue<Array>, ArrayIrOperation<Array>>,
                        updates: Vec<f64>| {
            let mut primal_outputs = linearization
                .primal()
                .interpret(vec![
                    ArrayIrValue::Dimension(DimensionValue::new(items_type.clone(), 3).unwrap()),
                    matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
                    ArrayIrValue::Array(
                        Array::from_elements(ArrayType::new_static(DataType::I32, [3, 1, 1]), &[2_i32, 0, 1]).unwrap(),
                    ),
                    column(updates),
                ])
                .unwrap();
            let residuals = primal_outputs.split_off(1);
            let mut tangent_inputs =
                vec![matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]), column(vec![10.0, 20.0, 30.0])];
            tangent_inputs.extend(residuals.clone());
            let mut pullback_inputs = vec![matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])];
            pullback_inputs.extend(residuals);
            (
                primal_outputs.remove(0),
                linearization.tangent().interpret(tangent_inputs).unwrap().remove(0),
                linearization.pullback().unwrap().interpret(pullback_inputs).unwrap(),
            )
        };

        let minimum = ScatterOperation::new(dimensions.clone(), ScatterReductionKind::Min);
        let linearization = jointly_mapped_dynamic_scatter_program(items.clone(), 0, minimum)
            .linearize_with_respect_to(&[1, 3])
            .unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:dimension<items ∈ [0, 9)>, %1:f64[items, 3], %2:i32[items, 1, 1], %3:f64[items, 1] .
                let %4:f64[items, 3] = scatter [
                    kind=min,
                    dimensions=(update_window=[], inserted_window=[1], scatter_to_operand=[1], operand_batching=[0], \
                        scatter_indices_batching=[0]),
                ] %1 %2 %3
                    %5:f64[items, 3] = zero_like %1
                    %6:f64[items, 1] = one_like %3
                    %7:f64[items, 3] = scatter [
                        kind=add,
                        dimensions=(update_window=[], inserted_window=[1], scatter_to_operand=[1], \
                            operand_batching=[0], scatter_indices_batching=[0]),
                    ] %5 %2 %6
                    %8:f64[items, 3] = zero_like %7
                    %9:bool[items, 3] = compare [direction=Equal] %7 %8
                    %10:bool[items, 3] = compare [direction=Equal] %1 %4
                    %11:bool[items, 3] = one_like %10
                    %12:bool[items, 3] = select %9 %11 %10
                    %13:f64[items, 1] = gather [
                        dimensions=(offset=[], collapsed_slice=[1], start_index_map=[1], batching=[(0, 0)]),
                        slice_sizes=[0, 1],
                    ] %4 %2
                    %14:bool[items, 1] = compare [direction=Equal] %3 %13
                    %15:f64[items, 3] = convert_element_type [data_type=f64] %12
                    %16:f64[items, 1] = convert_element_type [data_type=f64] %14
                    %17:f64[items, 3] = scatter [
                        kind=add,
                        dimensions=(update_window=[], inserted_window=[1], scatter_to_operand=[1], \
                            operand_batching=[0], scatter_indices_batching=[0]),
                    ] %15 %2 %16
                    %18:f64[items, 3] = zero_like %17
                    %19:bool[items, 3] = compare [direction=Equal] %17 %18
                    %20:f64[items, 3] = one_like %17
                    %21:f64[items, 3] = select %19 %20 %17
                    %22:f64[items, 3] = one_like %21
                    %23:f64[items, 3] = div %22 %21
                in (%4, %12, %14, %2, %23)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.pullback().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[items, 3], %1:bool[items, 3], %2:bool[items, 1], %3:i32[items, 1, 1], %4:f64[items, 3] .
                let %5:f64[items, 3] = mul %4 %0
                    %6:f64[items, 1] = gather [
                        dimensions=(offset=[], collapsed_slice=[1], start_index_map=[1], batching=[(0, 0)]),
                        slice_sizes=[0, 1],
                    ] %5 %3
                    %7:f64[items, 1] = zero_like %6
                    %8:f64[items, 1] = select %2 %6 %7
                    %9:f64[items, 1] = select %2 %7 %6
                    %10:f64[items, 3] = zero_like %5
                    %11:f64[items, 3] = select %1 %5 %10
                    %12:f64[items, 3] = select %1 %10 %5
                in (%11, %8)
            "}
            .trim_end(),
        );
        // Update 0 (1) wins row 0 at column 2, the input (4) keeps row 1 at column 0, and update 2 (5) wins row 2 at
        // column 1.
        assert_eq!(
            evaluate(&linearization, vec![1.0, 20.0, 5.0]),
            (
                matrix(vec![1.0, 2.0, 1.0, 4.0, 5.0, 6.0, 7.0, 5.0, 9.0]),
                matrix(vec![1.0, 2.0, 10.0, 4.0, 5.0, 6.0, 7.0, 30.0, 9.0]),
                vec![matrix(vec![1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 7.0, 0.0, 9.0]), column(vec![3.0, 0.0, 8.0])],
            ),
        );

        // Multiplication scales one element per row, so its input coefficient is the batched scatter of the updates
        // into ones and its update cotangent gathers the input times the cotangent.
        let multiply = ScatterOperation::new(dimensions, ScatterReductionKind::Mul).with_unique_indices(true);
        let linearization = jointly_mapped_dynamic_scatter_program(items, 0, multiply)
            .linearize_with_respect_to(&[1, 3])
            .unwrap();
        assert_eq!(
            evaluate(&linearization, vec![10.0, 20.0, 30.0]),
            (
                matrix(vec![1.0, 2.0, 30.0, 80.0, 5.0, 6.0, 7.0, 240.0, 9.0]),
                matrix(vec![1.0, 2.0, 60.0, 160.0, 5.0, 6.0, 7.0, 480.0, 9.0]),
                vec![matrix(vec![1.0, 2.0, 30.0, 80.0, 5.0, 6.0, 7.0, 240.0, 9.0]), column(vec![9.0, 16.0, 64.0])],
            ),
        );
    }

    #[test]
    fn test_scatter_differentiation_dynamic_inserted_axis() {
        // Gather collapses an inserted axis through a window of exactly one element, which an axis that may be empty
        // at runtime cannot provide, so no dual gather exists for such a scatter. The additive rule only needs it in
        // its transpose, while the extremal rule needs it to read back the update targets during linearization.
        let dimensions = ScatterDimensionNumbers::new(vec![], vec![0], vec![0]);
        let indices = Array::matrix(2, 1, vec![1_i32, 3]).unwrap();
        let updates_type = ArrayType::new_static(DataType::F64, [2]);
        let items = DimensionVariable::new("items", DimensionBounds::new(0, Some(9)).unwrap());
        let possibly_empty = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(items)]));
        let expected = DifferentiationError::Program(ProgramError::UnsupportedOperation {
            message: format!(
                "`{SCATTER_OPERATION_NAME}` differentiation requires inserted window axis 0 to have a nonzero minimum \
                 extent, because its dual gather collapses that axis through a one-element window"
            ),
        });
        let linearization = constant_index_scatter_program(
            ScatterOperation::new(dimensions.clone(), ScatterReductionKind::Add),
            possibly_empty.clone(),
            indices.clone(),
            updates_type.clone(),
        )
        .linearize()
        .unwrap();
        assert_eq!(linearization.pullback().unwrap_err(), expected);
        assert_eq!(
            constant_index_scatter_program(
                ScatterOperation::new(dimensions.clone(), ScatterReductionKind::Min),
                possibly_empty,
                indices.clone(),
                updates_type.clone(),
            )
            .linearize()
            .unwrap_err(),
            expected,
        );

        // A positive guaranteed minimum extent admits the one-element collapsed window.
        let nonempty = DimensionVariable::new("items", DimensionBounds::new(4, Some(9)).unwrap());
        let nonempty_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(nonempty)]));
        let linearization = constant_index_scatter_program(
            ScatterOperation::new(dimensions, ScatterReductionKind::Min),
            nonempty_type.clone(),
            indices,
            updates_type.clone(),
        )
        .linearize()
        .unwrap();
        assert_eq!(
            linearization.pullback().unwrap().output_types(),
            vec![
                ArrayIrType::Array(nonempty_type.cotangent().unwrap()),
                ArrayIrType::Array(updates_type.cotangent().unwrap()),
            ],
        );
    }

    #[test]
    fn test_scatter_differentiation_reduced_state() {
        // Reduced input and updates promise one value across the reduced axis, so indices that vary across it are
        // rejected up front rather than by the gathers that every derivative stages.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let reduced = Sharding::replicated(mesh.clone(), 1).with_reduced_axes(["x"]).unwrap();
        let input_type = ArrayType::new_static(DataType::F64, [4]).with_sharding(reduced.clone()).unwrap();
        let updates_type = ArrayType::new_static(DataType::F64, [2]).with_sharding(reduced).unwrap();
        let sharded_indices_type = ArrayType::new_static(DataType::I32, [2, 1])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated])
                    .unwrap(),
            )
            .unwrap();
        let varying_indices_type = ArrayType::new_static(DataType::I32, [2, 1])
            .with_sharding(Sharding::replicated(mesh, 2).with_varying_manual_axes(["m"]).unwrap())
            .unwrap();
        let dimensions = ScatterDimensionNumbers::new(vec![], vec![0], vec![0]);
        let add = ScatterOperation::new(dimensions.clone(), ScatterReductionKind::Add);
        let minimum = ScatterOperation::new(dimensions, ScatterReductionKind::Min);
        let expected = Err(TypeError::invalid(format!(
            "`{SCATTER_OPERATION_NAME}` reduction-state inputs require replicated, invariant indices"
        ))
        .into());
        assert_eq!(
            input_type.scatter(&sharded_indices_type, &updates_type, add.dimensions(), add.kind(), add.options()),
            expected
        );
        assert_eq!(
            input_type.scatter(
                &sharded_indices_type,
                &updates_type,
                minimum.dimensions(),
                minimum.kind(),
                minimum.options()
            ),
            expected
        );
        assert_eq!(
            input_type.scatter(&varying_indices_type, &updates_type, add.dimensions(), add.kind(), add.options()),
            expected
        );
        assert_eq!(
            input_type.scatter(
                &varying_indices_type,
                &updates_type,
                minimum.dimensions(),
                minimum.kind(),
                minimum.options()
            ),
            expected
        );

        // Replicated, invariant indices keep reduced state differentiable end to end: the primal, the tangent, and
        // the pullback all run, with cotangents taking the dual (unreduced) state.
        let indices = Array::matrix(2, 1, vec![1_i32, 3]).unwrap();
        let input = ArrayIrValue::Array(Array::from_elements(input_type.clone(), &[1.0_f64, 2.0, 3.0, 4.0]).unwrap());
        let updates = ArrayIrValue::Array(Array::from_elements(updates_type.clone(), &[10.0_f64, 20.0]).unwrap());
        let input_tangent = ArrayIrValue::Array(
            Array::from_elements(input_type.tangent().unwrap(), &[1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
        );
        let updates_tangent =
            ArrayIrValue::Array(Array::from_elements(updates_type.tangent().unwrap(), &[5.0_f64, 6.0]).unwrap());
        let cotangent = ArrayIrValue::Array(
            Array::from_elements(input_type.cotangent().unwrap(), &[10.0_f64, 20.0, 30.0, 40.0]).unwrap(),
        );
        let expected_cotangent_types = vec![
            ArrayIrType::Array(input_type.cotangent().unwrap()),
            ArrayIrType::Array(updates_type.cotangent().unwrap()),
        ];

        let linearization =
            constant_index_scatter_program(add, input_type.clone(), indices.clone(), updates_type.clone())
                .linearize()
                .unwrap();
        let mut primal_outputs = linearization.primal().interpret(vec![input.clone(), updates.clone()]).unwrap();
        assert_eq!(
            primal_outputs[0],
            ArrayIrValue::Array(Array::from_elements(input_type.clone(), &[1.0_f64, 12.0, 3.0, 24.0]).unwrap()),
        );
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![input_tangent.clone(), updates_tangent.clone()];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(
                Array::from_elements(input_type.tangent().unwrap(), &[1.0_f64, 7.0, 3.0, 10.0]).unwrap(),
            )]),
        );
        let pullback = linearization.pullback().unwrap();
        assert_eq!(pullback.output_types(), expected_cotangent_types);
        let mut pullback_inputs = vec![cotangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(
            pullback.interpret(pullback_inputs),
            Ok(vec![
                cotangent.clone(),
                ArrayIrValue::Array(
                    Array::from_elements(updates_type.cotangent().unwrap(), &[20.0_f64, 40.0]).unwrap(),
                ),
            ]),
        );

        // The minimum keeps the input at both targets (2 < 10 and 4 < 20), so the updates receive nothing.
        let linearization = constant_index_scatter_program(minimum, input_type.clone(), indices, updates_type.clone())
            .linearize()
            .unwrap();
        let mut primal_outputs = linearization.primal().interpret(vec![input.clone(), updates]).unwrap();
        assert_eq!(primal_outputs[0], input);
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![input_tangent.clone(), updates_tangent];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(linearization.tangent().interpret(tangent_inputs), Ok(vec![input_tangent]));
        let pullback = linearization.pullback().unwrap();
        assert_eq!(pullback.output_types(), expected_cotangent_types);
        let mut pullback_inputs = vec![cotangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(
            pullback.interpret(pullback_inputs),
            Ok(vec![
                cotangent,
                ArrayIrValue::Array(Array::from_elements(updates_type.cotangent().unwrap(), &[0.0_f64, 0.0]).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_scatter_differentiation_disconnected_dynamic_input() {
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
    fn test_scatter_transposition() {
        // Empty base arrays leave every update cotangent zero. This includes both nonempty queries targeting an
        // empty selected axis and a size-zero paired batching axis, whose dual gather cannot use a size-one window.
        for kind in [ScatterReductionKind::Add, ScatterReductionKind::Overwrite] {
            check_operation_transposition!(
                @exact,
                operation = ScatterOperation::new(
                    ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), kind,
                ).with_unique_indices(true).with_mode(ScatterMode::Clip),
                cases = [{
                    inputs = [
                        (@linear(type = ArrayType::new_static(DataType::F64, [0]))),
                        (@known, Array::matrix(1, 1, vec![0_i32]).unwrap()),
                        (@linear(type = ArrayType::new_static(DataType::F64, [1]))),
                    ],
                    output_cotangents = [Array::vector(Vec::<f64>::new()).unwrap()],
                    input_cotangents = [Array::vector(Vec::<f64>::new()).unwrap(), Array::vector(vec![0.0]).unwrap()],
                }],
            );
            check_operation_transposition!(
                @exact,
                operation = ScatterOperation::new(
                    ScatterDimensionNumbers::new(vec![], vec![1], vec![1])
                        .with_batching_dimensions(vec![0], vec![0]), kind,
                ).with_unique_indices(true),
                cases = [{
                    inputs = [
                        (@linear(type = ArrayType::new_static(DataType::F64, [0, 3]))),
                        (@known, Array::from_elements(
                            ArrayType::new_static(DataType::I32, [0, 1, 1]),
                            &[] as &[i32],
                        ).unwrap()),
                        (@linear(type = ArrayType::new_static(DataType::F64, [0, 1]))),
                    ],
                    output_cotangents = [Array::matrix(0, 3, Vec::<f64>::new()).unwrap()],
                    input_cotangents = [
                        Array::matrix(0, 3, Vec::<f64>::new()).unwrap(),
                        Array::matrix(0, 1, Vec::<f64>::new()).unwrap(),
                    ],
                }],
            );
        }

        // With unique replacement windows, the input cotangent is erased at the written locations.
        check_operation_transposition!(
            @exact,
            operation = ScatterOperation::new(
                ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Overwrite,
            ).with_unique_indices(true).with_mode(ScatterMode::Drop),
            cases = [{
                inputs = [
                    (@linear(type = ArrayType::new_static(DataType::F64, [4]))),
                    (@known, Array::matrix(3, 1, vec![-1_i32, 1, 4]).unwrap()),
                    (@linear(type = ArrayType::new_static(DataType::F64, [3]))),
                ],
                output_cotangents = [Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap()],
                input_cotangents = [
                    Array::vector(vec![1.0, 0.0, 3.0, 4.0]).unwrap(),
                    Array::vector(vec![0.0, 2.0, 0.0]).unwrap(),
                ],
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
                    (@known, Array::matrix(2, 1, vec![1_i32, 3]).unwrap()),
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
            ).with_mode(ScatterMode::Drop),
            cases = [{
                inputs = [
                    (@linear(type = ArrayType::new_static(DataType::F64, [4]))),
                    (@known, Array::matrix(3, 1, vec![-1_i32, 1, 4]).unwrap()),
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
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![4.into()])).with_memory(Memory::Host { pinned: true });
        let update_type = ArrayType::new(DataType::F64, Shape::new(vec![2.into()]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![8])))
            .with_memory(Memory::Host { pinned: true });
        let indices = Array::from_elements::<i32>(
            ArrayType::new_static(DataType::I32, [2, 1]).with_memory(Memory::Host { pinned: true }),
            &[1, 3],
        )
        .unwrap();
        check_operation_transposition!(
            @exact,
            operation = ScatterOperation::new(
                ScatterDimensionNumbers::new(vec![], vec![0], vec![0]),
                ScatterReductionKind::Add,
            ),
            cases = [{
                inputs = [
                    (@linear(type = input_type.clone())),
                    (@known, indices),
                    (@linear(type = update_type.clone())),
                ],
                output_cotangents = [Array::from_elements::<f64>(input_type.clone(), &[1.0, 2.0, 3.0, 4.0]).unwrap()],
                input_cotangents = [
                    Array::from_elements::<f64>(input_type, &[1.0, 2.0, 3.0, 4.0]).unwrap(),
                    Array::from_elements::<f64>(update_type, &[2.0, 4.0]).unwrap(),
                ],
            }],
        );
    }

    #[test]
    fn test_scatter_transposition_zero_cotangent() {
        // A structural-zero output cotangent contributes nothing for every combiner, including the ones without a
        // transpose: the rule returns before the combiner check and leaves every accumulator at its structural-zero
        // default. The same holds when no cotangent is needed.
        let dimensions = ScatterDimensionNumbers::new(vec![], vec![0], vec![0]);
        let input_type = ArrayType::new_static(DataType::F64, [4]);
        let updates_type = ArrayType::new_static(DataType::F64, [2]);
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let mut transpose = TranspositionContext::new(context.clone());
        let indices = context.lift(Array::matrix(2, 1, vec![1_i32, 3]).unwrap()).unwrap();
        let inputs = [
            PartialValue::Unknown(input_type.clone()),
            PartialValue::Known(indices),
            PartialValue::Unknown(updates_type.clone()),
        ];
        let accumulators = transpose.cotangent_accumulators(&inputs, &[]).unwrap();
        let zero_outputs = [MaybeZero::Zero(input_type.cotangent().unwrap())];
        for kind in [
            ScatterReductionKind::Overwrite,
            ScatterReductionKind::Add,
            ScatterReductionKind::Mul,
            ScatterReductionKind::Min,
            ScatterReductionKind::Max,
        ] {
            ScatterOperation::new(dimensions.clone(), kind)
                .transpose(&mut transpose, &EmptyRegionDriver, &inputs, &zero_outputs, &accumulators)
                .unwrap();
        }
        let cotangents = transpose.take_cotangents(&accumulators).unwrap();
        assert_eq!(cotangents.len(), 3);
        assert!(cotangents[0].is_zero());
        assert_eq!(cotangents[0].r#type().as_ref(), &input_type.cotangent().unwrap());
        assert!(cotangents[2].is_zero());
        assert_eq!(cotangents[2].r#type().as_ref(), &updates_type.cotangent().unwrap());
        assert!(context.builder().borrow().instructions().is_empty());

        let operation = ScatterOperation::new(dimensions, ScatterReductionKind::Add);
        let unneeded = transpose.cotangent_accumulators(&inputs, &[false, false, false]).unwrap();
        let outputs = [MaybeZero::Value(context.input(input_type.cotangent().unwrap()))];
        operation.transpose(&mut transpose, &EmptyRegionDriver, &inputs, &outputs, &unneeded).unwrap();
        assert!(context.builder().borrow().instructions().is_empty());

        // Arity is validated before any cotangent is inspected.
        assert_eq!(
            operation
                .transpose(&mut transpose, &EmptyRegionDriver, &inputs[..2], &outputs, &accumulators)
                .unwrap_err(),
            DifferentiationError::Program(ProgramError::InvalidInputCount { expected: 3, actual: 2 }),
        );
        assert_eq!(
            operation.transpose(&mut transpose, &EmptyRegionDriver, &inputs, &[], &accumulators).unwrap_err(),
            DifferentiationError::Program(ProgramError::InvalidOutputCount { expected: 1, actual: 0 }),
        );
        assert_eq!(
            operation
                .transpose(&mut transpose, &EmptyRegionDriver, &inputs, &outputs, &accumulators[..2])
                .unwrap_err(),
            DifferentiationError::InvalidAccumulatorCount { expected: 3, actual: 2 },
        );
    }

    #[test]
    fn test_scatter_transposition_unsupported() {
        // A nonzero cotangent through a combiner without a direct linear rule is rejected, naming the combiner.
        let dimensions = ScatterDimensionNumbers::new(vec![], vec![0], vec![0]);
        let input_type = ArrayType::new_static(DataType::F64, [4]);
        let updates_type = ArrayType::new_static(DataType::F64, [2]);
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let mut transpose = TranspositionContext::new(context.clone());
        let indices = context.lift(Array::matrix(2, 1, vec![1_i32, 3]).unwrap()).unwrap();
        let inputs = [
            PartialValue::Unknown(input_type.clone()),
            PartialValue::Known(indices),
            PartialValue::Unknown(updates_type.clone()),
        ];
        let accumulators = transpose.cotangent_accumulators(&inputs, &[]).unwrap();
        let outputs = [MaybeZero::Value(context.input(input_type.cotangent().unwrap()))];
        let unsupported = |kind: ScatterReductionKind| {
            DifferentiationError::Program(ProgramError::UnsupportedOperation {
                message: format!(
                    "transposition of `{SCATTER_OPERATION_NAME}` with the `{kind}` combiner requires scatter-add or \
                     unique-index overwrite"
                ),
            })
        };
        assert_eq!(
            ScatterOperation::new(dimensions.clone(), ScatterReductionKind::Overwrite)
                .transpose(&mut transpose, &EmptyRegionDriver, &inputs, &outputs, &accumulators)
                .unwrap_err(),
            unsupported(ScatterReductionKind::Overwrite),
        );
        assert_eq!(
            ScatterOperation::new(dimensions.clone(), ScatterReductionKind::Mul)
                .with_unique_indices(true)
                .transpose(&mut transpose, &EmptyRegionDriver, &inputs, &outputs, &accumulators)
                .unwrap_err(),
            unsupported(ScatterReductionKind::Mul),
        );
        assert_eq!(
            ScatterOperation::new(dimensions.clone(), ScatterReductionKind::Min)
                .transpose(&mut transpose, &EmptyRegionDriver, &inputs, &outputs, &accumulators)
                .unwrap_err(),
            unsupported(ScatterReductionKind::Min),
        );
        assert_eq!(
            ScatterOperation::new(dimensions.clone(), ScatterReductionKind::Max)
                .transpose(&mut transpose, &EmptyRegionDriver, &inputs, &outputs, &accumulators)
                .unwrap_err(),
            unsupported(ScatterReductionKind::Max),
        );

        // The indices are the known input of every valid pullback; both linear rules reject a linear index input.
        let unknown_indices = [
            PartialValue::Unknown(input_type.clone()),
            PartialValue::Unknown(ArrayType::new_static(DataType::I32, [2, 1])),
            PartialValue::Unknown(updates_type),
        ];
        let unknown_accumulators = transpose.cotangent_accumulators(&unknown_indices, &[]).unwrap();
        let expected = DifferentiationError::Program(ProgramError::Type(TypeError::invalid(format!(
            "`{SCATTER_OPERATION_NAME}` transpose requires known indices"
        ))));
        assert_eq!(
            ScatterOperation::new(dimensions.clone(), ScatterReductionKind::Add)
                .transpose(&mut transpose, &EmptyRegionDriver, &unknown_indices, &outputs, &unknown_accumulators)
                .unwrap_err(),
            expected,
        );
        assert_eq!(
            ScatterOperation::new(dimensions.clone(), ScatterReductionKind::Overwrite)
                .with_unique_indices(true)
                .transpose(&mut transpose, &EmptyRegionDriver, &unknown_indices, &outputs, &unknown_accumulators)
                .unwrap_err(),
            expected,
        );

        // The dual gather needs static update windows.
        let window = DimensionVariable::new("window", DimensionBounds::new(1, Some(2)).unwrap());
        let matrix_type = ArrayType::new_static(DataType::F64, [3, 2]);
        let dynamic_window = [
            PartialValue::Unknown(matrix_type.clone()),
            PartialValue::Known(context.lift(Array::matrix(2, 1, vec![0_i32, 2]).unwrap()).unwrap()),
            PartialValue::Unknown(ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(window)]),
            )),
        ];
        let dynamic_accumulators = transpose.cotangent_accumulators(&dynamic_window, &[]).unwrap();
        let matrix_outputs = [MaybeZero::Value(context.input(matrix_type.cotangent().unwrap()))];
        assert_eq!(
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![1], vec![0], vec![0]), ScatterReductionKind::Add)
                .transpose(&mut transpose, &EmptyRegionDriver, &dynamic_window, &matrix_outputs, &dynamic_accumulators)
                .unwrap_err(),
            DifferentiationError::Program(ProgramError::UnsupportedOperation {
                message: format!(
                    "`{SCATTER_OPERATION_NAME}` differentiation requires a static update window on axis 1 but its \
                     extent is `window`"
                ),
            }),
        );
    }

    #[test]
    fn test_scatter_scatter_axis() {
        let input = Array::matrix(2, 3, vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
        let indices = Array::vector(vec![2_i32, 0]).unwrap();
        let updates = Array::matrix(2, 2, vec![10_i32, 20, 30, 40]).unwrap();
        assert_eq!(
            input.scatter_axis(&indices, &updates, -1, ScatterReductionKind::Add, ScatterMode::Clip),
            Array::matrix(2, 3, vec![21_i32, 2, 13, 44, 5, 36]),
        );
        assert_eq!(
            input.scatter_axis(
                &Array::scalar(0_i32).unwrap(),
                &Array::vector(vec![7_i32, 8, 9]).unwrap(),
                0,
                ScatterReductionKind::Overwrite,
                ScatterMode::Clip,
            ),
            Array::matrix(2, 3, vec![7_i32, 8, 9, 4, 5, 6]),
        );
        assert_eq!(
            input.scatter_axis(
                &Array::scalar(-1_i32).unwrap(),
                &Array::vector(vec![7_i32, 8, 9]).unwrap(),
                0,
                ScatterReductionKind::Overwrite,
                ScatterMode::Drop,
            ),
            Ok(input.clone()),
        );
        let mismatched = input.scatter_axis(
            &indices,
            &Array::matrix(1, 2, vec![1_i32, 2]).unwrap(),
            1,
            ScatterReductionKind::Add,
            ScatterMode::Clip,
        );
        assert_eq!(
            mismatched,
            Err(TypeError::invalid("`scatter_axis` updates shape must be `[2, 2]` but got `[1, 2]`").into()),
        );

        // The update replaces the selected axis, so its window shape does not depend on that input extent.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(6)).unwrap());
        let input_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let (output_type, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |(input, indices, updates)| {
                input.scatter_axis(&indices, &updates, 0, ScatterReductionKind::Add, ScatterMode::Clip)
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
                        input.dynamic_scatter_axis(&queries, &updates, 0, ScatterReductionKind::Add, ScatterMode::Clip)
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

        // One retained query shape specializes at both lengths, and duplicate indices accumulate all updates.
        let query = DimensionVariable::new("queries", DimensionBounds::new(2, Some(4)).unwrap());
        let indices_type = ArrayType::new(DataType::I32, Shape::new(vec![query.clone().into()]));
        let updates_type = ArrayType::new(DataType::F64, Shape::new(vec![query.into()]));
        let input_type = ArrayType::new_static(DataType::F64, [4]);
        let (output_type, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(input, indices, updates)| {
                input.dynamic_scatter_axis(&indices, &updates, 0, ScatterReductionKind::Add, ScatterMode::Clip)
            },
            (ArrayIrType::from(input_type.clone()), ArrayIrType::from(indices_type), ArrayIrType::from(updates_type)),
        )
        .unwrap();
        assert_eq!(output_type, ArrayIrType::from(input_type));
        let input = ArrayIrValue::Array(Array::vector(vec![10_f64, 20., 30., 40.]).unwrap());
        assert_eq!(
            program.interpret((
                input.clone(),
                ArrayIrValue::Array(Array::vector(vec![1_i32, 1]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![2_f64, 2.]).unwrap()),
            )),
            Ok(ArrayIrValue::Array(Array::vector(vec![10_f64, 24., 30., 40.]).unwrap())),
        );
        assert_eq!(
            program.interpret((
                input,
                ArrayIrValue::Array(Array::vector(vec![1_i32, 1, 1]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![2_f64, 2., 2.]).unwrap()),
            )),
            Ok(ArrayIrValue::Array(Array::vector(vec![10_f64, 26., 30., 40.]).unwrap())),
        );

        let input = ArrayIrValue::Array(Array::vector(vec![10_i32, 20, 30]).unwrap());
        let indices = ArrayIrValue::Array(Array::vector(vec![0_i32, 2]).unwrap());
        let updates = ArrayIrValue::Array(Array::vector(vec![1_i32]).unwrap());
        assert_eq!(
            input.dynamic_scatter_axis(&indices, &updates, 0, ScatterReductionKind::Add, ScatterMode::Clip),
            Err(TypeError::invalid("`dynamic_scatter_axis` updates shape must be `[2]` but got `[1]`").into()),
        );
    }
}
