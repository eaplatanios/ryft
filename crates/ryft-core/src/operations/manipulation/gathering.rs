use std::collections::BTreeSet;
use std::fmt::Display;
use std::sync::Arc;

use crate::arrays::{
    Array, ArrayAddressing, ArrayBatch, ArrayBatchingPolicy, ArrayElement, ArrayExtentBatchingPolicy, ArrayIrType,
    ArrayIrValue, ArrayType, DataType, Dimension, LinearResiduals, LogicalMesh, MeshAxisType, Shape, Sharding,
    ShardingDimension, i1, i2, i4, u1, u2, u4,
};
use crate::axes::Axis;
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain, EagerContext, ProjectedContext, StagingContext};
use crate::differentiation::{
    CotangentAccumulator, DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDriver,
    DifferentiationDual, DifferentiationError, DifferentiationPolicy, MemberDifferentiableOperation,
    TransposableOperation, TranspositionContext, TranspositionDriver, jvp_projected_operation,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, dispatch_on_array_element_type};
use crate::operations::constants::constant::DimensionConstant;
use crate::operations::constants::zero::{DynamicZero, Zero, ZeroOperation};
use crate::operations::differentiation::linear_call::LinearCallOperation;
use crate::operations::dimensions::dimension_size::{DimensionSize, DimensionSizeOperation};
use crate::operations::manipulation::broadcasting::{BroadcastOperation, DynamicBroadcast};
use crate::operations::manipulation::reshaping::{Reshape, lift_output_sharding_for_leading_batch_axis};
use crate::operations::manipulation::scattering::{ScatterDimensionNumbers, ScatterOperation, ScatterReductionKind};
use crate::operations::manipulation::transposition::Transpose;
use crate::operations::math::add::AddOperation;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, OperationProjection, ProgramError, RegionInterface, TypeError, Typed,
    Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

// TODO(eaplatanios): Review this.

/// Out-of-bounds index handling for [`gather`](Gather) and [`scatter`](super::scattering::Scatter). The mode does not
/// affect the output [`Type`](crate::programs::types::Type)—only how a start index that would read or write outside
/// the input is treated at execution time. It is shared by both operations; the scatter combiner kind lives in
/// [`super::scattering`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum GatherScatterMode {
    /// The caller promises every index is in bounds; out-of-bounds behavior is undefined (and gradients are wrong if
    /// the promise is violated). This is the default and lowers directly to the bare StableHLO operation.
    #[default]
    PromiseInBounds,

    /// Each start index is clamped so the whole window stays in bounds.
    Clip,

    /// A window that falls partly out of bounds is filled by gather and discarded by scatter. Gather uses its
    /// explicit scalar fill when supplied; otherwise it uses NaN for floating-point and complex values, the minimum
    /// signed integer, the maximum unsigned integer, or `true` for Booleans.
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

/// Specification of how the index input and the sliced windows map onto the input and output axes of a
/// [`gather`](Gather), following StableHLO's [`gather`](https://openxla.org/stablehlo/spec#gather) dimension numbers.
///
/// The index vector dimension is implicit and always the last axis of the indices input: the
/// indices input has shape `[batch..., index_vector]`, where each length-`index_vector` slice is one start-index
/// vector whose components map onto input axes through [`start_index_map`](Self::start_index_map). To gather with a
/// scalar index per query, give the indices a trailing size-1 axis.
///
/// The output rank is `offset_dimensions.len() + indices.rank() - 1`. Each output axis named in
/// [`offset_dimensions`](Self::offset_dimensions) carries one sliced window axis (in input-axis order, skipping the
/// collapsed and batching axes); the remaining output axes carry the indices' batch axes in order.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct GatherDimensionNumbers {
    /// Output axes that hold the sliced window (the "offset" axes), in ascending order. Their count equals the number
    /// of input axes that are neither collapsed nor batching.
    offset_dimensions: Vec<usize>,

    /// Input axes whose slice size is `1` and that are removed from the output, in ascending order.
    collapsed_slice_dimensions: Vec<usize>,

    /// For each component of a start-index vector (the last axis of the indices input), the input axis it indexes
    /// into. Its length equals the extent of the indices' index vector dimension.
    start_index_map: Vec<usize>,

    /// Input axes batched against [`start_indices_batching_dimensions`](Self::start_indices_batching_dimensions),
    /// aligned 1:1, in ascending order. Each has slice size at most `1`.
    operand_batching_dimensions: Vec<usize>,

    /// Indices axes (other than the index vector dimension) that align 1:1 with
    /// [`operand_batching_dimensions`](Self::operand_batching_dimensions).
    start_indices_batching_dimensions: Vec<usize>,
}

impl GatherDimensionNumbers {
    /// Creates gather dimension numbers from explicit axis lists. The batching axis lists default to empty; use
    /// [`with_batching_dimensions`](Self::with_batching_dimensions) to set them.
    ///
    /// # Parameters
    ///
    ///   - `offset_dimensions`: Sorted output positions occupied by the retained window axes.
    ///   - `collapsed_slice_dimensions`: Sorted input axes with size-one windows that are omitted from the result.
    ///   - `start_index_map`: Input axis addressed by each component of the trailing index vector, in component order.
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

    /// Returns the output offset axes.
    #[inline]
    pub fn offset_dimensions(&self) -> &[usize] {
        &self.offset_dimensions
    }

    /// Returns the collapsed (size-1, removed) input axes.
    #[inline]
    pub fn collapsed_slice_dimensions(&self) -> &[usize] {
        &self.collapsed_slice_dimensions
    }

    /// Returns the start-index-to-input-axis map.
    #[inline]
    pub fn start_index_map(&self) -> &[usize] {
        &self.start_index_map
    }

    /// Returns the input batching axes.
    #[inline]
    pub fn operand_batching_dimensions(&self) -> &[usize] {
        &self.operand_batching_dimensions
    }

    /// Returns the indices batching axes.
    #[inline]
    pub fn start_indices_batching_dimensions(&self) -> &[usize] {
        &self.start_indices_batching_dimensions
    }
    /// Attaches the input/indices batching axis pair (aligned 1:1).
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

/// Canonical operation name for [`GatherOperation`].
pub const GATHER_OPERATION_NAME: &str = "gather";

/// [`Operation`] that reads slices ("windows") out of an input at positions named by an integer index input,
/// assembling them into a new array. Refer to the documentation of [`Gather`] for the full semantics.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GatherOperation {
    /// Dimension numbers mapping the index input and sliced windows onto the input and output axes.
    dimensions: GatherDimensionNumbers,

    /// Dimension of the sliced window along each input axis (length equals the input rank).
    slice_sizes: Vec<usize>,

    /// Out-of-bounds index handling.
    mode: GatherScatterMode,

    /// Optional scalar fill encoded canonically, preserving exact equality and hashing even for NaNs.
    fill_value: Option<(DataType, Vec<u8>)>,

    /// Whether the caller guarantees the index vectors are sorted (a lowering hint only).
    indices_are_sorted: bool,

    /// Whether the caller guarantees the gathered windows do not overlap (a lowering hint only).
    unique_indices: bool,

    /// Optional requested output [`Sharding`], used when the inferred placement is ambiguous (see
    /// [`Self::with_output_sharding`]).
    output_sharding: Option<Sharding>,
}

impl GatherOperation {
    /// Creates a new [`GatherOperation`] with the provided dimension numbers and per-input-axis slice sizes. The
    /// mode defaults to [`GatherScatterMode::PromiseInBounds`] and both index hints default to `false`; use the
    /// chained `with_*` builders to override them.
    ///
    /// # Parameters
    ///
    ///   - `dimensions`: Mapping from index components and window axes to input and output axes.
    ///   - `slice_sizes`: Nonnegative window size for each input axis. Collapsed axes have size one; batching axes
    ///     have size at most one. Each size must fit its input extent.
    #[inline]
    pub fn new(dimensions: GatherDimensionNumbers, slice_sizes: Vec<usize>) -> Self {
        Self {
            dimensions,
            slice_sizes,
            mode: GatherScatterMode::PromiseInBounds,
            fill_value: None,
            indices_are_sorted: false,
            unique_indices: false,
            output_sharding: None,
        }
    }

    /// Returns the dimension numbers.
    #[inline]
    pub fn dimensions(&self) -> &GatherDimensionNumbers {
        &self.dimensions
    }

    /// Returns the per-input-axis slice sizes.
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

    /// Returns the explicit scalar fill constant without memory or layout annotations, if one was supplied.
    pub fn fill_value(&self) -> Option<Array> {
        self.fill_value.as_ref().map(|(data_type, bytes)| {
            Array::new_unchecked(ArrayType::scalar(*data_type), std::sync::Arc::new(bytes.clone()))
        })
    }

    /// Returns the requested output sharding, if any.
    #[inline]
    pub fn output_sharding(&self) -> Option<&Sharding> {
        self.output_sharding.as_ref()
    }
    /// Uses a scalar constant for windows outside the input in [`GatherScatterMode::FillOrDrop`] mode.
    /// The scalar must have the input element data type; its memory and layout metadata are discarded. Other modes
    /// ignore this value. Without an override, floating-point and complex inputs use NaN, signed integers use their
    /// minimum value, unsigned integers use their maximum value, and Booleans use `true`.
    ///
    /// # Parameters
    ///
    ///   - `fill_value`: Rank-zero array containing the replacement element in its exact data type.
    pub fn with_fill_value(mut self, fill_value: Array) -> Result<Self, TypeError> {
        let r#type = fill_value.r#type();
        if r#type.rank() != 0 || !(r#type.data_type().is_numeric() || r#type.data_type().is_boolean()) {
            return Err(TypeError::invalid("`gather` fill value must be a numeric or Boolean scalar"));
        }
        let addressing =
            ArrayAddressing::new(r#type.into_owned()).map_err(|error| TypeError::invalid(error.to_string()))?;
        self.fill_value = Some((
            fill_value.r#type().data_type(),
            fill_value.storage_bytes()[addressing.byte_range_for_flat_index(0)].to_vec(),
        ));
        Ok(self)
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

    /// Requests `output_sharding` for the result. The gather sharding rule replicates the input axes named by
    /// [`GatherDimensionNumbers::start_index_map`] (and the index vector axis); when that leaves the output placement
    /// ambiguous — for example because a sliced input axis is sharded over an explicit mesh axis — a requested
    /// output sharding resolves it, bypassing inference. This mirrors `dot`/`reduce`'s `with_output_sharding`.
    #[inline]
    pub fn with_output_sharding(mut self, output_sharding: impl Into<Option<Sharding>>) -> Self {
        self.output_sharding = output_sharding.into();
        self
    }
    /// Resolves the scalar used for out-of-bounds windows in the requested input data type.
    /// Floating formats without NaN use their normal NaN conversion result. Complex NaN has a zero imaginary part.
    /// This function is shared by eager interpretation and native lowering so both use identical element encodings.
    ///
    /// # Parameters
    ///
    ///   - `data_type`: Element data type of the gathered input.
    pub fn resolved_fill_value(&self, data_type: DataType) -> Result<Array, ProgramError> {
        if let Some(value) = self.fill_value() {
            if value.r#type().data_type() != data_type {
                return Err(TypeError::invalid(format!(
                    "`gather` fill data type `{}` does not match input data type `{data_type}`",
                    value.r#type().data_type()
                ))
                .into());
            }
            return Ok(value);
        }
        dispatch_on_array_element_type!(data_type, |Element| {
            let element = if data_type.is_signed() {
                Element::max_identity()
            } else if data_type.is_unsigned() || data_type.is_boolean() {
                Element::min_identity()
            } else {
                Element::from_real(f64::NAN)?
            };
            Array::scalar(element)
        })
    }
}

impl Display for GatherOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for GatherOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        GATHER_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("region", region_interfaces, 0, TypeError);
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
            if let Some(fill_value) = self.fill_value() {
                operation.field("fill_value", &fill_value)?;
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

// Partial evaluation defers to the default fold-or-residualize behavior of
// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for GatherOperation where
    C::Operation: From<GatherOperation>
{
}

// Batching lifts the dimension numbers into one gather. A mapped input alone becomes a full-window offset axis;
// mapped indices alone add an output batch axis; jointly mapped inputs gain a paired input/indices batching axis.
impl<C, P: ArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>> for GatherOperation
where
    C: Context<Type = ArrayType>,
    C::Value: Transpose,
    GatherOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        check_count!("input", inputs, 2, ProgramError);
        if inputs.iter().any(|input| !input.ragged_axes().is_empty()) {
            return Err(BatchingError::UnsupportedOperation {
                message: format!("`{GATHER_OPERATION_NAME}` does not support bounded ragged array inputs"),
            });
        }
        let mapped_input = inputs[0].batch_axis_position().is_some();
        let mapped_indices = inputs[1].batch_axis_position().is_some();
        if !mapped_input && !mapped_indices {
            return Ok(self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()])?.into());
        }

        // Put each mapped axis first so all three lifting cases produce one leading mapped output axis. Unlike
        // expanding one gather per item, this also handles empty batches without constructing a nonempty zero.
        let axis_dimension = P::axis_dimension(context)?;
        let aligned = [inputs[0].move_axis(0)?, inputs[1].move_axis(0)?];
        for input in &aligned {
            if input.batch_axis_position().is_some() && input.r#type().dimension(0) != axis_dimension {
                return Err(BatchingError::MisalignedBatchAxes {
                    message: format!(
                        "`{GATHER_OPERATION_NAME}` mapped input extent {} does not match batching extent {axis_dimension}",
                        input.r#type().dimension(0),
                    ),
                });
            }
        }
        let dimensions = self.dimensions();
        let mut operation = self.clone();
        if mapped_input && !mapped_indices {
            // The same indices select a complete window along the new input axis, so that axis is an output
            // offset dimension. Its window size must be representable in the operation's static slice sizes.
            let Dimension::Static(axis_size) = axis_dimension else {
                return Err(BatchingError::UnsupportedOperation {
                    message: "`gather` with only its input mapped requires a statically known mapped extent"
                        .to_string(),
                });
            };
            operation.slice_sizes.insert(0, axis_size);
            let mut offsets = vec![0];
            offsets.extend(dimensions.offset_dimensions().iter().map(|axis| axis + 1));
            operation.dimensions = GatherDimensionNumbers::new(
                offsets,
                dimensions.collapsed_slice_dimensions().iter().map(|axis| axis + 1).collect(),
                dimensions.start_index_map().iter().map(|axis| axis + 1).collect(),
            )
            .with_batching_dimensions(
                dimensions.operand_batching_dimensions().iter().map(|axis| axis + 1).collect(),
                dimensions.start_indices_batching_dimensions().to_vec(),
            );
        } else if !mapped_input {
            // An extra indices batch dimension simply adds one leading output batch dimension. Indices from
            // different mapped items need not remain jointly sorted or unique.
            operation.dimensions = GatherDimensionNumbers::new(
                dimensions.offset_dimensions().iter().map(|axis| axis + 1).collect(),
                dimensions.collapsed_slice_dimensions().to_vec(),
                dimensions.start_index_map().to_vec(),
            )
            .with_batching_dimensions(
                dimensions.operand_batching_dimensions().to_vec(),
                dimensions.start_indices_batching_dimensions().iter().map(|axis| axis + 1).collect(),
            );
            operation.indices_are_sorted = false;
            operation.unique_indices = false;
        } else {
            // Pair the new input and indices dimensions: every item reads only its own input. A statically empty
            // mapped dimension uses a zero window; otherwise it is a size-one batching dimension.
            operation.slice_sizes.insert(0, usize::from(axis_dimension != Dimension::Static(0)));
            let mut input_batching = vec![0];
            input_batching.extend(dimensions.operand_batching_dimensions().iter().map(|axis| axis + 1));
            let mut indices_batching = vec![0];
            indices_batching.extend(dimensions.start_indices_batching_dimensions().iter().map(|axis| axis + 1));
            operation.dimensions = GatherDimensionNumbers::new(
                dimensions.offset_dimensions().iter().map(|axis| axis + 1).collect(),
                dimensions.collapsed_slice_dimensions().iter().map(|axis| axis + 1).collect(),
                dimensions.start_index_map().iter().map(|axis| axis + 1).collect(),
            )
            .with_batching_dimensions(input_batching, indices_batching);
        }
        if let Some(output_sharding) = self.output_sharding() {
            operation.output_sharding = Some(lift_output_sharding_for_leading_batch_axis(
                output_sharding,
                ArrayBatch::sharding_for_inputs(inputs)?,
            )?);
        }
        Ok(operation.interpret_with_batch_axes(context, &aligned, &[BatchAxis::from_position(0)])?.into())
    }
}

// Forward-mode differentiation gathers the data tangent at the primal indices. The indices and out-of-bounds fill
// are constant with respect to the input data, so the tangent uses zero fill. A zero input tangent stays typed zero.
impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for GatherOperation
where
    C::Operation: From<GatherOperation>,
    C::Value: Gather,
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        let indices = inputs[1].primal();
        let primal = inputs[0].primal().gather(indices, self)?;
        let tangent = match inputs[0].tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
            MaybeZero::Value(tangent) => {
                // An out-of-bounds fill is constant with respect to the gathered input. Its derivative is zero,
                // including when the primal uses NaN or a custom nonzero replacement.
                let operation = if self.mode() == GatherScatterMode::FillOrDrop {
                    self.clone().with_fill_value(
                        EagerContext::<Array>::new().zero(&ArrayType::scalar(tangent.r#type().data_type()))?,
                    )?
                } else {
                    self.clone()
                };
                MaybeZero::Value(tangent.gather(&context.primal_to_tangent(indices.clone())?, &operation)?)
            }
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

// Partition-aware transpose rule for the primal [`GatherOperation`]. The integer index input (input 1) has no
// tangent space, so in a valid pushforward it is the known input and the gathered input (input 0) is the
// linear one. The forward map `t ↦ gather(t, indices)` has, as its adjoint, the dual scatter-add that writes the
// output cotangent back into a zero input at the gathered windows: the scatter geometry mirrors the gather
// axis-for-axis. The transpose reads the known indices from the pullback boundary and stages an ordinary additive
// [`ScatterOperation`], so linearization retains the indices as regular SSA residuals. The indices receive a
// structural zero, and a zero output cotangent stays a structural zero.
//
// **Contract:** this homogeneous rule requires a statically shaped input. The scatter target is a zero of the
// input's cotangent type, and the homogeneous [`ArrayType`] operation family owns no first-class dimension
// operations, so it has no constructor that can supply a runtime extent for that zero. A dynamically shaped input
// is therefore rejected here with an exact diagnostic. Mixed [`ArrayIrType`](crate::ArrayIrType) programs are
// unaffected: the [`MemberDifferentiableOperation`](crate::MemberDifferentiableOperation) rule above routes a
// dynamically shaped gather into a residual-carrying [`LinearCallOperation`](crate::LinearCallOperation) whose
// transpose region rebuilds the same zero from the retained exact extents.
impl<V: Value<Type = ArrayType>, O> TransposableOperation<V, O> for GatherOperation
where
    O: Operation<Type = ArrayType>
        + From<AddOperation<ArrayType>>
        + From<ZeroOperation<ArrayType>>
        + From<ScatterOperation>
        + From<BroadcastOperation>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        check_count!("accumulator", accumulators, 2, DifferentiationError);
        match &outputs[0] {
            MaybeZero::Zero(_) => Ok(()),
            MaybeZero::Value(cotangent) => {
                if !accumulators[0].is_needed() {
                    return Ok(());
                }
                // The indices are the known input; the dispatch guarantees a `Known` input carries its pullback
                // value, so read the tracer directly.
                let indices = inputs[1]
                    .as_known()
                    .expect("dispatch guarantees a known operand carries its pullback value")
                    .clone();
                // Only the nullary zero is available in the homogeneous family, so enforce this rule's static-shape
                // contract explicitly instead of letting a dynamic input surface the constructor's own diagnostic.
                let operand_cotangent_type = inputs[0].r#type().cotangent()?;
                if operand_cotangent_type.static_shape().is_none() {
                    return Err(TypeError::invalid(format!(
                        "`{GATHER_OPERATION_NAME}` transpose requires a statically shaped operand but got \
                         {operand_cotangent_type}",
                    ))
                    .into());
                }
                let output_sharding = operand_cotangent_type.sharding().cloned();
                let zeros = MaybeZero::Zero(operand_cotangent_type.clone()).materialize(&**context)?;
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
                    .with_unique_indices(self.unique_indices())
                    .with_output_sharding(output_sharding);
                let outputs =
                    context.stage_operation(scatter_operation, Vec::new(), &[zeros, indices, cotangent.clone()])?;
                check_count!("output", outputs, 1, ProgramError);
                let mut contribution = outputs.into_iter().next().unwrap();
                if contribution.r#type().as_ref() != &operand_cotangent_type {
                    let mut outputs = context.stage_operation(
                        BroadcastOperation::new(
                            operand_cotangent_type.clone(),
                            (0..operand_cotangent_type.rank()).collect(),
                        ),
                        Vec::new(),
                        std::slice::from_ref(&contribution),
                    )?;
                    check_count!("output", outputs, 1, ProgramError);
                    contribution = outputs.remove(0);
                }
                accumulators[0].accumulate(context, MaybeZero::Value(contribution))
            }
        }
    }
}

// Projected array IR JVP rule for [`GatherOperation`]. A dynamically shaped input retains its exact extents
// and indices as ordinary residual values; a static input delegates to the homogeneous projected rule.
impl<C> MemberDifferentiableOperation<C> for GatherOperation
where
    C: Context<Type = ArrayIrType>,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Operation:
        From<DimensionSizeOperation> + From<LinearCallOperation<ArrayIrType>> + OperationProjection<ArrayType>,
    <C::Operation as OperationProjection<ArrayType>>::Projected: DifferentiableOperation<ProjectedContext<C, ArrayType>>
        + From<GatherOperation>
        + From<ScatterOperation>
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
        let [operand, indices] = inputs else {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
        };
        let operand_type = <&ArrayType>::try_from(operand.primal().r#type().as_ref())?.clone();
        if operand_type.shape().dimensions().iter().all(|dimension| matches!(dimension, Dimension::Static(_))) {
            let operation = <C::Operation as OperationProjection<ArrayType>>::Projected::from(self.clone());
            return jvp_projected_operation(destinations, &operation, inputs);
        }

        let operation = <C::Operation as OperationProjection<ArrayType>>::Projected::from(self.clone());
        let mut outputs = context.bind(operation, Vec::new(), &[operand.primal().clone(), indices.primal().clone()])?;
        check_count!("output", outputs, 1, ProgramError);
        let primal = outputs.remove(0);
        let output_primal = primal;
        let primal = destinations.primal_to_tangent(output_primal.clone())?;
        let tangent_inputs = destinations.dual_primal_to_tangent(inputs)?;
        let inputs = tangent_inputs.as_slice();
        let operand = &inputs[0];
        let indices = &inputs[1];
        let context = destinations.tangent();
        let tangent = match operand.tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
            MaybeZero::Value(operand_tangent) => {
                let mut residuals = LinearResiduals::new();
                let indices_index = residuals.retain(indices.primal().clone());
                let operand_shape = residuals.retain_shape(context, operand.primal())?;
                // The linear region differentiates input data, not the primal's constant replacement value.
                let forward_operation = if self.mode() == GatherScatterMode::FillOrDrop {
                    self.clone().with_fill_value(EagerContext::<Array>::new().zero(&ArrayType::scalar(
                        <&ArrayType>::try_from(operand_tangent.r#type().as_ref())?.data_type(),
                    ))?)?
                } else {
                    self.clone()
                };
                let transpose_operand_type = operand_type.cotangent()?;
                let dimensions = self.dimensions();
                let transpose_operation = ScatterOperation::new(
                    ScatterDimensionNumbers::new(
                        dimensions.offset_dimensions().to_vec(),
                        dimensions.collapsed_slice_dimensions().to_vec(),
                        dimensions.start_index_map().to_vec(),
                    )
                    .with_batching_dimensions(
                        dimensions.operand_batching_dimensions().to_vec(),
                        dimensions.start_indices_batching_dimensions().to_vec(),
                    ),
                    ScatterReductionKind::Add,
                )
                .with_mode(self.mode())
                .with_indices_are_sorted(self.indices_are_sorted())
                .with_unique_indices(self.unique_indices())
                .with_output_sharding(transpose_operand_type.sharding().cloned());
                let mut tangent_outputs = LinearCallOperation::stage(
                    context,
                    residuals.into_values(),
                    vec![operand_tangent.clone()],
                    move |residuals, linear_inputs| {
                        linear_inputs[0].dispatch_domain().bind(
                            <C::Operation as OperationProjection<ArrayType>>::Projected::from(forward_operation),
                            Vec::new(),
                            &[linear_inputs[0].clone(), residuals[indices_index].clone()],
                        )
                    },
                    move |residuals, output_cotangents| {
                        let transpose_context = output_cotangents[0].dispatch_domain();
                        let mut zero_outputs = transpose_context.bind(
                            <C::Operation as OperationProjection<ArrayType>>::Projected::from(ZeroOperation::new(
                                transpose_operand_type.clone(),
                            )),
                            Vec::new(),
                            operand_shape.dynamic_dimensions(residuals).as_slice(),
                        )?;
                        check_count!("output", zero_outputs, 1, ProgramError);
                        let zeros = zero_outputs.remove(0);
                        let mut contributions = transpose_context.bind(
                            <C::Operation as OperationProjection<ArrayType>>::Projected::from(transpose_operation),
                            Vec::new(),
                            &[zeros, residuals[indices_index].clone(), output_cotangents[0].clone()],
                        )?;
                        check_count!("output", contributions, 1, ProgramError);
                        let contribution = contributions.remove(0);
                        // Residual extents may refine singleton dynamic dimensions to static dimensions. Restore
                        // the original cotangent signature, including its dimension identities and storage metadata.
                        let contribution =
                            if <&ArrayType>::try_from(contribution.r#type().as_ref())? != &transpose_operand_type {
                                let mut outputs = transpose_context.bind(
                                    <C::Operation as OperationProjection<ArrayType>>::Projected::from(
                                        BroadcastOperation::new(
                                            transpose_operand_type.clone(),
                                            (0..transpose_operand_type.rank()).collect(),
                                        ),
                                    ),
                                    Vec::new(),
                                    std::slice::from_ref(&contribution),
                                )?;
                                check_count!("output", outputs, 1, ProgramError);
                                outputs.remove(0)
                            } else {
                                contribution
                            };
                        Ok(vec![contribution])
                    },
                )?;
                check_count!("output", tangent_outputs, 1, ProgramError);
                MaybeZero::Value(tangent_outputs.remove(0))
            }
        };
        Ok(vec![DifferentiationDual::new(output_primal, tangent)?])
    }
}

/// Value-level gather capability: the receiver-style entry point for staging or executing [`GatherOperation`].
///
/// The receiver is the input (the data source); `indices` is a separate integer-typed value whose last axis holds
/// each start-index vector. The output assembles the sliced windows according to `operation`'s
/// [`GatherDimensionNumbers`]; see that type for the shape rule and the implicit index-vector-dimension convention.
/// The input and indices must reside in the same memory space. The result retains that memory placement and clears
/// explicit physical layout metadata because gathering changes the logical relationship between axes and storage.
///
/// # Example
///
/// ```rust
/// use ryft_core::{Array, Gather, GatherDimensionNumbers, GatherOperation};
///
/// let input = Array::matrix(3, 2, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
/// let indices = Array::matrix(2, 1, vec![0_i32, 2]).unwrap();
/// // Each query selects a row: input axis 0 is collapsed, while output axis 1 retains the full row window.
/// let dimensions = GatherDimensionNumbers::new(vec![1], vec![0], vec![0]);
/// let operation = GatherOperation::new(dimensions, vec![1, 2]);
/// let output = input.gather(&indices, &operation).unwrap();
/// assert_eq!(output, Array::matrix(2, 2, vec![0.0, 1.0, 4.0, 5.0]).unwrap());
/// ```
pub trait Gather: Sized {
    /// Reads windows from the input at the starts given by `indices` and assembles them using `operation`.
    /// Negative starts are out of bounds; they do not count backward from an axis end. Bounds handling applies to
    /// whole windows, so one invalid start fills the entire window in fill mode.
    ///
    /// # Parameters
    ///
    ///   - `indices`: Integer array with one trailing index-vector axis. Its remaining axes enumerate queries.
    ///   - `operation`: Dimension mapping, window sizes, bounds mode, optional fill, and placement hints.
    fn gather(&self, indices: &Self, operation: &GatherOperation) -> Result<Self, ProgramError>;

    /// Gathers complete slices along one axis using raw integer indices. The index array's shape replaces that
    /// input axis in the result, and all other input axes retain their order and full size. Unlike indexing APIs that
    /// count negative indices backward from the end, this function treats every negative index as out of bounds and
    /// applies `mode` directly. It does not change or wrap index values.
    ///
    /// The selected axis may have a dynamic extent. All other input extents must be statically known because the
    /// underlying [`GatherOperation`] stores their complete window sizes as host integers. The index shape must also
    /// support the homogeneous [`Reshape`] used to append its index-vector axis. Use an explicit operation for dynamic query shapes, partial windows, or a custom fill.
    ///
    /// # Parameters
    ///
    ///   - `indices`: Integer indices of any rank. A scalar selects one slice and removes the selected axis.
    ///   - `axis`: Input axis to select. Negative axes count backward from the input rank.
    ///   - `mode`: Out-of-bounds policy. [`GatherScatterMode::Clip`] clamps indices; [`GatherScatterMode::FillOrDrop`]
    ///     fills invalid slices using the input data type's default fill; the promise mode requires valid indices.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ryft_core::{Array, Gather, GatherScatterMode};
    ///
    /// let input = Array::matrix(2, 3, vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
    /// let indices = Array::vector(vec![2_i32, 0]).unwrap();
    /// let output = input.gather_axis(&indices, 1, GatherScatterMode::Clip).unwrap();
    /// assert_eq!(output, Array::matrix(2, 2, vec![3_i32, 1, 6, 4]).unwrap());
    /// ```
    fn gather_axis<A: Into<Axis>>(&self, indices: &Self, axis: A, mode: GatherScatterMode) -> Result<Self, ProgramError>
    where
        Self: Typed<Type = ArrayType> + Reshape,
    {
        let input_type = self.r#type();
        let axis = axis.into().normalize(input_type.rank()).map_err(|error| TypeError::invalid(error.to_string()))?;
        let slice_sizes = input_type
            .shape()
            .dimensions()
            .iter()
            .enumerate()
            .map(|(input_axis, dimension)| match dimension {
                _ if input_axis == axis => Ok(1),
                Dimension::Static(size) => Ok(*size),
                _ => Err(TypeError::invalid(format!(
                    "`gather_axis` requires a static extent on unselected axis {input_axis}",
                ))),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let indices_type = indices.r#type();
        let mut indices_dimensions = indices_type.shape().dimensions().to_vec();
        indices_dimensions.push(Dimension::Static(1));
        let expanded_indices = indices.reshape(Shape::new(indices_dimensions))?;
        let offset_dimensions = (0..input_type.rank())
            .filter(|input_axis| *input_axis != axis)
            .map(|input_axis| if input_axis < axis { input_axis } else { input_axis + indices_type.rank() - 1 })
            .collect();
        let operation =
            GatherOperation::new(GatherDimensionNumbers::new(offset_dimensions, vec![axis], vec![axis]), slice_sizes)
                .with_mode(mode);
        self.gather(&expanded_indices, &operation)
    }
}

impl Gather for ArrayType {
    // Type-level gather: validates the dimension numbers and slice sizes against the input and indices types and
    // computes the output shape and placement.
    fn gather(&self, indices: &Self, operation: &GatherOperation) -> Result<Self, ProgramError> {
        let operand = self;
        let dimensions = operation.dimensions();
        let slice_sizes = operation.slice_sizes();
        let operand_rank = operand.rank();
        let indices_rank = indices.rank();

        if indices_rank == 0 {
            return Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` indices must have rank at least 1 (the trailing index vector)"
            ))
            .into());
        }
        if !indices.data_type().is_integer() {
            return Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` indices must be integer-typed but have type {indices}"
            ))
            .into());
        }
        if operation.mode() == GatherScatterMode::FillOrDrop {
            operation.resolved_fill_value(operand.data_type())?;
        }
        if operand.memory() != indices.memory() {
            return Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` operand and indices must share one memory space but reside in {} and {}",
                operand.memory(),
                indices.memory(),
            ))
            .into());
        }
        let index_vector_dimension = indices_rank - 1;
        let Dimension::Static(index_vector_extent) = indices.dimension(index_vector_dimension) else {
            return Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` indices index vector dimension must have a static extent"
            ))
            .into());
        };

        // Output rank, and the constituent input-axis classification.
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
                "`{GATHER_OPERATION_NAME}` start_index_map has length {} but the index vector extent is \
                     {index_vector_extent}",
                dimensions.start_index_map().len(),
            ))
            .into());
        }
        validate_unique_in_range(GATHER_OPERATION_NAME, "start_index_map", dimensions.start_index_map(), operand_rank)?;

        if dimensions.start_indices_batching_dimensions().len() != dimensions.operand_batching_dimensions().len() {
            return Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` operand and start-indices batching dimensions must align 1:1, but got {} \
                     and {}",
                dimensions.operand_batching_dimensions().len(),
                dimensions.start_indices_batching_dimensions().len(),
            ))
            .into());
        }
        validate_unique_in_range(
            GATHER_OPERATION_NAME,
            "start_indices_batching_dimensions",
            dimensions.start_indices_batching_dimensions(),
            indices_rank,
        )?;
        if dimensions
            .start_index_map()
            .iter()
            .any(|axis| dimensions.operand_batching_dimensions().contains(axis))
        {
            return Err(TypeError::invalid(
                "`gather` `start_index_map` and `operand_batching_dimensions` must be disjoint",
            )
            .into());
        }
        for &dimension in dimensions.start_indices_batching_dimensions() {
            if dimension >= indices_rank || dimension == index_vector_dimension {
                return Err(TypeError::invalid(format!(
                    "`{GATHER_OPERATION_NAME}` start_indices_batching_dimensions entry {dimension} is out of range \
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
                "`{GATHER_OPERATION_NAME}` collapsed_slice_dimensions and operand_batching_dimensions must be \
                     disjoint"
            ))
            .into());
        }

        // Slice sizes: one per input axis; size 1 on collapsed axes; size at most 1 on batching axes; within the
        // input extent when that extent is static.
        if slice_sizes.len() != operand_rank {
            return Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` slice_sizes has length {} but the operand has rank {operand_rank}",
                slice_sizes.len(),
            ))
            .into());
        }
        for (axis, &size) in slice_sizes.iter().enumerate() {
            match operand.dimension(axis) {
                Dimension::Static(extent) if size > extent => {
                    return Err(TypeError::invalid(format!(
                        "`{GATHER_OPERATION_NAME}` slice size {size} at axis {axis} exceeds the operand extent \
                         {extent}"
                    ))
                    .into());
                }
                Dimension::Dynamic(variable) if size > variable.bounds().lower() => {
                    return Err(TypeError::invalid(format!(
                        "`{GATHER_OPERATION_NAME}` slice size {size} exceeds the guaranteed minimum extent {} of \
                         dynamic operand axis {axis}",
                        variable.bounds().lower(),
                    ))
                    .into());
                }
                _ => {}
            }
            if collapsed.contains(&axis) && size != 1 {
                return Err(TypeError::invalid(format!(
                    "`{GATHER_OPERATION_NAME}` collapsed slice dimension {axis} must have slice size 1 but has {size}"
                ))
                .into());
            }
            if operand_batching.contains(&axis) && size > 1 {
                return Err(TypeError::invalid(format!(
                    "`{GATHER_OPERATION_NAME}` operand batching dimension {axis} must have slice size at most 1 but \
                         has {size}"
                ))
                .into());
            }
        }

        let offset_count = operand_rank - collapsed.len() - operand_batching.len();
        if dimensions.offset_dimensions().len() != offset_count {
            return Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` offset_dimensions has length {} but the operand has {offset_count} \
                     non-collapsed, non-batching axes",
                dimensions.offset_dimensions().len(),
            ))
            .into());
        }

        // Batch-dimension extents must match between input and indices.
        for (&operand_axis, &indices_axis) in
            dimensions.operand_batching_dimensions().iter().zip(dimensions.start_indices_batching_dimensions())
        {
            if !dimensions_have_equal_extents(&operand.dimension(operand_axis), &indices.dimension(indices_axis)) {
                return Err(TypeError::invalid(format!(
                    "`{GATHER_OPERATION_NAME}` batching dimensions must have equal extents, but operand axis \
                         {operand_axis} and indices axis {indices_axis} differ"
                ))
                .into());
            }
        }

        // Output shape: offset positions take the (non-collapsed, non-batching) input window sizes in input-axis
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

        // Retained full-window axes preserve input placement, and query axes inherit index placement. Partial
        // windows on explicitly sharded axes need an explicit output placement. Reduction and manual-axis state
        // remain part of the contract even when a placement is supplied explicitly.
        let operand_sharding = operand.sharding();
        let indices_sharding = indices.sharding();
        let mesh = resolve_mesh(operand_sharding, indices_sharding)?;
        if indices_sharding
            .is_some_and(|sharding| !sharding.unreduced_axes().is_empty() || !sharding.reduced_axes().is_empty())
        {
            return Err(TypeError::invalid("`gather` indices cannot carry reduced or unreduced mesh axes").into());
        }
        let unreduced_axes = operand_sharding.map(Sharding::unreduced_axes).cloned().unwrap_or_default();
        let reduced_axes = operand_sharding.map(Sharding::reduced_axes).cloned().unwrap_or_default();
        let mut varying_manual_axes = operand_sharding.map(Sharding::varying_manual_axes).cloned().unwrap_or_default();
        if let Some(sharding) = indices_sharding {
            varying_manual_axes.extend(sharding.varying_manual_axes().iter().cloned());
            if (!unreduced_axes.is_empty() || !reduced_axes.is_empty())
                && (sharding.dimensions().iter().any(|dimension| *dimension != ShardingDimension::Replicated)
                    || !sharding.varying_manual_axes().is_empty())
            {
                return Err(TypeError::invalid(
                    "`gather` reduction-state inputs require replicated, invariant indices",
                )
                .into());
            }
        }
        if !unreduced_axes.is_empty() && operation.mode() == GatherScatterMode::FillOrDrop {
            return Err(TypeError::invalid("`gather` fill mode does not support unreduced inputs").into());
        }
        let sharding = if let Some(requested) = operation.output_sharding() {
            if mesh.as_ref().is_some_and(|mesh| mesh != requested.mesh()) {
                return Err(TypeError::invalid("`gather` requested output sharding uses a different mesh").into());
            }
            if requested.unreduced_axes() != &unreduced_axes
                || requested.reduced_axes() != &reduced_axes
                || requested.varying_manual_axes() != &varying_manual_axes
            {
                return Err(TypeError::invalid(
                    "`gather` requested output sharding changes reduction or manual-axis state",
                )
                .into());
            }

            if requested.rank() != output_rank {
                return Err(TypeError::invalid(format!(
                    "`{GATHER_OPERATION_NAME}` output sharding rank ({}) does not match the output rank \
                         ({output_rank})",
                    requested.rank(),
                ))
                .into());
            }
            if requested.references_auto_axis() {
                return Err(TypeError::invalid(format!(
                    "`{GATHER_OPERATION_NAME}` output sharding cannot reference auto mesh axes"
                ))
                .into());
            }
            Some(requested.clone())
        } else if let Some(mesh) = mesh {
            // Indexed or collapsed axes require replication when the gather reads only part of their extent.
            // Full-extent windows can retain their placement. The index-vector axis always requires replication.
            let replicated_operand_axes: BTreeSet<usize> = dimensions
                .start_index_map()
                .iter()
                .chain(dimensions.collapsed_slice_dimensions())
                .copied()
                .collect();
            if let Some(sharding) = operand_sharding {
                for &axis in &replicated_operand_axes {
                    if operand.dimension(axis) != Dimension::Static(slice_sizes[axis])
                        && dimension_has_explicit_axis(&mesh, &sharding.dimensions()[axis])
                    {
                        return Err(TypeError::invalid(format!(
                            "`{GATHER_OPERATION_NAME}` operand axis {axis} is indexed by the start indices and must \
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
                    "`{GATHER_OPERATION_NAME}` indices index vector dimension must be replicated over explicit \
                         mesh axes"
                ))
                .into());
            }

            // A partial window does not preserve the placement of the complete input axis, even when its
            // start index is implicit zero rather than supplied in the index vector.
            for &axis in &operand_offset_axes {
                if operand.dimension(axis) != Dimension::Static(slice_sizes[axis])
                    && operand_sharding
                        .is_some_and(|sharding| dimension_has_explicit_axis(&mesh, &sharding.dimensions()[axis]))
                {
                    return Err(TypeError::invalid(
                        "`gather` partial sharded windows require explicit output sharding",
                    )
                    .into());
                }
            }
            let mut indices_placement = indices_sharding
                .map(|sharding| sharding.dimensions().to_vec())
                .unwrap_or_else(|| vec![ShardingDimension::Replicated; indices_rank]);
            for (&input_axis, &indices_axis) in
                dimensions.operand_batching_dimensions().iter().zip(dimensions.start_indices_batching_dimensions())
            {
                let input_placement = operand_sharding
                    .map(|sharding| sharding.dimensions()[input_axis].clone())
                    .unwrap_or(ShardingDimension::Replicated);
                let indices_dimension = &mut indices_placement[indices_axis];
                if *indices_dimension == ShardingDimension::Replicated {
                    *indices_dimension = input_placement;
                } else if input_placement != ShardingDimension::Replicated && input_placement != *indices_dimension {
                    return Err(TypeError::invalid(
                        "`gather` conflicting batching-axis shardings require explicit output sharding",
                    )
                    .into());
                }
            }

            // Propagate placement: offset positions inherit the input window axes; the remaining positions inherit
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
                        indices_placement[indices_axis].clone()
                    }
                })
                .collect();

            // Gather preserves the input reduction state only with the invariant index contract checked above.
            let map_sharding_error = |error| {
                TypeError::invalid(format!("`{GATHER_OPERATION_NAME}` output sharding construction failed: {error}"))
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

impl Gather for Array {
    fn gather(&self, indices: &Self, operation: &GatherOperation) -> Result<Self, ProgramError> {
        let output_type = self.r#type().gather(indices.r#type().as_ref(), operation)?;
        let dimensions = operation.dimensions();
        let slice_sizes = operation.slice_sizes();
        let operand_shape = self.r#type().static_shape().unwrap();
        let indices_shape = indices.r#type().static_shape().unwrap();
        let operand_rank = operand_shape.rank();
        let indices_rank = indices_shape.rank();
        let output_rank = output_type.rank();
        let index_vector_dimension = indices_rank - 1;
        let index_vector_extent = indices_shape[index_vector_dimension];

        // Classify input axes (window axes carry the slice; collapsed/batching do not) and output axes (offset
        // positions carry the window, the rest carry the indices' batch coordinates).
        let collapsed: BTreeSet<usize> = dimensions.collapsed_slice_dimensions().iter().copied().collect();
        let batching: BTreeSet<usize> = dimensions.operand_batching_dimensions().iter().copied().collect();
        let operand_window_axes: Vec<usize> =
            (0..operand_rank).filter(|axis| !collapsed.contains(axis) && !batching.contains(axis)).collect();
        let offset_positions: BTreeSet<usize> = dimensions.offset_dimensions().iter().copied().collect();
        let batch_output_positions: Vec<usize> =
            (0..output_rank).filter(|position| !offset_positions.contains(position)).collect();
        let indices_batch_axes: Vec<usize> = (0..indices_rank).filter(|axis| *axis != index_vector_dimension).collect();

        // Resolve a fill only when the mode can use it, preserving the exact element encoding of explicit fills.
        let dropped_fill = if operation.mode() == GatherScatterMode::FillOrDrop {
            let value = operation.resolved_fill_value(output_type.data_type())?;
            let addressing = ArrayAddressing::new(value.r#type().into_owned())?;
            Some((value, addressing))
        } else {
            None
        };
        let input_addressing = ArrayAddressing::new(self.r#type().into_owned())?;
        let indices_addressing = ArrayAddressing::new(indices.r#type().into_owned())?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let mut bytes = vec![0; output_addressing.storage_byte_len()];
        let mut output_index = vec![0usize; output_rank];
        let mut indices_index = vec![0usize; indices_rank];
        let mut starts = vec![0i128; index_vector_extent];
        let mut operand_index = vec![0i128; operand_rank];
        let mut operand_storage_index = vec![0usize; operand_rank];
        for output_element in 0..output_addressing.element_count() {
            // Place the output's batch coordinates into the indices multi-index and read this query's start vector.
            indices_index.fill(0);
            for (position, &output_position) in batch_output_positions.iter().enumerate() {
                indices_index[indices_batch_axes[position]] = output_index[output_position];
            }
            for (component, start) in starts.iter_mut().enumerate() {
                indices_index[index_vector_dimension] = component;
                let index_bytes = &indices.storage_bytes()[indices_addressing.byte_range_unchecked(&indices_index)];
                *start = match indices.r#type().data_type() {
                    DataType::I1 => i128::from(i1::decode(index_bytes).value()),
                    DataType::I2 => i128::from(i2::decode(index_bytes).value()),
                    DataType::I4 => i128::from(i4::decode(index_bytes).value()),
                    DataType::I8 => i128::from(i8::decode(index_bytes)),
                    DataType::I16 => i128::from(i16::decode(index_bytes)),
                    DataType::I32 => i128::from(i32::decode(index_bytes)),
                    DataType::I64 => i128::from(i64::decode(index_bytes)),
                    DataType::U1 => i128::from(u1::decode(index_bytes).value()),
                    DataType::U2 => i128::from(u2::decode(index_bytes).value()),
                    DataType::U4 => i128::from(u4::decode(index_bytes).value()),
                    DataType::U8 => i128::from(u8::decode(index_bytes)),
                    DataType::U16 => i128::from(u16::decode(index_bytes)),
                    DataType::U32 => i128::from(u32::decode(index_bytes)),
                    DataType::U64 => i128::from(u64::decode(index_bytes)),
                    _ => unreachable!(),
                };
            }
            // Assemble the input multi-index: window offsets, then batching coordinates, then start offsets.
            operand_index.fill(0);
            for (window, &operand_axis) in operand_window_axes.iter().enumerate() {
                operand_index[operand_axis] = output_index[dimensions.offset_dimensions()[window]] as i128;
            }
            for (batch, &operand_axis) in dimensions.operand_batching_dimensions().iter().enumerate() {
                operand_index[operand_axis] =
                    indices_index[dimensions.start_indices_batching_dimensions()[batch]] as i128;
            }
            let mut dropped = false;
            for (component, &operand_axis) in dimensions.start_index_map().iter().enumerate() {
                let raw = starts[component];
                let maximum = (operand_shape[operand_axis] - slice_sizes[operand_axis]) as i128;
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
            let source = if dropped {
                let (value, addressing) = dropped_fill.as_ref().unwrap();
                &value.storage_bytes()[addressing.byte_range_for_flat_index(0)]
            } else {
                for axis in 0..operand_rank {
                    operand_storage_index[axis] = operand_index[axis] as usize;
                }
                &self.storage_bytes()[input_addressing.byte_range_unchecked(&operand_storage_index)]
            };
            bytes[output_addressing.byte_range_for_flat_index(output_element)].copy_from_slice(source);
            output_addressing.advance_index(&mut output_index);
        }
        Ok(Self::new_unchecked(output_type, Arc::new(bytes)))
    }
}

impl<A: Gather + Value<Type = ArrayType>> Gather for ArrayIrValue<A> {
    fn gather(&self, indices: &Self, operation: &GatherOperation) -> Result<Self, ProgramError> {
        let input = <Self as ValueProjection<ArrayType>>::projected(self)?;
        let indices = <Self as ValueProjection<ArrayType>>::projected(indices)?;
        Ok(Self::Array(input.gather(indices, operation)?))
    }
}

// Bind homogeneous array values through their context. Mixed tracers use the canonical array projection;
// requiring a homogeneous type here keeps array-operation trait obligations from becoming recursive.
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

/// Gathers complete slices with first-class dimensions for the untouched input axes and query shape.
///
/// This is a composition of [`DynamicBroadcast`](crate::operations::DynamicBroadcast) and [`GatherOperation`].
/// Untouched input axes become paired gather batching axes instead of runtime-sized windows. The selected axis
/// uses a size-one window; paired axes use size zero or one according to their bounds. The output retains the exact
/// runtime dimensions of the input and queries. Like [`Gather::gather_axis`], negative indices are out of bounds
/// rather than indexing backward from the end.
///
/// # Examples
///
/// ```rust
/// # use ryft_core::{Array, ArrayIrValue, DynamicGather, GatherScatterMode};
/// let input = ArrayIrValue::Array(Array::matrix(2, 3, vec![1_i32, 2, 3, 4, 5, 6]).unwrap());
/// let indices = ArrayIrValue::Array(Array::vector(vec![2_i32, 0]).unwrap());
/// let output = input.dynamic_gather_axis(&indices, 1, GatherScatterMode::Clip).unwrap();
/// assert_eq!(output, ArrayIrValue::Array(Array::matrix(2, 2, vec![3_i32, 1, 6, 4]).unwrap()));
/// ```
pub trait DynamicGather: Value<Type = ArrayIrType> + Sized {
    /// Gathers along `axis`, replacing that axis with the complete shape of `indices` in the result.
    ///
    /// # Parameters
    ///
    ///   - `indices`: Integer query array. A scalar removes the selected input axis; a dynamic query shape is retained.
    ///   - `axis`: Input axis to select, with negative axes counted from the end of the input rank.
    ///   - `mode`: Bounds handling applied to each raw query index; see [`GatherScatterMode`].
    fn dynamic_gather_axis<A: Into<Axis>>(
        &self,
        indices: &Self,
        axis: A,
        mode: GatherScatterMode,
    ) -> Result<Self, ProgramError>;
}

impl<A: Value<Type = ArrayType> + Gather + Reshape> DynamicGather for ArrayIrValue<A>
where
    A::DispatchDomain: Zero<A>,
{
    fn dynamic_gather_axis<AxisValue: Into<Axis>>(
        &self,
        indices: &Self,
        axis: AxisValue,
        mode: GatherScatterMode,
    ) -> Result<Self, ProgramError> {
        let input = <Self as ValueProjection<ArrayType>>::projected(self)?;
        let indices = <Self as ValueProjection<ArrayType>>::projected(indices)?;
        let input_type = input.r#type();
        let axis = axis.into().normalize(input_type.rank()).map_err(|error| TypeError::invalid(error.to_string()))?;
        if input_type.dimension(axis) == Dimension::Static(0)
            && indices.r#type().element_count().map_err(|error| TypeError::invalid(error.to_string()))? == Some(0)
        {
            // Validate the integer query type and placement even though the empty result reads no elements.
            let mut validation_shape = input_type.shape().dimensions().to_vec();
            validation_shape[axis] = Dimension::Static(1);
            let output_type = input_type.clone().into_owned().with_shape(Shape::new(validation_shape)).gather_axis(
                indices.r#type().as_ref(),
                axis,
                mode,
            )?;
            return Ok(Self::Array(input.dispatch_domain().zero(&output_type)?));
        }
        Ok(Self::Array(input.gather_axis(indices, axis, mode)?))
    }
}

impl<V> DynamicGather for V
where
    V: Value<Type = ArrayIrType> + DimensionSize + DynamicBroadcast + ValueProjection<ArrayType, Projected: Gather>,
    V::DispatchDomain: Context<Type = ArrayIrType> + DimensionConstant + DynamicZero<V>,
{
    fn dynamic_gather_axis<A: Into<Axis>>(
        &self,
        indices: &Self,
        axis: A,
        mode: GatherScatterMode,
    ) -> Result<Self, ProgramError> {
        let input_type = self.r#type();
        let input_type = <&ArrayType>::try_from(input_type.as_ref())?;
        let indices_type = indices.r#type();
        let indices_type = <&ArrayType>::try_from(indices_type.as_ref())?;
        let axis = axis.into().normalize(input_type.rank()).map_err(|error| TypeError::invalid(error.to_string()))?;
        let mut dimensions = Vec::new();
        let mut input_batching = Vec::new();
        let mut indices_batching = Vec::new();
        for input_axis in 0..input_type.rank() {
            if input_axis == axis {
                for query_axis in 0..indices_type.rank() {
                    dimensions.push(indices.dimension_size(query_axis)?);
                }
            } else {
                input_batching.push(input_axis);
                indices_batching.push(dimensions.len());
                dimensions.push(self.dimension_size(input_axis)?);
            }
        }
        dimensions.push(self.dispatch_domain().dimension_constant(1)?);
        // Broadcast each scalar query over the untouched input coordinates. Those coordinates select matching
        // input/indices batches, so no symbolic extent is encoded as a host-sized gather window.
        let indices =
            indices.dynamic_broadcast(&dimensions, &(axis..axis + indices_type.rank()).collect::<Vec<_>>())?;
        let operation = GatherOperation::new(
            GatherDimensionNumbers::new(vec![], vec![axis], vec![axis])
                .with_batching_dimensions(input_batching, indices_batching),
            (0..input_type.rank())
                .map(|input_axis| {
                    if input_axis == axis {
                        1
                    } else {
                        match input_type.dimension(input_axis) {
                            Dimension::Static(size) => usize::from(size != 0),
                            Dimension::Dynamic(variable) => usize::from(variable.bounds().lower() != 0),
                        }
                    }
                })
                .collect(),
        )
        .with_mode(mode);
        if indices_type.element_count().map_err(|error| TypeError::invalid(error.to_string()))? == Some(0) {
            // Empty queries read no input elements, including when the selected axis itself is empty. Use the
            // ordinary gather metadata rules with a placeholder selected extent of one, then construct its empty
            // result. This preserves placement validation without staging an invalid collapsed size-one window.
            let mut shape = input_type.shape().dimensions().to_vec();
            shape[axis] = Dimension::Static(1);
            let output_type = input_type
                .clone()
                .with_shape(Shape::new(shape))
                .gather(<&ArrayType>::try_from(indices.r#type().as_ref())?, &operation)?;
            let dynamic_dimensions = output_type
                .shape()
                .dimensions()
                .iter()
                .zip(&dimensions)
                .filter_map(|(dimension, value)| matches!(dimension, Dimension::Dynamic(_)).then_some(value.clone()))
                .collect::<Vec<_>>();
            return self.dispatch_domain().dynamic_zero(&output_type, &dynamic_dimensions);
        }
        Ok(V::from_projected(self.clone().into_projected()?.gather(&indices.into_projected()?, &operation)?))
    }
}

/// Returns whether two indexing dimensions provably have the same extent.
///
/// Nominally equal dimensions are equal without additional evidence. Distinct dimensions are equal only when both
/// bounds describe the same single integer, for example a retained `n` with bounds `[0, 1)` and a static zero. This
/// occurs when specialization retains an index-array signature but materializes a concrete-shaped zero cotangent.
/// Equal non-singleton bounds never establish equality between independent nominal dimensions.
pub(crate) fn dimensions_have_equal_extents(left: &Dimension, right: &Dimension) -> bool {
    let bounds = left.bounds();
    left == right
        || (bounds == right.bounds()
            && bounds.upper().is_some_and(|upper| bounds.lower().checked_add(1) == Some(upper)))
}

/// Returns whether `dimension` is sharded over at least one explicit mesh axis of `mesh` (the explicit-axis gate used
/// by the dot/reduce/slice sharding rules). Shared with [`super::scattering`].
pub(crate) fn dimension_has_explicit_axis(mesh: &LogicalMesh, dimension: &ShardingDimension) -> bool {
    matches!(dimension, ShardingDimension::Sharded(axis_names)
        if axis_names.iter().any(|name| mesh.axis_type(name) == Some(MeshAxisType::Explicit)))
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
                    "`{GATHER_OPERATION_NAME}` operand and indices shardings must use the same mesh"
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
                "`{operation_name}` `{field}` must be sorted and unique but got {axes:?}"
            )));
        }
    }
    if let Some(&axis) = axes.iter().find(|&&axis| axis >= bound) {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` `{field}` entry {axis} is out of range for bound {bound}"
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
                "`{operation_name}` `{field}` entry {axis} is out of range for bound {bound}"
            )));
        }
        if !seen.insert(axis) {
            return Err(TypeError::invalid(format!("`{operation_name}` `{field}` must be unique but got {axes:?}")));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::RaggedAxis;
    use crate::arrays::batching::DynamicArrayExtentBatchingPolicy;
    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, DataType, DimensionBounds, DimensionType,
        DimensionValue, DimensionVariable, Layout, LogicalMesh, Memory, MeshAxis, MeshAxisType, Sharding,
        ShardingDimension, StridedLayout,
    };
    use crate::batching::batch;
    use crate::contexts::Context;
    use crate::differentiation::differentiate_at;
    use crate::macros::{
        check_operation_batching, check_operation_partial_evaluation, check_operation_transposition,
        check_operation_type_inference,
    };
    use crate::operations::manipulation::slicing::Slice;
    use crate::parameters::Placeholder;
    use crate::programs::{EmptyRegionDriver, ProgramBuilder};
    use crate::tracing::Trace;

    use super::*;

    fn indices_type(dimensions: Vec<usize>) -> ArrayType {
        ArrayType::new(DataType::I32, Shape::new(dimensions.into_iter().map(Dimension::Static).collect()))
    }

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
        Add(crate::operations::math::AddOperation<ArrayType>),
        Gather(GatherOperation),
        Scatter(ScatterOperation),
        Broadcast(BroadcastOperation),
    }

    impl<V: Value<Type = ArrayType>> TransposableOperation<V, TestGatherOperation<V>> for TestGatherOperation<V> {
        fn transpose<D: TranspositionDriver<V, TestGatherOperation<V>>>(
            &self,
            context: &mut TranspositionContext<V, TestGatherOperation<V>>,
            driver: &D,
            inputs: &[PartialValue<Tracer<TracingContext<V, TestGatherOperation<V>>>>],
            outputs: &[MaybeZero<Tracer<TracingContext<V, TestGatherOperation<V>>>>],
            accumulators: &[CotangentAccumulator],
        ) -> Result<(), DifferentiationError> {
            match self {
                Self::Gather(operation) => operation.transpose(context, driver, inputs, outputs, accumulators),
                _ => Err(ProgramError::UnsupportedOperation {
                    message: format!("{} is not transposed in this test enum", self.name()),
                }
                .into()),
            }
        }
    }

    #[test]
    fn test_gather() {
        // Take whole rows of a [3, 2] matrix indexed by a [2, 1] index array: offset axis 1 carries the row (slice
        // sizes [1, 2]); axis 0 (the collapsed row axis) is driven by the start index.
        let dimensions = GatherDimensionNumbers::new(vec![1], vec![0], vec![0]);
        let operation = GatherOperation::new(dimensions, vec![1, 2]);
        assert_eq!(operation.name(), GATHER_OPERATION_NAME);
        assert_eq!(operation.slice_sizes(), &[1, 2]);

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
    }

    #[test]
    fn test_gather_type_inference() {
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
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
                    error = "`gather` indices must be integer-typed but have type f32[2, 1]",
                },
                {
                    input_types = [host_operand.clone(), host_indices],
                    output_types = [host_output],
                },
                {
                    input_types = [host_operand, indices.clone()],
                    error = "`gather` operand and indices must share one memory space but reside in Host[Pinned] and \
                             Device",
                },
            ],
        );

        // Query-batch axes come directly from the indices array. A dynamic query extent therefore preserves the same
        // identity in the output and needs no separate first-class dimension input on `gather`.
        let query = DimensionVariable::new("query", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_indices =
            ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(query.clone()), Dimension::Static(1)]));
        assert_eq!(
            operation.infer_output_types(&[operand.clone(), dynamic_indices], &[]),
            Ok(vec![ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(query), Dimension::Static(2)]),)]),
        );
    }

    #[test]
    fn test_gather_type_inference_invalid_dimension_maps() {
        let operation = GatherOperation::new(
            GatherDimensionNumbers::new(vec![], vec![2], vec![2]).with_batching_dimensions(vec![0, 1], vec![0, 0]),
            vec![1, 1, 1],
        );
        assert_eq!(
            operation.infer_output_types(
                &[ArrayType::new_static(DataType::F32, [2, 2, 3]), ArrayType::new_static(DataType::I32, [2, 1])],
                &[]
            ),
            Err(TypeError::invalid("`gather` `start_indices_batching_dimensions` must be unique but got [0, 0]"))
        );
        let operation = GatherOperation::new(
            GatherDimensionNumbers::new(vec![1], vec![], vec![0]).with_batching_dimensions(vec![0], vec![0]),
            vec![1, 3],
        );
        assert_eq!(
            operation.infer_output_types(
                &[ArrayType::new_static(DataType::F32, [2, 3]), ArrayType::new_static(DataType::I32, [2, 1])],
                &[]
            ),
            Err(TypeError::invalid("`gather` `start_index_map` and `operand_batching_dimensions` must be disjoint"))
        );
        assert_eq!(
            operation
                .infer_output_types(&[], &[RegionInterface::new(vec![], vec![], crate::programs::EffectClasses::NONE)]),
            Err(TypeError::invalid("expected 0 regions but got 1"))
        );
    }

    #[test]
    fn test_gather_type_inference_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let indices = ArrayType::new_static(DataType::I32, [1, 1]);
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![], vec![0]), vec![4]);
        let expected = ArrayType::new_static(DataType::F32, [1, 4])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::Replicated, ShardingDimension::sharded(["x"])])
                    .unwrap(),
            )
            .unwrap();
        // Full windows retain placement even on an explicitly indexed axis.
        assert_eq!(input.gather(&indices, &operation), Ok(expected));
        let partial = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![], vec![]), vec![2]);
        assert_eq!(
            input.gather(&ArrayType::new_static(DataType::I32, [1, 0]), &partial),
            Err(TypeError::invalid("`gather` partial sharded windows require explicit output sharding").into())
        );

        let batched_input = ArrayType::new_static(DataType::F32, [2, 3])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated])
                    .unwrap(),
            )
            .unwrap();
        let batched_indices = ArrayType::new_static(DataType::I32, [2, 1]);
        let batched = GatherOperation::new(
            GatherDimensionNumbers::new(vec![], vec![1], vec![1]).with_batching_dimensions(vec![0], vec![0]),
            vec![1, 1],
        );
        let expected = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        assert_eq!(batched_input.gather(&batched_indices, &batched), Ok(expected));

        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let requested = Sharding::new(other_mesh, vec![ShardingDimension::Replicated; 2]).unwrap();
        assert_eq!(
            input.gather(&indices, &operation.clone().with_output_sharding(requested)),
            Err(TypeError::invalid("`gather` requested output sharding uses a different mesh").into())
        );
        let requested = Sharding::new(mesh.clone(), vec![ShardingDimension::Replicated; 2])
            .unwrap()
            .with_unreduced_axes(["x".to_string()])
            .unwrap();
        assert_eq!(
            input.gather(&indices, &operation.clone().with_output_sharding(requested)),
            Err(TypeError::invalid("`gather` requested output sharding changes reduction or manual-axis state").into())
        );
        let reduced_indices = indices
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::Replicated; 2])
                    .unwrap()
                    .with_reduced_axes(["x".to_string()])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            input.gather(&reduced_indices, &operation),
            Err(TypeError::invalid("`gather` indices cannot carry reduced or unreduced mesh axes").into())
        );
    }
    #[test]
    fn test_gather_type_inference_reduction_state() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::Replicated])
                    .unwrap()
                    .with_unreduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        let indices = ArrayType::new_static(DataType::I32, [1, 1]);
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![], vec![0]), vec![4]);
        let expected = ArrayType::new_static(DataType::F32, [1, 4])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::Replicated; 2])
                    .unwrap()
                    .with_unreduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(input.gather(&indices, &operation), Ok(expected));
        assert_eq!(
            input.gather(&indices, &operation.clone().with_mode(GatherScatterMode::FillOrDrop)),
            Err(TypeError::invalid("`gather` fill mode does not support unreduced inputs").into())
        );
        let distributed_indices = ArrayType::new_static(DataType::I32, [2, 1])
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated]).unwrap(),
            )
            .unwrap();
        assert_eq!(
            input.gather(&distributed_indices, &operation),
            Err(TypeError::invalid("`gather` reduction-state inputs require replicated, invariant indices").into())
        );
    }

    #[test]
    fn test_gather_interpretation() {
        // Interpretation handles each out-of-bounds mode explicitly.
        let scalar_dimensions = GatherDimensionNumbers::new(vec![], vec![0], vec![0]);
        let scalar_indices = Array::from_elements::<i32>(indices_type(vec![2, 1]), &[1, 5]).unwrap();
        let run = |mode| {
            Array::vector(vec![10.0, 20.0, 30.0, 40.0])
                .unwrap()
                .gather(&scalar_indices, &GatherOperation::new(scalar_dimensions.clone(), vec![1]).with_mode(mode))
                .unwrap()
                .to_f64s()
        };
        assert_eq!(run(GatherScatterMode::Clip), vec![20.0, 40.0]);
        assert_eq!(run(GatherScatterMode::PromiseInBounds), vec![20.0, 40.0]);
        let filled = run(GatherScatterMode::FillOrDrop);
        assert_eq!(filled[0], 20.0);
        assert!(filled[1].is_nan());

        // Slicing and reshaping change the gather window geometry without changing the element type.
        let input = Array::from_elements(
            ArrayType::new_static(DataType::F32, [2, 4]),
            &[0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        )
        .unwrap();
        let windows = input
            .slice(&[0, 1], &[2, 4], &[1, 1])
            .unwrap()
            .reshape(Shape::new(vec![3.into(), 2.into()]))
            .unwrap();
        let indices = Array::from_elements(ArrayType::new_static(DataType::I32, [2, 1]), &[2_i32, 0]).unwrap();
        assert_eq!(
            windows.gather(
                &indices,
                &GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]),
            ),
            Ok(Array::from_elements(ArrayType::new_static(DataType::F32, [2, 2]), &[6.0_f32, 7.0, 1.0, 2.0]).unwrap()),
        );
    }

    #[test]
    fn test_gather_partial_evaluation() {
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
        // Partial evaluation folds fully known gathers and residualizes an unknown data input with known indices.
        let operand_value = Array::matrix(3, 2, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let indices_value = Array::from_elements::<i32>(indices_type(vec![2, 1]), &[0, 2]).unwrap();
        let expected = Array::matrix(2, 2, vec![0.0, 1.0, 4.0, 5.0]).unwrap();
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
    }

    #[test]
    fn test_gather_batching() {
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
        let indices_value = Array::from_elements::<i32>(indices_type(vec![2, 1]), &[0, 2]).unwrap();
        // Dimension-number lifting preserves item boundaries without expanding one operation per item.
        check_operation_batching!(
            @exact,
            operation = operation.clone(),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::from_elements::<f64>(
                        ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into(), 2.into()])),
                        &(0..12).map(|value| value as f64).collect::<Vec<_>>(),
                    ).unwrap()),
                    (@replicated, indices_value),
                ],
                outputs = [(@mapped(axis = 0), Array::from_elements::<f64>(
                    ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 2.into(), 2.into()])),
                    &[0.0, 1.0, 4.0, 5.0, 6.0, 7.0, 10.0, 11.0],
                ).unwrap())],
            }],
        );
        check_operation_batching!(
            @exact,
            operation = GatherOperation::new(GatherDimensionNumbers::new(Vec::new(), vec![0], vec![0]), vec![1]),
            axis_size = 0,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::from_elements::<f64>(
                        ArrayType::new(DataType::F64, Shape::new(vec![0.into(), 3.into()])),
                        &[],
                    ).unwrap()),
                    (@mapped(axis = 0), Array::from_elements::<i32>(
                        ArrayType::new(DataType::I32, Shape::new(vec![0.into(), 1.into(), 1.into()])),
                        &[],
                    ).unwrap()),
                ],
                outputs = [(@mapped(axis = 0), Array::from_elements::<f64>(
                    ArrayType::new(DataType::F64, Shape::new(vec![0.into(), 1.into()])),
                    &[],
                ).unwrap())],
            }],
        );

        // Mapped indices add a leading output axis, whether the input is shared or independently mapped at a
        // nonleading axis. Repeated indices remain repeated reads within the corresponding item.
        check_operation_batching!(
            @exact,
            operation = GatherOperation::new(GatherDimensionNumbers::new(Vec::new(), vec![0], vec![0]), vec![1]),
            axis_size = 2,
            cases = [
                {
                    inputs = [
                        (@replicated, Array::vector(vec![1_f64, 2., 3.]).unwrap()),
                        (@mapped(axis = 0), Array::from_elements(ArrayType::new_static(DataType::I32, [2, 2, 1]), &[2_i32, 0, 1, 1]).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 0), Array::matrix(2, 2, vec![3_f64, 1., 2., 2.]).unwrap())],
                },
                {
                    inputs = [
                        (@mapped(axis = 1), Array::matrix(3, 2, vec![1_f64, 4., 2., 5., 3., 6.]).unwrap()),
                        (@mapped(axis = 0), Array::from_elements(ArrayType::new_static(DataType::I32, [2, 2, 1]), &[2_i32, 0, 1, 1]).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 0), Array::matrix(2, 2, vec![3_f64, 1., 5., 5.]).unwrap())],
                },
            ],
        );

        // Empty batching must not request an intermediate zero value from an element format without zero.
        check_operation_batching!(
            @exact,
            operation = GatherOperation::new(GatherDimensionNumbers::new(Vec::new(), vec![0], vec![0]), vec![1]),
            axis_size = 0,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::new(ArrayType::new_static(DataType::F8E8M0FNU, [0, 3]), Vec::new()).unwrap()),
                    (@mapped(axis = 0), Array::from_elements(ArrayType::new_static(DataType::I32, [0, 1, 1]), &[] as &[i32]).unwrap()),
                ],
                outputs = [(@mapped(axis = 0), Array::new(ArrayType::new_static(DataType::F8E8M0FNU, [0, 1]), Vec::new()).unwrap())],
            }],
        );

        // Indices-only and jointly mapped gathers can preserve a first-class mapped extent: neither needs that
        // extent encoded in the static slice sizes.
        for mapped_input in [false, true] {
            let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(9)).unwrap());
            let extent = trace.input(DimensionType::new(items.clone()).into());
            let input_shape = if mapped_input {
                Shape::new(vec![Dimension::Dynamic(items.clone()), Dimension::Static(3)])
            } else {
                Shape::new(vec![Dimension::Static(3)])
            };
            let input = trace.input(ArrayType::new(DataType::F32, input_shape).into());
            let input = <_ as ValueProjection<ArrayType>>::into_projected(input).unwrap();
            let input = if mapped_input {
                ArrayBatch::new(input, BatchAxis::new(0)).unwrap()
            } else {
                ArrayBatch::replicated(input)
            };
            let indices = trace.input(
                ArrayType::new(
                    DataType::I32,
                    Shape::new(vec![Dimension::Dynamic(items.clone()), Dimension::Static(1), Dimension::Static(1)]),
                )
                .into(),
            );
            let indices = <_ as ValueProjection<ArrayType>>::into_projected(indices).unwrap();
            let context = BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
                ProjectedContext::new(trace.clone()),
                extent,
            );
            let operation = GatherOperation::new(GatherDimensionNumbers::new(Vec::new(), vec![0], vec![0]), vec![1]);
            let (outputs, _) = operation
                .batch(
                    &context,
                    &EmptyRegionDriver,
                    &[input.clone(), ArrayBatch::new(indices, BatchAxis::new(0)).unwrap()],
                )
                .unwrap()
                .into_parts();
            assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
            assert_eq!(outputs[0].r#type().shape(), &Shape::new(vec![Dimension::Dynamic(items), Dimension::Static(1)]));
            if mapped_input {
                let indices = trace.constant(ArrayIrValue::Array(Array::matrix(1, 1, vec![0_i32]).unwrap()));
                let indices = <_ as ValueProjection<ArrayType>>::into_projected(indices).unwrap();
                assert_eq!(
                    operation
                        .batch(&context, &EmptyRegionDriver, &[input, ArrayBatch::replicated(indices)])
                        .unwrap_err(),
                    BatchingError::UnsupportedOperation {
                        message: "`gather` with only its input mapped requires a statically known mapped extent"
                            .to_string(),
                    }
                );
            }
        }

        // Ragged input extents must never be replaced with packed storage extents while selecting windows.
        let variable = DimensionVariable::new("length", DimensionBounds::new(1, Some(4)).unwrap());
        let input = ArrayBatch::new(Array::matrix(2, 3, vec![1_f64, 2., 3., 4., 5., 6.]).unwrap(), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, Array::vector(vec![1_i32, 3]).unwrap(), variable, vec![0])])
            .unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array>::new(), 2);
        let operation = GatherOperation::new(GatherDimensionNumbers::new(Vec::new(), vec![0], vec![0]), vec![1]);
        assert_eq!(
            operation
                .batch(
                    &context,
                    &EmptyRegionDriver,
                    &[input, ArrayBatch::replicated(Array::matrix(1, 1, vec![0_i32]).unwrap())]
                )
                .unwrap_err(),
            BatchingError::UnsupportedOperation {
                message: "`gather` does not support bounded ragged array inputs".to_string(),
            }
        );
    }

    #[test]
    fn test_gather_differentiation() {
        // Forward mode selects the input coordinate feeding each gathered output.
        let jacobian = differentiate_at(Array::matrix(3, 2, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap())
            .jacobian_forward(|operand| {
                let indices = index_array(&operand, vec![2, 1], vec![0, 2]);
                let operation =
                    GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
                Ok(operand.gather(&indices, &operation).unwrap())
            })
            .unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.output_type().static_shape().unwrap().as_slice(), &[2, 2]);
        assert_eq!(block.input_type().static_shape().unwrap().as_slice(), &[3, 2]);
        assert_eq!(
            block.value().to_f64s(),
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, //
            ],
        );

        // Both the default NaN fill and an explicit nonzero fill are constant in the input. Neither may appear in
        // the tangent: only the one in-bounds selected coordinate contributes to this Jacobian.
        for fill in [None, Some(Array::scalar(99_f64).unwrap())] {
            let mut operation =
                GatherOperation::new(GatherDimensionNumbers::new(Vec::new(), vec![0], vec![0]), vec![1])
                    .with_mode(GatherScatterMode::FillOrDrop);
            if let Some(fill) = fill {
                operation = operation.with_fill_value(fill).unwrap();
            }
            let jacobian = differentiate_at(Array::vector(vec![10_f64, 20.]).unwrap())
                .jacobian_forward(|input| {
                    let indices = index_array(&input, vec![3, 1], vec![-1, 1, 5]);
                    input.gather(&indices, &operation)
                })
                .unwrap();
            assert_eq!(
                jacobian.iter_blocks().next().unwrap().value().elements::<f64>(),
                Ok(vec![0., 0., 0., 1., 0., 0.])
            );
        }
    }

    #[test]
    fn test_gather_transposition() {
        // Take rows 0 and 2 of a [3, 2] input: the input is linear and the [2, 1] index array is the known
        // input. The gathered output and its cotangent have shape [2, 2].
        let dimensions = GatherDimensionNumbers::new(vec![1], vec![0], vec![0]);
        let operation = GatherOperation::new(dimensions, vec![1, 2]);
        let operand = Array::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let indices = Array::from_elements::<i32>(indices_type(vec![2, 1]), &[0, 2]).unwrap();
        let cotangent = Array::matrix(2, 2, vec![10.0, 20.0, 30.0, 40.0]).unwrap();
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
                input_cotangents = [Array::matrix(3, 2, vec![10.0, 20.0, 0.0, 0.0, 30.0, 40.0]).unwrap()],
            }],
        );

        // The transpose explicitly restores the input's distribution even when the forward gather requested a
        // different output placement. Scatter's zero base also preserves input strides and host memory.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let extent = DimensionVariable::new("extent", DimensionBounds::new(4, Some(5)).unwrap());
        for (dimension, sharding) in [
            (Dimension::Static(4), None),
            (Dimension::Static(4), Some(sharded.clone())),
            (Dimension::Dynamic(extent.clone()), None),
            (Dimension::Dynamic(extent), Some(sharded)),
        ] {
            let input_type = ArrayType::new(DataType::F64, Shape::new(vec![dimension]))
                .with_layout(Layout::Strided(StridedLayout::new(vec![16])))
                .with_memory(Memory::Host { pinned: true })
                .with_sharding(sharding)
                .unwrap();
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let input = builder.add_input(input_type.clone().into());
            let indices = builder.add_constant(ArrayIrValue::Array(
                Array::from_elements(
                    ArrayType::new_static(DataType::I32, [2, 1]).with_memory(Memory::Host { pinned: true }),
                    &[1_i32, 3],
                )
                .unwrap(),
            ));
            let operation = GatherOperation::new(GatherDimensionNumbers::new(Vec::new(), vec![0], vec![0]), vec![1])
                .with_output_sharding(Sharding::replicated(mesh.clone(), 1));
            let output = builder
                .add_instruction(
                    ArrayIrOperation::Array(ArrayOperation::Gather(operation)),
                    Vec::new(),
                    vec![input, indices],
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
            assert_eq!(
                program.linearize().unwrap().pullback().unwrap().output_types(),
                vec![ArrayIrType::Array(input_type.cotangent().unwrap())]
            );
        }
    }

    // The homogeneous gather transpose scatters into a zero of the input's cotangent type, and the homogeneous
    // `ArrayType` family has no constructor that can supply a runtime extent for one. A dynamically shaped input is
    // therefore part of the rule's rejected contract rather than an accident of zero construction.
    #[test]
    fn test_gather_transposition_rejects_dynamic_operand_shapes() {
        let rows = DimensionVariable::new("rows", DimensionBounds::new(4, Some(8)).unwrap());
        let dynamic_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows), Dimension::Static(2)]));

        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let operand = builder.add_input(dynamic_type);
        let indices = builder.add_constant(Array::from_elements::<i32>(indices_type(vec![2, 1]), &[0, 2]).unwrap());
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
        let output = builder.add_instruction(operation, Vec::new(), vec![operand, indices], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(
            program.transpose_with_respect_to(&[0], &[]).unwrap_err(),
            TypeError::invalid("`gather` transpose requires a statically shaped operand but got f32[rows, 2]").into(),
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
                "`gather` start_index_map has length 2 but the index vector extent is 1".to_string()
            )),
        );

        // A collapsed axis must have slice size 1.
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![2, 2]);
        assert_eq!(
            operation.infer_output_types(&[operand.clone(), indices.clone()], &[]),
            Err(TypeError::invalid(
                "`gather` collapsed slice dimension 0 must have slice size 1 but has 2".to_string()
            )),
        );

        // offset_dimensions count must equal the non-collapsed, non-batching input axes (here 1).
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1, 2], vec![0], vec![0]), vec![1, 2]);
        assert_eq!(
            operation.infer_output_types(&[operand, indices], &[]),
            Err(TypeError::invalid(
                "`gather` offset_dimensions has length 2 but the operand has 1 non-collapsed, non-batching \
                          axes"
                    .to_string()
            )),
        );
    }

    #[test]
    fn test_array_type_gather() {
        let exact = Dimension::Dynamic(DimensionVariable::new("batch", DimensionBounds::new(0, Some(1)).unwrap()));
        let operation = GatherOperation::new(
            GatherDimensionNumbers::new(vec![], vec![1], vec![1]).with_batching_dimensions(vec![0], vec![0]),
            vec![0, 1],
        );
        // Specialization can retain an exact nominal dimension on one side of a paired batch while the other is
        // already static. Both descriptions prove the same extent without equating unrelated symbolic dimensions.
        for input_batch in [Dimension::Static(0), exact.clone()] {
            for query_batch in [Dimension::Static(0), exact.clone()] {
                let input = ArrayType::new(DataType::F64, Shape::new(vec![input_batch.clone(), Dimension::Static(4)]));
                let indices = ArrayType::new(
                    DataType::I32,
                    Shape::new(vec![query_batch.clone(), Dimension::Static(2), Dimension::Static(1)]),
                );
                assert_eq!(
                    input.gather(&indices, &operation).unwrap().shape(),
                    &Shape::new(vec![query_batch, Dimension::Static(2)])
                );
            }
        }
        let input = ArrayType::new(
            DataType::F64,
            Shape::new(vec![
                Dimension::Dynamic(DimensionVariable::new("batch", DimensionBounds::new(0, Some(3)).unwrap())),
                Dimension::Static(4),
            ]),
        );
        let indices = ArrayType::new(
            DataType::I32,
            Shape::new(vec![
                Dimension::Dynamic(DimensionVariable::new("batch", DimensionBounds::new(0, Some(3)).unwrap())),
                Dimension::Static(2),
                Dimension::Static(1),
            ]),
        );
        assert!(matches!(input.gather(&indices, &operation), Err(ProgramError::Type(_))));

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

        // Sharding the start-indexed input axis over an explicit mesh axis is ambiguous without an output sharding.
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

    #[test]
    fn test_array_ir_gather_differentiation() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(6)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent.clone())]));
        let indices_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(3), Dimension::Static(1)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.clone().into());
        let indices = builder.add_input(indices_type.clone().into());
        let operation = GatherOperation::new(GatherDimensionNumbers::new(Vec::new(), vec![0], vec![0]), vec![1]);
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Gather(operation)),
                Vec::new(),
                vec![input, indices],
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

        // A dynamically shaped input reaches the mixed member rule, never the homogeneous array rule: the composite
        // rule intercepts it and delegates only fully static inputs downward. That routing is what keeps the
        // homogeneous `gather` and `slice` transpose rules static-only, and it is observable in the residual
        // signature, because retaining a runtime extent as a first-class dimension is something the homogeneous rule
        // cannot express. The tangent boundary is therefore the input tangent followed by the indices and that
        // extent.
        assert_eq!(linearization.residual_count(), 2);
        assert_eq!(
            linearization.tangent().input_types(),
            &[
                input_type.tangent().unwrap().into(),
                indices_type.clone().into(),
                ArrayIrType::Dimension(DimensionType::new(extent)),
            ],
        );
        assert!(linearization.tangent().to_string().contains("linear_call [residual_count=2]"));
        let indices = ArrayIrValue::Array(Array::from_elements::<i32>(indices_type, &[1, 1, 3]).unwrap());
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0]).unwrap()), indices])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::vector(vec![20.0_f64, 20.0, 40.0]).unwrap()));
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap())];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![2.0_f64, 2.0, 4.0]).unwrap())]),
        );
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::vector(vec![2.0_f64, 3.0, 5.0]).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0.0_f64, 5.0, 0.0, 5.0]).unwrap())]),
        );

        // The dynamic member's residual-carrying linear region must use zero fill as well, and its scatter adjoint
        // drops the same out-of-bounds coordinates.
        for fill in [None, Some(Array::scalar(99_f64).unwrap())] {
            let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(6)).unwrap());
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let input =
                builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)])).into());
            let indices = builder.add_constant(ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::I32, [3, 1]), &[-1_i32, 1, 5]).unwrap(),
            ));
            let mut operation =
                GatherOperation::new(GatherDimensionNumbers::new(Vec::new(), vec![0], vec![0]), vec![1])
                    .with_mode(GatherScatterMode::FillOrDrop);
            if let Some(fill) = fill {
                operation = operation.with_fill_value(fill).unwrap();
            }
            let output = builder
                .add_instruction(
                    ArrayIrOperation::Array(ArrayOperation::Gather(operation)),
                    Vec::new(),
                    vec![input, indices],
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
            let mut primals = linearization
                .primal()
                .interpret(vec![ArrayIrValue::Array(Array::vector(vec![10_f64, 20.]).unwrap())])
                .unwrap();
            let residuals = primals.split_off(1);
            let mut tangent_inputs = vec![ArrayIrValue::Array(Array::vector(vec![2_f64, 3.]).unwrap())];
            tangent_inputs.extend(residuals.clone());
            assert_eq!(
                linearization.tangent().interpret(tangent_inputs),
                Ok(vec![ArrayIrValue::Array(Array::vector(vec![0_f64, 3., 0.]).unwrap())])
            );
            let mut cotangent_inputs = vec![ArrayIrValue::Array(Array::vector(vec![1_f64, 1., 1.]).unwrap())];
            cotangent_inputs.extend(residuals);
            assert_eq!(
                linearization.pullback().unwrap().interpret(cotangent_inputs),
                Ok(vec![ArrayIrValue::Array(Array::vector(vec![0_f64, 1.]).unwrap())])
            );
        }
    }

    #[test]
    fn test_array_gather() {
        // Gather rows 2 and 0 of a 3x2 matrix.
        let operand = Array::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let indices = Array::matrix(2, 1, vec![2i64, 0]).unwrap();
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
        let gathered = operand.gather(&indices, &operation).unwrap();
        assert_eq!(gathered.r#type().into_owned(), ArrayType::new_static(DataType::F64, [2, 2]));
        assert_eq!(gathered.to_f64s(), vec![5.0, 6.0, 1.0, 2.0]);

        // In-bounds and clipping modes do not materialize an unused zero fill, so they work for formats that cannot
        // represent zero.
        let operand = Array::new(ArrayType::new_static(DataType::F8E8M0FNU, [2]), vec![0x7f, 0x80]).unwrap();
        let indices = Array::matrix(1, 1, vec![1i64]).unwrap();
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![], vec![0], vec![0]), vec![1]);
        assert_eq!(
            operand.gather(&indices, &operation),
            Array::new(ArrayType::new_static(DataType::F8E8M0FNU, [1]), vec![0x80])
        );

        // Gather reads both a reversed input and reversed sub-byte indices through their physical addressing. An
        // out-of-bounds query in fill-or-drop mode writes the default unsigned maximum into the dense result.
        let operand_type =
            ArrayType::new_static(DataType::U16, [3]).with_layout(Layout::Strided(StridedLayout::new(vec![-2])));
        let operand = Array::from_elements(operand_type, &[10u16, 20, 30]).unwrap();
        let indices_type =
            ArrayType::new_static(DataType::I4, [3, 1]).with_layout(Layout::Strided(StridedLayout::new(vec![-1, 1])));
        let indices =
            Array::from_elements(indices_type, &[i4::new(2).unwrap(), i4::new(-1).unwrap(), i4::new(1).unwrap()])
                .unwrap();
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![], vec![0], vec![0]), vec![1])
            .with_mode(GatherScatterMode::FillOrDrop);
        let gathered = operand.gather(&indices, &operation).unwrap();
        assert_eq!(gathered.elements::<u16>(), Ok(vec![30, u16::MAX, 20]));
        assert_eq!(gathered.storage_bytes(), [30, 0, 255, 255, 20, 0]);
    }

    #[test]
    fn test_array_gather_extreme_indices() {
        let input = Array::vector(vec![10_i32, 20, 30, 40]).unwrap();
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![], vec![0]), vec![2])
            .with_mode(GatherScatterMode::Clip);
        assert_eq!(
            input.gather(&Array::matrix(2, 1, vec![0_u64, u64::MAX]).unwrap(), &operation),
            Array::matrix(2, 2, vec![10_i32, 20, 30, 40]),
        );
        assert_eq!(
            input.gather(&Array::matrix(2, 1, vec![i64::MIN, i64::MAX]).unwrap(), &operation),
            Array::matrix(2, 2, vec![10_i32, 20, 30, 40]),
        );

        // Adding a window offset to an invalid maximal start must not overflow before the query is filled.
        let operation = operation.with_mode(GatherScatterMode::FillOrDrop);
        let signed = input.gather(&Array::matrix(1, 1, vec![i64::MAX]).unwrap(), &operation).unwrap();
        let unsigned = input.gather(&Array::matrix(1, 1, vec![u64::MAX]).unwrap(), &operation).unwrap();
        assert_eq!(signed, unsigned);
        assert_eq!(signed, Array::matrix(1, 2, vec![i32::MIN, i32::MIN]).unwrap());
    }

    #[test]
    fn test_array_ir_gather() {
        let input = ArrayIrValue::Array(Array::vector(vec![10_i32, 20, 30]).unwrap());
        let indices = ArrayIrValue::Array(Array::matrix(2, 1, vec![2_i32, 0]).unwrap());
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![], vec![0], vec![0]), vec![1]);
        assert_eq!(
            input.gather(&indices, &operation),
            Ok(ArrayIrValue::Array(Array::vector(vec![30_i32, 10]).unwrap()))
        );
        let dimension = ArrayIrValue::Dimension(DimensionValue::constant(1).unwrap());
        assert_eq!(
            input.gather(&dimension, &operation),
            Err(TypeError::invalid("expected array type but got dimension type").into())
        );
        assert_eq!(
            dimension.gather(&indices, &operation),
            Err(TypeError::invalid("expected array type but got dimension type").into())
        );

        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let staged_input = context.input(input.r#type().into_owned());
        let staged_indices = context.lift(indices).unwrap();
        assert_eq!(
            ValueProjection::<ArrayType>::into_projected(staged_input.clone())
                .unwrap()
                .gather(&ValueProjection::<ArrayType>::into_projected(staged_indices.clone()).unwrap(), &operation)
                .unwrap()
                .r#type()
                .into_owned(),
            ArrayType::new_static(DataType::I32, [2])
        );
    }

    #[test]
    fn test_gather_resolved_fill_value() {
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![], vec![0], vec![0]), vec![1]);
        assert_eq!(operation.resolved_fill_value(DataType::I64), Array::scalar(i64::MIN));
        assert_eq!(operation.resolved_fill_value(DataType::U64), Array::scalar(u64::MAX));
        assert_eq!(operation.resolved_fill_value(DataType::Boolean), Array::scalar(true));
        assert_eq!(operation.resolved_fill_value(DataType::I1), Array::scalar(i1::new(-1).unwrap()));
        assert_eq!(operation.resolved_fill_value(DataType::U4), Array::scalar(u4::new(15).unwrap()));
        assert!(operation.resolved_fill_value(DataType::F32).unwrap().elements::<f32>().unwrap()[0].is_nan());
        let complex = operation
            .resolved_fill_value(DataType::C64)
            .unwrap()
            .elements::<num_complex::Complex<f32>>()
            .unwrap()[0];
        assert!(complex.re.is_nan());
        assert_eq!(complex.im, 0.0);

        // Explicit NaN payloads survive operation cloning and eager filling without numeric conversion.
        let fill = Array::new(ArrayType::scalar(DataType::F32), 0x7fc12345_u32.to_ne_bytes().to_vec()).unwrap();
        let operation = operation.with_mode(GatherScatterMode::FillOrDrop).with_fill_value(fill.clone()).unwrap();
        assert_eq!(operation.fill_value().unwrap().storage_bytes(), fill.storage_bytes());
        assert_eq!(operation.resolved_fill_value(DataType::F32).unwrap().storage_bytes(), fill.storage_bytes());
        assert_eq!(
            operation.resolved_fill_value(DataType::I32),
            Err(TypeError::invalid("`gather` fill data type `f32` does not match input data type `i32`").into())
        );
        let output = Array::vector(vec![1.0_f32])
            .unwrap()
            .gather(&Array::matrix(1, 1, vec![2_i32]).unwrap(), &operation)
            .unwrap();
        assert_eq!(output.storage_bytes(), fill.storage_bytes());
        assert_eq!(
            operation.clone().with_fill_value(Array::vector(vec![1.0_f32]).unwrap()),
            Err(TypeError::invalid("`gather` fill value must be a numeric or Boolean scalar"))
        );
    }

    #[test]
    fn test_array_gather_axis() {
        let input = Array::matrix(2, 3, vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(
            input.gather_axis(&Array::vector(vec![2_i32, 0]).unwrap(), -1, GatherScatterMode::Clip),
            Array::matrix(2, 2, vec![3_i32, 1, 6, 4])
        );
        assert_eq!(
            input.gather_axis(&Array::scalar(1_i32).unwrap(), 0, GatherScatterMode::Clip),
            Array::vector(vec![4_i32, 5, 6])
        );
        assert_eq!(
            input.gather_axis(&Array::matrix(1, 2, vec![-1_i32, 9]).unwrap(), 1, GatherScatterMode::Clip),
            Array::from_elements(ArrayType::new_static(DataType::I32, [2, 1, 2]), &[1_i32, 3, 4, 6])
        );
        assert_eq!(
            input.gather_axis(&Array::vector(vec![-1_i32, 1]).unwrap(), 0, GatherScatterMode::FillOrDrop),
            Array::matrix(2, 3, vec![i32::MIN, i32::MIN, i32::MIN, 4, 5, 6])
        );
        assert_eq!(
            input.gather_axis(&Array::vector(Vec::<i32>::new()).unwrap(), 0, GatherScatterMode::Clip),
            Array::matrix(0, 3, Vec::<i32>::new())
        );
        assert_eq!(
            input.gather_axis(&Array::scalar(0_i32).unwrap(), 2, GatherScatterMode::Clip),
            Err(TypeError::invalid("axis 2 is out of bounds for rank 2").into())
        );

        // Selecting one element does not require the selected axis's runtime extent as a window parameter.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(6)).unwrap());
        let input_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let (output_type, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |(input, indices)| input.gather_axis(&indices, 0, GatherScatterMode::Clip),
            (input_type, ArrayType::new_static(DataType::I32, [2])),
        )
        .unwrap();
        assert_eq!(output_type, ArrayType::new_static(DataType::I32, [2]));
        assert_eq!(
            program.interpret((
                Array::from_elements(ArrayType::new_static(DataType::I32, [3]), &[10_i32, 20, 30]).unwrap(),
                Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[2_i32, 0]).unwrap()
            )),
            Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[30_i32, 10]),
        );

        // A complete window along an unselected axis still needs a host-known size.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(6)).unwrap());
        let result = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |(input, indices)| input.gather_axis(&indices, 1, GatherScatterMode::Clip),
            (
                ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(extent), Dimension::Static(2)])),
                ArrayType::new_static(DataType::I32, [2]),
            ),
        );
        assert_eq!(
            result.unwrap_err(),
            ProgramError::Type(TypeError::invalid("`gather_axis` requires a static extent on unselected axis 0")),
        );
    }

    #[test]
    fn test_dynamic_gather_dynamic_gather_axis() {
        // Both packed inputs are mapped in a mixed operation context. This exercises dimension queries,
        // dynamic query broadcasting, and the projected gather batching rule in one retained graph.
        let input = ArrayIrValue::Array(Array::matrix(2, 4, vec![0_f64, 1., 2., 3., 4., 5., 6., 7.]).unwrap());
        let queries = ArrayIrValue::Array(Array::matrix(2, 2, vec![3_i32, 0, 1, 2]).unwrap());
        let (_, batched_program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(input, queries)| {
                batch(
                    |(input, queries)| input.dynamic_gather_axis(&queries, 0, GatherScatterMode::Clip),
                    (input, queries),
                    (BatchAxis::new(0), BatchAxis::new(0)),
                    BatchAxis::new(0),
                    None,
                )
                .map_err(ProgramError::from)
            },
            (input.r#type().into_owned(), queries.r#type().into_owned()),
        )
        .unwrap();
        assert_eq!(
            batched_program.interpret((input, queries)),
            Ok(ArrayIrValue::Array(Array::matrix(2, 2, vec![3_f64, 0., 5., 6.]).unwrap())),
        );

        let empty = ArrayIrValue::Array(
            Array::from_elements(ArrayType::new_static(DataType::F64, [0]), &[] as &[f64]).unwrap(),
        );
        let indices = ArrayIrValue::Array(
            Array::from_elements(ArrayType::new_static(DataType::I32, [0]), &[] as &[i32]).unwrap(),
        );
        let (_, empty_program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(input, indices)| input.dynamic_gather_axis(&indices, 0, GatherScatterMode::Clip),
            (empty.r#type().into_owned(), indices.r#type().into_owned()),
        )
        .unwrap();
        assert_eq!(empty.dynamic_gather_axis(&indices, 0, GatherScatterMode::Clip).unwrap(), empty);
        assert_eq!(empty_program.interpret((empty.clone(), indices)).unwrap(), empty);
        // The eager empty shortcut uses the gather output metadata, including cleared layout and query placement.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh, vec![ShardingDimension::Replicated]).unwrap();
        let input_type = ArrayType::new_static(DataType::F64, [0])
            .with_layout(Layout::Strided(StridedLayout::new(vec![8])))
            .with_sharding(sharding.clone())
            .unwrap();
        let indices_type = ArrayType::new_static(DataType::I32, [0]).with_sharding(sharding.clone()).unwrap();
        let input = ArrayIrValue::Array(Array::from_elements(input_type.clone(), &[] as &[f64]).unwrap());
        let indices = ArrayIrValue::Array(Array::from_elements(indices_type.clone(), &[] as &[i32]).unwrap());
        let eager = input.dynamic_gather_axis(&indices, 0, GatherScatterMode::Clip).unwrap();
        let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(input, indices)| input.dynamic_gather_axis(&indices, 0, GatherScatterMode::Clip),
            (ArrayIrType::Array(input_type), ArrayIrType::Array(indices_type)),
        )
        .unwrap();
        assert_eq!(program.interpret((input, indices)).unwrap(), eager);
        assert_eq!(
            eager.r#type().into_owned(),
            ArrayIrType::Array(ArrayType::new_static(DataType::F64, [0]).with_sharding(sharding).unwrap(),)
        );
        // Query dimensions replace the selected axis, while paired batching preserves both nonleading and empty
        // untouched dimensions. The same symbolic program is replayed for two concrete input and query extents.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(0, Some(6)).unwrap());
        let queries = DimensionVariable::new("queries", DimensionBounds::new(0, Some(4)).unwrap());
        let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(input, indices)| input.dynamic_gather_axis(&indices, 1, GatherScatterMode::Clip),
            (
                ArrayIrType::Array(ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![Dimension::Dynamic(rows), Dimension::Static(4)]),
                )),
                ArrayIrType::Array(ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(queries)]))),
            ),
        )
        .unwrap();
        for (rows, queries) in [(0, 2), (4, 2), (5, 3), (4, 0)] {
            let input = Array::from_elements(
                ArrayType::new_static(DataType::F64, [rows, 4]),
                &(0..rows * 4).map(|value| value as f64).collect::<Vec<_>>(),
            )
            .unwrap();
            let indices = [2_i32, 0, 3][..queries].to_vec();
            let output = program
                .interpret((
                    ArrayIrValue::Array(input),
                    ArrayIrValue::Array(
                        Array::from_elements(ArrayType::new_static(DataType::I32, [queries]), &indices).unwrap(),
                    ),
                ))
                .unwrap();
            let expected = (0..rows)
                .flat_map(|row| indices.iter().map(move |index| (row * 4 + *index as usize) as f64))
                .collect::<Vec<_>>();
            assert_eq!(
                output,
                ArrayIrValue::Array(
                    Array::from_elements(ArrayType::new_static(DataType::F64, [rows, queries]), &expected,).unwrap()
                )
            );
        }
    }
}
