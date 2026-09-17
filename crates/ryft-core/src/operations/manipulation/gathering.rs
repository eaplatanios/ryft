use std::collections::BTreeSet;
use std::fmt::Display;
use std::sync::Arc;

use crate::arrays::{
    Array, ArrayAddressing, ArrayBatch, ArrayBatchingPolicy, ArrayElement, ArrayExtentBatchingPolicy, ArrayIrType,
    ArrayIrValue, ArrayType, DataType, Dimension, DimensionVariable, LinearResiduals, Shape, Sharding,
    ShardingDimension,
};
use crate::axes::Axis;
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain, EagerContext, ProjectedContext, StagingContext};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDriver, DifferentiationDual,
    DifferentiationError, DifferentiationPolicy, MemberDifferentiableOperation, jvp_projected_operation,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{
    check_count, dispatch_on_array_element_type, impl_differentiable_operation, impl_reference_dischargeable_operation,
};
use crate::operations::constants::constant::DimensionConstant;
use crate::operations::constants::zero::{DynamicZero, Zero, ZeroOperation};
use crate::operations::differentiation::linear_call::LinearCallOperation;
use crate::operations::dimensions::dimension_size::{DimensionSize, DimensionSizeOperation};
use crate::operations::manipulation::broadcasting::{BroadcastOperation, DynamicBroadcast};
use crate::operations::manipulation::reshaping::{Reshape, lift_output_sharding_for_leading_batch_axis};
use crate::operations::manipulation::scattering::{
    ScatterDimensionNumbers, ScatterMode, ScatterOperation, ScatterReductionKind,
};
use crate::operations::manipulation::transposition::Transpose;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, OperationProjection, ProgramError, RegionInterface, TypeError,
    TypeIdentityRenaming, Typed, Value, ValueProjection,
};

/// Determines how [`Gather`] handles windows extending outside its input. Negative indices are out of bounds and they
/// do not count backward from an axis end. Refer to the documentation of [`Gather`] for examples of each policy.
///
/// `V` is the stored constant representation, independent of the gathered value or tracer. Built-in operation families
/// use [`Array`] literals, just as their [`ConstantOperation`](crate::ConstantOperation) variants do. A custom
/// operation family can choose another [`Value`] representation. An explicit fill is a constant attribute and does
/// not introduce another operation input or receive a gradient.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum GatherMode<V: Value<Type = ArrayType> = Array> {
    /// The caller promises every window is in bounds. Violating the promise leaves results and gradients undefined.
    #[default]
    PromiseInBounds,

    /// Clamps each start so the whole window stays in bounds.
    Clip,

    /// Replaces an out-of-bounds window in its entirety. Without an explicit value, uses NaN for floating-point and
    /// complex values, the minimum signed integer, the maximum unsigned integer, or `true` for Booleans.
    Fill {
        /// Optional constant scalar of the input element data type.
        value: Option<Box<V>>,
    },
}

impl<V: Value<Type = ArrayType>> GatherMode<V> {
    /// Returns the canonical name of this [`GatherMode`].
    #[inline]
    pub fn name(&self) -> &'static str {
        match self {
            Self::PromiseInBounds => "promise_in_bounds",
            Self::Clip => "clip",
            Self::Fill { .. } => "fill",
        }
    }
}

impl<V: Value<Type = ArrayType>> Display for GatherMode<V> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fill { value: Some(value) } => write!(formatter, "fill(value={value})"),
            _ => write!(formatter, "{}", self.name()),
        }
    }
}

/// Specification of how the index input and the sliced windows map onto the input and output axes of a [`Gather`]
/// operation, following StableHLO's [`gather`](https://openxla.org/stablehlo/spec#gather) dimension numbers. The index
/// vector dimension is implicit and always the last axis of the indices input (the indices input has shape `[batch...,
/// index_vector]`, where each length-`index_vector` slice is one start-index vector whose components map onto input
/// axes through [`start_index_map`](Self::start_index_map)). To gather with a scalar index per query, give the indices
/// a trailing size-1 axis. The output rank is `offset_dimensions.len() + indices.rank() - 1`. Each output axis named in
/// [`offset_dimensions`](Self::offset_dimensions) carries one sliced window axis (in input-axis order, skipping the
/// collapsed and batching axes). The remaining output axes carry the indices' batch axes in order. See [`Gather`]
/// for diagrams and examples of axis mapping, window sizes, and paired batching.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct GatherDimensionNumbers {
    /// Refer to the documentation of [`offset_dimensions`](Self::offset_dimensions) for more information.
    offset_dimensions: Vec<usize>,

    /// Refer to the documentation of [`collapsed_slice_dimensions`](Self::collapsed_slice_dimensions)
    /// for more information.
    collapsed_slice_dimensions: Vec<usize>,

    /// Refer to the documentation of [`start_index_map`](Self::start_index_map) for more information.
    start_index_map: Vec<usize>,

    /// Refer to the documentation of [`batching_dimensions`](Self::batching_dimensions) for more information.
    batching_dimensions: Vec<(usize, usize)>,
}

impl GatherDimensionNumbers {
    /// Creates a new [`GatherDimensionNumbers`] instance from the provided explicit axis lists. The batching pairs
    /// default to empty; use [`with_batching_dimensions`](Self::with_batching_dimensions) to set them.
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
        Self { offset_dimensions, collapsed_slice_dimensions, start_index_map, batching_dimensions: Vec::new() }
    }

    /// Pairs input axes with query axes so that each query reads from its corresponding input batch. Paired axes
    /// must have equal extents. Input batching axes cannot also be collapsed or indexed by a start vector.
    /// These constraints are checked when inferring the gather result type.
    ///
    /// # Parameters
    ///
    ///   - `batching_dimensions`: Pairs of `(input_axis, indices_axis)`, sorted by input axis. Each input axis has
    ///     window size at most one; indices axes must be distinct and cannot name the trailing index-vector axis.
    #[inline]
    pub fn with_batching_dimensions(mut self, batching_dimensions: Vec<(usize, usize)>) -> Self {
        self.batching_dimensions = batching_dimensions;
        self
    }

    /// Returns the output axes that hold the sliced window (i.e., the "offset" axes), in ascending order. Their count
    /// equals the number of input axes that are neither collapsed nor batching.
    #[inline]
    pub fn offset_dimensions(&self) -> &[usize] {
        &self.offset_dimensions
    }

    /// Returns the input axes whose slice size is `1` and that are removed from the output, in ascending order.
    #[inline]
    pub fn collapsed_slice_dimensions(&self) -> &[usize] {
        &self.collapsed_slice_dimensions
    }

    /// Returns the input axis indexed by each component of a start-index vector (i.e., the last axis of the indices
    /// input). The map's length equals the extent of the indices' index vector dimension.
    #[inline]
    pub fn start_index_map(&self) -> &[usize] {
        &self.start_index_map
    }

    /// Returns the `(input_axis, indices_axis)` pairs, ordered by ascending input axis. Each pair selects the input
    /// batch coordinate from the matching query axis. Input batching window sizes are at most one.
    #[inline]
    pub fn batching_dimensions(&self) -> &[(usize, usize)] {
        &self.batching_dimensions
    }
}

/// Optional behavior and output placement for [`Gather`]. Geometry is supplied separately as [`GatherDimensionNumbers`]
/// and slice sizes. The default promises in-bounds indices, makes no sortedness or non-overlap promise, and infers
/// output [`Sharding`]. Explicit fill values remain constant attributes and do not receive gradients.
#[derive(Clone, Debug, PartialEq)]
pub struct GatherOptions<V: Value<Type = ArrayType> = Array> {
    /// Refer to the documentation of [`mode`](Self::mode) for more information.
    mode: GatherMode<V>,

    /// Refer to the documentation of [`indices_are_sorted`](Self::indices_are_sorted) for more information.
    indices_are_sorted: bool,

    /// Refer to the documentation of [`unique_indices`](Self::unique_indices) for more information.
    unique_indices: bool,

    /// Refer to the documentation of [`output_sharding`](Self::output_sharding) for more information.
    output_sharding: Option<Sharding>,
}

impl<V: Value<Type = ArrayType>> GatherOptions<V> {
    /// Creates options with [`GatherMode::PromiseInBounds`], no sortedness or non-overlap promises, and inferred output
    /// [`Sharding`]. Use the consuming `with_*` functions to override these defaults.
    #[inline]
    pub fn new() -> Self {
        Self {
            mode: GatherMode::PromiseInBounds,
            indices_are_sorted: false,
            unique_indices: false,
            output_sharding: None,
        }
    }

    /// Returns a copy of this [`GatherOptions`] with its out-of-bounds index handling [`GatherMode`] replaced by
    /// `mode`.
    #[inline]
    pub fn with_mode(mut self, mode: GatherMode<V>) -> Self {
        self.mode = mode;
        self
    }

    /// Returns a copy of this [`GatherOptions`] with its sorted-indices promise set to `indices_are_sorted`. When
    /// `true`, the caller promises that start-index vectors are sorted; `false` makes no such promise. Implementations
    /// and transformations may rely on this property. This function does not sort or validate the indices.
    #[inline]
    pub fn with_indices_are_sorted(mut self, indices_are_sorted: bool) -> Self {
        self.indices_are_sorted = indices_are_sorted;
        self
    }

    /// Returns a copy of this [`GatherOptions`] with its unique-indices promise set to `unique_indices`. When `true`,
    /// the caller promises that gathered windows do not overlap; `false` makes no such promise. Implementations and
    /// transformations may rely on this property. This function does not test the windows for overlap.
    #[inline]
    pub fn with_unique_indices(mut self, unique_indices: bool) -> Self {
        self.unique_indices = unique_indices;
        self
    }

    /// Returns a copy of this [`GatherOptions`] with its requested output [`Sharding`] replaced by `output_sharding`.
    /// Passing `None` restores inferred placement. A request specifies result placement when the window geometry does
    /// not determine one unambiguously: complete window axes inherit input placement and query axes inherit index
    /// placement. Partial windows on explicitly sharded input axes require a request, as do incompatible placements
    /// on paired batching axes.
    ///
    /// A request selects per-axis placement while preserving the common mesh, reduction state, and manual-axis
    /// variation. It must have the output rank and cannot reference automatic mesh axes. Validation takes place when
    /// inferring the result type.
    #[inline]
    pub fn with_output_sharding<S: Into<Option<Sharding>>>(mut self, output_sharding: S) -> Self {
        self.output_sharding = output_sharding.into();
        self
    }

    /// Returns the out-of-bounds index handling [`GatherMode`] of this [`GatherOptions`].
    #[inline]
    pub fn mode(&self) -> &GatherMode<V> {
        &self.mode
    }

    /// Returns whether the caller promises that the index vectors are sorted for this [`GatherOptions`]. This
    /// property is not checked. Implementations and transformations may rely on it; `false` makes no such promise.
    #[inline]
    pub fn indices_are_sorted(&self) -> bool {
        self.indices_are_sorted
    }

    /// Returns whether the caller promises that the gathered windows do not overlap. This property is not checked.
    /// Implementations and transformations may rely on it, including when constructing the adjoint scatter;
    /// `false` makes no such promise.
    #[inline]
    pub fn unique_indices(&self) -> bool {
        self.unique_indices
    }

    /// Returns the requested output [`Sharding`], if any, used when the inferred placement is ambiguous. Refer to the
    /// documentation of [`Self::with_output_sharding`] for more information.
    #[inline]
    pub fn output_sharding(&self) -> Option<&Sharding> {
        self.output_sharding.as_ref()
    }

    /// Validates the stored fill value if [`Self::mode`] is [`GatherMode::Fill`] without materializing it. Type
    /// inference and eager execution share this check, including for empty outputs that would otherwise bypass
    /// reading the fill value.
    fn validate_fill_value(&self, data_type: DataType) -> Result<(), TypeError> {
        if let GatherMode::Fill { value: Some(value) } = &self.mode {
            value.validate_as_constant()?;
            let r#type = value.r#type();
            if r#type.rank() != 0 || !(r#type.data_type().is_numeric() || r#type.data_type().is_boolean()) {
                return Err(TypeError::invalid(format!(
                    "`{GATHER_OPERATION_NAME}` fill value must be a numeric or Boolean scalar",
                )));
            }
            if r#type.data_type() != data_type {
                return Err(TypeError::invalid(format!(
                    "`{}` fill data type `{}` does not match input data type `{}`",
                    GATHER_OPERATION_NAME,
                    r#type.data_type(),
                    data_type,
                )));
            }
        }
        Ok(())
    }
}

impl GatherOptions<Array> {
    /// Returns the constant scalar used to fill out-of-bounds windows for the input `data_type`. An explicit
    /// [`GatherMode::Fill`] value must be a numeric or Boolean scalar of that exact data type; it is cloned without
    /// converting its element encoding, preserving signed zeros and NaN payloads.
    ///
    /// Without an explicit fill, this returns the minimum signed integer, maximum unsigned integer, `true` for
    /// Booleans, or NaN for floating-point and complex values. Floating-point formats without NaN use their normal
    /// NaN conversion result; complex NaN values have a zero imaginary part. Non-fill modes also resolve this default,
    /// but do not use it to replace out-of-bounds windows.
    pub fn resolved_fill_value(&self, data_type: DataType) -> Result<Array, ProgramError> {
        self.validate_fill_value(data_type)?;
        if let GatherMode::Fill { value: Some(value) } = &self.mode {
            return Ok(value.as_ref().clone());
        }
        dispatch_on_array_element_type!(data_type, |Element| {
            // The reduction identities give the extreme values. That is, the identity of a maximum reduction is the
            // smallest signed integer or `false`, and the identity of a minimum reduction is the largest unsigned
            // integer or `true`.
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

impl<V: Value<Type = ArrayType>> Default for GatherOptions<V> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Canonical operation name for [`GatherOperation`].
pub const GATHER_OPERATION_NAME: &str = "gather";

/// [`Operation`] that reads slices (i.e., "windows") out of an input at positions named by an integer index input,
/// assembling them into a new array. Refer to the documentation [`Gather`] for more information on the operation.
///
/// [`GatherDimensionNumbers`] describes the axis mapping independently of window sizes. This operation combines that
/// mapping with [`slice_sizes`](Self::slice_sizes), bounds handling, an optional fill value, index promises, and output
/// placement. Construction stores these settings and type inference validates them against the input and indices.
/// The `V` parameter describes the stored fill constant, not the gathered input. See [`GatherMode`] for the
/// distinction between a stored literal and a flowing value.
#[derive(Clone, Debug, PartialEq)]
pub struct GatherOperation<V: Value<Type = ArrayType> = Array> {
    /// Refer to the documentation of [`dimensions`](Self::dimensions) for more information.
    dimensions: GatherDimensionNumbers,

    /// Refer to the documentation of [`slice_sizes`](Self::slice_sizes) for more information.
    slice_sizes: Vec<usize>,

    /// Refer to the documentation of [`options`](Self::options) for more information.
    options: GatherOptions<V>,
}

impl<V: Value<Type = ArrayType>> GatherOperation<V> {
    /// Creates a new [`GatherOperation`] with the provided dimension numbers and per-input-axis slice sizes. The mode
    /// defaults to [`GatherMode::PromiseInBounds`] and both index promises default to `false`; use the chained `with_*`
    /// builders to override them.
    ///
    /// # Parameters
    ///
    ///   - `dimensions`: Mapping from index components and window axes to input and output axes.
    ///   - `slice_sizes`: Non-negative window size for each input axis. Collapsed axes have size one and batching axes
    ///     have size at most one. Each size must fit its input extent.
    #[inline]
    pub fn new(dimensions: GatherDimensionNumbers, slice_sizes: Vec<usize>) -> Self {
        Self { dimensions, slice_sizes, options: GatherOptions::new() }
    }

    /// Returns a copy of this [`GatherOperation`] with its optional behavior and output placement
    /// replaced by `options`.
    #[inline]
    pub fn with_options(mut self, options: GatherOptions<V>) -> Self {
        self.options = options;
        self
    }

    /// Returns a copy of this [`GatherOperation`] with its out-of-bounds index handling [`GatherMode`]
    /// replaced by `mode`.
    #[inline]
    pub fn with_mode(mut self, mode: GatherMode<V>) -> Self {
        self.options = self.options.with_mode(mode);
        self
    }

    /// Returns a copy of this [`GatherOperation`] with its sorted-indices promise set to `indices_are_sorted`. When
    /// `true`, the caller promises that start-index vectors are sorted; `false` makes no such promise. Implementations
    /// and transformations may rely on this property. This function does not sort or validate the indices.
    #[inline]
    pub fn with_indices_are_sorted(mut self, indices_are_sorted: bool) -> Self {
        self.options = self.options.with_indices_are_sorted(indices_are_sorted);
        self
    }

    /// Returns a copy of this [`GatherOperation`] with its unique-indices promise set to `unique_indices`. When
    /// `true`, the caller promises that gathered windows do not overlap; `false` makes no such promise. Implementations
    /// and transformations may rely on this property. This function does not test the windows for overlap.
    #[inline]
    pub fn with_unique_indices(mut self, unique_indices: bool) -> Self {
        self.options = self.options.with_unique_indices(unique_indices);
        self
    }

    /// Returns a copy of this [`GatherOperation`] with its requested output [`Sharding`] replaced by `output_sharding`.
    /// Passing `None` restores inferred placement. A request specifies result placement when the window geometry does
    /// not determine one unambiguously: complete window axes inherit input placement and query axes inherit index
    /// placement. Partial windows on explicitly sharded input axes require a request, as do incompatible placements
    /// on paired batching axes.
    ///
    /// A request selects per-axis placement while preserving the common mesh, reduction state, and manual-axis
    /// variation. It must have the output rank and cannot reference automatic mesh axes. Validation takes place when
    /// inferring the result type.
    #[inline]
    pub fn with_output_sharding<S: Into<Option<Sharding>>>(mut self, output_sharding: S) -> Self {
        self.options = self.options.with_output_sharding(output_sharding);
        self
    }

    /// Returns the [`GatherDimensionNumbers`] of this [`GatherOperation`] mapping the index input and sliced windows
    /// onto the input and output axes.
    #[inline]
    pub fn dimensions(&self) -> &GatherDimensionNumbers {
        &self.dimensions
    }

    /// Returns the size of the sliced window along each input axis of this [`GatherOperation`]. The number of sizes
    /// equals the input rank.
    #[inline]
    pub fn slice_sizes(&self) -> &[usize] {
        &self.slice_sizes
    }

    /// Returns the optional bounds behavior, index promises, and output placement of this [`GatherOperation`].
    pub fn options(&self) -> &GatherOptions<V> {
        &self.options
    }

    /// Returns the out-of-bounds index handling [`GatherMode`] of this [`GatherOperation`].
    #[inline]
    pub fn mode(&self) -> &GatherMode<V> {
        &self.options.mode
    }

    /// Returns whether the caller promises that the index vectors are sorted for this [`GatherOperation`]. This
    /// property is not checked. Implementations and transformations may rely on it; `false` makes no such promise.
    #[inline]
    pub fn indices_are_sorted(&self) -> bool {
        self.options.indices_are_sorted
    }

    /// Returns whether the caller promises that the gathered windows do not overlap. This property is not checked.
    /// Implementations and transformations may rely on it, including when constructing the adjoint scatter;
    /// `false` makes no such promise.
    #[inline]
    pub fn unique_indices(&self) -> bool {
        self.options.unique_indices
    }

    /// Returns the requested output [`Sharding`], if any, used when the inferred placement is ambiguous. Refer to the
    /// documentation of [`Self::with_output_sharding`] for more information.
    #[inline]
    pub fn output_sharding(&self) -> Option<&Sharding> {
        self.options.output_sharding.as_ref()
    }

    /// Builds the [`ScatterOperation`] used to propagate this [`GatherOperation`]'s output cotangents back to its
    /// input array. The caller applies the scatter to a zero array with the original input shape, using the original
    /// indices and the output cotangents as updates. Each cotangent is added at the position read by the gather, so
    /// repeated reads accumulate rather than overwrite contributions. For example, gathering scalar elements at indices
    /// `[2, 0, 2]` from a length-three input maps output cotangents `[a, b, c]` back to `[b, 0, a + c]`.
    ///
    /// The dimension numbers reverse the gather's window mapping: output window axes become update window axes,
    /// collapsed input axes become inserted window axes, and the index map and paired batching axes are retained.
    /// Sortedness and non-overlap promises are also carried over.
    ///
    /// In-bounds and clipping modes retain their respective policies. Fill mode becomes drop mode because an
    /// out-of-bounds gather window returns a constant fill and contributes no cotangent to the input array. This
    /// function only constructs the operation; it does not allocate the zero array or execute the scatter.
    ///
    /// # Parameters
    ///
    ///   - `output_sharding`: Requested placement of the resulting input cotangent, passed to
    ///     [`ScatterOperation::with_output_sharding`]. `None` leaves placement to [`ScatterOperation`]'s
    ///     type inference.
    fn adjoint_scatter_operation(&self, output_sharding: Option<Sharding>) -> ScatterOperation {
        let dimensions = ScatterDimensionNumbers::new(
            self.dimensions.offset_dimensions().to_vec(),
            self.dimensions.collapsed_slice_dimensions().to_vec(),
            self.dimensions.start_index_map().to_vec(),
        )
        .with_batching_dimensions(
            self.dimensions.batching_dimensions().iter().map(|&(input_axis, _)| input_axis).collect(),
            self.dimensions.batching_dimensions().iter().map(|&(_, indices_axis)| indices_axis).collect(),
        );
        ScatterOperation::new(dimensions, ScatterReductionKind::Add)
            .with_mode(match &self.options.mode {
                GatherMode::PromiseInBounds => ScatterMode::PromiseInBounds,
                GatherMode::Clip => ScatterMode::Clip,
                GatherMode::Fill { .. } => ScatterMode::Drop,
            })
            .with_indices_are_sorted(self.options.indices_are_sorted)
            .with_unique_indices(self.options.unique_indices)
            .with_output_sharding(output_sharding)
    }
}

impl<V: Value<Type = ArrayType>> Display for GatherOperation<V> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<V: Value<Type = ArrayType>> Operation for GatherOperation<V> {
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
        match input_types[0].gather(&input_types[1], self.dimensions(), self.slice_sizes(), self.options()) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError::invalid(error.to_string())),
        }
    }

    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        let mut operation = self.clone();
        if let GatherMode::Fill { value: Some(value) } = &mut operation.options.mode {
            **value = value.as_ref().rename_type_identities(renaming)?;
        }
        Ok(operation)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field(
                "dimensions",
                format_args!(
                    "(offset={:?}, collapsed_slice={:?}, start_index_map={:?}, batching={:?})",
                    self.dimensions.offset_dimensions,
                    self.dimensions.collapsed_slice_dimensions,
                    self.dimensions.start_index_map,
                    self.dimensions.batching_dimensions,
                ),
            )?;
            operation.field("slice_sizes", format_args!("{:?}", self.slice_sizes))?;
            if !matches!(&self.options.mode, GatherMode::PromiseInBounds) {
                operation.field("mode", &self.options.mode)?;
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

impl_reference_dischargeable_operation!(@reference_free <V> GatherOperation<V> where V: Value<Type = ArrayType>);

impl<Stored: Value<Type = ArrayType>, C: Domain<Type = ArrayType, Value: Gather<Stored>>> InterpretableOperation<C>
    for GatherOperation<Stored>
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        // Direct interpretation need not have passed through a builder's type inference. Validate literal storage
        // here too, before handing the value to a custom capability implementation.
        check_count!("input", inputs, 2, ProgramError);
        self.options.validate_fill_value(inputs[0].r#type().data_type())?;
        Ok(vec![inputs[0].gather(&inputs[1], self.dimensions(), self.slice_sizes(), self.options())?])
    }
}

impl<Stored: Value<Type = ArrayType>, C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C>
    for GatherOperation<Stored>
where
    C::Operation: From<GatherOperation<Stored>>,
{
}

impl<Stored: Value<Type = ArrayType>, C: Context<Type = ArrayType>, P: ArrayExtentBatchingPolicy<C>>
    BatchableOperation<C, ArrayBatchingPolicy<P>> for GatherOperation<Stored>
where
    C::Value: Transpose,
    GatherOperation<Stored>: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        // Batching lifts the dimension numbers into one gather. A mapped input alone becomes a full-window offset axis,
        // mapped indices alone add an output batch axis, and jointly mapped inputs gain a paired input/indices batching
        // axis.
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
                        "`{}` mapped input extent {} does not match batching extent {}",
                        GATHER_OPERATION_NAME,
                        input.r#type().dimension(0),
                        axis_dimension,
                    ),
                });
            }
        }

        let dimensions = self.dimensions();
        let mut operation = self.clone();
        if mapped_input && !mapped_indices {
            // The same indices select a complete window along the new input axis, so that axis is an output offset
            // dimension. Its window size must be representable in the operation's static slice sizes. The index
            // promises stay valid: every query gains the same complete window, so disjoint windows stay disjoint.
            let Dimension::Static(axis_size) = axis_dimension else {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "`{GATHER_OPERATION_NAME}` with only its input mapped requires a statically known mapped \
                         extent",
                    ),
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
                dimensions
                    .batching_dimensions()
                    .iter()
                    .map(|&(input_axis, indices_axis)| (input_axis + 1, indices_axis))
                    .collect(),
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
                dimensions
                    .batching_dimensions()
                    .iter()
                    .map(|&(input_axis, indices_axis)| (input_axis, indices_axis + 1))
                    .collect(),
            );
            operation.options.indices_are_sorted = false;
            operation.options.unique_indices = false;
        } else {
            // Pair the new input and indices dimensions (every item reads only its own input, so the index promises
            // stay valid per item). The paired axes take a size-one window, or a zero window when the mapped extent
            // is statically empty or may be empty at runtime. The output batch extent comes from the indices, so a
            // zero batching window does not empty a nonempty batch; it only keeps the window within the input axis's
            // guaranteed minimum extent.
            operation.slice_sizes.insert(0, axis_dimension.bounds().lower().min(1));
            let mut batching = vec![(0, 0)];
            batching.extend(
                dimensions
                    .batching_dimensions()
                    .iter()
                    .map(|&(input_axis, indices_axis)| (input_axis + 1, indices_axis + 1)),
            );
            operation.dimensions = GatherDimensionNumbers::new(
                dimensions.offset_dimensions().iter().map(|axis| axis + 1).collect(),
                dimensions.collapsed_slice_dimensions().iter().map(|axis| axis + 1).collect(),
                dimensions.start_index_map().iter().map(|axis| axis + 1).collect(),
            )
            .with_batching_dimensions(batching);
        }

        if let Some(output_sharding) = self.output_sharding() {
            operation.options.output_sharding = Some(lift_output_sharding_for_leading_batch_axis(
                output_sharding,
                ArrayBatch::sharding_for_inputs(inputs)?,
            )?);
        }

        Ok(operation.interpret_with_batch_axes(context, &aligned, &[BatchAxis::from_position(0)])?.into())
    }
}

// Differentiation must construct a literal zero in the stored family, independently of the input tracer's domain.
// Requiring its eager zero capability avoids embedding a live tangent tracer in the operation's constant payload.
impl_differentiable_operation! {
    <Stored> GatherOperation<Stored>,
    jvp<C>
    where
        Stored: Value<Type = ArrayType>,
        C: Context<Type = ArrayType>,
        C::Value: Gather<Stored>,
        C::Operation: From<GatherOperation<Stored>>,
        EagerContext<Stored>: Zero<Stored>,
    {
        |operation, context, _driver, inputs| {
            // Forward mode differentiation gathers the data tangent at the primal indices. The indices and
            // out-of-bounds fill are constant with respect to the input data, so the tangent uses zero fill.
            // A zero input tangent stays a typed zero.
            check_count!("input", inputs, 2, ProgramError);
            let indices = inputs[1].primal();
            let primal = inputs[0].primal().gather(
                indices,
                operation.dimensions(),
                operation.slice_sizes(),
                operation.options(),
            )?;
            let tangent = match inputs[0].tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
                MaybeZero::Value(tangent) => {
                    // An out-of-bounds fill is constant with respect to the gathered input. Its derivative is zero,
                    // including when the primal uses NaN or a custom nonzero replacement.
                    let tangent_operation = if matches!(operation.mode(), GatherMode::Fill { .. }) {
                        operation.clone().with_mode(GatherMode::Fill {
                            value: Some(Box::new(EagerContext::<Stored>::new().zero(
                                &ArrayType::scalar(tangent.r#type().data_type()),
                            )?)),
                        })
                    } else {
                        operation.clone()
                    };
                    MaybeZero::Value(tangent.gather(
                        &context.primal_to_tangent(indices.clone())?,
                        tangent_operation.dimensions(),
                        tangent_operation.slice_sizes(),
                        tangent_operation.options())?,
                    )
                }
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        Stored: Value<Type = ArrayType>,
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType>
            + From<ZeroOperation<ArrayType>>
            + From<BroadcastOperation>
            + From<ScatterOperation>,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            // The integer index input (i.e., input 1) has no tangent space, so in a valid pushforward it is the known
            // input and the gathered input (i.e., input 0) is the linear one. The forward map `t ↦ gather(t, indices)`
            // has, as its adjoint, the dual scatter-add that writes the output cotangent back into a zero input at the
            // gathered windows: the scatter geometry mirrors  the gather axis-for-axis. The transpose reads the known
            // indices from the pullback boundary and stages an ordinary additive `ScatterOperation`, so linearization
            // retains the indices as regular Single Static Assignment (SSA) residuals. The indices receive a structural
            // zero, and a zero output cotangent stays a structural zero.
            //
            // **Contract:** This homogeneous rule requires a statically shaped input. The scatter target is a zero
            // of the input's cotangent type, and the homogeneous `ArrayType` operation family owns no first-class
            // dimension operations, so it has no constructor that can supply a runtime extent for that zero. A
            // dynamically shaped input is therefore rejected here with an exact diagnostic. Mixed `ArrayIrType`
            // programs are unaffected: the `MemberDifferentiableOperation` rule below routes a dynamically shaped
            // gather into a residual-carrying `LinearCallOperation` whose transpose region rebuilds the same zero
            // from the retained exact extents.
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
                    let indices = inputs[1].as_known().unwrap().clone();

                    // Only the nullary zero is available in the homogeneous family, so enforce this rule's static-shape
                    // contract explicitly instead of letting a dynamic input surface the constructor's own diagnostic.
                    let input_cotangent_type = inputs[0].r#type().cotangent()?;
                    if input_cotangent_type.static_shape().is_none() {
                        return Err(TypeError::invalid(format!(
                            "`{GATHER_OPERATION_NAME}` transpose requires a statically shaped input but got \
                             `{input_cotangent_type}`",
                        ))
                        .into());
                    }

                    let zeros = MaybeZero::Zero(input_cotangent_type.clone()).materialize(&**context)?;
                    let scatter_operation =
                        operation.adjoint_scatter_operation(input_cotangent_type.sharding().cloned());
                    let outputs =
                        context.stage_operation(scatter_operation, Vec::new(), &[zeros, indices, cotangent.clone()])?;
                    check_count!("output", outputs, 1, ProgramError);

                    let mut contribution = outputs.into_iter().next().unwrap();
                    if contribution.r#type().as_ref() != &input_cotangent_type {
                        let mut outputs = context.stage_operation(
                            BroadcastOperation::new(
                                input_cotangent_type.clone(),
                                (0..input_cotangent_type.rank()).collect(),
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
    },
}

impl<Stored: Value<Type = ArrayType>, C: Context<Type = ArrayIrType>> MemberDifferentiableOperation<C>
    for GatherOperation<Stored>
where
    C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Operation: From<DimensionSizeOperation>
        + From<LinearCallOperation<ArrayIrType>>
        + OperationProjection<
            ArrayType,
            Projected: DifferentiableOperation<ProjectedContext<C, ArrayType>>
                           + From<GatherOperation<Stored>>
                           + From<ScatterOperation>
                           + From<BroadcastOperation>
                           + From<ZeroOperation<ArrayType>>,
        >,
    EagerContext<Stored>: Zero<Stored>,
{
    fn jvp_in_parent<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // A dynamically shaped input retains its exact extents and indices as ordinary residual values,
        // and a static input delegates to the homogeneous projected rule.
        let destinations = context;
        let [input, indices] = inputs else {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
        };

        let input_type = <&ArrayType>::try_from(input.primal().r#type().as_ref())?.clone();
        if input_type.shape().dimensions().iter().all(|dimension| matches!(dimension, Dimension::Static(_))) {
            let operation = <C::Operation as OperationProjection<ArrayType>>::Projected::from(self.clone());
            return jvp_projected_operation(destinations, &operation, inputs);
        }

        let operation = <C::Operation as OperationProjection<ArrayType>>::Projected::from(self.clone());
        let mut primal_outputs =
            destinations
                .primal()
                .bind(operation, Vec::new(), &[input.primal().clone(), indices.primal().clone()])?;
        check_count!("output", primal_outputs, 1, ProgramError);

        let output_primal = primal_outputs.remove(0);
        let tangent_primal = destinations.primal_to_tangent(output_primal.clone())?;
        let tangent_inputs = destinations.dual_primal_to_tangent(inputs)?;
        let input = &tangent_inputs[0];
        let indices = &tangent_inputs[1];
        let tangent_context = destinations.tangent();
        let tangent = match input.tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(tangent_primal.r#type().tangent()?),
            MaybeZero::Value(input_tangent) => {
                let mut residuals = LinearResiduals::new();
                let indices_index = residuals.retain(indices.primal().clone());
                let input_shape = residuals.retain_shape(tangent_context, input.primal())?;

                // The linear region differentiates input data, not the primal's constant replacement value.
                let forward_operation = if matches!(self.mode(), GatherMode::Fill { .. }) {
                    self.clone().with_mode(GatherMode::Fill {
                        value: Some(Box::new(EagerContext::<Stored>::new().zero(&ArrayType::scalar(
                            <&ArrayType>::try_from(input_tangent.r#type().as_ref())?.data_type(),
                        ))?)),
                    })
                } else {
                    self.clone()
                };

                let transpose_operand_type = input_type.cotangent()?;
                let transpose_operation = self.adjoint_scatter_operation(transpose_operand_type.sharding().cloned());
                let mut tangent_outputs = LinearCallOperation::stage(
                    tangent_context,
                    residuals.into_values(),
                    vec![input_tangent.clone()],
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
                            input_shape.dynamic_dimensions(residuals).as_slice(),
                        )?;
                        check_count!("output", zero_outputs, 1, ProgramError);
                        let zeros = zero_outputs.remove(0);
                        let mut contributions = transpose_context.bind(
                            <C::Operation as OperationProjection<ArrayType>>::Projected::from(transpose_operation),
                            Vec::new(),
                            &[zeros, residuals[indices_index].clone(), output_cotangents[0].clone()],
                        )?;
                        check_count!("output", contributions, 1, ProgramError);

                        // Residual extents may refine singleton dynamic dimensions to static dimensions. Restore
                        // the original cotangent signature, including its dimension identities and storage metadata.
                        let contribution = contributions.remove(0);
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

/// Reads "windows" from an array at positions supplied by an integer index array. The receiver is the data source.
/// The `indices` input describes where each window starts, [`GatherDimensionNumbers`] maps the axes, `slice_sizes`
/// gives the window sizes, and [`GatherOptions`] selects out-of-bounds behavior. Gathering can select individual
/// elements, entire rows, or multi-dimensional blocks; the output contains one window per query. Use
/// [`Self::gather_axis`] for complete slices along one axis without constructing dimension numbers. Use
/// [`DynamicGather::dynamic_gather_axis`] when query or untouched input extents must remain dynamic.
/// The general [`Self::gather`] function exposes the mapping below.
///
/// # From Indices to Output Axes
///
/// Suppose `indices` has shape `[Q0, Q1, ..., K]`. Every position in `[Q0, Q1, ...]` is a _query_, and its last-axis
/// vector contains `K` start coordinates. The last axis is always the index-vector axis and it does not appear in the
/// output. For scalar indices, include a trailing size-one axis as in `[number_of_queries, 1]`, and not
/// `[number_of_queries]`. A shape `[K]` represents one query with no query axes.
///
/// The following settings describe three distinct coordinate systems (all axis numbers are zero-based):
///
/// | Setting                      | Axes Refer To | Meaning                                          |
/// | ---------------------------- | ------------- | ------------------------------------------------ |
/// | `start_index_map`            | Input         | Axis addressed by each index-vector component.   |
/// | `slice_sizes`                | Input         | Window extent on each input axis.                |
/// | `collapsed_slice_dimensions` | Input         | Size-one window axes omitted from the output.    |
/// | `offset_dimensions`          | Output        | Positions of retained window axes.               |
/// | `batching_dimensions`        | Input/Indices | Pairs linking input axes to matching query axes. |
///
/// [`GatherDimensionNumbers`] holds the mappings and the `slice_sizes` argument holds the window sizes. For each query,
/// `start_index_map[j]` says which input axis receives index-vector component `j`. For example, `[1, 0]` interprets a
/// vector `[column, row]` as a start in a matrix. Input axes absent from this map start at zero, except paired batching
/// axes, whose coordinates come from the query itself.
///
/// After extracting a window, remove its collapsed and paired batching axes. Place the retained window axes, still
/// in input-axis order, at `offset_dimensions`. Fill all remaining output positions with the query axes, still in
/// indices-axis order. Thus, `offset_dimensions` interleaves window and query axes; it does not arbitrarily permute
/// window axes. Its entries must be sorted and distinct, as must the collapsed-axis list.
///
/// The output rank is `offset_dimensions.len() + indices.rank() - 1`. Its window-axis extents come from `slice_sizes`;
/// its query-axis extents come from `indices`. Collapsed axes must have window size one. Window sizes are static and
/// must fit the input: for a dynamic input axis, its guaranteed minimum extent must be at least the window size.
///
/// # Bounds Handling and Optional Settings
///
/// [`GatherMode`] determines what happens when a start would put any part of a window outside the input. The default
/// is [`GatherMode::PromiseInBounds`]. Negative starts are out of bounds; they do not count backward from the end.
/// For input `[0, 1, 2, 3, 4]`, window size `[2]`, and start `[4]`:
///
/// | Mode              | Result                                                                     |
/// | ----------------- | -------------------------------------------------------------------------- |
/// | `PromiseInBounds` | Violates the caller's promise; no result or gradient behavior is promised. |
/// | `Clip`            | Moves the start to `3`, producing `[3, 4]`.                                |
/// | `Fill { value }`  | Fills the whole window, for example `[-1, -1]` with an explicit `-1` fill. |
///
/// [`GatherMode::Fill`] owns an optional boxed constant scalar of the input element data type. Set the mode with
/// [`GatherOptions::with_mode`]. Without a value, the fill mode uses NaN for floating-point and complex values, the
/// minimum signed integer, the maximum unsigned integer, or `true` for Booleans. Other modes carry no fill value. The
/// mode never changes the output shape. Clipping shifts a whole window, and filling replaces a whole window, rather
/// than preserving its in-bounds portion.
///
/// ```rust
/// # use ryft_core::{Array, Gather, GatherDimensionNumbers, GatherMode, GatherOptions};
/// // Shapes: input [5], indices [1, 1], fill [] (scalar) -> output [1, 2].
/// let input = Array::vector(vec![0_i32, 1, 2, 3, 4]).unwrap();
/// let indices = Array::matrix(1, 1, vec![4_i32]).unwrap();
/// let dimensions = GatherDimensionNumbers::new(vec![1], vec![], vec![0]);
/// let options = GatherOptions::new().with_mode(GatherMode::Fill {
///     value: Some(Box::new(Array::scalar(-1_i32).unwrap())),
/// });
/// assert_eq!(
///     input.gather(&indices, &dimensions, &[2], &options),
///     Ok(Array::matrix(1, 2, vec![-1_i32, -1]).unwrap()),
/// );
/// ```
///
/// [`GatherOptions::with_indices_are_sorted`] and [`GatherOptions::with_unique_indices`] declare unchecked caller
/// promises that implementations and transformations may rely on. They do not sort or deduplicate queries. Leave them
/// false unless the index vectors are sorted or the gathered windows do not overlap, respectively. Neither setting
/// changes the intended result for inputs satisfying the promises.
///
/// [`GatherOptions::with_output_sharding`] requests output placement, which can resolve otherwise ambiguous placement
/// when gathering partial windows on explicitly sharded axes. It does not change the axis mapping or numerical result.
/// Input and indices must use compatible meshes; requested placement must preserve reduction and manual-axis state.
/// Without a request, window axes and query axes infer placement from the input and indices. Indices cannot carry
/// reduction state. Inputs with reduction state require replicated, invariant indices, and fill mode is unsupported
/// for unreduced inputs.
///
/// The input and indices must reside in the same memory space. The result keeps the input element data type and memory
/// placement, and clears explicit physical layout metadata because gathering changes the relationship between logical
/// axes and storage. Shape and mapping validation occurs when inferring or executing the operation, not merely when
/// constructing its dimension numbers.
///
/// The `Stored` parameter selects the fill's constant representation independently of `Self`. For example, gathering
/// a staged tracer still uses an [`Array`] literal for its fill in the built-in operation families; it does not embed
/// a tracer in the operation payload. Other operation families can implement this capability for their own stored
/// values. Fills are validated as scalar constants of the input element data type before execution.
///
/// # Examples
///
/// ## Selecting Rows and Choosing Output Order
///
/// For the matrix below, the query vectors `[0]` and `[2]` select its first and last rows. The window size `[1, 2]`
/// selects one row and both columns. Collapsing input axis `0` removes the singleton row axis from each window.
/// `offset_dimensions = [1]` places the retained column axis at output position `1`, leaving position `0` for queries.
///
/// ```rust
/// # use ryft_core::{Array, Gather, GatherDimensionNumbers, GatherOptions};
///
/// // Shapes: input [3, 2], indices [2, 1] -> output [2, 2] for either axis order below.
/// let input = Array::matrix(3, 2, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
/// let indices = Array::matrix(2, 1, vec![0_i32, 2]).unwrap();
/// let dimensions = GatherDimensionNumbers::new(vec![1], vec![0], vec![0]);
/// assert_eq!(
///     input.gather(&indices, &dimensions, &[1, 2], &GatherOptions::new()),
///     Ok(Array::matrix(2, 2, vec![0.0, 1.0, 4.0, 5.0]).unwrap()),
/// );
///
/// // Moving the column axis to output position 0 makes position 1 the query axis.
/// let dimensions = GatherDimensionNumbers::new(vec![0], vec![0], vec![0]);
/// assert_eq!(
///     input.gather(&indices, &dimensions, &[1, 2], &GatherOptions::new()),
///     Ok(Array::matrix(2, 2, vec![0.0, 4.0, 1.0, 5.0]).unwrap()),
/// );
/// ```
///
/// ## Rectangular Windows
///
/// Collapsing is optional. With two-component indices, `start_index_map = [0, 1]`, and `slice_sizes = [2, 2]`,
/// each query selects a two-row, two-column block. Keeping both window axes at output positions `[1, 2]` gives
/// shape `[queries, 2, 2]`. The starts `[0, 1]` and `[1, 2]` below produce blocks `[[1, 2], [5, 6]]` and
/// `[[6, 7], [10, 11]]`, respectively. Overlapping windows are allowed.
///
/// ```rust
/// # use ryft_core::{Array, ArrayType, DataType, Gather, GatherDimensionNumbers, GatherOptions};
///
/// // Shapes: input [3, 4], indices [2, 2] -> output [2, 2, 2] (i.e., two [2, 2] windows).
/// let input = Array::matrix(3, 4, vec![0_i32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).unwrap();
/// let indices = Array::matrix(2, 2, vec![0_i32, 1, 1, 2]).unwrap();
/// let dimensions = GatherDimensionNumbers::new(vec![1, 2], vec![], vec![0, 1]);
/// let expected = Array::from_elements(
///     ArrayType::new_static(DataType::I32, [2, 2, 2]),
///     &[1_i32, 2, 5, 6, 6, 7, 10, 11],
/// ).unwrap();
/// assert_eq!(input.gather(&indices, &dimensions, &[2, 2], &GatherOptions::new()), Ok(expected));
/// ```
///
/// ## Pairing Queries with Input Batch Items
///
/// Ordinary query axes all read from the same input. Paired batching instead ties a query coordinate to an input
/// coordinate: each `(input_axis, indices_axis)` entry of `batching_dimensions` takes the input coordinate from
/// that indices axis. These paired axes must have equal extents. Input batching axes cannot also be indexed by
/// `start_index_map` or collapsed. Their window sizes are at most one; they do not contribute window axes to the
/// output. Their matching query axes still appear once in the output.
///
/// For an input of shape `[batch, columns]` and indices of shape `[batch, 1]`, pair input axis `0` with indices axis
/// `0`. Each vector then supplies only a column index (`start_index_map = [1]`). This selects one column per row,
/// rather than applying every query to every row:
///
/// ```rust
/// # use ryft_core::{Array, Gather, GatherDimensionNumbers, GatherOptions};
///
/// // Shapes: input [2, 3], indices [2, 1] -> output [2] (i.e., one element per batch item).
/// let input = Array::matrix(2, 3, vec![10_i32, 20, 30, 40, 50, 60]).unwrap();
/// let indices = Array::matrix(2, 1, vec![2_i32, 0]).unwrap();
/// let dimensions = GatherDimensionNumbers::new(vec![], vec![1], vec![1])
///     .with_batching_dimensions(vec![(0, 0)]);
/// assert_eq!(
///     input.gather(&indices, &dimensions, &[1, 1], &GatherOptions::new()),
///     Ok(Array::vector(vec![30_i32, 40]).unwrap()),
/// );
/// ```
pub trait Gather<Stored: Value<Type = ArrayType> = Array>: Sized {
    /// Reads windows of `slice_sizes` from the input at the starts given by `indices`, using `dimensions` to arrange
    /// their axes and `options` to select bounds behavior, index promises, and output placement. Refer to the
    /// documentation of [`Gather`] for how index vectors, window sizes, and output-axis positions interact. Negative
    /// starts are out of bounds; they do not count backward from an axis end. Bounds handling applies to whole windows,
    /// so one invalid start fills the entire window in fill mode.
    ///
    /// # Parameters
    ///
    ///   - `indices`: Integer array with one trailing index-vector axis. Its remaining axes enumerate queries.
    ///   - `dimensions`: Mapping from index components and window axes to input and output axes.
    ///   - `slice_sizes`: Window size along each input axis.
    ///   - `options`: Bounds mode, optional constant fill, unchecked index promises, and requested output placement.
    fn gather(
        &self,
        indices: &Self,
        dimensions: &GatherDimensionNumbers,
        slice_sizes: &[usize],
        options: &GatherOptions<Stored>,
    ) -> Result<Self, ProgramError>;

    /// Gathers complete slices along one axis using raw integer indices. The index array's shape replaces that input
    /// axis in the result, and all other input axes retain their order and full size. Unlike indexing APIs that count
    /// negative indices backward from the end, this function treats every negative index as out of bounds and applies
    /// `mode` directly. It does not change or wrap index values.
    ///
    /// The selected axis may have a dynamic extent. All other input extents must be statically known because the
    /// underlying [`GatherOperation`] stores their complete window sizes as host integers. The index shape must also
    /// support the homogeneous [`Reshape`] used to append its index-vector axis. Use [`DynamicGather`] for dynamic
    /// query shapes or an explicit operation for partial windows and custom fill values.
    ///
    /// # Parameters
    ///
    ///   - `indices`: Integer indices of any rank. A scalar selects one slice and removes the selected axis.
    ///   - `axis`: Input axis to select. Negative axes count backward from the input rank.
    ///   - `mode`: Out-of-bounds policy. [`GatherMode::Clip`] clamps indices; [`GatherMode::Fill`] fills invalid slices
    ///     using the input data type's default fill; the promise mode requires valid indices.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::{Array, Gather, GatherMode};
    ///
    /// // Shapes: input [2, 3], indices [2] -> output [2, 2].
    /// let input = Array::matrix(2, 3, vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
    /// let indices = Array::vector(vec![2_i32, 0]).unwrap();
    /// let output = input.gather_axis(&indices, 1, GatherMode::Clip).unwrap();
    /// assert_eq!(output, Array::matrix(2, 2, vec![3_i32, 1, 6, 4]).unwrap());
    /// ```
    fn gather_axis<A: Into<Axis>>(
        &self,
        indices: &Self,
        axis: A,
        mode: GatherMode<Stored>,
    ) -> Result<Self, ProgramError>
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
        let dimensions = GatherDimensionNumbers::new(offset_dimensions, vec![axis], vec![axis]);
        let options = GatherOptions::new().with_mode(mode);
        self.gather(&expanded_indices, &dimensions, &slice_sizes, &options)
    }
}

impl<Stored: Value<Type = ArrayType>> Gather<Stored> for ArrayType {
    fn gather(
        &self,
        indices: &Self,
        dimensions: &GatherDimensionNumbers,
        slice_sizes: &[usize],
        options: &GatherOptions<Stored>,
    ) -> Result<Self, ProgramError> {
        let input = self;
        let (input_batching_dimensions, indices_batching_dimensions): (Vec<_>, Vec<_>) =
            dimensions.batching_dimensions().iter().copied().unzip();
        let input_rank = input.rank();
        let indices_rank = indices.rank();

        if indices_rank == 0 {
            return Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` indices must have rank at least 1 (the trailing index vector)",
            ))
            .into());
        }

        if !indices.data_type().is_integer() {
            return Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` indices must be integer-typed but have type `{indices}`",
            ))
            .into());
        }

        options.validate_fill_value(input.data_type())?;

        if input.memory() != indices.memory() {
            return Err(TypeError::invalid(format!(
                "`{}` input and indices must share one memory space but reside in `{}` and `{}`",
                GATHER_OPERATION_NAME,
                input.memory(),
                indices.memory(),
            ))
            .into());
        }

        let index_vector_dimension = indices_rank - 1;
        let Dimension::Static(index_vector_extent) = indices.dimension(index_vector_dimension) else {
            return Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` indices index vector dimension must have a static extent",
            ))
            .into());
        };

        // Output rank, then each dimension-number list against its rank bound.
        let output_rank = dimensions.offset_dimensions().len() + indices_rank - 1;
        validate_unique_in_range(
            GATHER_OPERATION_NAME,
            "offset_dimensions",
            dimensions.offset_dimensions(),
            output_rank,
            true,
        )?;

        validate_unique_in_range(
            GATHER_OPERATION_NAME,
            "collapsed_slice_dimensions",
            dimensions.collapsed_slice_dimensions(),
            input_rank,
            true,
        )?;

        validate_unique_in_range(
            GATHER_OPERATION_NAME,
            "batching_dimensions input axes",
            &input_batching_dimensions,
            input_rank,
            true,
        )?;

        if dimensions.start_index_map().len() != index_vector_extent {
            return Err(TypeError::invalid(format!(
                "`{}` `start_index_map` has length {} but the index vector extent is {}",
                GATHER_OPERATION_NAME,
                dimensions.start_index_map().len(),
                index_vector_extent,
            ))
            .into());
        }

        validate_unique_in_range(
            GATHER_OPERATION_NAME,
            "start_index_map",
            dimensions.start_index_map(),
            input_rank,
            false,
        )?;

        validate_unique_in_range(
            GATHER_OPERATION_NAME,
            "batching_dimensions indices axes",
            &indices_batching_dimensions,
            indices_rank,
            false,
        )?;

        if dimensions.start_index_map().iter().any(|axis| input_batching_dimensions.contains(axis)) {
            return Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` `start_index_map` and `batching_dimensions input axes` must be disjoint"
            ))
            .into());
        }

        if indices_batching_dimensions.contains(&index_vector_dimension) {
            return Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` `batching_dimensions indices axes` cannot name the index vector dimension \
                 {index_vector_dimension}"
            ))
            .into());
        }

        // The collapsed, batching, and start-index-map axis sets must be mutually disjoint where required.
        let collapsed: BTreeSet<usize> = dimensions.collapsed_slice_dimensions().iter().copied().collect();
        let operand_batching: BTreeSet<usize> = input_batching_dimensions.iter().copied().collect();
        if collapsed.intersection(&operand_batching).next().is_some() {
            return Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` `collapsed_slice_dimensions` and `batching_dimensions input axes` must be \
                 disjoint"
            ))
            .into());
        }

        // Slice sizes must have one per input axis, size 1 on collapsed axes, size at most 1 on batching axes,
        // and within the input extent when that extent is static.
        if slice_sizes.len() != input_rank {
            return Err(TypeError::invalid(format!(
                "`{}` `slice_sizes` has length {} but the input has rank {}",
                GATHER_OPERATION_NAME,
                slice_sizes.len(),
                input_rank,
            ))
            .into());
        }

        for (axis, &size) in slice_sizes.iter().enumerate() {
            match input.dimension(axis) {
                Dimension::Static(extent) if size > extent => {
                    return Err(TypeError::invalid(format!(
                        "`{GATHER_OPERATION_NAME}` slice size {size} at axis {axis} exceeds the input extent {extent}",
                    ))
                    .into());
                }
                Dimension::Dynamic(variable) if size > variable.bounds().lower() => {
                    return Err(TypeError::invalid(format!(
                        "`{}` slice size {} exceeds the guaranteed minimum extent {} of dynamic input axis {}",
                        GATHER_OPERATION_NAME,
                        size,
                        variable.bounds().lower(),
                        axis,
                    ))
                    .into());
                }
                _ => {}
            }

            if collapsed.contains(&axis) && size != 1 {
                return Err(TypeError::invalid(format!(
                    "`{GATHER_OPERATION_NAME}` collapsed slice dimension {axis} must have slice size 1 but has {size}",
                ))
                .into());
            }

            if operand_batching.contains(&axis) && size > 1 {
                return Err(TypeError::invalid(format!(
                    "`{GATHER_OPERATION_NAME}` input batching dimension {axis} must have slice size at most 1 but \
                     has {size}",
                ))
                .into());
            }
        }

        let offset_count = input_rank - collapsed.len() - operand_batching.len();
        if dimensions.offset_dimensions().len() != offset_count {
            return Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` `offset_dimensions` has length {} but the number of non-collapsed, \
                 non-batching input axes is {offset_count}",
                dimensions.offset_dimensions().len(),
            ))
            .into());
        }

        // Batch-dimension extents must match between input and indices.
        for &(input_axis, indices_axis) in dimensions.batching_dimensions() {
            if !input.dimension(input_axis).has_equal_extents(&indices.dimension(indices_axis)) {
                return Err(TypeError::invalid(format!(
                    "`{GATHER_OPERATION_NAME}` batching dimensions must have equal extents, but input axis \
                     {input_axis} and indices axis {indices_axis} differ"
                ))
                .into());
            }
        }

        // In the output shape, offset positions take the (non-collapsed, non-batching) input window sizes in input-axis
        // order and the remaining positions take the indices' batch axes (i.e., every axis but the index vector)
        // in order.
        let input_offset_axes: Vec<usize> = (0..input_rank)
            .filter(|axis| !collapsed.contains(axis) && !operand_batching.contains(axis))
            .collect();
        let indices_batch_axes: Vec<usize> = (0..indices_rank).filter(|axis| *axis != index_vector_dimension).collect();
        let batch_query_sizes: Vec<Dimension> =
            indices_batch_axes.iter().map(|&axis| indices.dimension(axis)).collect();
        let offset_position: BTreeSet<usize> = dimensions.offset_dimensions().iter().copied().collect();
        let mut offset_iterator = input_offset_axes.iter();
        let mut batch_iterator = batch_query_sizes.iter();
        let output_dimensions: Vec<Dimension> = (0..output_rank)
            .map(|position| {
                if offset_position.contains(&position) {
                    let &input_axis = offset_iterator.next().unwrap();
                    Dimension::Static(slice_sizes[input_axis])
                } else {
                    batch_iterator.next().unwrap().clone()
                }
            })
            .collect();

        // Retained full-window axes preserve input placement, and query axes inherit index placement. Partial windows
        // on explicitly sharded axes need an explicit output placement. Reduction and manual-axis state remain part of
        // the contract even when a placement is supplied explicitly.
        let input_sharding = input.sharding();
        let indices_sharding = indices.sharding();
        let mesh = match (input_sharding, indices_sharding) {
            (Some(input), Some(indices)) if input.mesh() != indices.mesh() => {
                return Err(TypeError::invalid(format!(
                    "`{GATHER_OPERATION_NAME}` input and indices shardings must use the same mesh",
                ))
                .into());
            }
            (Some(sharding), _) | (_, Some(sharding)) => Some(sharding.mesh().clone()),
            (None, None) => None,
        };

        if indices_sharding
            .is_some_and(|sharding| !sharding.unreduced_axes().is_empty() || !sharding.reduced_axes().is_empty())
        {
            return Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` indices cannot carry reduced or unreduced mesh axes",
            ))
            .into());
        }

        let unreduced_axes = input_sharding.map(Sharding::unreduced_axes).cloned().unwrap_or_default();
        let reduced_axes = input_sharding.map(Sharding::reduced_axes).cloned().unwrap_or_default();
        let mut varying_manual_axes = input_sharding.map(Sharding::varying_manual_axes).cloned().unwrap_or_default();
        if let Some(sharding) = indices_sharding {
            varying_manual_axes.extend(sharding.varying_manual_axes().iter().cloned());
            if (!unreduced_axes.is_empty() || !reduced_axes.is_empty())
                && (sharding.dimensions().iter().any(|dimension| *dimension != ShardingDimension::Replicated)
                    || !sharding.varying_manual_axes().is_empty())
            {
                return Err(TypeError::invalid(format!(
                    "`{GATHER_OPERATION_NAME}` reduction-state inputs require replicated, invariant indices",
                ))
                .into());
            }
        }

        // An unreduced input holds per-device partial sums, so a fill constant written on every device would be counted
        // once per device by the pending reduction. Reduced inputs are complete on each device and may be filled.
        if !unreduced_axes.is_empty() && matches!(options.mode(), GatherMode::Fill { .. }) {
            return Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` fill mode does not support unreduced inputs",
            ))
            .into());
        }

        let sharding = if let Some(requested) = options.output_sharding() {
            if mesh.as_ref().is_some_and(|mesh| mesh != requested.mesh()) {
                return Err(TypeError::invalid(format!(
                    "`{GATHER_OPERATION_NAME}` requested output sharding uses a different mesh",
                ))
                .into());
            }

            if requested.unreduced_axes() != &unreduced_axes
                || requested.reduced_axes() != &reduced_axes
                || requested.varying_manual_axes() != &varying_manual_axes
            {
                return Err(TypeError::invalid(format!(
                    "`{GATHER_OPERATION_NAME}` requested output sharding changes reduction or manual-axis state",
                ))
                .into());
            }

            if requested.rank() != output_rank {
                return Err(TypeError::invalid(format!(
                    "`{}` output sharding rank ({}) does not match the output rank ({})",
                    GATHER_OPERATION_NAME,
                    requested.rank(),
                    output_rank,
                ))
                .into());
            }

            if requested.references_auto_axis() {
                return Err(TypeError::invalid(format!(
                    "`{GATHER_OPERATION_NAME}` output sharding cannot reference auto mesh axes",
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
            if let Some(sharding) = input_sharding {
                for &axis in &replicated_operand_axes {
                    if input.dimension(axis) != Dimension::Static(slice_sizes[axis])
                        && sharding.dimensions()[axis].has_explicit_axis(&mesh)
                    {
                        return Err(TypeError::invalid(format!(
                            "`{GATHER_OPERATION_NAME}` input axis {axis} is indexed by the start indices and must \
                             be replicated over explicit mesh axes; request an explicit output sharding to resolve \
                             placement"
                        ))
                        .into());
                    }
                }
            }

            if let Some(sharding) = indices_sharding
                && sharding.dimensions()[index_vector_dimension].has_explicit_axis(&mesh)
            {
                return Err(TypeError::invalid(format!(
                    "`{GATHER_OPERATION_NAME}` indices index vector dimension must be replicated over explicit \
                     mesh axes"
                ))
                .into());
            }

            // A partial window does not preserve the placement of the complete input axis, even when its
            // start index is implicit zero rather than supplied in the index vector.
            for &axis in &input_offset_axes {
                if input.dimension(axis) != Dimension::Static(slice_sizes[axis])
                    && input_sharding.is_some_and(|sharding| sharding.dimensions()[axis].has_explicit_axis(&mesh))
                {
                    return Err(TypeError::invalid(format!(
                        "`{GATHER_OPERATION_NAME}` partial sharded windows require explicit output sharding",
                    ))
                    .into());
                }
            }

            let mut indices_placement = indices_sharding
                .map(|sharding| sharding.dimensions().to_vec())
                .unwrap_or_else(|| vec![ShardingDimension::Replicated; indices_rank]);
            for &(input_axis, indices_axis) in dimensions.batching_dimensions() {
                let input_placement = input_sharding
                    .map(|sharding| sharding.dimensions()[input_axis].clone())
                    .unwrap_or(ShardingDimension::Replicated);
                let indices_dimension = &mut indices_placement[indices_axis];
                if *indices_dimension == ShardingDimension::Replicated {
                    *indices_dimension = input_placement;
                } else if input_placement != ShardingDimension::Replicated && input_placement != *indices_dimension {
                    return Err(TypeError::invalid(format!(
                        "`{GATHER_OPERATION_NAME}` conflicting batching-axis shardings require explicit \
                         output sharding",
                    ))
                    .into());
                }
            }

            // On sharding/placement, offset positions inherit the input window axes and the remaining positions inherit
            // the indices' batch axes (i.e., every axis but the index vector), in order.
            let mut offset_iterator = input_offset_axes.iter();
            let mut batch_iterator = indices_batch_axes.iter();
            let placement: Vec<ShardingDimension> = (0..output_rank)
                .map(|position| {
                    if offset_position.contains(&position) {
                        let &input_axis = offset_iterator.next().unwrap();
                        input_sharding
                            .map(|sharding| sharding.dimensions()[input_axis].clone())
                            .unwrap_or(ShardingDimension::Replicated)
                    } else {
                        let &indices_axis = batch_iterator.next().unwrap();
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

        ArrayType::new(input.data_type(), Shape::new(output_dimensions))
            .with_memory(input.memory())
            .with_sharding(sharding)
            .map_err(|error| {
                TypeError::invalid(format!("`{GATHER_OPERATION_NAME}` output type is invalid: {error}")).into()
            })
    }
}

impl Gather for Array {
    fn gather(
        &self,
        indices: &Self,
        dimensions: &GatherDimensionNumbers,
        slice_sizes: &[usize],
        options: &GatherOptions,
    ) -> Result<Self, ProgramError> {
        let output_type = self.r#type().gather(indices.r#type().as_ref(), dimensions, slice_sizes, options)?;
        let input_shape = self.r#type().static_shape().unwrap();
        let indices_shape = indices.r#type().static_shape().unwrap();
        let input_rank = input_shape.rank();
        let indices_rank = indices_shape.rank();
        let output_rank = output_type.rank();
        let index_vector_dimension = indices_rank - 1;
        let indices_data_type = indices.r#type().data_type();

        // Classify input axes (window axes carry the slice while collapsed/batching do not) and output axes (offset
        // positions carry the window and the rest carry the indices' batch coordinates).
        let collapsed: BTreeSet<usize> = dimensions.collapsed_slice_dimensions().iter().copied().collect();
        let batching: BTreeSet<usize> =
            dimensions.batching_dimensions().iter().map(|&(input_axis, _)| input_axis).collect();
        let input_window_axes: Vec<usize> =
            (0..input_rank).filter(|axis| !collapsed.contains(axis) && !batching.contains(axis)).collect();
        let offset_positions: BTreeSet<usize> = dimensions.offset_dimensions().iter().copied().collect();
        let batch_output_positions: Vec<usize> =
            (0..output_rank).filter(|position| !offset_positions.contains(position)).collect();
        let indices_batch_axes: Vec<usize> = (0..indices_rank).filter(|axis| *axis != index_vector_dimension).collect();

        // Resolve a fill only when the mode can use it, preserving the exact element encoding of explicit fills.
        let dropped_fill = if matches!(options.mode(), GatherMode::Fill { .. }) {
            let value = options.resolved_fill_value(output_type.data_type())?;
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
        let mut input_origin = vec![0usize; input_rank];
        let mut input_index = vec![0usize; input_rank];
        let mut dropped = false;
        let drop_out_of_bounds = matches!(options.mode(), GatherMode::Fill { .. });
        for output_element in 0..output_addressing.element_count() {
            // Consecutive window elements often use the same query. Cache only that query's origin and bounds
            // decision, retaining the original element traversal order even when query/window axes interleave.
            // The first iteration also initializes queries with an empty index vector or no query axes.
            let mut query_changed = output_element == 0;
            for (position, &axis) in batch_output_positions.iter().enumerate() {
                let indices_axis = indices_batch_axes[position];
                let coordinate = output_index[axis];
                query_changed |= indices_index[indices_axis] != coordinate;
                indices_index[indices_axis] = coordinate;
            }

            if query_changed {
                input_origin.fill(0);
                dropped = false;

                for &(input_axis, indices_axis) in dimensions.batching_dimensions() {
                    input_origin[input_axis] = indices_index[indices_axis];
                }

                for (component, &input_axis) in dimensions.start_index_map().iter().enumerate() {
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
                    let maximum = (input_shape[input_axis] - slice_sizes[input_axis]) as i128;
                    dropped |= drop_out_of_bounds && (raw < 0 || raw > maximum);

                    // Invalid fill/drop origins are never accessed. Promise mode uses defensive clipping without
                    // guaranteeing any particular out-of-bounds result to callers.
                    input_origin[input_axis] = raw.clamp(0, maximum) as usize;
                }
            }

            let source = if dropped {
                let (value, addressing) = dropped_fill.as_ref().unwrap();
                &value.storage_bytes()[addressing.byte_range_for_flat_index(0)]
            } else {
                input_index.copy_from_slice(&input_origin);
                for (window, &input_axis) in input_window_axes.iter().enumerate() {
                    input_index[input_axis] += output_index[dimensions.offset_dimensions()[window]];
                }
                &self.storage_bytes()[input_addressing.byte_range_unchecked(&input_index)]
            };

            bytes[output_addressing.byte_range_for_flat_index(output_element)].copy_from_slice(source);
            output_addressing.advance_index(&mut output_index);
        }

        Ok(Self::new_unchecked(output_type, Arc::new(bytes)))
    }
}

impl<Stored: Value<Type = ArrayType>, A: Gather<Stored> + Value<Type = ArrayType>> Gather<Stored> for ArrayIrValue<A> {
    #[inline]
    fn gather(
        &self,
        indices: &Self,
        dimensions: &GatherDimensionNumbers,
        slice_sizes: &[usize],
        options: &GatherOptions<Stored>,
    ) -> Result<Self, ProgramError> {
        let input = <Self as ValueProjection<ArrayType>>::projected(self)?;
        let indices = <Self as ValueProjection<ArrayType>>::projected(indices)?;
        Ok(Self::Array(input.gather(indices, dimensions, slice_sizes, options)?))
    }
}

impl<Stored: Value<Type = ArrayType>, V: Value<Type = ArrayType>> Gather<Stored> for V
where
    V::DispatchDomain: Context<Type = ArrayType, Operation: From<GatherOperation<Stored>>>,
{
    fn gather(
        &self,
        indices: &Self,
        dimensions: &GatherDimensionNumbers,
        slice_sizes: &[usize],
        options: &GatherOptions<Stored>,
    ) -> Result<Self, ProgramError> {
        // Bind homogeneous array values through their context. Mixed tracers use the canonical array projection;
        // requiring a homogeneous type here keeps array-operation trait obligations from becoming recursive.
        let operation = GatherOperation::new(dimensions.clone(), slice_sizes.to_vec()).with_options(options.clone());
        let mut outputs = self.dispatch_domain().bind(operation, Vec::new(), &[self.clone(), indices.clone()])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Gathers complete slices with first-class dimensions for the untouched input axes and query shape. This is a
/// composition of [`DynamicBroadcast`] and [`GatherOperation`]. Untouched input axes become paired gather batching axes
/// instead of runtime-sized windows. The selected axis uses a size-one window, and paired axes use size zero or one
/// according to their bounds. The output retains the exact runtime dimensions of the input and queries. Like
/// [`Gather::gather_axis`], negative indices are out of bounds rather than indexing backward from the end.
///
/// # Examples
///
/// ```rust
/// # use ryft_core::{Array, ArrayIrValue, DynamicGather, GatherMode};
/// // Shapes: input [2, 3], indices [2] -> output [2, 2].
/// let input = ArrayIrValue::Array(Array::matrix(2, 3, vec![1_i32, 2, 3, 4, 5, 6]).unwrap());
/// let indices = ArrayIrValue::Array(Array::vector(vec![2_i32, 0]).unwrap());
/// let output = input.dynamic_gather_axis(&indices, 1, GatherMode::Clip).unwrap();
/// assert_eq!(output, ArrayIrValue::Array(Array::matrix(2, 2, vec![3_i32, 1, 6, 4]).unwrap()));
/// ```
pub trait DynamicGather<Stored: Value<Type = ArrayType> = Array>: Value<Type = ArrayIrType> + Sized {
    /// Gathers along `axis`, replacing that axis with the complete shape of `indices` in the result.
    ///
    /// # Parameters
    ///
    ///   - `indices`: Integer query array. A scalar removes the selected input axis; a dynamic query shape is retained.
    ///   - `axis`: Input axis to select, with negative axes counted from the end of the input rank.
    ///   - `mode`: Bounds handling applied to each raw query index; see [`GatherMode`].
    fn dynamic_gather_axis<A: Into<Axis>>(
        &self,
        indices: &Self,
        axis: A,
        mode: GatherMode<Stored>,
    ) -> Result<Self, ProgramError>;
}

impl<Stored: Value<Type = ArrayType>, A: Value<Type = ArrayType, DispatchDomain: Zero<A>> + Gather<Stored> + Reshape>
    DynamicGather<Stored> for ArrayIrValue<A>
{
    fn dynamic_gather_axis<AxisValue: Into<Axis>>(
        &self,
        indices: &Self,
        axis: AxisValue,
        mode: GatherMode<Stored>,
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

impl<Stored: Value<Type = ArrayType>, V: Value<Type = ArrayIrType>> DynamicGather<Stored> for V
where
    V: DimensionSize + DynamicBroadcast + ValueProjection<ArrayType, Projected: Gather<Stored>>,
    V::DispatchDomain: Context<Type = ArrayIrType> + DimensionConstant + DynamicZero<V>,
{
    fn dynamic_gather_axis<A: Into<Axis>>(
        &self,
        indices: &Self,
        axis: A,
        mode: GatherMode<Stored>,
    ) -> Result<Self, ProgramError> {
        let input_type = self.r#type();
        let input_type = <&ArrayType>::try_from(input_type.as_ref())?;
        let indices_type = indices.r#type();
        let indices_type = <&ArrayType>::try_from(indices_type.as_ref())?;
        let axis = axis.into().normalize(input_type.rank()).map_err(|error| TypeError::invalid(error.to_string()))?;
        let mut dimensions = Vec::new();
        let mut batching = Vec::new();
        for input_axis in 0..input_type.rank() {
            if input_axis == axis {
                for query_axis in 0..indices_type.rank() {
                    dimensions.push(indices.dimension_size(query_axis)?);
                }
            } else {
                batching.push((input_axis, dimensions.len()));
                dimensions.push(self.dimension_size(input_axis)?);
            }
        }
        dimensions.push(self.dispatch_domain().dimension_constant(1)?);

        // Broadcast each scalar query over the untouched input coordinates. Those coordinates select matching
        // input/indices batches, so no symbolic extent is encoded as a host-sized gather window.
        let indices =
            indices.dynamic_broadcast(&dimensions, &(axis..axis + indices_type.rank()).collect::<Vec<_>>())?;
        let gather_dimensions =
            GatherDimensionNumbers::new(vec![], vec![axis], vec![axis]).with_batching_dimensions(batching);

        // Paired axes use a zero window when they may be empty. Their output extents come from the indices
        // dimensions, independently of these window sizes.
        let slice_sizes = (0..input_type.rank())
            .map(
                |input_axis| {
                    if input_axis == axis { 1 } else { input_type.dimension(input_axis).bounds().lower().min(1) }
                },
            )
            .collect::<Vec<_>>();
        let options = GatherOptions::new().with_mode(mode);
        if indices_type.element_count().map_err(|error| TypeError::invalid(error.to_string()))? == Some(0) {
            // Empty queries read no input elements, including when the selected axis itself is empty. Use the
            // ordinary gather metadata rules with a placeholder selected extent of one, then construct its empty
            // result. This preserves placement validation without staging an invalid collapsed size-one window.
            let mut shape = input_type.shape().dimensions().to_vec();
            shape[axis] = Dimension::Static(1);
            let output_type = input_type.clone().with_shape(Shape::new(shape)).gather(
                <&ArrayType>::try_from(indices.r#type().as_ref())?,
                &gather_dimensions,
                &slice_sizes,
                &options,
            )?;
            let dynamic_dimensions = output_type
                .shape()
                .dimensions()
                .iter()
                .zip(&dimensions)
                .filter_map(|(dimension, value)| matches!(dimension, Dimension::Dynamic(_)).then_some(value.clone()))
                .collect::<Vec<_>>();
            return self.dispatch_domain().dynamic_zero(&output_type, &dynamic_dimensions);
        }

        Ok(V::from_projected(self.clone().into_projected()?.gather(
            &indices.into_projected()?,
            &gather_dimensions,
            &slice_sizes,
            &options,
        )?))
    }
}

/// Validates that `axes` contains distinct axis indices in `0..bound`, optionally requiring strictly ascending order.
/// Empty lists are valid, including when `bound` is zero. This function neither reorders nor deduplicates the entries;
/// it returns a [`TypeError`] identifying the operation and field when a constraint is violated.
///
/// When `sorted` is `true`, ordering and uniqueness are checked before any range checks. Otherwise, entries are visited
/// in their supplied order, checking each entry's range before checking whether it duplicates an earlier entry. This
/// determines which diagnostic is returned when more than one constraint is violated.
///
/// # Parameters
///
///   - `operation_name`: Name of the operation being validated, used to identify it in diagnostics.
///   - `field`: Name of the operation field containing the axis indices, used to identify it in diagnostics.
///   - `axes`: Axis indices to validate. Duplicate entries are rejected regardless of `sorted`.
///   - `bound`: Exclusive upper bound for each index, typically the rank of the corresponding input or output.
///   - `sorted`: Whether to require strictly ascending indices. If `false`, any ordering of distinct, in-range
///     indices is accepted. If `true`, each index must also be greater than the preceding index.
pub(crate) fn validate_unique_in_range(
    operation_name: &'static str,
    field: &str,
    axes: &[usize],
    bound: usize,
    sorted: bool,
) -> Result<(), TypeError> {
    // Strictly increasing entries are necessarily unique, so sorted inputs need no set of previously seen axes.
    if sorted {
        for window in axes.windows(2) {
            if window[0] >= window[1] {
                return Err(TypeError::invalid(format!(
                    "`{operation_name}` `{field}` must be sorted and unique but got {axes:?}",
                )));
            }
        }
    }

    let mut seen = BTreeSet::new();
    for &axis in axes {
        if axis >= bound {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` `{field}` entry {axis} is out of range for bound {bound}",
            )));
        }
        if !sorted && !seen.insert(axis) {
            return Err(TypeError::invalid(format!("`{operation_name}` `{field}` must be unique but got {axes:?}")));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::batching::DynamicArrayExtentBatchingPolicy;
    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReferenceDischarge, DataType,
        DimensionBounds, DimensionType, DimensionValue, DimensionVariable, Layout, LogicalMesh, Memory, MeshAxis,
        MeshAxisType, RaggedAxis, Sharding, ShardingDimension, StridedLayout, i1, i4, u4,
    };
    use crate::batching::batch;
    use crate::differentiation::{TransposableOperation, TranspositionContext, differentiate_at};
    use crate::macros::{
        check_operation_batching, check_operation_partial_evaluation, check_operation_transposition,
        check_operation_type_inference,
    };
    use crate::parameters::{Parameter, Placeholder};
    use crate::partial::PartialValue;
    use crate::programs::{
        EffectClasses, EmptyRegionDriver, Program, ProgramBuilder, ReferenceDischargeContext, ReferenceDischargeValue,
        ReferenceDischargeableOperation,
    };
    use crate::tracing::{Trace, Tracer, TracingContext};

    use super::*;

    /// Tracer of the mixed array IR tracing context used by the batching and differentiation edge cases.
    type IrTracer = Tracer<TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>;

    /// Mixed program with the dynamic `items` extent, a packed `f32[items, 3]` input, and packed `i32[items, 1, 1]`
    /// indices as inputs, staging one `gather` that selects one element per row with both inputs jointly mapped over
    /// `items` through the dynamic-extent batching policy.
    fn jointly_mapped_dynamic_gather_program(
        items: DimensionVariable,
    ) -> Program<
        ArrayIrValue<Array>,
        ArrayIrOperation<Array>,
        (ArrayIrValue<Array>, ArrayIrValue<Array>, ArrayIrValue<Array>),
        ArrayIrValue<Array>,
    > {
        let input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(items.clone()), Dimension::Static(3)]));
        let indices_type = ArrayType::new(
            DataType::I32,
            Shape::new(vec![Dimension::Dynamic(items.clone()), Dimension::Static(1), Dimension::Static(1)]),
        );
        let operation = GatherOperation::new(GatherDimensionNumbers::new(Vec::new(), vec![0], vec![0]), vec![1]);
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(extent, input, indices): (IrTracer, IrTracer, IrTracer)| {
                let context = BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
                    ProjectedContext::new(extent.context().clone()),
                    extent,
                );
                let input = ValueProjection::<ArrayType>::into_projected(input)?;
                let indices = ValueProjection::<ArrayType>::into_projected(indices)?;
                let (outputs, _) = operation
                    .batch(
                        &context,
                        &EmptyRegionDriver,
                        &[ArrayBatch::new(input, BatchAxis::new(0))?, ArrayBatch::new(indices, BatchAxis::new(0))?],
                    )?
                    .into_parts();
                assert_eq!(outputs.len(), 1);
                assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
                let output = outputs.into_iter().next().unwrap().into_value();
                Ok(<IrTracer as ValueProjection<ArrayType>>::from_projected(output))
            },
            (
                ArrayIrType::Dimension(DimensionType::new(items)),
                ArrayIrType::Array(input_type),
                ArrayIrType::Array(indices_type),
            ),
        )
        .unwrap();
        program
    }

    /// Mixed program that gathers elements 1 and 3 of a rank-1 input of the given type while requesting a replicated
    /// output placement on `mesh`, with the constant indices placed in pinned host memory like the input.
    fn placed_take_program(
        input_type: ArrayType,
        mesh: &LogicalMesh,
    ) -> Program<ArrayIrValue<Array>, ArrayIrOperation<Array>, Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>> {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
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
        builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap()
    }

    /// Mixed program that gathers three elements of an `f64[extent]` input at the constant indices `[-1, 1, 5]` with
    /// the provided fill-mode operation, so that one query is in bounds and the other two are out of bounds.
    fn dynamic_fill_gather_program(
        operation: GatherOperation,
    ) -> Program<ArrayIrValue<Array>, ArrayIrOperation<Array>, Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>> {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(6)).unwrap());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input =
            builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)])).into());
        let indices = builder.add_constant(ArrayIrValue::Array(Array::matrix(3, 1, vec![-1_i32, 1, 5]).unwrap()));
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Gather(operation)),
                Vec::new(),
                vec![input, indices],
                None,
            )
            .unwrap()[0];
        builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap()
    }

    #[test]
    fn test_gather_mode() {
        assert_eq!(GatherMode::<Array>::default(), GatherMode::<Array>::PromiseInBounds);
        assert_eq!(GatherMode::<Array>::PromiseInBounds.name(), "promise_in_bounds");
        assert_eq!(GatherMode::<Array>::Clip.name(), "clip");
        assert_eq!(GatherMode::<Array>::Fill { value: None }.name(), "fill");
        assert_eq!(GatherMode::<Array>::PromiseInBounds.to_string(), "promise_in_bounds");
        assert_eq!(GatherMode::<Array>::Clip.to_string(), "clip");
        assert_eq!(GatherMode::<Array>::Fill { value: None }.to_string(), "fill");
        assert_eq!(format!("{:?}", GatherMode::<Array>::PromiseInBounds), "PromiseInBounds");
        assert_eq!(format!("{:?}", GatherMode::<Array>::Clip), "Clip");
        assert_eq!(format!("{:?}", GatherMode::<Array>::Fill { value: None }), "Fill { value: None }");
    }

    #[test]
    fn test_gather_dimension_numbers_new() {
        let dimensions = GatherDimensionNumbers::new(vec![1], vec![0], vec![0]);
        assert_eq!(dimensions.offset_dimensions(), &[1]);
        assert_eq!(dimensions.collapsed_slice_dimensions(), &[0]);
        assert_eq!(dimensions.start_index_map(), &[0]);
        assert_eq!(dimensions.batching_dimensions(), &[] as &[(usize, usize)]);
        assert_eq!(
            format!("{dimensions:?}"),
            "GatherDimensionNumbers { offset_dimensions: [1], collapsed_slice_dimensions: [0], start_index_map: [0], \
             batching_dimensions: [] }",
        );
        let lookup = HashMap::from([(dimensions.clone(), 7)]);
        assert_eq!(lookup.get(&dimensions), Some(&7));
        assert_eq!(lookup.get(&GatherDimensionNumbers::default()), None);
    }

    #[test]
    fn test_gather_dimension_numbers_with_batching_dimensions() {
        let dimensions = GatherDimensionNumbers::new(vec![2], vec![1], vec![1]).with_batching_dimensions(vec![(0, 1)]);
        assert_eq!(dimensions.offset_dimensions(), &[2]);
        assert_eq!(dimensions.collapsed_slice_dimensions(), &[1]);
        assert_eq!(dimensions.start_index_map(), &[1]);
        assert_eq!(dimensions.batching_dimensions(), &[(0, 1)]);
    }

    #[test]
    fn test_gather_options_new() {
        let options = GatherOptions::<Array>::new();
        assert_eq!(options, GatherOptions::default());
        assert_eq!(options.mode(), &GatherMode::PromiseInBounds);
        assert!(!options.indices_are_sorted());
        assert!(!options.unique_indices());
        assert_eq!(options.output_sharding(), None);
    }

    #[test]
    fn test_gather_options_with_mode() {
        let options =
            GatherOptions::new().with_mode(GatherMode::Fill { value: Some(Box::new(Array::scalar(7_i32).unwrap())) });
        assert_eq!(options.mode(), &GatherMode::Fill { value: Some(Box::new(Array::scalar(7_i32).unwrap())) });
        assert!(!options.indices_are_sorted());
        assert!(!options.unique_indices());
        assert_eq!(options.output_sharding(), None);
        let options = options.with_mode(GatherMode::Clip);
        assert_eq!(options.mode(), &GatherMode::Clip);
        assert_eq!(options.with_mode(GatherMode::Fill { value: None }).mode(), &GatherMode::Fill { value: None });
    }

    #[test]
    fn test_gather_options_with_indices_are_sorted() {
        let options = GatherOptions::<Array>::new().with_indices_are_sorted(true);
        assert!(options.indices_are_sorted());
        assert_eq!(options.with_indices_are_sorted(false), GatherOptions::new());
    }

    #[test]
    fn test_gather_options_with_unique_indices() {
        let options = GatherOptions::<Array>::new().with_unique_indices(true);
        assert!(options.unique_indices());
        assert_eq!(options.with_unique_indices(false), GatherOptions::new());
    }

    #[test]
    fn test_gather_options_with_output_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 2);
        let options = GatherOptions::<Array>::new().with_output_sharding(sharding.clone());
        assert_eq!(options.output_sharding(), Some(&sharding));
        assert_eq!(options.with_output_sharding(None), GatherOptions::new());
    }

    #[test]
    fn test_gather_options_resolved_fill_value() {
        let options = GatherOptions::new();
        assert_eq!(options.resolved_fill_value(DataType::I64), Array::scalar(i64::MIN));
        assert_eq!(options.resolved_fill_value(DataType::U64), Array::scalar(u64::MAX));
        assert_eq!(options.resolved_fill_value(DataType::Boolean), Array::scalar(true));
        assert_eq!(options.resolved_fill_value(DataType::I1), Array::scalar(i1::new(-1).unwrap()));
        assert_eq!(options.resolved_fill_value(DataType::U4), Array::scalar(u4::new(15).unwrap()));
        let float_fill = options.resolved_fill_value(DataType::F32).unwrap();
        assert!(float_fill.elements::<f32>().unwrap()[0].is_nan());
        let complex_fill = options.resolved_fill_value(DataType::C64).unwrap();
        let complex = complex_fill.elements::<num_complex::Complex<f32>>().unwrap()[0];
        assert!(complex.re.is_nan());
        assert_eq!(complex.im, 0.0);

        // Resolving an explicit fill preserves its encoding and validates its shape and element data type.
        let fill = Array::new(ArrayType::scalar(DataType::F32), 0x7fc12345_u32.to_le_bytes().to_vec()).unwrap();
        let options = options.with_mode(GatherMode::Fill { value: Some(Box::new(fill.clone())) });
        assert_eq!(options.resolved_fill_value(DataType::F32).unwrap().storage_bytes(), fill.storage_bytes());
        assert_eq!(
            options.resolved_fill_value(DataType::I32),
            Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` fill data type `f32` does not match input data type `i32`"
            ))
            .into()),
        );
        assert_eq!(
            options
                .with_mode(GatherMode::Fill { value: Some(Box::new(Array::vector(vec![1.0_f32]).unwrap())) })
                .resolved_fill_value(DataType::F32),
            Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` fill value must be a numeric or Boolean scalar"
            ))
            .into()),
        );
    }

    #[test]
    fn test_gather() {
        // Take whole rows of a [3, 2] matrix indexed by a [2, 1] index array: offset axis 1 carries the row (slice
        // sizes [1, 2]); axis 0 (the collapsed row axis) is driven by the start index.
        let dimensions = GatherDimensionNumbers::new(vec![1], vec![0], vec![0]);
        let operation = GatherOperation::new(dimensions, vec![1, 2]);
        assert_eq!(operation.name(), GATHER_OPERATION_NAME);
        assert_eq!(operation.dimensions(), &GatherDimensionNumbers::new(vec![1], vec![0], vec![0]));
        assert_eq!(operation.slice_sizes(), &[1, 2]);
        assert_eq!(operation.mode(), &GatherMode::PromiseInBounds);
        assert!(!operation.indices_are_sorted());
        assert!(!operation.unique_indices());
        assert_eq!(operation.output_sharding(), None);
        assert_eq!(operation.effects().classes(), EffectClasses::NONE);
        assert_eq!(
            format!("{operation}"),
            indoc! {"
                gather [
                    dimensions=(offset=[1], collapsed_slice=[0], start_index_map=[0], batching=[]),
                    slice_sizes=[1, 2],
                ]
            "}
            .trim_end(),
        );

        // Combining the builders preserves the gather geometry and renders each non-default option.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let configured = operation
            .clone()
            .with_mode(GatherMode::Fill { value: Some(Box::new(Array::scalar(0.5_f32).unwrap())) })
            .with_indices_are_sorted(true)
            .with_unique_indices(true)
            .with_output_sharding(Sharding::replicated(mesh.clone(), 2));
        assert_eq!(configured.dimensions(), operation.dimensions());
        assert_eq!(configured.slice_sizes(), operation.slice_sizes());
        assert_eq!(configured.mode(), &GatherMode::Fill { value: Some(Box::new(Array::scalar(0.5_f32).unwrap())) });
        assert!(configured.indices_are_sorted());
        assert!(configured.unique_indices());
        assert_eq!(configured.output_sharding(), Some(&Sharding::replicated(mesh, 2)));
        assert_eq!(
            format!("{configured}"),
            indoc! {"
                gather [
                    dimensions=(offset=[1], collapsed_slice=[0], start_index_map=[0], batching=[]),
                    slice_sizes=[1, 2],
                    mode=fill(value=0.5),
                    indices_are_sorted=true,
                    unique_indices=true,
                    output_sharding={mesh<['x'=2:explicit]>, [{}, {}]},
                ]
            "}
            .trim_end(),
        );
        assert_eq!(configured.clone().with_output_sharding(None).output_sharding(), None);
        let dimensions = GatherDimensionNumbers::new(vec![], vec![0], vec![0]);
        let clipping = GatherOptions::<Array>::new().with_mode(GatherMode::Clip);
        assert_eq!(
            format!("{}", GatherOperation::new(dimensions.clone(), vec![1]).with_options(clipping.clone())),
            indoc! {"
                gather [
                    dimensions=(offset=[], collapsed_slice=[0], start_index_map=[0], batching=[]),
                    slice_sizes=[1],
                    mode=clip,
                ]
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_gather_with_options() {
        let options = GatherOptions::<Array>::new().with_mode(GatherMode::Clip).with_unique_indices(true);
        let dimensions = GatherDimensionNumbers::new(vec![1], vec![0], vec![0]);
        let operation = GatherOperation::new(dimensions.clone(), vec![1, 2]).with_options(options.clone());
        assert_eq!(operation.dimensions(), &dimensions);
        assert_eq!(operation.slice_sizes(), &[1, 2]);
        assert_eq!(operation.options(), &options);
        let operation = operation.with_options(GatherOptions::new());
        assert_eq!(operation.mode(), &GatherMode::PromiseInBounds);
        assert!(!operation.unique_indices());
    }

    #[test]
    fn test_gather_type_inference() {
        let operation =
            GatherOperation::<Array>::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
        let input = ArrayType::new_static(DataType::F32, [3, 2]);
        let indices = ArrayType::new_static(DataType::I32, [2, 1]);
        let host_input = input.clone().with_memory(Memory::Host { pinned: true });
        let host_indices = indices.clone().with_memory(Memory::Host { pinned: true });
        let host_output = ArrayType::new_static(DataType::F32, [2, 2]).with_memory(Memory::Host { pinned: true });
        let vector = DimensionVariable::new("vector", DimensionBounds::new(1, Some(2)).unwrap());
        let dynamic_vector_indices =
            ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(vector)]));
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [
                {
                    input_types = [input.clone(), indices.clone()],
                    output_types = [ArrayType::new_static(DataType::F32, [2, 2])],
                },
                {
                    input_types = [input.clone()],
                    error = "expected 2 inputs but got 1",
                },
                {
                    input_types = [input.clone(), ArrayType::new_static(DataType::F32, [2, 1])],
                    error = format!(
                        "`{GATHER_OPERATION_NAME}` indices must be integer-typed but have type `f32[2, 1]`",
                    ),
                },
                {
                    input_types = [input.clone(), ArrayType::scalar(DataType::I32)],
                    error = format!(
                        "`{GATHER_OPERATION_NAME}` indices must have rank at least 1 (the trailing index vector)",
                    ),
                },
                {
                    input_types = [input.clone(), dynamic_vector_indices],
                    error = format!(
                        "`{GATHER_OPERATION_NAME}` indices index vector dimension must have a static extent",
                    ),
                },
                {
                    input_types = [host_input.clone(), host_indices],
                    output_types = [host_output],
                },
                {
                    input_types = [host_input, indices.clone()],
                    error = format!(
                        "`{GATHER_OPERATION_NAME}` input and indices must share one memory space but reside in \
                         `Host[Pinned]` and `Device`",
                    ),
                },
            ],
        );

        // Query-batch axes come directly from the indices array. A dynamic query extent therefore preserves the same
        // identity in the output and needs no separate first-class dimension input on `gather`.
        let query = DimensionVariable::new("query", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_indices =
            ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(query.clone()), Dimension::Static(1)]));
        assert_eq!(
            operation.infer_output_types(&[input, dynamic_indices], &[]),
            Ok(vec![ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(query), Dimension::Static(2)]))]),
        );
    }

    #[test]
    fn test_gather_type_inference_invalid_fill() {
        /// A scalar-typed value whose identity cannot be embedded in a literal payload.
        #[derive(Clone, Debug, ryft_macros::Parameter)]
        struct NonLiteralFill;

        impl Display for NonLiteralFill {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("non_literal_fill")
            }
        }

        impl Typed for NonLiteralFill {
            type Type = ArrayType;

            fn r#type(&self) -> std::borrow::Cow<'_, ArrayType> {
                std::borrow::Cow::Owned(ArrayType::scalar(DataType::F32))
            }
        }

        impl Value for NonLiteralFill {
            type DispatchDomain = EagerContext<Self>;
            type ExecutionDomain = EagerContext<Self>;

            fn dispatch_domain(&self) -> Self::DispatchDomain {
                EagerContext::new()
            }

            fn execution_domain(&self) -> Self::ExecutionDomain {
                EagerContext::new()
            }

            fn validate_as_constant(&self) -> Result<(), TypeError> {
                Err(TypeError::invalid("non-literal fill cannot be stored as a constant"))
            }
        }

        // Even an empty query must validate constant storage; neither type inference nor interpretation may
        // bypass the stored value's contract merely because the result will contain no elements.
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![], vec![0], vec![0]), vec![1])
            .with_mode(GatherMode::Fill { value: Some(Box::new(NonLiteralFill)) });
        let input_types = [ArrayType::new_static(DataType::F32, [3]), ArrayType::new_static(DataType::I32, [0, 1])];
        assert_eq!(
            operation.infer_output_types(&input_types, &[]),
            Err(TypeError::invalid("non-literal fill cannot be stored as a constant")),
        );
        assert_eq!(
            operation.interpret(&EagerContext::<ArrayType>::new(), &EmptyRegionDriver, &input_types),
            Err(TypeError::invalid("non-literal fill cannot be stored as a constant").into()),
        );
    }

    #[test]
    fn test_gather_type_inference_paired_extents() {
        // Specialization can retain an exact nominal dimension on one side of a paired batch while the other is
        // already static. Both descriptions prove the same extent without equating unrelated symbolic dimensions.
        let exact = Dimension::Dynamic(DimensionVariable::new("batch", DimensionBounds::new(0, Some(1)).unwrap()));
        let operation = GatherOperation::<Array>::new(
            GatherDimensionNumbers::new(vec![], vec![1], vec![1]).with_batching_dimensions(vec![(0, 0)]),
            vec![0, 1],
        );
        let static_input = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0), Dimension::Static(4)]));
        let exact_input = ArrayType::new(DataType::F64, Shape::new(vec![exact.clone(), Dimension::Static(4)]));
        let static_indices = ArrayType::new(
            DataType::I32,
            Shape::new(vec![Dimension::Static(0), Dimension::Static(2), Dimension::Static(1)]),
        );
        let exact_indices =
            ArrayType::new(DataType::I32, Shape::new(vec![exact.clone(), Dimension::Static(2), Dimension::Static(1)]));
        let static_output = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0), Dimension::Static(2)]));
        let exact_output = ArrayType::new(DataType::F64, Shape::new(vec![exact, Dimension::Static(2)]));
        assert_eq!(
            static_input.gather(&static_indices, operation.dimensions(), operation.slice_sizes(), operation.options()),
            Ok(static_output.clone())
        );
        assert_eq!(
            static_input.gather(&exact_indices, operation.dimensions(), operation.slice_sizes(), operation.options()),
            Ok(exact_output.clone())
        );
        assert_eq!(
            exact_input.gather(&static_indices, operation.dimensions(), operation.slice_sizes(), operation.options()),
            Ok(static_output)
        );
        assert_eq!(
            exact_input.gather(&exact_indices, operation.dimensions(), operation.slice_sizes(), operation.options()),
            Ok(exact_output)
        );

        // Independent dynamic dimensions with equal bounds do not prove equal extents.
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
        assert_eq!(
            input.gather(&indices, operation.dimensions(), operation.slice_sizes(), operation.options()),
            Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` batching dimensions must have equal extents, but input axis 0 and indices \
                 axis 0 differ",
            ))
            .into()),
        );
    }

    #[test]
    fn test_gather_type_inference_invalid_dimension_maps() {
        let input = ArrayType::new_static(DataType::F32, [3, 2]);
        let indices = ArrayType::new_static(DataType::I32, [2, 1]);

        // Each dimension-number list is validated against its own rank bound: offset dimensions against the output
        // rank, input axis lists against the input rank, and indices axis lists against the indices rank.
        check_operation_type_inference!(
            operation = GatherOperation::<Array>::new(GatherDimensionNumbers::new(vec![1, 0], vec![], vec![0]), vec![1, 2]),
            cases = [{
                input_types = [input.clone(), indices.clone()],
                error = format!(
                    "`{GATHER_OPERATION_NAME}` `offset_dimensions` must be sorted and unique but got [1, 0]",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = GatherOperation::<Array>::new(GatherDimensionNumbers::new(vec![2], vec![0], vec![0]), vec![1, 2]),
            cases = [{
                input_types = [input.clone(), indices.clone()],
                error = format!("`{GATHER_OPERATION_NAME}` `offset_dimensions` entry 2 is out of range for bound 2"),
            }],
        );
        check_operation_type_inference!(
            operation = GatherOperation::<Array>::new(GatherDimensionNumbers::new(vec![1], vec![2], vec![0]), vec![1, 2]),
            cases = [{
                input_types = [input.clone(), indices.clone()],
                error = format!(
                    "`{GATHER_OPERATION_NAME}` `collapsed_slice_dimensions` entry 2 is out of range for bound 2",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = GatherOperation::<Array>::new(
                GatherDimensionNumbers::new(vec![1], vec![0], vec![0]).with_batching_dimensions(vec![(2, 0)]),
                vec![1, 2],
            ),
            cases = [{
                input_types = [input.clone(), indices.clone()],
                error = format!(
                    "`{GATHER_OPERATION_NAME}` `batching_dimensions input axes` entry 2 is out of range for bound 2",
                ),
            }],
        );

        // The start index map has one entry per index vector component, each naming a distinct input axis.
        check_operation_type_inference!(
            operation = GatherOperation::<Array>::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0, 1]), vec![1, 2]),
            cases = [{
                input_types = [input.clone(), indices.clone()],
                error = format!(
                    "`{GATHER_OPERATION_NAME}` `start_index_map` has length 2 but the index vector extent is 1",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = GatherOperation::<Array>::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![2]), vec![1, 2]),
            cases = [{
                input_types = [input.clone(), indices.clone()],
                error = format!("`{GATHER_OPERATION_NAME}` `start_index_map` entry 2 is out of range for bound 2"),
            }],
        );
        check_operation_type_inference!(
            operation = GatherOperation::<Array>::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0, 0]), vec![1, 2]),
            cases = [{
                input_types = [input.clone(), ArrayType::new_static(DataType::I32, [2, 2])],
                error = format!("`{GATHER_OPERATION_NAME}` `start_index_map` must be unique but got [0, 0]"),
            }],
        );

        // Batching axes pair 1:1, name distinct in-range indices axes other than the index vector, and are disjoint
        // from both the start index map and the collapsed axes.
        check_operation_type_inference!(
            operation = GatherOperation::<Array>::new(
                GatherDimensionNumbers::new(vec![1], vec![0], vec![0]).with_batching_dimensions(vec![(1, 2)]),
                vec![1, 2],
            ),
            cases = [{
                input_types = [input.clone(), indices.clone()],
                error = format!(
                    "`{GATHER_OPERATION_NAME}` `batching_dimensions indices axes` entry 2 is out of range for bound 2",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = GatherOperation::<Array>::new(
                GatherDimensionNumbers::new(vec![], vec![2], vec![2]).with_batching_dimensions(vec![(0, 0), (1, 0)]),
                vec![1, 1, 1],
            ),
            cases = [{
                input_types = [ArrayType::new_static(DataType::F32, [2, 2, 3]), indices.clone()],
                error = format!(
                    "`{GATHER_OPERATION_NAME}` `batching_dimensions indices axes` must be unique but got [0, 0]",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = GatherOperation::<Array>::new(
                GatherDimensionNumbers::new(vec![1], vec![], vec![0]).with_batching_dimensions(vec![(0, 0)]),
                vec![1, 3],
            ),
            cases = [{
                input_types = [ArrayType::new_static(DataType::F32, [2, 3]), indices.clone()],
                error = format!(
                    "`{GATHER_OPERATION_NAME}` `start_index_map` and `batching_dimensions input axes` must be disjoint",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = GatherOperation::<Array>::new(
                GatherDimensionNumbers::new(vec![1], vec![], vec![0]).with_batching_dimensions(vec![(1, 1)]),
                vec![1, 1],
            ),
            cases = [{
                input_types = [input.clone(), indices.clone()],
                error = format!(
                    "`{GATHER_OPERATION_NAME}` `batching_dimensions indices axes` cannot name the index vector \
                     dimension 1",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = GatherOperation::<Array>::new(
                GatherDimensionNumbers::new(vec![], vec![0], vec![1]).with_batching_dimensions(vec![(0, 0)]),
                vec![1, 1],
            ),
            cases = [{
                input_types = [input.clone(), indices.clone()],
                error = format!(
                    "`{GATHER_OPERATION_NAME}` `collapsed_slice_dimensions` and `batching_dimensions input axes` \
                     must be disjoint",
                ),
            }],
        );

        // The offset dimensions account for exactly the input axes that are neither collapsed nor batching.
        check_operation_type_inference!(
            operation = GatherOperation::<Array>::new(GatherDimensionNumbers::new(vec![1, 2], vec![0], vec![0]), vec![1, 2]),
            cases = [{
                input_types = [input.clone(), indices],
                error = format!(
                    "`{GATHER_OPERATION_NAME}` `offset_dimensions` has length 2 but the number of non-collapsed, \
                     non-batching input axes is 1",
                ),
            }],
        );

        // `gather` carries no regions.
        let operation =
            GatherOperation::<Array>::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
        assert_eq!(
            operation.infer_output_types(&[], &[RegionInterface::new(vec![], vec![], EffectClasses::NONE)]),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_gather_type_inference_invalid_slice_sizes() {
        let input = ArrayType::new_static(DataType::F32, [3, 2]);
        let indices = ArrayType::new_static(DataType::I32, [2, 1]);
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(5)).unwrap());

        // One slice size per input axis, each within the static extent or the guaranteed minimum dynamic extent.
        check_operation_type_inference!(
            operation = GatherOperation::<Array>::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1]),
            cases = [{
                input_types = [input.clone(), indices.clone()],
                error = format!("`{GATHER_OPERATION_NAME}` `slice_sizes` has length 1 but the input has rank 2"),
            }],
        );
        check_operation_type_inference!(
            operation = GatherOperation::<Array>::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]),
            cases = [
                {
                    input_types = [ArrayType::new_static(DataType::F32, [3, 1]), indices.clone()],
                    error = format!("`{GATHER_OPERATION_NAME}` slice size 2 at axis 1 exceeds the input extent 1"),
                },
                {
                    input_types = [
                        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Dynamic(rows)])),
                        indices.clone(),
                    ],
                    error = format!(
                        "`{GATHER_OPERATION_NAME}` slice size 2 exceeds the guaranteed minimum extent 1 of dynamic \
                         input axis 1",
                    ),
                },
            ],
        );

        // Collapsed axes take exactly one element and batching axes at most one.
        check_operation_type_inference!(
            operation = GatherOperation::<Array>::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![2, 2]),
            cases = [{
                input_types = [input.clone(), indices],
                error = format!(
                    "`{GATHER_OPERATION_NAME}` collapsed slice dimension 0 must have slice size 1 but has 2",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = GatherOperation::<Array>::new(
                GatherDimensionNumbers::new(vec![], vec![1], vec![1]).with_batching_dimensions(vec![(0, 0)]),
                vec![2, 1],
            ),
            cases = [{
                input_types = [input, ArrayType::new_static(DataType::I32, [3, 1])],
                error = format!(
                    "`{GATHER_OPERATION_NAME}` input batching dimension 0 must have slice size at most 1 but has 2",
                ),
            }],
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
        let operation = GatherOperation::<Array>::new(GatherDimensionNumbers::new(vec![1], vec![], vec![0]), vec![4]);
        let expected = ArrayType::new_static(DataType::F32, [1, 4])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::Replicated; 2])
                    .unwrap()
                    .with_unreduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();

        // Complete windows of an unreduced input keep its pending reduction; a fill written on every device would be
        // counted once per device by that reduction, so fill mode is rejected for unreduced inputs.
        assert_eq!(
            input.gather(&indices, operation.dimensions(), operation.slice_sizes(), operation.options()),
            Ok(expected)
        );
        assert_eq!(
            input.gather(
                &indices,
                operation.dimensions(),
                operation.slice_sizes(),
                &operation.options().clone().with_mode(GatherMode::Fill { value: None }),
            ),
            Err(TypeError::invalid(format!("`{GATHER_OPERATION_NAME}` fill mode does not support unreduced inputs"))
                .into()),
        );
        let distributed_indices = ArrayType::new_static(DataType::I32, [2, 1])
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated]).unwrap(),
            )
            .unwrap();
        assert_eq!(
            input.gather(&distributed_indices, operation.dimensions(), operation.slice_sizes(), operation.options()),
            Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` reduction-state inputs require replicated, invariant indices",
            ))
            .into()),
        );
    }

    #[test]
    fn test_gather_type_inference_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let indices = ArrayType::new_static(DataType::I32, [1, 1]);
        let operation = GatherOperation::<Array>::new(GatherDimensionNumbers::new(vec![1], vec![], vec![0]), vec![4]);
        let expected = ArrayType::new_static(DataType::F32, [1, 4])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::Replicated, ShardingDimension::sharded(["x"])])
                    .unwrap(),
            )
            .unwrap();
        // Full windows retain placement even on an explicitly indexed axis.
        assert_eq!(
            input.gather(&indices, operation.dimensions(), operation.slice_sizes(), operation.options()),
            Ok(expected)
        );
        let partial = GatherOperation::<Array>::new(GatherDimensionNumbers::new(vec![1], vec![], vec![]), vec![2]);
        assert_eq!(
            input.gather(
                &ArrayType::new_static(DataType::I32, [1, 0]),
                partial.dimensions(),
                partial.slice_sizes(),
                partial.options(),
            ),
            Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` partial sharded windows require explicit output sharding"
            ))
            .into()),
        );

        // Paired batching axes inherit the input placement when the indices leave theirs replicated.
        let batched_input = ArrayType::new_static(DataType::F32, [2, 3])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated])
                    .unwrap(),
            )
            .unwrap();
        let batched_indices = ArrayType::new_static(DataType::I32, [2, 1]);
        let batched = GatherOperation::<Array>::new(
            GatherDimensionNumbers::new(vec![], vec![1], vec![1]).with_batching_dimensions(vec![(0, 0)]),
            vec![1, 1],
        );
        let expected = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        assert_eq!(
            batched_input.gather(&batched_indices, batched.dimensions(), batched.slice_sizes(), batched.options()),
            Ok(expected)
        );

        // Paired batching axes sharded differently are ambiguous unless the output placement is requested.
        let two_axis_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let conflicting_input = ArrayType::new_static(DataType::F32, [2, 3])
            .with_sharding(
                Sharding::new(
                    two_axis_mesh.clone(),
                    vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated],
                )
                .unwrap(),
            )
            .unwrap();
        let conflicting_indices = ArrayType::new_static(DataType::I32, [2, 1])
            .with_sharding(
                Sharding::new(
                    two_axis_mesh.clone(),
                    vec![ShardingDimension::sharded(["y"]), ShardingDimension::Replicated],
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(
            conflicting_input.gather(
                &conflicting_indices,
                batched.dimensions(),
                batched.slice_sizes(),
                batched.options(),
            ),
            Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` conflicting batching-axis shardings require explicit output sharding"
            ))
            .into()),
        );
        let requested = Sharding::new(two_axis_mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
        assert_eq!(
            conflicting_input.gather(
                &conflicting_indices,
                batched.dimensions(),
                batched.slice_sizes(),
                &batched.options().clone().with_output_sharding(requested.clone()),
            ),
            Ok(ArrayType::new_static(DataType::F32, [2]).with_sharding(requested).unwrap()),
        );

        // The index vector axis must be replicated over explicit mesh axes, and both inputs must share one mesh.
        let sharded_vector_indices = indices
            .clone()
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::Replicated, ShardingDimension::sharded(["x"])])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            input.gather(&sharded_vector_indices, operation.dimensions(), operation.slice_sizes(), operation.options()),
            Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` indices index vector dimension must be replicated over explicit mesh axes"
            ))
            .into()),
        );
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let other_mesh_indices = indices.clone().with_sharding(Sharding::replicated(other_mesh.clone(), 2)).unwrap();
        assert_eq!(
            input.gather(&other_mesh_indices, operation.dimensions(), operation.slice_sizes(), operation.options()),
            Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` input and indices shardings must use the same mesh"
            ))
            .into()),
        );

        // A requested output sharding must use the common mesh, preserve reduction and manual-axis state, have the
        // output rank, and avoid automatic mesh axes.
        let requested = Sharding::replicated(other_mesh, 2);
        assert_eq!(
            input.gather(
                &indices,
                operation.dimensions(),
                operation.slice_sizes(),
                &operation.options().clone().with_output_sharding(requested),
            ),
            Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` requested output sharding uses a different mesh"
            ))
            .into()),
        );
        let requested = Sharding::replicated(mesh.clone(), 2).with_unreduced_axes(["x"]).unwrap();
        assert_eq!(
            input.gather(
                &indices,
                operation.dimensions(),
                operation.slice_sizes(),
                &operation.options().clone().with_output_sharding(requested),
            ),
            Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` requested output sharding changes reduction or manual-axis state"
            ))
            .into()),
        );
        let requested = Sharding::replicated(mesh.clone(), 1);
        assert_eq!(
            input.gather(
                &indices,
                operation.dimensions(),
                operation.slice_sizes(),
                &operation.options().clone().with_output_sharding(requested),
            ),
            Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` output sharding rank (1) does not match the output rank (2)"
            ))
            .into()),
        );
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let requested =
            Sharding::new(auto_mesh, vec![ShardingDimension::Replicated, ShardingDimension::sharded(["a"])]).unwrap();
        assert_eq!(
            ArrayType::new_static(DataType::F32, [4]).gather(
                &indices,
                operation.dimensions(),
                operation.slice_sizes(),
                &operation.options().clone().with_output_sharding(requested),
            ),
            Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` output sharding cannot reference auto mesh axes"
            ))
            .into()),
        );

        // Indices never carry reduction state of their own.
        let reduced_indices =
            indices.with_sharding(Sharding::replicated(mesh, 2).with_reduced_axes(["x"]).unwrap()).unwrap();
        assert_eq!(
            input.gather(&reduced_indices, operation.dimensions(), operation.slice_sizes(), operation.options()),
            Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` indices cannot carry reduced or unreduced mesh axes"
            ))
            .into()),
        );

        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();

        // Input [4, 2] sharded only on the feature axis (axis 1); axis 0 (indexed by the start index) is replicated.
        let input = ArrayType::new_static(DataType::F32, [4, 2])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["y"])])
                    .unwrap(),
            )
            .unwrap();
        let indices = ArrayType::new_static(DataType::I32, [3, 1]);
        let operation =
            GatherOperation::<Array>::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);

        // Output [3, 2]: the query axis (from the indices) is replicated, the feature axis keeps `y`.
        let output = operation.infer_output_types(&[input, indices.clone()], &[]).unwrap();
        assert_eq!(
            output[0].sharding().unwrap().dimensions(),
            &[ShardingDimension::Replicated, ShardingDimension::sharded(["y"])],
        );

        // Sharding the start-indexed input axis over an explicit mesh axis is ambiguous without an output sharding.
        let input = ArrayType::new_static(DataType::F32, [4, 2])
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap(),
            )
            .unwrap();
        assert_eq!(
            operation.infer_output_types(&[input, indices], &[]),
            Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` input axis 0 is indexed by the start indices and must be replicated over \
                 explicit mesh axes; request an explicit output sharding to resolve placement",
            ))),
        );
    }

    #[test]
    fn test_gather_reference_discharge() {
        // Reference-free replay preserves the complete configured gather payload. Generic replay behavior and
        // reference rejection are covered by the reference-discharge macro tests.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let expected = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2])
            .with_mode(GatherMode::Fill { value: Some(Box::new(Array::scalar(7_f64).unwrap())) })
            .with_indices_are_sorted(true)
            .with_unique_indices(true)
            .with_output_sharding(Sharding::replicated(mesh.clone(), 2));
        let operation = ArrayIrOperation::Array(ArrayOperation::Gather(expected.clone()));
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let staging = ReferenceDischargeContext::<_, ArrayReferenceDischarge>::new(trace.clone());
        let inputs = [
            ReferenceDischargeValue::Value(trace.input(ArrayType::new_static(DataType::F64, [3, 2]).into())),
            ReferenceDischargeValue::Value(trace.input(ArrayType::new_static(DataType::I32, [2, 1]).into())),
        ];
        let outputs = operation.discharge_references(&staging, &EmptyRegionDriver, &inputs).unwrap();
        assert_eq!(outputs.len(), 1);
        let ReferenceDischargeValue::Value(output) = &outputs[0] else {
            panic!("expected a value carrier but got {}", outputs[0]);
        };
        assert_eq!(
            output.r#type().as_ref(),
            &ArrayIrType::Array(
                ArrayType::new_static(DataType::F64, [2, 2]).with_sharding(Sharding::replicated(mesh, 2)).unwrap(),
            ),
        );
        let builder = trace.builder().borrow();
        assert_eq!(builder.instructions().len(), 1);
        let ArrayIrOperation::Array(ArrayOperation::Gather(staged)) = builder.instructions()[0].operation() else {
            panic!("expected a staged gather");
        };
        assert_eq!(staged, &expected);
    }

    #[test]
    fn test_gather_interpretation() {
        // Index vectors store [column, row], reversing the input axis order. Each query selects a whole 1x2 window.
        let input = Array::matrix(3, 4, vec![0_i32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).unwrap();
        let indices = Array::matrix(3, 2, vec![1_i32, 2, 0, 0, 2, 1]).unwrap();
        assert_eq!(
            input.gather(
                &indices,
                &GatherDimensionNumbers::new(vec![1], vec![0], vec![1, 0]),
                &[1, 2],
                &GatherOptions::new(),
            ),
            Array::matrix(3, 2, vec![9_i32, 10, 0, 1, 6, 7]),
        );

        // A start inside the input can still define an invalid window. Clip moves the complete window left;
        // fill replaces both elements, including the one whose coordinate would otherwise be in bounds.
        let input = Array::vector(vec![10_i32, 20, 30, 40]).unwrap();
        let indices = Array::matrix(2, 1, vec![2_i32, 3]).unwrap();
        let dimensions = GatherDimensionNumbers::new(vec![1], vec![], vec![0]);
        assert_eq!(
            input.gather(&indices, &dimensions, &[2], &GatherOptions::new().with_mode(GatherMode::Clip)),
            Array::matrix(2, 2, vec![30_i32, 40, 30, 40]),
        );
        assert_eq!(
            input.gather(
                &indices,
                &dimensions,
                &[2],
                &GatherOptions::new()
                    .with_mode(GatherMode::Fill { value: Some(Box::new(Array::scalar(-99_i32).unwrap())) }),
            ),
            Array::matrix(2, 2, vec![30_i32, 40, -99, -99]),
        );

        // Window elements reuse a query, while interleaved output axes revisit queries. Both must reset the
        // whole-window fill decision correctly when moving between in-bounds and out-of-bounds starts.
        let input = Array::vector(vec![10_i32, 20, 30, 40]).unwrap();
        let indices = Array::matrix(4, 1, vec![-1_i32, 1, 5, 0]).unwrap();
        let options =
            GatherOptions::new().with_mode(GatherMode::Fill { value: Some(Box::new(Array::scalar(-99_i32).unwrap())) });
        assert_eq!(
            input.gather(&indices, &GatherDimensionNumbers::new(vec![1], vec![], vec![0]), &[2], &options),
            Array::matrix(4, 2, vec![-99_i32, -99, 20, 30, -99, -99, 10, 20]),
        );
        assert_eq!(
            input.gather(&indices, &GatherDimensionNumbers::new(vec![0], vec![], vec![0]), &[2], &options),
            Array::matrix(2, 4, vec![-99_i32, 20, -99, 10, -99, 30, -99, 20]),
        );

        let dimensions = GatherDimensionNumbers::new(vec![], vec![0], vec![0]);
        let input = Array::vector(vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        let indices = Array::matrix(2, 1, vec![1_i32, 3]).unwrap();
        let operation = GatherOperation::new(dimensions, vec![1]);
        let context = EagerContext::<Array>::new();
        assert_eq!(
            operation.interpret(&context, &EmptyRegionDriver, &[input.clone(), indices]),
            Ok(vec![Array::vector(vec![20.0, 40.0]).unwrap()]),
        );
        assert_eq!(
            operation.interpret(&context, &EmptyRegionDriver, &[]),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        );

        // Clipping moves a complete window in bounds, while fill mode replaces each invalid window.
        let indices = Array::matrix(2, 1, vec![1_i32, 5]).unwrap();
        assert_eq!(
            input.gather(
                &indices,
                operation.dimensions(),
                operation.slice_sizes(),
                &operation.options().clone().with_mode(GatherMode::Clip),
            ),
            Array::vector(vec![20.0, 40.0]),
        );
        let filled = input
            .gather(
                &indices,
                operation.dimensions(),
                operation.slice_sizes(),
                &operation.options().clone().with_mode(GatherMode::Fill { value: None }),
            )
            .unwrap();
        assert_eq!(filled.r#type().as_ref(), &ArrayType::new_static(DataType::F64, [2]));
        let filled_elements = filled.elements::<f64>().unwrap();
        assert_eq!(filled_elements[0], 20.0);
        assert!(filled_elements[1].is_nan());

        // An abstract stored value participates in inference and interpretation without host scalar bytes.
        let abstract_operation =
            GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2])
                .with_mode(GatherMode::Fill { value: Some(Box::new(ArrayType::scalar(DataType::F32))) });
        let input_types = [ArrayType::new_static(DataType::F32, [3, 2]), ArrayType::new_static(DataType::I32, [2, 1])];
        assert_eq!(
            abstract_operation.interpret(&EagerContext::<ArrayType>::new(), &EmptyRegionDriver, &input_types),
            Ok(vec![ArrayType::new_static(DataType::F32, [2, 2])]),
        );

        let context = TracingContext::<ArrayType, GatherOperation<ArrayType>>::new();
        let staged_input = context.input(input_types[0].clone());
        let staged_indices = context.input(input_types[1].clone());
        assert_eq!(
            staged_input
                .gather(
                    &staged_indices,
                    abstract_operation.dimensions(),
                    abstract_operation.slice_sizes(),
                    abstract_operation.options(),
                )
                .unwrap()
                .r#type()
                .as_ref(),
            &ArrayType::new_static(DataType::F32, [2, 2]),
        );
    }

    #[test]
    fn test_gather_interpretation_fill_encodings() {
        // Explicit NaN payloads survive options cloning and eager filling without numeric conversion.
        let fill = Array::new(ArrayType::scalar(DataType::F32), 0x7fc12345_u32.to_le_bytes().to_vec()).unwrap();
        let filling = GatherOptions::new().with_mode(GatherMode::Fill { value: Some(Box::new(fill.clone())) });
        let input = Array::vector(vec![1.0_f32]).unwrap();
        let out_of_bounds = Array::matrix(1, 1, vec![2_i32]).unwrap();
        let dimensions = GatherDimensionNumbers::new(vec![], vec![0], vec![0]);
        let output = input.gather(&out_of_bounds, &dimensions, &[1], &filling).unwrap();
        assert_eq!(output.storage_bytes(), fill.storage_bytes());
        let negative_zero = filling
            .clone()
            .with_mode(GatherMode::Fill { value: Some(Box::new(Array::scalar(-0.0_f32).unwrap())) });
        assert_eq!(
            input.gather(&out_of_bounds, &dimensions, &[1], &negative_zero).unwrap().elements::<f32>().unwrap()[0]
                .to_bits(),
            (-0.0_f32).to_bits(),
        );

        // A scalar fill can carry storage placement without changing its logical value.
        let pinned_fill = Array::new(
            ArrayType::scalar(DataType::F32).with_memory(Memory::Host { pinned: true }),
            7.0_f32.to_le_bytes().to_vec(),
        )
        .unwrap();
        assert_eq!(
            input.gather(
                &out_of_bounds,
                &dimensions,
                &[1],
                &filling.clone().with_mode(GatherMode::Fill { value: Some(Box::new(pinned_fill)) }),
            ),
            Array::vector(vec![7.0_f32]),
        );
        let invalid_fill = filling
            .clone()
            .with_mode(GatherMode::Fill { value: Some(Box::new(Array::vector(vec![1.0_f32]).unwrap())) });
        assert_eq!(
            input.gather(
                &Array::new(ArrayType::new_static(DataType::I32, [0, 1]), Vec::new()).unwrap(),
                &dimensions,
                &[1],
                &invalid_fill,
            ),
            Err(TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` fill value must be a numeric or Boolean scalar"
            ))
            .into()),
        );

        // Changing mode discards its fill payload. Returning to default fill cannot resurrect the old NaN bits.
        let clipping = filling.clone().with_mode(GatherMode::Clip);
        assert_eq!(clipping.mode(), &GatherMode::Clip);
        assert_eq!(
            input.gather(&out_of_bounds, &dimensions, &[1], &clipping),
            Ok(Array::vector(vec![1.0_f32]).unwrap())
        );
        let refilling = clipping.with_mode(GatherMode::Fill { value: None });
        assert_ne!(refilling, filling);
        assert_eq!(refilling.mode(), &GatherMode::Fill { value: None });
        assert_eq!(
            input.gather(&out_of_bounds, &dimensions, &[1], &refilling).unwrap().storage_bytes(),
            GatherOptions::new().resolved_fill_value(DataType::F32).unwrap().storage_bytes(),
        );
    }

    #[test]
    fn test_gather_interpretation_array_ir() {
        let input = ArrayIrValue::Array(Array::vector(vec![10_i32, 20, 30]).unwrap());
        let indices = ArrayIrValue::Array(Array::matrix(2, 1, vec![2_i32, 0]).unwrap());
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![], vec![0], vec![0]), vec![1]);
        assert_eq!(
            input.gather(&indices, operation.dimensions(), operation.slice_sizes(), operation.options()),
            Ok(ArrayIrValue::Array(Array::vector(vec![30_i32, 10]).unwrap())),
        );
        let dimension = ArrayIrValue::Dimension(DimensionValue::constant(1).unwrap());
        assert_eq!(
            input.gather(&dimension, operation.dimensions(), operation.slice_sizes(), operation.options()),
            Err(TypeError::invalid("expected array type but got dimension type").into()),
        );
        assert_eq!(
            dimension.gather(&indices, operation.dimensions(), operation.slice_sizes(), operation.options()),
            Err(TypeError::invalid("expected array type but got dimension type").into()),
        );

        // Mixed tracers gather through their array projection.
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let staged_input = context.input(input.r#type().into_owned());
        let staged_indices = context.lift(indices).unwrap();
        let projected_input = ValueProjection::<ArrayType>::into_projected(staged_input).unwrap();
        let projected_indices = ValueProjection::<ArrayType>::into_projected(staged_indices).unwrap();
        let output = projected_input
            .gather(&projected_indices, operation.dimensions(), operation.slice_sizes(), operation.options())
            .unwrap();
        assert_eq!(output.r#type().into_owned(), ArrayType::new_static(DataType::I32, [2]));
    }

    #[test]
    fn test_gather_interpretation_extreme_indices() {
        let input = Array::vector(vec![10_i32, 20, 30, 40]).unwrap();
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![], vec![0]), vec![2])
            .with_mode(GatherMode::Clip);
        assert_eq!(
            input.gather(
                &Array::matrix(2, 1, vec![0_u64, u64::MAX]).unwrap(),
                operation.dimensions(),
                operation.slice_sizes(),
                operation.options(),
            ),
            Array::matrix(2, 2, vec![10_i32, 20, 30, 40]),
        );
        assert_eq!(
            input.gather(
                &Array::matrix(2, 1, vec![i64::MIN, i64::MAX]).unwrap(),
                operation.dimensions(),
                operation.slice_sizes(),
                operation.options(),
            ),
            Array::matrix(2, 2, vec![10_i32, 20, 30, 40]),
        );

        // Adding a window offset to an invalid maximal start must not overflow before the query is filled.
        let operation = operation.with_mode(GatherMode::Fill { value: None });
        let signed = input
            .gather(
                &Array::matrix(1, 1, vec![i64::MAX]).unwrap(),
                operation.dimensions(),
                operation.slice_sizes(),
                operation.options(),
            )
            .unwrap();
        let unsigned = input
            .gather(
                &Array::matrix(1, 1, vec![u64::MAX]).unwrap(),
                operation.dimensions(),
                operation.slice_sizes(),
                operation.options(),
            )
            .unwrap();
        assert_eq!(signed, unsigned);
        assert_eq!(signed, Array::matrix(1, 2, vec![i32::MIN, i32::MIN]).unwrap());
    }

    #[test]
    fn test_gather_interpretation_layouts_and_element_types() {
        // Gather rows 2 and 0 of a 3x2 matrix.
        let input = Array::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let indices = Array::matrix(2, 1, vec![2i64, 0]).unwrap();
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
        let gathered = input
            .gather(&indices, operation.dimensions(), operation.slice_sizes(), operation.options())
            .unwrap();
        assert_eq!(gathered.r#type().into_owned(), ArrayType::new_static(DataType::F64, [2, 2]));
        assert_eq!(gathered.to_f64s(), vec![5.0, 6.0, 1.0, 2.0]);

        // In-bounds and clipping modes do not materialize an unused zero fill, so they work for formats that cannot
        // represent zero.
        let input = Array::new(ArrayType::new_static(DataType::F8E8M0FNU, [2]), vec![0x7f, 0x80]).unwrap();
        let indices = Array::matrix(1, 1, vec![1i64]).unwrap();
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![], vec![0], vec![0]), vec![1]);
        assert_eq!(
            input.gather(&indices, operation.dimensions(), operation.slice_sizes(), operation.options()),
            Array::new(ArrayType::new_static(DataType::F8E8M0FNU, [1]), vec![0x80]),
        );

        assert_eq!(
            input.gather(
                &Array::matrix(1, 1, vec![9_i64]).unwrap(),
                operation.dimensions(),
                operation.slice_sizes(),
                &operation.options().clone().with_mode(GatherMode::Clip),
            ),
            Array::new(ArrayType::new_static(DataType::F8E8M0FNU, [1]), vec![0x80]),
        );

        // Gather reads both a reversed input and reversed sub-byte indices through their physical addressing.
        // An out-of-bounds query in fill-or-drop mode writes the default unsigned maximum into the dense result.
        let input_type =
            ArrayType::new_static(DataType::U16, [3]).with_layout(Layout::Strided(StridedLayout::new(vec![-2])));
        let input = Array::from_elements(input_type, &[10u16, 20, 30]).unwrap();
        let indices_type =
            ArrayType::new_static(DataType::I4, [3, 1]).with_layout(Layout::Strided(StridedLayout::new(vec![-1, 1])));
        let indices =
            Array::from_elements(indices_type, &[i4::new(2).unwrap(), i4::new(-1).unwrap(), i4::new(1).unwrap()])
                .unwrap();
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![], vec![0], vec![0]), vec![1])
            .with_mode(GatherMode::Fill { value: None });
        let gathered = input
            .gather(&indices, operation.dimensions(), operation.slice_sizes(), operation.options())
            .unwrap();
        assert_eq!(gathered.elements::<u16>(), Ok(vec![30, u16::MAX, 20]));
        assert_eq!(gathered.storage_bytes(), [30, 0, 255, 255, 20, 0]);
    }

    #[test]
    fn test_gather_partial_evaluation() {
        // Partial evaluation folds fully known gathers and residualizes an unknown data input with known indices.
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
        let input_value = Array::matrix(3, 2, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let indices_value = Array::matrix(2, 1, vec![0_i32, 2]).unwrap();
        let expected = Array::matrix(2, 2, vec![0.0, 1.0, 4.0, 5.0]).unwrap();
        check_operation_partial_evaluation!(
            backend = (Array, ArrayOperation<Array>),
            operation = operation.clone(),
            cases = [
                {
                    inputs = [(@known, input_value.clone()), (@known, indices_value.clone())],
                    outputs = [(@known, expected.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = input_value.r#type().into_owned(), replay = input_value.clone())),
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
        let indices_value = Array::matrix(2, 1, vec![0_i32, 2]).unwrap();

        // Unmapped inputs take the fast path and produce a replicated output.
        check_operation_batching!(
            @exact,
            operation = operation.clone(),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@replicated, Array::matrix(3, 2, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap()),
                    (@replicated, indices_value.clone()),
                ],
                outputs = [(@replicated, Array::matrix(2, 2, vec![0.0, 1.0, 4.0, 5.0]).unwrap())],
            }],
        );

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
                    (@replicated, indices_value.clone()),
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
                        (@mapped(axis = 0), Array::from_elements(
                            ArrayType::new_static(DataType::I32, [2, 2, 1]),
                            &[2_i32, 0, 1, 1],
                        ).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 0), Array::matrix(2, 2, vec![3_f64, 1., 2., 2.]).unwrap())],
                },
                {
                    inputs = [
                        (@mapped(axis = 1), Array::matrix(3, 2, vec![1_f64, 4., 2., 5., 3., 6.]).unwrap()),
                        (@mapped(axis = 0), Array::from_elements(
                            ArrayType::new_static(DataType::I32, [2, 2, 1]),
                            &[2_i32, 0, 1, 1],
                        ).unwrap()),
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
                    (@mapped(axis = 0), Array::new(
                        ArrayType::new_static(DataType::F8E8M0FNU, [0, 3]), Vec::new(),
                    ).unwrap()),
                    (@mapped(axis = 0), Array::from_elements(
                        ArrayType::new_static(DataType::I32, [0, 1, 1]),
                        &[] as &[i32],
                    ).unwrap()),
                ],
                outputs = [(@mapped(axis = 0), Array::new(
                    ArrayType::new_static(DataType::F8E8M0FNU, [0, 1]), Vec::new(),
                ).unwrap())],
            }],
        );

        // A mapped extent that differs from the batching extent is rejected before any lifting, as is a missing input.
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array>::new(), 2);
        let operation = GatherOperation::new(GatherDimensionNumbers::new(Vec::new(), vec![0], vec![0]), vec![1]);
        let misaligned =
            ArrayBatch::new(Array::matrix(3, 3, vec![1_f64, 2., 3., 4., 5., 6., 7., 8., 9.]).unwrap(), 0).unwrap();
        let indices = ArrayBatch::replicated(Array::matrix(1, 1, vec![0_i32]).unwrap());
        assert_eq!(
            operation.batch(&context, &EmptyRegionDriver, &[misaligned, indices.clone()]).unwrap_err(),
            BatchingError::MisalignedBatchAxes {
                message: format!("`{GATHER_OPERATION_NAME}` mapped input extent 3 does not match batching extent 2"),
            },
        );
        assert_eq!(
            operation.batch(&context, &EmptyRegionDriver, &[]).unwrap_err(),
            BatchingError::Program(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        );

        // Ragged input extents must never be replaced with packed storage extents while selecting windows.
        let variable = DimensionVariable::new("length", DimensionBounds::new(1, Some(4)).unwrap());
        let ragged = ArrayBatch::new(Array::matrix(2, 3, vec![1_f64, 2., 3., 4., 5., 6.]).unwrap(), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, Array::vector(vec![1_i32, 3]).unwrap(), variable, vec![0])])
            .unwrap();
        assert_eq!(
            operation.batch(&context, &EmptyRegionDriver, &[ragged, indices]).unwrap_err(),
            BatchingError::UnsupportedOperation {
                message: format!("`{GATHER_OPERATION_NAME}` does not support bounded ragged array inputs"),
            },
        );
    }

    #[test]
    fn test_gather_batching_dynamic_extent() {
        // Jointly mapped inputs pair the mapped axes as batching axes, so the mapped extent never has to be encoded in
        // the static slice sizes. A dynamic extent that may be empty at runtime takes a zero batching window, because
        // a size-one window would exceed the guaranteed minimum extent. These axes do not contribute window dimensions;
        // their extents enter the output through the indices dimensions, so a zero window does not empty the batch.
        let items = DimensionVariable::new("items", DimensionBounds::new(0, Some(9)).unwrap());
        let program = jointly_mapped_dynamic_gather_program(items.clone());
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<items ∈ [0, 9)>, %1:f32[items, 3], %2:i32[items, 1, 1] .
                let %3:f32[items, 1] = gather [
                    dimensions=(offset=[], collapsed_slice=[1], start_index_map=[1], batching=[(0, 0)]),
                    slice_sizes=[0, 1],
                ] %1 %2
                in (%3)
            "}
            .trim_end(),
        );
        let items_type = DimensionType::new(items);
        assert_eq!(
            program.interpret((
                ArrayIrValue::Dimension(DimensionValue::new(items_type.clone(), 3).unwrap()),
                ArrayIrValue::Array(
                    Array::from_elements(
                        ArrayType::new_static(DataType::F32, [3, 3]),
                        &[0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                    )
                    .unwrap(),
                ),
                ArrayIrValue::Array(
                    Array::from_elements(ArrayType::new_static(DataType::I32, [3, 1, 1]), &[2_i32, 0, 1]).unwrap(),
                ),
            )),
            Ok(ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::F32, [3, 1]), &[2.0_f32, 3.0, 7.0]).unwrap(),
            )),
        );
        assert_eq!(
            program.interpret((
                ArrayIrValue::Dimension(DimensionValue::new(items_type, 0).unwrap()),
                ArrayIrValue::Array(
                    Array::from_elements(ArrayType::new_static(DataType::F32, [0, 3]), &[] as &[f32]).unwrap()
                ),
                ArrayIrValue::Array(
                    Array::from_elements(ArrayType::new_static(DataType::I32, [0, 1, 1]), &[] as &[i32]).unwrap(),
                ),
            )),
            Ok(ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::F32, [0, 1]), &[] as &[f32]).unwrap(),
            )),
        );

        // A positive guaranteed minimum extent admits the ordinary size-one batching window.
        let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(9)).unwrap());
        let program = jointly_mapped_dynamic_gather_program(items);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<items ∈ [1, 9)>, %1:f32[items, 3], %2:i32[items, 1, 1] .
                let %3:f32[items, 1] = gather [
                    dimensions=(offset=[], collapsed_slice=[1], start_index_map=[1], batching=[(0, 0)]),
                    slice_sizes=[1, 1],
                ] %1 %2
                in (%3)
            "}
            .trim_end(),
        );

        // Mapped indices alone add a leading batch axis whose extent stays first-class as well.
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(9)).unwrap());
        let extent = trace.input(DimensionType::new(items.clone()).into());
        let input = trace.input(ArrayType::new_static(DataType::F32, [3]).into());
        let input = ValueProjection::<ArrayType>::into_projected(input).unwrap();
        let indices = trace.input(
            ArrayType::new(
                DataType::I32,
                Shape::new(vec![Dimension::Dynamic(items.clone()), Dimension::Static(1), Dimension::Static(1)]),
            )
            .into(),
        );
        let indices = ValueProjection::<ArrayType>::into_projected(indices).unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
            ProjectedContext::new(trace.clone()),
            extent,
        );
        let operation = GatherOperation::new(GatherDimensionNumbers::new(Vec::new(), vec![0], vec![0]), vec![1]);
        let (outputs, _) = operation
            .batch(
                &context,
                &EmptyRegionDriver,
                &[ArrayBatch::replicated(input.clone()), ArrayBatch::new(indices, BatchAxis::new(0)).unwrap()],
            )
            .unwrap()
            .into_parts();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(
            outputs[0].r#type().shape(),
            &Shape::new(vec![Dimension::Dynamic(items.clone()), Dimension::Static(1)]),
        );

        // A mapped input alone needs the mapped extent as a complete static window, so a dynamic extent is rejected.
        let mapped_input = trace.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(items), Dimension::Static(3)])).into(),
        );
        let mapped_input = ValueProjection::<ArrayType>::into_projected(mapped_input).unwrap();
        let shared_indices = trace.constant(ArrayIrValue::Array(Array::matrix(1, 1, vec![0_i32]).unwrap()));
        let shared_indices = ValueProjection::<ArrayType>::into_projected(shared_indices).unwrap();
        assert_eq!(
            operation
                .batch(
                    &context,
                    &EmptyRegionDriver,
                    &[
                        ArrayBatch::new(mapped_input, BatchAxis::new(0)).unwrap(),
                        ArrayBatch::replicated(shared_indices)
                    ],
                )
                .unwrap_err(),
            BatchingError::UnsupportedOperation {
                message: format!(
                    "`{GATHER_OPERATION_NAME}` with only its input mapped requires a statically known mapped extent"
                ),
            },
        );
    }

    #[test]
    fn test_gather_batching_static_extent() {
        // Static mapped extents stage the same paired batching axes with a zero window for an empty batch and a
        // size-one window otherwise. Mapped axes away from position zero are moved to the front first, and a requested
        // output placement gains a leading replicated batch dimension.
        let operation = GatherOperation::new(GatherDimensionNumbers::new(Vec::new(), vec![0], vec![0]), vec![1]);
        let (output_type, empty_program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |(input, indices)| {
                batch(
                    |(input, indices)| {
                        input.gather(&indices, operation.dimensions(), operation.slice_sizes(), operation.options())
                    },
                    (input, indices),
                    (BatchAxis::new(0), BatchAxis::new(0)),
                    BatchAxis::new(0),
                    None,
                )
                .map_err(ProgramError::from)
            },
            (ArrayType::new_static(DataType::F32, [0, 3]), ArrayType::new_static(DataType::I32, [0, 1, 1])),
        )
        .unwrap();
        assert_eq!(output_type, ArrayType::new_static(DataType::F32, [0, 1]));
        assert_eq!(
            empty_program.to_string(),
            indoc! {"
                lambda %0:f32[0, 3], %1:i32[0, 1, 1] .
                let %2:f32[0, 1] = gather [
                    dimensions=(offset=[], collapsed_slice=[1], start_index_map=[1], batching=[(0, 0)]),
                    slice_sizes=[0, 1],
                ] %0 %1
                in (%2)
            "}
            .trim_end(),
        );

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let operation = GatherOperation::new(GatherDimensionNumbers::new(Vec::new(), vec![0], vec![0]), vec![1])
            .with_output_sharding(Sharding::replicated(mesh.clone(), 1));
        let (output_type, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |(input, indices)| {
                batch(
                    |(input, indices)| {
                        input.gather(&indices, operation.dimensions(), operation.slice_sizes(), operation.options())
                    },
                    (input, indices),
                    (BatchAxis::new(1), BatchAxis::new(1)),
                    BatchAxis::new(0),
                    None,
                )
                .map_err(ProgramError::from)
            },
            (ArrayType::new_static(DataType::F32, [3, 2]), ArrayType::new_static(DataType::I32, [1, 2, 1])),
        )
        .unwrap();
        assert_eq!(
            output_type,
            ArrayType::new_static(DataType::F32, [2, 1]).with_sharding(Sharding::replicated(mesh, 2)).unwrap(),
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[3, 2], %1:i32[1, 2, 1] .
                let %2:f32[2, 3] = transpose [permutation=[1, 0]] %0
                    %3:i32[2, 1, 1] = transpose [permutation=[1, 0, 2]] %1
                    %4:f32[2, 1][sharding={mesh<['x'=2:explicit]>, [{}, {}]}] = gather [
                        dimensions=(offset=[], collapsed_slice=[1], start_index_map=[1], batching=[(0, 0)]),
                        slice_sizes=[1, 1],
                        output_sharding={mesh<['x'=2:explicit]>, [{}, {}]},
                    ] %2 %3
                in (%4)
            "}
            .trim_end(),
        );

        // Item 0 is column 0 of the input read at row 2, and item 1 is column 1 read at row 0.
        assert_eq!(
            program.interpret((
                Array::matrix(3, 2, vec![1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap(),
                Array::from_elements(ArrayType::new_static(DataType::I32, [1, 2, 1]), &[2_i32, 0]).unwrap(),
            )),
            Ok(Array::from_elements(output_type, &[3.0_f32, 4.0]).unwrap()),
        );
    }

    #[test]
    fn test_gather_differentiation() {
        // Forward mode selects the input coordinate feeding each gathered output.
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
        let jacobian = differentiate_at(Array::matrix(3, 2, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap())
            .jacobian_forward(|input| {
                let indices = input.dispatch_domain().lift(Array::matrix(2, 1, vec![0_i32, 2]).unwrap())?;
                input.gather(&indices, operation.dimensions(), operation.slice_sizes(), operation.options())
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
        let filling = GatherOperation::new(GatherDimensionNumbers::new(Vec::new(), vec![0], vec![0]), vec![1])
            .with_mode(GatherMode::Fill { value: None });
        let jacobian = differentiate_at(Array::vector(vec![10_f64, 20.]).unwrap())
            .jacobian_forward(|input| {
                let indices = input.dispatch_domain().lift(Array::matrix(3, 1, vec![-1_i32, 1, 5]).unwrap())?;
                input.gather(&indices, filling.dimensions(), filling.slice_sizes(), filling.options())
            })
            .unwrap();
        assert_eq!(jacobian.iter_blocks().next().unwrap().value().elements::<f64>(), Ok(vec![0., 0., 0., 1., 0., 0.]),);
        let explicitly_filling =
            filling.with_mode(GatherMode::Fill { value: Some(Box::new(Array::scalar(99_f64).unwrap())) });
        let jacobian = differentiate_at(Array::vector(vec![10_f64, 20.]).unwrap())
            .jacobian_forward(|input| {
                let indices = input.dispatch_domain().lift(Array::matrix(3, 1, vec![-1_i32, 1, 5]).unwrap())?;
                input.gather(
                    &indices,
                    explicitly_filling.dimensions(),
                    explicitly_filling.slice_sizes(),
                    explicitly_filling.options(),
                )
            })
            .unwrap();
        assert_eq!(jacobian.iter_blocks().next().unwrap().value().elements::<f64>(), Ok(vec![0., 0., 0., 1., 0., 0.]),);
    }

    #[test]
    fn test_gather_differentiation_zero_tangent() {
        // The shared all-zero fast path lives in the differentiation context's bind, so a direct rule call reaches
        // the body with a structural-zero input tangent, which stays a typed zero of the output type.
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
        let context = DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new());
        let input = Array::matrix(3, 2, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let indices = Array::matrix(2, 1, vec![0_i32, 2]).unwrap();
        let outputs = operation
            .jvp(
                &context,
                &EmptyRegionDriver,
                &[
                    DifferentiationDual::new_with_zero_tangent(input).unwrap(),
                    DifferentiationDual::new_with_zero_tangent(indices).unwrap(),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(*outputs[0].primal(), Array::matrix(2, 2, vec![0.0, 1.0, 4.0, 5.0]).unwrap());
        assert!(matches!(
            outputs[0].tangent(),
            MaybeZero::Zero(tangent_type) if tangent_type == &ArrayType::new_static(DataType::F64, [2, 2]),
        ));
        assert_eq!(
            operation.jvp(&context, &EmptyRegionDriver, &[]).unwrap_err(),
            DifferentiationError::Program(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        );
    }

    #[test]
    fn test_gather_differentiation_array_ir() {
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
        // extent, and the tangent program is one residual-carrying linear call whose transpose region rebuilds the
        // dynamic zero from the retained extent.
        assert_eq!(linearization.residual_count(), 2);
        assert_eq!(
            linearization.tangent().input_types(),
            &[
                input_type.tangent().unwrap().into(),
                indices_type.clone().into(),
                ArrayIrType::Dimension(DimensionType::new(extent)),
            ],
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[extent], %1:i32[3, 1], %2:dimension<extent ∈ [1, 6)> .
                let %3:f64[3] = linear_call [residual_count=2] %1 %2 %0 [
                    forward={
                        lambda %0:i32[3, 1], %1:dimension<extent ∈ [1, 6)>, %2:f64[extent] .
                        let %3:f64[3] = gather [
                            dimensions=(offset=[], collapsed_slice=[0], start_index_map=[0], batching=[]),
                            slice_sizes=[1],
                        ] %2 %0
                        in (%3)
                    },
                    transpose={
                        lambda %0:i32[3, 1], %1:dimension<extent ∈ [1, 6)>, %2:f64[3] .
                        let %3:f64[extent] = zero [type=f64[extent]] %1
                            %4:f64[extent] = scatter [
                                kind=add,
                                dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                                    operand_batching=[], scatter_indices_batching=[]),
                            ] %3 %0 %2
                        in (%4)
                    },
                ]
                in (%3)
            "}
            .trim_end(),
        );
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
        // drops the same out-of-bounds coordinates, both for the default NaN fill and for an explicit nonzero fill.
        let filling = GatherOperation::new(GatherDimensionNumbers::new(Vec::new(), vec![0], vec![0]), vec![1])
            .with_mode(GatherMode::Fill { value: None });
        let linearization = dynamic_fill_gather_program(filling.clone()).linearize().unwrap();
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Array(Array::vector(vec![10_f64, 20.]).unwrap())])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![ArrayIrValue::Array(Array::vector(vec![2_f64, 3.]).unwrap())];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0_f64, 3., 0.]).unwrap())]),
        );
        let mut cotangent_inputs = vec![ArrayIrValue::Array(Array::vector(vec![1_f64, 1., 1.]).unwrap())];
        cotangent_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(cotangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0_f64, 1.]).unwrap())]),
        );
        let explicitly_filling =
            filling.with_mode(GatherMode::Fill { value: Some(Box::new(Array::scalar(99_f64).unwrap())) });
        let linearization = dynamic_fill_gather_program(explicitly_filling).linearize().unwrap();
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Array(Array::vector(vec![10_f64, 20.]).unwrap())])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::vector(vec![99_f64, 20., 99.]).unwrap()));
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![ArrayIrValue::Array(Array::vector(vec![2_f64, 3.]).unwrap())];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0_f64, 3., 0.]).unwrap())]),
        );
        let mut cotangent_inputs = vec![ArrayIrValue::Array(Array::vector(vec![1_f64, 1., 1.]).unwrap())];
        cotangent_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(cotangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0_f64, 1.]).unwrap())]),
        );
    }

    #[test]
    fn test_gather_differentiation_array_ir_zero_tangent() {
        // The member rule stages only the primal gather when the dynamically shaped input carries a structural-zero
        // tangent, and it validates its arity before touching any input.
        let operation = GatherOperation::new(GatherDimensionNumbers::new(Vec::new(), vec![0], vec![0]), vec![1]);
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let context = DifferentiationContext::fused(trace.clone());
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(6)).unwrap());
        let input_type =
            ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)])));
        let indices_type = ArrayIrType::Array(ArrayType::new_static(DataType::I32, [3, 1]));
        let outputs = operation
            .jvp_in_parent(
                &context,
                &EmptyRegionDriver,
                &[
                    DifferentiationDual::new_with_zero_tangent(trace.input(input_type)).unwrap(),
                    DifferentiationDual::new_with_zero_tangent(trace.input(indices_type)).unwrap(),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        let output_type = ArrayIrType::Array(ArrayType::new_static(DataType::F64, [3]));
        assert_eq!(outputs[0].primal().r#type().as_ref(), &output_type);
        assert!(matches!(outputs[0].tangent(), MaybeZero::Zero(tangent_type) if tangent_type == &output_type));
        let builder = trace.builder().borrow();
        assert_eq!(builder.instructions().len(), 1);
        assert_eq!(builder.instructions()[0].operation().name(), GATHER_OPERATION_NAME);
        drop(builder);
        assert_eq!(
            operation.jvp_in_parent(&context, &EmptyRegionDriver, &[]).unwrap_err(),
            DifferentiationError::Program(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        );
    }

    #[test]
    fn test_gather_transposition() {
        // Take rows 0 and 2 of a [3, 2] input: the input is linear and the [2, 1] index array is the known
        // input. The gathered output and its cotangent have shape [2, 2].
        let dimensions = GatherDimensionNumbers::new(vec![1], vec![0], vec![0]);
        let operation = GatherOperation::new(dimensions, vec![1, 2]);
        let input = Array::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let indices = Array::matrix(2, 1, vec![0_i32, 2]).unwrap();
        let cotangent = Array::matrix(2, 2, vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        check_operation_transposition!(
            @exact,
            backend = (Array, ArrayOperation<Array>),
            operation = operation,
            cases = [{
                inputs = [
                    (@linear(type = input.r#type().into_owned())),
                    (@known, indices),
                ],
                output_cotangents = [cotangent],
                input_cotangents = [Array::matrix(3, 2, vec![10.0, 20.0, 0.0, 0.0, 30.0, 40.0]).unwrap()],
            }],
        );

        // The transpose explicitly restores the input's distribution even when the forward gather requested a
        // different output placement. Scatter's zero base also preserves input strides and host memory. A static input
        // takes the homogeneous rule, while a dynamic input takes the residual-carrying member rule.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let extent = DimensionVariable::new("extent", DimensionBounds::new(4, Some(5)).unwrap());
        let placed_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(4)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![16])))
            .with_memory(Memory::Host { pinned: true });
        let dynamic_placed_type = placed_type.clone().with_shape(Shape::new(vec![Dimension::Dynamic(extent)]));
        let cotangent = ArrayIrValue::Array(
            Array::from_elements(
                ArrayType::new_static(DataType::F64, [2])
                    .with_memory(Memory::Host { pinned: true })
                    .with_sharding(Sharding::replicated(mesh.clone(), 1))
                    .unwrap(),
                &[10_f64, 20.],
            )
            .unwrap(),
        );

        let pullback = placed_take_program(placed_type.clone(), &mesh).linearize().unwrap().pullback().unwrap();
        assert_eq!(pullback.output_types(), vec![ArrayIrType::Array(placed_type.cotangent().unwrap())]);
        assert_eq!(
            pullback.interpret(vec![cotangent.clone()]),
            Ok(vec![ArrayIrValue::Array(
                Array::from_elements(placed_type.cotangent().unwrap(), &[0_f64, 10., 0., 20.]).unwrap(),
            )]),
        );
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[2][sharding={mesh<['x'=2:explicit]>, [{}]}]@Host[Pinned] .
                let %1:i32[2, 1]@Host[Pinned] = const [[1], [3]]
                    %2:f64[4][layout=strided{16}]@Host[Pinned] = zero [type=f64[4][layout=strided{16}]@Host[Pinned]]
                    %3:f64[4][layout=strided{16}][sharding={mesh<['x'=2:explicit]>, [{}]}]@Host[Pinned] = scatter [
                        kind=add,
                        dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                            operand_batching=[], scatter_indices_batching=[]),
                    ] %2 %1 %0
                    %4:f64[4][layout=strided{16}]@Host[Pinned] = broadcast \
                        [output_type=f64[4][layout=strided{16}]@Host[Pinned], output_axes=[0]] %3
                in (%4)
            "}
            .trim_end(),
        );

        let sharded_type = placed_type.with_sharding(sharded.clone()).unwrap();
        let pullback = placed_take_program(sharded_type.clone(), &mesh).linearize().unwrap().pullback().unwrap();
        assert_eq!(pullback.output_types(), vec![ArrayIrType::Array(sharded_type.cotangent().unwrap())]);
        assert_eq!(
            pullback.interpret(vec![cotangent.clone()]),
            Ok(vec![ArrayIrValue::Array(
                Array::from_elements(sharded_type.cotangent().unwrap(), &[0_f64, 10., 0., 20.]).unwrap(),
            )]),
        );
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[2][sharding={mesh<['x'=2:explicit]>, [{}]}]@Host[Pinned] .
                let %1:i32[2, 1]@Host[Pinned] = const [[1], [3]]
                    %2:f64[4][layout=strided{16}][sharding={mesh<['x'=2:explicit]>, [{'x'}]}]@Host[Pinned] = zero [
                        type=f64[4][layout=strided{16}][sharding={mesh<['x'=2:explicit]>, [{'x'}]}]@Host[Pinned],
                    ]
                    %3:f64[4][layout=strided{16}][sharding={mesh<['x'=2:explicit]>, [{'x'}]}]@Host[Pinned] = scatter [
                        kind=add,
                        dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                            operand_batching=[], scatter_indices_batching=[]),
                        output_sharding={mesh<['x'=2:explicit]>, [{'x'}]},
                    ] %2 %1 %0
                in (%3)
            "}
            .trim_end(),
        );

        let pullback = placed_take_program(dynamic_placed_type.clone(), &mesh).linearize().unwrap().pullback().unwrap();
        assert_eq!(pullback.output_types(), vec![ArrayIrType::Array(dynamic_placed_type.cotangent().unwrap())]);
        let ArrayIrType::Dimension(extent_type) = &pullback.input_types()[1] else {
            panic!("expected a retained input extent");
        };
        let extent = ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 4).unwrap());
        assert_eq!(
            pullback.interpret(vec![cotangent.clone(), extent]),
            Ok(vec![ArrayIrValue::Array(
                Array::from_elements(
                    dynamic_placed_type.cotangent().unwrap().with_shape(Shape::new(vec![Dimension::Static(4)])),
                    &[0_f64, 10., 0., 20.]
                )
                .unwrap(),
            )]),
        );
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[2][sharding={mesh<['x'=2:explicit]>, [{}]}]@Host[Pinned], %1:dimension<4> .
                let %2:i32[2, 1]@Host[Pinned] = const [[1], [3]]
                    %3:f64[extent][layout=strided{16}]@Host[Pinned] = linear_call [residual_count=2] %2 %1 %0 [
                        forward={
                            lambda %0:i32[2, 1]@Host[Pinned], %1:dimension<4>, \
                                %2:f64[2][sharding={mesh<['x'=2:explicit]>, [{}]}]@Host[Pinned] .
                            let %3:f64[extent][layout=strided{16}]@Host[Pinned] = zero \
                                [type=f64[extent][layout=strided{16}]@Host[Pinned]] %1
                                %4:f64[extent][layout=strided{16}][sharding={mesh<['x'=2:explicit]>, \
                                    [{}]}]@Host[Pinned] = scatter [
                                    kind=add,
                                    dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                                        operand_batching=[], scatter_indices_batching=[]),
                                ] %3 %0 %2
                                %5:f64[extent][layout=strided{16}]@Host[Pinned] = broadcast \
                                    [output_type=f64[extent][layout=strided{16}]@Host[Pinned], output_axes=[0]] %4
                            in (%5)
                        },
                        transpose={
                            lambda %0:i32[2, 1]@Host[Pinned], %1:dimension<4>, \
                                %2:f64[extent][layout=strided{16}]@Host[Pinned] .
                            let %3:f64[2][sharding={mesh<['x'=2:explicit]>, [{}]}]@Host[Pinned] = gather [
                                dimensions=(offset=[], collapsed_slice=[0], start_index_map=[0], batching=[]),
                                slice_sizes=[1],
                                output_sharding={mesh<['x'=2:explicit]>, [{}]},
                            ] %2 %0
                            in (%3)
                        },
                    ]
                in (%3)
            "}
            .trim_end(),
        );

        let dynamic_sharded_type = dynamic_placed_type.with_sharding(sharded).unwrap();
        let pullback =
            placed_take_program(dynamic_sharded_type.clone(), &mesh).linearize().unwrap().pullback().unwrap();
        assert_eq!(pullback.output_types(), vec![ArrayIrType::Array(dynamic_sharded_type.cotangent().unwrap())]);
        let ArrayIrType::Dimension(extent_type) = &pullback.input_types()[1] else {
            panic!("expected a retained input extent");
        };
        let extent = ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 4).unwrap());
        assert_eq!(
            pullback.interpret(vec![cotangent.clone(), extent]),
            Ok(vec![ArrayIrValue::Array(
                Array::from_elements(
                    dynamic_sharded_type.cotangent().unwrap().with_shape(Shape::new(vec![Dimension::Static(4)])),
                    &[0_f64, 10., 0., 20.]
                )
                .unwrap(),
            )]),
        );
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[2][sharding={mesh<['x'=2:explicit]>, [{}]}]@Host[Pinned], %1:dimension<4> .
                let %2:i32[2, 1]@Host[Pinned] = const [[1], [3]]
                    %3:f64[extent][layout=strided{16}][sharding={mesh<['x'=2:explicit]>, [{'x'}]}]@Host[Pinned] = \
                        linear_call [residual_count=2] %2 %1 %0 [
                        forward={
                            lambda %0:i32[2, 1]@Host[Pinned], %1:dimension<4>, \
                                %2:f64[2][sharding={mesh<['x'=2:explicit]>, [{}]}]@Host[Pinned] .
                            let %3:f64[extent][layout=strided{16}][sharding={mesh<['x'=2:explicit]>, \
                                [{'x'}]}]@Host[Pinned] = zero [
                                type=f64[extent][layout=strided{16}][sharding={mesh<['x'=2:explicit]>, \
                                    [{'x'}]}]@Host[Pinned],
                            ] %1
                                %4:f64[extent][layout=strided{16}][sharding={mesh<['x'=2:explicit]>, \
                                    [{'x'}]}]@Host[Pinned] = scatter [
                                    kind=add,
                                    dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                                        operand_batching=[], scatter_indices_batching=[]),
                                    output_sharding={mesh<['x'=2:explicit]>, [{'x'}]},
                                ] %3 %0 %2
                            in (%4)
                        },
                        transpose={
                            lambda %0:i32[2, 1]@Host[Pinned], %1:dimension<4>, \
                                %2:f64[extent][layout=strided{16}][sharding={mesh<['x'=2:explicit]>, \
                                [{'x'}]}]@Host[Pinned] .
                            let %3:f64[2][sharding={mesh<['x'=2:explicit]>, [{}]}]@Host[Pinned] = gather [
                                dimensions=(offset=[], collapsed_slice=[0], start_index_map=[0], batching=[]),
                                slice_sizes=[1],
                                output_sharding={mesh<['x'=2:explicit]>, [{}]},
                            ] %2 %0
                            in (%3)
                        },
                    ]
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_gather_transposition_dynamic_input_shapes() {
        // The homogeneous gather transpose scatters into a zero of the input's cotangent type, and the homogeneous
        // `ArrayType` family has no constructor that can supply a runtime extent for one. A dynamically shaped input is
        // therefore part of the rule's rejected contract rather than an accident of zero construction.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(4, Some(8)).unwrap());
        let dynamic_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows), Dimension::Static(2)]));

        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(dynamic_type);
        let indices = builder.add_constant(Array::matrix(2, 1, vec![0_i32, 2]).unwrap());
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
        let output = builder.add_instruction(operation, Vec::new(), vec![input, indices], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(
            program.transpose_with_respect_to(&[0], &[]).unwrap_err(),
            TypeError::invalid(format!(
                "`{GATHER_OPERATION_NAME}` transpose requires a statically shaped input but got `f32[rows, 2]`"
            ))
            .into(),
        );
    }

    #[test]
    fn test_gather_transposition_dynamic_indices() {
        // Only the input must be statically shaped: the query axes come from the indices, so a dynamic query extent
        // reaches the mixed member rule and delegates to the homogeneous rule when the input is static. Its scatter
        // adjoint accepts dynamic query dimensions as long as the trailing index vector extent is static. Repeated
        // indices accumulate into the same input coordinate.
        let queries = DimensionVariable::new("queries", DimensionBounds::new(0, Some(5)).unwrap());
        let indices_type =
            ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(queries), Dimension::Static(1)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F64, [4]).into());
        let indices = builder.add_input(indices_type.into());
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
        let pullback = linearization.pullback().unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[queries], %1:i32[queries, 1] .
                let %2:f64[4] = zero [type=f64[4]]
                    %3:f64[4] = scatter [
                        kind=add,
                        dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], \
                            operand_batching=[], scatter_indices_batching=[]),
                    ] %2 %1 %0
                in (%3)
            "}
            .trim_end(),
        );

        let input = ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0]).unwrap());
        let empty_indices = ArrayIrValue::Array(
            Array::from_elements(ArrayType::new_static(DataType::I32, [0, 1]), &[] as &[i32]).unwrap(),
        );
        let mut primal_outputs = linearization.primal().interpret(vec![input.clone(), empty_indices]).unwrap();
        assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::vector(Vec::<f64>::new()).unwrap()));
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::vector(Vec::<f64>::new()).unwrap())];
        pullback_inputs.extend(primal_outputs.split_off(1));
        assert_eq!(
            pullback.interpret(pullback_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0.0_f64; 4]).unwrap())]),
        );

        let repeated_indices = ArrayIrValue::Array(Array::matrix(3, 1, vec![1_i32, 1, 3]).unwrap());
        let mut primal_outputs = linearization.primal().interpret(vec![input, repeated_indices]).unwrap();
        assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::vector(vec![20.0_f64, 20.0, 40.0]).unwrap()));
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::vector(vec![2.0_f64, 3.0, 5.0]).unwrap())];
        pullback_inputs.extend(primal_outputs.split_off(1));
        assert_eq!(
            pullback.interpret(pullback_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0.0_f64, 5.0, 0.0, 5.0]).unwrap())]),
        );
    }

    #[test]
    fn test_gather_transposition_zero_cotangent() {
        // A structural-zero output cotangent contributes nothing: the rule returns before staging anything and leaves
        // the input accumulator at its structural-zero default. The same holds when no cotangent is needed.
        let operation =
            GatherOperation::<Array>::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
        let input_type = ArrayType::new_static(DataType::F64, [3, 2]);
        let output_type = ArrayType::new_static(DataType::F64, [2, 2]);
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let mut transpose = TranspositionContext::new(context.clone());
        let indices = context.lift(Array::matrix(2, 1, vec![0_i32, 2]).unwrap()).unwrap();
        let inputs = [PartialValue::Unknown(input_type.clone()), PartialValue::Known(indices)];
        let accumulators = transpose.cotangent_accumulators(&inputs, &[]).unwrap();
        let zero_outputs = [MaybeZero::Zero(output_type.cotangent().unwrap())];
        operation
            .transpose(&mut transpose, &EmptyRegionDriver, &inputs, &zero_outputs, &accumulators)
            .unwrap();
        let cotangents = transpose.take_cotangents(&accumulators).unwrap();
        assert_eq!(cotangents.len(), 2);
        assert!(cotangents[0].is_zero());
        assert_eq!(cotangents[0].r#type().as_ref(), &input_type.cotangent().unwrap());
        assert!(context.builder().borrow().instructions().is_empty());

        let unneeded = transpose.cotangent_accumulators(&inputs, &[false, false]).unwrap();
        let outputs = [MaybeZero::Value(context.input(output_type.cotangent().unwrap()))];
        operation.transpose(&mut transpose, &EmptyRegionDriver, &inputs, &outputs, &unneeded).unwrap();
        assert!(context.builder().borrow().instructions().is_empty());

        // Arity is validated before any cotangent is inspected.
        assert_eq!(
            operation
                .transpose(&mut transpose, &EmptyRegionDriver, &inputs[..1], &outputs, &accumulators)
                .unwrap_err(),
            DifferentiationError::Program(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        );
        assert_eq!(
            operation.transpose(&mut transpose, &EmptyRegionDriver, &inputs, &[], &accumulators).unwrap_err(),
            DifferentiationError::Program(ProgramError::InvalidOutputCount { expected: 1, actual: 0 }),
        );
        assert_eq!(
            operation
                .transpose(&mut transpose, &EmptyRegionDriver, &inputs, &outputs, &accumulators[..1])
                .unwrap_err(),
            DifferentiationError::InvalidAccumulatorCount { expected: 2, actual: 1 },
        );
    }

    #[test]
    fn test_gather_gather_axis() {
        let input = Array::matrix(2, 3, vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(
            input.gather_axis(&Array::vector(vec![2_i32, 0]).unwrap(), -1, GatherMode::Clip),
            Array::matrix(2, 2, vec![3_i32, 1, 6, 4]),
        );
        assert_eq!(
            input.gather_axis(&Array::scalar(1_i32).unwrap(), 0, GatherMode::Clip),
            Array::vector(vec![4_i32, 5, 6]),
        );
        assert_eq!(
            input.gather_axis(&Array::matrix(1, 2, vec![-1_i32, 9]).unwrap(), 1, GatherMode::Clip),
            Array::from_elements(ArrayType::new_static(DataType::I32, [2, 1, 2]), &[1_i32, 3, 4, 6]),
        );
        assert_eq!(
            input.gather_axis(&Array::vector(vec![-1_i32, 1]).unwrap(), 0, GatherMode::Fill { value: None }),
            Array::matrix(2, 3, vec![i32::MIN, i32::MIN, i32::MIN, 4, 5, 6]),
        );
        assert_eq!(
            input.gather_axis(&Array::vector(Vec::<i32>::new()).unwrap(), 0, GatherMode::Clip),
            Array::matrix(0, 3, Vec::<i32>::new()),
        );
        assert_eq!(
            input.gather_axis(&Array::scalar(0_i32).unwrap(), 2, GatherMode::Clip),
            Err(TypeError::invalid("axis 2 is out of bounds for rank 2").into()),
        );

        // Selecting one element does not require the selected axis's runtime extent as a window parameter.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(6)).unwrap());
        let input_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let (output_type, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |(input, indices)| input.gather_axis(&indices, 0, GatherMode::Clip),
            (input_type, ArrayType::new_static(DataType::I32, [2])),
        )
        .unwrap();
        assert_eq!(output_type, ArrayType::new_static(DataType::I32, [2]));
        assert_eq!(
            program.interpret((
                Array::from_elements(ArrayType::new_static(DataType::I32, [3]), &[10_i32, 20, 30]).unwrap(),
                Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[2_i32, 0]).unwrap(),
            )),
            Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[30_i32, 10]),
        );

        // A complete window along an unselected axis still needs a host-known size.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(6)).unwrap());
        let result = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |(input, indices)| input.gather_axis(&indices, 1, GatherMode::Clip),
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
                    |(input, queries)| input.dynamic_gather_axis(&queries, 0, GatherMode::Clip),
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
            |(input, indices)| input.dynamic_gather_axis(&indices, 0, GatherMode::Clip),
            (empty.r#type().into_owned(), indices.r#type().into_owned()),
        )
        .unwrap();
        assert_eq!(empty.dynamic_gather_axis(&indices, 0, GatherMode::Clip).unwrap(), empty);
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
        let eager = input.dynamic_gather_axis(&indices, 0, GatherMode::Clip).unwrap();
        let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(input, indices)| input.dynamic_gather_axis(&indices, 0, GatherMode::Clip),
            (ArrayIrType::Array(input_type), ArrayIrType::Array(indices_type)),
        )
        .unwrap();
        assert_eq!(program.interpret((input, indices)).unwrap(), eager);
        assert_eq!(
            eager.r#type().into_owned(),
            ArrayIrType::Array(ArrayType::new_static(DataType::F64, [0]).with_sharding(sharding).unwrap()),
        );

        // Query dimensions replace the selected axis, while paired batching preserves both nonleading and empty
        // untouched dimensions. The same symbolic program is replayed for several concrete input and query extents.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(0, Some(6)).unwrap());
        let queries = DimensionVariable::new("queries", DimensionBounds::new(0, Some(4)).unwrap());
        let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(input, indices)| input.dynamic_gather_axis(&indices, 1, GatherMode::Clip),
            (
                ArrayIrType::Array(ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![Dimension::Dynamic(rows), Dimension::Static(4)]),
                )),
                ArrayIrType::Array(ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(queries)]))),
            ),
        )
        .unwrap();
        let four_rows = ArrayIrValue::Array(
            Array::from_elements(
                ArrayType::new_static(DataType::F64, [4, 4]),
                &(0..16).map(|value| value as f64).collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        let five_rows = ArrayIrValue::Array(
            Array::from_elements(
                ArrayType::new_static(DataType::F64, [5, 4]),
                &(0..20).map(|value| value as f64).collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        let no_rows = ArrayIrValue::Array(
            Array::from_elements(ArrayType::new_static(DataType::F64, [0, 4]), &[] as &[f64]).unwrap(),
        );
        let two_queries = ArrayIrValue::Array(Array::vector(vec![2_i32, 0]).unwrap());
        let three_queries = ArrayIrValue::Array(Array::vector(vec![2_i32, 0, 3]).unwrap());
        let no_queries = ArrayIrValue::Array(Array::vector(Vec::<i32>::new()).unwrap());
        assert_eq!(
            program.interpret((no_rows, two_queries.clone())),
            Ok(ArrayIrValue::Array(Array::matrix(0, 2, Vec::<f64>::new()).unwrap())),
        );
        assert_eq!(
            program.interpret((four_rows.clone(), two_queries)),
            Ok(ArrayIrValue::Array(Array::matrix(4, 2, vec![2_f64, 0., 6., 4., 10., 8., 14., 12.]).unwrap())),
        );
        assert_eq!(
            program.interpret((five_rows, three_queries)),
            Ok(ArrayIrValue::Array(
                Array::matrix(5, 3, vec![2_f64, 0., 3., 6., 4., 7., 10., 8., 11., 14., 12., 15., 18., 16., 19.],)
                    .unwrap(),
            )),
        );
        assert_eq!(
            program.interpret((four_rows, no_queries)),
            Ok(ArrayIrValue::Array(Array::matrix(4, 0, Vec::<f64>::new()).unwrap())),
        );
    }

    #[test]
    fn test_validate_unique_in_range() {
        // Sorted validation rejects non-increasing entries before checking their range.
        assert_eq!(validate_unique_in_range("gather", "axes", &[], 0, true), Ok(()));
        assert_eq!(validate_unique_in_range("gather", "axes", &[0, 2], 3, true), Ok(()));
        assert_eq!(
            validate_unique_in_range("gather", "axes", &[2, 0], 3, true),
            Err(TypeError::invalid("`gather` `axes` must be sorted and unique but got [2, 0]")),
        );
        assert_eq!(
            validate_unique_in_range("gather", "axes", &[0, 0], 3, true),
            Err(TypeError::invalid("`gather` `axes` must be sorted and unique but got [0, 0]")),
        );
        assert_eq!(
            validate_unique_in_range("gather", "axes", &[3], 3, true),
            Err(TypeError::invalid("`gather` `axes` entry 3 is out of range for bound 3")),
        );

        assert_eq!(
            validate_unique_in_range("gather", "axes", &[3, 0], 3, true),
            Err(TypeError::invalid("`gather` `axes` must be sorted and unique but got [3, 0]")),
        );

        // Unrestricted ordering still requires unique entries and checks each entry's range first.
        assert_eq!(validate_unique_in_range("gather", "axes", &[], 0, false), Ok(()));
        assert_eq!(
            validate_unique_in_range("gather", "axes", &[3, 0], 3, false),
            Err(TypeError::invalid("`gather` `axes` entry 3 is out of range for bound 3")),
        );
        assert_eq!(validate_unique_in_range("gather", "axes", &[2, 0], 3, false), Ok(()));
        assert_eq!(
            validate_unique_in_range("gather", "axes", &[0, 0], 3, false),
            Err(TypeError::invalid("`gather` `axes` must be unique but got [0, 0]")),
        );
        assert_eq!(
            validate_unique_in_range("gather", "axes", &[3], 3, false),
            Err(TypeError::invalid("`gather` `axes` entry 3 is out of range for bound 3")),
        );
    }
}
