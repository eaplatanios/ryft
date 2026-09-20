use std::marker::PhantomData;
use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

use crate::arrays::{Array, ArrayIrType, ArrayType, Broadcastable, DataType, Dimension, Shape};
use crate::contexts::{Context, Domain};
use crate::macros::check_count;
use crate::operations::compare::Compare;
use crate::operations::constants::constant::{ConstantOperation, DimensionConstant};
use crate::operations::control_flow::select::Select;
use crate::operations::dimensions::dimension_size::DimensionSize;
use crate::operations::dimensions::dimension_to_scalar::DimensionToScalar;
use crate::operations::manipulation::broadcasting::{Broadcast, DynamicBroadcast};
use crate::operations::manipulation::concatenation::Concatenate;
use crate::operations::manipulation::conversions::ConvertElementType;
use crate::operations::manipulation::gathering::{
    DynamicGather, Gather, GatherDimensionNumbers, GatherMode, GatherOptions,
};
use crate::operations::manipulation::memory::TransferToMemory;
use crate::operations::manipulation::reshaping::{DynamicReshape, Reshape};
use crate::operations::manipulation::reversing::Reverse;
use crate::operations::manipulation::scattering::{
    DynamicScatter, Scatter, ScatterDimensionNumbers, ScatterMode, ScatterOptions, ScatterReductionKind,
};
use crate::operations::manipulation::slicing::Slice;
use crate::operations::math::add::Add;
use crate::programs::{ProgramError, Type, TypeError, Typed, Value, ValueProjection};

/// A host integer that can be represented exactly as an indexing coordinate, slice endpoint, or stride. The [`index!`]
/// macro and range conversions into [`IndexSelector`] use this trait to accept integer types such as `usize` without
/// losing precision or permitting floating-point truncation.
///
/// Implementations must preserve the integer's value when converting to `i128`. All primitive integer types except
/// `u128` (because the full range of `u128` cannot be represented by `i128`) implement this trait.
pub trait IndexInteger {
    /// Converts this integer to the signed representation used by indexing descriptors without losing precision.
    fn to_index_integer(self) -> i128;
}

macro_rules! impl_index_integer {
    ($integer:ty) => {
        impl IndexInteger for $integer {
            fn to_index_integer(self) -> i128 {
                self as i128
            }
        }
    };
}

impl_index_integer!(i8);
impl_index_integer!(i16);
impl_index_integer!(i32);
impl_index_integer!(i64);
impl_index_integer!(i128);
impl_index_integer!(isize);
impl_index_integer!(u8);
impl_index_integer!(u16);
impl_index_integer!(u32);
impl_index_integer!(u64);
impl_index_integer!(usize);

/// A signed, half-open slice of one array axis. Omitted endpoints select the corresponding end of the axis. Negative
/// endpoints count backward from its extent. A negative step traverses the selected positions in reverse. In
/// particular, an omitted stop differs from an explicit `-1` when the step is negative as the former includes
/// index `0`, whereas the latter refers to the last element. Slice bounds are clipped to the axis independently of
/// the bounds mode used for integer indices. A zero step is rejected when the slice is normalized.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct IndexSlice {
    /// Refer to the documentation of [`start`](Self::start) for more information.
    start: Option<i128>,

    /// Refer to the documentation of [`stop`](Self::stop) for more information.
    stop: Option<i128>,

    /// Refer to the documentation of [`step`](Self::step) for more information.
    step: i128,
}

// TODO(eaplatanios): Review from here onwards.

impl IndexSlice {
    /// Creates a signed slice with the provided optional endpoints and step. Endpoints are exclusive at the stop
    /// and are clipped to the axis when normalized. This constructor preserves omitted endpoints so that reverse
    /// slicing can distinguish an omitted stop from an explicit negative index, and it accepts any step so that the
    /// [`index!`](crate::index) macro stays infallible; a zero step is rejected when the slice is normalized against
    /// an axis.
    pub fn new(start: Option<i128>, stop: Option<i128>, step: i128) -> Self {
        Self { start, stop, step }
    }

    /// Returns the inclusive start index, or [`None`] to start at the first position in the traversal direction.
    pub fn start(&self) -> Option<i128> {
        self.start
    }

    /// Returns the exclusive stop index, or [`None`] to continue through the end in the traversal direction.
    pub fn stop(&self) -> Option<i128> {
        self.stop
    }

    /// Returns the signed distance between selected positions. Positive values traverse forward, negative values
    /// traverse backward, and zero is invalid. Omitting the step in indexing syntax uses `1`.
    pub fn step(&self) -> i128 {
        self.step
    }

    /// Resolves signed endpoints against an axis extent, producing a positive-stride slice. For a negative step,
    /// the coordinates refer to the reversed input axis. Endpoint clipping and unsigned step magnitudes avoid
    /// overflow even for `i128::MIN`, and empty intervals remain valid empty slices.
    fn normalize(&self, extent: usize) -> Result<NormalizedIndexSlice, ProgramError> {
        if self.step == 0 {
            return Err(TypeError::invalid("index slice step must not be zero").into());
        }
        let reversed = self.step < 0;
        let extent = extent as i128;
        let lower = if reversed { -1 } else { 0 };
        let upper = if reversed { extent - 1 } else { extent };
        // Only explicit negative endpoints count backward from the extent. The default reverse stop is the
        // sentinel before index zero and must not undergo that translation.
        let start = self.start.map_or(if reversed { upper } else { lower }, |start| {
            if start < 0 { start + extent } else { start }.clamp(lower, upper)
        });
        let stop = self.stop.map_or(if reversed { lower } else { upper }, |stop| {
            if stop < 0 { stop + extent } else { stop }.clamp(lower, upper)
        });
        let (start, limit) = if reversed {
            ((extent - 1 - start) as usize, (extent - 1 - stop) as usize)
        } else {
            (start as usize, stop as usize)
        };
        let limit = limit.max(start);
        // A stride larger than the axis extent still selects at most one element. Cap it at that extent (or one
        // for an empty axis), preserving the positions without sending an enormous unsigned stride to backends
        // whose slice configuration uses signed integers.
        let stride = self.step.unsigned_abs().min((extent as usize).max(1) as u128) as usize;
        let length = (limit - start).div_ceil(stride);
        Ok(NormalizedIndexSlice { start, limit, stride, length, reversed })
    }
}

/// A basic selector that consumes one axis for an integer or slice, inserts an axis for [`Self::NewAxis`], or
/// expands over the axes not explicitly selected for [`Self::Ellipsis`]. Integer indices remove the selected axis.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BasicIndex {
    /// A signed scalar index. Negative indices count backward from the axis extent.
    Integer(i128),

    /// A signed, potentially strided slice that preserves the selected axis.
    Slice(IndexSlice),

    /// Inserts a new axis of extent one without consuming an input axis.
    NewAxis,

    /// Expands to full slices over the otherwise unspecified input axes.
    Ellipsis,
}

/// Positive-stride coordinates for a normalized basic slice, optionally applied after reversing its input axis.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct NormalizedIndexSlice {
    /// Inclusive coordinate on the input axis, or on its reversal when `reversed` is true.
    start: usize,

    /// Exclusive coordinate, at least `start`, on the same axis as `start`.
    limit: usize,

    /// Positive distance between selected positions.
    stride: usize,

    /// Number of selected positions.
    length: usize,

    /// Whether the coordinates apply to the reversal of the input axis.
    reversed: bool,
}

/// A host-known Boolean mask with explicit shape. Constructing it validates the number of entries. Indexing converts
/// its true positions into constant integer coordinates; it never reads a device value or tracer back to the host.
/// A scalar mask inserts an advanced axis of size one (`true`) or zero (`false`) without consuming an input axis.
/// Boolean indexing requires this concrete descriptor; runtime Boolean compaction and host reads of device or
/// traced masks are unsupported.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexMask {
    /// Refer to the documentation of [`shape`](Self::shape) for more information.
    shape: Vec<usize>,

    /// Refer to the documentation of [`values`](Self::values) for more information.
    values: Vec<bool>,
}

impl IndexMask {
    /// Creates a concrete mask with row-major `values` and the specified axis extents. The product of the extents
    /// must equal the number of values and fit in `usize`. A scalar mask has an empty shape and exactly one value.
    ///
    /// # Parameters
    ///
    ///   - `shape`: Extents of the consecutive input axes consumed by this mask.
    ///   - `values`: Host-known Boolean entries in row-major order; each `true` entry selects its position.
    pub fn new(shape: Vec<usize>, values: Vec<bool>) -> Result<Self, ProgramError> {
        let count = shape
            .iter()
            .try_fold(1usize, |count, extent| count.checked_mul(*extent))
            .ok_or_else(|| TypeError::invalid("index mask shape overflows `usize`"))?;
        if count != values.len() {
            return Err(TypeError::invalid(format!(
                "index mask shape requires {count} values but got {}",
                values.len()
            ))
            .into());
        }
        Ok(Self { shape, values })
    }

    /// Returns the extents of the input axes consumed by this mask. An empty shape denotes a scalar mask.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the concrete Boolean entries in row-major order.
    pub fn values(&self) -> &[bool] {
        &self.values
    }
}

/// One component of an array selection. Use [`index!`](crate::index) or the standard conversions to construct a list.
/// The value parameter is inferred from the receiver of [`Indexing::at`], including for a list containing only basic
/// indices. Array selectors borrow the receiver's value family; lift constants into the trace before using them.
///
/// Negative integer indices count backward from the axis end once. Remaining invalid integer indices follow the
/// bounds options supplied to the read or update function. Array indices broadcast jointly; separated advanced
/// indices put their broadcast axes first, while adjacent ones insert those axes in place.
#[derive(Clone, Debug, PartialEq)]
pub enum IndexSelector<'i, V: Value> {
    /// Host integer, slice, new axis, or ellipsis.
    Basic(BasicIndex),

    /// Integer array of coordinates, including rank-zero arrays. Boolean arrays are rejected; use [`IndexMask`] for
    /// concrete masks. Keeping rank-zero arrays distinct from host integers preserves advanced-index semantics.
    Array(&'i V),

    /// Explicitly host-known mask, consuming one input axis per mask axis.
    Mask(&'i IndexMask),
}

impl<V: Value> From<BasicIndex> for IndexSelector<'_, V> {
    fn from(value: BasicIndex) -> Self {
        Self::Basic(value)
    }
}

impl<V: Value> From<IndexSlice> for IndexSelector<'_, V> {
    fn from(value: IndexSlice) -> Self {
        Self::Basic(BasicIndex::Slice(value))
    }
}

impl<'i, V: Value> From<&'i V> for IndexSelector<'i, V> {
    fn from(value: &'i V) -> Self {
        Self::Array(value)
    }
}

impl<'i, V: Value> From<&'i IndexMask> for IndexSelector<'i, V> {
    fn from(value: &'i IndexMask) -> Self {
        Self::Mask(value)
    }
}

// Implement host integer conversions explicitly so they cannot overlap the borrowed-value conversion.
macro_rules! index_integer_conversions {
    // Implements descriptor conversion for one primitive host integer with an exact signed representation.
    ($integer:ty) => {
        impl<V: Value> From<$integer> for IndexSelector<'_, V> {
            fn from(value: $integer) -> Self {
                Self::Basic(BasicIndex::Integer(value.to_index_integer()))
            }
        }
    };
}

index_integer_conversions!(i8);
index_integer_conversions!(i16);
index_integer_conversions!(i32);
index_integer_conversions!(i64);
index_integer_conversions!(i128);
index_integer_conversions!(isize);
index_integer_conversions!(u8);
index_integer_conversions!(u16);
index_integer_conversions!(u32);
index_integer_conversions!(u64);
index_integer_conversions!(usize);

impl<I: IndexInteger, V: Value> From<Range<I>> for IndexSelector<'_, V> {
    fn from(value: Range<I>) -> Self {
        IndexSlice::new(Some(value.start.to_index_integer()), Some(value.end.to_index_integer()), 1).into()
    }
}

impl<I: IndexInteger, V: Value> From<RangeFrom<I>> for IndexSelector<'_, V> {
    fn from(value: RangeFrom<I>) -> Self {
        IndexSlice::new(Some(value.start.to_index_integer()), None, 1).into()
    }
}

impl<I: IndexInteger, V: Value> From<RangeTo<I>> for IndexSelector<'_, V> {
    fn from(value: RangeTo<I>) -> Self {
        IndexSlice::new(None, Some(value.end.to_index_integer()), 1).into()
    }
}

impl<V: Value> From<RangeFull> for IndexSelector<'_, V> {
    fn from(_: RangeFull) -> Self {
        IndexSlice::new(None, None, 1).into()
    }
}

/// Creates borrowed selections for reads and functional updates. The wrapper has no effects until a terminal
/// function is called, and its functions require only the capabilities needed for their direction.
///
/// # Examples
///
/// ```rust
/// use ryft_core::{Array, GatherMode, GatherOptions, Indexing, index};
/// let input = Array::matrix(3, 2, vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
/// let options = GatherOptions::new().with_mode(GatherMode::Clip);
/// let output = input.at(&index![.. by -1, 1]).get(&options).unwrap();
/// assert_eq!(output, Array::vector(vec![5_i32, 3, 1]).unwrap());
/// ```
pub trait Indexing: Value {
    /// Borrows this value and its [`IndexSelector`]s in an [`Indexed`] wrapper. Its `get` function reads the selection;
    /// `set`, `add`, `multiply`, `min`, and `max` return a new array, leaving the input unchanged. Validation occurs in
    /// those terminal functions, so constructing a wrapper is infallible. Refer to [`Indexed`] for supported geometry
    /// and the shared bounds and index-promise contracts.
    ///
    /// # Parameters
    ///
    ///   - `selectors`: Ordered selection components, usually constructed with [`index!`](crate::index). Omitted
    ///     trailing axes are selected in full. The wrapper borrows this list and its array indices until its last use.
    fn at<'a, 's, 'i>(&'a self, selectors: &'s [IndexSelector<'i, Self>]) -> Indexed<'a, 's, 'i, Self> {
        Indexed { input: self, selectors, marker: PhantomData }
    }
}

impl<V: Value> Indexing for V {}

/// Borrowed selection of a value. Separate input, selector-list, and index-value lifetimes allow both reusable lists
/// and temporary lists used within a chained call. The type parameter selects homogeneous or mixed-IR execution.
/// Selections have value semantics: reads produce values and updates return new values, leaving the input unchanged.
/// No selection exposes a mutable view.
///
/// # Supported Geometry
///
/// Concrete shapes support host integer indices, positive and negative strides, inserted axes, ellipses, broadcast
/// integer-array indices, and explicit host-known [`IndexMask`]s. Eager arrays and staged arrays use the same selection
/// rules. Reads compose slice, reverse, reshape, and gather; updates compose broadcast, reshape, and scatter.
///
/// Symbolic shapes use the mixed [`ArrayIrType`] value family so dimension values remain available during staging.
/// This path supports at most one indexed axis, full forward or reverse slices on the other axes, and inserted axes.
/// It also supports a symbolic query-array shape on that indexed axis. When working with a projected array tracer,
/// use its parent mixed-IR value for this path. General symbolic slice bounds, multiple symbolic advanced indices,
/// explicit output sharding, and index promises are currently rejected by the symbolic frontend. Untouched axes and
/// query axes may have zero runtime extents; symbolic reads impose an additional indexed-axis requirement documented
/// on their `get` function.
///
/// # Bounds and Index Promises
///
/// Bounds modes apply after negative-index normalization. A uniqueness promise applies to the final selected input
/// positions, including aliases introduced by normalization or clipping; this frontend never establishes uniqueness
/// for the caller. The concrete-shape frontend clears the sortedness promise before gathering or scattering because
/// normalization and interleaved slice coordinates can change index order. Existing operation type rules validate
/// memory, sharding, and reduction-state compatibility for the composed operations.
///
/// The value parameter precedes the type parameter, unlike the usual generic parameter order, because `T` defaults to
/// the value's type and a defaulted parameter must follow the parameter it depends on.
#[derive(Debug)]
pub struct Indexed<'a, 's, 'i, V: Value, T: Type = <V as Typed>::Type> {
    /// The input whose elements are read or functionally updated.
    input: &'a V,

    /// The borrowed list of host and value-level selectors.
    selectors: &'s [IndexSelector<'i, V>],

    /// [`PhantomData`] marker identifying the input's type universe.
    marker: PhantomData<fn() -> T>,
}

/// An expanded selector; ellipses remain separators even when they consume no axes.
#[derive(Clone)]
enum ExpandedIndex<V> {
    /// Host selector consuming one input axis or inserting an output axis.
    Basic(BasicIndex),

    /// Owned array coordinate, cloned or lifted from the original descriptor.
    Array(V),

    /// Scalar Boolean mask contributing an advanced axis without an input axis.
    Boolean(bool),
}

/// Shared gather/scatter geometry. The result without inserted new axes is exactly the gather result shape and the
/// scatter update shape; reshaping at the boundary adds/removes only known singleton axes.
struct IndexPlan<V> {
    /// Jointly broadcast coordinate components with a final index-vector axis.
    indices: V,

    /// Gather dimension mapping.
    dimensions: GatherDimensionNumbers,

    /// Static window size on every input axis.
    sizes: Vec<usize>,

    /// Gather/scatter shape before inserting new singleton axes.
    shape: Vec<usize>,

    /// Public selection shape including inserted singleton axes.
    output_shape: Vec<usize>,

    /// Positions of singleton axes absent from the gather result.
    new_axes: Vec<usize>,
}

/// Axis contribution in the public selection order, before the advanced broadcast dimensions are inserted.
#[derive(Copy, Clone)]
enum OutputAxis {
    /// A contiguous slice becomes a gather window dimension.
    Window(usize),

    /// A strided slice contributes a coordinate-query dimension.
    Query(usize),

    /// A newly inserted extent-one axis is restored after gathering.
    New,

    /// Position at which jointly broadcast integer-array dimensions appear.
    Advanced,
}

impl<V> Indexed<'_, '_, '_, V, ArrayType>
where
    V: Value<Type = ArrayType>
        + Broadcast
        + Reshape
        + Concatenate
        + ConvertElementType
        + Compare
        + Add
        + Select
        + Slice
        + Reverse
        + Gather,
    V::ExecutionDomain: Context,
    <V::ExecutionDomain as Domain>::Operation: From<ConstantOperation<Array>>,
{
    /// Reads this selection. Invalid scalar/array indices follow `options` after negative-index normalization.
    /// Slices clip their endpoints independently. Floating-point fill literals preserve their original encodings.
    /// Refer to [`Indexed`] for supported geometry and the shared bounds and index-promise contracts.
    /// A host integer that remains out of bounds under [`GatherMode::PromiseInBounds`] is rejected before staging.
    ///
    /// # Parameters
    ///
    ///   - `options`: Bounds handling, explicit fill, output placement, and caller promises for the selected input
    ///     positions. Uniqueness refers to coordinates after normalization and clipping. Sortedness is cleared before
    ///     composing the gather because selection normalization can change coordinate order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::{Array, GatherOptions, Indexing, index};
    /// let input = Array::matrix(2, 3, vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    /// assert_eq!(
    ///     input.at(&index![.., .. by -2]).get(&GatherOptions::new()),
    ///     Array::matrix(2, 2, vec![2_i32, 0, 5, 3]),
    /// );
    /// ```
    pub fn get(&self, options: &GatherOptions) -> Result<V, ProgramError> {
        let expanded = self.expanded()?;
        // Positive slices and valid scalar indices need only one slice and a rank adjustment. Reversal is a separate
        // linear operation, preserving symbolic extents and avoiding index tensors for ordinary reverse slicing.
        let input_type = self.input.r#type();
        if let GatherMode::Fill { value: Some(value) } = options.mode() {
            value.validate_as_constant()?;
            if value.r#type().rank() != 0 || value.r#type().data_type() != input_type.data_type() {
                return Err(TypeError::invalid("index fill must be a scalar of the input data type").into());
            }
        }
        if expanded.iter().all(|index| matches!(index, ExpandedIndex::Basic(_)))
            && input_type.shape().dimensions().iter().all(|dimension| dimension.value().is_some())
        {
            let mut starts = Vec::new();
            let mut limits = Vec::new();
            let mut strides = Vec::new();
            let mut reversed = Vec::new();
            let mut output = Vec::new();
            let mut axis = 0;
            let mut can_slice = true;
            for index in &expanded {
                match index {
                    ExpandedIndex::Basic(BasicIndex::NewAxis) => output.push(1),
                    ExpandedIndex::Basic(BasicIndex::Ellipsis) => {}
                    ExpandedIndex::Basic(BasicIndex::Slice(slice)) => {
                        let normalized = slice.normalize(input_type.dimension(axis).value().unwrap())?;
                        starts.push(normalized.start);
                        limits.push(normalized.limit);
                        strides.push(normalized.stride);
                        output.push(normalized.length);
                        if normalized.reversed {
                            reversed.push(axis);
                        }
                        axis += 1;
                    }
                    ExpandedIndex::Basic(BasicIndex::Integer(integer)) => {
                        let extent = input_type.dimension(axis).value().unwrap();
                        let integer = if *integer < 0 { integer.saturating_add(extent as i128) } else { *integer };
                        let valid = integer >= 0 && integer < extent as i128;
                        if !valid && matches!(options.mode(), GatherMode::PromiseInBounds) {
                            return Err(TypeError::invalid(format!(
                                "index {integer} is out of bounds for axis {axis} with extent {extent} under \
                                 `PromiseInBounds`",
                            ))
                            .into());
                        }
                        if extent == 0 || (!valid && matches!(options.mode(), GatherMode::Fill { .. })) {
                            can_slice = false;
                            break;
                        }
                        let integer = integer.clamp(0, extent as i128 - 1) as usize;
                        starts.push(integer);
                        limits.push(integer + 1);
                        strides.push(1);
                        axis += 1;
                    }
                    _ => unreachable!("only basic selectors reach the slicing fast path"),
                }
            }
            if can_slice {
                let input = if reversed.is_empty() { self.input.clone() } else { self.input.reverse(reversed)? };
                return input
                    .slice(&starts, &limits, &strides)?
                    .reshape_with_output_sharding(Shape::from(output), options.output_sharding().cloned());
            }
        }
        let plan = self.plan(&expanded, matches!(options.mode(), GatherMode::PromiseInBounds))?;
        // Negative-coordinate normalization and interleaving slice coordinates can change lexicographic order.
        // Keep uniqueness as a caller promise about normalized selected positions, but do not forward sortedness.
        let mut gather_options = options.clone().with_indices_are_sorted(false);
        if let Some(sharding) = options.output_sharding() {
            if sharding.dimensions().len() != plan.output_shape.len() {
                return Err(TypeError::invalid("index output sharding rank does not match selection rank").into());
            }
            gather_options = gather_options.with_output_sharding(
                sharding
                    .with_dimensions(
                        sharding
                            .dimensions()
                            .iter()
                            .enumerate()
                            .filter(|(axis, _)| !plan.new_axes.contains(axis))
                            .map(|(_, dimension)| dimension.clone())
                            .collect::<Vec<_>>(),
                    )
                    .map_err(|error| TypeError::invalid(error.to_string()))?,
            );
        }
        let empty_indexed_axis = plan
            .dimensions
            .collapsed_slice_dimensions()
            .iter()
            .any(|&axis| input_type.dimension(axis).value() == Some(0));
        let gathered = if empty_indexed_axis {
            let mut validation_shape = input_type.shape().dimensions().to_vec();
            for &axis in plan.dimensions.collapsed_slice_dimensions() {
                if validation_shape[axis].value() == Some(0) {
                    validation_shape[axis] = Dimension::Static(1);
                }
            }
            let output_type = input_type.clone().into_owned().with_shape(Shape::new(validation_shape)).gather(
                plan.indices.r#type().as_ref(),
                &plan.dimensions,
                &plan.sizes,
                &gather_options,
            )?;
            if plan.shape.iter().all(|&extent| extent != 0) && !matches!(options.mode(), GatherMode::Fill { .. }) {
                return Err(TypeError::invalid(
                    "cannot index a nonempty selection from an empty axis without fill mode",
                )
                .into());
            }
            let fill = gather_options.resolved_fill_value(input_type.data_type())?;
            self.constant(fill)?.broadcast(output_type, &[])?
        } else {
            self.input.gather(&plan.indices, &plan.dimensions, &plan.sizes, &gather_options)?
        };
        gathered.reshape_with_output_sharding(Shape::from(plan.output_shape), options.output_sharding().cloned())
    }
}

impl<V> Indexed<'_, '_, '_, V, ArrayType>
where
    V: Value<Type = ArrayType>
        + Broadcast
        + Reshape
        + Concatenate
        + ConvertElementType
        + Compare
        + Add
        + Select
        + Scatter,
    V::ExecutionDomain: Context,
    <V::ExecutionDomain as Domain>::Operation: From<ConstantOperation<Array>>,
{
    /// Returns a value with selected elements overwritten by `updates`. Conflicting repeated indices do not promise
    /// a deterministic winner. The input is unchanged, and updates broadcast to the selection shape.
    ///
    /// # Parameters
    ///
    ///   - `updates`: Values broadcast to the selected shape, with the same data type as the input.
    ///   - `options`: Bounds handling, output placement, and promises about the final normalized selected positions.
    ///     Sortedness is cleared before scatter; uniqueness remains an unchecked caller promise. Refer to
    ///     [`Indexed`] for how normalization and clipping affect these promises.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::{Array, Indexing, ScatterOptions, index};
    /// let input = Array::vector(vec![0_i32, 1, 2, 3]).unwrap();
    /// assert_eq!(
    ///     input.at(&index![1..3]).set(&Array::scalar(9_i32).unwrap(), &ScatterOptions::new()),
    ///     Array::vector(vec![0_i32, 9, 9, 3]),
    /// );
    /// ```
    pub fn set(&self, updates: &V, options: &ScatterOptions) -> Result<V, ProgramError> {
        self.update(updates, ScatterReductionKind::Overwrite, options)
    }

    /// Returns a value with every selected update added, including all updates at repeated indices. Updates broadcast
    /// to the selection shape, and the input remains unchanged.
    ///
    /// # Parameters
    ///
    ///   - `updates`: Values with the input's data type, broadcast to the selection shape.
    ///   - `options`: Bounds handling, output placement, and promises about normalized selected positions.
    ///     Sortedness is cleared before scatter; uniqueness remains an unchecked caller promise. Refer to
    ///     [`Indexed`] for how normalization and clipping affect these promises.
    pub fn add(&self, updates: &V, options: &ScatterOptions) -> Result<V, ProgramError> {
        self.update(updates, ScatterReductionKind::Add, options)
    }

    /// Returns a value with selected updates multiplied into it. Derivatives with respect to the updates require the
    /// existing scatter unique-indices promise; this function does not establish that promise. Updates broadcast
    /// to the selection shape, and the input remains unchanged.
    ///
    /// # Parameters
    ///
    ///   - `updates`: Factors with the input's data type, broadcast to the selection shape.
    ///   - `options`: Bounds handling, output placement, and promises about normalized selected positions.
    ///     Sortedness is cleared before scatter; uniqueness remains an unchecked caller promise. Refer to
    ///     [`Indexed`] for how normalization and clipping affect these promises.
    pub fn multiply(&self, updates: &V, options: &ScatterOptions) -> Result<V, ProgramError> {
        self.update(updates, ScatterReductionKind::Mul, options)
    }

    /// Returns a value with selected updates combined by the elementwise minimum. Updates broadcast
    /// to the selection shape, and the input remains unchanged.
    ///
    /// # Parameters
    ///
    ///   - `updates`: Values with the input's data type, broadcast to the selection shape.
    ///   - `options`: Bounds handling, output placement, and promises about normalized selected positions.
    ///     Sortedness is cleared before scatter; uniqueness remains an unchecked caller promise. Refer to
    ///     [`Indexed`] for how normalization and clipping affect these promises.
    pub fn min(&self, updates: &V, options: &ScatterOptions) -> Result<V, ProgramError> {
        self.update(updates, ScatterReductionKind::Min, options)
    }

    /// Returns a value with selected updates combined by the elementwise maximum. Updates broadcast
    /// to the selection shape, and the input remains unchanged.
    ///
    /// # Parameters
    ///
    ///   - `updates`: Values with the input's data type, broadcast to the selection shape.
    ///   - `options`: Bounds handling, output placement, and promises about normalized selected positions.
    ///     Sortedness is cleared before scatter; uniqueness remains an unchecked caller promise. Refer to
    ///     [`Indexed`] for how normalization and clipping affect these promises.
    pub fn max(&self, updates: &V, options: &ScatterOptions) -> Result<V, ProgramError> {
        self.update(updates, ScatterReductionKind::Max, options)
    }

    /// Uses the same window/query mapping as reads, removing only inserted singleton axes from the updates.
    fn update(&self, updates: &V, kind: ScatterReductionKind, options: &ScatterOptions) -> Result<V, ProgramError> {
        if updates.r#type().data_type() != self.input.r#type().data_type() {
            return Err(TypeError::invalid("index updates must have the input data type").into());
        }
        let expanded = self.expanded()?;
        let plan = self.plan(&expanded, matches!(options.mode(), ScatterMode::PromiseInBounds))?;
        let updates = updates.broadcast_to(Shape::from(plan.output_shape))?.reshape(Shape::from(plan.shape))?;
        let dimensions = ScatterDimensionNumbers::new(
            plan.dimensions.offset_dimensions().to_vec(),
            plan.dimensions.collapsed_slice_dimensions().to_vec(),
            plan.dimensions.start_index_map().to_vec(),
        );
        self.input
            .scatter(&plan.indices, &updates, &dimensions, kind, &options.clone().with_indices_are_sorted(false))
    }
}
impl<V> Indexed<'_, '_, '_, V, ArrayType>
where
    V: Value<Type = ArrayType> + Broadcast + Reshape + Concatenate + ConvertElementType + Compare + Add + Select,
    V::ExecutionDomain: Context,
    <V::ExecutionDomain as Domain>::Operation: From<ConstantOperation<Array>>,
{
    /// Expands ellipses and concrete masks once, keeping separators for advanced-index axis ordering.
    fn expanded(&self) -> Result<Vec<ExpandedIndex<V>>, ProgramError> {
        let rank = self.input.r#type().rank();
        let consumed = self
            .selectors
            .iter()
            .map(|selector| match selector {
                IndexSelector::Basic(BasicIndex::NewAxis | BasicIndex::Ellipsis) => 0,
                IndexSelector::Mask(mask) => mask.shape.len(),
                _ => 1,
            })
            .sum::<usize>();
        if consumed > rank {
            return Err(TypeError::invalid(format!(
                "index selection consumes {consumed} axes but input rank is {rank}"
            ))
            .into());
        }
        if self
            .selectors
            .iter()
            .filter(|selector| matches!(selector, IndexSelector::Basic(BasicIndex::Ellipsis)))
            .count()
            > 1
        {
            return Err(TypeError::invalid("index selection contains more than one ellipsis").into());
        }
        let mut result = Vec::new();
        let mut axis = 0;
        let mut ellipsis = false;
        for selector in self.selectors {
            match selector {
                IndexSelector::Basic(BasicIndex::Ellipsis) => {
                    ellipsis = true;
                    // A zero-width ellipsis still separates two advanced groups.
                    result.push(ExpandedIndex::Basic(BasicIndex::Ellipsis));
                    for _ in 0..rank - consumed {
                        result.push(ExpandedIndex::Basic(BasicIndex::Slice(IndexSlice::new(None, None, 1))));
                        axis += 1;
                    }
                }
                IndexSelector::Basic(value) => {
                    result.push(ExpandedIndex::Basic(*value));
                    if !matches!(value, BasicIndex::NewAxis) {
                        axis += 1;
                    }
                }
                IndexSelector::Array(value) => {
                    let r#type = value.r#type();
                    if !r#type.data_type().is_integer() {
                        return Err(TypeError::invalid(
                            "index arrays must have an integer data type; use `IndexMask` for concrete Boolean masks",
                        )
                        .into());
                    }
                    if r#type.memory() != self.input.r#type().memory() {
                        return Err(TypeError::invalid("index arrays and input must share one memory space").into());
                    }
                    result.push(ExpandedIndex::Array((*value).clone()));
                    axis += 1;
                }
                IndexSelector::Mask(mask) => {
                    if mask.shape.is_empty() {
                        result.push(ExpandedIndex::Boolean(mask.values[0]));
                        continue;
                    }
                    for (offset, &extent) in mask.shape.iter().enumerate() {
                        if self.input.r#type().dimension(axis + offset).value() != Some(extent) {
                            return Err(
                                TypeError::invalid("index mask shape does not match the consumed input axes").into()
                            );
                        }
                    }
                    for coordinate_axis in 0..mask.shape.len() {
                        // An empty mask can have huge trailing extents whose product does not fit in `usize`.
                        // No coordinate is decoded in that case. For nonempty masks the constructor has checked
                        // the entire positive shape product, so every trailing product fits as well.
                        let stride = if mask.values.is_empty() {
                            1
                        } else {
                            mask.shape[coordinate_axis + 1..].iter().product::<usize>()
                        };
                        let coordinates = mask
                            .values
                            .iter()
                            .enumerate()
                            .filter(|(_, active)| **active)
                            .map(|(index, _)| ((index / stride) % mask.shape[coordinate_axis]) as i64)
                            .collect::<Vec<_>>();
                        result.push(ExpandedIndex::Array(self.constant(Array::vector(coordinates)?)?));
                    }
                    axis += mask.shape.len();
                }
            }
        }
        if !ellipsis {
            for _ in consumed..rank {
                result.push(ExpandedIndex::Basic(BasicIndex::Slice(IndexSlice::new(None, None, 1))));
            }
        }
        Ok(result)
    }

    /// Lifts a host coordinate using the input's memory space and execution domain.
    #[inline]
    fn constant(&self, value: Array) -> Result<V, ProgramError> {
        portable_constant(self.input, value)
    }

    /// Converts integer queries without wrapping large unsigned values into valid signed positions. Negative signed
    /// indices add the axis extent once; values still outside the axis are left to the requested bounds policy.
    fn normalized_indices(&self, value: &V, extent: usize) -> Result<V, ProgramError> {
        let extent = i64::try_from(extent).map_err(|_| TypeError::invalid("indexed axis extent exceeds `i64::MAX`"))?;
        let value = if value.r#type().data_type() == DataType::U64 {
            let maximum = self.constant(Array::scalar(i64::MAX as u64)?)?;
            V::select(&value.greater_than(&maximum)?, &maximum, value)?.convert_element_type(DataType::I64)?
        } else {
            value.convert_element_type(DataType::I64)?
        };
        let zero = self.constant(Array::scalar(0_i64)?)?;
        let extent = self.constant(Array::scalar(extent)?)?;
        V::select(&value.less_than(&zero)?, &value.add(&extent)?, &value)
    }

    /// Builds one shared window/query mapping, preserving contiguous windows rather than constructing a full grid
    /// over every output element. Only genuinely strided slices add coordinate-query axes.
    fn plan(&self, expanded: &[ExpandedIndex<V>], promise: bool) -> Result<IndexPlan<V>, ProgramError> {
        let input_type = self.input.r#type();
        let shape = input_type
            .shape()
            .dimensions()
            .iter()
            .map(|dimension| {
                dimension.value().ok_or_else(|| ProgramError::UnsupportedOperation {
                    message: "general indexing requires concrete extents; use the supported single-axis mixed-IR \
                              selection for symbolic shapes"
                        .into(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if shape.iter().any(|&extent| i64::try_from(extent).is_err()) {
            return Err(TypeError::invalid("indexed axis extent exceeds `i64::MAX`").into());
        }
        let advanced =
            expanded.iter().any(|index| matches!(index, ExpandedIndex::Array(_) | ExpandedIndex::Boolean(_)));
        let is_advanced = |index: &ExpandedIndex<V>| {
            matches!(index, ExpandedIndex::Array(_) | ExpandedIndex::Boolean(_))
                || (advanced && matches!(index, ExpandedIndex::Basic(BasicIndex::Integer(_))))
        };
        let positions = expanded
            .iter()
            .enumerate()
            .filter_map(|(position, index)| is_advanced(index).then_some(position))
            .collect::<Vec<_>>();
        let contiguous = positions.windows(2).all(|pair| pair[1] == pair[0] + 1);
        let mut broadcast_shape = Shape::new(vec![]);
        for index in expanded {
            let next = match index {
                ExpandedIndex::Array(value) => Some(value.r#type().shape().clone()),
                ExpandedIndex::Boolean(value) => Some(Shape::new(vec![Dimension::Static(usize::from(*value))])),
                _ => None,
            };
            if let Some(next) = next {
                broadcast_shape = broadcast_shape
                    .broadcast(&next)
                    .map_err(|error| TypeError::invalid(format!("index arrays cannot broadcast: {error}")))?;
            }
        }
        let broadcast = broadcast_shape
            .dimensions()
            .iter()
            .map(|dimension| {
                dimension.value().ok_or_else(|| TypeError::invalid("general index query shape must be concrete"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut output = Vec::new();
        if advanced && !contiguous {
            output.push(OutputAxis::Advanced);
        }
        let mut components = Vec::<(usize, V, Option<usize>)>::new();
        let mut sizes = vec![1; shape.len()];
        let mut collapsed = Vec::new();
        let mut query_lengths = Vec::new();
        let mut axis = 0;
        for (position, index) in expanded.iter().enumerate() {
            if advanced && contiguous && positions.first() == Some(&position) {
                output.push(OutputAxis::Advanced);
            }
            match index {
                ExpandedIndex::Basic(BasicIndex::NewAxis) => output.push(OutputAxis::New),
                ExpandedIndex::Basic(BasicIndex::Ellipsis) | ExpandedIndex::Boolean(_) => {}
                ExpandedIndex::Basic(BasicIndex::Integer(integer)) => {
                    let integer = if *integer < 0 { integer.saturating_add(shape[axis] as i128) } else { *integer };
                    if promise && (integer < 0 || integer >= shape[axis] as i128) {
                        return Err(TypeError::invalid(format!(
                            "index {integer} is out of bounds for axis {axis} with extent {} under `PromiseInBounds`",
                            shape[axis]
                        ))
                        .into());
                    }
                    let integer = integer.clamp(i64::MIN as i128, i64::MAX as i128) as i64;
                    components.push((axis, self.constant(Array::scalar(integer)?)?, None));
                    collapsed.push(axis);
                    axis += 1;
                }
                ExpandedIndex::Array(value) => {
                    components.push((axis, self.normalized_indices(value, shape[axis])?, None));
                    collapsed.push(axis);
                    axis += 1;
                }
                ExpandedIndex::Basic(BasicIndex::Slice(slice)) => {
                    let normalized = slice.normalize(shape[axis])?;
                    if normalized.stride == 1 && !normalized.reversed {
                        sizes[axis] = normalized.length;
                        output.push(OutputAxis::Window(axis));
                        // Unindexed window axes start at zero already. Omit their coordinate component rather
                        // than broadcasting zeros and concatenating a redundant index-vector column.
                        if normalized.start != 0 {
                            components.push((axis, self.constant(Array::scalar(normalized.start as i64)?)?, None));
                        }
                    } else {
                        let query_axis = query_lengths.len();
                        query_lengths.push(normalized.length);
                        output.push(OutputAxis::Query(query_axis));
                        let coordinates = (0..normalized.length)
                            .map(|index| {
                                let coordinate = normalized.start + index * normalized.stride;
                                (if normalized.reversed { shape[axis] - 1 - coordinate } else { coordinate }) as i64
                            })
                            .collect::<Vec<_>>();
                        components.push((axis, self.constant(Array::vector(coordinates)?)?, Some(query_axis)));
                        collapsed.push(axis);
                    }
                    axis += 1;
                }
            }
        }
        let mut query_shape = Vec::new();
        let mut shape = Vec::new();
        let mut output_shape = Vec::new();
        let mut offsets = Vec::new();
        let mut new_axes = Vec::new();
        let mut advanced_axes = Vec::new();
        let mut slice_query_axes = vec![0; query_lengths.len()];
        for contribution in output {
            match contribution {
                OutputAxis::New => {
                    new_axes.push(output_shape.len());
                    output_shape.push(1);
                }
                OutputAxis::Advanced => {
                    for &extent in &broadcast {
                        advanced_axes.push(query_shape.len());
                        query_shape.push(extent);
                        shape.push(extent);
                        output_shape.push(extent);
                    }
                }
                OutputAxis::Query(query) => {
                    slice_query_axes[query] = query_shape.len();
                    query_shape.push(query_lengths[query]);
                    shape.push(query_lengths[query]);
                    output_shape.push(query_lengths[query]);
                }
                OutputAxis::Window(axis) => {
                    offsets.push(shape.len());
                    shape.push(sizes[axis]);
                    output_shape.push(sizes[axis]);
                }
            }
        }
        let mut vector_shape = query_shape.clone();
        vector_shape.push(1);
        let mut vectors = Vec::new();
        let mut start_map = Vec::new();
        for (axis, value, query) in components {
            let axes = if let Some(query) = query {
                vec![slice_query_axes[query]]
            } else {
                advanced_axes[advanced_axes.len() - value.r#type().rank()..].to_vec()
            };
            // Coordinate components may have different ranks. Move their sharding with the broadcast axes instead
            // of attaching the old rank's sharding to the joint query shape. This also retains reduction metadata.
            let value_type = value.r#type();
            let sharding = value_type
                .sharding()
                .map(|sharding| sharding.with_broadcasted_dimensions(query_shape.len(), &axes))
                .transpose()
                .map_err(|error| TypeError::invalid(error.to_string()))?;
            let output_type = ArrayType::new(value_type.data_type(), Shape::from(query_shape.clone()))
                .with_memory(value_type.memory())
                .with_sharding(sharding)
                .map_err(|error| TypeError::invalid(error.to_string()))?;
            let broadcast_value = value.broadcast(output_type, &axes)?;
            vectors.push(broadcast_value.reshape(Shape::from(vector_shape.clone()))?);
            start_map.push(axis);
        }
        let indices = if vectors.is_empty() {
            let mut empty_vector_shape = query_shape;
            empty_vector_shape.push(0);
            self.constant(Array::new(ArrayType::new_static(DataType::I64, empty_vector_shape), vec![])?)?
        } else {
            V::concatenate(vectors.iter(), -1)?
        };
        Ok(IndexPlan {
            indices,
            dimensions: GatherDimensionNumbers::new(offsets, collapsed, start_map),
            sizes,
            shape,
            output_shape,
            new_axes,
        })
    }
}

/// Constructs a fixed-size array of indexing descriptors for [`Indexing::at`].
///
/// Integer expressions select one position, ordinary exclusive Rust ranges select a slice, and `range by step`
/// supplies a signed stride. `..` selects a complete axis, `...` expands to the remaining axes, and `new_axis`
/// inserts an axis of size one. Borrowed integer arrays select multiple positions, and borrowed [`IndexMask`]
/// descriptors select positions identified by a concrete Boolean mask.
///
/// This macro only constructs selectors; it does not execute an operation. Each expression is evaluated once. Use
/// parentheses around expressions containing top-level commas, such as explicit generic argument lists. The receiver
/// of `at` determines the array-value parameter of the descriptors, so basic indexing requires no explicit type
/// annotation when used directly with a receiver.
///
/// # Examples
///
/// ```rust
/// # use ryft_core::{Array, BasicIndex, IndexSelector, IndexSlice};
/// use ryft_core::index;
/// let selection: [IndexSelector<'_, Array>; 3] = index![1..9 by 2, new_axis, ...];
/// assert_eq!(selection[0], IndexSelector::Basic(BasicIndex::Slice(
///     IndexSlice::new(Some(1), Some(9), 2),
/// )));
/// ```
#[macro_export]
macro_rules! index {
    // Finish the accumulated descriptor array, including the empty selection.
    (@items [$($items:expr,)*] []) => { [$($items,)*] };
    // A final selector does not need a trailing comma.
    (@items [$($items:expr,)*] [$($selector:tt)+]) => {
        [$($items,)* $crate::index!(@selector [] $($selector)+)]
    };
    // Commas separate selectors; grouped token trees retain their internal commas.
    (@items [$($items:expr,)*] [$($selector:tt)+] , $($rest:tt)*) => {
        $crate::index!(@items [$($items,)* $crate::index!(@selector [] $($selector)+),] [] $($rest)*)
    };
    // Collect one selector without evaluating any of its expressions.
    (@items [$($items:expr,)*] [$($selector:tt)*] $next:tt $($rest:tt)*) => {
        $crate::index!(@items [$($items,)*] [$($selector)* $next] $($rest)*)
    };
    // Ellipsis consumes all otherwise unmentioned input axes.
    (@selector [] ...) => {
        $crate::operations::manipulation::IndexSelector::Basic(
            $crate::operations::manipulation::BasicIndex::Ellipsis,
        )
    };
    // A new axis contributes a size-one output axis without consuming an input axis.
    (@selector [] new_axis) => {
        $crate::operations::manipulation::IndexSelector::Basic(
            $crate::operations::manipulation::BasicIndex::NewAxis,
        )
    };
    // A range with an omitted start retains that omission for negative-stride normalization.
    (@selector [] .. $($rest:tt)*) => {
        $crate::index!(@stop [None] [] $($rest)*)
    };
    // Separate an exclusive range's start from its stop and optional stride.
    (@selector [$($start:tt)+] .. $($rest:tt)*) => {
        $crate::index!(@stop [Some($crate::index!(@integer $($start)+))] [] $($rest)*)
    };
    // Plain expressions use type-directed conversions, including borrowed index arrays.
    (@selector [$($value:tt)+]) => {
        $crate::operations::manipulation::IndexSelector::from($($value)+)
    };
    // Scan for an exclusive range token while preserving grouped expressions.
    (@selector [$($value:tt)*] $next:tt $($rest:tt)*) => {
        $crate::index!(@selector [$($value)* $next] $($rest)*)
    };
    // A supplied stride and omitted stop preserve Python-style reversal semantics.
    (@stop [$start:expr] [] by $step:expr) => {
        $crate::index!(@slice $start, None, $crate::index!(@integer $step))
    };
    // A supplied stop and stride each evaluate exactly once.
    (@stop [$start:expr] [$($stop:tt)+] by $step:expr) => {
        $crate::index!(@slice $start, Some($crate::index!(@integer $($stop)+)), $crate::index!(@integer $step))
    };
    // An omitted stop and stride denote the remainder of the axis in forward order.
    (@stop [$start:expr] []) => {
        $crate::index!(@slice $start, None, 1)
    };
    // An ordinary exclusive range defaults to a stride of one.
    (@stop [$start:expr] [$($stop:tt)+]) => {
        $crate::index!(@slice $start, Some($crate::index!(@integer $($stop)+)), 1)
    };
    // Collect the stop expression up to the optional `by` keyword.
    (@stop [$start:expr] [$($stop:tt)*] $next:tt $($rest:tt)*) => {
        $crate::index!(@stop [$start] [$($stop)* $next] $($rest)*)
    };
    // Construct a basic slice without erasing omitted endpoints.
    (@slice $start:expr, $stop:expr, $step:expr) => {
        $crate::operations::manipulation::IndexSelector::Basic(
            $crate::operations::manipulation::BasicIndex::Slice(
                $crate::operations::manipulation::IndexSlice::new($start, $stop, $step),
            ),
        )
    };
    // Convert integer expressions through the supported, lossless host-integer conversions.
    (@integer $value:expr) => {
        $crate::operations::manipulation::IndexInteger::to_index_integer($value)
    };
    // Reject malformed internal parser states without recursively treating them as public syntax.
    (@$state:ident $($rest:tt)*) => {
        compile_error!("invalid indexing selector syntax")
    };
    // The public form accepts comma-separated selectors and an optional trailing comma.
    ($($selectors:tt)*) => {
        $crate::index!(@items [] [] $($selectors)*)
    };
}

// The macro is exported at the crate root by `#[macro_export]`; this re-export lets callers that import the
// manipulation facade reach it through the module path as well.
pub use crate::index;

/// Runtime geometry for a selection with at most one array or scalar index. Full slices preserve their dimension
/// identities; inserted axes are recorded separately so gather/scatter continue to operate on the original rank.
struct DynamicIndexPlan<V> {
    /// Optional input axis and normalized signed query array.
    query: Option<(usize, V)>,

    /// Full input axes traversed backward.
    reversed_axes: Vec<usize>,

    /// Positions of inserted size-one axes in the selected result.
    inserted_axes: Vec<usize>,
}

/// Binds a portable coordinate literal in the execution domain of `exemplar`, placed in its memory space. A
/// backend's stored constant family may contain capture handles, so coordinate literals use the ordinary constant
/// operation payload instead, which compiled contexts lower directly.
fn portable_constant<V>(exemplar: &V, value: Array) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType>,
    V::ExecutionDomain: Context,
    <V::ExecutionDomain as Domain>::Operation: From<ConstantOperation<Array>>,
{
    let r#type = value.r#type().into_owned().with_memory(exemplar.r#type().memory());
    let mut outputs = exemplar.execution_domain().bind(
        ConstantOperation::new(Array::new(r#type, value.storage_bytes().to_vec())?),
        Vec::new(),
        &[],
    )?;
    check_count!("output", outputs, 1, ProgramError);
    Ok(outputs.remove(0))
}

/// Binds a portable coordinate literal in the projected array execution domain of a mixed-IR `input`. Refer to the
/// documentation of [`portable_constant`] for more information.
fn dynamic_index_constant<V>(input: &V, value: Array) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    <V::Projected as Value>::ExecutionDomain: Context,
    <<V::Projected as Value>::ExecutionDomain as Domain>::Operation: From<ConstantOperation<Array>>,
{
    Ok(V::from_projected(portable_constant(&input.clone().into_projected()?, value)?))
}

/// Rejects the read and update options that the symbolic frontend does not remap yet: explicit output sharding and
/// the sortedness and uniqueness index promises. Bounds modes and fills remain available.
fn validate_symbolic_index_options(
    has_output_sharding: bool,
    indices_are_sorted: bool,
    unique_indices: bool,
) -> Result<(), ProgramError> {
    if has_output_sharding || indices_are_sorted || unique_indices {
        return Err(TypeError::invalid(
            "symbolic indexing does not yet support explicit output sharding or index promises",
        )
        .into());
    }
    Ok(())
}

/// Returns `input` with the projected array member reversed along `axes`, or an unchanged clone when no axis is
/// reversed, so that callers keep the mixed-IR value family on both paths.
fn reversed_projection<V>(input: &V, axes: &[usize]) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType> + Reverse>,
{
    if axes.is_empty() {
        return Ok(input.clone());
    }
    Ok(V::from_projected(input.clone().into_projected()?.reverse(axes.to_vec())?))
}

/// Resolves the supported symbolic selection geometry without reading traced array data on the host.
fn dynamic_index_plan<V>(
    input: &V,
    selectors: &[IndexSelector<'_, V>],
    promise_in_bounds: bool,
) -> Result<DynamicIndexPlan<V>, ProgramError>
where
    V: Value<Type = ArrayIrType> + DimensionSize + DimensionToScalar + ValueProjection<ArrayType>,
    V::Projected: Value<Type = ArrayType> + ConvertElementType + Compare + Add + Select + TransferToMemory,
    <V::Projected as Value>::ExecutionDomain: Context,
    <<V::Projected as Value>::ExecutionDomain as Domain>::Operation: From<ConstantOperation<Array>>,
{
    let input_type = input.r#type();
    let input_type = <&ArrayType>::try_from(input_type.as_ref())?;
    let consumed = selectors
        .iter()
        .filter(|selector| !matches!(selector, IndexSelector::Basic(BasicIndex::NewAxis | BasicIndex::Ellipsis)))
        .count();
    let rank = input_type.rank();
    if consumed > rank {
        return Err(
            TypeError::invalid(format!("index selection consumes {consumed} axes but input rank is {rank}")).into()
        );
    }
    let ellipses = selectors
        .iter()
        .filter(|selector| matches!(selector, IndexSelector::Basic(BasicIndex::Ellipsis)))
        .count();
    if ellipses > 1 {
        return Err(TypeError::invalid("index selection contains more than one ellipsis").into());
    }
    // Validate the whole selector list before lifting constants or staging dimension reads. A bad later selector
    // must not leave a partly staged indexing expression in the caller's context.
    let mut validation_axis = 0;
    let mut query_count = 0;
    for selector in selectors {
        match selector {
            IndexSelector::Basic(BasicIndex::Ellipsis) => validation_axis += input_type.rank() - consumed,
            IndexSelector::Basic(BasicIndex::NewAxis) => {}
            IndexSelector::Basic(BasicIndex::Slice(slice)) => {
                if slice.start().is_some() || slice.stop().is_some() || !matches!(slice.step(), -1 | 1) {
                    return Err(TypeError::invalid(
                        "symbolic indexing currently requires full slices with step `1` or `-1`",
                    )
                    .into());
                }
                validation_axis += 1;
            }
            IndexSelector::Basic(BasicIndex::Integer(index)) => {
                i64::try_from(*index).map_err(|_| {
                    TypeError::invalid("symbolic indexing requires host integer indices representable as `i64`")
                })?;
                if promise_in_bounds {
                    // Dynamic upper bounds are exclusive. Even when the actual extent is unavailable, an index
                    // outside the largest permitted extent cannot satisfy the caller's in-bounds promise.
                    let dimension = input_type.dimension(validation_axis);
                    let maximum = dimension.value().or_else(|| dimension.bounds().upper().map(|upper| upper - 1));
                    if maximum.is_some_and(|maximum| *index >= maximum as i128 || *index < -(maximum as i128)) {
                        return Err(
                            TypeError::invalid("host integer index is out of bounds under `PromiseInBounds`").into()
                        );
                    }
                }
                query_count += 1;
                validation_axis += 1;
            }
            IndexSelector::Array(indices) => {
                let indices_type = indices.r#type();
                let indices_type = <&ArrayType>::try_from(indices_type.as_ref())?;
                if !indices_type.data_type().is_integer() || !indices_type.data_type().is_signed() {
                    return Err(TypeError::invalid("symbolic indexing requires signed integer query arrays").into());
                }
                if indices_type.memory() != input_type.memory() {
                    return Err(TypeError::invalid("index arrays and input must share one memory space").into());
                }
                query_count += 1;
                validation_axis += 1;
            }
            IndexSelector::Mask(_) => {
                return Err(
                    TypeError::invalid("index masks require a concrete input shape in symbolic indexing").into()
                );
            }
        }
    }
    if query_count > 1 {
        return Err(TypeError::invalid("symbolic indexing currently supports one indexed axis").into());
    }
    let mut query = None;
    let mut reversed_axes = Vec::new();
    let mut inserted_axes = Vec::new();
    let mut input_axis = 0;
    let mut output_axis = 0;
    for selector in selectors {
        let indices = match selector {
            IndexSelector::Basic(BasicIndex::Ellipsis) => {
                let omitted = input_type.rank() - consumed;
                input_axis += omitted;
                output_axis += omitted;
                continue;
            }
            IndexSelector::Basic(BasicIndex::NewAxis) => {
                inserted_axes.push(output_axis);
                output_axis += 1;
                continue;
            }
            IndexSelector::Basic(BasicIndex::Slice(slice)) => {
                if slice.step() < 0 {
                    reversed_axes.push(input_axis);
                }
                input_axis += 1;
                output_axis += 1;
                continue;
            }
            IndexSelector::Basic(BasicIndex::Integer(index)) => {
                let index = i64::try_from(*index).unwrap();
                dynamic_index_constant(input, Array::scalar(index)?)?
            }
            IndexSelector::Array(indices) => (*indices).clone(),
            IndexSelector::Mask(_) => unreachable!("masks are rejected while validating the selector list"),
        };
        let indices_type = indices.r#type();
        let indices_type = <&ArrayType>::try_from(indices_type.as_ref())?;
        output_axis += indices_type.rank();
        // Normalize negatives using the actual retained dimension value. This is ordinary array arithmetic,
        // preserving the index array's symbolic shape and structural-zero tangent rather than concretizing it.
        let indices = indices.clone().into_projected()?.convert_element_type(DataType::I64)?;
        let extent = input
            .dimension_size(input_axis)?
            .to_scalar()?
            .into_projected()?
            .transfer_to_memory(input_type.memory())?;
        let zero = dynamic_index_constant(input, Array::scalar(0_i64)?)?.into_projected()?;
        let negative = indices.less_than(&zero)?;
        let wrapped = indices.add(&extent)?;
        let indices = V::from_projected(V::Projected::select(&negative, &wrapped, &indices)?);
        query = Some((input_axis, indices));
        input_axis += 1;
    }
    Ok(DynamicIndexPlan { query, reversed_axes, inserted_axes })
}

/// Selects through existing mixed-IR capabilities, retaining dimensions rather than encoding symbolic window sizes
/// as host integers. Explicit placement and index promises are rejected until their frontend remapping is defined.
fn dynamic_index_get<V>(
    input: &V,
    selectors: &[IndexSelector<'_, V>],
    options: &GatherOptions,
) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayIrType>
        + DimensionSize
        + DimensionToScalar
        + DynamicGather
        + DynamicReshape
        + ValueProjection<ArrayType>,
    V::Projected: Value<Type = ArrayType> + ConvertElementType + Compare + Add + Select + TransferToMemory + Reverse,
    V::DispatchDomain: Context<Type = ArrayIrType> + DimensionConstant,
    <V::Projected as Value>::ExecutionDomain: Context,
    <<V::Projected as Value>::ExecutionDomain as Domain>::Operation: From<ConstantOperation<Array>>,
{
    validate_symbolic_index_options(
        options.output_sharding().is_some(),
        options.indices_are_sorted(),
        options.unique_indices(),
    )?;
    // Identity and full-reversal selections skip gather, but explicit fills still have the same scalar/data-type
    // contract as indexed reads. Validate before those fast paths instead of silently accepting a malformed fill.
    if matches!(options.mode(), GatherMode::Fill { value: Some(_) }) {
        let input_type = input.r#type();
        options.resolved_fill_value(<&ArrayType>::try_from(input_type.as_ref())?.data_type())?;
    }
    let plan = dynamic_index_plan(input, selectors, matches!(options.mode(), GatherMode::PromiseInBounds))?;
    let mut output = reversed_projection(input, &plan.reversed_axes)?;
    if let Some((axis, indices)) = plan.query {
        output = output.dynamic_gather_axis(&indices, axis, options.mode().clone())?;
    }
    for axis in plan.inserted_axes {
        output = output.dynamic_expand_dimensions(axis)?;
    }
    Ok(output)
}

/// Applies an indexed update using runtime dimension values for broadcasting and removal of inserted axes. The
/// reversed input is updated in selection order and reversed back, so the returned value has the original geometry.
fn dynamic_index_update<V>(
    input: &V,
    selectors: &[IndexSelector<'_, V>],
    updates: &V,
    kind: ScatterReductionKind,
    options: &ScatterOptions,
) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayIrType>
        + DimensionSize
        + DimensionToScalar
        + DynamicScatter
        + DynamicReshape
        + DynamicBroadcast
        + ValueProjection<ArrayType>,
    V::Projected:
        Value<Type = ArrayType> + ConvertElementType + Compare + Add + Select + TransferToMemory + Reverse + Scatter,
    V::DispatchDomain: Context<Type = ArrayIrType> + DimensionConstant,
    <V::Projected as Value>::ExecutionDomain: Context,
    <<V::Projected as Value>::ExecutionDomain as Domain>::Operation: From<ConstantOperation<Array>>,
{
    validate_symbolic_index_options(
        options.output_sharding().is_some(),
        options.indices_are_sorted(),
        options.unique_indices(),
    )?;
    let plan = dynamic_index_plan(input, selectors, options.mode() == ScatterMode::PromiseInBounds)?;
    let input_type = input.r#type();
    let input_type = <&ArrayType>::try_from(input_type.as_ref())?;
    let mut dimensions = Vec::new();
    for axis in 0..input_type.rank() {
        if let Some((query_axis, indices)) = &plan.query
            && *query_axis == axis
        {
            let indices_type = indices.r#type();
            let indices_type = <&ArrayType>::try_from(indices_type.as_ref())?;
            for query_axis in 0..indices_type.rank() {
                dimensions.push(indices.dimension_size(query_axis)?);
            }
            continue;
        }
        dimensions.push(input.dimension_size(axis)?);
    }
    let mut selected_dimensions = dimensions.clone();
    for &axis in &plan.inserted_axes {
        selected_dimensions.insert(axis, input.dispatch_domain().dimension_constant(1)?);
    }
    let updates = updates.dynamic_broadcast_to(&selected_dimensions)?;
    let updates = if plan.inserted_axes.is_empty() { updates } else { updates.dynamic_reshape(&dimensions)? };
    let base = reversed_projection(input, &plan.reversed_axes)?;
    let output = if let Some((axis, indices)) = plan.query {
        base.dynamic_scatter_axis(&indices, &updates, axis, kind, options.mode())?
    } else {
        // A zero-width index vector describes a single whole-array update without inventing a runtime-sized
        // window. All update axes are window axes and every input element is updated exactly once.
        let indices = dynamic_index_constant(input, Array::vector(Vec::<i64>::new())?)?;
        V::from_projected(base.into_projected()?.scatter(
            &indices.into_projected()?,
            &updates.into_projected()?,
            &ScatterDimensionNumbers::new((0..input_type.rank()).collect(), vec![], vec![]),
            kind,
            options,
        )?)
    };
    reversed_projection(&output, &plan.reversed_axes)
}

impl<V> Indexed<'_, '_, '_, V, ArrayIrType>
where
    V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    /// Serves a concrete-geometry selection through the homogeneous frontend by projecting the input and every index
    /// array into the array member family, re-borrowing the selectors over those projections, and handing the
    /// projected selection to `select`.
    fn with_projected_selection<R, F>(&self, select: F) -> Result<R, ProgramError>
    where
        F: FnOnce(Indexed<'_, '_, '_, V::Projected, ArrayType>) -> Result<R, ProgramError>,
    {
        let input = self.input.clone().into_projected()?;
        let values = self
            .selectors
            .iter()
            .map(|selector| match selector {
                IndexSelector::Array(value) => Ok(Some((*value).clone().into_projected()?)),
                _ => Ok(None),
            })
            .collect::<Result<Vec<_>, TypeError>>()?;
        let selectors = self
            .selectors
            .iter()
            .zip(&values)
            .map(|(selector, value)| match selector {
                IndexSelector::Basic(value) => IndexSelector::Basic(*value),
                IndexSelector::Array(_) => IndexSelector::Array(value.as_ref().unwrap()),
                IndexSelector::Mask(mask) => IndexSelector::Mask(mask),
            })
            .collect::<Vec<_>>();
        select(input.at(&selectors))
    }

    /// Returns whether the input or an index array retains a non-concrete dimension.
    fn has_symbolic_shape(&self) -> Result<bool, ProgramError> {
        let r#type = self.input.r#type();
        if <&ArrayType>::try_from(r#type.as_ref())?
            .shape()
            .dimensions()
            .iter()
            .any(|dimension| dimension.value().is_none())
        {
            return Ok(true);
        }
        for selector in self.selectors {
            if let IndexSelector::Array(value) = selector {
                let r#type = value.r#type();
                if <&ArrayType>::try_from(r#type.as_ref())?
                    .shape()
                    .dimensions()
                    .iter()
                    .any(|dimension| dimension.value().is_none())
                {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }
}

impl<V> Indexed<'_, '_, '_, V, ArrayIrType>
where
    V: Value<Type = ArrayIrType>
        + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + DimensionSize
        + DimensionToScalar
        + DynamicGather
        + DynamicReshape,
    V::Projected: Broadcast
        + Reshape
        + Concatenate
        + ConvertElementType
        + Compare
        + Add
        + Select
        + Slice
        + Reverse
        + Gather
        + TransferToMemory,
    <V::Projected as Value>::ExecutionDomain: Context,
    <<V::Projected as Value>::ExecutionDomain as Domain>::Operation: From<ConstantOperation<Array>>,
    V::DispatchDomain: Context<Type = ArrayIrType> + DimensionConstant,
{
    /// Reads a mixed-IR array selection. Concrete geometry shares the homogeneous implementation. Symbolic shapes
    /// support one indexed axis, full forward/reverse slices, and inserted axes through retained dimension values.
    /// Use the parent mixed-IR value when starting from a projected array tracer with symbolic dimensions. Refer to
    /// [`Indexed`] for the complete supported-geometry contract.
    ///
    /// Symbolic reads inherit [`DynamicGather`]'s one-element window requirement on the indexed axis: its minimum
    /// extent must be positive unless the query is statically empty. Untouched axes and query axes may have zero
    /// runtime extents.
    ///
    /// # Parameters
    ///
    ///   - `options`: Bounds handling and optional fill. Symbolic geometry rejects explicit output sharding and
    ///     index promises. Concrete geometry also supports placement and uniqueness as described on [`Indexed`].
    pub fn get(&self, options: &GatherOptions) -> Result<V, ProgramError> {
        if self.has_symbolic_shape()? {
            return dynamic_index_get(self.input, self.selectors, options);
        }
        self.with_projected_selection(|selection| selection.get(options)).map(V::from_projected)
    }
}

impl<V> Indexed<'_, '_, '_, V, ArrayIrType>
where
    V: Value<Type = ArrayIrType>
        + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + DimensionSize
        + DimensionToScalar
        + DynamicScatter
        + DynamicReshape
        + DynamicBroadcast,
    V::Projected: Broadcast
        + Reshape
        + Concatenate
        + ConvertElementType
        + Compare
        + Add
        + Select
        + Reverse
        + Scatter
        + TransferToMemory,
    <V::Projected as Value>::ExecutionDomain: Context,
    <<V::Projected as Value>::ExecutionDomain as Domain>::Operation: From<ConstantOperation<Array>>,
    V::DispatchDomain: Context<Type = ArrayIrType> + DimensionConstant,
{
    /// Overwrites selected mixed-IR array elements, preserving the input and the existing scatter bounds contract.
    /// Updates broadcast to the selected shape using retained dimension values when necessary.
    ///
    /// # Parameters
    ///
    ///   - `updates`: Array value with the input's data type and shape broadcastable to the selection.
    ///   - `options`: Scatter bounds policy and placement/promises for concrete geometry. Symbolic geometry rejects
    ///     explicit output sharding and index promises. Refer to [`Indexed`] for supported geometry and the shared
    ///     bounds and index-promise contracts.
    pub fn set(&self, updates: &V, options: &ScatterOptions) -> Result<V, ProgramError> {
        self.update(updates, ScatterReductionKind::Overwrite, options)
    }

    /// Adds every selected update, including duplicates, to a mixed-IR array. Updates broadcast to the selected shape
    /// using retained dimension values when necessary. Refer to the documentation of [`set`](Self::set) for the
    /// shared parameter contract.
    pub fn add(&self, updates: &V, options: &ScatterOptions) -> Result<V, ProgramError> {
        self.update(updates, ScatterReductionKind::Add, options)
    }

    /// Multiplies selected updates into a mixed-IR array; scatter's differentiation restrictions apply. Updates
    /// broadcast to the selected shape using retained dimension values when necessary. Refer to the documentation of
    /// [`set`](Self::set) for the shared parameter contract.
    pub fn multiply(&self, updates: &V, options: &ScatterOptions) -> Result<V, ProgramError> {
        self.update(updates, ScatterReductionKind::Mul, options)
    }

    /// Combines selected mixed-IR updates with the input using the elementwise minimum. Updates broadcast to the
    /// selected shape using retained dimension values when necessary. Refer to the documentation of
    /// [`set`](Self::set) for the shared parameter contract.
    pub fn min(&self, updates: &V, options: &ScatterOptions) -> Result<V, ProgramError> {
        self.update(updates, ScatterReductionKind::Min, options)
    }

    /// Combines selected mixed-IR updates with the input using the elementwise maximum. Updates broadcast to the
    /// selected shape using retained dimension values when necessary. Refer to the documentation of
    /// [`set`](Self::set) for the shared parameter contract.
    pub fn max(&self, updates: &V, options: &ScatterOptions) -> Result<V, ProgramError> {
        self.update(updates, ScatterReductionKind::Max, options)
    }

    /// Projects concrete geometry or retains explicit dimensions for a symbolic selection.
    fn update(&self, updates: &V, kind: ScatterReductionKind, options: &ScatterOptions) -> Result<V, ProgramError> {
        if self.has_symbolic_shape()? {
            return dynamic_index_update(self.input, self.selectors, updates, kind, options);
        }
        let updates = updates.clone().into_projected()?;
        self.with_projected_selection(|selection| selection.update(&updates, kind, options))
            .map(V::from_projected)
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayIrOperation, ArrayIrValue, ArrayOperation, DimensionBounds, DimensionVariable, LogicalMesh, Memory,
        MeshAxis, MeshAxisType, Sharding, ShardingDimension,
    };
    use crate::batching::{BatchAxis, batch};
    use crate::contexts::{Context, EagerContext};
    use crate::differentiation::differentiate_at;
    use crate::partial::PartialValue;
    use crate::tracing::{Trace, Tracer, TracingContext};

    use super::*;

    /// Tracer of the mixed-IR eager context used by the symbolic-geometry tests.
    type ArrayIrTracer = Tracer<TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>;

    #[test]
    fn test_index_integer_to_index_integer() {
        assert_eq!(i128::MIN.to_index_integer(), i128::MIN);
        assert_eq!(i128::MAX.to_index_integer(), i128::MAX);
        assert_eq!(usize::MAX.to_index_integer(), usize::MAX as i128);
        assert_eq!(u64::MAX.to_index_integer(), u64::MAX as i128);
        assert_eq!((-1_i32).to_index_integer(), -1);
    }

    #[test]
    fn test_index_slice_new() {
        assert_eq!(IndexSlice::new(Some(-2), None, -1), IndexSlice { start: Some(-2), stop: None, step: -1 });
    }

    #[test]
    fn test_index_slice_start() {
        assert_eq!(IndexSlice::new(Some(-2), None, 1).start(), Some(-2));
        assert_eq!(IndexSlice::new(None, None, 1).start(), None);
    }

    #[test]
    fn test_index_slice_stop() {
        assert_eq!(IndexSlice::new(None, Some(3), 1).stop(), Some(3));
        assert_eq!(IndexSlice::new(None, None, 1).stop(), None);
    }

    #[test]
    fn test_index_slice_step() {
        assert_eq!(IndexSlice::new(None, None, -2).step(), -2);
        assert_eq!(IndexSlice::new(None, None, 1).step(), 1);
    }

    #[test]
    fn test_index_slice_normalize() {
        assert_eq!(
            IndexSlice::new(Some(1), Some(8), 2).normalize(10).unwrap(),
            NormalizedIndexSlice { start: 1, limit: 8, stride: 2, length: 4, reversed: false },
        );
        assert_eq!(
            IndexSlice::new(Some(-3), None, 1).normalize(10).unwrap(),
            NormalizedIndexSlice { start: 7, limit: 10, stride: 1, length: 3, reversed: false },
        );
        assert_eq!(
            IndexSlice::new(Some(8), Some(1), 1).normalize(10).unwrap(),
            NormalizedIndexSlice { start: 8, limit: 8, stride: 1, length: 0, reversed: false },
        );
        assert!(matches!(
            IndexSlice::new(None, None, 0).normalize(10),
            Err(ProgramError::Type(TypeError::Invalid { message, .. }))
                if message == "index slice step must not be zero",
        ));
    }

    #[test]
    fn test_index_slice_normalize_reversed() {
        assert_eq!(
            IndexSlice::new(None, None, -1).normalize(5).unwrap(),
            NormalizedIndexSlice { start: 0, limit: 5, stride: 1, length: 5, reversed: true },
        );
        // An explicit -1 is the last element, whereas the omitted reverse stop lies before the first element.
        assert_eq!(
            IndexSlice::new(None, Some(-1), -1).normalize(5).unwrap(),
            NormalizedIndexSlice { start: 0, limit: 0, stride: 1, length: 0, reversed: true },
        );
        assert_eq!(
            IndexSlice::new(Some(3), Some(0), -2).normalize(5).unwrap(),
            NormalizedIndexSlice { start: 1, limit: 4, stride: 2, length: 2, reversed: true },
        );
        assert_eq!(
            IndexSlice::new(None, None, -1).normalize(0).unwrap(),
            NormalizedIndexSlice { start: 0, limit: 0, stride: 1, length: 0, reversed: true },
        );
    }

    #[test]
    fn test_index_slice_normalize_extreme_endpoints() {
        assert_eq!(
            IndexSlice::new(Some(i128::MIN), Some(i128::MAX), 1).normalize(5).unwrap(),
            NormalizedIndexSlice { start: 0, limit: 5, stride: 1, length: 5, reversed: false },
        );
        assert_eq!(
            IndexSlice::new(Some(i128::MAX), Some(i128::MIN), i128::MIN).normalize(5).unwrap(),
            NormalizedIndexSlice { start: 0, limit: 5, stride: 5, length: 1, reversed: true },
        );
        assert_eq!(
            IndexSlice::new(None, None, i128::MAX).normalize(0).unwrap(),
            NormalizedIndexSlice { start: 0, limit: 0, stride: 1, length: 0, reversed: false },
        );
    }

    #[test]
    fn test_index_mask_new() {
        assert_eq!(IndexMask::new(vec![2], vec![true, false]).unwrap().values(), &[true, false]);
        assert!(matches!(
            IndexMask::new(vec![2], vec![true]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "index mask shape requires 2 values but got 1",
        ));
        assert!(matches!(
            IndexMask::new(vec![usize::MAX, 2], vec![]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "index mask shape overflows `usize`",
        ));
    }

    #[test]
    fn test_index_mask_shape() {
        assert_eq!(IndexMask::new(vec![2, 0], vec![]).unwrap().shape(), &[2, 0]);
        assert_eq!(IndexMask::new(vec![], vec![true]).unwrap().shape(), &[] as &[usize]);
    }

    #[test]
    fn test_index_mask_values() {
        assert_eq!(IndexMask::new(vec![2], vec![false, true]).unwrap().values(), &[false, true]);
    }

    #[test]
    fn test_index_selector_from() {
        assert_eq!(IndexSelector::<Array>::from(-1_i32), IndexSelector::Basic(BasicIndex::Integer(-1)));
        assert_eq!(
            IndexSelector::<Array>::from(usize::MAX),
            IndexSelector::Basic(BasicIndex::Integer(usize::MAX as i128))
        );
        assert_eq!(
            IndexSelector::<Array>::from(1..3),
            IndexSelector::Basic(BasicIndex::Slice(IndexSlice::new(Some(1), Some(3), 1)))
        );
        assert_eq!(
            IndexSelector::<Array>::from(1..),
            IndexSelector::Basic(BasicIndex::Slice(IndexSlice::new(Some(1), None, 1)))
        );
        assert_eq!(
            IndexSelector::<Array>::from(..3),
            IndexSelector::Basic(BasicIndex::Slice(IndexSlice::new(None, Some(3), 1)))
        );
        assert_eq!(
            IndexSelector::<Array>::from(..),
            IndexSelector::Basic(BasicIndex::Slice(IndexSlice::new(None, None, 1)))
        );
        assert_eq!(IndexSelector::<Array>::from(BasicIndex::NewAxis), IndexSelector::Basic(BasicIndex::NewAxis));
        assert_eq!(
            IndexSelector::<Array>::from(IndexSlice::new(Some(-2), None, -1)),
            IndexSelector::Basic(BasicIndex::Slice(IndexSlice::new(Some(-2), None, -1)))
        );
        let rows = Array::vector(vec![0_i32, 2]).unwrap();
        assert_eq!(IndexSelector::from(&rows), IndexSelector::Array(&rows));
        let mask = IndexMask::new(vec![2], vec![true, false]).unwrap();
        assert_eq!(IndexSelector::<Array>::from(&mask), IndexSelector::Mask(&mask));
    }

    #[test]
    fn test_indexing_at() {
        let input = Array::vector(vec![10_i32, 20, 30]).unwrap();
        let selectors = index![1];
        let indexed = input.at(&selectors);
        assert_eq!(indexed.get(&GatherOptions::new()), Array::scalar(20_i32));
        assert_eq!(input, Array::vector(vec![10_i32, 20, 30]).unwrap());
    }

    #[test]
    fn test_indexed_get() {
        let input = Array::matrix(3, 4, (0_i32..12).collect()).unwrap();
        assert_eq!(
            input.at(&index![1..3, 1..4 by 2]).get(&GatherOptions::new()),
            Array::matrix(2, 2, vec![5_i32, 7, 9, 11])
        );
        assert_eq!(input.at(&index![..., -1]).get(&GatherOptions::new()), Array::vector(vec![3_i32, 7, 11]));
        assert_eq!(input.at(&index![]).get(&GatherOptions::new()), Ok(input.clone()));
        assert_eq!(
            input.at(&index![new_axis, 1, ..]).get(&GatherOptions::new()),
            Array::matrix(1, 4, vec![4_i32, 5, 6, 7])
        );
    }

    #[test]
    fn test_indexed_get_reverse_and_empty() {
        let input = Array::vector(vec![0_i32, 1, 2, 3, 4]).unwrap();
        assert_eq!(input.at(&index![..by - 2]).get(&GatherOptions::new()), Array::vector(vec![4_i32, 2, 0]));
        assert_eq!(input.at(&index![..-1 by -1]).get(&GatherOptions::new()), Array::vector(Vec::<i32>::new()));
        assert_eq!(input.at(&index![4..0 by -2]).get(&GatherOptions::new()), Array::vector(vec![4_i32, 2]));
        assert_eq!(input.at(&index![(i128::MIN)..(i128::MAX)]).get(&GatherOptions::new()), Ok(input.clone()));
        assert_eq!(input.at(&index![..by(i128::MIN)]).get(&GatherOptions::new()), Array::vector(vec![4_i32]));
        assert_eq!(input.at(&index![4..2]).get(&GatherOptions::new()), Array::vector(Vec::<i32>::new()));
    }

    #[test]
    fn test_indexed_get_advanced() {
        // Expected values were independently generated with NumPy 2.3.5.
        let input =
            Array::from_elements(ArrayType::new_static(DataType::I32, [3, 4, 5]), &(0_i32..60).collect::<Vec<_>>())
                .unwrap();
        let rows = Array::vector(vec![0_i32, 2]).unwrap();
        let columns = Array::vector(vec![1_i32, 3]).unwrap();
        assert_eq!(
            input.at(&index![.., &rows, &columns]).get(&GatherOptions::new()),
            Array::matrix(3, 2, vec![1_i32, 13, 21, 33, 41, 53])
        );
        // An ellipsis remains a separator even when it expands to zero axes.
        assert_eq!(
            input.at(&index![.., &rows, ..., &columns]).get(&GatherOptions::new()),
            Array::matrix(2, 3, vec![1_i32, 21, 41, 13, 33, 53])
        );
        assert_eq!(
            input.at(&index![1, .., &columns]).get(&GatherOptions::new()),
            Array::matrix(2, 4, vec![21_i32, 26, 31, 36, 23, 28, 33, 38])
        );
        assert_eq!(
            input.at(&index![.., 1, &columns]).get(&GatherOptions::new()),
            Array::matrix(3, 2, vec![6_i32, 8, 26, 28, 46, 48])
        );
        let scalar = Array::scalar(1_i32).unwrap();
        assert_eq!(
            input.at(&index![&scalar, .., &columns]).get(&GatherOptions::new()),
            Array::matrix(2, 4, vec![21_i32, 26, 31, 36, 23, 28, 33, 38])
        );
        assert_eq!(
            input.at(&index![.., &rows, new_axis, &columns]).get(&GatherOptions::new()),
            Array::from_elements(ArrayType::new_static(DataType::I32, [2, 3, 1]), &[1_i32, 21, 41, 13, 33, 53])
        );
        let rows = Array::matrix(2, 1, vec![0_i32, 2]).unwrap();
        assert_eq!(
            input.at(&index![&rows, &columns, 1]).get(&GatherOptions::new()),
            Array::matrix(2, 2, vec![6_i32, 16, 46, 56])
        );
    }

    #[test]
    fn test_indexed_get_masks() {
        let input = Array::matrix(2, 3, vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let mask = IndexMask::new(vec![2, 3], vec![true, false, true, false, true, false]).unwrap();
        assert_eq!(input.at(&index![&mask]).get(&GatherOptions::new()), Array::vector(vec![0_i32, 2, 4]));
        let empty = IndexMask::new(vec![2], vec![false, false]).unwrap();
        assert_eq!(
            input.at(&index![&empty]).get(&GatherOptions::new()),
            Array::from_elements::<i32>(ArrayType::new_static(DataType::I32, [0, 3]), &[])
        );
        let active = IndexMask::new(vec![], vec![true]).unwrap();
        assert_eq!(
            input.at(&index![&active]).get(&GatherOptions::new()),
            Array::from_elements(ArrayType::new_static(DataType::I32, [1, 2, 3]), &[0_i32, 1, 2, 3, 4, 5])
        );
        let inactive = IndexMask::new(vec![], vec![false]).unwrap();
        assert_eq!(
            input.at(&index![&inactive]).get(&GatherOptions::new()),
            Array::from_elements::<i32>(ArrayType::new_static(DataType::I32, [0, 2, 3]), &[])
        );
    }

    #[test]
    fn test_indexed_get_bounds() {
        let input = Array::vector(vec![10_i32, 20, 30]).unwrap();
        let indices = Array::vector(vec![-4_i64, -1, 0, 3, 9]).unwrap();
        let clip = GatherOptions::new().with_mode(GatherMode::Clip);
        let fill =
            GatherOptions::new().with_mode(GatherMode::Fill { value: Some(Box::new(Array::scalar(-99_i32).unwrap())) });
        // Negative coordinates count from the end once; the remaining invalid coordinates clip or fill.
        assert_eq!(input.at(&index![&indices]).get(&clip), Array::vector(vec![10_i32, 30, 10, 30, 30]));
        assert_eq!(input.at(&index![&indices]).get(&fill), Array::vector(vec![-99_i32, 30, 10, -99, -99]));
        assert_eq!(input.at(&index![(i128::MAX)]).get(&clip), Array::scalar(30_i32));
        assert_eq!(input.at(&index![(i128::MIN)]).get(&fill), Array::scalar(-99_i32));

        // The default fill is the gather's data-type default (the minimum for signed integers), for host integers as
        // well as for query arrays.
        let default_fill = GatherOptions::new().with_mode(GatherMode::Fill { value: None });
        assert_eq!(input.at(&index![7]).get(&default_fill), Array::scalar(i32::MIN));
        assert_eq!(
            input.at(&index![&indices]).get(&default_fill),
            Array::vector(vec![i32::MIN, 30, 10, i32::MIN, i32::MIN]),
        );

        // An explicit fill must be a scalar of the input data type.
        let mismatched_fill =
            GatherOptions::new().with_mode(GatherMode::Fill { value: Some(Box::new(Array::scalar(0_f32).unwrap())) });
        assert!(matches!(
            input.at(&index![..]).get(&mismatched_fill),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "index fill must be a scalar of the input data type",
        ));

        // An empty axis can only be read through fill mode.
        let empty = Array::vector(Vec::<i32>::new()).unwrap();
        assert_eq!(empty.at(&index![0]).get(&fill), Array::scalar(-99_i32));
        assert!(matches!(
            empty.at(&index![0]).get(&clip),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot index a nonempty selection from an empty axis without fill mode",
        ));
    }

    #[test]
    fn test_indexed_get_validation() {
        let input = Array::vector(vec![10_i32, 20, 30]).unwrap();
        // A host integer that stays out of bounds after normalization violates the default in-bounds promise.
        assert!(matches!(
            input.at(&index![3]).get(&GatherOptions::new()),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "index 3 is out of bounds for axis 0 with extent 3 under `PromiseInBounds`",
        ));

        // Selector lists are validated as a whole: one ellipsis at most, no more consumed axes than the rank, and a
        // nonzero slice step.
        assert!(matches!(
            input.at(&index![..., ...]).get(&GatherOptions::new()),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "index selection contains more than one ellipsis",
        ));
        assert!(matches!(
            input.at(&index![0, 0]).get(&GatherOptions::new()),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "index selection consumes 2 axes but input rank is 1",
        ));
        assert!(matches!(
            input.at(&index![.. by 0]).get(&GatherOptions::new()),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "index slice step must not be zero",
        ));

        // Query arrays must be integers, and concrete masks must match the axes they consume.
        let boolean = Array::vector(vec![true, false, true]).unwrap();
        assert!(matches!(
            input.at(&index![&boolean]).get(&GatherOptions::new()),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "index arrays must have an integer data type; use `IndexMask` for concrete Boolean masks",
        ));
        let mask = IndexMask::new(vec![2], vec![true, false]).unwrap();
        assert!(matches!(
            input.at(&index![&mask]).get(&GatherOptions::new()),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "index mask shape does not match the consumed input axes",
        ));
    }

    #[test]
    fn test_indexed_get_advanced_strides() {
        let input =
            Array::from_elements(ArrayType::new_static(DataType::I32, [3, 4, 5]), &(0_i32..60).collect::<Vec<_>>())
                .unwrap();
        let rows = Array::vector(vec![0_i32, 2]).unwrap();
        let columns = Array::vector(vec![1_i32, 3]).unwrap();
        assert_eq!(
            input.at(&index![&rows, ..by - 2, &columns]).get(&GatherOptions::new()),
            Array::matrix(2, 2, vec![16_i32, 6, 58, 48]),
        );
        assert_eq!(
            input.at(&index![..by - 2, &rows, ..by - 2]).get(&GatherOptions::new()),
            Array::from_elements(
                ArrayType::new_static(DataType::I32, [2, 2, 3]),
                &[44_i32, 42, 40, 54, 52, 50, 4, 2, 0, 14, 12, 10],
            ),
        );
    }

    #[test]
    fn test_indexed_get_unsigned_extremes() {
        let input = Array::vector(vec![10_i32, 20, 30]).unwrap();
        let indices = Array::vector(vec![0_u64, 2, u64::MAX]).unwrap();
        let fill =
            GatherOptions::new().with_mode(GatherMode::Fill { value: Some(Box::new(Array::scalar(-99_i32).unwrap())) });
        assert_eq!(input.at(&index![&indices]).get(&fill), Array::vector(vec![10_i32, 30, -99]),);
        assert_eq!(
            input.at(&index![&indices]).get(&GatherOptions::new().with_mode(GatherMode::Clip)),
            Array::vector(vec![10_i32, 30, 30]),
        );
        assert_eq!(
            input
                .at(&index![&indices])
                .set(&Array::scalar(7_i32).unwrap(), &ScatterOptions::new().with_mode(ScatterMode::Drop)),
            Array::vector(vec![7_i32, 20, 7]),
        );
    }

    #[test]
    fn test_indexed_get_staging() {
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |input| input.at(&index![.., 1]).get(&GatherOptions::new()),
            ArrayType::new_static(DataType::F64, [2, 3]),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
            lambda %0:f64[2, 3] .
            let %1:f64[2, 1] = slice [start_indices=[0, 1], limit_indices=[2, 2]] %0
                %2:f64[2] = reshape [shape=[2]] %1
            in (%2)
        "}
            .trim_end()
        );
        assert_eq!(
            program.interpret(Array::matrix(2, 3, vec![10_f64, 20., 30., 40., 50., 60.]).unwrap()),
            Ok(Array::vector(vec![20_f64, 50.]).unwrap()),
        );
    }

    #[test]
    fn test_indexed_get_staging_array_ir() {
        let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(input, indices)| input.at(&index![.., &indices]).get(&GatherOptions::new()),
            (
                ArrayIrType::Array(ArrayType::new_static(DataType::F64, [2, 3])),
                ArrayIrType::Array(ArrayType::new_static(DataType::I32, [2])),
            ),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2, 3], %1:i32[2] .
                let %2:i64[2] = convert_element_type [data_type=i64] %1
                    %3:i64[] = constant [value=0]
                    %4:i64[] = constant [value=3]
                    %5:bool[2] = compare [direction=LessThan] %2 %3
                    %6:i64[2] = add %2 %4
                    %7:i64[2] = select %5 %6 %2
                    %8:i64[2, 1] = reshape [shape=[2, 1]] %7
                    %9:f64[2, 2] = gather [
                        dimensions=(offset=[0], collapsed_slice=[1], start_index_map=[1], batching=[]),
                        slice_sizes=[2, 1],
                    ] %0 %8
                in (%9)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.interpret((
                ArrayIrValue::Array(Array::matrix(2, 3, vec![10_f64, 20., 30., 40., 50., 60.]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![-1_i32, 0]).unwrap()),
            )),
            Ok(ArrayIrValue::Array(Array::matrix(2, 2, vec![30_f64, 10., 60., 40.]).unwrap())),
        );
    }

    #[test]
    fn test_indexed_get_staging_symbolic() {
        let rows = DimensionVariable::new("rows", DimensionBounds::new(0, Some(6)).unwrap());
        let queries = DimensionVariable::new("queries", DimensionBounds::new(0, Some(4)).unwrap());
        let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(input, indices)| input.at(&index![.., &indices]).get(&GatherOptions::new().with_mode(GatherMode::Clip)),
            (
                ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(rows), 3.into()]))),
                ArrayIrType::Array(ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(queries)]))),
            ),
        )
        .unwrap();
        assert_eq!(
            program.interpret((
                ArrayIrValue::Array(Array::matrix(2, 3, vec![10_f64, 20., 30., 40., 50., 60.]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![-1_i32, 0]).unwrap()),
            )),
            Ok(ArrayIrValue::Array(Array::matrix(2, 2, vec![30_f64, 10., 60., 40.]).unwrap())),
        );
        assert_eq!(
            program.interpret((
                ArrayIrValue::Array(Array::matrix(0, 3, Vec::<f64>::new()).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![-1_i32, 0]).unwrap()),
            )),
            Ok(ArrayIrValue::Array(Array::matrix(0, 2, Vec::<f64>::new()).unwrap())),
        );
        assert_eq!(
            program.interpret((
                ArrayIrValue::Array(Array::matrix(2, 3, vec![10_f64, 20., 30., 40., 50., 60.]).unwrap()),
                ArrayIrValue::Array(Array::vector(Vec::<i32>::new()).unwrap()),
            )),
            Ok(ArrayIrValue::Array(Array::matrix(2, 0, Vec::<f64>::new()).unwrap())),
        );
    }

    #[test]
    fn test_indexed_get_staging_symbolic_selections() {
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(6)).unwrap());
        let input_type = ArrayIrType::Array(ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Static(3)]),
        ));
        let matrix = ArrayIrValue::Array(Array::matrix(2, 3, vec![10_f64, 20., 30., 40., 50., 60.]).unwrap());

        // A host integer index lifts a portable literal, counts from the end through the retained extent, and
        // gathers along the symbolic axis.
        let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |input| input.at(&index![-1, ..]).get(&GatherOptions::new()),
            input_type.clone(),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[rows, 3] .
                let %1:i64[] = constant [value=-1]
                    %2:dimension<rows ∈ [1, 6)> = dimension_size [axis=0] %0
                    %3:i64[] = dimension_to_scalar %2
                    %4:i64[] = transfer_to_memory [destination=Device] %3
                    %5:i64[] = constant [value=0]
                    %6:bool[] = compare [direction=LessThan] %1 %5
                    %7:i64[] = add %1 %4
                    %8:i64[] = select %6 %7 %1
                    %9:dimension<3> = constant [value=3]
                    %10:dimension<1> = constant [value=1]
                    %11:i64[3, 1] = broadcast [output_axes=[]] %8 %9 %10
                    %12:f64[3] = gather [
                        dimensions=(offset=[], collapsed_slice=[0], start_index_map=[0], batching=[(1, 0)]),
                        slice_sizes=[1, 1],
                    ] %0 %11
                in (%12)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.interpret(matrix.clone()),
            Ok(ArrayIrValue::Array(Array::vector(vec![40_f64, 50., 60.]).unwrap())),
        );

        // A reversed full slice composes with a query on another axis, and an inserted axis is restored last.
        let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(input, indices)| input.at(&index![..by - 1, &indices]).get(&GatherOptions::new()),
            (input_type.clone(), ArrayIrType::Array(ArrayType::new_static(DataType::I32, [2]))),
        )
        .unwrap();
        let indices = ArrayIrValue::Array(Array::vector(vec![-1_i32, 0]).unwrap());
        assert_eq!(
            program.interpret((matrix.clone(), indices.clone())),
            Ok(ArrayIrValue::Array(Array::matrix(2, 2, vec![60_f64, 40., 30., 10.]).unwrap())),
        );
        let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(input, indices)| input.at(&index![new_axis, .., &indices]).get(&GatherOptions::new()),
            (input_type.clone(), ArrayIrType::Array(ArrayType::new_static(DataType::I32, [2]))),
        )
        .unwrap();
        assert_eq!(
            program.interpret((matrix.clone(), indices.clone())),
            Ok(ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::F64, [1, 2, 2]), &[30_f64, 10., 60., 40.])
                    .unwrap()
            )),
        );

        // Bounds modes apply after normalization: an explicit fill replaces the out-of-range query.
        let fill =
            GatherOptions::new().with_mode(GatherMode::Fill { value: Some(Box::new(Array::scalar(-1_f64).unwrap())) });
        let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            move |(input, indices)| input.at(&index![.., &indices]).get(&fill),
            (input_type.clone(), ArrayIrType::Array(ArrayType::new_static(DataType::I32, [2]))),
        )
        .unwrap();
        assert_eq!(
            program.interpret((matrix.clone(), ArrayIrValue::Array(Array::vector(vec![-1_i32, 5]).unwrap()))),
            Ok(ArrayIrValue::Array(Array::matrix(2, 2, vec![30_f64, -1., 60., -1.]).unwrap())),
        );

        // An empty selector list is the identity on a symbolic input.
        let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |input| input.at(&index![]).get(&GatherOptions::new()),
            input_type,
        )
        .unwrap();
        assert!(program.instructions().is_empty());
        assert_eq!(program.interpret(matrix.clone()), Ok(matrix));
    }

    #[test]
    fn test_indexed_get_staging_symbolic_validation() {
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(6)).unwrap());
        let input_type = ArrayIrType::Array(ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Static(3)]),
        ));
        let indices_type = ArrayIrType::Array(ArrayType::new_static(DataType::I32, [2]));
        let trace = |selectors: fn(&ArrayIrTracer, &ArrayIrTracer) -> Result<ArrayIrTracer, ProgramError>,
                     indices_type: ArrayIrType| {
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
                move |(input, indices)| selectors(&input, &indices),
                (input_type.clone(), indices_type),
            )
            .map(|_| ())
        };

        // Only full forward or reverse slices, one indexed axis, and `i64`-representable host integers are supported.
        assert_eq!(
            trace(|input, _| input.at(&index![0.., ..]).get(&GatherOptions::new()), indices_type.clone()),
            Err(TypeError::invalid("symbolic indexing currently requires full slices with step `1` or `-1`").into()),
        );
        assert_eq!(
            trace(|input, _| input.at(&index![(i128::MAX), ..]).get(&GatherOptions::new()), indices_type.clone()),
            Err(TypeError::invalid("symbolic indexing requires host integer indices representable as `i64`").into()),
        );
        assert_eq!(
            trace(
                |input, indices| input.at(&index![indices, indices]).get(&GatherOptions::new()),
                indices_type.clone()
            ),
            Err(TypeError::invalid("symbolic indexing currently supports one indexed axis").into()),
        );

        // The in-bounds promise is checked against the largest admitted extent for host integers.
        assert_eq!(
            trace(|input, _| input.at(&index![9, ..]).get(&GatherOptions::new()), indices_type.clone()),
            Err(TypeError::invalid("host integer index is out of bounds under `PromiseInBounds`").into()),
        );

        // Query arrays must be signed integers placed with the input, and masks need concrete geometry.
        assert_eq!(
            trace(
                |input, indices| input.at(&index![.., indices]).get(&GatherOptions::new()),
                ArrayIrType::Array(ArrayType::new_static(DataType::U32, [2])),
            ),
            Err(TypeError::invalid("symbolic indexing requires signed integer query arrays").into()),
        );
        assert_eq!(
            trace(
                |input, indices| input.at(&index![.., indices]).get(&GatherOptions::new()),
                ArrayIrType::Array(
                    ArrayType::new_static(DataType::I32, [2]).with_memory(Memory::Host { pinned: true }),
                ),
            ),
            Err(TypeError::invalid("index arrays and input must share one memory space").into()),
        );
        assert_eq!(
            trace(
                |input, _| {
                    let mask = IndexMask::new(vec![3], vec![true, false, true])?;
                    input.at(&index![.., &mask]).get(&GatherOptions::new())
                },
                indices_type.clone(),
            ),
            Err(TypeError::invalid("index masks require a concrete input shape in symbolic indexing").into()),
        );

        // Selector-list structure errors share the concrete frontend's wording.
        assert_eq!(
            trace(|input, _| input.at(&index![0, 0, 0]).get(&GatherOptions::new()), indices_type.clone()),
            Err(TypeError::invalid("index selection consumes 3 axes but input rank is 2").into()),
        );
        assert_eq!(
            trace(|input, _| input.at(&index![..., ...]).get(&GatherOptions::new()), indices_type.clone()),
            Err(TypeError::invalid("index selection contains more than one ellipsis").into()),
        );

        // Explicit placement and index promises are not remapped by the symbolic frontend yet.
        assert_eq!(
            trace(
                |input, indices| input
                    .at(&index![.., indices])
                    .get(&GatherOptions::new().with_indices_are_sorted(true)),
                indices_type.clone(),
            ),
            Err(TypeError::invalid(
                "symbolic indexing does not yet support explicit output sharding or index promises"
            )
            .into()),
        );
        assert_eq!(
            trace(
                |input, indices| {
                    let updates = ValueProjection::<ArrayType>::into_projected(indices.clone())?
                        .convert_element_type(DataType::F64)?;
                    let updates = <ArrayIrTracer as ValueProjection<ArrayType>>::from_projected(updates);
                    input.at(&index![.., indices]).set(&updates, &ScatterOptions::new().with_unique_indices(true))
                },
                indices_type,
            ),
            Err(TypeError::invalid(
                "symbolic indexing does not yet support explicit output sharding or index promises"
            )
            .into()),
        );
    }

    #[test]
    fn test_indexed_get_batching() {
        let output = batch(
            |input| input.at(&index![..by - 1]).get(&GatherOptions::new()),
            Array::matrix(2, 3, vec![1_i32, 2, 3, 4, 5, 6]).unwrap(),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(output, Array::matrix(2, 3, vec![3_i32, 2, 1, 6, 5, 4]).unwrap());
    }

    #[test]
    fn test_indexed_get_differentiation() {
        // Duplicate queries select the same tangent twice in forward mode.
        let (value, tangent) = differentiate_at(Array::vector(vec![10_f64, 20., 30.]).unwrap())
            .jvp(Array::vector(vec![2_f64, 3., 5.]).unwrap(), |input| {
                let indices = input.dispatch_domain().lift(Array::vector(vec![2_i32, 0, 2]).unwrap())?;
                input.at(&index![&indices]).get(&GatherOptions::new())
            })
            .unwrap();
        assert_eq!(value, Array::vector(vec![30_f64, 10., 30.]).unwrap());
        assert_eq!(tangent, Array::vector(vec![5_f64, 2., 5.]).unwrap());

        // Reverse mode sums contributions from every occurrence of a repeated query.
        let (value, pullback) = differentiate_at(Array::vector(vec![10_f64, 20., 30.]).unwrap())
            .vjp(|input| {
                let indices = input.dispatch_domain().lift(Array::vector(vec![2_i32, 0, 2]).unwrap())?;
                input.at(&index![&indices]).get(&GatherOptions::new())
            })
            .unwrap();
        assert_eq!(value, Array::vector(vec![30_f64, 10., 30.]).unwrap());
        assert_eq!(
            pullback.apply(Array::vector(vec![2_f64, 3., 5.]).unwrap()),
            Ok(Array::vector(vec![3_f64, 0., 7.]).unwrap()),
        );
    }

    #[test]
    fn test_indexed_get_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input = Array::from_elements(
            ArrayType::new_static(DataType::I32, [3, 4])
                .with_sharding(Sharding::replicated(mesh.clone(), 2))
                .unwrap(),
            &[0_i32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        )
        .unwrap();
        let rows = Array::from_elements(
            ArrayType::new_static(DataType::I32, [2, 1])
                .with_sharding(
                    Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated])
                        .unwrap(),
                )
                .unwrap(),
            &[0_i32, 2],
        )
        .unwrap();
        let columns = Array::matrix(1, 2, vec![1_i32, 3]).unwrap();
        let requested = Sharding::new(
            mesh,
            vec![ShardingDimension::Replicated, ShardingDimension::sharded(["x"]), ShardingDimension::Replicated],
        )
        .unwrap();
        // Query arrays broadcast to [2, 2]; the inserted leading axis must not shift the requested query placement.
        let output = input
            .at(&index![new_axis, &rows, &columns])
            .get(&GatherOptions::new().with_output_sharding(requested.clone()))
            .unwrap();
        assert_eq!(
            output,
            Array::from_elements(
                ArrayType::new_static(DataType::I32, [1, 2, 2]).with_sharding(requested).unwrap(),
                &[1_i32, 3, 9, 11],
            )
            .unwrap()
        );
    }

    #[test]
    fn test_indexed_get_memory() {
        let memory = Memory::Host { pinned: true };
        let input =
            Array::from_elements(ArrayType::new_static(DataType::I32, [3]).with_memory(memory), &[10_i32, 20, 30])
                .unwrap();
        let indices =
            Array::from_elements(ArrayType::new_static(DataType::I32, [2]).with_memory(memory), &[-1_i32, 0]).unwrap();
        // Generated zero and extent literals must share the query/input memory before index normalization.
        assert_eq!(
            input.at(&index![&indices]).get(&GatherOptions::new()),
            Ok(Array::from_elements(ArrayType::new_static(DataType::I32, [2]).with_memory(memory), &[30_i32, 10],)
                .unwrap())
        );
    }

    #[test]
    fn test_indexed_get_empty_placement_validation() {
        let input = Array::from_elements::<f64>(
            ArrayType::new_static(DataType::F64, [0]).with_memory(Memory::Host { pinned: true }),
            &[],
        )
        .unwrap();
        let indices = Array::vector(Vec::<i32>::new()).unwrap();
        assert!(matches!(
            input.at(&index![&indices]).get(&GatherOptions::new()),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "index arrays and input must share one memory space",
        ));

        let input_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let query_mesh = LogicalMesh::new(vec![MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input = Array::from_elements::<f64>(
            ArrayType::new_static(DataType::F64, [0])
                .with_sharding(Sharding::replicated(input_mesh, 1))
                .unwrap(),
            &[],
        )
        .unwrap();
        let indices = Array::from_elements::<i32>(
            ArrayType::new_static(DataType::I32, [0])
                .with_sharding(Sharding::replicated(query_mesh, 1))
                .unwrap(),
            &[],
        )
        .unwrap();
        assert!(matches!(
            input.at(&index![&indices]).get(&GatherOptions::new()),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`gather` input and indices shardings must use the same mesh",
        ));
    }

    #[test]
    fn test_indexed_get_partial_evaluation() {
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |(input, indices)| input.at(&index![&indices]).get(&GatherOptions::new()),
            (ArrayType::new_static(DataType::F64, [3]), ArrayType::new_static(DataType::I32, [2])),
        )
        .unwrap();
        let program = program.into_flat_program();
        let input = Array::vector(vec![10_f64, 20., 30.]).unwrap();
        let indices = Array::vector(vec![-1_i32, 0]).unwrap();
        let expected = Array::vector(vec![30_f64, 10.]).unwrap();
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();

        let known = program
            .partially_evaluate(&[PartialValue::Known(input.clone()), PartialValue::Known(indices.clone())])
            .unwrap();
        assert!(known.program().instructions().is_empty());
        assert_eq!(known.interpret(&context, &[]), Ok(vec![expected.clone()]));

        // Known query normalization folds, while the unknown source remains a residual gather input.
        let residual = program
            .partially_evaluate(&[PartialValue::Unknown(input.r#type().into_owned()), PartialValue::Known(indices)])
            .unwrap();
        assert!(residual.outputs()[0].is_unknown());
        assert_eq!(residual.interpret(&context, &[input]), Ok(vec![expected]));
    }

    #[test]
    fn test_indexed_get_batching_nested() {
        let input = Array::from_elements(
            ArrayType::new_static(DataType::F64, [2, 2, 3]),
            &[0_f64, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        )
        .unwrap();
        let output = batch(
            |matrix| {
                batch(
                    |row| {
                        let indices = row.dispatch_domain().lift(Array::vector(vec![2_i32, 0]).unwrap())?;
                        row.at(&index![&indices]).get(&GatherOptions::new())
                    },
                    matrix,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )
                .map_err(ProgramError::from)
            },
            input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(
            output,
            Array::from_elements(
                ArrayType::new_static(DataType::F64, [2, 2, 2]),
                &[2_f64, 0., 5., 3., 8., 6., 11., 9.],
            )
            .unwrap()
        );
    }

    #[test]
    fn test_indexed_set() {
        let input = Array::matrix(2, 3, vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let updates = Array::scalar(9_i32).unwrap();
        assert_eq!(
            input.at(&index![.., 1..3]).set(&updates, &ScatterOptions::new()),
            Array::matrix(2, 3, vec![0_i32, 9, 9, 3, 9, 9])
        );
        let indices = Array::vector(vec![0_i32, 2]).unwrap();
        assert_eq!(
            input.at(&index![1, &indices]).set(&Array::vector(vec![7_i32, 8]).unwrap(), &ScatterOptions::new()),
            Array::matrix(2, 3, vec![0_i32, 1, 2, 7, 4, 8])
        );
        assert_eq!(input, Array::matrix(2, 3, vec![0_i32, 1, 2, 3, 4, 5]).unwrap());
    }

    #[test]
    fn test_indexed_set_reversed_and_masked() {
        let input = Array::vector(vec![0_i32, 1, 2, 3, 4]).unwrap();
        assert_eq!(
            input.at(&index![..by - 2]).set(&Array::vector(vec![7_i32, 8, 9]).unwrap(), &ScatterOptions::new()),
            Array::vector(vec![9_i32, 1, 8, 3, 7])
        );
        let mask = IndexMask::new(vec![5], vec![false, true, false, true, false]).unwrap();
        assert_eq!(
            input.at(&index![&mask]).set(&Array::scalar(6_i32).unwrap(), &ScatterOptions::new()),
            Array::vector(vec![0_i32, 6, 2, 6, 4])
        );
        let invalid = Array::vector(vec![-6_i32, 5]).unwrap();
        assert_eq!(
            input
                .at(&index![&invalid])
                .set(&Array::scalar(9_i32).unwrap(), &ScatterOptions::new().with_mode(ScatterMode::Drop)),
            Ok(input)
        );
    }

    #[test]
    fn test_indexed_set_empty_axis() {
        let input = Array::vector(Vec::<i32>::new()).unwrap();
        let indices = Array::vector(vec![0_i32]).unwrap();
        let updates = Array::scalar(7_i32).unwrap();
        assert_eq!(
            input.at(&index![&indices]).set(&updates, &ScatterOptions::new().with_mode(ScatterMode::Drop)),
            Ok(input.clone()),
        );
        assert_eq!(
            input.at(&index![&indices]).set(&updates, &ScatterOptions::new().with_mode(ScatterMode::Clip)),
            Ok(input.clone()),
        );
        assert_eq!(input.at(&index![..]).set(&updates, &ScatterOptions::new()), Ok(input),);
    }

    #[test]
    fn test_indexed_add() {
        let input = Array::vector(vec![10_i32, 20, 30]).unwrap();
        let indices = Array::vector(vec![1_i32, 1, -1]).unwrap();
        let updates = Array::vector(vec![2_i32, 3, 4]).unwrap();
        assert_eq!(
            input.at(&index![&indices]).add(&updates, &ScatterOptions::new()),
            Array::vector(vec![10_i32, 25, 34])
        );
    }

    #[test]
    fn test_indexed_add_staging_symbolic() {
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(6)).unwrap());
        let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(input, indices, updates)| input.at(&index![.., &indices]).add(&updates, &ScatterOptions::new()),
            (
                ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(rows), 3.into()]))),
                ArrayIrType::Array(ArrayType::new_static(DataType::I32, [2])),
                ArrayIrType::Array(ArrayType::scalar(DataType::F64)),
            ),
        )
        .unwrap();
        assert_eq!(
            program.interpret((
                ArrayIrValue::Array(Array::matrix(2, 3, vec![10_f64, 20., 30., 40., 50., 60.]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![-1_i32, 0]).unwrap()),
                ArrayIrValue::Array(Array::scalar(5_f64).unwrap()),
            )),
            Ok(ArrayIrValue::Array(Array::matrix(2, 3, vec![15_f64, 20., 35., 45., 50., 65.]).unwrap())),
        );
    }

    #[test]
    fn test_indexed_set_staging_symbolic() {
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(6)).unwrap());
        let input_type = ArrayIrType::Array(ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(rows), Dimension::Static(3)]),
        ));
        let matrix = ArrayIrValue::Array(Array::matrix(2, 3, vec![10_f64, 20., 30., 40., 50., 60.]).unwrap());
        let indices = ArrayIrValue::Array(Array::vector(vec![-1_i32, 0]).unwrap());

        // A reversed axis is reversed before the scatter and reversed back afterwards, so the result keeps the input
        // geometry while the updates land in selection order.
        let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(input, indices, updates)| input.at(&index![..by - 1, &indices]).set(&updates, &ScatterOptions::new()),
            (
                input_type.clone(),
                ArrayIrType::Array(ArrayType::new_static(DataType::I32, [2])),
                ArrayIrType::Array(ArrayType::new_static(DataType::F64, [2])),
            ),
        )
        .unwrap();
        assert_eq!(
            program.interpret((
                matrix.clone(),
                indices.clone(),
                ArrayIrValue::Array(Array::vector(vec![7_f64, 8.]).unwrap()),
            )),
            Ok(ArrayIrValue::Array(Array::matrix(2, 3, vec![8_f64, 20., 7., 8., 50., 7.]).unwrap())),
        );
        let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(input, indices, updates)| input.at(&index![..by - 1, &indices]).max(&updates, &ScatterOptions::new()),
            (
                input_type.clone(),
                ArrayIrType::Array(ArrayType::new_static(DataType::I32, [2])),
                ArrayIrType::Array(ArrayType::new_static(DataType::F64, [2])),
            ),
        )
        .unwrap();
        assert_eq!(
            program.interpret((
                matrix.clone(),
                indices.clone(),
                ArrayIrValue::Array(Array::vector(vec![70_f64, 80.]).unwrap()),
            )),
            Ok(ArrayIrValue::Array(Array::matrix(2, 3, vec![80_f64, 20., 70., 80., 50., 70.]).unwrap())),
        );

        // Without a query, the whole array is updated through a zero-width index vector.
        let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(input, updates)| input.at(&index![..by - 1]).set(&updates, &ScatterOptions::new()),
            (input_type.clone(), ArrayIrType::Array(ArrayType::scalar(DataType::F64))),
        )
        .unwrap();
        assert_eq!(
            program.interpret((matrix.clone(), ArrayIrValue::Array(Array::scalar(5_f64).unwrap()))),
            Ok(ArrayIrValue::Array(Array::matrix(2, 3, vec![5_f64; 6]).unwrap())),
        );

        // An inserted axis is broadcast into the updates and removed again before the scatter.
        let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(input, indices, updates)| input.at(&index![new_axis, .., &indices]).set(&updates, &ScatterOptions::new()),
            (
                input_type,
                ArrayIrType::Array(ArrayType::new_static(DataType::I32, [2])),
                ArrayIrType::Array(ArrayType::scalar(DataType::F64)),
            ),
        )
        .unwrap();
        assert_eq!(
            program.interpret((matrix, indices, ArrayIrValue::Array(Array::scalar(5_f64).unwrap()))),
            Ok(ArrayIrValue::Array(Array::matrix(2, 3, vec![5_f64, 20., 5., 5., 50., 5.]).unwrap())),
        );
    }

    #[test]
    fn test_indexed_add_reduction_state() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let input = Array::from_elements(
            ArrayType::new_static(DataType::F64, [3]).with_sharding(sharding.clone()).unwrap(),
            &[10_f64, 20., 30.],
        )
        .unwrap();
        let indices = Array::vector(vec![2_i32, 0]).unwrap();
        let updates = Array::from_elements(
            ArrayType::new_static(DataType::F64, [2]).with_sharding(sharding.clone()).unwrap(),
            &[5_f64, 7.],
        )
        .unwrap();
        assert_eq!(
            input.at(&index![&indices]).get(&GatherOptions::new()),
            Ok(Array::from_elements(
                ArrayType::new_static(DataType::F64, [2]).with_sharding(sharding.clone()).unwrap(),
                &[30_f64, 10.],
            )
            .unwrap())
        );
        assert_eq!(
            input.at(&index![&indices]).add(&updates, &ScatterOptions::new()),
            Ok(Array::from_elements(
                ArrayType::new_static(DataType::F64, [3]).with_sharding(sharding).unwrap(),
                &[17_f64, 20., 35.],
            )
            .unwrap())
        );
    }

    #[test]
    fn test_indexed_add_differentiation() {
        // Differentiating only the updates keeps the source constant and sums tangents for duplicate destinations.
        let (value, tangent) = differentiate_at(Array::vector(vec![2_f64, 3.]).unwrap())
            .jvp(Array::vector(vec![5_f64, 7.]).unwrap(), |updates| {
                let input = updates.dispatch_domain().lift(Array::vector(vec![10_f64, 20., 30.]).unwrap())?;
                let indices = updates.dispatch_domain().lift(Array::vector(vec![1_i32, 1]).unwrap())?;
                input.at(&index![&indices]).add(&updates, &ScatterOptions::new())
            })
            .unwrap();
        assert_eq!(value, Array::vector(vec![10_f64, 25., 30.]).unwrap());
        assert_eq!(tangent, Array::vector(vec![0_f64, 12., 0.]).unwrap());
    }

    #[test]
    fn test_indexed_multiply() {
        let input = Array::vector(vec![10_i32, 20, 30]).unwrap();
        let indices = Array::vector(vec![1_i32, 1, -1]).unwrap();
        let updates = Array::vector(vec![2_i32, 3, 4]).unwrap();
        assert_eq!(
            input.at(&index![&indices]).multiply(&updates, &ScatterOptions::new()),
            Array::vector(vec![10_i32, 120, 120])
        );
    }

    #[test]
    fn test_indexed_min() {
        let input = Array::vector(vec![10_i32, 20, 30]).unwrap();
        let indices = Array::vector(vec![1_i32, 1, -1]).unwrap();
        let updates = Array::vector(vec![2_i32, 3, 4]).unwrap();
        assert_eq!(
            input.at(&index![&indices]).min(&updates, &ScatterOptions::new()),
            Array::vector(vec![10_i32, 2, 4])
        );
    }

    #[test]
    fn test_indexed_max() {
        let input = Array::vector(vec![10_i32, 20, 30]).unwrap();
        let indices = Array::vector(vec![1_i32, 1, -1]).unwrap();
        let updates = Array::vector(vec![2_i32, 3, 4]).unwrap();
        assert_eq!(
            input.at(&index![&indices]).max(&updates, &ScatterOptions::new()),
            Array::vector(vec![10_i32, 20, 30])
        );
    }

    #[test]
    fn test_index() {
        let selection: [IndexSelector<'_, Array>; 9] =
            index![1..9 by 2, .. by -1, ..., new_axis, -1, ..3, 2.., .., ..-1 by -1];
        assert_eq!(
            selection,
            [
                IndexSelector::Basic(BasicIndex::Slice(IndexSlice::new(Some(1), Some(9), 2))),
                IndexSelector::Basic(BasicIndex::Slice(IndexSlice::new(None, None, -1))),
                IndexSelector::Basic(BasicIndex::Ellipsis),
                IndexSelector::Basic(BasicIndex::NewAxis),
                IndexSelector::Basic(BasicIndex::Integer(-1)),
                IndexSelector::Basic(BasicIndex::Slice(IndexSlice::new(None, Some(3), 1))),
                IndexSelector::Basic(BasicIndex::Slice(IndexSlice::new(Some(2), None, 1))),
                IndexSelector::Basic(BasicIndex::Slice(IndexSlice::new(None, None, 1))),
                IndexSelector::Basic(BasicIndex::Slice(IndexSlice::new(None, Some(-1), -1))),
            ],
        );
        let empty: [IndexSelector<'_, Array>; 0] = index![];
        assert_eq!(empty, []);
        let trailing: [IndexSelector<'_, Array>; 1] = index![1,];
        assert_eq!(trailing, [IndexSelector::Basic(BasicIndex::Integer(1))]);
    }

    #[test]
    fn test_index_expressions() {
        // Bounds and strides retain ordinary expression semantics and evaluate once from left to right.
        let mut evaluations = Vec::new();
        let selection: [IndexSelector<'_, Array>; 1] = index![{
            evaluations.push("start");
            1usize
        }..{
            evaluations.push("stop");
            9usize
        } by {
            evaluations.push("step");
            2isize
        }];
        assert_eq!(evaluations, ["start", "stop", "step"]);
        assert_eq!(selection, [IndexSelector::Basic(BasicIndex::Slice(IndexSlice::new(Some(1), Some(9), 2)))],);
        let grouped: [IndexSelector<'_, Array>; 1] = index![(|first, second| first + second)(1, 2)..9 by 2];
        assert_eq!(grouped, [IndexSelector::Basic(BasicIndex::Slice(IndexSlice::new(Some(3), Some(9), 2)))],);
    }

    #[test]
    fn test_index_array_borrow() {
        let rows = Array::vector(vec![0_i32, 2]).unwrap();
        let selection = index![&rows, ...];
        assert_eq!(selection, [IndexSelector::Array(&rows), IndexSelector::Basic(BasicIndex::Ellipsis)],);
        // Constructing descriptors keeps the original array available to the caller.
        assert_eq!(rows, Array::vector(vec![0_i32, 2]).unwrap());
    }
}
