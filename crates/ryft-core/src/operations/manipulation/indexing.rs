use std::marker::PhantomData;
use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

use crate::arrays::{Array, ArrayIrType, ArraySliceAxis, ArrayType, Broadcastable, DataType, Dimension, Shape};
use crate::contexts::Context;
use crate::macros::check_count;
use crate::operations::arithmetic::Add;
use crate::operations::comparisons::Compare;
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
use crate::operations::references::{
    ReferenceAddUpdate, ReferenceDynamicIndex, ReferenceIndex, ReferenceRead, ReferenceSlice, ReferenceSwap,
    ReferenceWrite,
};
use crate::programs::{ProgramError, ReferenceType, Type, TypeError, Typed, Value, ValueProjection};

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
            #[inline]
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

impl IndexSlice {
    /// Creates a new signed [`IndexSlice`] with the provided optional endpoints and step. Endpoints are exclusive
    /// at the stop and are clipped to the axis when normalized. This constructor preserves omitted endpoints so that
    /// reverse slicing can distinguish an omitted stop from an explicit negative index, and it accepts any step so that
    /// the [`index!`] macro stays infallible; a zero step is rejected when the slice is normalized against an axis.
    #[inline]
    pub fn new(start: Option<i128>, stop: Option<i128>, step: i128) -> Self {
        Self { start, stop, step }
    }

    /// Returns the inclusive start index for this [`IndexSlice`], or [`None`] to start at the first position
    /// in the traversal direction.
    #[inline]
    pub fn start(&self) -> Option<i128> {
        self.start
    }

    /// Returns the exclusive stop index for this [`IndexSlice`], or [`None`] to continue through the end
    /// in the traversal direction.
    #[inline]
    pub fn stop(&self) -> Option<i128> {
        self.stop
    }

    /// Returns the signed distance between selected positions for this [`IndexSlice`]. Positive values traverse
    /// forward, negative values traverse backward, and zero is invalid. Omitting the step in indexing syntax uses `1`.
    #[inline]
    pub fn step(&self) -> i128 {
        self.step
    }

    /// Resolves signed endpoints against an axis extent, producing a positive-stride slice. For a negative step,
    /// the coordinates refer to the reversed input axis. Endpoint clipping and unsigned step magnitudes avoid
    /// overflow even for [`i128::MIN`], and empty intervals remain valid empty slices.
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

/// Positive-stride coordinates for a normalized [`IndexSlice`], optionally applied after reversing its input axis.
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

/// Basic index that is a host-known selector that acts on one axis at a time without any broadcasting.
/// Basic indexing is one of the two categories of [`IndexSelector`] that is inspired by
/// [NumPy's basic indexing](https://numpy.org/doc/stable/user/basics.indexing.html#basic-indexing). The other is
/// inspired by [NumPy's advanced indexing](https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing),
/// consists of the [`Array`](IndexSelector::Array) and [`Mask`](IndexSelector::Mask) selectors, and has no dedicated
/// type because its two members carry unrelated payloads (i.e., a borrowed value and a host-known [`IndexMask`]). An
/// integer or slice consumes one input axis, where integers remove the axis and slices preserve it,
/// [`NewAxis`](Self::NewAxis) inserts an output axis of extent one without consuming an input axis, and
/// [`Ellipsis`](Self::Ellipsis) expands to full slices over the axes not explicitly selected. Basic indices leave
/// the relative order of the axes they touch unchanged, whereas advanced indices broadcast jointly and may move their
/// result axes to the front. Refer to the documentation of [`IndexSelector`] for how the two categories combine
/// in one selection.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BasicIndex {
    /// A signed scalar index. Negative indices count backward from the axis extent.
    Integer(i128),

    /// A signed, potentially strided slice that preserves the selected axis.
    Slice(IndexSlice),

    /// Index that inserts a new axis of extent one without consuming an input axis.
    NewAxis,

    /// Index that expands to full slices over the otherwise unspecified input axes.
    Ellipsis,
}

/// A host-known Boolean mask with explicit shape, which is the Boolean form of advanced indexing
/// (the other form is an integer [`Array`](IndexSelector::Array) selector), following NumPy's
/// [Boolean array indexing](https://numpy.org/doc/stable/user/basics.indexing.html#boolean-array-indexing).
/// Constructing it validates the number of entries. Indexing converts its true positions into constant integer
/// coordinates, one advanced integer index per mask axis, exactly as NumPy replaces a mask with the arrays returned
/// by [`numpy.nonzero`](https://numpy.org/doc/stable/reference/generated/numpy.nonzero.html); it never reads a device
/// value or tracer back to the host. A scalar mask inserts an advanced axis of size one (i.e., `true`) or zero
/// (i.e., `false`) without consuming an input axis. Boolean indexing requires this concrete descriptor; runtime
/// Boolean compaction and host reads of device or traced masks are unsupported.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexMask {
    /// Refer to the documentation of [`shape`](Self::shape) for more information.
    shape: Vec<usize>,

    /// Refer to the documentation of [`values`](Self::values) for more information.
    values: Vec<bool>,
}

impl IndexMask {
    /// Creates a new concrete [`IndexMask`] with row-major `values` and the specified axis extents. The product of the
    /// extents must equal the number of values and fit in `usize`. A scalar mask has an empty shape and exactly one
    /// value.
    ///
    /// # Parameters
    ///
    ///   - `shape`: Extents of the consecutive input axes consumed by this mask.
    ///   - `values`: Host-known Boolean entries in row-major order, each `true` entry selects its position.
    pub fn new(shape: Vec<usize>, values: Vec<bool>) -> Result<Self, ProgramError> {
        let count = shape
            .iter()
            .try_fold(1usize, |count, extent| count.checked_mul(*extent))
            .ok_or_else(|| TypeError::invalid("index mask shape overflows `usize`"))?;
        if count != values.len() {
            return Err(TypeError::invalid(format!(
                "index mask shape requires {} values but got {}",
                count,
                values.len()
            ))
            .into());
        }
        Ok(Self { shape, values })
    }

    /// Returns the extents of the input axes consumed by this [`IndexMask`]. An empty shape denotes a scalar mask.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the concrete Boolean entries of this [`IndexMask`] in row-major order.
    #[inline]
    pub fn values(&self) -> &[bool] {
        &self.values
    }
}

/// One component of an array selection. Use [`index!`] or the standard conversions to construct lists of such
/// selectors. The value parameter is inferred from the receiver of [`Indexing::at`], including for a list containing
/// only basic indices. Array selectors borrow the receiver's value family and so the caller must lift constants into
/// the trace before using them.
///
/// Selectors fall into NumPy's two [indexing](https://numpy.org/doc/stable/user/basics.indexing.html) categories.
/// [`Basic`](Self::Basic) selectors are host-known and act on one axis at a time (refer to the documentation of
/// [`BasicIndex`] for more information). [`Array`](Self::Array) and [`Mask`](Self::Mask) selectors are _advanced_
/// indices. All advanced selectors in one selection broadcast jointly to a single coordinate shape, and the selection
/// gathers (or scatters) at those coordinates. When a selection mixes the two categories, the basic selectors are
/// applied axis by axis around the advanced gather, and host integers count as advanced for the purpose of axis
/// placement, as in NumPy's rules for [combining advanced and basic indexing](
/// https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing). Advanced selectors
/// that are adjacent in the selection insert their broadcast axes in place of the first consumed axis, while advanced
/// selectors separated by a slice, new axis, or ellipsis move their broadcast axes to the front of the result.
///
/// Negative integer indices count backward from the axis end once. Remaining invalid integer indices follow the bounds
/// options supplied to the read or update function.
#[derive(Clone, Debug, PartialEq)]
pub enum IndexSelector<'i, V: Value> {
    /// Basic index represented as a host integer, a slice, a new axis, or an ellipsis.
    Basic(BasicIndex),

    /// Advanced index represented as an integer array of coordinates, including rank-zero arrays. Boolean arrays are
    /// rejected (callers must use [`IndexMask`] for concrete Boolean masks). Keeping rank-zero arrays distinct from
    /// host integers preserves advanced-index semantics (i.e., a rank-zero array contributes to the joint broadcast
    /// and does not remove the axis the way a host integer does).
    Array(&'i V),

    /// Advanced index represented as an explicitly host-known mask, consuming one input axis per mask axis and
    /// contributing one advanced axis whose extent is the number of true entries.
    Mask(&'i IndexMask),
}

impl<V: Value> From<BasicIndex> for IndexSelector<'_, V> {
    #[inline]
    fn from(value: BasicIndex) -> Self {
        Self::Basic(value)
    }
}

impl<V: Value> From<IndexSlice> for IndexSelector<'_, V> {
    #[inline]
    fn from(value: IndexSlice) -> Self {
        Self::Basic(BasicIndex::Slice(value))
    }
}

impl<'i, V: Value> From<&'i V> for IndexSelector<'i, V> {
    #[inline]
    fn from(value: &'i V) -> Self {
        Self::Array(value)
    }
}

impl<'i, V: Value> From<&'i IndexMask> for IndexSelector<'i, V> {
    #[inline]
    fn from(value: &'i IndexMask) -> Self {
        Self::Mask(value)
    }
}

macro_rules! impl_from_integer_for_index_selector {
    ($integer:ty) => {
        impl<V: Value> From<$integer> for IndexSelector<'_, V> {
            #[inline]
            fn from(value: $integer) -> Self {
                Self::Basic(BasicIndex::Integer(value.to_index_integer()))
            }
        }
    };
}

impl_from_integer_for_index_selector!(i8);
impl_from_integer_for_index_selector!(i16);
impl_from_integer_for_index_selector!(i32);
impl_from_integer_for_index_selector!(i64);
impl_from_integer_for_index_selector!(i128);
impl_from_integer_for_index_selector!(isize);
impl_from_integer_for_index_selector!(u8);
impl_from_integer_for_index_selector!(u16);
impl_from_integer_for_index_selector!(u32);
impl_from_integer_for_index_selector!(u64);
impl_from_integer_for_index_selector!(usize);

impl<I: IndexInteger, V: Value> From<Range<I>> for IndexSelector<'_, V> {
    #[inline]
    fn from(value: Range<I>) -> Self {
        IndexSlice::new(Some(value.start.to_index_integer()), Some(value.end.to_index_integer()), 1).into()
    }
}

impl<I: IndexInteger, V: Value> From<RangeFrom<I>> for IndexSelector<'_, V> {
    #[inline]
    fn from(value: RangeFrom<I>) -> Self {
        IndexSlice::new(Some(value.start.to_index_integer()), None, 1).into()
    }
}

impl<I: IndexInteger, V: Value> From<RangeTo<I>> for IndexSelector<'_, V> {
    #[inline]
    fn from(value: RangeTo<I>) -> Self {
        IndexSlice::new(None, Some(value.end.to_index_integer()), 1).into()
    }
}

impl<V: Value> From<RangeFull> for IndexSelector<'_, V> {
    #[inline]
    fn from(_: RangeFull) -> Self {
        IndexSlice::new(None, None, 1).into()
    }
}

/// Creates borrowed selections for reads and functional updates. The wrapper has no effects until a terminal
/// function is called, and its functions require only the capabilities needed for their direction.
///
/// # Example
///
/// ```rust
/// # use ryft_core::{Array, GatherMode, GatherOptions, Indexing, index};
/// let input = Array::matrix(3, 2, vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
/// let options = GatherOptions::new().with_mode(GatherMode::Clip);
/// let output = input.at(&index![.. by -1, 1]).get(&options).unwrap();
/// assert_eq!(output, Array::vector(vec![5_i32, 3, 1]).unwrap());
/// ```
pub trait Indexing: Value {
    /// Borrows this value and the provided [`IndexSelector`]s in an [`Indexed`] wrapper. For array values, its `get`
    /// function reads the selection while `set`, `add`, `mul`, `min`, and `max` return a new array, leaving the
    /// input unchanged. For reference values, `view` derives a reference to the selected region and `read`, `write`,
    /// `add_update`, and `swap` access the reference's state through that view in place. Validation occurs in those
    /// terminal functions, so constructing a wrapper is infallible. Refer to [`Indexed`] for supported geometry and
    /// the shared bounds and index-promise contracts.
    ///
    /// # Parameters
    ///
    ///   - `selectors`: Ordered [`IndexSelector`]s, usually constructed with [`index!`]. Omitted trailing axes are
    ///     selected in full. The wrapper borrows this list and its array indices until its last use.
    #[inline]
    fn at<'v, 's, 'i>(&'v self, selectors: &'s [IndexSelector<'i, Self>]) -> Indexed<'v, 's, 'i, Self> {
        Indexed { input: self, selectors, marker: PhantomData }
    }
}

impl<V: Value> Indexing for V {}

/// Borrowed selection of a value. Separate input, selector-list, and index-value lifetimes allow both reusable lists
/// and temporary lists used within a chained call. Selections of array values have value/functional semantics meaning
/// that reads produce values and updates return new values, leaving the input unchanged, exactly like `x.at[...]` in
/// [JAX](https://docs.jax.dev/en/latest/_autosummary/jax.Array.at.html). Selections of reference values instead derive
/// a reference view of the selected region and access the reference's state through it in place, exactly like
/// `ref.at[...]` in JAX.
///
/// # Supported Geometry
///
/// Concrete shapes support host integer indices, positive and negative strides, inserted axes, ellipses, broadcast
/// integer-array indices, and explicit host-known [`IndexMask`]s. Eager arrays and staged arrays use the same selection
/// rules. Reads compose slice, reverse, reshape, and gather, and updates compose broadcast, reshape, and scatter.
///
/// Arrays with symbolic shapes are selected as [`ArrayIrType`] values, a family that carries dimension values alongside
/// arrays so that runtime extents remain available while staging. Such selections support at most one indexed axis,
/// full forward or reverse slices on the other axes, inserted axes, and a symbolic query-array shape on the indexed
/// axis. When starting from an array tracer projected out of such a value, index the original value instead. General
/// symbolic slice bounds, several symbolic advanced indices, explicit output sharding, and index promises are
/// rejected. Untouched axes and query axes may have zero runtime extents, and symbolic reads impose one further
/// requirement on the indexed axis that is documented on their `get` function.
///
/// # Reference Views
///
/// A reference input (i.e., a value of [`ReferenceType`]) is selected without accessing its state: [`view`](Self::view)
/// derives a reference sharing the input's allocation, and [`read`](Self::read), [`write`](Self::write),
/// [`add_update`](Self::add_update), and [`swap`](Self::swap) go through that view. Reference views are limited to the
/// transforms the reference machinery can reconstruct (i.e., host integers, unit-stride forward slices, an ellipsis,
/// and scalar integer index arrays that select one position at run time). Inserted axes, masks, non-scalar index
/// arrays, and strided or reversed slices are rejected, and axes touched by host integers or slices must have static
/// extents. The bounds and index-promise options below do not apply to reference views, whose host integers must lie
/// within their axis after negative-index normalization.
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
pub struct Indexed<'v, 's, 'i, V: Value, T: Type = <V as Typed>::Type> {
    /// The input whose elements are read or functionally updated.
    input: &'v V,

    /// Borrowed list of host and value-level [`IndexSelector`]s.
    selectors: &'s [IndexSelector<'i, V>],

    /// [`PhantomData`] marker identifying the input's type universe.
    marker: PhantomData<fn() -> T>,
}

impl<
    V: Value<Type = ArrayType, ExecutionDomain: Context<Operation: From<ConstantOperation<Array>>>>
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
> Indexed<'_, '_, '_, V, ArrayType>
{
    /// Reads this selection. Invalid scalar/array indices follow `options` after negative-index normalization.
    /// Slices clip their endpoints independently. Floating-point fill literals preserve their original encodings.
    /// Refer to [`Indexed`] for supported geometry and the shared bounds and index-promise contracts. A host integer
    /// that remains out of bounds under [`GatherMode::PromiseInBounds`] is rejected before staging.
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

        if expanded.iter().all(|index| matches!(index, ExpandedIndexSelector::Basic(_)))
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
                    ExpandedIndexSelector::Basic(BasicIndex::NewAxis) => output.push(1),
                    ExpandedIndexSelector::Basic(BasicIndex::Ellipsis) => {}
                    ExpandedIndexSelector::Basic(BasicIndex::Slice(slice)) => {
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
                    ExpandedIndexSelector::Basic(BasicIndex::Integer(integer)) => {
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

            if plan.output_shape.iter().all(|&extent| extent != 0) && !matches!(options.mode(), GatherMode::Fill { .. })
            {
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

impl<
    V: Value<Type = ArrayType, ExecutionDomain: Context<Operation: From<ConstantOperation<Array>>>>
        + Broadcast
        + Reshape
        + Concatenate
        + ConvertElementType
        + Compare
        + Add
        + Select
        + Scatter,
> Indexed<'_, '_, '_, V, ArrayType>
{
    /// Returns a value with selected elements overwritten by `updates`. Conflicting repeated indices do not promise
    /// a deterministic winner. The input is unchanged, and updates broadcast to the selection shape.
    ///
    /// # Parameters
    ///
    ///   - `updates`: Values broadcast to the selected shape, with the same data type as the input.
    ///   - `options`: Bounds handling, output placement, and promises about the final normalized selected positions.
    ///     Sortedness is cleared before scatter while uniqueness remains an unchecked caller promise. Refer to
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
    #[inline]
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
    ///     Sortedness is cleared before scatter while uniqueness remains an unchecked caller promise. Refer to
    ///     [`Indexed`] for how normalization and clipping affect these promises.
    #[inline]
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
    ///     Sortedness is cleared before scatter while uniqueness remains an unchecked caller promise. Refer to
    ///     [`Indexed`] for how normalization and clipping affect these promises.
    #[inline]
    pub fn mul(&self, updates: &V, options: &ScatterOptions) -> Result<V, ProgramError> {
        self.update(updates, ScatterReductionKind::Mul, options)
    }

    /// Returns a value with selected updates combined by the elementwise minimum. Updates broadcast
    /// to the selection shape, and the input remains unchanged.
    ///
    /// # Parameters
    ///
    ///   - `updates`: Values with the input's data type, broadcast to the selection shape.
    ///   - `options`: Bounds handling, output placement, and promises about normalized selected positions.
    ///     Sortedness is cleared before scatter while uniqueness remains an unchecked caller promise. Refer to
    ///     [`Indexed`] for how normalization and clipping affect these promises.
    #[inline]
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
    ///     Sortedness is cleared before scatter while uniqueness remains an unchecked caller promise. Refer to
    ///     [`Indexed`] for how normalization and clipping affect these promises.
    #[inline]
    pub fn max(&self, updates: &V, options: &ScatterOptions) -> Result<V, ProgramError> {
        self.update(updates, ScatterReductionKind::Max, options)
    }

    /// Shared implementation of [`set`](Self::set), [`add`](Self::add), [`mul`](Self::mul), [`min`](Self::min), and
    /// [`max`](Self::max), which differ only in how `kind` combines the updates with the selected elements. It builds
    /// the same [`IndexPlan`] as [`get`](Self::get), so an update touches exactly the positions that a read of the same
    /// selection would return, and then scatters into the input. The updates are broadcast to the selection shape the
    /// caller sees, the inserted extent-one axes are reshaped away to obtain the scatter update shape, and the plan's
    /// gather dimension numbers are reused as scatter dimension numbers. The updates must already have the input's data
    /// type, because scatter does not promote. The sortedness promise is cleared, since normalization and interleaved
    /// slice coordinates may reorder the selected positions, while the uniqueness promise is passed through unchecked.
    /// The input is left unchanged.
    ///
    /// # Parameters
    ///
    ///   - `updates`: Values with the input's data type, broadcastable to the selection shape.
    ///   - `kind`: Scatter reduction combining each update with the element it lands on.
    ///   - `options`: Bounds handling, output placement, and index promises, as described on [`Indexed`].
    fn update(&self, updates: &V, kind: ScatterReductionKind, options: &ScatterOptions) -> Result<V, ProgramError> {
        if updates.r#type().data_type() != self.input.r#type().data_type() {
            return Err(TypeError::invalid("index updates must have the input data type").into());
        }

        let expanded = self.expanded()?;
        let plan = self.plan(&expanded, matches!(options.mode(), ScatterMode::PromiseInBounds))?;

        // The scatter update has the selection shape without the inserted axes, which consume no input axis.
        let update_shape = plan
            .output_shape
            .iter()
            .enumerate()
            .filter(|(axis, _)| !plan.new_axes.contains(axis))
            .map(|(_, &extent)| extent)
            .collect::<Vec<_>>();
        let updates = updates.broadcast_to(Shape::from(plan.output_shape))?.reshape(Shape::from(update_shape))?;

        let dimensions = ScatterDimensionNumbers::new(
            plan.dimensions.offset_dimensions().to_vec(),
            plan.dimensions.collapsed_slice_dimensions().to_vec(),
            plan.dimensions.start_index_map().to_vec(),
        );

        let options = options.clone().with_indices_are_sorted(false);
        self.input.scatter(&plan.indices, &updates, &dimensions, kind, &options)
    }
}

impl<V: Value<Type = ArrayType, ExecutionDomain: Context<Operation: From<ConstantOperation<Array>>>>>
    Indexed<'_, '_, '_, V, ArrayType>
{
    /// Lifts a host coordinate literal into the input's execution domain, placed in the input's memory space.
    /// A backend's stored constant family may contain capture handles, so coordinate literals use the ordinary
    /// constant operation payload instead, which compiled contexts lower directly.
    fn constant(&self, value: Array) -> Result<V, ProgramError> {
        let r#type = value.r#type().into_owned().with_memory(self.input.r#type().memory());
        let mut outputs = self.input.execution_domain().bind(
            ConstantOperation::new(Array::new(r#type, value.storage_bytes().to_vec())?),
            Vec::new(),
            &[],
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<
    V: Value<Type = ArrayType, ExecutionDomain: Context<Operation: From<ConstantOperation<Array>>>>
        + Broadcast
        + Reshape
        + Concatenate
        + ConvertElementType
        + Compare
        + Add
        + Select,
> Indexed<'_, '_, '_, V, ArrayType>
{
    /// Turns the caller's [`IndexSelector`] list into the explicit [`ExpandedIndexSelector`] list that
    /// [`plan`](Self::plan) consumes, validating the selection against the input on the way. It rejects a selection
    /// that consumes more axes than the input has or that contains more than one ellipsis, requires index arrays to be
    /// integer-typed and to share the input's memory space, and requires each mask to match the extents of the axes it
    /// consumes. The ellipsis (or, without one, the omitted trailing axes) is expanded into full slices over the
    /// unspecified axes, while the ellipsis entry itself is kept as a zero-width separator between advanced-index
    /// groups. Each non-scalar mask is replaced by one host coordinate array per mask axis holding the positions
    /// of its true entries, lifted as constants in the input's memory space, and a scalar mask becomes a
    /// [`Boolean`](ExpandedIndexSelector::Boolean) entry.
    fn expanded(&self) -> Result<Vec<ExpandedIndexSelector<V>>, ProgramError> {
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
                "index selection consumes {consumed} axes but input rank is {rank}",
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
                    // A zero-width ellipsis still separates two advanced groups.
                    ellipsis = true;
                    result.push(ExpandedIndexSelector::Basic(BasicIndex::Ellipsis));
                    for _ in 0..rank - consumed {
                        result.push(ExpandedIndexSelector::Basic(BasicIndex::Slice(IndexSlice::new(None, None, 1))));
                        axis += 1;
                    }
                }
                IndexSelector::Basic(value) => {
                    result.push(ExpandedIndexSelector::Basic(*value));
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

                    result.push(ExpandedIndexSelector::Array((*value).clone()));
                    axis += 1;
                }
                IndexSelector::Mask(mask) => {
                    if mask.shape.is_empty() {
                        result.push(ExpandedIndexSelector::Boolean(mask.values[0]));
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
                        // No coordinate is decoded in that case. For non-empty masks the constructor has checked
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
                        result.push(ExpandedIndexSelector::Array(self.constant(Array::vector(coordinates)?)?));
                    }

                    axis += mask.shape.len();
                }
            }
        }

        if !ellipsis {
            for _ in consumed..rank {
                result.push(ExpandedIndexSelector::Basic(BasicIndex::Slice(IndexSlice::new(None, None, 1))));
            }
        }

        Ok(result)
    }

    /// Builds a shared window/query mapping, preserving contiguous windows rather than constructing a full grid
    /// over every output element. Only genuinely strided slices add coordinate-query axes.
    fn plan(&self, expanded: &[ExpandedIndexSelector<V>], promise: bool) -> Result<IndexPlan<V>, ProgramError> {
        let input_type = self.input.r#type();
        let shape = input_type
            .shape()
            .dimensions()
            .iter()
            .map(|dimension| {
                dimension.value().ok_or_else(|| ProgramError::UnsupportedOperation {
                    message: "general indexing requires concrete extents; index an array with a symbolic shape as an \
                         `ArrayIrType` value instead"
                        .into(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        if shape.iter().any(|&extent| i64::try_from(extent).is_err()) {
            return Err(TypeError::invalid("indexed axis extent exceeds `i64::MAX`").into());
        }

        let advanced = expanded
            .iter()
            .any(|index| matches!(index, ExpandedIndexSelector::Array(_) | ExpandedIndexSelector::Boolean(_)));

        // Host integers join the advanced group only when an array or mask selector is present.
        let positions = expanded
            .iter()
            .enumerate()
            .filter(|(_, index)| match index {
                ExpandedIndexSelector::Array(_) | ExpandedIndexSelector::Boolean(_) => true,
                ExpandedIndexSelector::Basic(BasicIndex::Integer(_)) => advanced,
                ExpandedIndexSelector::Basic(_) => false,
            })
            .map(|(position, _)| position)
            .collect::<Vec<_>>();
        let contiguous = positions.windows(2).all(|pair| pair[1] == pair[0] + 1);

        let mut broadcast_shape = Shape::new(vec![]);
        for index in expanded {
            let next = match index {
                ExpandedIndexSelector::Array(value) => Some(value.r#type().shape().clone()),
                ExpandedIndexSelector::Boolean(value) => Some(Shape::new(vec![Dimension::Static(usize::from(*value))])),
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
                ExpandedIndexSelector::Basic(BasicIndex::NewAxis) => output.push(OutputAxis::New),
                ExpandedIndexSelector::Basic(BasicIndex::Ellipsis) | ExpandedIndexSelector::Boolean(_) => {}
                ExpandedIndexSelector::Basic(BasicIndex::Integer(integer)) => {
                    let integer = if *integer < 0 { integer.saturating_add(shape[axis] as i128) } else { *integer };
                    if promise && (integer < 0 || integer >= shape[axis] as i128) {
                        return Err(TypeError::invalid(format!(
                            "index {} is out of bounds for axis {} with extent {} under `PromiseInBounds`",
                            integer, axis, shape[axis]
                        ))
                        .into());
                    }
                    let integer = integer.clamp(i64::MIN as i128, i64::MAX as i128) as i64;
                    components.push((axis, self.constant(Array::scalar(integer)?)?, None));
                    collapsed.push(axis);
                    axis += 1;
                }
                ExpandedIndexSelector::Array(value) => {
                    // Convert the query to `i64` without wrapping large unsigned values into valid signed positions,
                    // then add the axis extent once to negative indices. Values still outside the axis are left to the
                    // requested bounds policy.
                    let extent = i64::try_from(shape[axis])
                        .map_err(|_| TypeError::invalid("indexed axis extent exceeds `i64::MAX`"))?;
                    let indices = if value.r#type().data_type() == DataType::U64 {
                        let maximum = self.constant(Array::scalar(i64::MAX as u64)?)?;
                        V::select(&value.greater_than(&maximum)?, &maximum, value)?
                            .convert_element_type(DataType::I64)?
                    } else {
                        value.convert_element_type(DataType::I64)?
                    };
                    let zero = self.constant(Array::scalar(0_i64)?)?;
                    let extent = self.constant(Array::scalar(extent)?)?;
                    let indices = V::select(&indices.less_than(&zero)?, &indices.add(&extent)?, &indices)?;
                    components.push((axis, indices, None));
                    collapsed.push(axis);
                    axis += 1;
                }
                ExpandedIndexSelector::Basic(BasicIndex::Slice(slice)) => {
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
                        output_shape.push(extent);
                    }
                }
                OutputAxis::Query(query) => {
                    slice_query_axes[query] = query_shape.len();
                    query_shape.push(query_lengths[query]);
                    output_shape.push(query_lengths[query]);
                }
                OutputAxis::Window(axis) => {
                    // Offsets index the gather result, which has no inserted axes.
                    offsets.push(output_shape.len() - new_axes.len());
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
            output_shape,
            new_axes,
        })
    }
}

impl<
    V: Value<Type = ArrayIrType, DispatchDomain: Context<Type = ArrayIrType> + DimensionConstant>
        + ValueProjection<
            ArrayType,
            Projected: Value<Type = ArrayType, ExecutionDomain: Context<Operation: From<ConstantOperation<Array>>>>
                           + Broadcast
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
        > + DimensionSize
        + DimensionToScalar
        + DynamicGather
        + DynamicReshape,
> Indexed<'_, '_, '_, V, ArrayIrType>
{
    /// Reads the selection of an array that may have a symbolic shape. A selection whose input and index arrays all
    /// have concrete shapes is served by the implementation for array values, with the result lifted back into this
    /// value family. A selection with a symbolic shape instead keeps its extents as dimension values and supports one
    /// indexed axis, full forward or reverse slices on the other axes, and inserted axes. When starting from an array
    /// tracer projected out of an [`ArrayIrType`] value, read the selection through that value instead. Refer to
    /// [`Indexed`] for the complete supported-geometry contract.
    ///
    /// Symbolic reads inherit [`DynamicGather`]'s one-element window requirement on the indexed axis. Specifically,
    /// its minimum extent must be positive unless the query is statically empty. Untouched axes and query axes may have
    /// zero runtime extents.
    ///
    /// # Parameters
    ///
    ///   - `options`: Bounds handling and optional fill. Symbolic geometry rejects explicit output sharding and
    ///     index promises. Concrete geometry also supports placement and uniqueness as described on [`Indexed`].
    pub fn get(&self, options: &GatherOptions) -> Result<V, ProgramError> {
        if !self.has_symbolic_shape()? {
            return self.with_projected_selection(|selection| selection.get(options)).map(V::from_projected);
        }

        // The symbolic path does not remap explicit output sharding or the index promises yet;
        // bounds modes and fills remain available.
        if options.output_sharding().is_some() || options.indices_are_sorted() || options.unique_indices() {
            return Err(TypeError::invalid(
                "symbolic indexing does not yet support explicit output sharding or index promises",
            )
            .into());
        }

        // Identity and full-reversal selections skip gather, but explicit fills still have the same scalar/data-type
        // contract as indexed reads. Validate before those fast paths instead of silently accepting a malformed fill.
        if matches!(options.mode(), GatherMode::Fill { value: Some(_) }) {
            let input_type = self.input.r#type();
            options.resolved_fill_value(<&ArrayType>::try_from(input_type.as_ref())?.data_type())?;
        }

        let plan = self.dynamic_plan(matches!(options.mode(), GatherMode::PromiseInBounds))?;
        let mut output = Self::reversed(self.input, &plan.reversed_axes)?;

        if let Some((axis, indices)) = plan.query {
            output = output.dynamic_gather_axis(&indices, axis, options.mode().clone())?;
        }

        for axis in plan.inserted_axes {
            output = output.dynamic_expand_dimensions(axis)?;
        }

        Ok(output)
    }
}

impl<
    V: Value<Type = ArrayIrType, DispatchDomain: Context<Type = ArrayIrType> + DimensionConstant>
        + ValueProjection<
            ArrayType,
            Projected: Value<Type = ArrayType, ExecutionDomain: Context<Operation: From<ConstantOperation<Array>>>>
                           + Broadcast
                           + Reshape
                           + Concatenate
                           + ConvertElementType
                           + Compare
                           + Add
                           + Select
                           + Reverse
                           + Scatter
                           + TransferToMemory,
        > + DimensionSize
        + DimensionToScalar
        + DynamicScatter
        + DynamicReshape
        + DynamicBroadcast,
> Indexed<'_, '_, '_, V, ArrayIrType>
{
    /// Overwrites the selected elements of an array that may have a symbolic shape, returning a new value and leaving
    /// the input unchanged. Updates broadcast to the selected shape, using dimension values for symbolic extents.
    ///
    /// # Parameters
    ///
    ///   - `updates`: Array value with the input's data type and shape broadcastable to the selection.
    ///   - `options`: Scatter bounds policy and placement/promises for concrete geometry. Symbolic geometry rejects
    ///     explicit output sharding and index promises. Refer to [`Indexed`] for supported geometry and the shared
    ///     bounds and index-promise contracts.
    #[inline]
    pub fn set(&self, updates: &V, options: &ScatterOptions) -> Result<V, ProgramError> {
        self.update(updates, ScatterReductionKind::Overwrite, options)
    }

    /// Adds every selected update, including duplicates, into an array that may have a symbolic shape. Updates
    /// broadcast to the selected shape, using dimension values for symbolic extents. Refer to the documentation of
    /// [`set`](Self::set) for the shared parameter contract.
    #[inline]
    pub fn add(&self, updates: &V, options: &ScatterOptions) -> Result<V, ProgramError> {
        self.update(updates, ScatterReductionKind::Add, options)
    }

    /// Multiplies selected updates into an array that may have a symbolic shape; scatter's differentiation restrictions
    /// apply. Updates broadcast to the selected shape, using dimension values for symbolic extents. Refer to the
    /// documentation of [`set`](Self::set) for the shared parameter contract.
    #[inline]
    pub fn mul(&self, updates: &V, options: &ScatterOptions) -> Result<V, ProgramError> {
        self.update(updates, ScatterReductionKind::Mul, options)
    }

    /// Combines selected updates with an array that may have a symbolic shape using the elementwise minimum. Updates
    /// broadcast to the selected shape, using dimension values for symbolic extents. Refer to the documentation of
    /// [`set`](Self::set) for the shared parameter contract.
    #[inline]
    pub fn min(&self, updates: &V, options: &ScatterOptions) -> Result<V, ProgramError> {
        self.update(updates, ScatterReductionKind::Min, options)
    }

    /// Combines selected updates with an array that may have a symbolic shape using the elementwise maximum. Updates
    /// broadcast to the selected shape, using dimension values for symbolic extents. Refer to the documentation of
    /// [`set`](Self::set) for the shared parameter contract.
    #[inline]
    pub fn max(&self, updates: &V, options: &ScatterOptions) -> Result<V, ProgramError> {
        self.update(updates, ScatterReductionKind::Max, options)
    }

    /// Shared implementation of [`set`](Self::set), [`add`](Self::add), [`mul`](Self::mul), [`min`](Self::min), and
    /// [`max`](Self::max) for arrays that may have symbolic shapes. A selection whose input and index arrays all have
    /// concrete shapes is served by the implementation for array values, with the result lifted back into this value
    /// family. A selection with a symbolic shape is instead resolved by [`dynamic_plan`](Self::dynamic_plan), which
    /// keeps the dimension values available, and then scattered along its single queried axis with dynamic operations.
    /// The reversed input is updated in selection order and reversed back, so the returned value has the original
    /// geometry.
    ///
    /// # Parameters
    ///
    ///   - `updates`: Array value with the input's data type, broadcastable to the selection shape.
    ///   - `kind`: Scatter reduction combining each update with the element it lands on.
    ///   - `options`: Bounds handling, output placement, and index promises, as described on [`Indexed`].
    fn update(&self, updates: &V, kind: ScatterReductionKind, options: &ScatterOptions) -> Result<V, ProgramError> {
        if !self.has_symbolic_shape()? {
            let updates = updates.clone().into_projected()?;
            return self
                .with_projected_selection(|selection| selection.update(&updates, kind, options))
                .map(V::from_projected);
        }

        // The symbolic path does not remap explicit output sharding or the index promises yet;
        // bounds modes and fills remain available.
        if options.output_sharding().is_some() || options.indices_are_sorted() || options.unique_indices() {
            return Err(TypeError::invalid(
                "symbolic indexing does not yet support explicit output sharding or index promises",
            )
            .into());
        }

        let plan = self.dynamic_plan(options.mode() == ScatterMode::PromiseInBounds)?;
        let input_type = self.input.r#type();
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
            dimensions.push(self.input.dimension_size(axis)?);
        }

        let mut selected_dimensions = dimensions.clone();
        for &axis in &plan.inserted_axes {
            selected_dimensions.insert(axis, self.input.dispatch_domain().dimension_constant(1)?);
        }

        let updates = updates.dynamic_broadcast_to(&selected_dimensions)?;
        let updates = if plan.inserted_axes.is_empty() { updates } else { updates.dynamic_reshape(&dimensions)? };
        let base = Self::reversed(self.input, &plan.reversed_axes)?;
        let output = if let Some((axis, indices)) = plan.query {
            base.dynamic_scatter_axis(&indices, &updates, axis, kind, options.mode())?
        } else {
            // A zero-width index vector describes a single whole-array update without inventing a runtime-sized
            // window. All update axes are window axes and every input element is updated exactly once.
            let indices = self.constant(Array::vector(Vec::<i64>::new())?)?;
            V::from_projected(base.into_projected()?.scatter(
                &indices.into_projected()?,
                &updates.into_projected()?,
                &ScatterDimensionNumbers::new((0..input_type.rank()).collect(), vec![], vec![]),
                kind,
                options,
            )?)
        };

        Self::reversed(&output, &plan.reversed_axes)
    }
}

impl<V: Value<Type = ArrayIrType> + ReferenceIndex + ReferenceSlice + ReferenceDynamicIndex>
    Indexed<'_, '_, '_, V, ArrayIrType>
{
    /// Derives a reference view of the selected region of a reference input without accessing its state. The view
    /// shares the input's allocation, so reads through it observe later writes to the input and writes through it are
    /// visible to the input, which is the reference counterpart of NumPy's basic-indexing views and of `ref.at[...]`
    /// in [JAX](https://docs.jax.dev/en/latest/jax.ref.html). Reference views are restricted to the transforms the
    /// reference machinery can reconstruct:host integers remove their axis, unit-stride forward slices keep theirs, an
    /// ellipsis expands over the unspecified axes, and a scalar integer array selects one position on its axis at run
    /// time (clamped into bounds, as for [`reference_dynamic_index`](ReferenceDynamicIndex::reference_dynamic_index)).
    /// Inserted axes, masks, non-scalar index arrays, strided or reversed slices, and out-of-bounds host integers are
    /// rejected. Every axis that a host integer or slice touches must have a static extent, and when any slice is
    /// present the whole referent shape must be static. A selection that touches no axis returns the input itself.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{Array, ArrayIrValue, Indexing, ProgramError, ReferenceNew, ReferenceRead, index};
    /// # fn main() -> Result<(), ProgramError> {
    /// let buffer = ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])?).reference_new()?;
    /// let row = buffer.at(&index![1, 1..]).view()?;
    /// assert_eq!(row.read()?, ArrayIrValue::Array(Array::vector(vec![5.0f32, 6.0])?));
    /// # Ok(())
    /// # }
    /// ```
    pub fn view(&self) -> Result<V, ProgramError> {
        let input_type = self.input.r#type();
        let referent = <&ReferenceType<ArrayType>>::try_from(input_type.as_ref())?.referent();
        let rank = referent.rank();

        // Reject unsupported selector kinds and validate the selection against the referent rank before assigning
        // selectors to axes, so that the ellipsis can be expanded over exactly the unspecified axes.
        let mut ellipses = 0;
        let mut consumed = 0;
        for selector in self.selectors {
            match selector {
                IndexSelector::Basic(BasicIndex::Ellipsis) => ellipses += 1,
                IndexSelector::Basic(BasicIndex::NewAxis) => {
                    return Err(TypeError::invalid("reference views cannot insert axes").into());
                }
                IndexSelector::Mask(_) => {
                    return Err(TypeError::invalid("reference views do not support index masks").into());
                }
                IndexSelector::Basic(_) | IndexSelector::Array(_) => consumed += 1,
            }
        }

        if ellipses > 1 {
            return Err(TypeError::invalid("index selection contains more than one ellipsis").into());
        }

        if consumed > rank {
            return Err(TypeError::invalid(format!(
                "index selection consumes {consumed} axes but input rank is {rank}",
            ))
            .into());
        }

        let mut assigned = vec![None; rank];
        let mut axis = 0;
        for selector in self.selectors {
            if matches!(selector, IndexSelector::Basic(BasicIndex::Ellipsis)) {
                axis += rank - consumed;
            } else {
                assigned[axis] = Some(selector);
                axis += 1;
            }
        }

        // Resolve every selector against its axis. Host integers and slices need the static extent of their axis to
        // normalize negative coordinates and to prove that the selection lies within the referent.
        let extent = |axis: usize| -> Result<usize, ProgramError> {
            referent.dimension(axis).value().ok_or_else(|| {
                TypeError::invalid(format!(
                    "reference views require a static extent on axis {axis} but got `{referent}`"
                ))
                .into()
            })
        };

        let mut selections = Vec::with_capacity(rank);
        for (axis, selector) in assigned.iter().enumerate() {
            let selection = match selector {
                None
                | Some(IndexSelector::Basic(BasicIndex::Ellipsis | BasicIndex::NewAxis))
                | Some(IndexSelector::Mask(_)) => ReferenceAxisSelection::Full,
                Some(IndexSelector::Basic(BasicIndex::Integer(index))) => {
                    let extent = extent(axis)?;
                    let normalized = if *index < 0 { *index + extent as i128 } else { *index };
                    if normalized < 0 || normalized >= extent as i128 {
                        return Err(TypeError::invalid(format!(
                            "index {index} is out of bounds for axis {axis} with extent {extent}",
                        ))
                        .into());
                    }
                    ReferenceAxisSelection::Index(normalized as usize)
                }
                Some(IndexSelector::Basic(BasicIndex::Slice(slice))) => {
                    if slice.step() < 0 {
                        return Err(TypeError::invalid("reference views do not support reversed slices").into());
                    }

                    // A complete unit-stride slice selects its axis in full whatever the extent, so it never needs a
                    // static extent, which keeps dynamic axes that are not touched selectable through `..`.
                    if slice.start().is_none() && slice.stop().is_none() && slice.step() == 1 {
                        selections.push(ReferenceAxisSelection::Full);
                        continue;
                    }

                    let extent = extent(axis)?;
                    let normalized = slice.normalize(extent)?;
                    if normalized.stride != 1 {
                        return Err(TypeError::invalid("reference views do not support strided slices").into());
                    }

                    if normalized.start == 0 && normalized.length == extent {
                        ReferenceAxisSelection::Full
                    } else {
                        ReferenceAxisSelection::Window(ArraySliceAxis::new(normalized.start, normalized.length, 1))
                    }
                }
                Some(IndexSelector::Array(value)) => {
                    let index_type = value.r#type();
                    let index_type = <&ArrayType>::try_from(index_type.as_ref())?;
                    if index_type.rank() != 0 || !index_type.data_type().is_integer() {
                        return Err(TypeError::invalid(format!(
                            "reference views support only scalar integer index arrays but got `{index_type}`"
                        ))
                        .into());
                    }
                    ReferenceAxisSelection::Dynamic(*value)
                }
            };
            selections.push(selection);
        }

        // Windows are applied first through one rank-preserving slice over every axis, and then the indexed axes
        // are removed from the sliced view in ascending order, adjusting for the axes removed before them.
        let mut view = self.input.clone();
        if selections.iter().any(|selection| matches!(selection, ReferenceAxisSelection::Window(_))) {
            let axes = selections
                .iter()
                .enumerate()
                .map(|(axis, selection)| match selection {
                    ReferenceAxisSelection::Window(window) => Ok(*window),
                    _ => Ok(ArraySliceAxis::new(0, extent(axis)?, 1)),
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            view = view.reference_slice(&axes)?;
        }

        let mut removed = 0;
        for (axis, selection) in selections.iter().enumerate() {
            match selection {
                ReferenceAxisSelection::Index(index) => {
                    view = view.reference_index(axis - removed, *index)?;
                    removed += 1;
                }
                ReferenceAxisSelection::Dynamic(index) => {
                    view = view.reference_dynamic_index(axis - removed, index)?;
                    removed += 1;
                }
                ReferenceAxisSelection::Full | ReferenceAxisSelection::Window(_) => {}
            }
        }

        Ok(view)
    }
}

impl<V: Value<Type = ArrayIrType> + ReferenceIndex + ReferenceSlice + ReferenceDynamicIndex + ReferenceRead>
    Indexed<'_, '_, '_, V, ArrayIrType>
{
    /// Reads the selected region of a reference input as an immutable snapshot. This is [`view`](Self::view) followed
    /// by [`read`](ReferenceRead::read), so it supports exactly the selections that [`view`](Self::view) accepts and
    /// observes the reference state at the point of the read in program order.
    #[inline]
    pub fn read(&self) -> Result<V, ProgramError> {
        self.view()?.read()
    }
}

impl<
    V: Value<Type = ArrayIrType>
        + ReferenceIndex
        + ReferenceSlice
        + ReferenceDynamicIndex
        + ReferenceWrite
        + ReferenceAddUpdate
        + ReferenceSwap,
> Indexed<'_, '_, '_, V, ArrayIrType>
{
    /// Overwrites the selected region of a reference input in place. This is [`view`](Self::view) followed by
    /// [`write`](ReferenceWrite::write), so it supports exactly the selections that [`view`](Self::view) accepts. It
    /// is the in-place counterpart of the functional [`set`](Self::set) on array values: the reference's state is
    /// updated in program order and nothing is returned.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{Array, ArrayIrValue, Indexing, ProgramError, ReferenceNew, ReferenceRead, index};
    /// # fn main() -> Result<(), ProgramError> {
    /// let buffer = ArrayIrValue::Array(Array::vector(vec![0_i32, 1, 2, 3])?).reference_new()?;
    /// buffer.at(&index![1..3]).write(&ArrayIrValue::Array(Array::vector(vec![9_i32, 9])?))?;
    /// assert_eq!(buffer.read()?, ArrayIrValue::Array(Array::vector(vec![0_i32, 9, 9, 3])?));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn write(&self, replacement: &V) -> Result<(), ProgramError> {
        self.view()?.write(replacement)
    }

    /// Adds `update` into the selected region of a reference input in place. This is [`view`](Self::view)
    /// followed by [`add_update`](ReferenceAddUpdate::add_update), and the in-place counterpart of the functional
    /// [`add`](Self::add) on array values. Refer to the documentation of [`write`](Self::write) for the shared
    /// selection contract.
    #[inline]
    pub fn add_update(&self, update: &V) -> Result<(), ProgramError> {
        self.view()?.add_update(update)
    }

    /// Overwrites the selected region of a reference input in place and returns the previously stored region.
    /// This is [`view`](Self::view) followed by [`swap`](ReferenceSwap::swap). Refer to the documentation of
    /// [`write`](Self::write) for the shared selection contract.
    #[inline]
    pub fn swap(&self, replacement: &V) -> Result<V, ProgramError> {
        self.view()?.swap(replacement)
    }
}

impl<V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>>
    Indexed<'_, '_, '_, V, ArrayIrType>
{
    /// Returns whether the input or an index array has a non-concrete dimension in its shape.
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

    /// Rebuilds this selection over the array projected out of the input and calls `select` with it, so that a
    /// selection with concrete shapes can be served by the implementation for array values. Every index array is
    /// projected the same way and the selectors are re-borrowed over those projections, which is why the projected
    /// selection only lives for the duration of the call. The result of `select` is returned as is, and callers lift
    /// it back into this value family when it is a value.
    ///
    /// # Parameters
    ///
    ///   - `select`: Terminal function applied to the projected selection (e.g., its `get` or `update`).
    fn with_projected_selection<
        R,
        F: FnOnce(Indexed<'_, '_, '_, V::Projected, ArrayType>) -> Result<R, ProgramError>,
    >(
        &self,
        select_fn: F,
    ) -> Result<R, ProgramError> {
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
        select_fn(input.at(&selectors))
    }
}

impl<
    V: Value<Type = ArrayIrType>
        + DimensionSize
        + DimensionToScalar
        + ValueProjection<
            ArrayType,
            Projected: Value<Type = ArrayType, ExecutionDomain: Context<Operation: From<ConstantOperation<Array>>>>
                           + ConvertElementType
                           + Compare
                           + Add
                           + Select
                           + TransferToMemory
                           + Reverse,
        >,
> Indexed<'_, '_, '_, V, ArrayIrType>
{
    /// Lifts a host coordinate literal through the array projected out of the input. Refer to the documentation of the
    /// array-valued [`constant`](Indexed::constant) for more information.
    fn constant(&self, value: Array) -> Result<V, ProgramError> {
        Ok(V::from_projected(self.input.clone().into_projected()?.at(&[]).constant(value)?))
    }

    /// Returns `value` with its projected array reversed along `axes`, or an unchanged clone when no axis is reversed,
    /// so that both paths stay in the caller's value family.
    fn reversed(value: &V, axes: &[usize]) -> Result<V, ProgramError> {
        if axes.is_empty() {
            return Ok(value.clone());
        }
        Ok(V::from_projected(value.clone().into_projected()?.reverse(axes.to_vec())?))
    }

    /// Resolves the supported symbolic selection geometry without reading traced array data on the host. One host-side
    /// pass over the selectors validates every one of them and records the reversed axes, the inserted axes, and the
    /// single queried axis; only then is that query staged, so that a bad later selector never leaves a partly staged
    /// indexing expression in the caller's context.
    ///
    /// # Parameters
    ///
    ///   - `promise_in_bounds`: Whether the caller promised in-bounds indices, in which case host integers that cannot
    ///     lie within the largest admitted extent of their axis are rejected here.
    fn dynamic_plan(&self, promise_in_bounds: bool) -> Result<DynamicIndexPlan<V>, ProgramError> {
        let input_type = self.input.r#type();
        let input_type = <&ArrayType>::try_from(input_type.as_ref())?;
        let rank = input_type.rank();

        let consumed = self
            .selectors
            .iter()
            .filter(|selector| !matches!(selector, IndexSelector::Basic(BasicIndex::NewAxis | BasicIndex::Ellipsis)))
            .count();
        if consumed > rank {
            return Err(TypeError::invalid(format!(
                "index selection consumes {consumed} axes but input rank is {rank}",
            ))
            .into());
        }

        let ellipses = self
            .selectors
            .iter()
            .filter(|selector| matches!(selector, IndexSelector::Basic(BasicIndex::Ellipsis)))
            .count();
        if ellipses > 1 {
            return Err(TypeError::invalid("index selection contains more than one ellipsis").into());
        }

        let mut query = None;
        let mut query_count = 0;
        let mut reversed_axes = Vec::new();
        let mut inserted_axes = Vec::new();
        let mut input_axis = 0;
        let mut output_axis = 0;
        for selector in self.selectors {
            match selector {
                IndexSelector::Basic(BasicIndex::Ellipsis) => {
                    let omitted = rank - consumed;
                    input_axis += omitted;
                    output_axis += omitted;
                }
                IndexSelector::Basic(BasicIndex::NewAxis) => {
                    inserted_axes.push(output_axis);
                    output_axis += 1;
                }
                IndexSelector::Basic(BasicIndex::Slice(slice)) => {
                    if slice.start().is_some() || slice.stop().is_some() || !matches!(slice.step(), -1 | 1) {
                        return Err(TypeError::invalid(
                            "symbolic indexing currently requires full slices with step `1` or `-1`",
                        )
                        .into());
                    }

                    if slice.step() < 0 {
                        reversed_axes.push(input_axis);
                    }

                    input_axis += 1;
                    output_axis += 1;
                }
                IndexSelector::Basic(BasicIndex::Integer(index)) => {
                    i64::try_from(*index).map_err(|_| {
                        TypeError::invalid("symbolic indexing requires host integer indices representable as `i64`")
                    })?;

                    if promise_in_bounds {
                        // Dynamic upper bounds are exclusive. Even when the actual extent is unavailable, an index
                        // outside the largest permitted extent cannot satisfy the caller's in-bounds promise.
                        let dimension = input_type.dimension(input_axis);
                        let maximum = dimension.value().or_else(|| dimension.bounds().upper().map(|upper| upper - 1));
                        if maximum.is_some_and(|maximum| *index >= maximum as i128 || *index < -(maximum as i128)) {
                            return Err(TypeError::invalid(
                                "host integer index is out of bounds under `PromiseInBounds`",
                            )
                            .into());
                        }
                    }

                    query.get_or_insert((input_axis, selector));
                    query_count += 1;
                    input_axis += 1;
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

                    query.get_or_insert((input_axis, selector));
                    query_count += 1;
                    input_axis += 1;
                    output_axis += indices_type.rank();
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

        let query = match query {
            None => None,
            Some((axis, selector)) => {
                let indices = match selector {
                    IndexSelector::Basic(BasicIndex::Integer(index)) => {
                        self.constant(Array::scalar(i64::try_from(*index).unwrap())?)?
                    }
                    IndexSelector::Array(indices) => (*indices).clone(),
                    _ => unreachable!("only host integers and index arrays are recorded as queries"),
                };

                // Normalize negatives using the actual retained dimension value. This is ordinary array arithmetic,
                // preserving the index array's symbolic shape and structural-zero tangent rather than concretizing it.
                let indices = indices.into_projected()?.convert_element_type(DataType::I64)?;
                let extent = self
                    .input
                    .dimension_size(axis)?
                    .to_scalar()?
                    .into_projected()?
                    .transfer_to_memory(input_type.memory())?;
                let zero = self.constant(Array::scalar(0i64)?)?.into_projected()?;
                let negative = indices.less_than(&zero)?;
                let wrapped = indices.add(&extent)?;

                Some((axis, V::from_projected(V::Projected::select(&negative, &wrapped, &indices)?)))
            }
        };

        Ok(DynamicIndexPlan { query, reversed_axes, inserted_axes })
    }
}

/// Constructs a fixed-size array of [`IndexSelector`]s for [`Indexing::at`] from a comma-separated list of selectors
/// written in [NumPy-style](https://numpy.org/doc/stable/user/basics.indexing.html) notation, one per input axis in
/// order. Axes not mentioned at the end of the list are selected in full.
///
/// # Syntax
///
/// | Selector      | Meaning                                                                                       |
/// |---------------|-----------------------------------------------------------------------------------------------|
/// | `i`           | Selects position `i` and removes the axis; a negative `i` counts from the end of the axis.    |
/// | `a..b`        | Selects the exclusive range with a unit stride; either endpoint may be omitted, as in `..`.   |
/// | `a..b by s`   | Strides the range by `s`; a negative `s` walks the axis backward.                             |
/// | `&values`     | Borrows an integer array whose elements select positions.                                     |
/// | `&mask`       | Borrows an [`IndexMask`] whose `true` entries select positions.                               |
/// | `new_axis`    | Inserts an axis of extent one without consuming an input axis.                                |
/// | `...`         | Expands to full slices over every axis not mentioned elsewhere in the list.                   |
///
/// Integer positions, endpoints, and strides may be any expression of a type implementing [`IndexInteger`], and
/// each expression is evaluated exactly once. Wrap an expression that contains a top-level comma, such as an explicit
/// generic argument list, in parentheses. The macro only builds descriptors; nothing executes until a terminal function
/// of [`Indexed`] is called. The value type of the descriptors is inferred from the receiver of `at`, so a selection
/// passed straight to it needs no annotation.
///
/// # Examples
///
/// Each selector becomes one descriptor:
///
/// ```rust
/// # use ryft_core::{Array, BasicIndex, IndexSelector, IndexSlice, index};
/// let selection: [IndexSelector<'_, Array>; 4] = index![1, 1..9 by 2, new_axis, ...];
/// assert_eq!(selection[0], IndexSelector::Basic(BasicIndex::Integer(1)));
/// assert_eq!(selection[1], IndexSelector::Basic(BasicIndex::Slice(IndexSlice::new(Some(1), Some(9), 2))));
/// assert_eq!(selection[2], IndexSelector::Basic(BasicIndex::NewAxis));
/// assert_eq!(selection[3], IndexSelector::Basic(BasicIndex::Ellipsis));
/// ```
///
/// Selections read like array indexing when passed to `at`:
///
/// ```rust
/// # use ryft_core::{Array, GatherOptions, Indexing, ProgramError, index};
/// # fn main() -> Result<(), ProgramError> {
/// let matrix = Array::matrix(3, 4, (0..12).collect::<Vec<i32>>())?;
///
/// // The last row, every other column.
/// assert_eq!(matrix.at(&index![-1, .. by 2]).get(&GatherOptions::new())?, Array::vector(vec![8i32, 10])?);
///
/// // Rows picked by an index array, with the columns reversed.
/// let rows = Array::vector(vec![2i32, 0])?;
/// assert_eq!(
///     matrix.at(&index![&rows, .. by -1]).get(&GatherOptions::new())?,
///     Array::matrix(2, 4, vec![11i32, 10, 9, 8, 3, 2, 1, 0])?,
/// );
/// # Ok(())
/// # }
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
        $crate::IndexSelector::Basic($crate::BasicIndex::Ellipsis)
    };

    // A new axis contributes a size-one output axis without consuming an input axis.
    (@selector [] new_axis) => {
        $crate::IndexSelector::Basic($crate::BasicIndex::NewAxis)
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
        $crate::IndexSelector::from($($value)+)
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
        $crate::IndexSelector::Basic($crate::BasicIndex::Slice($crate::IndexSlice::new($start, $stop, $step)))
    };

    // Convert integer expressions through the supported, lossless host-integer conversions.
    (@integer $value:expr) => {
        $crate::IndexInteger::to_index_integer($value)
    };

    // Reject malformed internal parser states without recursively treating them as public syntax.
    (@$state:ident $($rest:tt)*) => {
        compile_error!("invalid indexing selector syntax")
    };

    // The public form accepts comma-separated selectors and an optional trailing comma. It must follow the internal
    // `@`-prefixed arms, because it matches any token sequence, including theirs.
    ($($selectors:tt)*) => {
        $crate::index!(@items [] [] $($selectors)*)
    };
}

// The macro is exported at the crate root by `#[macro_export]`; this re-export lets callers that import the
// manipulation facade reach it through the module path as well.
pub use crate::index;

/// One entry of a selection after [`expanded`](Indexed::expanded) has made it explicit. The public [`IndexSelector`]
/// list is what the caller wrote; this is what the planner reads. By this point the ellipsis has been replaced by
/// full slices over the unspecified axes, every mask has been turned into the integer coordinate arrays of its true
/// positions, and index arrays are owned by the working value family rather than borrowed. The `Ellipsis` entry itself
/// is kept as a zero-width marker, because an ellipsis separates advanced-index groups even when it consumes no axes.
#[derive(Clone)]
enum ExpandedIndexSelector<V> {
    /// Host integer, slice, new axis, or the ellipsis marker.
    Basic(BasicIndex),

    /// Owned integer coordinate array, cloned from the caller's index array or produced from a mask.
    Array(V),

    /// Scalar mask, contributing an advanced axis of extent one (i.e., `true`) or zero (i.e., `false`)
    /// without consuming an input axis.
    Boolean(bool),
}

/// The gather or scatter that realizes one expanded selection. [`plan`](Indexed::plan) builds it once from the
/// [`ExpandedIndexSelector`]s, and reads and updates share it so that both address exactly the same input positions.
/// Contiguous slices become gather windows, and only strided slices and advanced indices contribute coordinates, which
/// keeps the coordinate array small. The caller sees [`output_shape`](Self::output_shape), whereas the gather result
/// and the scatter update have that shape without the axes at [`new_axes`](Self::new_axes), because an inserted axis
/// consumes no input axis; reads reshape after gathering and updates reshape before scattering.
struct IndexPlan<V> {
    /// Jointly broadcast coordinates of every strided slice and advanced index, with a trailing index-vector axis.
    indices: V,

    /// [`GatherDimensionNumbers`] mapping windows, collapsed axes, and coordinate components onto the input axes.
    /// The scatter reuses the same mapping.
    dimensions: GatherDimensionNumbers,

    /// Window size on every input axis: the slice length for a contiguous slice and one everywhere else.
    sizes: Vec<usize>,

    /// Shape the caller sees, including the inserted extent-one axes.
    output_shape: Vec<usize>,

    /// Positions of the inserted extent-one axes within [`output_shape`](Self::output_shape).
    new_axes: Vec<usize>,
}

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

/// Per-axis outcome of resolving a selection against a reference's referent shape.
enum ReferenceAxisSelection<'i, V> {
    /// The axis is selected in full and needs no view transform.
    Full,

    /// A static unit-stride window that keeps the axis.
    Window(ArraySliceAxis),

    /// A normalized host index that removes the axis.
    Index(usize),

    /// A scalar integer value that removes the axis at run time.
    Dynamic(&'i V),
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayIrOperation, ArrayIrValue, ArrayOperation, DimensionBounds, DimensionValue, DimensionVariable,
        LogicalMesh, Memory, MeshAxis, MeshAxisType, Sharding, ShardingDimension,
    };
    use crate::batching::{BatchAxis, batch};
    use crate::contexts::{Context, EagerContext};
    use crate::differentiation::differentiate_at;
    use crate::operations::references::{ReferenceFreeze, ReferenceNew};
    use crate::partial::PartialValue;
    use crate::tracing::{Trace, Tracer, TracingContext};

    use super::*;

    /// Tracer over [`ArrayIrValue`]s used by the symbolic-geometry tests.
    type ArrayIrTracer = Tracer<TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>;

    /// Allocates the 2x3 `f32` reference used by the reference-view tests.
    fn reference_matrix() -> ArrayIrValue<Array> {
        ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap())
            .reference_new()
            .unwrap()
    }

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
        assert_eq!(
            // An explicit -1 is the last element, whereas the omitted reverse stop lies before the first element.
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
            IndexSelector::Basic(BasicIndex::Integer(usize::MAX as i128)),
        );
        assert_eq!(
            IndexSelector::<Array>::from(1..3),
            IndexSelector::Basic(BasicIndex::Slice(IndexSlice::new(Some(1), Some(3), 1))),
        );
        assert_eq!(
            IndexSelector::<Array>::from(1..),
            IndexSelector::Basic(BasicIndex::Slice(IndexSlice::new(Some(1), None, 1))),
        );
        assert_eq!(
            IndexSelector::<Array>::from(..3),
            IndexSelector::Basic(BasicIndex::Slice(IndexSlice::new(None, Some(3), 1))),
        );
        assert_eq!(
            IndexSelector::<Array>::from(..),
            IndexSelector::Basic(BasicIndex::Slice(IndexSlice::new(None, None, 1))),
        );
        assert_eq!(IndexSelector::<Array>::from(BasicIndex::NewAxis), IndexSelector::Basic(BasicIndex::NewAxis));
        assert_eq!(
            IndexSelector::<Array>::from(IndexSlice::new(Some(-2), None, -1)),
            IndexSelector::Basic(BasicIndex::Slice(IndexSlice::new(Some(-2), None, -1))),
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
            Array::matrix(2, 2, vec![5_i32, 7, 9, 11]),
        );
        assert_eq!(input.at(&index![..., -1]).get(&GatherOptions::new()), Array::vector(vec![3_i32, 7, 11]));
        assert_eq!(input.at(&index![]).get(&GatherOptions::new()), Ok(input.clone()));
        assert_eq!(
            input.at(&index![new_axis, 1, ..]).get(&GatherOptions::new()),
            Array::matrix(1, 4, vec![4_i32, 5, 6, 7]),
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
        let input =
            Array::from_elements(ArrayType::new_static(DataType::I32, [3, 4, 5]), &(0_i32..60).collect::<Vec<_>>())
                .unwrap();
        let rows = Array::vector(vec![0_i32, 2]).unwrap();
        let columns = Array::vector(vec![1_i32, 3]).unwrap();
        assert_eq!(
            input.at(&index![.., &rows, &columns]).get(&GatherOptions::new()),
            Array::matrix(3, 2, vec![1_i32, 13, 21, 33, 41, 53]),
        );
        assert_eq!(
            // An ellipsis remains a separator even when it expands to zero axes.
            input.at(&index![.., &rows, ..., &columns]).get(&GatherOptions::new()),
            Array::matrix(2, 3, vec![1_i32, 21, 41, 13, 33, 53]),
        );
        assert_eq!(
            input.at(&index![1, .., &columns]).get(&GatherOptions::new()),
            Array::matrix(2, 4, vec![21_i32, 26, 31, 36, 23, 28, 33, 38]),
        );
        assert_eq!(
            input.at(&index![.., 1, &columns]).get(&GatherOptions::new()),
            Array::matrix(3, 2, vec![6_i32, 8, 26, 28, 46, 48]),
        );
        let scalar = Array::scalar(1_i32).unwrap();
        assert_eq!(
            input.at(&index![&scalar, .., &columns]).get(&GatherOptions::new()),
            Array::matrix(2, 4, vec![21_i32, 26, 31, 36, 23, 28, 33, 38]),
        );
        assert_eq!(
            input.at(&index![.., &rows, new_axis, &columns]).get(&GatherOptions::new()),
            Array::from_elements(ArrayType::new_static(DataType::I32, [2, 3, 1]), &[1_i32, 21, 41, 13, 33, 53]),
        );
        let rows = Array::matrix(2, 1, vec![0_i32, 2]).unwrap();
        assert_eq!(
            input.at(&index![&rows, &columns, 1]).get(&GatherOptions::new()),
            Array::matrix(2, 2, vec![6_i32, 16, 46, 56]),
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
            Array::from_elements::<i32>(ArrayType::new_static(DataType::I32, [0, 3]), &[]),
        );
        let active = IndexMask::new(vec![], vec![true]).unwrap();
        assert_eq!(
            input.at(&index![&active]).get(&GatherOptions::new()),
            Array::from_elements(ArrayType::new_static(DataType::I32, [1, 2, 3]), &[0_i32, 1, 2, 3, 4, 5]),
        );
        let inactive = IndexMask::new(vec![], vec![false]).unwrap();
        assert_eq!(
            input.at(&index![&inactive]).get(&GatherOptions::new()),
            Array::from_elements::<i32>(ArrayType::new_static(DataType::I32, [0, 2, 3]), &[]),
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
        assert_eq!(input.at(&index![&indices]).get(&fill), Array::vector(vec![10_i32, 30, -99]));
        assert_eq!(
            input.at(&index![&indices]).get(&GatherOptions::new().with_mode(GatherMode::Clip)),
            Array::vector(vec![10_i32, 30, 30]),
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
            .trim_end(),
        );
        assert_eq!(
            program.interpret(Array::matrix(2, 3, vec![10_f64, 20., 30., 40., 50., 60.]).unwrap()),
            Ok(Array::vector(vec![20_f64, 50.]).unwrap()),
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
            .unwrap(),
        );
    }

    #[test]
    fn test_indexed_get_reduction_state() {
        // A gather from an unreduced input keeps its reduction state on the selection.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 1).with_unreduced_axes(["x"]).unwrap();
        let input = Array::from_elements(
            ArrayType::new_static(DataType::F64, [3]).with_sharding(sharding.clone()).unwrap(),
            &[10_f64, 20., 30.],
        )
        .unwrap();
        let indices = Array::vector(vec![2_i32, 0]).unwrap();
        assert_eq!(
            input.at(&index![&indices]).get(&GatherOptions::new()),
            Ok(Array::from_elements(
                ArrayType::new_static(DataType::F64, [2]).with_sharding(sharding).unwrap(),
                &[30_f64, 10.],
            )
            .unwrap()),
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
            Ok(Array::from_elements(ArrayType::new_static(DataType::I32, [2]).with_memory(memory), &[30_i32, 10])
                .unwrap()),
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
            .unwrap(),
        );
    }

    #[test]
    fn test_indexed_set() {
        let input = Array::matrix(2, 3, vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let updates = Array::scalar(9_i32).unwrap();
        assert_eq!(
            input.at(&index![.., 1..3]).set(&updates, &ScatterOptions::new()),
            Array::matrix(2, 3, vec![0_i32, 9, 9, 3, 9, 9]),
        );
        let indices = Array::vector(vec![0_i32, 2]).unwrap();
        assert_eq!(
            input.at(&index![1, &indices]).set(&Array::vector(vec![7_i32, 8]).unwrap(), &ScatterOptions::new()),
            Array::matrix(2, 3, vec![0_i32, 1, 2, 7, 4, 8]),
        );
        assert_eq!(input, Array::matrix(2, 3, vec![0_i32, 1, 2, 3, 4, 5]).unwrap());
    }

    #[test]
    fn test_indexed_set_reversed_and_masked() {
        let input = Array::vector(vec![0_i32, 1, 2, 3, 4]).unwrap();
        assert_eq!(
            input.at(&index![..by - 2]).set(&Array::vector(vec![7_i32, 8, 9]).unwrap(), &ScatterOptions::new()),
            Array::vector(vec![9_i32, 1, 8, 3, 7]),
        );
        let mask = IndexMask::new(vec![5], vec![false, true, false, true, false]).unwrap();
        assert_eq!(
            input.at(&index![&mask]).set(&Array::scalar(6_i32).unwrap(), &ScatterOptions::new()),
            Array::vector(vec![0_i32, 6, 2, 6, 4]),
        );
    }

    #[test]
    fn test_indexed_set_bounds() {
        // Negative coordinates count from the end once; coordinates that remain invalid are dropped under `Drop`.
        let input = Array::vector(vec![0_i32, 1, 2, 3, 4]).unwrap();
        let invalid = Array::vector(vec![-6_i32, 5]).unwrap();
        assert_eq!(
            input
                .at(&index![&invalid])
                .set(&Array::scalar(9_i32).unwrap(), &ScatterOptions::new().with_mode(ScatterMode::Drop)),
            Ok(input),
        );

        // Unsigned extremes are decoded exactly rather than reinterpreted as negative coordinates.
        let input = Array::vector(vec![10_i32, 20, 30]).unwrap();
        let indices = Array::vector(vec![0_u64, 2, u64::MAX]).unwrap();
        assert_eq!(
            input
                .at(&index![&indices])
                .set(&Array::scalar(7_i32).unwrap(), &ScatterOptions::new().with_mode(ScatterMode::Drop)),
            Array::vector(vec![7_i32, 20, 7]),
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
        assert_eq!(input.at(&index![..]).set(&updates, &ScatterOptions::new()), Ok(input));
    }

    #[test]
    fn test_indexed_add() {
        let input = Array::vector(vec![10_i32, 20, 30]).unwrap();
        let indices = Array::vector(vec![1_i32, 1, -1]).unwrap();
        let updates = Array::vector(vec![2_i32, 3, 4]).unwrap();
        assert_eq!(
            input.at(&index![&indices]).add(&updates, &ScatterOptions::new()),
            Array::vector(vec![10_i32, 25, 34]),
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

        // The unreduced input and updates agree, so the scattered result carries the same reduction state.
        assert_eq!(
            input.at(&index![&indices]).add(&updates, &ScatterOptions::new()),
            Ok(Array::from_elements(
                ArrayType::new_static(DataType::F64, [3]).with_sharding(sharding).unwrap(),
                &[17_f64, 20., 35.],
            )
            .unwrap()),
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
    fn test_indexed_mul() {
        let input = Array::vector(vec![10_i32, 20, 30]).unwrap();
        let indices = Array::vector(vec![1_i32, 1, -1]).unwrap();
        let updates = Array::vector(vec![2_i32, 3, 4]).unwrap();
        assert_eq!(
            input.at(&index![&indices]).mul(&updates, &ScatterOptions::new()),
            Array::vector(vec![10_i32, 120, 120]),
        );
    }

    #[test]
    fn test_indexed_min() {
        let input = Array::vector(vec![10_i32, 20, 30]).unwrap();
        let indices = Array::vector(vec![1_i32, 1, -1]).unwrap();
        let updates = Array::vector(vec![2_i32, 3, 4]).unwrap();
        assert_eq!(
            input.at(&index![&indices]).min(&updates, &ScatterOptions::new()),
            Array::vector(vec![10_i32, 2, 4]),
        );
    }

    #[test]
    fn test_indexed_max() {
        let input = Array::vector(vec![10_i32, 20, 30]).unwrap();
        let indices = Array::vector(vec![1_i32, 1, -1]).unwrap();
        let updates = Array::vector(vec![2_i32, 3, 4]).unwrap();
        assert_eq!(
            input.at(&index![&indices]).max(&updates, &ScatterOptions::new()),
            Array::vector(vec![10_i32, 20, 30]),
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

        // Only array inputs can be indexed.
        assert_eq!(
            ArrayIrValue::<Array>::Dimension(DimensionValue::constant(1).unwrap())
                .at(&index![0])
                .get(&GatherOptions::new()),
            Err(TypeError::invalid("expected array type but got dimension type").into()),
        );

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
                indices_type,
            ),
            Err(TypeError::invalid(
                "symbolic indexing does not yet support explicit output sharding or index promises",
            )
            .into()),
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
                input_type.clone(),
                ArrayIrType::Array(ArrayType::new_static(DataType::I32, [2])),
                ArrayIrType::Array(ArrayType::scalar(DataType::F64)),
            ),
        )
        .unwrap();
        assert_eq!(
            program.interpret((matrix, indices, ArrayIrValue::Array(Array::scalar(5_f64).unwrap()))),
            Ok(ArrayIrValue::Array(Array::matrix(2, 3, vec![5_f64, 20., 5., 5., 50., 5.]).unwrap())),
        );

        // Explicit placement and index promises are not remapped by the symbolic frontend yet,
        // for updates as for reads.
        assert_eq!(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
                |(input, indices, updates)| input
                    .at(&index![.., &indices])
                    .set(&updates, &ScatterOptions::new().with_unique_indices(true)),
                (
                    input_type,
                    ArrayIrType::Array(ArrayType::new_static(DataType::I32, [2])),
                    ArrayIrType::Array(ArrayType::scalar(DataType::F64)),
                ),
            )
            .map(|_| ()),
            Err(TypeError::invalid(
                "symbolic indexing does not yet support explicit output sharding or index promises",
            )
            .into()),
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
    fn test_indexed_max_staging_symbolic() {
        // A reduction kind other than overwrite follows the same reversed scatter and reversal back.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(6)).unwrap());
        let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(input, indices, updates)| input.at(&index![..by - 1, &indices]).max(&updates, &ScatterOptions::new()),
            (
                ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(rows), 3.into()]))),
                ArrayIrType::Array(ArrayType::new_static(DataType::I32, [2])),
                ArrayIrType::Array(ArrayType::new_static(DataType::F64, [2])),
            ),
        )
        .unwrap();
        assert_eq!(
            program.interpret((
                ArrayIrValue::Array(Array::matrix(2, 3, vec![10_f64, 20., 30., 40., 50., 60.]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![-1_i32, 0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![70_f64, 80.]).unwrap()),
            )),
            Ok(ArrayIrValue::Array(Array::matrix(2, 3, vec![80_f64, 20., 70., 80., 50., 70.]).unwrap())),
        );
    }
    #[test]
    fn test_indexed_view() {
        let buffer = reference_matrix();

        // Host integers remove their axis, slices keep theirs, an ellipsis and omitted trailing axes select in full,
        // and negative coordinates count from the end.
        let view = buffer.at(&index![1]).view().unwrap();
        assert_eq!(view.read(), Ok(ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]).unwrap())));
        let view = buffer.at(&index![.., 1..3]).view().unwrap();
        assert_eq!(view.read(), Ok(ArrayIrValue::Array(Array::matrix(2, 2, vec![2.0_f32, 3.0, 5.0, 6.0]).unwrap())));
        let view = buffer.at(&index![-1, 1..]).view().unwrap();
        assert_eq!(view.read(), Ok(ArrayIrValue::Array(Array::vector(vec![5.0_f32, 6.0]).unwrap())));
        let view = buffer.at(&index![..., -1]).view().unwrap();
        assert_eq!(view.read(), Ok(ArrayIrValue::Array(Array::vector(vec![3.0_f32, 6.0]).unwrap())));

        // Full selections derive no view transform and hand back the input allocation.
        let view = buffer.at(&index![.., 0..3]).view().unwrap();
        assert_eq!(view.r#type(), buffer.r#type());
        assert_eq!(view.read(), buffer.read());
        assert_eq!(buffer.at(&[]).view().unwrap().read(), buffer.read());

        // Scalar integer arrays select one position at run time, clamped into bounds.
        let index = ArrayIrValue::Array(Array::scalar(1_i32).unwrap());
        let view = buffer.at(&index![&index, 2]).view().unwrap();
        assert_eq!(view.read(), Ok(ArrayIrValue::Array(Array::scalar(6.0_f32).unwrap())));
        let index = ArrayIrValue::Array(Array::scalar(7_i32).unwrap());
        let view = buffer.at(&index![.., &index]).view().unwrap();
        assert_eq!(view.read(), Ok(ArrayIrValue::Array(Array::vector(vec![3.0_f32, 6.0]).unwrap())));

        // Views alias the allocation, so writes through the input are visible through an earlier view.
        let view = buffer.at(&index![0, 0]).view().unwrap();
        buffer.write(&ArrayIrValue::Array(Array::matrix(2, 3, vec![9.0_f32; 6]).unwrap())).unwrap();
        assert_eq!(view.read(), Ok(ArrayIrValue::Array(Array::scalar(9.0_f32).unwrap())));
    }

    #[test]
    fn test_indexed_view_validation() {
        // Unsupported selector kinds, malformed selector lists, and out-of-range host integers are rejected before any
        // view is derived.
        let buffer = reference_matrix();
        assert_eq!(
            buffer.at(&index![new_axis]).view(),
            Err(TypeError::invalid("reference views cannot insert axes").into()),
        );
        let mask = IndexMask::new(vec![2], vec![true, false]).unwrap();
        assert_eq!(
            buffer.at(&index![&mask]).view(),
            Err(TypeError::invalid("reference views do not support index masks").into()),
        );
        assert_eq!(
            buffer.at(&index![.., .., ..]).view(),
            Err(TypeError::invalid("index selection consumes 3 axes but input rank is 2").into()),
        );
        assert_eq!(
            buffer.at(&index![..., ...]).view(),
            Err(TypeError::invalid("index selection contains more than one ellipsis").into()),
        );
        assert_eq!(
            buffer.at(&index![2]).view(),
            Err(TypeError::invalid("index 2 is out of bounds for axis 0 with extent 2").into()),
        );
        assert_eq!(
            buffer.at(&index![.., -4]).view(),
            Err(TypeError::invalid("index -4 is out of bounds for axis 1 with extent 3").into()),
        );
        assert_eq!(
            buffer.at(&index![..by - 1]).view(),
            Err(TypeError::invalid("reference views do not support reversed slices").into()),
        );
        assert_eq!(
            buffer.at(&index![.., .. by 2]).view(),
            Err(TypeError::invalid("reference views do not support strided slices").into()),
        );
        let vector = ArrayIrValue::Array(Array::vector(vec![0_i32]).unwrap());
        assert_eq!(
            buffer.at(&index![&vector]).view(),
            Err(TypeError::invalid("reference views support only scalar integer index arrays but got `i32[1]`").into()),
        );
        let float = ArrayIrValue::Array(Array::scalar(0.5_f32).unwrap());
        assert_eq!(
            buffer.at(&index![&float]).view(),
            Err(TypeError::invalid("reference views support only scalar integer index arrays but got `f32[]`").into()),
        );

        // Array inputs have no reference to view.
        let array = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap());
        assert_eq!(
            array.at(&index![0]).view(),
            Err(TypeError::invalid("expected reference type but got array type").into()),
        );

        // Axes touched by host integers or slices must have static extents.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(4)).unwrap());
        let dynamic_type = ArrayIrType::Array(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(rows), Dimension::Static(3)]),
        ));
        let static_extent_error =
            Err(TypeError::invalid("reference views require a static extent on axis 0 but got `f32[rows, 3]`").into());
        assert_eq!(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
                |input| input.reference_new()?.at(&index![0]).view().map(|_| ()),
                dynamic_type.clone(),
            )
            .map(|_| ()),
            static_extent_error,
        );
        assert_eq!(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
                |input| input.reference_new()?.at(&index![.., 1..3]).view().map(|_| ()),
                dynamic_type,
            )
            .map(|_| ()),
            static_extent_error,
        );
    }

    #[test]
    fn test_indexed_view_staging() {
        let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(input, replacement, index): (ArrayIrTracer, ArrayIrTracer, ArrayIrTracer)| {
                let buffer = input.reference_new()?;
                buffer.at(&index![1..3]).write(&replacement)?;
                buffer.at(&index![..2]).add_update(&replacement)?;
                let element = buffer.at(&index![&index]).read()?;
                let last = buffer.at(&index![-1]).swap(&element)?;
                Ok((last, buffer.freeze()?))
            },
            (
                ArrayIrType::Array(ArrayType::new_static(DataType::F32, [4])),
                ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2])),
                ArrayIrType::Array(ArrayType::scalar(DataType::I32)),
            ),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4], %1:f32[2], %2:i32[] .
                let %3:ref<f32[4]> = reference_new %0
                    %4:ref<f32[2]> = reference_slice [axes=[ArraySliceAxis { start: 1, size: 2, stride: 1 }]] %3
                    () = reference_write %4 %1
                    %5:ref<f32[2]> = reference_slice [axes=[ArraySliceAxis { start: 0, size: 2, stride: 1 }]] %3
                    () = reference_add_update %5 %1
                    %6:ref<f32[]> = reference_dynamic_index [axis=0] %3 %2
                    %7:f32[] = reference_read %6
                    %8:ref<f32[]> = reference_index [axis=0, index=3] %3
                    %9:f32[] = reference_swap %8 %7
                    %10:f32[4] = reference_freeze %3
                in (%9, %10)"},
        );
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap());
        let replacement = ArrayIrValue::Array(Array::vector(vec![10.0_f32, 20.0]).unwrap());
        let index = ArrayIrValue::Array(Array::scalar(0_i32).unwrap());
        assert_eq!(
            program.interpret((input, replacement, index)),
            Ok((
                ArrayIrValue::Array(Array::scalar(4.0_f32).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![11.0_f32, 30.0, 20.0, 11.0]).unwrap()),
            )),
        );
    }

    #[test]
    fn test_indexed_read() {
        let buffer = reference_matrix();
        assert_eq!(
            buffer.at(&index![1, 1..]).read(),
            Ok(ArrayIrValue::Array(Array::vector(vec![5.0_f32, 6.0]).unwrap())),
        );

        // Reads observe the reference state at the time of the read.
        buffer.write(&ArrayIrValue::Array(Array::matrix(2, 3, vec![0.0_f32; 6]).unwrap())).unwrap();
        assert_eq!(
            buffer.at(&index![1, 1..]).read(),
            Ok(ArrayIrValue::Array(Array::vector(vec![0.0_f32, 0.0]).unwrap())),
        );
    }

    #[test]
    fn test_indexed_write() {
        let buffer = reference_matrix();
        buffer
            .at(&index![1, 1..])
            .write(&ArrayIrValue::Array(Array::vector(vec![50.0_f32, 60.0]).unwrap()))
            .unwrap();
        buffer
            .at(&index![.., 0])
            .write(&ArrayIrValue::Array(Array::vector(vec![10.0_f32, 40.0]).unwrap()))
            .unwrap();
        assert_eq!(
            buffer.read(),
            Ok(ArrayIrValue::Array(Array::matrix(2, 3, vec![10.0_f32, 2.0, 3.0, 40.0, 50.0, 60.0]).unwrap())),
        );

        // The replacement must match the selected region's type.
        assert!(
            buffer
                .at(&index![1])
                .write(&ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap()))
                .is_err(),
        );
    }

    #[test]
    fn test_indexed_add_update() {
        let buffer = reference_matrix();
        buffer
            .at(&index![0])
            .add_update(&ArrayIrValue::Array(Array::vector(vec![1.0_f32, 1.0, 1.0]).unwrap()))
            .unwrap();
        assert_eq!(
            buffer.read(),
            Ok(ArrayIrValue::Array(Array::matrix(2, 3, vec![2.0_f32, 3.0, 4.0, 4.0, 5.0, 6.0]).unwrap())),
        );
    }

    #[test]
    fn test_indexed_swap() {
        let buffer = reference_matrix();
        assert_eq!(
            buffer
                .at(&index![.., 1..3])
                .swap(&ArrayIrValue::Array(Array::matrix(2, 2, vec![0.0_f32; 4]).unwrap())),
            Ok(ArrayIrValue::Array(Array::matrix(2, 2, vec![2.0_f32, 3.0, 5.0, 6.0]).unwrap())),
        );
        assert_eq!(
            buffer.read(),
            Ok(ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 0.0, 0.0, 4.0, 0.0, 0.0]).unwrap())),
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
        assert_eq!(selection, [IndexSelector::Basic(BasicIndex::Slice(IndexSlice::new(Some(1), Some(9), 2)))]);
        let grouped: [IndexSelector<'_, Array>; 1] = index![(|first, second| first + second)(1, 2)..9 by 2];
        assert_eq!(grouped, [IndexSelector::Basic(BasicIndex::Slice(IndexSlice::new(Some(3), Some(9), 2)))]);
    }

    #[test]
    fn test_index_array_borrow() {
        let rows = Array::vector(vec![0_i32, 2]).unwrap();
        let selection = index![&rows, ...];
        assert_eq!(selection, [IndexSelector::Array(&rows), IndexSelector::Basic(BasicIndex::Ellipsis)]);

        // Constructing descriptors keeps the original array available to the caller.
        assert_eq!(rows, Array::vector(vec![0_i32, 2]).unwrap());
    }
}
