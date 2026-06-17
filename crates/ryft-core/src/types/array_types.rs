use std::borrow::Cow;
use std::fmt::Display;
use std::ops::Index;

use ryft_macros::Parameter;

use crate::Error;
use crate::broadcasting::Broadcastable;
use crate::parameters::Parameter;
use crate::programs::Value;
use crate::sharding::{DeviceMesh, Sharding, ShardingDimension, ShardingError};
use crate::types::data_types::DataType;
use crate::types::layouts::Layout;
use crate::types::memories::Memory;
use crate::types::{Type, TypeError, Typed};

/// Represents the size of an array dimension. Array dimensions can be either statically known at compilation time or
/// dynamic, in which case their sizes will only be known at runtime. Dynamic dimensions may optionally have an upper
/// bound for their size that may be used for optimizations by the compiler. Note that by compilation here we do not
/// refer to the compilation of the Rust program but rather to the compilation of an array program within our Rust
/// library.
///
/// Note that the [`Display`] implementation of [`Size`] renders static sizes as just a number, dynamic sizes
/// with an upper bound as `<` followed by the upper bound, and dynamic sizes with no upper bound as `*`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum Size {
    /// Static size that is known at compilation time.
    Static(usize),

    /// Dynamic size that is not known until runtime and which has an optional upper bound. The upper bound, if present,
    /// represents an exclusive upper bound on the value that this size can have (i.e., the maximum possible value plus
    /// one). This can enable certain optimizations and static checks (though, of course, not as powerful as what a
    /// static size enables).
    Dynamic(Option<usize>),
}

impl Size {
    /// Returns the value of this [`Size`] if it is a [`Size::Static`] and `None` otherwise.
    #[inline]
    pub fn value(&self) -> Option<usize> {
        match &self {
            Self::Static(size) => Some(*size),
            Self::Dynamic(_) => None,
        }
    }

    /// Returns an (exclusive) upper bound for the value of this [`Size`] if such a bound is known. For [`Size::Static`]
    /// sizes, this function will return the underlying value plus one as the upper bound. For [`Size::Dynamic`] sizes,
    /// this function will return the upper bound for that size if one exists, and `None` otherwise.
    #[inline]
    pub fn upper_bound(&self) -> Option<usize> {
        match &self {
            Self::Static(size) => Some(*size + 1),
            Self::Dynamic(upper_bound) => *upper_bound,
        }
    }

    /// Returns `true` if `other` refines this [`Size`] (i.e., if every concrete size allowed by `other` is also allowed
    /// by this [`Size`]). The receiver is the more general size (e.g., a declared size), and the argument is the more
    /// precise one (e.g., the one carried by a runtime value's type), and so the relation is directional:
    /// `declared.is_refined_by(&actual)`.
    ///
    /// The relation is defined as follows, recalling that the upper bound carried by [`Size::Dynamic`] is *exclusive*:
    ///
    /// - `Static(n)` is refined by `Static(m)` only when `n == m`.
    /// - `Static(_)` is never refined by `Dynamic(_)`.
    /// - `Dynamic(None)` is refined by every size.
    /// - `Dynamic(Some(bound))` is refined by `Static(m)` only when `m < bound`.
    /// - `Dynamic(Some(bound))` is refined by `Dynamic(Some(other))` only when `other <= bound`.
    /// - `Dynamic(Some(_))` is never refined by `Dynamic(None)`.
    #[inline]
    pub fn is_refined_by(&self, other: &Size) -> bool {
        match (self, other) {
            (Self::Static(declared), Self::Static(actual)) => declared == actual,
            (Self::Static(_), Self::Dynamic(_)) => false,
            (Self::Dynamic(None), _) => true,
            (Self::Dynamic(Some(bound)), Self::Static(actual)) => actual < bound,
            (Self::Dynamic(Some(bound)), Self::Dynamic(Some(other_bound))) => other_bound <= bound,
            (Self::Dynamic(Some(_)), Self::Dynamic(None)) => false,
        }
    }
}

impl Display for Size {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            Self::Static(size) => write!(formatter, "{size}"),
            Self::Dynamic(Some(upper_bound)) => write!(formatter, "<{upper_bound}"),
            Self::Dynamic(None) => write!(formatter, "*"),
        }
    }
}

impl From<usize> for Size {
    fn from(value: usize) -> Self {
        Self::Static(value)
    }
}

/// Represents the shape of an array (i.e., the number of dimensions in the array and the [`Size`] of each dimension).
///
/// Note that the [`Display`] implementation of [`Shape`] renders shapes as the rendered dimension sizes
/// in a comma-separated list surrounded by square brackets.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct Shape {
    /// [`Size`]s of the array dimensions ordered from outermost to innermost.
    dimensions: Vec<Size>,
}

impl Shape {
    /// Constructs a new [`Shape`] with the provided dimension [`Size`]s.
    #[inline]
    pub fn new(dimensions: Vec<Size>) -> Self {
        Self { dimensions }
    }

    /// Constructs a new scalar [`Shape`]. The resulting [`Shape::dimensions`] will be empty.
    #[inline]
    pub fn scalar() -> Self {
        Self::new(Vec::new())
    }

    /// Returns the [`Size`]s of the array dimensions ordered from outermost to innermost.
    #[inline]
    pub fn dimensions(&self) -> &[Size] {
        self.dimensions.as_slice()
    }

    /// Returns the rank (i.e., the number of dimensions) of this [`Shape`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::types::{Shape, Size};
    ///
    /// // Scalar.
    /// assert_eq!(Shape::scalar().rank(), 0);
    ///
    /// // Vector with 42 elements.
    /// assert_eq!(Shape::new(vec![Size::Static(42)]).rank(), 1);
    ///
    /// // Matrix with 42 rows and up to 10 columns.
    /// assert_eq!(Shape::new(vec![Size::Static(42), Size::Dynamic(Some(10))]).rank(), 2);
    ///
    /// // Matrix with an unknown number of rows and 42 columns.
    /// assert_eq!(Shape::new(vec![Size::Dynamic(None), Size::Static(42)]).rank(), 2);
    /// ```
    #[inline]
    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }

    /// Returns the [`Size`] of the `index`-th dimension of this [`Shape`]. A negative `index` can be used to obtain
    /// dimension sizes using the end of the dimensions vector as the reference point. For example, an index value of
    /// `-1` will result in the last dimension (i.e., innermost) `Size` being returned.
    #[inline]
    pub fn dimension(&self, index: isize) -> Size {
        if index >= 0 {
            self.dimensions[index as usize]
        } else {
            self.dimensions[(self.dimensions.len() as isize + index) as usize]
        }
    }

    /// Returns the number of elements in arrays with this [`Shape`] or `Ok(None)` if any of its dimensions is
    /// dynamic. Returns an [`Error`] wrapping a [`TypeError`] if the static element count does not fit in [`usize`].
    #[inline]
    pub fn element_count(&self) -> Result<Option<usize>, Error> {
        let mut count = 1usize;
        for size in &self.dimensions {
            match size {
                Size::Static(size) => {
                    count = count.checked_mul(*size).ok_or_else(|| TypeError {
                        message: format!("shape {self} element count does not fit in usize"),
                    })?;
                }
                Size::Dynamic(_) => return Ok(None),
            }
        }
        Ok(Some(count))
    }

    /// Returns `true` if every [`Shape`] admitted by `other` is also admitted by this [`Shape`]. The receiver is the
    /// more general shape (e.g., a declared shape), and the argument is the more precise one (e.g., the one carried by
    /// a runtime value's type), and so the relation is directional: `declared.is_refined_by(&actual)`. The two shapes
    /// must have equal rank and every dimension [`Size`] of the receiver must be refined by the corresponding dimension
    /// of `other` per [`Size::is_refined_by`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::types::{Shape, Size};
    ///
    /// let declared = Shape::new(vec![Size::Dynamic(None), Size::Static(3)]);
    ///
    /// // Dynamic declared dimensions are refined by any static size, while static ones require equality.
    /// assert!(declared.is_refined_by(&Shape::new(vec![Size::Static(2), Size::Static(3)])));
    /// assert!(declared.is_refined_by(&Shape::new(vec![Size::Static(5), Size::Static(3)])));
    /// assert!(!declared.is_refined_by(&Shape::new(vec![Size::Static(2), Size::Static(4)])));
    ///
    /// // The ranks must match exactly.
    /// assert!(!declared.is_refined_by(&Shape::new(vec![Size::Static(3)])));
    /// ```
    #[inline]
    pub fn is_refined_by(&self, other: &Shape) -> bool {
        self.rank() == other.rank()
            && self
                .dimensions
                .iter()
                .zip(other.dimensions.iter())
                .all(|(declared, actual)| declared.is_refined_by(actual))
    }
}

impl Display for Shape {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "[{}]",
            self.dimensions.iter().map(|dimension| dimension.to_string()).collect::<Vec<_>>().join(", ")
        )
    }
}

impl Index<usize> for Shape {
    type Output = Size;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.dimensions[index]
    }
}

impl Index<isize> for Shape {
    type Output = Size;

    /// Indexes into [`Self::dimensions`] with support for negative indices. A negative index `i` resolves to
    /// `self.dimensions.len() as isize + i`, so `shape[-1]` returns the innermost dimension.
    #[inline]
    fn index(&self, index: isize) -> &Self::Output {
        let normalized = if index >= 0 { index } else { self.dimensions.len() as isize + index };
        &self.dimensions[normalized as usize]
    }
}

/// Represents the shape of an array (i.e., the number of dimensions in the array and the [`Size`] of each dimension),
/// whose dimension [`Size`]s are all [`Size::Static`] (in contrast to [`Shape`] which supports dynamic dimensions).
///
/// Note that the [`Display`] implementation of [`StaticShape`] renders shapes as the rendered dimension sizes
/// in a comma-separated list surrounded by square brackets.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct StaticShape {
    /// Static dimension sizes ordered from outermost to innermost.
    dimensions: Vec<usize>,
}

impl StaticShape {
    /// Constructs a new [`StaticShape`] with the provided static dimension sizes.
    #[inline]
    pub fn new(dimensions: Vec<usize>) -> Self {
        Self { dimensions }
    }

    /// Constructs a new scalar [`StaticShape`]. The resulting [`StaticShape::dimensions`] will be empty.
    #[inline]
    pub fn scalar() -> Self {
        Self::new(Vec::new())
    }

    /// Returns the static dimension sizes ordered from outermost to innermost.
    #[inline]
    pub fn dimensions(&self) -> &[usize] {
        self.dimensions.as_slice()
    }

    /// Returns the rank (i.e., the number of dimensions) of this [`StaticShape`].
    #[inline]
    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }

    /// Returns the size of the `index`-th dimension of this [`StaticShape`]. A negative `index` can be used to obtain
    /// dimension sizes using the end of the dimensions vector as the reference point. For example, an index value of
    /// `-1` will result in the last dimension (i.e., innermost) size being returned.
    #[inline]
    pub fn dimension(&self, index: isize) -> usize {
        if index >= 0 {
            self.dimensions[index as usize]
        } else {
            self.dimensions[(self.dimensions.len() as isize + index) as usize]
        }
    }

    /// Returns the static dimension sizes as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[usize] {
        &self.dimensions
    }

    /// Returns the row-major (i.e., the last axis corresponds to the fastest moving index) strides over element indices
    /// for arrays with this [`StaticShape`].
    pub fn row_major_strides(&self) -> Vec<usize> {
        let mut strides = vec![0usize; self.dimensions.len()];
        if self.dimensions.is_empty() {
            return strides;
        }
        let mut stride = 1usize;
        for axis in (0..self.dimensions.len()).rev() {
            strides[axis] = stride;
            stride *= self.dimensions[axis];
        }
        strides
    }
}

impl Display for StaticShape {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "[{}]",
            self.dimensions.iter().map(|dimension| dimension.to_string()).collect::<Vec<_>>().join(", ")
        )
    }
}

impl Index<usize> for StaticShape {
    type Output = usize;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.dimensions[index]
    }
}

impl Index<isize> for StaticShape {
    type Output = usize;

    /// Indexes into [`Self::dimensions`] with support for negative indices. A negative index `i` resolves to
    /// `self.dimensions.len() as isize + i`, so `shape[-1]` returns the innermost dimension.
    #[inline]
    fn index(&self, index: isize) -> &Self::Output {
        let normalized = if index >= 0 { index } else { self.dimensions.len() as isize + index };
        &self.dimensions[normalized as usize]
    }
}

impl From<StaticShape> for Shape {
    fn from(value: StaticShape) -> Self {
        Self::new(value.dimensions.into_iter().map(Size::Static).collect())
    }
}

impl From<&StaticShape> for Shape {
    fn from(value: &StaticShape) -> Self {
        Self::new(value.dimensions.iter().copied().map(Size::Static).collect())
    }
}

impl TryFrom<Shape> for StaticShape {
    type Error = TypeError;

    fn try_from(value: Shape) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl TryFrom<&Shape> for StaticShape {
    type Error = TypeError;

    fn try_from(value: &Shape) -> Result<Self, Self::Error> {
        let mut dimensions = Vec::with_capacity(value.dimensions.len());
        for (dimension, size) in value.dimensions.iter().enumerate() {
            match size {
                Size::Static(size) => dimensions.push(*size),
                Size::Dynamic(_) => {
                    return Err(TypeError {
                        message: format!("shape dimension {dimension} must be static, but got {size}"),
                    });
                }
            }
        }
        Ok(Self::new(dimensions))
    }
}

/// Represents the [`Type`] of a potentially multi-dimensional array.
///
/// Note that the [`Display`] implementation of [`ArrayType`] renders array types simply as their [`DataType`]s
/// followed by their [`Shape`]s, optionally followed by their [`Layout`] and [`Sharding`], if present, and finally
/// an `@` followed by the [`Memory`] space when the array resides outside the default [`Memory::Device`] memory.
///
/// # Examples
///
/// ```rust
/// # use ryft_core::{ArrayType, DataType, Memory, Shape, Size};
///
/// // Boolean scalar.
/// assert_eq!(
///   ArrayType::new(DataType::Boolean, Shape::scalar()).to_string(),
///   "bool[]",
/// );
///
/// // 32-bit floating-point number vector with 42 elements residing in pinned host memory.
/// assert_eq!(
///   ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(42)]))
///       .with_memory(Memory::Host { pinned: true })
///       .to_string(),
///   "f32[42]@Host[Pinned]",
/// );
///
/// // 64-bit unsigned integer vector with 42 elements.
/// assert_eq!(
///   ArrayType::new(DataType::U64, Shape::new(vec![Size::Static(42)])).to_string(),
///   "u64[42]",
/// );
///
/// // 32-bit floating-point number matrix with 42 rows and up to 10 columns.
/// assert_eq!(
///   ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(42), Size::Dynamic(Some(10))])).to_string(),
///   "f32[42, <10]",
/// );
///
/// // 64-bit complex number matrix with an unknown number of rows and 42 columns.
/// assert_eq!(
///   ArrayType::new(DataType::C64, Shape::new(vec![Size::Dynamic(None), Size::Static(42)])).to_string(),
///   "c64[*, 42]",
/// );
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct ArrayType {
    /// [`DataType`] of the elements stored in the array.
    pub(crate) data_type: DataType,

    /// [`Shape`] of the array.
    pub(crate) shape: Shape,

    /// Optional physical memory/storage [`Layout`] of the array.
    pub(crate) layout: Option<Layout>,

    /// Optional [`Sharding`] information about the array.
    pub(crate) sharding: Option<Sharding>,

    /// [`Memory`] in which the array resides. For sharded arrays, the memory applies uniformly to every shard,
    /// each residing in its own device's memory of this kind.
    pub(crate) memory: Memory,
}

impl ArrayType {
    /// Constructs a new [`ArrayType`] with the provided [`DataType`] and [`Shape`], no [`Layout`] or [`Sharding`]
    /// information, and residing in the default [`Memory::Device`] memory space. Use [`Self::with_layout`],
    /// [`Self::with_sharding`], and [`Self::with_memory`] to attach optional metadata.
    #[inline]
    pub fn new(data_type: DataType, shape: Shape) -> Self {
        Self { data_type, shape, layout: None, sharding: None, memory: Memory::Device }
    }

    /// Returns this [`ArrayType`] with the provided physical memory/storage [`Layout`] replacing its current layout
    /// (or without any [`Layout`] information when [`None`] is provided).
    #[inline]
    pub fn with_layout(mut self, layout: impl Into<Option<Layout>>) -> Self {
        self.layout = layout.into();
        self
    }

    /// Returns a copy of this [`ArrayType`] with the provided [`Sharding`] replacing its current sharding metadata
    /// (or without any [`Sharding`] information when [`None`] is provided), after validating that any provided
    /// [`Sharding`] has the same rank as [`Self::shape`].
    #[inline]
    pub fn with_sharding(mut self, sharding: impl Into<Option<Sharding>>) -> Result<Self, ShardingError> {
        let sharding = sharding.into();
        if let Some(sharding) = &sharding {
            let sharding_rank = sharding.rank();
            let array_rank = self.shape.rank();
            if sharding_rank != array_rank {
                return Err(ShardingError::ShardingRankMismatch { sharding_rank, array_rank });
            }
        }
        self.sharding = sharding;
        Ok(self)
    }

    /// Returns a copy of this [`ArrayType`] with the array residing in the provided [`Memory`]. This kind of placement
    /// information is metadata about where the array lives, and it does not affect the array's [`DataType`], [`Shape`],
    /// [`Layout`], or [`Sharding`] (for sharded arrays, every shard resides in its own device's memory of this kind).
    #[inline]
    pub fn with_memory(mut self, memory: Memory) -> Self {
        self.memory = memory;
        self
    }

    /// Constructs a new "scalar" [`ArrayType`] with the provided [`DataType`]. The resulting [`ArrayType::shape`]
    /// will be a scalar (i.e., have rank 0).
    #[inline]
    pub fn scalar(data_type: DataType) -> Self {
        Self { data_type, shape: Shape::scalar(), layout: None, sharding: None, memory: Memory::default() }
    }

    /// Returns the [`DataType`] of the elements stored in the array.
    #[inline]
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Returns the [`Shape`] of the array.
    #[inline]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns a new [`StaticShape`] for this [`ArrayType`] if all dimensions of [`Self::shape`] have static size.
    /// This method computes the [`StaticShape`] from [`Self::shape`] on each call and returns an owned value, not a
    /// reference to cached shape metadata.
    #[inline]
    pub fn static_shape(&self) -> Option<StaticShape> {
        self.shape.dimensions().iter().map(Size::value).collect::<Option<Vec<_>>>().map(StaticShape::new)
    }

    /// Returns the rank (i.e., the number of dimensions) of this [`ArrayType`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::types::DataType;
    /// # use ryft_core::types::{ArrayType, Shape, Size};
    ///
    /// // Boolean scalar.
    /// assert_eq!(ArrayType::new(DataType::Boolean, Shape::scalar()).rank(), 0);
    ///
    /// // 64-bit unsigned integer vector with 42 elements.
    /// assert_eq!(ArrayType::new(DataType::U64, Shape::new(vec![Size::Static(42)])).rank(), 1);
    ///
    /// // 32-bit floating-point number matrix with 42 rows and up to 10 columns.
    /// assert_eq!(
    ///     ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(42), Size::Dynamic(Some(10))])).rank(),
    ///     2,
    /// );
    ///
    /// // 64-bit complex number matrix with an unknown number of rows and 42 columns.
    /// assert_eq!(
    ///     ArrayType::new(DataType::C64, Shape::new(vec![Size::Dynamic(None), Size::Static(42)])).rank(),
    ///     2,
    /// );
    /// ```
    #[inline]
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Returns the [`Size`] of the `index`-th dimension of this array type's [`Shape`]. A negative `index` can be used
    /// to obtain dimension sizes using the end of the dimensions vector as the reference point. For example, an index
    /// value of `-1` will result in the last dimension (i.e., innermost) `Size` being returned.
    #[inline]
    pub fn dimension(&self, index: isize) -> Size {
        self.shape.dimension(index)
    }

    /// Returns the number of elements in arrays of this [`ArrayType`] or `Ok(None)` if any dimension in [`Self::shape`]
    /// is dynamic. Returns an [`Error`] wrapping a [`TypeError`] if the static element count does not fit in [`usize`].
    #[inline]
    pub fn element_count(&self) -> Result<Option<usize>, Error> {
        self.shape.element_count()
    }

    /// Returns the physical memory/storage [`Layout`] of the array if it is known.
    #[inline]
    pub fn layout(&self) -> Option<&Layout> {
        self.layout.as_ref()
    }

    /// Returns [`Sharding`] information about the array if it is known.
    #[inline]
    pub fn sharding(&self) -> Option<&Sharding> {
        self.sharding.as_ref()
    }

    /// Returns the [`Memory`] space in which the array resides.
    #[inline]
    pub fn memory(&self) -> Memory {
        self.memory
    }

    /// Returns a copy of this [`ArrayType`] with a dimension inserted at the provided index. Rank-changing operations
    /// clear explicit [`Layout`] information because [`Layout`]s do not carry enough information to infer a correct
    /// stride or tiling for a newly inserted logical axis. [`Sharding`] information is preserved by inserting a
    /// replicated sharding dimension at the same index and shifting the existing dimension annotations.
    pub fn with_inserted_dimension(&self, index: usize, size: Size) -> Result<Self, TypeError> {
        if index > self.rank() {
            return Err(TypeError {
                message: format!("cannot insert dimension at index {index} for rank-{} array type", self.rank()),
            });
        }
        let mut dimensions = self.shape.dimensions.clone();
        dimensions.insert(index, size);
        
        // TODO(eaplatanios): Review this portion.
        // The inserted array dimension is replicated; reuse the sharding-level insertion so this method stays a thin
        // structural wrapper (refer to the documentation of [`Sharding::inserting_dimension`]).
        let sharding = self
            .sharding
            .as_ref()
            .map(|sharding| sharding.inserting_dimension(index, ShardingDimension::Replicated))
            .transpose()
            .map_err(|error| TypeError { message: error.to_string() })?;

        Ok(Self {
            data_type: self.data_type,
            shape: Shape::new(dimensions),
            layout: None,
            sharding,
            memory: self.memory,
        })
    }

    /// Returns a copy of this [`ArrayType`] with its `axis`-th dimension removed, paired with the [`Size`] of the
    /// removed dimension. Rank-changing operations clear explicit [`Layout`] information because [`Layout`]s do not
    /// carry enough information to infer a correct stride or tiling after removing a logical axis. [`Sharding`]
    /// information is preserved when the removed dimension is replicated or unconstrained. When the removed dimension
    /// is sharded over manual mesh axes, those axes become varying manual axes because the value can still differ
    /// across shards even though the ranked array dimension is gone. Removing a dimension sharded over non-manual
    /// axes is rejected because there is no equivalent rank-independent metadata field for those axes.
    pub fn without_dimension(&self, axis: usize) -> Result<(Self, Size), TypeError> {
        if axis >= self.rank() {
            return Err(TypeError {
                message: format!("cannot remove dimension at index {axis} for rank-{} array type", self.rank()),
            });
        }
        let mut dimensions = self.shape.dimensions.clone();
        let dimension = dimensions.remove(axis);
        
        // TODO(eaplatanios): Review this portion.
        // Delegate the per-dimension sharding bookkeeping (manual axes become varying, non-manual sharded dimensions
        // cannot be dropped) to the sharding itself; refer to the documentation of [`Sharding::without_dimension`].
        let sharding = self
            .sharding
            .as_ref()
            .map(|sharding| sharding.without_dimension(axis))
            .transpose()
            .map_err(|error| TypeError { message: error.to_string() })?;

        Ok((
            Self {
                data_type: self.data_type,
                shape: Shape::new(dimensions),
                layout: None,
                sharding,
                memory: self.memory,
            },
            dimension,
        ))
    }

    /// Returns a copy of this [`ArrayType`] with a replicated [`Sharding`] over the provided [`DeviceMesh`]. The
    /// [`Layout`] information and [`Memory`] placement are preserved.
    pub fn replicated(&self, mesh: &DeviceMesh) -> Result<Self, ShardingError> {
        self.clone().with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), self.shape.rank()))
    }
}

impl Display for ArrayType {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}{}", self.data_type, self.shape)?;
        if let Some(layout) = &self.layout {
            write!(formatter, "[layout={layout}]")?;
        }
        if let Some(sharding) = &self.sharding {
            write!(formatter, "[sharding={sharding}]")?;
        }
        if self.memory != Memory::Device {
            write!(formatter, "@{}", self.memory)?;
        }
        Ok(())
    }
}

impl Type for ArrayType {
    #[inline]
    fn is_compatible_with(&self, other: &Self) -> bool {
        // Note that this compatibility relationship is defined here as a "broadcastability" relationship.
        self.is_broadcastable_to(other)
    }

    #[inline]
    fn is_refined_by(&self, other: &Self) -> bool {
        self.data_type == other.data_type
            && self.shape.is_refined_by(&other.shape)
            && self.layout == other.layout
            && self.sharding == other.sharding
            && self.memory == other.memory
    }

    #[inline]
    fn is_scalar(&self) -> bool {
        self.rank() == 0
    }
}

// Some staged XLA programs use `ArrayType` itself as the value carrier (e.g., with `T = ArrayType` and
// `V = ArrayType`) because the program stores boundary metadata rather than runtime arrays. In that mode the abstract
// value is self-describing: its value-type descriptor is itself. This is not a type-theoretic universe claim (i.e.,
// `ArrayType : ArrayType`). It is the `Typed` witness required by `Value<ArrayType>` for metadata-only program storage,
// lowering, and transformation.
impl Typed<ArrayType> for ArrayType {
    #[inline]
    fn r#type(&self) -> Cow<'_, ArrayType> {
        Cow::Borrowed(self)
    }
}

impl Value<ArrayType> for ArrayType {}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::Error;
    use crate::sharding::{
        Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension, ShardingError,
    };
    use crate::types::DataType::{BF16, Boolean, C64, F8E3M4, F8E4M3FN, F16, F32, F64};
    use crate::types::{
        ArrayType, Layout, Memory, Shape, Size, StaticShape, StridedLayout, Tile, TileDimension, TiledLayout, Type,
        TypeError,
    };

    #[test]
    fn test_size_value() {
        assert_eq!(Size::Static(1).value(), Some(1));
        assert_eq!(Size::Static(42).value(), Some(42));
        assert_eq!(Size::Dynamic(None).value(), None);
        assert_eq!(Size::Dynamic(Some(42)).value(), None);
    }

    #[test]
    fn test_size_upper_bound() {
        assert_eq!(Size::Static(1).upper_bound(), Some(2));
        assert_eq!(Size::Static(42).upper_bound(), Some(43));
        assert_eq!(Size::Dynamic(None).upper_bound(), None);
        assert_eq!(Size::Dynamic(Some(42)).upper_bound(), Some(42));
    }

    #[test]
    fn test_size_is_refined_by() {
        // Static declared sizes accept only equal static actual sizes.
        assert!(Size::Static(3).is_refined_by(&Size::Static(3)));
        assert!(!Size::Static(3).is_refined_by(&Size::Static(4)));
        assert!(!Size::Static(3).is_refined_by(&Size::Dynamic(None)));
        assert!(!Size::Static(3).is_refined_by(&Size::Dynamic(Some(3))));

        // Unbounded dynamic declared sizes accept every actual size.
        assert!(Size::Dynamic(None).is_refined_by(&Size::Static(0)));
        assert!(Size::Dynamic(None).is_refined_by(&Size::Static(42)));
        assert!(Size::Dynamic(None).is_refined_by(&Size::Dynamic(None)));
        assert!(Size::Dynamic(None).is_refined_by(&Size::Dynamic(Some(42))));

        // Bounded dynamic declared sizes accept static actual sizes strictly below the exclusive bound.
        assert!(Size::Dynamic(Some(4)).is_refined_by(&Size::Static(0)));
        assert!(Size::Dynamic(Some(4)).is_refined_by(&Size::Static(3)));
        assert!(!Size::Dynamic(Some(4)).is_refined_by(&Size::Static(4)));
        assert!(!Size::Dynamic(Some(4)).is_refined_by(&Size::Static(5)));

        // Bounded dynamic declared sizes accept bounded dynamic actual sizes with bounds at most as large,
        // and never accept unbounded dynamic actual sizes.
        assert!(Size::Dynamic(Some(4)).is_refined_by(&Size::Dynamic(Some(3))));
        assert!(Size::Dynamic(Some(4)).is_refined_by(&Size::Dynamic(Some(4))));
        assert!(!Size::Dynamic(Some(4)).is_refined_by(&Size::Dynamic(Some(5))));
        assert!(!Size::Dynamic(Some(4)).is_refined_by(&Size::Dynamic(None)));
    }

    #[test]
    fn test_size_to_string() {
        assert_eq!(Size::Static(1).to_string(), "1");
        assert_eq!(Size::Static(42).to_string(), "42");
        assert_eq!(Size::Dynamic(None).to_string(), "*");
        assert_eq!(Size::Dynamic(Some(42)).to_string(), "<42");
    }

    #[test]
    fn test_shape_rank() {
        let s0 = Shape::scalar();
        let s1 = Shape::new(vec![Size::Static(42)]);
        let s2 = Shape::new(vec![Size::Static(4), Size::Dynamic(None)]);

        assert_eq!(s0.rank(), 0);
        assert_eq!(s1.rank(), 1);
        assert_eq!(s2.rank(), 2);
    }

    #[test]
    fn test_shape_dimension() {
        let s0 = Shape::new(vec![Size::Static(42)]);
        let s1 = Shape::new(vec![Size::Static(4), Size::Dynamic(None)]);

        assert_eq!(s0.dimension(0), Size::Static(42));
        assert_eq!(s1.dimension(1), Size::Dynamic(None));
        assert_eq!(s1.dimension(-2), Size::Static(4));

        assert_eq!(s0[0usize], Size::Static(42));
        assert_eq!(s1[0usize], Size::Static(4));
        assert_eq!(s1[1usize], Size::Dynamic(None));
        assert_eq!(s1[-1isize], Size::Dynamic(None));
        assert_eq!(s1[-2isize], Size::Static(4));
    }

    #[test]
    fn test_shape_element_count() {
        assert_eq!(Shape::scalar().element_count(), Ok(Some(1)));
        assert_eq!(Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]).element_count(), Ok(Some(336)),);
        assert_eq!(Shape::new(vec![Size::Static(42), Size::Static(0)]).element_count(), Ok(Some(0)));
        assert_eq!(Shape::new(vec![Size::Static(42), Size::Dynamic(None)]).element_count(), Ok(None));
        assert_eq!(Shape::new(vec![Size::Static(42), Size::Dynamic(Some(8))]).element_count(), Ok(None));
        assert_eq!(
            Shape::new(vec![Size::Static(usize::MAX), Size::Static(2)]).element_count(),
            Err(Error::from(TypeError {
                message: format!("shape [{}, 2] element count does not fit in usize", usize::MAX),
            })),
        );
    }

    #[test]
    fn test_shape_is_refined_by() {
        let declared = Shape::new(vec![Size::Dynamic(None), Size::Static(3)]);

        // Pairwise size compatibility with matching ranks.
        assert!(declared.is_refined_by(&Shape::new(vec![Size::Static(2), Size::Static(3)])));
        assert!(declared.is_refined_by(&Shape::new(vec![Size::Static(5), Size::Static(3)])));
        assert!(declared.is_refined_by(&declared));
        assert!(!declared.is_refined_by(&Shape::new(vec![Size::Static(2), Size::Static(4)])));

        // Rank mismatches are always rejected.
        assert!(!declared.is_refined_by(&Shape::new(vec![Size::Static(3)])));
        assert!(!declared.is_refined_by(&Shape::new(vec![Size::Static(2), Size::Static(3), Size::Static(1)])));
        assert!(!declared.is_refined_by(&Shape::scalar()));
        assert!(Shape::scalar().is_refined_by(&Shape::scalar()));
    }

    #[test]
    fn test_shape_display() {
        let s0 = Shape::scalar();
        let s1 = Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]);
        let s2 = Shape::new(vec![Size::Static(4), Size::Static(1)]);
        let s3 = Shape::new(vec![Size::Static(4), Size::Dynamic(Some(1))]);
        let s4 = Shape::new(vec![Size::Dynamic(None), Size::Static(42), Size::Dynamic(None)]);
        let s5 = Shape::new(vec![Size::Static(42), Size::Dynamic(None)]);

        assert_eq!(format!("{s0}"), "[]");
        assert_eq!(format!("{s1}"), "[42, 4, 2]");
        assert_eq!(format!("{s2}"), "[4, 1]");
        assert_eq!(format!("{s3}"), "[4, <1]");
        assert_eq!(format!("{s4}"), "[*, 42, *]");
        assert_eq!(format!("{s5}"), "[42, *]");
    }

    #[test]
    fn test_static_shape_rank_dimension_and_slice() {
        let s0 = StaticShape::scalar();
        let s1 = StaticShape::new(vec![42]);
        let s2 = StaticShape::new(vec![4, 1]);

        assert_eq!(s0.rank(), 0);
        assert_eq!(s0.as_slice(), &[] as &[usize]);
        assert_eq!(s1.rank(), 1);
        assert_eq!(s1.dimension(0), 42);
        assert_eq!(s2.rank(), 2);
        assert_eq!(s2.dimension(1), 1);
        assert_eq!(s2.dimension(-2), 4);
        assert_eq!(s2.as_slice(), &[4, 1]);

        assert_eq!(s1[0usize], 42);
        assert_eq!(s2[0usize], 4);
        assert_eq!(s2[1usize], 1);
        assert_eq!(s2[-1isize], 1);
        assert_eq!(s2[-2isize], 4);
    }

    #[test]
    fn test_static_shape_row_major_strides() {
        assert_eq!(StaticShape::new(vec![2, 3, 4]).row_major_strides(), vec![12, 4, 1]);
        assert_eq!(StaticShape::new(vec![5]).row_major_strides(), vec![1]);
        assert_eq!(StaticShape::scalar().row_major_strides(), Vec::<usize>::new());
    }

    #[test]
    fn test_static_shape_display() {
        let s0 = StaticShape::scalar();
        let s1 = StaticShape::new(vec![42, 4, 2]);
        let s2 = StaticShape::new(vec![4, 1]);

        assert_eq!(format!("{s0}"), "[]");
        assert_eq!(format!("{s1}"), "[42, 4, 2]");
        assert_eq!(format!("{s2}"), "[4, 1]");
    }

    #[test]
    fn test_static_shape_to_shape() {
        let static_shape = StaticShape::new(vec![42, 4, 2]);
        let shape = Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]);

        assert_eq!(Shape::from(static_shape.clone()), shape);
        assert_eq!(Shape::from(&static_shape), shape);
    }

    #[test]
    fn test_static_shape_from_shape() {
        let static_shape = StaticShape::new(vec![42, 4, 2]);
        let shape = Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]);
        let dynamic_shape = Shape::new(vec![Size::Static(42), Size::Dynamic(None)]);
        let bounded_dynamic_shape = Shape::new(vec![Size::Static(42), Size::Dynamic(Some(8))]);

        assert_eq!(StaticShape::try_from(shape.clone()), Ok(static_shape.clone()));
        assert_eq!(StaticShape::try_from(&shape), Ok(static_shape));
        assert_eq!(
            StaticShape::try_from(dynamic_shape),
            Err(TypeError { message: "shape dimension 1 must be static, but got *".to_string() }),
        );
        assert_eq!(
            StaticShape::try_from(&bounded_dynamic_shape),
            Err(TypeError { message: "shape dimension 1 must be static, but got <8".to_string() }),
        );
    }

    #[test]
    fn test_array_type_static_shape() {
        let static_shape = Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]);
        let dynamic_shape = Shape::new(vec![Size::Static(42), Size::Dynamic(None)]);

        let scalar = ArrayType::scalar(Boolean);
        let static_array_type = ArrayType::new(F32, static_shape);
        let dynamic_array_type = ArrayType::new(F8E3M4, dynamic_shape);

        assert_eq!(scalar.static_shape(), Some(StaticShape::scalar()));
        assert_eq!(static_array_type.static_shape(), Some(StaticShape::new(vec![42, 4, 2])));
        assert_eq!(dynamic_array_type.static_shape(), None);
    }

    #[test]
    fn test_array_type_rank() {
        let s1 = Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]);
        let s2 = Shape::new(vec![Size::Static(42), Size::Dynamic(None)]);

        let t0 = ArrayType::scalar(Boolean);
        let t1 = ArrayType::new(F32, s1);
        let t2 = ArrayType::new(F8E3M4, s2);

        assert_eq!(t0.rank(), 0);
        assert_eq!(t1.rank(), 3);
        assert_eq!(t2.rank(), 2);
    }

    #[test]
    fn test_array_type_dimension() {
        let s0 = Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]);
        let s1 = Shape::new(vec![Size::Static(42), Size::Dynamic(None)]);

        let t0 = ArrayType::new(F32, s0);
        let t1 = ArrayType::new(F8E3M4, s1);

        assert_eq!(t0.dimension(0), Size::Static(42));
        assert_eq!(t0.dimension(2), Size::Static(2));
        assert_eq!(t0.dimension(-2), Size::Static(4));
        assert_eq!(t1.dimension(0), Size::Static(42));
        assert_eq!(t1.dimension(1), Size::Dynamic(None));
        assert_eq!(t1.dimension(-1), Size::Dynamic(None));
    }

    #[test]
    fn test_array_type_element_count() {
        let static_shape = Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]);
        let dynamic_shape = Shape::new(vec![Size::Static(42), Size::Dynamic(None)]);

        let scalar = ArrayType::scalar(Boolean);
        let static_array_type = ArrayType::new(F32, static_shape);
        let dynamic_array_type = ArrayType::new(F8E3M4, dynamic_shape);

        assert_eq!(scalar.element_count(), Ok(Some(1)));
        assert_eq!(static_array_type.element_count(), Ok(Some(336)));
        assert_eq!(dynamic_array_type.element_count(), Ok(None));
    }

    #[test]
    fn test_array_type_memory() {
        // Arrays reside in device memory by default, and `with_memory` re-places them.
        let t0 = ArrayType::new(F32, Shape::new(vec![2.into(), 3.into()]));
        assert_eq!(t0.memory(), Memory::Device);
        let t1 = t0.clone().with_memory(Memory::Host { pinned: true });
        assert_eq!(t1.memory(), Memory::Host { pinned: true });
        assert_eq!(t1.data_type(), t0.data_type());
        assert_eq!(t1.shape(), t0.shape());

        // Placement participates in type equality and is rendered only for non-default memories.
        assert_ne!(t0, t1);
        assert_eq!(t0.to_string(), "f32[2, 3]");
        assert_eq!(t1.to_string(), "f32[2, 3]@Host[Pinned]");
        assert_eq!(t0.clone().with_memory(Memory::Host { pinned: false }).to_string(), "f32[2, 3]@Host[Unpinned]");

        // Rank-changing helpers preserve the placement.
        let t2 = t1.with_inserted_dimension(1, 5.into()).unwrap();
        assert_eq!(t2.memory(), Memory::Host { pinned: true });
        let (t3, removed_dimension) = t2.without_dimension(1).unwrap();
        assert_eq!(removed_dimension, Size::Static(5));
        assert_eq!(t3, t1);

        // Replication preserves the placement.
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(0, 0), Device::new(1, 0)],
        )
        .unwrap();
        assert_eq!(t1.replicated(&mesh).unwrap().memory(), Memory::Host { pinned: true });
    }

    #[test]
    fn test_array_type_insert_and_remove_dimensions() {
        let t0 = ArrayType::new(F32, Shape::new(vec![2.into(), 3.into()]));
        let t1 = t0.with_inserted_dimension(1, 5.into()).unwrap();
        let t2 = ArrayType::new(F32, Shape::new(vec![2.into(), 5.into(), 3.into()]));

        assert_eq!(t1, t2);
        assert_eq!(t1.without_dimension(1).unwrap(), (t0, Size::Static(5)));

        let t3 = ArrayType::new(F32, Shape::new(vec![2.into(), 3.into()]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![12, 4])));
        let t4 = t3.with_inserted_dimension(1, 5.into()).unwrap();

        assert_eq!(t4.layout, None);
        assert_eq!(t4.shape, Shape::new(vec![2.into(), 5.into(), 3.into()]));

        let (t5, removed_dimension) = t4.without_dimension(1).unwrap();

        assert_eq!(removed_dimension, Size::Static(5));
        assert_eq!(t5.layout, None);
        assert_eq!(t5.shape, Shape::new(vec![2.into(), 3.into()]));

        let m0 = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let s0 = Sharding::with_manual_axes(
            m0.clone(),
            vec![ShardingDimension::sharded(["x"])],
            Vec::<&str>::new(),
            Vec::<&str>::new(),
            ["x"],
        )
        .unwrap();
        let t6 = ArrayType::new(F32, Shape::new(vec![8.into()])).with_sharding(s0).unwrap();
        let t7 = t6.with_inserted_dimension(0, 2.into()).unwrap();
        let s1 = t7.sharding().unwrap();

        assert_eq!(s1.dimensions(), &[ShardingDimension::replicated(), ShardingDimension::sharded(["x"])]);
        assert_eq!(s1.varying_manual_axes(), &["x".to_string()].into_iter().collect());

        let (t8, removed_dimension) = t7.without_dimension(0).unwrap();

        assert_eq!(removed_dimension, Size::Static(2));
        assert_eq!(t8, t6);

        let s2 = Sharding::new(m0, vec![ShardingDimension::sharded(["x"])]).unwrap();
        let t9 = ArrayType::new(F32, Shape::new(vec![8.into()])).with_sharding(s2).unwrap();

        let (t10, removed_dimension) = t9.without_dimension(0).unwrap();
        let s3 = t10.sharding().unwrap();

        assert_eq!(removed_dimension, Size::Static(8));
        assert_eq!(s3.dimensions(), &Vec::<ShardingDimension>::new());
        assert_eq!(s3.varying_manual_axes(), &["x".to_string()].into_iter().collect());

        let m1 = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        let t11 = ArrayType::new(F32, Shape::new(vec![8.into()]))
            .with_sharding(Sharding::new(m1, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();

        assert_eq!(
            t11.without_dimension(0),
            Err(TypeError {
                message: "cannot remove dimension 0 because it is sharded over the non-manual mesh axis 'x'"
                    .to_string(),
            })
        );
    }

    #[test]
    fn test_array_type_replicated() {
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(0, 0), Device::new(1, 0)],
        )
        .unwrap();
        let r#type = ArrayType::new(F32, Shape::new(vec![Size::Static(2), Size::Static(3)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![12, 4])));
        let replicated = r#type.replicated(&mesh).unwrap();

        assert_eq!(replicated.data_type(), F32);
        assert_eq!(replicated.shape(), r#type.shape());
        assert_eq!(replicated.layout(), r#type.layout());
        assert_eq!(replicated.sharding(), Some(&Sharding::replicated(mesh.logical_mesh().clone(), 2)));
    }

    #[test]
    fn test_array_type_display() {
        let s1 = Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]);
        let s2 = Shape::new(vec![Size::Static(4), Size::Static(1)]);
        let s3 = Shape::new(vec![Size::Static(4), Size::Dynamic(Some(1))]);
        let s4 = Shape::new(vec![Size::Dynamic(None), Size::Static(42), Size::Dynamic(None)]);
        let s5 = Shape::new(vec![Size::Static(42), Size::Dynamic(None)]);

        let t0 = ArrayType::scalar(Boolean);
        let t1 = ArrayType::new(F32, s1);
        let t2 = ArrayType::new(BF16, s2);
        let t3 = ArrayType::new(F16, s3);
        let t4 = ArrayType::new(C64, s4);
        let t5 = ArrayType::new(F8E4M3FN, s5);
        let t6 = ArrayType::new(F32, Shape::new(vec![Size::Static(4), Size::Static(2)]))
            .with_layout(Layout::Tiled(TiledLayout::new(vec![1, 0], vec![Tile::new(vec![TileDimension::Sized(2)])])));
        let t7 = ArrayType::new(F32, Shape::new(vec![Size::Static(4), Size::Static(2)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![8, 4])));
        let t8 = ArrayType::new(F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(
                Sharding::with_manual_axes(
                    LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(),
                    vec![ShardingDimension::sharded(["x"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["x"],
                )
                .unwrap(),
            )
            .unwrap();

        assert_eq!(format!("{t0}"), "bool[]");
        assert_eq!(format!("{t1}"), "f32[42, 4, 2]");
        assert_eq!(format!("{t2}"), "bf16[4, 1]");
        assert_eq!(format!("{t3}"), "f16[4, <1]");
        assert_eq!(format!("{t4}"), "c64[*, 42, *]");
        assert_eq!(format!("{t5}"), "f8e4m3fn[42, *]");
        assert_eq!(format!("{t6}"), "f32[4, 2][layout=tiled{1,0:T(2)}]");
        assert_eq!(format!("{t7}"), "f32[4, 2][layout=strided{8,4}]");
        assert_eq!(format!("{t8}"), "f32[8][sharding={mesh<['x'=4]>, [{'x'}], varying_manual={'x'}}]");
    }

    #[test]
    fn test_array_type_is_compatible_with() {
        // `Type::is_compatible_with` is the interoperability relation (i.e., the "broadcastability" relation),
        // and it is distinct from the refinement relation tested by `test_array_type_is_refined_by`.
        let vector = ArrayType::new(F32, Shape::new(vec![Size::Static(1), Size::Static(3)]));
        let matrix = ArrayType::new(F32, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        assert!(vector.is_compatible_with(&matrix));
        assert!(!matrix.is_compatible_with(&vector));
        assert!(matrix.is_compatible_with(&matrix));
    }

    #[test]
    fn test_array_type_is_refined_by() {
        let declared = ArrayType::new(F32, Shape::new(vec![Size::Dynamic(None), Size::Static(3)]));
        let actual = ArrayType::new(F32, Shape::new(vec![Size::Static(2), Size::Static(3)]));

        // Identical types and refining shapes are accepted; the relation is directional.
        assert!(declared.is_refined_by(&declared));
        assert!(actual.is_refined_by(&actual));
        assert!(declared.is_refined_by(&actual));
        assert!(!actual.is_refined_by(&declared));

        // Bounded dynamic declared dimensions enforce their exclusive bound on static actual sizes.
        let bounded = ArrayType::new(F32, Shape::new(vec![Size::Dynamic(Some(4)), Size::Static(3)]));
        assert!(bounded.is_refined_by(&ArrayType::new(F32, Shape::new(vec![Size::Static(3), Size::Static(3)]))));
        assert!(!bounded.is_refined_by(&ArrayType::new(F32, Shape::new(vec![Size::Static(4), Size::Static(3)]))));

        // Data types must match exactly; broadcastable shapes do not make types compatible.
        assert!(!declared.is_refined_by(&ArrayType::new(F64, Shape::new(vec![Size::Static(2), Size::Static(3)]))));
        assert!(!actual.is_refined_by(&ArrayType::new(F32, Shape::new(vec![Size::Static(3)]))));

        // Layout, sharding, and memory metadata must match exactly.
        let strided = actual.clone().with_layout(Layout::Strided(StridedLayout::new(vec![12, 4])));
        assert!(!declared.is_refined_by(&strided));
        assert!(strided.is_refined_by(&strided));
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded = actual
            .clone()
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap(),
            )
            .unwrap();
        assert!(!declared.is_refined_by(&sharded));
        assert!(sharded.is_refined_by(&sharded));
        let pinned = actual.clone().with_memory(Memory::Host { pinned: true });
        assert!(!declared.is_refined_by(&pinned));
        assert!(pinned.is_refined_by(&pinned));
    }

    #[test]
    fn test_array_type_with_mismatched_sharding_rank() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        assert_eq!(
            ArrayType::new(F32, Shape::new(vec![Size::Static(4), Size::Static(2)])).with_sharding(sharding),
            Err(ShardingError::ShardingRankMismatch { sharding_rank: 1, array_rank: 2 }),
        );
    }

}
