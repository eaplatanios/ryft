use std::fmt::Display;

use ryft_macros::Parameter;

use crate::broadcasting::Broadcastable;
use crate::parameters::Parameter;
use crate::sharding::{MeshAxisType, Sharding, ShardingDimension, ShardingError};
use crate::types::{DataType, Layout, Type, TypeError};

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
/// followed by their [`Shape`]s, optionally followed by their [`Layout`] and [`Sharding`], if present.
///
/// # Examples
///
/// ```rust
/// # use ryft_core::types::DataType;
/// # use ryft_core::types::{ArrayType, Shape, Size};
///
/// // Boolean scalar.
/// assert_eq!(
///   ArrayType::new(DataType::Boolean, Shape::scalar(), None, None).unwrap().to_string(),
///   "bool[]",
/// );
///
/// // 64-bit unsigned integer vector with 42 elements.
/// assert_eq!(
///   ArrayType::new(DataType::U64, Shape::new(vec![Size::Static(42)]), None, None).unwrap().to_string(),
///   "u64[42]",
/// );
///
/// // 32-bit floating-point number matrix with 42 rows and up to 10 columns.
/// assert_eq!(
///   ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(42), Size::Dynamic(Some(10))]), None, None)
///       .unwrap()
///       .to_string(),
///   "f32[42, <10]",
/// );
///
/// // 64-bit complex number matrix with an unknown number of rows and 42 columns.
/// assert_eq!(
///   ArrayType::new(DataType::C64, Shape::new(vec![Size::Dynamic(None), Size::Static(42)]), None, None)
///       .unwrap()
///       .to_string(),
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
}

impl ArrayType {
    /// Constructs a new [`ArrayType`], validating that any provided [`Sharding`] has the same rank as `shape`.
    #[inline]
    pub fn new(
        data_type: DataType,
        shape: Shape,
        layout: Option<Layout>,
        sharding: Option<Sharding>,
    ) -> Result<Self, ShardingError> {
        if let Some(sharding) = &sharding {
            let sharding_rank = sharding.rank();
            let array_rank = shape.rank();
            if sharding_rank != array_rank {
                return Err(ShardingError::ShardingRankMismatch { sharding_rank, array_rank });
            }
        }

        Ok(Self { data_type, shape, layout, sharding })
    }

    /// Constructs a new "scalar" [`ArrayType`] with the provided [`DataType`]. The resulting [`ArrayType::shape`]
    /// will be a scalar (i.e., have rank 0).
    #[inline]
    pub fn scalar(data_type: DataType) -> Self {
        Self { data_type, shape: Shape::scalar(), layout: None, sharding: None }
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

    /// Returns the rank (i.e., the number of dimensions) of this [`ArrayType`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::types::DataType;
    /// # use ryft_core::types::{ArrayType, Shape, Size};
    ///
    /// // Boolean scalar.
    /// assert_eq!(ArrayType::new(DataType::Boolean, Shape::scalar(), None, None).unwrap().rank(), 0);
    ///
    /// // 64-bit unsigned integer vector with 42 elements.
    /// assert_eq!(ArrayType::new(DataType::U64, Shape::new(vec![Size::Static(42)]), None, None).unwrap().rank(), 1);
    ///
    /// // 32-bit floating-point number matrix with 42 rows and up to 10 columns.
    /// assert_eq!(
    ///     ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(42), Size::Dynamic(Some(10))]), None, None)
    ///         .unwrap()
    ///         .rank(),
    ///     2,
    /// );
    ///
    /// // 64-bit complex number matrix with an unknown number of rows and 42 columns.
    /// assert_eq!(
    ///     ArrayType::new(DataType::C64, Shape::new(vec![Size::Dynamic(None), Size::Static(42)]), None, None)
    ///         .unwrap()
    ///         .rank(),
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
        let sharding = self
            .sharding
            .as_ref()
            .map(|sharding| {
                let mut sharding_dimensions = sharding.dimensions().to_vec();
                sharding_dimensions.insert(index, ShardingDimension::Replicated);
                Sharding::with_manual_axes(
                    sharding.mesh().clone(),
                    sharding_dimensions,
                    sharding.unreduced_axes().clone(),
                    sharding.reduced_manual_axes().clone(),
                    sharding.varying_manual_axes().clone(),
                )
                .map_err(|error| TypeError { message: error.to_string() })
            })
            .transpose()?;
        Ok(Self { data_type: self.data_type, shape: Shape::new(dimensions), layout: None, sharding })
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
        let sharding = self
            .sharding
            .as_ref()
            .map(|sharding| {
                let mut sharding_dimensions = sharding.dimensions().to_vec();
                let removed_sharding_dimension = sharding_dimensions.remove(axis);
                let mut varying_manual_axes = sharding.varying_manual_axes().clone();
                if let ShardingDimension::Sharded(axis_names) = removed_sharding_dimension {
                    for axis_name in axis_names {
                        if sharding.mesh().axis_type(&axis_name) != Some(MeshAxisType::Manual) {
                            return Err(TypeError {
                                message: format!(
                                    "cannot remove sharded dimension {axis} \
                                    because mesh axis '{axis_name}' is not manual"
                                ),
                            });
                        }
                        varying_manual_axes.insert(axis_name);
                    }
                }
                Sharding::with_manual_axes(
                    sharding.mesh().clone(),
                    sharding_dimensions,
                    sharding.unreduced_axes().clone(),
                    sharding.reduced_manual_axes().clone(),
                    varying_manual_axes,
                )
                .map_err(|error| TypeError { message: error.to_string() })
            })
            .transpose()?;
        Ok((Self { data_type: self.data_type, shape: Shape::new(dimensions), layout: None, sharding }, dimension))
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
        Ok(())
    }
}

impl Type for ArrayType {
    #[inline]
    fn is_compatible_with(&self, other: &Self) -> bool {
        // Note that this compatibility relationship is defined here as a "broadcastability" relationship.
        self.is_broadcastable_to(other)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension, ShardingError};
    use crate::types::DataType::{BF16, Boolean, C64, F8E3M4, F8E4M3FN, F16, F32};
    use crate::types::{
        ArrayType, Layout, Shape, Size, StaticShape, StridedLayout, Tile, TileDimension, TiledLayout, TypeError,
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
        assert_eq!(s0.as_slice(), &[]);
        assert_eq!(s1.rank(), 1);
        assert_eq!(s1.dimension(0), 42);
        assert_eq!(s2.rank(), 2);
        assert_eq!(s2.dimension(1), 1);
        assert_eq!(s2.dimension(-2), 4);
        assert_eq!(s2.as_slice(), &[4, 1]);
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
    fn test_array_type_rank() {
        let s1 = Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]);
        let s2 = Shape::new(vec![Size::Static(42), Size::Dynamic(None)]);

        let t0 = ArrayType::scalar(Boolean);
        let t1 = ArrayType::new(F32, s1, None, None).unwrap();
        let t2 = ArrayType::new(F8E3M4, s2, None, None).unwrap();

        assert_eq!(t0.rank(), 0);
        assert_eq!(t1.rank(), 3);
        assert_eq!(t2.rank(), 2);
    }

    #[test]
    fn test_array_type_dimension() {
        let s0 = Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]);
        let s1 = Shape::new(vec![Size::Static(42), Size::Dynamic(None)]);

        let t0 = ArrayType::new(F32, s0, None, None).unwrap();
        let t1 = ArrayType::new(F8E3M4, s1, None, None).unwrap();

        assert_eq!(t0.dimension(0), Size::Static(42));
        assert_eq!(t0.dimension(2), Size::Static(2));
        assert_eq!(t0.dimension(-2), Size::Static(4));
        assert_eq!(t1.dimension(0), Size::Static(42));
        assert_eq!(t1.dimension(1), Size::Dynamic(None));
        assert_eq!(t1.dimension(-1), Size::Dynamic(None));
    }

    #[test]
    fn test_array_type_insert_and_remove_dimensions() {
        let t0 = ArrayType::new(F32, Shape::new(vec![2.into(), 3.into()]), None, None).unwrap();
        let t1 = t0.with_inserted_dimension(1, 5.into()).unwrap();
        let t2 = ArrayType::new(F32, Shape::new(vec![2.into(), 5.into(), 3.into()]), None, None).unwrap();

        assert_eq!(t1, t2);
        assert_eq!(t1.without_dimension(1).unwrap(), (t0, Size::Static(5)));

        let t3 = ArrayType::new(
            F32,
            Shape::new(vec![2.into(), 3.into()]),
            Some(Layout::Strided(StridedLayout::new(vec![12, 4]))),
            None,
        )
        .unwrap();
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
        let t6 = ArrayType::new(F32, Shape::new(vec![8.into()]), None, Some(s0.clone())).unwrap();
        let t7 = t6.with_inserted_dimension(0, 2.into()).unwrap();
        let s1 = t7.sharding().unwrap();

        assert_eq!(s1.dimensions(), &[ShardingDimension::replicated(), ShardingDimension::sharded(["x"])]);
        assert_eq!(s1.varying_manual_axes(), &["x".to_string()].into_iter().collect());

        let (t8, removed_dimension) = t7.without_dimension(0).unwrap();

        assert_eq!(removed_dimension, Size::Static(2));
        assert_eq!(t8, t6);

        let s2 = Sharding::new(m0, vec![ShardingDimension::sharded(["x"])]).unwrap();
        let t9 = ArrayType::new(F32, Shape::new(vec![8.into()]), None, Some(s2)).unwrap();

        let (t10, removed_dimension) = t9.without_dimension(0).unwrap();
        let s3 = t10.sharding().unwrap();

        assert_eq!(removed_dimension, Size::Static(8));
        assert_eq!(s3.dimensions(), &Vec::<ShardingDimension>::new());
        assert_eq!(s3.varying_manual_axes(), &["x".to_string()].into_iter().collect());

        let m1 = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        let t11 = ArrayType::new(
            F32,
            Shape::new(vec![8.into()]),
            None,
            Some(Sharding::new(m1, vec![ShardingDimension::sharded(["x"])]).unwrap()),
        )
        .unwrap();

        assert_eq!(
            t11.without_dimension(0),
            Err(TypeError {
                message: "cannot remove sharded dimension 0 because mesh axis 'x' is not manual".to_string(),
            })
        );
    }

    #[test]
    fn test_array_type_display() {
        let s1 = Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]);
        let s2 = Shape::new(vec![Size::Static(4), Size::Static(1)]);
        let s3 = Shape::new(vec![Size::Static(4), Size::Dynamic(Some(1))]);
        let s4 = Shape::new(vec![Size::Dynamic(None), Size::Static(42), Size::Dynamic(None)]);
        let s5 = Shape::new(vec![Size::Static(42), Size::Dynamic(None)]);

        let t0 = ArrayType::scalar(Boolean);
        let t1 = ArrayType::new(F32, s1, None, None).unwrap();
        let t2 = ArrayType::new(BF16, s2, None, None).unwrap();
        let t3 = ArrayType::new(F16, s3, None, None).unwrap();
        let t4 = ArrayType::new(C64, s4, None, None).unwrap();
        let t5 = ArrayType::new(F8E4M3FN, s5, None, None).unwrap();
        let t6 = ArrayType::new(
            F32,
            Shape::new(vec![Size::Static(4), Size::Static(2)]),
            Some(Layout::Tiled(TiledLayout::new(vec![1, 0], vec![Tile::new(vec![TileDimension::Sized(2)])]))),
            None,
        )
        .unwrap();
        let t7 = ArrayType::new(
            F32,
            Shape::new(vec![Size::Static(4), Size::Static(2)]),
            Some(Layout::Strided(StridedLayout::new(vec![8, 4]))),
            None,
        )
        .unwrap();
        let t8 = ArrayType::new(
            F32,
            Shape::new(vec![Size::Static(8)]),
            None,
            Some(
                Sharding::with_manual_axes(
                    LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(),
                    vec![ShardingDimension::sharded(["x"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["x"],
                )
                .unwrap(),
            ),
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
    fn test_array_type_with_mismatched_sharding_rank() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        assert_eq!(
            ArrayType::new(F32, Shape::new(vec![Size::Static(4), Size::Static(2)]), None, Some(sharding)),
            Err(ShardingError::ShardingRankMismatch { sharding_rank: 1, array_rank: 2 }),
        );
    }
}
