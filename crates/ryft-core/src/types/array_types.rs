use std::fmt::Display;

use ryft_macros::Parameter;

use crate::broadcasting::Broadcastable;
use crate::parameters::Parameter;
use crate::types::{DataType, Layout, Type};
use crate::xla::sharding::Sharding;

/// Represents the size of an array dimension. Array dimensions can be either statically known at compilation time or
/// dynamic, in which case their sizes will only be known at runtime. Dynamic dimensions may optionally have an upper
/// bound for their size that may be used for optimizations by the compiler. Note that by compilation here we do not
/// refer to the compilation of the Rust program but rather to the compilation of an array program within our Rust
/// library.
///
/// Note that the [`Display`] implementation of [`Size`] renders static sizes as just a number, dynamic sizes
/// with an upper bound as `<` followed by the upper bound, and dynamic sizes with no upper bound as `*`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Parameter)]
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
#[derive(Clone, Debug, Eq, PartialEq, Hash, Parameter)]
pub struct Shape {
    /// [`Size`]s of the array dimensions ordered from outermost to innermost.
    pub dimensions: Vec<Size>,
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
    pub fn dimension(&self, index: i32) -> Size {
        if index >= 0 {
            self.dimensions[index as usize]
        } else {
            self.dimensions[(self.dimensions.len() as i32 + index) as usize]
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
///   ArrayType::new(DataType::Boolean, Shape::scalar(), None, None).to_string(),
///   "bool[]",
/// );
///
/// // 64-bit unsigned integer vector with 42 elements.
/// assert_eq!(
///   ArrayType::new(DataType::U64, Shape::new(vec![Size::Static(42)]), None, None).to_string(),
///   "u64[42]",
/// );
///
/// // 32-bit floating-point number matrix with 42 rows and up to 10 columns.
/// assert_eq!(
///   ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(42), Size::Dynamic(Some(10))]), None, None).to_string(),
///   "f32[42, <10]",
/// );
///
/// // 64-bit complex number matrix with an unknown number of rows and 42 columns.
/// assert_eq!(
///   ArrayType::new(DataType::C64, Shape::new(vec![Size::Dynamic(None), Size::Static(42)]), None, None).to_string(),
///   "c64[*, 42]",
/// );
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Hash, Parameter)]
pub struct ArrayType {
    /// [`DataType`] of the elements stored in the array.
    pub data_type: DataType,

    /// [`Shape`] of the array.
    pub shape: Shape,

    /// Optional physical memory/storage [`Layout`] of the array.
    pub layout: Option<Layout>,

    /// Optional [`Sharding`] information about the array.
    pub sharding: Option<Sharding>,
}

impl ArrayType {
    /// Constructs a new [`ArrayType`].
    #[inline]
    pub fn new(data_type: DataType, shape: Shape, layout: Option<Layout>, sharding: Option<Sharding>) -> Self {
        Self { data_type, shape, layout, sharding }
    }

    /// Constructs a new "scalar" [`ArrayType`] with the provided [`DataType`]. The resulting [`ArrayType::shape`]
    /// will be a scalar (i.e., have rank 0).
    #[inline]
    pub fn scalar(data_type: DataType) -> Self {
        Self::new(data_type, Shape::scalar(), None, None)
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
    /// assert_eq!(ArrayType::new(DataType::Boolean, Shape::scalar(), None, None).rank(), 0);
    ///
    /// // 64-bit unsigned integer vector with 42 elements.
    /// assert_eq!(ArrayType::new(DataType::U64, Shape::new(vec![Size::Static(42)]), None, None).rank(), 1);
    ///
    /// // 32-bit floating-point number matrix with 42 rows and up to 10 columns.
    /// assert_eq!(
    ///     ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(42), Size::Dynamic(Some(10))]), None, None)
    ///         .rank(),
    ///     2,
    /// );
    ///
    /// // 64-bit complex number matrix with an unknown number of rows and 42 columns.
    /// assert_eq!(
    ///     ArrayType::new(DataType::C64, Shape::new(vec![Size::Dynamic(None), Size::Static(42)]), None, None)
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
    pub fn dimension(&self, index: i32) -> Size {
        self.shape.dimension(index)
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
        self.is_broadcastable_to(&other)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, ShardingDimension};
    use crate::types::DataType::{BF16, Boolean, C64, F8E3M4, F8E4M3FN, F16, F32};
    use crate::types::{ArrayType, Layout, Shape, Size, StridedLayout, Tile, TileDimension, TiledLayout};
    use crate::xla::sharding::Sharding;

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
    fn test_array_type_rank() {
        let s1 = Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]);
        let s2 = Shape::new(vec![Size::Static(42), Size::Dynamic(None)]);

        let t0 = ArrayType::scalar(Boolean);
        let t1 = ArrayType::new(F32, s1, None, None);
        let t2 = ArrayType::new(F8E3M4, s2, None, None);

        assert_eq!(t0.rank(), 0);
        assert_eq!(t1.rank(), 3);
        assert_eq!(t2.rank(), 2);
    }

    #[test]
    fn test_array_type_dimension() {
        let s0 = Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]);
        let s1 = Shape::new(vec![Size::Static(42), Size::Dynamic(None)]);

        let t0 = ArrayType::new(F32, s0, None, None);
        let t1 = ArrayType::new(F8E3M4, s1, None, None);

        assert_eq!(t0.dimension(0), Size::Static(42));
        assert_eq!(t0.dimension(2), Size::Static(2));
        assert_eq!(t0.dimension(-2), Size::Static(4));
        assert_eq!(t1.dimension(0), Size::Static(42));
        assert_eq!(t1.dimension(1), Size::Dynamic(None));
        assert_eq!(t1.dimension(-1), Size::Dynamic(None));
    }

    #[test]
    fn test_array_type_display() {
        let s1 = Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]);
        let s2 = Shape::new(vec![Size::Static(4), Size::Static(1)]);
        let s3 = Shape::new(vec![Size::Static(4), Size::Dynamic(Some(1))]);
        let s4 = Shape::new(vec![Size::Dynamic(None), Size::Static(42), Size::Dynamic(None)]);
        let s5 = Shape::new(vec![Size::Static(42), Size::Dynamic(None)]);

        let t0 = ArrayType::scalar(Boolean);
        let t1 = ArrayType::new(F32, s1, None, None);
        let t2 = ArrayType::new(BF16, s2, None, None);
        let t3 = ArrayType::new(F16, s3, None, None);
        let t4 = ArrayType::new(C64, s4, None, None);
        let t5 = ArrayType::new(F8E4M3FN, s5, None, None);
        let t6 = ArrayType::new(
            F32,
            Shape::new(vec![Size::Static(4), Size::Static(2)]),
            Some(Layout::Tiled(TiledLayout::new(vec![1, 0], vec![Tile::new(vec![TileDimension::Sized(2)])]))),
            None,
        );
        let t7 = ArrayType::new(
            F32,
            Shape::new(vec![Size::Static(4), Size::Static(2)]),
            Some(Layout::Strided(StridedLayout::new(vec![8, 4]))),
            None,
        );
        let t8 = ArrayType::new(
            F32,
            Shape::new(vec![Size::Static(8)]),
            None,
            Some(
                Sharding::new(
                    LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(),
                    vec![ShardingDimension::sharded(["x"])],
                    vec![],
                    vec![],
                    vec!["x".into()],
                )
                .unwrap(),
            ),
        );

        assert_eq!(format!("{t0}"), "bool[]");
        assert_eq!(format!("{t1}"), "f32[42, 4, 2]");
        assert_eq!(format!("{t2}"), "bf16[4, 1]");
        assert_eq!(format!("{t3}"), "f16[4, <1]");
        assert_eq!(format!("{t4}"), "c64[*, 42, *]");
        assert_eq!(format!("{t5}"), "f8e4m3fn[42, *]");
        assert_eq!(format!("{t6}"), "f32[4, 2][layout=tiled{1,0:T(2)}]");
        assert_eq!(format!("{t7}"), "f32[4, 2][layout=strided{8,4}]");
        assert_eq!(format!("{t8}"), "f32[8][sharding={mesh<['x'=4]>, [{'x'}], varying={'x'}}]");
    }
}
