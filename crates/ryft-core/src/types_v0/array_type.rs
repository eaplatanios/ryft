//! [`Type`]s for multi-dimensional arrays with broadcasting semantics.
//!
//! This module provides [`Type`]s for representing multi-dimensional arrays with various data types and dimensions.
//! It implements [NumPy-like broadcasting semantics](https://numpy.org/doc/stable/user/basics.broadcasting.html)
//! and [JAX-like data type promotion rules](https://docs.jax.dev/en/latest/type_promotion.html).

use std::fmt::Display;

use thiserror::Error;

use ryft_macros::Parameter;

use crate::parameters::Parameter;
use crate::types::data_type::DataTypeError as CoreDataTypeError;
use crate::types::{DataType, Type};

/// Represents the size of an array dimension. Array dimensions can be either statically known at compilation time
/// or dynamic, in which case their sizes will only be known at runtime. Dynamic dimensions may optionally have an upper
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

    /// Dynamic size that is not known until runtime and which has an optional upper bound. The upper bound,
    /// if present, represents an exclusive upper bound on the value that this size can have (i.e., the maximum
    /// possible value plus one). This can enable certain optimizations and static checks (though it is, of course,
    /// not as powerful as a static size).
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

    /// Returns an (exclusive) upper bound for the value of this [`Size`] if such a bound is known.
    /// For [`Size::Static`] sizes, this function will return the underlying value plus one as the upper bound.
    /// For [`Size::Dynamic`] sizes, this function will return the upper bound for that size if one exists, and
    /// `None` otherwise.
    #[inline]
    pub fn upper_bound(&self) -> Option<usize> {
        match &self {
            Self::Static(size) => Some(*size + 1),
            Self::Dynamic(upper_bound) => *upper_bound,
        }
    }
}

impl Display for Size {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            Self::Static(size) => write!(f, "{size}"),
            Self::Dynamic(Some(upper_bound)) => write!(f, "<{upper_bound}"),
            Self::Dynamic(None) => write!(f, "*"),
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
    /// # use ryft_core::types_v0::array_type::{Shape, Size};
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

    /// Constructs a new [`Shape`] that is the "smallest" shape that all of the provided shapes can be broadcast to
    /// using [NumPy-like broadcasting semantics](https://numpy.org/doc/stable/user/basics.broadcasting.html). Since
    /// this shape is the "smallest" such shape, we also know that its number of dimensions must match that of one of
    /// the provided [`Shape`]s).
    ///
    /// Note that this operation is *order-invariant* meaning that it will return the same [`Shape`] irrespective
    /// of the order in which the input [`Shape`]s are provided.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::types_v0::array_type::{Shape, Size};
    /// let w = Shape::scalar();
    /// let x = Shape::new(vec![Size::Static(42), Size::Static(42)]);
    /// let y = Shape::new(vec![Size::Dynamic(Some(10))]);
    /// let z = Shape::new(vec![Size::Static(42), Size::Dynamic(Some(10))]);
    ///
    /// assert_eq!(Shape::broadcast(&[&w]), Ok(w.clone()));
    /// assert_eq!(Shape::broadcast(&[&w, &y]), Ok(y.clone()));
    /// assert_eq!(Shape::broadcast(&[&w, &z]), Ok(z.clone()));
    /// assert_eq!(Shape::broadcast(&[&z, &y]), Ok(z.clone()));
    /// assert_eq!(Shape::broadcast(&[&w, &y, &z]), Ok(z.clone()));
    /// assert!(Shape::broadcast(&[&x, &w, &y, &z]).is_err());
    /// ```
    pub fn broadcast(shapes: &[&Self]) -> Result<Self, ShapeBroadcastingError> {
        if shapes.is_empty() {
            return Err(ShapeBroadcastingError::Empty);
        }

        shapes.iter().fold(Ok(Shape::scalar()), |lhs, rhs| match lhs {
            Ok(lhs) => {
                // Handle differing array ranks by (conceptually) padding the shorter shape with ones
                // on the left (i.e., as a prefix), up to the rank of the longer shape.
                let broadcast_rank = lhs.rank().max(rhs.rank());
                let lhs_offset = broadcast_rank - lhs.rank();
                let rhs_offset = broadcast_rank - rhs.rank();
                let mut broadcast_dimensions = Vec::with_capacity(broadcast_rank);
                for i in 0..broadcast_rank {
                    let lhs_size = if i < lhs_offset { Size::Static(1) } else { lhs.dimensions[i - lhs_offset] };
                    let rhs_size = if i < rhs_offset { Size::Static(1) } else { rhs.dimensions[i - rhs_offset] };
                    let broadcast_size = match (&lhs_size, &rhs_size) {
                        (_, Size::Static(1)) => lhs_size,
                        (Size::Static(1), _) => rhs_size,
                        (Size::Static(a), Size::Static(b)) if a == b => lhs_size,
                        (Size::Dynamic(a), Size::Dynamic(b)) if a == b => lhs_size,
                        _ => {
                            return Err(ShapeBroadcastingError::Incompatible { lhs: lhs.clone(), rhs: (*rhs).clone() });
                        }
                    };
                    broadcast_dimensions.push(broadcast_size);
                }

                Ok(Shape { dimensions: broadcast_dimensions })
            }
            Err(error) => Err(error),
        })
    }

    /// Broadcasts this [`Shape`] to the provided [`Shape`] using
    /// [NumPy-like semantics](https://numpy.org/doc/stable/user/basics.broadcasting.html).
    ///
    /// Note that this operation is *not necessarily symmetric* meaning that `x.broadcast_to(y)` is not necessarily
    /// going to be equal `y.broadcast_to(x)` for all `x` and `y`. In fact, it is only going to be equal if `x == y`,
    /// in which case we will also have `x.broadcast_to(y) == y.broadcast_to(x) == x == y`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::types_v0::array_type::{Shape, Size};
    /// let w = Shape::new(vec![Size::Static(42), Size::Static(42)]);
    /// let x = Shape::new(vec![]);
    /// let y = Shape::new(vec![Size::Dynamic(Some(10))]);
    /// let z = Shape::new(vec![Size::Static(42), Size::Dynamic(Some(10))]);
    ///
    /// assert_eq!(x.broadcast_to(&x), Ok(x.clone()));
    /// assert_eq!(x.broadcast_to(&y), Ok(y.clone()));
    /// assert_eq!(x.broadcast_to(&z), Ok(z.clone()));
    /// assert!(z.broadcast_to(&y).is_err());
    /// assert_eq!(x.broadcast_to(&y).unwrap().broadcast_to(&z), Ok(z.clone()));
    /// assert_eq!(x.broadcast_to(&w), Ok(w.clone()));
    /// assert!(w.broadcast_to(&x).is_err());
    /// ```
    pub fn broadcast_to(&self, other: &Self) -> Result<Self, ShapeBroadcastingError> {
        if self.rank() > other.rank() {
            return Err(ShapeBroadcastingError::Incompatible { lhs: self.clone(), rhs: other.clone() });
        }

        // Handle differing array ranks by (conceptually) padding the dimension sizes of this shape
        // with ones on the left (i.e., as a prefix), up to the rank of the other shape.
        let broadcast_rank = other.rank();
        let offset = broadcast_rank - self.rank();
        let mut broadcast_shape = Vec::with_capacity(broadcast_rank);
        for i in 0..broadcast_rank {
            let lhs_size = if i < offset { Size::Static(1) } else { self.dimensions[i - offset] };
            let rhs_size = other.dimensions[i];
            let broadcast_size = match (&lhs_size, &rhs_size) {
                (Size::Static(1), _) => rhs_size,
                (Size::Static(a), Size::Static(b)) if a == b => rhs_size,
                (Size::Dynamic(a), Size::Dynamic(b)) if a == b => rhs_size,
                _ => return Err(ShapeBroadcastingError::Incompatible { lhs: self.clone(), rhs: other.clone() }),
            };
            broadcast_shape.push(broadcast_size);
        }

        Ok(Self { dimensions: broadcast_shape })
    }

    /// Returns `true` if this [`Shape`] can be broadcast to the provided [`Shape`], and `false` otherwise.
    /// Refer to the documentation of [`Shape::broadcast_to`] for more information on [`Shape`] broadcasting.
    pub fn broadcastable_to(&self, other: &Self) -> bool {
        if self.rank() > other.rank() {
            return false;
        }

        let broadcast_rank = other.rank();
        let offset = broadcast_rank - self.rank();
        for i in 0..broadcast_rank {
            let lhs_size = if i < offset { Size::Static(1) } else { self.dimensions[i - offset] };
            let rhs_size = other.dimensions[i];
            match (&lhs_size, &rhs_size) {
                (Size::Static(1), _) => continue,
                (Size::Static(a), Size::Static(b)) if a == b => continue,
                (Size::Dynamic(a), Size::Dynamic(b)) if a == b => continue,
                _ => return false,
            };
        }
        true
    }
}

impl Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]", self.dimensions.iter().map(|dimension| dimension.to_string()).collect::<Vec<_>>().join(", "))
    }
}

/// Error returned when a [`Shape`] cannot be broadcast to another [`Shape`].
#[derive(Error, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ShapeBroadcastingError {
    /// Error returned when attempting to compute a broadcast [`Shape`] for an empty collection
    /// of [`Shape`]s (i.e., using [`Shape::broadcast`]).
    #[error("cannot construct a broadcast shape from an empty collection of shapes")]
    Empty,

    /// Error returned when a [`Shape`] broadcasting fails due to incompatible shapes.
    #[error("cannot promote shape `{lhs}` to shape `{rhs}`")]
    Incompatible { lhs: Shape, rhs: Shape },
}

/// Represents the type of a potentially multi-dimensional array.
///
/// Note that the [`Display`] implementation of [`ArrayType`] renders array types simply as their
/// [`DataType`]s followed by their [`Shape`]s.
///
/// # Examples
///
/// ```rust
/// # use ryft_core::types::DataType;
/// # use ryft_core::types_v0::array_type::{ArrayType, Shape, Size};
///
/// // Boolean scalar.
/// assert_eq!(
///   ArrayType::new(DataType::Boolean, Shape::scalar()).to_string(),
///   "bool[]",
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
#[derive(Clone, Debug, Eq, PartialEq, Hash, Parameter)]
pub struct ArrayType {
    /// [`DataType`] of the elements stored in the array.
    pub data_type: DataType,

    /// [`Shape`] of the array.
    pub shape: Shape,
}

impl ArrayType {
    /// Constructs a new [`ArrayType`] with the provided [`DataType`] and [`Shape`].
    #[inline]
    pub fn new(data_type: DataType, shape: Shape) -> Self {
        Self { data_type, shape }
    }

    /// Constructs a new "scalar" [`ArrayType`] with the provided [`DataType`]. The resulting [`ArrayType::shape`]
    /// will be a scalar (i.e., have rank 0).
    #[inline]
    pub fn scalar(data_type: DataType) -> Self {
        Self::new(data_type, Shape::scalar())
    }

    /// Returns the rank (i.e., the number of dimensions) of this [`ArrayType`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::types::DataType;
    /// # use ryft_core::types_v0::array_type::{ArrayType, Shape, Size};
    ///
    /// // Boolean scalar.
    /// assert_eq!(ArrayType::new(DataType::Boolean, Shape::scalar()).rank(), 0);
    ///
    /// // 64-bit unsigned integer vector with 42 elements.
    /// assert_eq!(ArrayType::new(DataType::U64, Shape::new(vec![Size::Static(42)])).rank(), 1);
    ///
    /// // 32-bit floating-point number matrix with 42 rows and up to 10 columns.
    /// assert_eq!(ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(42), Size::Dynamic(Some(10))])).rank(), 2);
    ///
    /// // 64-bit complex number matrix with an unknown number of rows and 42 columns.
    /// assert_eq!(ArrayType::new(DataType::C64, Shape::new(vec![Size::Dynamic(None), Size::Static(42)])).rank(), 2);
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

    /// Constructs a new [`ArrayType`] that is the narrowest array type that all of the provided array types can be
    /// broadcast to. That new array type has the following properties:
    ///
    ///   1. All of the provided array type [`DataType`]s can be promoted to its [`DataType`]. Refer to
    ///      [`DataType::promoted`] for more information on data type promotion.
    ///   2. All of the provided array type [`Shape`]s can be broadcast to its [`Shape`]. Refer to [`Shape::broadcast`]
    ///      for more information on shape broadcasting.
    ///
    /// Note that this operation is *order-invariant* meaning that it will return the same [`ArrayType`] irrespective
    /// of the order in which the input [`ArrayType`]s are provided.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::types::DataType;
    /// # use ryft_core::types_v0::array_type::{ArrayType, Shape, Size};
    /// let w = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(42), Size::Static(42)]));
    /// let x = ArrayType::new(DataType::Boolean, Shape::scalar());
    /// let y = ArrayType::new(DataType::U16, Shape::new(vec![Size::Dynamic(Some(10))]));
    /// let z = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(42), Size::Dynamic(Some(10))]));
    ///
    /// assert_eq!(ArrayType::broadcast(&[&x]), Ok(x.clone()));
    /// assert_eq!(ArrayType::broadcast(&[&x, &y]), Ok(y.clone()));
    /// assert_eq!(ArrayType::broadcast(&[&x, &z]), Ok(z.clone()));
    /// assert_eq!(ArrayType::broadcast(&[&z, &y]), Ok(z.clone()));
    /// assert_eq!(ArrayType::broadcast(&[&x, &y, &z]), Ok(z.clone()));
    /// assert!(ArrayType::broadcast(&[&w, &x, &y, &z]).is_err());
    /// ```
    pub fn broadcast(types: &[&Self]) -> Result<Self, ArrayTypeBroadcastingError> {
        if types.is_empty() {
            return Err(ArrayTypeBroadcastingError::Empty);
        }

        types.iter().fold(Ok(Self::scalar(DataType::Boolean)), |lhs, rhs| match lhs {
            Ok(lhs) => {
                let broadcast_data_type = DataType::promoted(&[lhs.data_type, rhs.data_type])?;
                let broadcast_shape = Shape::broadcast(&[&lhs.shape, &rhs.shape])?;
                Ok(Self { data_type: broadcast_data_type, shape: broadcast_shape })
            }
            Err(error) => Err(error),
        })
    }

    /// Broadcasts this [`ArrayType`] to the provided [`ArrayType`]. This consists of:
    ///
    ///   1. Promoting the underlying [`DataType`] using
    ///      [JAX-like promotion rules](https://docs.jax.dev/en/latest/type_promotion.html) to the provided
    ///      array type's [`DataType`].
    ///   2. Broadcasting the underlying dimension [`Size`]s using
    ///      [NumPy-like semantics](https://numpy.org/doc/stable/user/basics.broadcasting.html) to the provided
    ///      array type's dimension [`Size`]s.
    ///
    /// Note that this operation is *not necessarily symmetric* meaning that `x.broadcast_to(y)` is not necessarily
    /// going to be equal `y.broadcast_to(x)` for all `x` and `y`. In fact, it is only going to be equal if `x == y`,
    /// in which case we will also have `x.broadcast_to(y) == y.broadcast_to(x) == x == y`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::types::DataType;
    /// # use ryft_core::types_v0::array_type::{ArrayType, Shape, Size};
    /// let w = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(42), Size::Static(42)]));
    /// let x = ArrayType::new(DataType::Boolean, Shape::scalar());
    /// let y = ArrayType::new(DataType::U16, Shape::new(vec![Size::Dynamic(Some(10))]));
    /// let z = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(42), Size::Dynamic(Some(10))]));
    ///
    /// assert_eq!(x.broadcast_to(&x), Ok(x.clone()));
    /// assert_eq!(x.broadcast_to(&y), Ok(y.clone()));
    /// assert_eq!(x.broadcast_to(&z), Ok(z.clone()));
    /// assert!(z.broadcast_to(&y).is_err());
    /// assert_eq!(x.broadcast_to(&y).unwrap().broadcast_to(&z), Ok(z.clone()));
    /// assert_eq!(x.broadcast_to(&w), Ok(w.clone()));
    /// assert!(w.broadcast_to(&x).is_err());
    /// ```
    #[inline]
    pub fn broadcast_to(&self, other: &Self) -> Result<Self, ArrayTypeBroadcastingError> {
        let broadcast_data_type = self.data_type.promote_to(other.data_type)?;
        let broadcast_shape = self.shape.broadcast_to(&other.shape)?;
        Ok(ArrayType { data_type: broadcast_data_type, shape: broadcast_shape })
    }

    /// Returns `true` if this [`ArrayType`] can be broadcast to the provided [`ArrayType`], and `false` otherwise.
    /// Refer to the documentation of [`ArrayType::broadcast_to`] for more information on [`ArrayType`] broadcasting.
    #[inline]
    pub fn broadcastable_to(&self, other: &Self) -> bool {
        self.data_type.is_promotable_to(other.data_type) && self.shape.broadcastable_to(&other.shape)
    }
}

impl Type for ArrayType {
    /// Returns `true` if this [`ArrayType`] is a subtype of the provided [`ArrayType`], and `false` otherwise.
    ///
    /// An array type is considered a subtype of another if it is broadcastable to the other array type. For more
    /// information on broadcasting semantics for array types, refer to [`ArrayType::broadcastable_to`] and
    /// [`ArrayType::broadcast_to`].
    #[inline]
    fn is_subtype_of(&self, other: &Self) -> bool {
        self.broadcastable_to(&other)
    }
}

impl Display for ArrayType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.data_type, self.shape)
    }
}

/// Represents an [`Error`] related to [`ArrayType`] broadcasting. For more information on broadcasting,
/// refer to [`ArrayType::broadcast`], [`ArrayType::broadcast_to`], and [`ArrayType::broadcastable_to`].
#[derive(Error, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ArrayTypeBroadcastingError {
    /// Error returned when attempting to compute a broadcast [`ArrayType`] for an empty collection of
    /// [`ArrayType`]s (i.e., using [`ArrayType::broadcast`]).
    #[error("cannot construct a broadcast array type from an empty collection of array types")]
    Empty,

    /// Error returned when failing to broadcast two [`ArrayType`]s due to their [`DataType`]s not being
    /// compatible (i.e., the [`DataType`] of one cannot be promoted to the [`DataType`] of the other).
    #[error("{0}")]
    IncompatibleDataTypes(#[from] CoreDataTypeError),

    /// Error returned when failing to broadcast two [`ArrayType`]s due to their [`Shape`]s not being
    /// compatible (i.e., the [`Shape`] of one cannot be broadcast to the [`Shape`] of the other).
    #[error("{0}")]
    IncompatibleShapes(#[from] ShapeBroadcastingError),
}

#[cfg(test)]
mod tests {
    use super::*;

    use DataType::*;

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
    fn test_shape_broadcast() {
        let s0 = Shape::scalar();
        let s1 = Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]);
        let s2 = Shape::new(vec![Size::Static(4), Size::Static(1)]);
        let s3 = Shape::new(vec![Size::Static(4), Size::Dynamic(Some(1))]);
        let s4 = Shape::new(vec![Size::Dynamic(None), Size::Static(42), Size::Dynamic(None)]);
        let s5 = Shape::new(vec![Size::Static(42), Size::Dynamic(None)]);

        assert_eq!(Shape::broadcast(&[&s0]), Ok(s0.clone()));
        assert_eq!(Shape::broadcast(&[&s0, &s0]), Ok(s0.clone()));
        assert_eq!(Shape::broadcast(&[&s0, &s1]), Ok(s1.clone()));
        assert_eq!(Shape::broadcast(&[&s0, &s2]), Ok(s2.clone()));
        assert_eq!(Shape::broadcast(&[&s0, &s3]), Ok(s3.clone()));
        assert_eq!(Shape::broadcast(&[&s1, &s0]), Ok(s1.clone()));
        assert_eq!(Shape::broadcast(&[&s1, &s2]), Ok(s1.clone()));
        assert_eq!(Shape::broadcast(&[&s2, &s0]), Ok(s2.clone()));
        assert_eq!(Shape::broadcast(&[&s2, &s3]), Ok(s3.clone()));
        assert_eq!(Shape::broadcast(&[&s3, &s0]), Ok(s3.clone()));
        assert_eq!(Shape::broadcast(&[&s3, &s2]), Ok(s3.clone()));
        assert_eq!(Shape::broadcast(&[&s4, &s5]), Ok(s4.clone()));

        assert!(Shape::broadcast(&[]).is_err());
        assert!(Shape::broadcast(&[&s1, &s3]).is_err());
        assert!(Shape::broadcast(&[&s1, &s4]).is_err());
        assert!(Shape::broadcast(&[&s1, &s5]).is_err());
    }

    #[test]
    fn test_shape_broadcast_to() {
        let s0 = Shape::scalar();
        let s1 = Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]);
        let s2 = Shape::new(vec![Size::Static(4), Size::Static(1)]);
        let s3 = Shape::new(vec![Size::Static(4), Size::Dynamic(Some(1))]);
        let s4 = Shape::new(vec![Size::Dynamic(None), Size::Static(42), Size::Dynamic(None)]);
        let s5 = Shape::new(vec![Size::Static(42), Size::Dynamic(None)]);

        assert_eq!(s0.broadcast_to(&s0), Ok(s0.clone()));
        assert_eq!(s0.broadcast_to(&s1), Ok(s1.clone()));
        assert_eq!(s0.broadcast_to(&s2), Ok(s2.clone()));
        assert_eq!(s0.broadcast_to(&s3), Ok(s3.clone()));
        assert_eq!(s2.broadcast_to(&s1), Ok(s1.clone()));
        assert_eq!(s5.broadcast_to(&s4), Ok(s4.clone()));

        assert!(s1.broadcast_to(&s0).is_err());
        assert!(s1.broadcast_to(&s2).is_err());
        assert!(s1.broadcast_to(&s3).is_err());
        assert!(s1.broadcast_to(&s4).is_err());
        assert!(s1.broadcast_to(&s5).is_err());
        assert!(s2.broadcast_to(&s0).is_err());
        assert!(s3.broadcast_to(&s0).is_err());
        assert!(s4.broadcast_to(&s5).is_err());
    }

    #[test]
    fn test_shape_broadcastable_to() {
        let s0 = Shape::scalar();
        let s1 = Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]);
        let s2 = Shape::new(vec![Size::Static(4), Size::Static(1)]);
        let s3 = Shape::new(vec![Size::Static(4), Size::Dynamic(Some(1))]);
        let s4 = Shape::new(vec![Size::Dynamic(None), Size::Static(42), Size::Dynamic(None)]);
        let s5 = Shape::new(vec![Size::Static(42), Size::Dynamic(None)]);

        assert!(s0.broadcastable_to(&s0));
        assert!(s0.broadcastable_to(&s1));
        assert!(s0.broadcastable_to(&s2));
        assert!(s0.broadcastable_to(&s3));
        assert!(s2.broadcastable_to(&s1));
        assert!(s5.broadcastable_to(&s4));

        assert!(!s1.broadcastable_to(&s0));
        assert!(!s1.broadcastable_to(&s2));
        assert!(!s1.broadcastable_to(&s3));
        assert!(!s1.broadcastable_to(&s4));
        assert!(!s1.broadcastable_to(&s5));
        assert!(!s2.broadcastable_to(&s0));
        assert!(!s3.broadcastable_to(&s0));
        assert!(!s4.broadcastable_to(&s5));
    }

    #[test]
    fn test_shape_to_string() {
        let s0 = Shape::scalar();
        let s1 = Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]);
        let s2 = Shape::new(vec![Size::Static(4), Size::Static(1)]);
        let s3 = Shape::new(vec![Size::Static(4), Size::Dynamic(Some(1))]);
        let s4 = Shape::new(vec![Size::Dynamic(None), Size::Static(42), Size::Dynamic(None)]);
        let s5 = Shape::new(vec![Size::Static(42), Size::Dynamic(None)]);

        assert_eq!(s0.to_string(), "[]");
        assert_eq!(s1.to_string(), "[42, 4, 2]");
        assert_eq!(s2.to_string(), "[4, 1]");
        assert_eq!(s3.to_string(), "[4, <1]");
        assert_eq!(s4.to_string(), "[*, 42, *]");
        assert_eq!(s5.to_string(), "[42, *]");
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
    fn test_array_type_broadcast() {
        let s1 = Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]);
        let s2 = Shape::new(vec![Size::Static(4), Size::Static(1)]);
        let s3 = Shape::new(vec![Size::Static(4), Size::Dynamic(Some(1))]);
        let s4 = Shape::new(vec![Size::Dynamic(None), Size::Static(42), Size::Dynamic(None)]);
        let s5 = Shape::new(vec![Size::Static(42), Size::Dynamic(None)]);

        let t0 = ArrayType::scalar(Boolean);
        let t1 = ArrayType::new(F32, s1);
        let t2 = ArrayType::new(BF16, s2);
        let t3 = ArrayType::new(F16, s3.clone());
        let t4 = ArrayType::new(C64, s4);
        let t5 = ArrayType::new(BF16, s5);

        assert_eq!(ArrayType::broadcast(&[&t0]), Ok(t0.clone()));
        assert_eq!(ArrayType::broadcast(&[&t0, &t0]), Ok(t0.clone()));
        assert_eq!(ArrayType::broadcast(&[&t0, &t1]), Ok(t1.clone()));
        assert_eq!(ArrayType::broadcast(&[&t0, &t2]), Ok(t2.clone()));
        assert_eq!(ArrayType::broadcast(&[&t0, &t3]), Ok(t3.clone()));
        assert_eq!(ArrayType::broadcast(&[&t1, &t0]), Ok(t1.clone()));
        assert_eq!(ArrayType::broadcast(&[&t1, &t2]), Ok(t1.clone()));
        assert_eq!(ArrayType::broadcast(&[&t2, &t0]), Ok(t2.clone()));
        assert_eq!(ArrayType::broadcast(&[&t4, &t5]), Ok(t4.clone()));

        assert!(ArrayType::broadcast(&[]).is_err());
        assert_eq!(ArrayType::broadcast(&[&t2, &t3]), Ok(ArrayType::new(F32, s3.clone())));
        assert_eq!(ArrayType::broadcast(&[&t3, &t2]), Ok(ArrayType::new(F32, s3.clone())));
        assert!(ArrayType::broadcast(&[&t1, &t3]).is_err());
        assert!(ArrayType::broadcast(&[&t1, &t4]).is_err());
        assert!(ArrayType::broadcast(&[&t1, &t5]).is_err());
    }

    #[test]
    fn test_array_type_broadcast_to() {
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
        let t5 = ArrayType::new(BF16, s5);

        assert_eq!(t0.broadcast_to(&t0), Ok(t0.clone()));
        assert_eq!(t0.broadcast_to(&t1), Ok(t1.clone()));
        assert_eq!(t0.broadcast_to(&t2), Ok(t2.clone()));
        assert_eq!(t0.broadcast_to(&t3), Ok(t3.clone()));
        assert_eq!(t2.broadcast_to(&t1), Ok(t1.clone()));
        assert_eq!(t5.broadcast_to(&t4), Ok(t4.clone()));

        assert!(t1.broadcast_to(&t0).is_err());
        assert!(t1.broadcast_to(&t2).is_err());
        assert!(t1.broadcast_to(&t3).is_err());
        assert!(t1.broadcast_to(&t4).is_err());
        assert!(t1.broadcast_to(&t5).is_err());
        assert!(t2.broadcast_to(&t3).is_err());
        assert!(t3.broadcast_to(&t0).is_err());
        assert!(t3.broadcast_to(&t2).is_err());
        assert!(t4.broadcast_to(&t5).is_err());
    }

    #[test]
    fn test_array_type_broadcastable_to() {
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
        let t5 = ArrayType::new(BF16, s5);

        assert!(t0.broadcastable_to(&t0));
        assert!(t0.broadcastable_to(&t1));
        assert!(t0.broadcastable_to(&t2));
        assert!(t0.broadcastable_to(&t3));
        assert!(t2.broadcastable_to(&t1));
        assert!(t5.broadcastable_to(&t4));

        assert!(!t1.broadcastable_to(&t0));
        assert!(!t1.broadcastable_to(&t2));
        assert!(!t1.broadcastable_to(&t3));
        assert!(!t1.broadcastable_to(&t4));
        assert!(!t1.broadcastable_to(&t5));
        assert!(!t2.broadcastable_to(&t3));
        assert!(!t3.broadcastable_to(&t0));
        assert!(!t3.broadcastable_to(&t2));
        assert!(!t4.broadcastable_to(&t5));
    }

    #[test]
    fn test_array_type_to_string() {
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

        assert_eq!(t0.to_string(), "bool[]");
        assert_eq!(t1.to_string(), "f32[42, 4, 2]");
        assert_eq!(t2.to_string(), "bf16[4, 1]");
        assert_eq!(t3.to_string(), "f16[4, <1]");
        assert_eq!(t4.to_string(), "c64[*, 42, *]");
        assert_eq!(t5.to_string(), "f8e4m3fn[42, *]");
    }
}
