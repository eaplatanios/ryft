//! Array dimension and shape type descriptors.

use std::fmt::Display;
use std::ops::Index;

use ryft_macros::Parameter;

use crate::axes::Axis;
use crate::parameters::Parameter;
use crate::programs::types::TypeError;

/// Represents the extent of one array axis. A dimension can be either statically known at compilation time or dynamic,
/// in which case its extent is only known at runtime. Dynamic dimensions may optionally have an upper bound that the
/// compiler can use for optimization. Compilation here refers to compiling an array program, not the Rust program
/// containing it.
///
/// The [`Display`] implementation renders static dimensions as a number, bounded dynamic dimensions as `<` followed by
/// the upper bound, and unbounded dynamic dimensions as `*`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum Dimension {
    /// Static extent that is known at compilation time.
    Static(usize),

    /// Dynamic extent that is not known until runtime and has an optional exclusive upper bound.
    Dynamic(Option<usize>),
}

impl Dimension {
    /// Returns the value of this [`Dimension`] if it is a [`Dimension::Static`] and `None` otherwise.
    #[inline]
    pub fn value(&self) -> Option<usize> {
        match &self {
            Self::Static(size) => Some(*size),
            Self::Dynamic(_) => None,
        }
    }

    /// Returns an exclusive upper bound for this [`Dimension`] when one is known. For [`Dimension::Static`], this is the
    /// static extent plus one. For [`Dimension::Dynamic`], this is its stored bound.
    #[inline]
    pub fn upper_bound(&self) -> Option<usize> {
        match &self {
            Self::Static(size) => Some(*size + 1),
            Self::Dynamic(upper_bound) => *upper_bound,
        }
    }

    /// Returns `true` if every concrete extent allowed by `other` is also allowed by this [`Dimension`]. The receiver is
    /// the more general dimension (e.g., a declared dimension), and the argument is the more precise one (e.g., the
    /// dimension carried by a runtime value's type), so the relation is directional:
    /// `declared.is_refined_by(&actual)`.
    ///
    /// The relation is defined as follows, recalling that the upper bound carried by [`Dimension::Dynamic`] is *exclusive*:
    ///
    /// - `Static(n)` is refined by `Static(m)` only when `n == m`.
    /// - `Static(_)` is never refined by `Dynamic(_)`.
    /// - `Dynamic(None)` is refined by every dimension.
    /// - `Dynamic(Some(bound))` is refined by `Static(m)` only when `m < bound`.
    /// - `Dynamic(Some(bound))` is refined by `Dynamic(Some(other))` only when `other <= bound`.
    /// - `Dynamic(Some(_))` is never refined by `Dynamic(None)`.
    #[inline]
    pub fn is_refined_by(&self, other: &Dimension) -> bool {
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

impl Display for Dimension {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            Self::Static(size) => write!(formatter, "{size}"),
            Self::Dynamic(Some(upper_bound)) => write!(formatter, "<{upper_bound}"),
            Self::Dynamic(None) => write!(formatter, "*"),
        }
    }
}

impl From<usize> for Dimension {
    fn from(value: usize) -> Self {
        Self::Static(value)
    }
}

/// Represents an array's ordered dimensions.
///
/// Note that the [`Display`] implementation of [`Shape`] renders shapes as the rendered dimension sizes
/// in a comma-separated list surrounded by square brackets.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct Shape {
    /// Dimensions ordered from outermost to innermost.
    dimensions: Vec<Dimension>,
}

impl Shape {
    /// Constructs a new [`Shape`] with the provided dimensions.
    #[inline]
    pub fn new(dimensions: Vec<Dimension>) -> Self {
        Self { dimensions }
    }

    /// Constructs a new scalar [`Shape`]. The resulting [`Shape::dimensions`] will be empty.
    #[inline]
    pub fn scalar() -> Self {
        Self::new(Vec::new())
    }

    /// Returns the dimensions ordered from outermost to innermost.
    #[inline]
    pub fn dimensions(&self) -> &[Dimension] {
        self.dimensions.as_slice()
    }

    /// Returns the rank (i.e., the number of dimensions) of this [`Shape`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::types::{Shape, Dimension};
    ///
    /// // Scalar.
    /// assert_eq!(Shape::scalar().rank(), 0);
    ///
    /// // Vector with 42 elements.
    /// assert_eq!(Shape::new(vec![Dimension::Static(42)]).rank(), 1);
    ///
    /// // Matrix with 42 rows and up to 10 columns.
    /// assert_eq!(Shape::new(vec![Dimension::Static(42), Dimension::Dynamic(Some(10))]).rank(), 2);
    ///
    /// // Matrix with an unknown number of rows and 42 columns.
    /// assert_eq!(Shape::new(vec![Dimension::Dynamic(None), Dimension::Static(42)]).rank(), 2);
    /// ```
    #[inline]
    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }

    /// Returns the `index`-th [`Dimension`] of this [`Shape`]. A negative `index` selects relative to the end, so `-1`
    /// returns the innermost dimension.
    #[inline]
    pub fn dimension<A: Into<Axis>>(&self, index: A) -> Dimension {
        self[index]
    }

    /// Returns the number of elements in arrays with this [`Shape`]. A statically zero dimension makes the result
    /// exactly zero even when another dimension is dynamic. Otherwise, a dynamic dimension produces `Ok(None)`.
    /// Returns a [`TypeError`] if the static element count does not fit in [`usize`].
    #[inline]
    pub fn element_count(&self) -> Result<Option<usize>, TypeError> {
        if self.dimensions.contains(&Dimension::Static(0)) {
            return Ok(Some(0));
        }
        let mut count = 1usize;
        for dimension in &self.dimensions {
            match dimension {
                Dimension::Static(extent) => {
                    count = count.checked_mul(*extent).ok_or_else(|| {
                        TypeError::invalid(format!("shape {self} element count does not fit in usize"))
                    })?;
                }
                Dimension::Dynamic(_) => return Ok(None),
            }
        }
        Ok(Some(count))
    }

    /// Returns `true` if every [`Shape`] admitted by `other` is also admitted by this [`Shape`]. The receiver is the
    /// more general shape (e.g., a declared shape), and the argument is the more precise one (e.g., the one carried by
    /// a runtime value's type), and so the relation is directional: `declared.is_refined_by(&actual)`. The two shapes
    /// must have equal rank and every receiver dimension must be refined by the corresponding dimension of `other` per
    /// [`Dimension::is_refined_by`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::types::{Shape, Dimension};
    ///
    /// let declared = Shape::new(vec![Dimension::Dynamic(None), Dimension::Static(3)]);
    ///
    /// // Dynamic declared dimensions are refined by any static size, while static ones require equality.
    /// assert!(declared.is_refined_by(&Shape::new(vec![Dimension::Static(2), Dimension::Static(3)])));
    /// assert!(declared.is_refined_by(&Shape::new(vec![Dimension::Static(5), Dimension::Static(3)])));
    /// assert!(!declared.is_refined_by(&Shape::new(vec![Dimension::Static(2), Dimension::Static(4)])));
    ///
    /// // The ranks must match exactly.
    /// assert!(!declared.is_refined_by(&Shape::new(vec![Dimension::Static(3)])));
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

impl<A: Into<Axis>> Index<A> for Shape {
    type Output = Dimension;

    #[inline]
    fn index(&self, axis: A) -> &Self::Output {
        let axis = axis.into();
        let position = axis
            .normalize(self.rank())
            .unwrap_or_else(|_| panic!("axis {axis} is out of bounds for shape {self}"));
        &self.dimensions[position]
    }
}

/// Represents an array shape whose dimensions are all [`Dimension::Static`], in contrast to [`Shape`], which also
/// supports dynamic dimensions.
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
    pub fn dimension<A: Into<Axis>>(&self, index: A) -> usize {
        self[index]
    }

    /// Returns the static dimension sizes as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[usize] {
        &self.dimensions
    }

    /// Returns the row-major (i.e., the last dimension corresponds to the fastest moving index) strides over element
    /// indices for arrays with this [`StaticShape`].
    pub fn row_major_strides(&self) -> Vec<usize> {
        let mut strides = vec![0usize; self.dimensions.len()];
        if self.dimensions.is_empty() {
            return strides;
        }
        let mut stride = 1usize;
        for index in (0..self.dimensions.len()).rev() {
            strides[index] = stride;
            stride *= self.dimensions[index];
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

impl<A: Into<Axis>> Index<A> for StaticShape {
    type Output = usize;

    #[inline]
    fn index(&self, axis: A) -> &Self::Output {
        let axis = axis.into();
        let position = axis
            .normalize(self.rank())
            .unwrap_or_else(|_| panic!("axis {axis} is out of bounds for static shape {self}"));
        &self.dimensions[position]
    }
}

impl From<StaticShape> for Shape {
    fn from(value: StaticShape) -> Self {
        Self::new(value.dimensions.into_iter().map(Dimension::Static).collect())
    }
}

impl From<&StaticShape> for Shape {
    fn from(value: &StaticShape) -> Self {
        Self::new(value.dimensions.iter().copied().map(Dimension::Static).collect())
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
        for (axis, dimension) in value.dimensions.iter().enumerate() {
            match dimension {
                Dimension::Static(extent) => dimensions.push(*extent),
                Dimension::Dynamic(_) => {
                    return Err(TypeError::invalid(format!(
                        "shape dimension {axis} must be static, but got {dimension}",
                    )));
                }
            }
        }
        Ok(Self::new(dimensions))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_dimension_value() {
        assert_eq!(Dimension::Static(1).value(), Some(1));
        assert_eq!(Dimension::Static(42).value(), Some(42));
        assert_eq!(Dimension::Dynamic(None).value(), None);
        assert_eq!(Dimension::Dynamic(Some(42)).value(), None);
    }

    #[test]
    fn test_dimension_upper_bound() {
        assert_eq!(Dimension::Static(1).upper_bound(), Some(2));
        assert_eq!(Dimension::Static(42).upper_bound(), Some(43));
        assert_eq!(Dimension::Dynamic(None).upper_bound(), None);
        assert_eq!(Dimension::Dynamic(Some(42)).upper_bound(), Some(42));
    }

    #[test]
    fn test_dimension_is_refined_by() {
        // Static declared sizes accept only equal static actual sizes.
        assert!(Dimension::Static(3).is_refined_by(&Dimension::Static(3)));
        assert!(!Dimension::Static(3).is_refined_by(&Dimension::Static(4)));
        assert!(!Dimension::Static(3).is_refined_by(&Dimension::Dynamic(None)));
        assert!(!Dimension::Static(3).is_refined_by(&Dimension::Dynamic(Some(3))));

        // Unbounded dynamic declared sizes accept every actual size.
        assert!(Dimension::Dynamic(None).is_refined_by(&Dimension::Static(0)));
        assert!(Dimension::Dynamic(None).is_refined_by(&Dimension::Static(42)));
        assert!(Dimension::Dynamic(None).is_refined_by(&Dimension::Dynamic(None)));
        assert!(Dimension::Dynamic(None).is_refined_by(&Dimension::Dynamic(Some(42))));

        // Bounded dynamic declared sizes accept static actual sizes strictly below the exclusive bound.
        assert!(Dimension::Dynamic(Some(4)).is_refined_by(&Dimension::Static(0)));
        assert!(Dimension::Dynamic(Some(4)).is_refined_by(&Dimension::Static(3)));
        assert!(!Dimension::Dynamic(Some(4)).is_refined_by(&Dimension::Static(4)));
        assert!(!Dimension::Dynamic(Some(4)).is_refined_by(&Dimension::Static(5)));

        // Bounded dynamic declared sizes accept bounded dynamic actual sizes with bounds at most as large,
        // and never accept unbounded dynamic actual sizes.
        assert!(Dimension::Dynamic(Some(4)).is_refined_by(&Dimension::Dynamic(Some(3))));
        assert!(Dimension::Dynamic(Some(4)).is_refined_by(&Dimension::Dynamic(Some(4))));
        assert!(!Dimension::Dynamic(Some(4)).is_refined_by(&Dimension::Dynamic(Some(5))));
        assert!(!Dimension::Dynamic(Some(4)).is_refined_by(&Dimension::Dynamic(None)));
    }

    #[test]
    fn test_dimension_to_string() {
        assert_eq!(Dimension::Static(1).to_string(), "1");
        assert_eq!(Dimension::Static(42).to_string(), "42");
        assert_eq!(Dimension::Dynamic(None).to_string(), "*");
        assert_eq!(Dimension::Dynamic(Some(42)).to_string(), "<42");
    }

    #[test]
    fn test_shape_rank() {
        let s0 = Shape::scalar();
        let s1 = Shape::new(vec![Dimension::Static(42)]);
        let s2 = Shape::new(vec![Dimension::Static(4), Dimension::Dynamic(None)]);

        assert_eq!(s0.rank(), 0);
        assert_eq!(s1.rank(), 1);
        assert_eq!(s2.rank(), 2);
    }

    #[test]
    fn test_shape_dimension() {
        let s0 = Shape::new(vec![Dimension::Static(42)]);
        let s1 = Shape::new(vec![Dimension::Static(4), Dimension::Dynamic(None)]);

        assert_eq!(s0.dimension(0), Dimension::Static(42));
        assert_eq!(s1.dimension(1), Dimension::Dynamic(None));
        assert_eq!(s1.dimension(-2), Dimension::Static(4));

        assert_eq!(s0[0usize], Dimension::Static(42));
        assert_eq!(s1[0usize], Dimension::Static(4));
        assert_eq!(s1[1usize], Dimension::Dynamic(None));
        assert_eq!(s1[-1isize], Dimension::Dynamic(None));
        assert_eq!(s1[-2isize], Dimension::Static(4));
    }

    #[test]
    fn test_shape_element_count() {
        assert_eq!(Shape::scalar().element_count(), Ok(Some(1)));
        assert_eq!(
            Shape::new(vec![Dimension::Static(42), Dimension::Static(4), Dimension::Static(2)]).element_count(),
            Ok(Some(336)),
        );
        assert_eq!(Shape::new(vec![Dimension::Static(42), Dimension::Static(0)]).element_count(), Ok(Some(0)));
        assert_eq!(Shape::new(vec![Dimension::Static(0), Dimension::Static(usize::MAX)]).element_count(), Ok(Some(0)));
        assert_eq!(
            Shape::new(vec![Dimension::Static(usize::MAX), Dimension::Static(0), Dimension::Static(usize::MAX)])
                .element_count(),
            Ok(Some(0)),
        );
        assert_eq!(
            Shape::new(vec![Dimension::Static(usize::MAX), Dimension::Static(2), Dimension::Static(0)]).element_count(),
            Ok(Some(0)),
        );
        assert_eq!(Shape::new(vec![Dimension::Static(usize::MAX), Dimension::Static(0)]).element_count(), Ok(Some(0)));
        assert_eq!(Shape::new(vec![Dimension::Static(42), Dimension::Dynamic(None)]).element_count(), Ok(None));
        assert_eq!(Shape::new(vec![Dimension::Static(42), Dimension::Dynamic(Some(8))]).element_count(), Ok(None));
        assert_eq!(Shape::new(vec![Dimension::Static(0), Dimension::Dynamic(None)]).element_count(), Ok(Some(0)));
        assert_eq!(
            Shape::new(vec![Dimension::Static(usize::MAX), Dimension::Static(2)]).element_count(),
            Err(TypeError::invalid(format!("shape [{}, 2] element count does not fit in usize", usize::MAX))),
        );
    }

    #[test]
    fn test_shape_is_refined_by() {
        let declared = Shape::new(vec![Dimension::Dynamic(None), Dimension::Static(3)]);

        // Pairwise size compatibility with matching ranks.
        assert!(declared.is_refined_by(&Shape::new(vec![Dimension::Static(2), Dimension::Static(3)])));
        assert!(declared.is_refined_by(&Shape::new(vec![Dimension::Static(5), Dimension::Static(3)])));
        assert!(declared.is_refined_by(&declared));
        assert!(!declared.is_refined_by(&Shape::new(vec![Dimension::Static(2), Dimension::Static(4)])));

        // Rank mismatches are always rejected.
        assert!(!declared.is_refined_by(&Shape::new(vec![Dimension::Static(3)])));
        assert!(!declared.is_refined_by(&Shape::new(vec![
            Dimension::Static(2),
            Dimension::Static(3),
            Dimension::Static(1)
        ])));
        assert!(!declared.is_refined_by(&Shape::scalar()));
        assert!(Shape::scalar().is_refined_by(&Shape::scalar()));
    }

    #[test]
    fn test_shape_display() {
        let s0 = Shape::scalar();
        let s1 = Shape::new(vec![Dimension::Static(42), Dimension::Static(4), Dimension::Static(2)]);
        let s2 = Shape::new(vec![Dimension::Static(4), Dimension::Static(1)]);
        let s3 = Shape::new(vec![Dimension::Static(4), Dimension::Dynamic(Some(1))]);
        let s4 = Shape::new(vec![Dimension::Dynamic(None), Dimension::Static(42), Dimension::Dynamic(None)]);
        let s5 = Shape::new(vec![Dimension::Static(42), Dimension::Dynamic(None)]);

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
        let shape = Shape::new(vec![Dimension::Static(42), Dimension::Static(4), Dimension::Static(2)]);

        assert_eq!(Shape::from(static_shape.clone()), shape);
        assert_eq!(Shape::from(&static_shape), shape);
    }

    #[test]
    fn test_static_shape_from_shape() {
        let static_shape = StaticShape::new(vec![42, 4, 2]);
        let shape = Shape::new(vec![Dimension::Static(42), Dimension::Static(4), Dimension::Static(2)]);
        let dynamic_shape = Shape::new(vec![Dimension::Static(42), Dimension::Dynamic(None)]);
        let bounded_dynamic_shape = Shape::new(vec![Dimension::Static(42), Dimension::Dynamic(Some(8))]);

        assert_eq!(StaticShape::try_from(shape.clone()), Ok(static_shape.clone()));
        assert_eq!(StaticShape::try_from(&shape), Ok(static_shape));
        assert_eq!(
            StaticShape::try_from(dynamic_shape),
            Err(TypeError::invalid("shape dimension 1 must be static, but got *".to_string())),
        );
        assert_eq!(
            StaticShape::try_from(&bounded_dynamic_shape),
            Err(TypeError::invalid("shape dimension 1 must be static, but got <8".to_string())),
        );
    }
}
