use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::ops::Index;
use std::sync::Arc;

use thiserror::Error;

use ryft_macros::Parameter;

use crate::axes::Axis;
use crate::parameters::Parameter;
use crate::programs::ProgramError;
use crate::programs::identities::{TypeIdentity, TypeIdentityPosition, TypeIdentityRenaming};
use crate::programs::types::{Type, TypeError, visit_type_signature_pairs};

/// Largest runtime dimension extent that every supported backend representation can carry. Host values use [`usize`],
/// while compiled dimension Single Static Assignment (SSA) values use signed 64-bit scalars. On narrower hosts every
/// [`usize`] fits. On 64-bit hosts this excludes values that cannot be lowered without changing their meaning.
pub const MAX_DIMENSION_EXTENT: usize = if usize::BITS < i64::BITS { usize::MAX } else { i64::MAX as usize };

/// Errors produced while constructing or validating [`Dimension`]s.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
pub enum DimensionError {
    #[error("invalid dimension bounds [{lower}, {upper})")]
    InvalidBounds {
        /// Inclusive lower bound.
        lower: usize,

        /// Exclusive upper bound.
        upper: usize,
    },

    #[error("dimension extent {value} exceeds backend maximum {maximum}")]
    ExtentExceedsBackendWidth {
        /// Concrete extent that exceeded the portable backend representation.
        value: usize,

        /// Largest extent supported by that representation.
        maximum: usize,
    },

    #[error("{message}")]
    ArithmeticOverflow {
        /// Complete user-facing diagnostic for overflowing dimension arithmetic.
        message: String,
    },

    #[error("input dimension `{variable}` = {value} is outside its declared bounds {bounds}")]
    BindingOutOfBounds {
        /// Diagnostic name of the dynamic dimension variable.
        variable: String,

        /// Concrete extent that violated the bounds.
        value: usize,

        /// Declared bounds of the variable.
        bounds: DimensionBounds,
    },

    #[error("{message}")]
    RequirementViolation {
        /// Complete user-facing requirement and observed-value diagnostic.
        message: String,
    },

    #[error("input dimension `{dimension}` expected {expected} but got {actual}")]
    InputDimensionMismatch {
        /// Diagnostic name of the shared dynamic dimension variable.
        dimension: String,

        /// Extent established by an earlier occurrence.
        expected: usize,

        /// Conflicting extent observed at this occurrence.
        actual: usize,
    },
}

impl From<DimensionError> for TypeError {
    #[inline]
    fn from(error: DimensionError) -> Self {
        Self::custom(error)
    }
}

impl From<DimensionError> for ProgramError {
    #[inline]
    fn from(error: DimensionError) -> Self {
        Self::custom(error)
    }
}

/// Inclusive lower and exclusive upper bounds for a dynamic dimension.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Parameter)]
pub struct DimensionBounds {
    /// Inclusive lower bound.
    lower: usize,

    /// Exclusive upper bound, or [`None`] when unbounded.
    upper: Option<usize>,
}

impl DimensionBounds {
    /// Creates a new [`DimensionBounds`] instance with an inclusive lower and optional exclusive upper bound.
    #[inline]
    pub fn new(lower: usize, upper: Option<usize>) -> Result<Self, DimensionError> {
        if let Some(upper) = upper
            && upper <= lower
        {
            return Err(DimensionError::InvalidBounds { lower, upper });
        }
        Ok(Self { lower, upper })
    }

    /// Creates a new [`DimensionBounds`] instance admitting non-negative extents below `upper`, when provided.
    #[inline]
    pub fn non_negative(upper: Option<usize>) -> Result<Self, DimensionError> {
        Self::new(0, upper)
    }

    /// Creates a new [`DimensionBounds`] instance admitting positive extents below `upper`, when provided.
    #[inline]
    pub fn positive(upper: Option<usize>) -> Result<Self, DimensionError> {
        Self::new(1, upper)
    }

    /// Creates a new [`DimensionBounds`] instance with the provided lower bound and no finite upper bound.
    #[inline]
    pub const fn at_least(lower: usize) -> Self {
        Self { lower, upper: None }
    }

    /// Creates a new [`DimensionBounds`] instance admitting every non-negative extent.
    #[inline]
    pub const fn unbounded() -> Self {
        Self::at_least(0)
    }

    /// Returns the inclusive lower bound of this [`DimensionBounds`] instance.
    #[inline]
    pub const fn lower(&self) -> usize {
        self.lower
    }

    /// Returns the exclusive upper bound of this [`DimensionBounds`] instance.
    #[inline]
    pub const fn upper(&self) -> Option<usize> {
        self.upper
    }

    /// Returns `true` if this [`DimensionBounds`] instance contains (i.e., admits) `value`.
    #[inline]
    pub fn contains(&self, value: usize) -> bool {
        value >= self.lower && self.upper.is_none_or(|upper| value < upper)
    }

    /// Returns `true` if every value contained in `other` is also contained in this [`DimensionBounds`] instance.
    #[inline]
    pub fn contains_bounds(&self, other: Self) -> bool {
        self.lower <= other.lower
            && match (self.upper, other.upper) {
                (None, _) => true,
                (Some(_), None) => false,
                (Some(left), Some(right)) => left >= right,
            }
    }
}

impl Display for DimensionBounds {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.upper {
            Some(upper) => write!(formatter, "[{}, {upper})", self.lower),
            None => write!(formatter, "[{}, ∞)", self.lower),
        }
    }
}

/// Shared allocation representing one unique symbolic [`DimensionVariable`] and owning its immutable metadata.
/// Cloning a [`DimensionVariable`] shares this allocation, so every clone continues to refer to the same symbolic
/// variable after either handle is moved. Each [`DimensionVariable::new`] call creates a fresh allocation and therefore
/// an independent variable, even when its name and bounds match an existing variable. Keeping the diagnostic name and
/// authoritative bounds in the shared payload also prevents handles to the same variable from observing different
/// metadata.
struct DimensionVariablePayload {
    /// Diagnostic name, excluded from semantic equality.
    name: String,

    /// [`DimensionBounds`] owned by the symbolic variable.
    bounds: DimensionBounds,
}

/// A symbolic variable representing a dynamic [`Dimension`]. Reusing a [`DimensionVariable`] in multiple array types
/// declares that those occurrences have the same runtime extent. Cloning preserves that relationship, whereas each call
/// to [`DimensionVariable::new`] creates an independent variable even when its name and bounds match another variable.
/// Names exist only for diagnostics, and the shared payload owns the authoritative immutable bounds observed by every
/// reference to the variable.
#[derive(Clone, Parameter)]
pub struct DimensionVariable {
    /// Shared symbolic variable and its authoritative metadata.
    payload: Arc<DimensionVariablePayload>,
}

impl DimensionVariable {
    /// Creates a new independent symbolic [`DimensionVariable`] with the provided name and [`DimensionBounds`].
    #[inline]
    pub fn new<N: Into<String>>(name: N, bounds: DimensionBounds) -> Self {
        Self { payload: Arc::new(DimensionVariablePayload { name: name.into(), bounds }) }
    }

    /// Returns the name of this [`DimensionVariable`], which is only meant be used for diagnostic purposes.
    #[inline]
    pub fn name(&self) -> &str {
        self.payload.name.as_str()
    }

    /// Returns the [`DimensionBounds`] of this [`DimensionVariable`].
    #[inline]
    pub fn bounds(&self) -> DimensionBounds {
        self.payload.bounds
    }
}

impl Display for DimensionVariable {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.payload.name)
    }
}

impl Debug for DimensionVariable {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DimensionVariable")
            .field("name", &self.payload.name)
            .field("bounds", &self.payload.bounds)
            .finish_non_exhaustive()
    }
}

impl PartialEq for DimensionVariable {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.payload, &other.payload)
    }
}

impl Eq for DimensionVariable {}

impl Hash for DimensionVariable {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.payload).hash(state);
    }
}

impl TypeIdentity for DimensionVariable {}

/// [`Type`] of a [`DimensionValue`](crate::DimensionValue). A value with this type defines `variable` as an ordinary
/// Single Static Assignment (SSA) value. Array axes may refer to that same [`DimensionVariable`], while arithmetic
/// relationships between dimension values remain explicit operation edges in the [`Program`](crate::Program) graph.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct DimensionType {
    /// [`DimensionVariable`] defined by values of this type.
    variable: DimensionVariable,
}

impl DimensionType {
    /// Creates a new [`DimensionType`] whose values define `variable`.
    #[inline]
    pub fn new(variable: DimensionVariable) -> Self {
        Self { variable }
    }

    /// Returns the [`DimensionVariable`] defined by values of this [`DimensionType`].
    #[inline]
    pub fn variable(&self) -> &DimensionVariable {
        &self.variable
    }

    /// Returns the authoritative [`DimensionBounds`] of this [`DimensionType`].
    #[inline]
    pub fn bounds(&self) -> DimensionBounds {
        self.variable.bounds()
    }

    /// Returns the most precise [`Dimension`] described by this [`DimensionType`]. Exact singleton bounds become a
    /// static dimension. All other bounds retain this type's [`DimensionVariable`] as a dynamic dimension.
    #[inline]
    pub fn to_dimension(&self) -> Dimension {
        let bounds = self.bounds();
        if bounds.lower().checked_add(1) == bounds.upper() {
            Dimension::Static(bounds.lower())
        } else {
            Dimension::Dynamic(self.variable.clone())
        }
    }

    /// Extends a complete-signature [`TypeIdentityRenaming`] with one declared/actual dimension-type pair. This is
    /// the per-pair fold step behind [`Type::derive_identity_renaming`] for dimension types: signature-level drivers
    /// (including the dimension arms of [`ArrayProgramType`](crate::types::ArrayProgramType)) call it once per member
    /// so that a repeated declared [`DimensionVariable`] renames consistently across the whole signature. The `actual`
    /// [`DimensionType`]'s bounds must be contained in the `declared` [`DimensionType`]'s bounds, and the declared
    /// [`DimensionVariable`] must not already be renamed to a different target by another member of the same signature.
    ///
    /// # Example
    ///
    /// Given the variables `n: [0, 8)`, `m: [0, 8)`, and `k: [0, 4)`, folding over a declared signature behaves as
    /// follows:
    ///
    ///   - `(dimension<n>, dimension<n>)` instantiated by `(dimension<m>, dimension<m>)`: Records the renaming
    ///     `n -> m` and accepts its consistent repetition.
    ///   - `(dimension<n>, dimension<n>)` instantiated by `(dimension<m>, dimension<k>)`: Fails because `n` would be
    ///     renamed to both `m` and `k`.
    ///   - `(dimension<k>)` instantiated by `(dimension<n>)`: Fails because the declared bounds `[0, 4)` do not
    ///     contain the actual bounds `[0, 8)`.
    pub(crate) fn extend_identity_renaming(
        declared: &Self,
        actual: &Self,
        renaming: &mut TypeIdentityRenaming<DimensionVariable>,
    ) -> Result<(), TypeError> {
        if !declared.bounds().contains_bounds(actual.bounds()) {
            return Err(TypeError::invalid(format!(
                "dimension type {actual} cannot instantiate declared type {declared}",
            )));
        }
        renaming.insert(declared.variable.clone(), actual.variable.clone())
    }
}

impl Display for DimensionType {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "dimension<{}: {}>", self.variable, self.bounds())
    }
}

impl Type for DimensionType {
    type Identity = DimensionVariable;
    type Refinements = ();

    #[inline]
    fn visit_identities(&self, visitor: &mut impl FnMut(TypeIdentityPosition, &Self::Identity)) {
        visitor(TypeIdentityPosition::Definition, &self.variable);
    }

    fn derive_identity_renaming(
        declared: &[Self],
        actual: &[Self],
    ) -> Result<TypeIdentityRenaming<Self::Identity>, TypeError> {
        let mut renaming = TypeIdentityRenaming::new();
        visit_type_signature_pairs(declared, actual, |declared, actual| {
            Self::extend_identity_renaming(declared, actual, &mut renaming)
        })?;
        Ok(renaming)
    }

    #[inline]
    fn rename_identities(&self, renaming: &TypeIdentityRenaming<Self::Identity>) -> Result<Self, TypeError> {
        let variable = renaming.rename(&self.variable);
        if !self.bounds().contains_bounds(variable.bounds()) {
            return Err(TypeError::invalid(format!(
                "cannot rename dimension variable {} with bounds {} to {variable} with bounds {}",
                self.variable,
                self.bounds(),
                variable.bounds(),
            )));
        }
        Ok(Self::new(variable))
    }

    #[inline]
    fn is_compatible_with(&self, other: &Self) -> bool {
        self == other
    }

    #[inline]
    fn is_refined_by(&self, other: &Self) -> bool {
        self.bounds().contains_bounds(other.bounds())
    }

    #[inline]
    fn is_scalar(&self) -> bool {
        false
    }

    #[inline]
    fn is_complex(&self) -> bool {
        false
    }
}

impl From<DimensionVariable> for DimensionType {
    #[inline]
    fn from(variable: DimensionVariable) -> Self {
        Self::new(variable)
    }
}

/// Represents the extent of one array axis as either a static value or one symbolic [`DimensionVariable`]. Reusing the
/// same variable in multiple dimensions declares that their runtime extents are equal. Arithmetic relationships between
/// dynamic extents are deliberately not represented in this type and instead belong in the ordinary program graph.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum Dimension {
    /// Extent known statically.
    Static(usize),

    /// Runtime extent represented by a symbolic [`DimensionVariable`].
    Dynamic(DimensionVariable),
}

impl Dimension {
    /// Returns the statically known extent of this [`Dimension`], or [`None`] if it is a [`Dimension::Dynamic`].
    #[inline]
    pub fn value(&self) -> Option<usize> {
        match self {
            Self::Static(value) => Some(*value),
            Self::Dynamic(_) => None,
        }
    }

    /// Returns the underlying [`DimensionVariable`] if this is a [`Dimension::Dynamic`], and [`None`] otherwise.
    #[inline]
    pub fn variable(&self) -> Option<&DimensionVariable> {
        match self {
            Self::Static(_) => None,
            Self::Dynamic(variable) => Some(variable),
        }
    }

    /// Returns the [`DimensionBounds`] of this [`Dimension`]. If this is a [`Dimension::Static`] dimension with size
    /// `n`, the resulting bounds will be `[n, n + 1)`. If `n + 1` cannot be represented, the upper bound will be set
    /// to `None`.
    #[inline]
    pub fn bounds(&self) -> DimensionBounds {
        match self {
            Self::Static(value) => DimensionBounds { lower: *value, upper: value.checked_add(1) },
            Self::Dynamic(variable) => variable.bounds(),
        }
    }

    /// Returns an exclusive upper bound for this [`Dimension`] if one is known.
    #[inline]
    pub fn upper_bound(&self) -> Option<usize> {
        self.bounds().upper()
    }

    /// Returns `true` if `other` is a valid refinement of this dimension. Static dimensions can only be refined by
    /// static dimensions with the same value. Dynamic dimensions can be refined by static extents within their bounds
    /// or by dynamic dimensions that are represented by the same symbolic [`DimensionVariable`]. That is because a
    /// different symbolic [`DimensionVariable`] represents an independent runtime extent.
    #[inline]
    pub fn is_refined_by(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Static(left), Self::Static(right)) => left == right,
            (Self::Static(_), Self::Dynamic(_)) => false,
            (Self::Dynamic(variable), Self::Static(value)) => variable.bounds().contains(*value),
            (Self::Dynamic(left), Self::Dynamic(right)) => left == right,
        }
    }

    /// Returns this dimension after applying `renaming` to its dynamic variable.
    #[inline]
    pub fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Self {
        match self {
            Self::Static(value) => Self::Static(*value),
            Self::Dynamic(variable) => Self::Dynamic(renaming.rename(variable)),
        }
    }
}

impl Display for Dimension {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Static(value) => Display::fmt(value, formatter),
            Self::Dynamic(variable) => Display::fmt(variable, formatter),
        }
    }
}

impl From<usize> for Dimension {
    #[inline]
    fn from(value: usize) -> Self {
        Self::Static(value)
    }
}

impl From<DimensionVariable> for Dimension {
    #[inline]
    fn from(variable: DimensionVariable) -> Self {
        Self::Dynamic(variable)
    }
}

impl From<&DimensionVariable> for Dimension {
    #[inline]
    fn from(variable: &DimensionVariable) -> Self {
        Self::Dynamic(variable.clone())
    }
}

/// Represents an array's ordered [`Dimension`]s. Note that the [`Display`] implementation of [`Shape`] renders shapes
/// as the rendered dimension sizes in a comma-separated list surrounded by square brackets.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct Shape {
    /// [`Dimension`]s ordered from outermost to innermost.
    dimensions: Vec<Dimension>,
}

impl Shape {
    /// Constructs a new [`Shape`] with the provided [`Dimension`]s.
    #[inline]
    pub fn new(dimensions: Vec<Dimension>) -> Self {
        Self { dimensions }
    }

    /// Constructs a new scalar [`Shape`]. The resulting [`Shape::dimensions`] will be empty.
    #[inline]
    pub fn scalar() -> Self {
        Self::new(Vec::new())
    }

    /// Returns the [`Dimension`]s of this [`Shape`] ordered from outermost to innermost.
    #[inline]
    pub fn dimensions(&self) -> &[Dimension] {
        self.dimensions.as_slice()
    }

    /// Returns the rank (i.e., the number of [`Dimension`]s) of this [`Shape`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::types::{Shape, Dimension};
    /// # use ryft_core::types::dimensions::{DimensionBounds, DimensionVariable};
    ///
    /// // Scalar.
    /// assert_eq!(Shape::scalar().rank(), 0);
    ///
    /// // Vector with 42 elements.
    /// assert_eq!(Shape::new(vec![Dimension::Static(42)]).rank(), 1);
    ///
    /// // Matrix with 42 rows and up to 10 columns.
    /// let columns = DimensionVariable::new("columns", DimensionBounds::non_negative(Some(10)).unwrap());
    /// assert_eq!(Shape::new(vec![Dimension::Static(42), columns.into()]).rank(), 2);
    ///
    /// // Matrix with an unknown number of rows and 42 columns.
    /// let rows = DimensionVariable::new("rows", DimensionBounds::unbounded());
    /// assert_eq!(Shape::new(vec![rows.into(), Dimension::Static(42)]).rank(), 2);
    /// ```
    #[inline]
    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }

    /// Returns the `index`-th [`Dimension`] of this [`Shape`]. A negative `index` selects relative to the end,
    /// so `-1` returns the innermost dimension.
    #[inline]
    pub fn dimension<A: Into<Axis>>(&self, index: A) -> Dimension {
        self[index].clone()
    }

    /// Returns the number of elements in arrays with this [`Shape`]. A statically zero [`Dimension`] makes the result
    /// exactly zero even when another [`Dimension`] is dynamic. Otherwise, a dynamic dimension produces `Ok(None)`.
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
    /// must have equal rank and every receiver dimension must be refined by the corresponding dimension of `other`
    /// per [`Dimension::is_refined_by`]. Repeated occurrences of a dynamic [`DimensionVariable`] must also refine
    /// to equal dimensions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::types::{Shape, Dimension};
    /// # use ryft_core::types::dimensions::{DimensionBounds, DimensionVariable};
    ///
    /// let rows = DimensionVariable::new("rows", DimensionBounds::unbounded());
    /// let declared = Shape::new(vec![rows.into(), Dimension::Static(3)]);
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
            && self.dimensions.iter().zip(&other.dimensions).enumerate().all(|(axis, (declared, actual))| {
                declared.is_refined_by(actual)
                    && declared.variable().is_none_or(|variable| {
                        self.dimensions[..axis].iter().zip(&other.dimensions[..axis]).all(
                            |(previous_declared, previous_actual)| {
                                previous_declared.variable() != Some(variable) || previous_actual == actual
                            },
                        )
                    })
            })
    }

    /// Returns this shape after simultaneously renaming every dynamic dimension variable.
    #[inline]
    pub fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Self {
        Self::new(self.dimensions.iter().map(|dimension| dimension.rename_type_identities(renaming)).collect())
    }
}

impl Display for Shape {
    #[inline]
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

/// Represents an array shape whose dimensions are all [`Dimension::Static`], in contrast to [`Shape`],
/// which also supports dynamic dimensions.
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
    use std::collections::HashSet;

    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_dimension_bounds() {
        assert_eq!(DimensionBounds::new(0, Some(0)), Err(DimensionError::InvalidBounds { lower: 0, upper: 0 }));
        assert_eq!(DimensionBounds::new(3, Some(3)), Err(DimensionError::InvalidBounds { lower: 3, upper: 3 }));
        assert_eq!(DimensionBounds::new(4, Some(3)), Err(DimensionError::InvalidBounds { lower: 4, upper: 3 }));

        let nonnegative = DimensionBounds::non_negative(Some(5)).unwrap();
        assert_eq!(nonnegative.lower(), 0);
        assert_eq!(nonnegative.upper(), Some(5));
        assert!(nonnegative.contains(0));
        assert!(nonnegative.contains(4));
        assert!(!nonnegative.contains(5));
        assert_eq!(nonnegative.to_string(), "[0, 5)");

        let positive = DimensionBounds::positive(Some(5)).unwrap();
        assert!(!positive.contains(0));
        assert!(positive.contains(1));
        assert!(nonnegative.contains_bounds(positive));
        assert!(!positive.contains_bounds(nonnegative));

        let unbounded = DimensionBounds::unbounded();
        assert_eq!(unbounded, DimensionBounds::at_least(0));
        assert!(unbounded.contains(usize::MAX));
        assert!(unbounded.contains_bounds(nonnegative));
        assert!(!nonnegative.contains_bounds(unbounded));
        assert_eq!(unbounded.to_string(), "[0, ∞)");

        let error = DimensionError::InvalidBounds { lower: 7, upper: 7 };
        assert_eq!(error.to_string(), "invalid dimension bounds [7, 7)");
        let type_error = TypeError::from(error.clone());
        assert_eq!(type_error.downcast_custom::<DimensionError>(), Some(&error));
    }

    #[test]
    fn test_dimension_variable() {
        let bounds = DimensionBounds::positive(Some(65)).unwrap();
        let batch = DimensionVariable::new("batch", bounds);
        let batch_clone = batch.clone();
        let same_declaration = DimensionVariable::new("batch", bounds);

        assert_eq!(batch, batch_clone);
        assert_ne!(batch, same_declaration);
        assert_eq!(batch.bounds(), bounds);
        assert_eq!(batch.to_string(), "batch");
        assert_eq!(
            format!("{batch:?}"),
            "DimensionVariable { name: \"batch\", bounds: DimensionBounds { lower: 1, upper: Some(65) }, .. }",
        );

        let mut variables = HashSet::new();
        variables.insert(batch);
        assert!(variables.contains(&batch_clone));
        assert!(!variables.contains(&same_declaration));
    }

    #[test]
    fn test_dimension_type() {
        let declared_variable = DimensionVariable::new("declared", DimensionBounds::new(1, Some(65)).unwrap());
        let actual_variable = DimensionVariable::new("actual", DimensionBounds::new(1, Some(33)).unwrap());
        let declared = DimensionType::new(declared_variable.clone());
        let actual = DimensionType::new(actual_variable.clone());

        assert_eq!(declared.variable(), &declared_variable);
        assert_eq!(declared.bounds(), DimensionBounds::new(1, Some(65)).unwrap());
        assert_eq!(declared.to_string(), "dimension<declared: [1, 65)>");
        assert!(!declared.is_scalar());
        assert!(!declared.is_complex());
        assert!(declared.is_refined_by(&actual));
        assert_eq!(declared.to_dimension(), Dimension::Dynamic(declared_variable.clone()));
        let exact_variable = DimensionVariable::new("7", DimensionBounds::new(7, Some(8)).unwrap());
        assert_eq!(DimensionType::new(exact_variable).to_dimension(), Dimension::Static(7));

        let mut identities = Vec::new();
        declared.visit_identities(&mut |position, variable| identities.push((position, variable.clone())));
        assert_eq!(identities, vec![(TypeIdentityPosition::Definition, declared_variable.clone())]);

        let renaming =
            DimensionType::derive_identity_renaming(std::slice::from_ref(&declared), std::slice::from_ref(&actual))
                .unwrap();
        assert_eq!(renaming.replacements(), &[(declared_variable.clone(), actual_variable.clone())]);
        assert_eq!(declared.rename_identities(&renaming), Ok(actual.clone()));

        let wider = DimensionType::new(DimensionVariable::new("wider", DimensionBounds::new(0, Some(129)).unwrap()));
        assert!(!declared.is_refined_by(&wider));
        assert_eq!(
            DimensionType::derive_identity_renaming(&[declared.clone()], &[wider]),
            Err(TypeError::invalid(
                "dimension type dimension<wider: [0, 129)> cannot instantiate declared type \
                 dimension<declared: [1, 65)>",
            )),
        );

        // A repeated declared variable must rename consistently across the complete signature.
        let other = DimensionType::new(DimensionVariable::new("other", DimensionBounds::new(1, Some(65)).unwrap()));
        assert_eq!(
            DimensionType::derive_identity_renaming(&[declared.clone(), declared], &[actual.clone(), other.clone()],),
            Err(TypeError::invalid(format!(
                "identity {} is renamed to both {} and {}",
                "declared",
                actual.variable(),
                other.variable(),
            ))),
        );
    }

    #[test]
    fn test_dimension_value() {
        let unbounded = DimensionVariable::new("unbounded", DimensionBounds::unbounded());
        let bounded = DimensionVariable::new("bounded", DimensionBounds::non_negative(Some(42)).unwrap());
        assert_eq!(Dimension::Static(1).value(), Some(1));
        assert_eq!(Dimension::Static(42).value(), Some(42));
        assert_eq!(Dimension::Dynamic(unbounded).value(), None);
        assert_eq!(Dimension::Dynamic(bounded).value(), None);
    }

    #[test]
    fn test_dimension_upper_bound() {
        let unbounded = DimensionVariable::new("unbounded", DimensionBounds::unbounded());
        let bounded = DimensionVariable::new("bounded", DimensionBounds::non_negative(Some(42)).unwrap());
        assert_eq!(Dimension::Static(1).upper_bound(), Some(2));
        assert_eq!(Dimension::Static(42).upper_bound(), Some(43));
        assert_eq!(Dimension::Dynamic(unbounded).upper_bound(), None);
        assert_eq!(Dimension::Dynamic(bounded).upper_bound(), Some(42));
    }

    #[test]
    fn test_dimension_is_refined_by() {
        let unbounded = DimensionVariable::new("unbounded", DimensionBounds::unbounded());
        let bounded = DimensionVariable::new("bounded", DimensionBounds::non_negative(Some(4)).unwrap());
        let other_unbounded = DimensionVariable::new("unbounded", DimensionBounds::unbounded());
        let other_bounded = DimensionVariable::new("bounded", DimensionBounds::non_negative(Some(4)).unwrap());

        // Static declared sizes accept only equal static actual sizes.
        assert!(Dimension::Static(3).is_refined_by(&Dimension::Static(3)));
        assert!(!Dimension::Static(3).is_refined_by(&Dimension::Static(4)));
        assert!(!Dimension::Static(3).is_refined_by(&Dimension::Dynamic(unbounded.clone())));
        assert!(!Dimension::Static(3).is_refined_by(&Dimension::Dynamic(bounded.clone())));

        // Dynamic declarations accept static extents admitted by their bounds.
        assert!(Dimension::Dynamic(unbounded.clone()).is_refined_by(&Dimension::Static(0)));
        assert!(Dimension::Dynamic(unbounded.clone()).is_refined_by(&Dimension::Static(42)));
        assert!(Dimension::Dynamic(bounded.clone()).is_refined_by(&Dimension::Static(0)));
        assert!(Dimension::Dynamic(bounded.clone()).is_refined_by(&Dimension::Static(3)));
        assert!(!Dimension::Dynamic(bounded.clone()).is_refined_by(&Dimension::Static(4)));
        assert!(!Dimension::Dynamic(bounded.clone()).is_refined_by(&Dimension::Static(5)));

        // A dynamic occurrence refines only the same symbolic variable, not an independent declaration with matching
        // name and bounds.
        assert!(Dimension::Dynamic(unbounded.clone()).is_refined_by(&Dimension::Dynamic(unbounded.clone())));
        assert!(Dimension::Dynamic(bounded.clone()).is_refined_by(&Dimension::Dynamic(bounded.clone())));
        assert!(!Dimension::Dynamic(unbounded).is_refined_by(&Dimension::Dynamic(other_unbounded)));
        assert!(!Dimension::Dynamic(bounded).is_refined_by(&Dimension::Dynamic(other_bounded)));
    }

    #[test]
    fn test_dimension_to_string() {
        let rows = DimensionVariable::new("rows", DimensionBounds::unbounded());
        let columns = DimensionVariable::new("columns", DimensionBounds::non_negative(Some(42)).unwrap());
        assert_eq!(Dimension::Static(1).to_string(), "1");
        assert_eq!(Dimension::Static(42).to_string(), "42");
        assert_eq!(Dimension::Dynamic(rows).to_string(), "rows");
        assert_eq!(Dimension::Dynamic(columns).to_string(), "columns");
    }

    #[test]
    fn test_shape_rank() {
        let s0 = Shape::scalar();
        let s1 = Shape::new(vec![Dimension::Static(42)]);
        let s2 = Shape::new(vec![
            Dimension::Static(4),
            Dimension::Dynamic(crate::types::dimensions::DimensionVariable::new(
                "dynamic",
                crate::types::dimensions::DimensionBounds::unbounded(),
            )),
        ]);

        assert_eq!(s0.rank(), 0);
        assert_eq!(s1.rank(), 1);
        assert_eq!(s2.rank(), 2);
    }

    #[test]
    fn test_shape_dimension() {
        let columns = DimensionVariable::new("columns", DimensionBounds::unbounded());
        let s0 = Shape::new(vec![Dimension::Static(42)]);
        let s1 = Shape::new(vec![Dimension::Static(4), Dimension::Dynamic(columns.clone())]);

        assert_eq!(s0.dimension(0), Dimension::Static(42));
        assert_eq!(s1.dimension(1), Dimension::Dynamic(columns.clone()));
        assert_eq!(s1.dimension(-2), Dimension::Static(4));

        assert_eq!(s0[0usize], Dimension::Static(42));
        assert_eq!(s1[0usize], Dimension::Static(4));
        assert_eq!(s1[1usize], Dimension::Dynamic(columns.clone()));
        assert_eq!(s1[-1isize], Dimension::Dynamic(columns));
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
        assert_eq!(
            Shape::new(vec![
                Dimension::Static(42),
                Dimension::Dynamic(crate::types::dimensions::DimensionVariable::new(
                    "dynamic",
                    crate::types::dimensions::DimensionBounds::unbounded()
                ))
            ])
            .element_count(),
            Ok(None)
        );
        assert_eq!(
            Shape::new(vec![
                Dimension::Static(42),
                Dimension::Dynamic(crate::types::dimensions::DimensionVariable::new(
                    "dynamic",
                    crate::types::dimensions::DimensionBounds::non_negative(Some(8)).unwrap()
                ))
            ])
            .element_count(),
            Ok(None)
        );
        assert_eq!(
            Shape::new(vec![
                Dimension::Static(0),
                Dimension::Dynamic(crate::types::dimensions::DimensionVariable::new(
                    "dynamic",
                    crate::types::dimensions::DimensionBounds::unbounded()
                ))
            ])
            .element_count(),
            Ok(Some(0))
        );
        assert_eq!(
            Shape::new(vec![Dimension::Static(usize::MAX), Dimension::Static(2)]).element_count(),
            Err(TypeError::invalid(format!("shape [{}, 2] element count does not fit in usize", usize::MAX))),
        );
    }

    #[test]
    fn test_shape_is_refined_by() {
        let dynamic = DimensionVariable::new("dynamic", DimensionBounds::unbounded());
        let declared = Shape::new(vec![Dimension::Dynamic(dynamic.clone()), Dimension::Static(3)]);

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

        // Repeated occurrences of one variable declare an equality relationship across axes.
        let square = Shape::new(vec![Dimension::Dynamic(dynamic.clone()), Dimension::Dynamic(dynamic)]);
        assert!(square.is_refined_by(&Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])));
        assert!(!square.is_refined_by(&Shape::new(vec![Dimension::Static(2), Dimension::Static(3)])));
    }

    #[test]
    fn test_shape_display() {
        let s0 = Shape::scalar();
        let s1 = Shape::new(vec![Dimension::Static(42), Dimension::Static(4), Dimension::Static(2)]);
        let s2 = Shape::new(vec![Dimension::Static(4), Dimension::Static(1)]);
        let s3 = Shape::new(vec![
            Dimension::Static(4),
            Dimension::Dynamic(DimensionVariable::new("depth", DimensionBounds::non_negative(Some(1)).unwrap())),
        ]);
        let s4 = Shape::new(vec![
            Dimension::Dynamic(DimensionVariable::new("rows", DimensionBounds::unbounded())),
            Dimension::Static(42),
            Dimension::Dynamic(DimensionVariable::new("columns", DimensionBounds::unbounded())),
        ]);
        let s5 = Shape::new(vec![
            Dimension::Static(42),
            Dimension::Dynamic(DimensionVariable::new("columns", DimensionBounds::unbounded())),
        ]);

        assert_eq!(format!("{s0}"), "[]");
        assert_eq!(format!("{s1}"), "[42, 4, 2]");
        assert_eq!(format!("{s2}"), "[4, 1]");
        assert_eq!(format!("{s3}"), "[4, depth]");
        assert_eq!(format!("{s4}"), "[rows, 42, columns]");
        assert_eq!(format!("{s5}"), "[42, columns]");
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
        let dynamic_shape = Shape::new(vec![
            Dimension::Static(42),
            Dimension::Dynamic(DimensionVariable::new("columns", DimensionBounds::unbounded())),
        ]);
        let bounded_dynamic_shape = Shape::new(vec![
            Dimension::Static(42),
            Dimension::Dynamic(DimensionVariable::new(
                "bounded_columns",
                DimensionBounds::non_negative(Some(8)).unwrap(),
            )),
        ]);

        assert_eq!(StaticShape::try_from(shape.clone()), Ok(static_shape.clone()));
        assert_eq!(StaticShape::try_from(&shape), Ok(static_shape));
        assert_eq!(
            StaticShape::try_from(dynamic_shape),
            Err(TypeError::invalid("shape dimension 1 must be static, but got columns".to_string())),
        );
        assert_eq!(
            StaticShape::try_from(&bounded_dynamic_shape),
            Err(TypeError::invalid("shape dimension 1 must be static, but got bounded_columns".to_string())),
        );
    }
}
