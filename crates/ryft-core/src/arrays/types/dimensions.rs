use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::ops::Index;
use std::sync::Arc;

use thiserror::Error;

use ryft_macros::Parameter;

use crate::axes::Axis;
use crate::parameters::Parameter;
use crate::programs::types::{type_signature_pairs, visit_type_signature_pairs};
use crate::programs::{
    ProgramError, Type, TypeError, TypeIdentity, TypeIdentityClass, TypeIdentityKind, TypeIdentityPosition,
    TypeIdentityRenaming, TypeRefinements, type_identities_needing_runtime_checks,
};

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
    pub const fn at_least(lower: usize) -> Self {
        Self { lower, upper: None }
    }

    /// Creates a new [`DimensionBounds`] instance admitting every non-negative extent.
    pub const fn unbounded() -> Self {
        Self::at_least(0)
    }

    /// Returns the inclusive lower bound of this [`DimensionBounds`] instance.
    pub const fn lower(&self) -> usize {
        self.lower
    }

    /// Returns the exclusive upper bound of this [`DimensionBounds`] instance.
    pub const fn upper(&self) -> Option<usize> {
        self.upper
    }

    /// Returns the single extent admitted by this [`DimensionBounds`] instance when it admits exactly one extent
    /// (i.e., when its bounds are `[n, n + 1)`), and [`None`] for every wider range.
    #[inline]
    pub fn extent(&self) -> Option<usize> {
        (self.lower.checked_add(1) == self.upper).then_some(self.lower)
    }

    /// Returns the inclusive lower and upper endpoints of the portable extents admitted by this [`DimensionBounds`].
    /// The upper endpoint is capped at [`MAX_DIMENSION_EXTENT`], including when the bounds are unbounded. Returns
    /// [`DimensionError::ExtentExceedsBackendWidth`] if the lower bound exceeds [`MAX_DIMENSION_EXTENT`].
    pub fn representable_extent_range(self) -> Result<(usize, usize), DimensionError> {
        if self.lower > MAX_DIMENSION_EXTENT {
            return Err(DimensionError::ExtentExceedsBackendWidth { value: self.lower, maximum: MAX_DIMENSION_EXTENT });
        }
        Ok((self.lower, self.upper.map(|upper| upper - 1).unwrap_or(MAX_DIMENSION_EXTENT).min(MAX_DIMENSION_EXTENT)))
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

    /// Returns the target of this [`DimensionVariable`] in `renaming` (itself when `renaming` does not rename it),
    /// after checking that the target's bounds are contained in this variable's bounds.
    pub fn renamed(&self, renaming: &TypeIdentityRenaming<Self>) -> Result<Self, TypeError> {
        let variable = renaming.rename(self);
        if !self.bounds().contains_bounds(variable.bounds()) {
            return Err(TypeError::invalid(format!(
                "cannot rename dimension variable {} with bounds {} to {} with bounds {}",
                self,
                self.bounds(),
                variable,
                variable.bounds(),
            )));
        }
        Ok(variable)
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

impl TypeIdentity for DimensionVariable {
    #[inline]
    fn fresh(&self) -> Self {
        Self::new(self.name(), self.bounds())
    }
}

/// [`Type`] of a first-class dimension value, described by a [`Dimension`] exactly like one array axis.
/// A static type (e.g., `dimension<3>`) describes a value whose extent is known, such as every eager
/// [`DimensionValue`](crate::DimensionValue), and has no identity, exactly like a [`Dimension::Static`] array axis.
/// A symbolic type describes a value whose extent is only known at runtime and defines its [`DimensionVariable`] as
/// an ordinary Single Static Assignment (SSA) value. Array axes may refer to that same [`DimensionVariable`], while
/// arithmetic relationships between dimension values remain explicit operation edges in the [`Program`](crate::Program)
/// graph.
///
/// A symbolic [`DimensionType`] is strictly its [`DimensionVariable`] identity plus that variable's authoritative
/// bounds. Concrete extents observed at execution boundaries are deliberately *not* part of such a type (i.e.,
/// structural type equality, hashing, display, and retained-compilation cache identity must not distinguish two calls
/// whose dynamic extents differ, or every concrete extent would acquire its own specialization). Boundary facts instead
/// live in the complete-signature [`TypeRefinements`] environment established from boundary inputs, where a static
/// dimension type that meets a symbolic one binds its variable to the static extent. A variable defined by an
/// instruction inside a program needs no input-boundary refinement (its producing instruction closes the identity
/// structurally and establishes its concrete value when the program executes).
///
/// # Motivation
///
/// First-class typed dimensions exist to make programs *shape polymorphic*. A program traced once over symbolic extents
/// is type checked once, transformed once (e.g., differentiated, batched, or partially evaluated), and then executed at
/// every concrete extent its bounds admit. All of that rests on one invariant that the entire tracing and transform
/// stack consumes: the type of every operation output is computable at trace time from the types of its inputs. Typed
/// dimensions preserve that invariant in the presence of dynamic extents, which buys us:
///
///   - **Trace-Time Shape Checking:** Shape-carrying operations prove facts such as "this reshape preserves the element
///     count" symbolically while tracing, and reject violations with a precise trace-time error instead of deferring
///     them to backend compilation or to a runtime crash on some particular input.
///   - **Transformability:** Transforms must synthesize types for values that do not exist in the original program,
///     such as batched intermediates, tangents, and cotangent zeros. Because every dynamic extent is defined by an
///     explicit dimension value in the same program, transforms can thread those definitions through the programs they
///     build (e.g., as residuals of a linearization), and the resulting programs remain well-typed and themselves shape
///     polymorphic.
///   - **One Compiled Artifact Across Extents:** Concrete extents are not part of the type, so retained compilations
///     are keyed by the symbolic signature and reused across every admitted extent, with the boundary
///     [`TypeRefinements`] environment binding actual extents per call.
///
/// The restriction this type encodes is therefore one of _provenance_. Ryft supports three increasingly dynamic tiers:
///
///   1. **Tier 1 — Static Extents:** A [`Dimension::Static`] axis, a static dimension type (e.g., that of every eager
///      [`DimensionValue`](crate::DimensionValue)), or an exact singleton-bounded dimension is known completely while
///      tracing.
///   2. **Tier 2 — Input-Derived Extents:** Dimension inputs, dynamic axes of array inputs,
///      [`DimensionSize`](crate::DimensionSize) results, and explicit arithmetic over those values remain symbolic
///      while tracing but are determined by the program boundary.
///   3. **Tier 3 — Bounded Data-Derived Extents:** The deliberate [`DimensionFromScalar`](crate::DimensionFromScalar)
///      gateway converts rank-zero integer array _data_ into a fresh bounded dimension. This is the only
///      data-to-dimension boundary. Ordinary array computation cannot implicitly acquire shape authority.
///
/// The tier-3 gateway keeps the trace-time type invariant intact because its output type is a fresh variable plus
/// declared bounds, which are computable before execution like every other dimension type. Its producing instruction
/// makes that variable an internal identity, so it neither appears in nor needs an input-boundary [`TypeRefinements`]
/// entry. Execution reads the scalar and checks the observed extent against the declared bounds before the value may
/// control array geometry. Backend representations are more permissive. For example, a StableHLO
/// [`dynamic_reshape`](https://openxla.org/stablehlo/spec#dynamic_reshape) consumes an ordinary runtime integer tensor
/// with no provenance requirement, so lowering does not need this machinery. The restriction is a frontend type-system
/// choice: an _unchecked_ data-derived extent would make output types depend on unbounded runtime values, and
/// data-dependent dynamism also remains unevenly supported across backends (refer to the
/// [dynamism in StableHLO](https://openxla.org/stablehlo/dynamism) documentation for more information).
/// The authoritative [`DimensionBounds`] carried by every [`DimensionVariable`] additionally keep each tier-2
/// or tier-3 dynamic extent inside a static envelope that backends can plan buffers and generate loops against.
///
/// Note that constructing a type with [`DimensionType::new`] creates a fresh variable identity while converting
/// an existing [`DimensionVariable`] through [`From`] preserves its identity, name, and bounds. A static type is
/// converted from a [`Dimension::Static`] extent instead.
///
/// # Example
///
/// The following program reshapes a `f64[source, 4]` input into `f64[2, source * 2]`. The input array type's dynamic
/// axis references the variable `source`, `dimension_size` defines that variable as a Single Static Assignment (SSA)
/// value, and `dimension_mul` derives a fresh variable whose diagnostic name and bounds record the arithmetic (i.e.,
/// `source ∈ [0, 9)` implies `source * 2 ∈ [0, 17)`):
///
/// ```text
/// lambda %0: f64[source, 4] .
/// let %1: dimension<2>                    = const 2
///     %2: dimension<source ∈ [0, 9)>      = dimension_size [axis=0] %0
///     %3: dimension<source * 2 ∈ [0, 17)> = dimension_mul %2 %1
///     %4: f64[2, source * 2]              = reshape %0 %1 %3
/// in (%4)
/// ```
///
/// This one program type checks, differentiates, and compiles once, and then executes at every `source ∈ [0, 9)`,
/// with the concrete extent bound at each execution boundary.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct DimensionType {
    /// [`Dimension`] described by values of this type: a static extent, or the [`DimensionVariable`]
    /// that such values define.
    dimension: Dimension,
}

impl DimensionType {
    /// Creates a new [`DimensionType`] whose values define a fresh [`DimensionVariable`] with the provided `name`
    /// and `bounds`. Each call creates a distinct identity, even when the name and bounds are identical.
    #[inline]
    pub fn new<N: Into<String>>(name: N, bounds: DimensionBounds) -> Self {
        Self::from(DimensionVariable::new(name, bounds))
    }

    /// Returns the [`Dimension`] described by this [`DimensionType`], whose [`Display`] rendering is the name of its
    /// variable if it is symbolic, and its extent if it is static.
    #[inline]
    pub fn dimension(&self) -> &Dimension {
        &self.dimension
    }

    /// Returns an owned [`Dimension`] with singleton-bounded variables replaced by their exact static extents.
    /// Variables with wider bounds retain their identity, and static dimensions retain their extent. Unlike
    /// [`Self::dimension`], this can simplify a symbolic representation; the stored dimension remains unchanged.
    #[inline]
    pub fn simplified_dimension(&self) -> Dimension {
        self.extent().map_or_else(|| self.dimension.clone(), Dimension::Static)
    }

    /// Returns the [`DimensionVariable`] defined by values of this [`DimensionType`] if it is symbolic,
    /// and [`None`] if it is static.
    #[inline]
    pub fn variable(&self) -> Option<&DimensionVariable> {
        self.dimension.variable()
    }

    /// Returns the [`DimensionVariable`] defined by values of this [`DimensionType`] if it is symbolic, and otherwise a
    /// fresh variable whose singleton bounds admit only its extent. This is the variable that decoding a runtime value
    /// into this type defines (e.g., [`DimensionFromScalar`](crate::DimensionFromScalar)), which checks the decoded
    /// extent against these bounds.
    #[inline]
    pub fn to_variable(&self) -> DimensionVariable {
        match &self.dimension {
            Dimension::Static(_) => DimensionVariable::new(self.dimension.to_string(), self.bounds()),
            Dimension::Dynamic(variable) => variable.clone(),
        }
    }

    /// Returns the authoritative [`DimensionBounds`] of this [`DimensionType`], which admit only its extent
    /// if it is static.
    #[inline]
    pub fn bounds(&self) -> DimensionBounds {
        self.dimension.bounds()
    }

    /// Returns the exact extent of this [`DimensionType`] when it is static or its bounds admit exactly one value,
    /// and [`None`] for every wider range.
    #[inline]
    pub fn extent(&self) -> Option<usize> {
        self.bounds().extent()
    }

    /// Returns the exclusive upper bound minus one, capped at [`MAX_DIMENSION_EXTENT`], or [`None`] when this
    /// [`DimensionType`]'s bounds have no finite upper bound. Unlike [`DimensionBounds::representable_extent_range`],
    /// this function preserves the distinction between finite and unbounded declared bounds.
    #[inline]
    pub fn maximum_extent(&self) -> Option<usize> {
        self.bounds().upper()?.checked_sub(1).map(|maximum| maximum.min(MAX_DIMENSION_EXTENT))
    }

    /// Returns `true` if values of this [`DimensionType`] and of `other` provably have equal extents (i.e., if both
    /// define the same [`DimensionVariable`] or their bounds admit the same single extent; refer to the documentation
    /// of [`Dimension::has_equal_extents`] for more information). Two static types are considered equal exactly when
    /// their extents are.
    #[inline]
    pub fn has_equal_extents(&self, other: &Self) -> bool {
        self.dimension.has_equal_extents(&other.dimension)
    }

    /// Checks that `actual` refines `declared` (i.e., that the actual bounds are contained in the declared bounds)
    /// and records the renaming of `declared`'s [`DimensionVariable`] to `actual`'s in `renaming` when both types are
    /// symbolic. A static type has no identity to rename or to rename to: a static `actual` instead binds the declared
    /// variable to its extent in `refinements`, exactly like a static array axis does in
    /// [`ArrayType::extend_renaming_and_refinements`](crate::ArrayType::extend_renaming_and_refinements),
    /// so that a later signature member observing a different extent for the same variable is rejected. As there,
    /// callers check that no variable is both renamed and bound using [`DimensionRefinements::require_disjoint_from`]
    /// once after folding over the complete signature.
    ///
    /// This is the single-pair step used by the [`Type::derive_identity_renaming`] implementations that fold over
    /// an entire signature like [`DimensionType`]'s own, and [`ArrayIrType`](crate::arrays::ArrayIrType)'s, which
    /// encounters dimension types as individual elements of a mixed signature. It is a separate function so that
    /// those callers can thread one shared [`TypeIdentityRenaming`] through every pair of the signature. Sharing the
    /// accumulator is what makes a declared [`DimensionVariable`] that appears in several signature positions rename
    /// consistently.
    ///
    /// # Example
    ///
    /// Given the variables `n: [0, 8)`, `m: [0, 8)`, and `k: [0, 4)`, folding over a declared signature behaves
    /// as follows:
    ///
    ///   - `(dimension<n>, dimension<n>)` instantiated by `(dimension<m>, dimension<m>)`: Records the renaming `n -> m`
    ///     and accepts its consistent repetition.
    ///   - `(dimension<n>, dimension<n>)` instantiated by `(dimension<m>, dimension<k>)`: Fails because `n` would be
    ///     renamed to both `m` and `k`.
    ///   - `(dimension<k>)` instantiated by `(dimension<n>)`: Fails because the declared bounds `[0, 4)` do not contain
    ///     the actual bounds `[0, 8)`.
    ///   - `(dimension<n>, dimension<n>)` instantiated by `(dimension<3>, dimension<3>)`: Records no renaming but the
    ///     static binding `n = 3`, and accepts its consistent repetition.
    ///   - `(dimension<n>, dimension<n>)` instantiated by `(dimension<3>, dimension<4>)`: Fails because `n` is bound
    ///     to static extent `3` and then observed as `4`.
    pub(crate) fn extend_renaming_and_refinements(
        declared: &Self,
        actual: &Self,
        renaming: &mut TypeIdentityRenaming<DimensionVariable>,
        refinements: &mut DimensionRefinements,
    ) -> Result<(), TypeError> {
        if !declared.is_refined_by(actual) {
            return Err(TypeError::invalid(format!(
                "dimension type {actual} cannot instantiate declared type {declared}",
            )));
        }
        match (declared.variable(), &actual.dimension) {
            (Some(declared), Dimension::Dynamic(actual)) => renaming.insert(declared.clone(), actual.clone()),
            (Some(declared), Dimension::Static(extent)) => refinements.bind(declared, *extent),
            (None, _) => Ok(()),
        }
    }
}

impl Display for DimensionType {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.extent(), &self.dimension) {
            (Some(extent), _) => write!(formatter, "dimension<{extent}>"),
            (None, dimension) => write!(formatter, "dimension<{dimension} ∈ {}>", self.bounds()),
        }
    }
}

impl Type for DimensionType {
    type Identity = DimensionVariable;
    type Refinements = DimensionTypeRefinements;

    #[inline]
    fn identities(&self) -> impl Iterator<Item = (TypeIdentityPosition, &Self::Identity)> {
        self.variable().into_iter().map(|variable| (TypeIdentityPosition::Definition, variable))
    }

    fn derive_identity_renaming<
        'a,
        D: IntoIterator<Item = &'a Self, IntoIter: ExactSizeIterator>,
        A: IntoIterator<Item = &'a Self, IntoIter: ExactSizeIterator>,
    >(
        declared: D,
        actual: A,
    ) -> Result<TypeIdentityRenaming<Self::Identity>, TypeError>
    where
        Self: 'a,
    {
        let mut renaming = TypeIdentityRenaming::new();
        let mut refinements = DimensionRefinements::default();
        visit_type_signature_pairs(declared, actual, |declared, actual| {
            Self::extend_renaming_and_refinements(declared, actual, &mut renaming, &mut refinements)
        })?;
        refinements.require_disjoint_from(&renaming)?;
        Ok(renaming)
    }

    #[inline]
    fn rename_identities(&self, renaming: &TypeIdentityRenaming<Self::Identity>) -> Result<Self, TypeError> {
        match self.variable() {
            Some(variable) => Ok(Self::from(variable.renamed(renaming)?)),
            None => Ok(self.clone()),
        }
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

impl From<Dimension> for DimensionType {
    #[inline]
    fn from(dimension: Dimension) -> Self {
        Self { dimension }
    }
}

impl From<DimensionVariable> for DimensionType {
    #[inline]
    fn from(variable: DimensionVariable) -> Self {
        Self { dimension: Dimension::Dynamic(variable) }
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

    /// Returns whether this [`Dimension`] and `other` _provably_ have the same extent. Equal dimensions have equal
    /// extents, including occurrences of the same symbolic variable. Distinct dimensions also have equal extents when
    /// their bounds both identify the same single integer. For example, a dynamic dimension with bounds `[3, 4)` and
    /// a static dimension of size three. Matching non-singleton bounds do not prove equality between independent
    /// variables. A `false` result means equality is unproven, not that the runtime extents necessarily differ.
    /// Unlike [`Self::is_refined_by`], this comparison is symmetric and proves equality rather than checking whether
    /// one dimension is an allowed specialization of another.
    #[inline]
    pub fn has_equal_extents(&self, other: &Self) -> bool {
        let bounds = self.bounds();
        self == other || (bounds == other.bounds() && bounds.extent().is_some())
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

impl From<&Dimension> for Dimension {
    #[inline]
    fn from(dimension: &Dimension) -> Self {
        dimension.clone()
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

/// [`TypeRefinements`] established for a complete [`DimensionType`] signature. They follow the rules of
/// first-class dimension members in the composite array universe: each actual dimension must have bounds within its
/// declared bounds, a differently named actual identity is linked to the declared one, and each connected class shares
/// its exact fact, so that a repeated declared identity rejects actual inputs of different extents (e.g.,
/// `[dimension<n>, dimension<n>]` entered with extents `2` and `3`). The formal and actual facts stay in separate
/// scopes, as documented on [`DimensionRefinements`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DimensionTypeRefinements {
    /// Underlying [`DimensionRefinements`].
    dimensions: DimensionRefinements,
}

impl DimensionTypeRefinements {
    /// Returns the [`DimensionRefinementFact`]s relating the provided [`DimensionType`]s, validating their bounds
    /// first. Differently named identities are linked, and identical singleton identities contribute their exact
    /// extent.
    pub(crate) fn dimension_refinement_facts<'t>(
        declared: &'t DimensionType,
        actual: &'t DimensionType,
    ) -> impl Iterator<Item = Result<DimensionRefinementFact<'t>, TypeError>> {
        let (error, facts) = if declared.is_refined_by(actual) {
            (
                None,
                Some(DimensionRefinements::dimension_refinement_facts(
                    declared.dimension(),
                    actual.dimension(),
                    true,
                    true,
                )),
            )
        } else {
            (Some(TypeError::invalid(format!("type {actual} does not refine declared type {declared}"))), None)
        };
        error.into_iter().map(Err).chain(facts.into_iter().flatten())
    }
}

impl TypeRefinements<DimensionType> for DimensionTypeRefinements {
    fn from_local_declarations<D: IntoIterator, A: IntoIterator>(declared: D, actual: A) -> Result<Self, TypeError>
    where
        D::IntoIter: ExactSizeIterator,
        A::IntoIter: ExactSizeIterator,
        D::Item: Borrow<DimensionType>,
        A::Item: Borrow<DimensionType>,
    {
        let declared = declared.into_iter().collect::<Vec<_>>();
        let actual = actual.into_iter().collect::<Vec<_>>();
        let dimensions = DimensionRefinements::from_refinement_facts(
            type_signature_pairs(declared.iter().map(Borrow::borrow), actual.iter().map(Borrow::borrow))?
                .flat_map(|(declared, actual)| Self::dimension_refinement_facts(declared, actual)),
        )?;
        let actual = actual.iter().map(|r#type| Some(r#type.borrow())).collect::<Vec<_>>();
        let unproven = type_identities_needing_runtime_checks::<DimensionType, _>(&declared, &actual)?;
        Ok(Self { dimensions: dimensions.with_identities_needing_runtime_checks(unproven) })
    }

    #[inline]
    fn identity_classes<'o>(&'o self) -> impl Iterator<Item = TypeIdentityClass<'o, <DimensionType as Type>::Identity>>
    where
        <DimensionType as Type>::Identity: 'o,
    {
        self.dimensions.identity_classes()
    }

    #[inline]
    fn identities_needing_runtime_checks(&self) -> &[DimensionVariable] {
        self.dimensions.identities_needing_runtime_checks()
    }

    fn validate_refinement<D: IntoIterator, A: IntoIterator>(
        &self,
        declared: D,
        actual: A,
        closed_identities: &[DimensionVariable],
    ) -> Result<(), TypeError>
    where
        D::IntoIter: ExactSizeIterator,
        A::IntoIter: ExactSizeIterator,
        D::Item: Borrow<DimensionType>,
        A::Item: Borrow<DimensionType>,
    {
        // Validation is "declared-only". It checks outputs against the declared facts and never records actual facts.
        let mut refinements = self.dimensions.clone();
        visit_type_signature_pairs(declared, actual, |declared, actual| {
            Self::dimension_refinement_facts(declared, actual).try_for_each(|refinement_fact| match refinement_fact? {
                DimensionRefinementFact::Declared(variable, extent) => {
                    refinements.validate_or_bind(variable, extent, closed_identities)
                }
                DimensionRefinementFact::Actual(..) | DimensionRefinementFact::Correspondence(..) => Ok(()),
            })
        })
    }

    #[inline]
    fn refine_declared(
        &self,
        r#type: &DimensionType,
        preserved_identities: &[DimensionVariable],
    ) -> Result<DimensionType, TypeError> {
        Ok(DimensionType::from(self.dimensions.refine_declared_dimension(r#type.dimension(), preserved_identities)))
    }

    #[inline]
    fn refine_actual(
        &self,
        r#type: &DimensionType,
        preserved_identities: &[DimensionVariable],
    ) -> Result<DimensionType, TypeError> {
        Ok(DimensionType::from(self.dimensions.refine_actual_dimension(r#type.dimension(), preserved_identities)))
    }

    #[inline]
    fn retype_actual(
        &self,
        declared: &DimensionType,
        actual: &DimensionType,
        preferred_identities: &[DimensionVariable],
    ) -> Result<DimensionType, TypeError> {
        Ok(DimensionType::from(self.dimensions.retype_actual_dimension(
            declared.dimension(),
            actual.dimension(),
            preferred_identities,
        )))
    }
}

// TODO(eaplatanios): Review this.
/// Shared dimension identity facts used by array, first-class dimension, and composite array IR [`TypeRefinements`].
/// [`Type`]-specific [`TypeRefinements`] implementations validate their input signatures and collect these facts.
/// Operations can also construct an empty set with [`Default`] and record exact extents with [`Self::bind`] or
/// [`Self::bind_dimension_input`]. To apply such facts to array types, convert this set into
/// [`ArrayTypeRefinements`](crate::ArrayTypeRefinements).
///
/// Facts are kept in two separate scopes. Formal facts bind identities of the declared signature (e.g., the formal `n`
/// of a region declared as `f32[n]` that receives an `f32[3]`) and apply to declared, intermediate, and output types
/// through [`TypeRefinements::refine_declared`]. Actual facts bind identities of the supplied signature and apply to
/// the supplied input types through [`TypeRefinements::refine_actual`]. The two scopes stay separate even when the same
/// identity object occurs on both sides of an invocation: matching declared `[dimension<n>, dimension<m>, f32[m]]`
/// against actual `[dimension<m>, dimension<n>, f32[2]]` fixes the formal `m` and the actual `n` to 2, while the formal
/// `n` and the actual `m` stay unresolved. Establishment links each formal identity to the actual identity in the same
/// position and collects their scoped classes, sharing one exact fact within each class. For example, matching declared
/// `[dimension<n>, dimension<p>, f32[n]]` against actual `[dimension<m>, dimension<m>, f32[2]]` also fixes the actual
/// `m` and the formal `p` to 2. These facts hold for one invocation only and never mutate a caller identity. Input
/// relationships that the supplied types do not prove must still be checked at runtime.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DimensionRefinements {
    /// Scoped identity classes, in recording order, with each class's exact extent when established.
    classes: Vec<DimensionIdentityClass>,

    /// Declared input identities whose relationships the supplied types do not prove.
    unproven: Vec<DimensionVariable>,
}

// TODO(eaplatanios): Review this.
impl DimensionRefinements {
    /// Consumes refinement facts in iterator order and builds their scoped identity classes.
    ///
    /// Each item contains a borrowed fact or a comparison error. Records each fact immediately and stops on the first
    /// comparison or recording error. After the iterator is exhausted, connected identities are merged and their
    /// shared extents are checked against bounds. Stored identities are owned; borrowed facts do not escape construction.
    pub(crate) fn from_refinement_facts<'t>(
        facts: impl IntoIterator<Item = Result<DimensionRefinementFact<'t>, TypeError>>,
    ) -> Result<Self, TypeError> {
        let mut refinements = Self::default();
        let mut indices = HashMap::new();
        let mut correspondences = Vec::new();
        // Record scoped identities, exact extents, and correspondence edges before joining their classes.
        for refinement_fact in facts {
            let refinement_fact = refinement_fact?;
            let mut index_of = |variable: &DimensionVariable, kind: TypeIdentityKind| {
                *indices.entry((variable.clone(), kind)).or_insert_with(|| {
                    let mut class = DimensionIdentityClass::default();
                    match kind {
                        TypeIdentityKind::Declared => class.formal.push(variable.clone()),
                        TypeIdentityKind::Actual => class.actual.push(variable.clone()),
                    }
                    refinements.classes.push(class);
                    refinements.classes.len() - 1
                })
            };
            match refinement_fact {
                DimensionRefinementFact::Declared(variable, extent) => {
                    let index = index_of(variable, TypeIdentityKind::Declared);
                    refinements.bind_class(index, variable, extent, TypeIdentityKind::Declared)?;
                }
                DimensionRefinementFact::Actual(variable, extent) => {
                    let index = index_of(variable, TypeIdentityKind::Actual);
                    refinements.bind_class(index, variable, extent, TypeIdentityKind::Actual)?;
                }
                DimensionRefinementFact::Correspondence(formal, actual) => {
                    correspondences.push((
                        index_of(formal, TypeIdentityKind::Declared),
                        index_of(actual, TypeIdentityKind::Actual),
                    ));
                }
            }
        }

        // Join linked identities, validate receiving bounds, and preserve recording order in the final classes.
        let mut parents = (0..refinements.classes.len()).collect::<Vec<_>>();
        let mut members = (0..refinements.classes.len()).map(|index| vec![index]).collect::<Vec<_>>();
        let mut extents = refinements.classes.iter().map(|class| class.extent).collect::<Vec<_>>();
        for &(formal, actual) in &correspondences {
            let formal_root = Self::root(&mut parents, formal);
            let actual_root = Self::root(&mut parents, actual);
            if formal_root == actual_root {
                continue;
            }
            let extent = match (extents[formal_root], extents[actual_root]) {
                (Some(expected), Some(actual)) if expected != actual => {
                    return Err(DimensionError::InputDimensionMismatch {
                        dimension: refinements.classes[formal].formal[0].to_string(),
                        expected,
                        actual,
                    }
                    .into());
                }
                (Some(extent), _) | (_, Some(extent)) => Some(extent),
                (None, None) => None,
            };
            if let Some(extent) = extent {
                // Each receiving node is checked only when its component first obtains a fact. Diagnose the first
                // recorded invalid member, independent of union roots and node-list concatenation order.
                let invalid = [formal_root, actual_root]
                    .into_iter()
                    .filter(|&root| extents[root].is_none())
                    .flat_map(|root| members[root].iter().copied())
                    .filter(|&index| {
                        refinements.classes[index]
                            .formal
                            .iter()
                            .chain(&refinements.classes[index].actual)
                            .any(|variable| !variable.bounds().contains(extent))
                    })
                    .min();
                if let Some(index) = invalid {
                    let variable = refinements.classes[index]
                        .formal
                        .iter()
                        .chain(&refinements.classes[index].actual)
                        .next()
                        .unwrap();
                    Self::validate_extent(variable, extent)?;
                }
            }
            // Joining the smaller node list into the larger bounds construction work without changing recording
            // order: final classes are emitted by the original node vector below, never by union roots or hash order.
            let (root, merged) = if members[formal_root].len() >= members[actual_root].len() {
                (formal_root, actual_root)
            } else {
                (actual_root, formal_root)
            };
            parents[merged] = root;
            let merged_members = std::mem::take(&mut members[merged]);
            members[root].extend(merged_members);
            extents[root] = extent;
        }
        let mut classes = Vec::<DimensionIdentityClass>::new();
        let mut positions = vec![None; refinements.classes.len()];
        for (index, class) in std::mem::take(&mut refinements.classes).into_iter().enumerate() {
            let root = Self::root(&mut parents, index);
            let position = *positions[root].get_or_insert_with(|| {
                classes.push(DimensionIdentityClass { extent: extents[root], ..DimensionIdentityClass::default() });
                classes.len() - 1
            });
            classes[position].formal.extend(class.formal);
            classes[position].actual.extend(class.actual);
        }
        refinements.classes = classes;
        Ok(refinements)
    }

    /// Attaches the original input signature's runtime obligations to its conditional facts.
    pub(crate) fn with_identities_needing_runtime_checks(mut self, unproven: Vec<DimensionVariable>) -> Self {
        self.unproven = unproven;
        self
    }

    /// Returns the connected classes of declared and actual identities, in recording order.
    /// Each class reports whether its shared extent is known; refer to the scope examples on [`Self`].
    pub fn identity_classes(&self) -> impl Iterator<Item = TypeIdentityClass<'_, DimensionVariable>> {
        self.classes
            .iter()
            .map(|class| TypeIdentityClass::new(&class.formal, &class.actual, class.extent.is_some()))
    }

    /// Returns declared input identities whose relationships still need runtime checking.
    /// Exact extent facts alone do not prove those relationships; refer to [`TypeRefinements`].
    pub fn identities_needing_runtime_checks(&self) -> &[DimensionVariable] {
        &self.unproven
    }

    /// Records one concrete extent for the declared identity `variable`, rejecting a conflicting extent fact.
    pub fn bind(&mut self, variable: &DimensionVariable, extent: usize) -> Result<(), TypeError> {
        let index = match self.class_index(variable, TypeIdentityKind::Declared) {
            Some(index) => index,
            None => {
                self.classes.push(DimensionIdentityClass {
                    formal: vec![variable.clone()],
                    ..DimensionIdentityClass::default()
                });
                self.classes.len() - 1
            }
        };
        self.bind_class(index, variable, extent, TypeIdentityKind::Declared)
    }

    /// Accepts the first-class dimension input of type `input_type` for a dynamic output axis whose identity is
    /// `variable`, as dynamic constructors and other operations whose inputs supply output extents require. An input
    /// that defines `variable` itself keeps that identity in the output. An exact input (i.e., one whose bounds admit a
    /// single extent, such as the type of an eager dimension value) whose extent lies within `variable`'s bounds binds
    /// `variable` to that extent, so that refining the output type with these refinements replaces the axis by that
    /// static extent. Returns whether the input is accepted.
    ///
    /// # Errors
    ///
    /// Returns an error when an exact input conflicts with an extent already bound for `variable` (e.g., two inputs of
    /// extents 2 and 3 for a repeated axis).
    pub fn bind_dimension_input(
        &mut self,
        variable: &DimensionVariable,
        input_type: &DimensionType,
    ) -> Result<bool, TypeError> {
        if input_type.variable() == Some(variable) {
            return Ok(true);
        }
        match input_type.extent() {
            Some(extent) if variable.bounds().contains(extent) => {
                self.bind(variable, extent)?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// Validates one observed extent against an established input fact, or binds it when `variable` belongs to the
    /// validated boundary's closed identity signature. This is the validation-phase counterpart of [`Self::bind`]. For
    /// a variable that already has a recorded extent, the two behave identically (i.e., a matching extent is accepted
    /// and a differing one is rejected). They differ on a variable with no recorded fact. [`Self::bind`]
    /// treats the refinement set as *open* and records the first supplied extent, which is correct while
    /// establishing facts from a complete input signature, where every variable may legitimately appear for the
    /// first time, as in [`TypeRefinements::from_local_declarations`] and
    /// [`ArrayType::extend_renaming_and_refinements`][extend_renaming_and_refinements].
    ///
    /// [`Self::validate_or_bind`] instead admits an unrecorded variable only when it appears in `closed_identities`,
    /// the boundary's complete closed identity set (i.e., an identity established by the formal input signature, whose
    /// first concrete fact may only become observable at an output, such as a first-class dimension input whose type is
    /// strictly identity plus bounds, or an identity defined by an instruction inside the program or region). In either
    /// case it delegates to [`Self::bind`] so repeated extent facts across several outputs must still agree; this is
    /// sound without inspecting runtime payloads because structural region closure already proves that every output
    /// reference identity is consumed or defined. Every variable outside the closed set is rejected, because an extent
    /// claimed for an identity foreign to the boundary cannot be justified. In short, use [`Self::bind`] where
    /// comparisons *establish* facts, and use this function where extent facts must be *justified* against the boundary's
    /// identity set, as [`TypeRefinements::validate_refinement`] does when checking an output signature against the
    /// facts established from its input boundary.
    ///
    /// [extend_renaming_and_refinements]: crate::arrays::types::arrays::ArrayType::extend_renaming_and_refinements
    pub fn validate_or_bind(
        &mut self,
        variable: &DimensionVariable,
        extent: usize,
        closed_identities: &[DimensionVariable],
    ) -> Result<(), TypeError> {
        match self.lookup(variable, TypeIdentityKind::Declared) {
            Some(expected) if expected != extent => Err(DimensionError::InputDimensionMismatch {
                dimension: variable.to_string(),
                expected,
                actual: extent,
            }
            .into()),
            Some(_) => Ok(()),
            None if closed_identities.contains(variable) => self.bind(variable, extent),
            None => Err(TypeError::invalid(format!(
                "dimension identity {variable} does not belong to the validated boundary signature",
            ))),
        }
    }

    /// Rejects a [`DimensionVariable`] that one signature member binds to a static extent while another member renames
    /// it to a variable whose bounds do not specify that exact extent. A singleton target bound equal to the static
    /// extent is consistent as first-class dimensions retain a nominal variable even when their extent is statically
    /// known. Wider bounds do not prove that the two extent facts agree. Complete signature renaming derivations call
    /// this after folding over every member, so the check is independent of the order in which the facts are recorded.
    pub fn require_disjoint_from(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<(), TypeError> {
        for (variable, extent) in self
            .classes
            .iter()
            .filter_map(|class| class.extent.map(|extent| (class, extent)))
            .flat_map(|(class, extent)| class.formal.iter().map(move |variable| (variable, extent)))
        {
            if let Some(target) = renaming.target(variable) {
                if target.bounds().extent() == Some(extent) {
                    continue;
                }
                return Err(TypeError::invalid(format!(
                    "dimension variable {variable} is renamed to {target} by one signature member and bound to \
                     static extent {extent} by another",
                )));
            }
        }
        Ok(())
    }

    /// Returns a declared dimension with its class's exact extent applied, unless its identity is listed in
    /// `preserved_identities` (e.g., identities defined by outputs).
    pub(crate) fn refine_declared_dimension(
        &self,
        dimension: &Dimension,
        preserved_identities: &[DimensionVariable],
    ) -> Dimension {
        match dimension {
            Dimension::Dynamic(variable) if !preserved_identities.contains(variable) => self
                .lookup(variable, TypeIdentityKind::Declared)
                .map(Dimension::Static)
                .unwrap_or_else(|| dimension.clone()),
            _ => dimension.clone(),
        }
    }

    /// Returns an actual dimension with its class's exact fact applied, unless its identity must remain symbolic.
    pub(crate) fn refine_actual_dimension(
        &self,
        dimension: &Dimension,
        preserved_identities: &[DimensionVariable],
    ) -> Dimension {
        match dimension {
            Dimension::Dynamic(variable) if !preserved_identities.contains(variable) => self
                .lookup(variable, TypeIdentityKind::Actual)
                .map(Dimension::Static)
                .unwrap_or_else(|| dimension.clone()),
            Dimension::Dynamic(_) | Dimension::Static(_) => dimension.clone(),
        }
    }

    /// Returns the supplied dimension under a retained member of its declared class, or its exact refinement.
    pub(crate) fn retype_actual_dimension(
        &self,
        declared: &Dimension,
        actual: &Dimension,
        preferred_identities: &[DimensionVariable],
    ) -> Dimension {
        if let Dimension::Dynamic(formal) = declared
            && let Some(index) = self.class_index(formal, TypeIdentityKind::Declared)
        {
            let class = &self.classes[index];
            // Several reference handles may retain different names in one checked class. Preserve a supplied
            // retained name rather than replacing it by another handle's identity.
            if let Dimension::Dynamic(variable) = actual
                && preferred_identities.contains(variable)
                && class.actual.contains(variable)
                && formal.bounds().contains_bounds(variable.bounds())
            {
                return actual.clone();
            }
            if let Some(preferred_identity) = preferred_identities.iter().find(|candidate| {
                class.actual.contains(candidate) && formal.bounds().contains_bounds(candidate.bounds())
            }) {
                return Dimension::Dynamic(preferred_identity.clone());
            }
            // An unfixed class pinned under a wider name still needs one compatible representation for this
            // component. Prefer the body's own identity when an actual input carries it; otherwise use the first
            // compatible actual member. Only the checked class supplies these alternatives, so unrelated names
            // cannot supply a new name, and classes without a preferred member remain unchanged.
            if class.extent.is_none() && preferred_identities.iter().any(|candidate| class.actual.contains(candidate)) {
                if class.actual.contains(formal) {
                    return Dimension::Dynamic(formal.clone());
                }
                if let Some(candidate) =
                    class.actual.iter().find(|candidate| formal.bounds().contains_bounds(candidate.bounds()))
                {
                    return Dimension::Dynamic(candidate.clone());
                }
            }
        }
        // A wider retained name cannot describe this declared component, even when the entry proves its extent.
        // Apply that exact fact instead of preserving the incompatible name through the preserved identity list.
        let preserves_actual = actual.variable().is_some_and(|variable| {
            preferred_identities.contains(variable) && declared.bounds().contains_bounds(variable.bounds())
        });
        self.refine_actual_dimension(actual, if preserves_actual { preferred_identities } else { &[] })
    }

    /// Returns the refinement facts for one declared and one actual dimension, or their comparison error.
    /// Differently named identities are admitted only when `admits_renamed` is set and the actual bounds fit within
    /// the declared bounds. Identical dimensions contribute facts only when `include_identical` is set.
    pub(crate) fn dimension_refinement_facts<'t>(
        declared: &'t Dimension,
        actual: &'t Dimension,
        admits_renamed: bool,
        include_identical: bool,
    ) -> impl Iterator<Item = Result<DimensionRefinementFact<'t>, TypeError>> {
        // One pair can establish at most three facts: declared extent, actual extent, and their correspondence.
        let mut facts = [None, None, None];
        if declared == actual && !include_identical {
            return facts.into_iter().flatten();
        }
        // Singleton identities supply the same exact extent as static dimensions while retaining their names.
        let (exact_actual, actual_variable) = match actual {
            Dimension::Static(extent) => (Some(*extent), None),
            Dimension::Dynamic(variable) => (variable.bounds().extent(), Some(variable)),
        };
        let mismatch =
            || TypeError::invalid(format!("dimension {actual} does not refine declared dimension {declared}"));
        match (declared, exact_actual, actual_variable) {
            (Dimension::Static(declared_extent), exact_actual, actual_variable) => {
                if exact_actual != Some(*declared_extent) {
                    facts[0] = Some(Err(mismatch()));
                } else if let Some(actual_variable) = actual_variable {
                    facts[0] = Some(Ok(DimensionRefinementFact::Actual(actual_variable, *declared_extent)));
                }
            }
            (Dimension::Dynamic(declared_variable), Some(extent), actual_variable) => {
                if !declared_variable.bounds().contains(extent) {
                    facts[0] = Some(Err(DimensionError::BindingOutOfBounds {
                        variable: declared_variable.to_string(),
                        value: extent,
                        bounds: declared_variable.bounds(),
                    }
                    .into()));
                } else {
                    facts[0] = Some(Ok(DimensionRefinementFact::Declared(declared_variable, extent)));
                    if let Some(actual_variable) = actual_variable {
                        facts[1] = Some(Ok(DimensionRefinementFact::Actual(actual_variable, extent)));
                        facts[2] =
                            Some(Ok(DimensionRefinementFact::Correspondence(declared_variable, actual_variable)));
                    }
                }
            }
            (Dimension::Dynamic(declared_variable), None, Some(actual_variable)) => {
                facts[0] = Some(
                    if actual_variable != declared_variable
                        && (!admits_renamed || !declared_variable.bounds().contains_bounds(actual_variable.bounds()))
                    {
                        Err(mismatch())
                    } else {
                        Ok(DimensionRefinementFact::Correspondence(declared_variable, actual_variable))
                    },
                );
            }
            (Dimension::Dynamic(_), None, None) => facts[0] = Some(Err(mismatch())),
        }
        facts.into_iter().flatten()
    }

    /// Finds the class containing `variable` in the requested scope.
    fn class_index(&self, variable: &DimensionVariable, kind: TypeIdentityKind) -> Option<usize> {
        self.classes.iter().position(|class| match kind {
            TypeIdentityKind::Declared => class.formal.contains(variable),
            TypeIdentityKind::Actual => class.actual.contains(variable),
        })
    }

    /// Returns the exact extent of the requested scoped identity's class.
    fn lookup(&self, variable: &DimensionVariable, kind: TypeIdentityKind) -> Option<usize> {
        self.class_index(variable, kind).and_then(|index| self.classes[index].extent)
    }

    /// Records a fact at a known class position, validating every other receiving member before publishing it.
    fn bind_class(
        &mut self,
        index: usize,
        variable: &DimensionVariable,
        extent: usize,
        kind: TypeIdentityKind,
    ) -> Result<(), TypeError> {
        match self.classes[index].extent {
            Some(expected) if expected != extent => Err(DimensionError::InputDimensionMismatch {
                dimension: variable.to_string(),
                expected,
                actual: extent,
            }
            .into()),
            Some(_) => Ok(()),
            None => {
                // A later declared extent fact (including output validation) fixes every linked member. Keep `bind`'s
                // direct binding contract, but validate the authoritative bounds of all receiving identities.
                let class = &self.classes[index];
                for (member, member_kind) in class
                    .formal
                    .iter()
                    .map(|member| (member, TypeIdentityKind::Declared))
                    .chain(class.actual.iter().map(|member| (member, TypeIdentityKind::Actual)))
                {
                    if member_kind != kind || member != variable {
                        Self::validate_extent(member, extent)?;
                    }
                }
                self.classes[index].extent = Some(extent);
                Ok(())
            }
        }
    }

    /// Returns the temporary union root, compressing paths so later lookups do not repeat the traversal.
    fn root(parents: &mut [usize], mut index: usize) -> usize {
        while parents[index] != index {
            parents[index] = parents[parents[index]];
            index = parents[index];
        }
        index
    }

    /// Validates an inferred extent against the authoritative bounds of its receiving identity.
    fn validate_extent(variable: &DimensionVariable, extent: usize) -> Result<(), TypeError> {
        if !variable.bounds().contains(extent) {
            return Err(DimensionError::BindingOutOfBounds {
                variable: variable.to_string(),
                value: extent,
                bounds: variable.bounds(),
            }
            .into());
        }
        Ok(())
    }
}

// TODO(eaplatanios): Review this.
/// One connected class of scoped dimension identities and its invocation-local exact fact.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct DimensionIdentityClass {
    /// Declared identities, in recording order.
    formal: Vec<DimensionVariable>,

    /// Supplied identities, in recording order.
    actual: Vec<DimensionVariable>,

    /// Exact extent shared by this class, when known.
    extent: Option<usize>,
}

// TODO(eaplatanios): Review this.
/// A dimension fact derived by comparing declared and actual types for [`DimensionRefinements`].
///
/// These facts describe how dimensions relate when the supplied values satisfy the declared input signature.
/// Any required runtime checks must succeed before relying on the facts. Deriving a refinement fact only inspects
/// types; it neither reads runtime dimension values nor performs those checks.
///
/// For example, declared `dimension<n>` and actual `dimension<m>` with bounds fixing `m` to `2` produce
/// `Declared(n, 2)`, `Actual(m, 2)`, and `Correspondence(n, m)`.
pub(crate) enum DimensionRefinementFact<'o> {
    /// The declared identity has this exact extent in the invocation.
    Declared(&'o DimensionVariable, usize),

    /// The actual identity has this exact extent in the invocation.
    Actual(&'o DimensionVariable, usize),

    /// The declared identity and the actual identity denote the same runtime extent in the invocation.
    Correspondence(&'o DimensionVariable, &'o DimensionVariable),
}

/// Represents an array's ordered [`Dimension`]s. Note that the [`Display`] implementation of [`Shape`] renders shapes
/// as the rendered dimension sizes in a comma-separated list surrounded by square brackets.
///
/// Converting host-size arrays, slices, or vectors creates static dimensions in the same order. Empty collections
/// represent a scalar shape, and zero sizes represent empty axes. Borrowed collections are copied into the shape.
///
/// # Examples
///
/// ```rust
/// # use ryft_core::Shape;
/// let shape = Shape::from([2, 3]);
/// assert_eq!(shape.to_string(), "[2, 3]");
/// assert_eq!(shape, Shape::from(vec![2, 3]));
/// assert_eq!(shape, Shape::from([2, 3].as_slice()));
/// ```
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
    /// # use ryft_core::arrays::{Dimension, DimensionBounds, DimensionVariable, Shape};
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
    pub fn element_count(&self) -> Result<Option<usize>, TypeError> {
        if self.dimensions.contains(&Dimension::Static(0)) {
            return Ok(Some(0));
        }
        let mut count = 1usize;
        for dimension in &self.dimensions {
            match dimension {
                Dimension::Static(extent) => {
                    count = count.checked_mul(*extent).ok_or_else(|| {
                        TypeError::invalid(format!("shape {self} element count does not fit in `usize`"))
                    })?;
                }
                Dimension::Dynamic(_) => return Ok(None),
            }
        }
        Ok(Some(count))
    }

    /// Returns whether this [`Shape`] and `other` _provably_ contain the same number of elements, even when their
    /// dimensions are dynamic. The proof compares the products of static extents and matches dynamic dimension
    /// identities with multiplicity, independently of axis order. For example, `[n, 4]` and `[2, n, 2]` have equal
    /// element counts, whereas `[n, 4]` and `[m, 4]` are not proven equal for independent variables `n` and `m`.
    ///
    /// A static zero makes a shape's product zero without multiplying its remaining extents. A scalar has product
    /// one. Dynamic dimension bounds and relationships between distinct variables are not used to establish equality.
    /// `Ok(false)` means equality was not proven, not necessarily that the runtime element counts differ. Returns a
    /// [`TypeError`] if either shape's static coefficient overflows [`usize`].
    ///
    /// # Parameters
    ///
    ///   - `other`: [`Shape`] whose element count is compared with this [`Shape`]'s element count.
    pub fn has_equal_element_count(&self, other: &Self) -> Result<bool, TypeError> {
        let coefficient = |shape: &Shape| {
            if shape.dimensions().contains(&Dimension::Static(0)) {
                return Ok(0usize);
            }
            shape.dimensions().iter().try_fold(1usize, |product, dimension| {
                product.checked_mul(dimension.value().unwrap_or(1)).ok_or_else(|| {
                    TypeError::invalid(format!(
                        "shape `{shape}` static element-count coefficient does not fit in `usize`",
                    ))
                })
            })
        };
        let self_coefficient = coefficient(self)?;
        let other_coefficient = coefficient(other)?;
        if self_coefficient != other_coefficient {
            return Ok(false);
        }
        if self_coefficient == 0 {
            return Ok(true);
        }
        let mut remaining =
            other.dimensions().iter().filter(|dimension| dimension.value().is_none()).collect::<Vec<_>>();
        for dimension in self.dimensions().iter().filter(|dimension| dimension.value().is_none()) {
            let Some(index) = remaining.iter().position(|candidate| *candidate == dimension) else {
                return Ok(false);
            };
            remaining.swap_remove(index);
        }
        Ok(remaining.is_empty())
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
    /// # use ryft_core::arrays::{Dimension, DimensionBounds, DimensionVariable, Shape};
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

impl<const N: usize> From<[usize; N]> for Shape {
    #[inline]
    fn from(dimensions: [usize; N]) -> Self {
        Self::new(dimensions.into_iter().map(Dimension::Static).collect())
    }
}

impl<const N: usize> From<&[usize; N]> for Shape {
    #[inline]
    fn from(dimensions: &[usize; N]) -> Self {
        Self::from(dimensions.as_slice())
    }
}

impl From<&[usize]> for Shape {
    #[inline]
    fn from(dimensions: &[usize]) -> Self {
        Self::new(dimensions.iter().copied().map(Dimension::Static).collect())
    }
}

impl From<Vec<usize>> for Shape {
    #[inline]
    fn from(dimensions: Vec<usize>) -> Self {
        Self::new(dimensions.into_iter().map(Dimension::Static).collect())
    }
}

impl From<&Vec<usize>> for Shape {
    #[inline]
    fn from(dimensions: &Vec<usize>) -> Self {
        Self::from(dimensions.as_slice())
    }
}

impl From<StaticShape> for Shape {
    #[inline]
    fn from(value: StaticShape) -> Self {
        Self::new(value.dimensions.into_iter().map(Dimension::Static).collect())
    }
}

impl From<&StaticShape> for Shape {
    #[inline]
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

/// [`Type`] universe whose types may describe first-class dimensions. Universe-parameterized dimension-valued
/// capabilities use this view in their provided functions to read the [`DimensionType`] of a value in any universe
/// that has first-class dimensions, so that those functions do not depend on one particular universe.
pub trait AsDimensionType: Type {
    /// Returns the [`DimensionType`] that this type describes, or a [`TypeError`] if it does not describe a first-class
    /// dimension (e.g., an array or a reference in the array IR universe).
    fn as_dimension_type(&self) -> Result<&DimensionType, TypeError>;
}

impl AsDimensionType for DimensionType {
    #[inline]
    fn as_dimension_type(&self) -> Result<&DimensionType, TypeError> {
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use pretty_assertions::assert_eq;

    use crate::arrays::dimensions::DimensionValue;
    use crate::arrays::operations::DimensionOperation;
    use crate::operations::GuardOperation;
    use crate::parameters::Placeholder;
    use crate::programs::{Operation, ProgramBuilder};

    use super::*;

    #[test]
    fn test_dimension_bounds() {
        assert_eq!(DimensionBounds::new(0, Some(0)), Err(DimensionError::InvalidBounds { lower: 0, upper: 0 }));
        assert_eq!(DimensionBounds::new(3, Some(3)), Err(DimensionError::InvalidBounds { lower: 3, upper: 3 }));
        assert_eq!(DimensionBounds::new(4, Some(3)), Err(DimensionError::InvalidBounds { lower: 4, upper: 3 }));

        let non_negative = DimensionBounds::non_negative(Some(5)).unwrap();
        assert_eq!(non_negative.lower(), 0);
        assert_eq!(non_negative.upper(), Some(5));
        assert_eq!(non_negative.representable_extent_range(), Ok((0, 4)));
        assert_eq!(DimensionBounds::new(0, Some(1)).unwrap().representable_extent_range(), Ok((0, 0)));
        assert!(non_negative.contains(0));
        assert!(non_negative.contains(4));
        assert!(!non_negative.contains(5));
        assert_eq!(non_negative.to_string(), "[0, 5)");

        let positive = DimensionBounds::positive(Some(5)).unwrap();
        assert_eq!(positive.representable_extent_range(), Ok((1, 4)));
        assert_eq!(DimensionBounds::new(2, Some(9)).unwrap().representable_extent_range(), Ok((2, 8)));
        assert!(!positive.contains(0));
        assert!(positive.contains(1));
        assert!(non_negative.contains_bounds(positive));
        assert!(!positive.contains_bounds(non_negative));

        // Only bounds that admit exactly one extent determine that extent.
        assert_eq!(DimensionBounds::new(0, Some(1)).unwrap().extent(), Some(0));
        assert_eq!(DimensionBounds::new(7, Some(8)).unwrap().extent(), Some(7));
        assert_eq!(non_negative.extent(), None);
        assert_eq!(DimensionBounds::at_least(7).extent(), None);

        let unbounded = DimensionBounds::unbounded();
        assert_eq!(unbounded, DimensionBounds::at_least(0));
        assert_eq!(unbounded.representable_extent_range(), Ok((0, MAX_DIMENSION_EXTENT)));
        assert!(unbounded.contains(usize::MAX));
        assert!(unbounded.contains_bounds(non_negative));
        assert!(!non_negative.contains_bounds(unbounded));
        assert_eq!(unbounded.to_string(), "[0, ∞)");

        assert_eq!(
            DimensionBounds::at_least(MAX_DIMENSION_EXTENT).representable_extent_range(),
            Ok((MAX_DIMENSION_EXTENT, MAX_DIMENSION_EXTENT)),
        );

        if let Some(unrepresentable_extent) = MAX_DIMENSION_EXTENT.checked_add(1) {
            assert_eq!(
                DimensionBounds::new(2, Some(usize::MAX)).unwrap().representable_extent_range(),
                Ok((2, MAX_DIMENSION_EXTENT)),
            );
            assert_eq!(
                DimensionBounds::at_least(unrepresentable_extent).representable_extent_range(),
                Err(DimensionError::ExtentExceedsBackendWidth {
                    value: unrepresentable_extent,
                    maximum: MAX_DIMENSION_EXTENT,
                }),
            );
        }

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
        let fresh = DimensionType::new("declared", declared_variable.bounds());
        let another_fresh = DimensionType::new(String::from("declared"), declared_variable.bounds());
        assert_eq!(fresh.variable().unwrap().name(), "declared");
        assert_eq!(fresh.bounds(), declared_variable.bounds());
        assert_ne!(fresh.variable(), Some(&declared_variable));
        assert_ne!(fresh.variable(), another_fresh.variable());

        let declared = DimensionType::from(declared_variable.clone());
        let actual = DimensionType::from(actual_variable.clone());
        assert_eq!(declared.simplified_dimension(), Dimension::Dynamic(declared_variable.clone()));

        assert_eq!(declared.variable(), Some(&declared_variable));
        assert_eq!(DimensionType::from(declared_variable.clone()), declared);
        assert_eq!(declared.bounds(), DimensionBounds::new(1, Some(65)).unwrap());
        assert_eq!(declared.to_string(), "dimension<declared ∈ [1, 65)>");
        assert!(!declared.is_scalar());
        assert!(!declared.is_complex());
        assert!(declared.is_refined_by(&actual));

        // A dimension type is strictly identity plus bounds. Exact singleton bounds are the only source of a static
        // extent, and no concrete boundary observation participates in the type itself.
        assert_eq!(declared.extent(), None);
        assert_eq!(declared.maximum_extent(), Some(64));
        let exact_variable = DimensionVariable::new("7", DimensionBounds::new(7, Some(8)).unwrap());
        let exact_type = DimensionType::from(exact_variable.clone());
        assert_eq!(exact_type.simplified_dimension(), Dimension::Static(7));
        // Simplification leaves the stored symbolic representation and its identity unchanged.
        assert_eq!(exact_type.dimension(), &Dimension::Dynamic(exact_variable));
        assert_eq!(exact_type.extent(), Some(7));
        assert_eq!(exact_type.maximum_extent(), Some(7));
        assert_eq!(exact_type.to_string(), "dimension<7>");
        assert!(declared.is_refined_by(&exact_type));
        assert!(!exact_type.is_refined_by(&declared));

        let unbounded = DimensionType::new("unbounded", DimensionBounds::unbounded());
        assert_eq!(unbounded.maximum_extent(), None);
        let zero = DimensionType::new("zero", DimensionBounds::new(0, Some(1)).unwrap());
        assert_eq!(zero.maximum_extent(), Some(0));
        let wide = DimensionType::new("wide", DimensionBounds::new(0, Some(usize::MAX)).unwrap());
        assert_eq!(wide.maximum_extent(), Some((usize::MAX - 1).min(MAX_DIMENSION_EXTENT)));

        let identities =
            declared.identities().map(|(position, variable)| (position, variable.clone())).collect::<Vec<_>>();
        assert_eq!(identities, vec![(TypeIdentityPosition::Definition, declared_variable.clone())]);

        let renaming =
            DimensionType::derive_identity_renaming(std::slice::from_ref(&declared), std::slice::from_ref(&actual))
                .unwrap();
        assert_eq!(renaming.replacements(), &[(declared_variable.clone(), actual_variable.clone())]);
        assert_eq!(declared.rename_identities(&renaming), Ok(actual.clone()));

        let wider = DimensionType::new("wider", DimensionBounds::new(0, Some(129)).unwrap());
        assert!(!declared.is_refined_by(&wider));
        assert_eq!(
            DimensionType::derive_identity_renaming(&[declared.clone()], &[wider]),
            Err(TypeError::invalid(
                "dimension type dimension<wider ∈ [0, 129)> cannot instantiate declared type \
                 dimension<declared ∈ [1, 65)>",
            )),
        );

        // Static types bind a repeated declared variable to their extents, which must agree. Separate symbolic
        // definitions remain distinct identities even when their bounds admit the same single extent.
        let seven = DimensionType::from(Dimension::Static(7));
        let eight = DimensionType::from(Dimension::Static(8));
        assert_eq!(
            DimensionType::derive_identity_renaming(
                &[declared.clone(), declared.clone()],
                &[seven.clone(), seven.clone()]
            )
            .map(|renaming| renaming.is_identity()),
            Ok(true),
        );
        assert_eq!(
            DimensionType::derive_identity_renaming(&[declared.clone(), declared.clone()], &[seven, eight]),
            Err(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "declared".to_string(),
                expected: 7,
                actual: 8,
            })),
        );
        let second_exact = DimensionType::new("another_seven", exact_type.bounds());
        assert!(
            DimensionType::derive_identity_renaming(&[declared.clone(), declared.clone()], &[exact_type, second_exact])
                .is_err()
        );

        // A repeated declared variable must rename consistently across the complete signature.
        let other = DimensionType::new("other", DimensionBounds::new(1, Some(65)).unwrap());
        assert_eq!(
            DimensionType::derive_identity_renaming(&[declared.clone(), declared], &[actual.clone(), other.clone()],),
            Err(TypeError::invalid(format!(
                "type identity `declared` is renamed to both `{}` and `{}`",
                actual.dimension(),
                other.dimension(),
            ))),
        );
    }

    #[test]
    fn test_dimension_type_programs_check_repeated_identities() {
        // A standalone dimension program and an entry guard over dimensions both reject different extents for a
        // repeated identity, like their composite counterparts.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let mut builder = ProgramBuilder::<DimensionValue, DimensionOperation<DimensionValue>>::new();
        let first = builder.add_input(n.clone());
        builder.add_input(n.clone());
        let program = builder
            .build::<Vec<DimensionValue>, Vec<DimensionValue>>(vec![first], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let value = |extent: usize| DimensionValue::constant(extent).unwrap();
        let mismatch = Err(ProgramError::Type(TypeError::from(DimensionError::InputDimensionMismatch {
            dimension: "n".to_string(),
            expected: 2,
            actual: 3,
        })));
        assert_eq!(program.interpret(vec![value(2), value(2)]), Ok(vec![value(2)]));
        assert_eq!(program.interpret(vec![value(2), value(3)]), mismatch);
        let exact = |extent: usize| DimensionType::from(Dimension::Static(extent));
        let guard = GuardOperation::new("region", vec![n.clone(), n]);
        assert_eq!(guard.infer_output_types(&[exact(2), exact(2)], &[]), Ok(vec![exact(2), exact(2)]));
        assert_eq!(
            guard.infer_output_types(&[exact(2), exact(3)], &[]).map_err(ProgramError::Type),
            mismatch.map(|_: Vec<DimensionValue>| Vec::new()),
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
    fn test_dimension_has_equal_extents() {
        let variable = DimensionVariable::new("size", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic = Dimension::Dynamic(variable);
        let independent = Dimension::Dynamic(DimensionVariable::new("size", DimensionBounds::new(1, Some(5)).unwrap()));
        let exact = Dimension::Dynamic(DimensionVariable::new("exact", DimensionBounds::new(3, Some(4)).unwrap()));
        assert!(dynamic.has_equal_extents(&dynamic));
        assert!(!dynamic.has_equal_extents(&independent));
        assert!(exact.has_equal_extents(&Dimension::Static(3)));
        assert!(Dimension::Static(3).has_equal_extents(&exact));
        assert!(!exact.has_equal_extents(&Dimension::Static(4)));
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
    fn test_dimension_type_refinements_from_local_declarations() {
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [n, m] = ["n", "m"].map(|name| DimensionType::new(name, bounds));
        let exact = |extent: usize| DimensionType::from(Dimension::Static(extent));

        // A repeated declared identity rejects actual inputs of different extents, and exact facts propagate to the
        // formal identity and to a linked actual identity.
        assert_eq!(
            DimensionTypeRefinements::from_local_declarations(&[n.clone(), n.clone()], &[exact(2), exact(3)]),
            Err(TypeError::from(DimensionError::InputDimensionMismatch {
                dimension: "n".to_string(),
                expected: 2,
                actual: 3,
            })),
        );
        let refinements =
            DimensionTypeRefinements::from_local_declarations(&[n.clone(), n.clone()], &[m.clone(), exact(2)]).unwrap();
        assert_eq!(refinements.refine_declared(&n, &[]), Ok(exact(2)));
        assert_eq!(refinements.refine_declared(&n, std::slice::from_ref(n.variable().unwrap())), Ok(n.clone()));
        assert_eq!(refinements.refine_actual(&m, &[]), Ok(exact(2)));

        // Actual bounds must lie within the declared bounds.
        let wide = DimensionType::new("wide", DimensionBounds::positive(Some(16)).unwrap());
        assert_eq!(
            DimensionTypeRefinements::from_local_declarations(std::slice::from_ref(&n), std::slice::from_ref(&wide)),
            Err(TypeError::invalid(format!("type {wide} does not refine declared type {n}"))),
        );

        // Validation checks outputs against the formal facts only.
        let closed = [n.variable().unwrap().clone()];
        assert_eq!(refinements.validate_refinement(std::slice::from_ref(&n), &[exact(2)], &closed), Ok(()));
        assert_eq!(
            refinements.validate_refinement(std::slice::from_ref(&n), &[exact(3)], &closed),
            Err(TypeError::from(DimensionError::InputDimensionMismatch {
                dimension: "n".to_string(),
                expected: 2,
                actual: 3,
            })),
        );
    }

    #[test]
    fn test_dimension_type_refinements_identity_classes() {
        let formal = DimensionType::new("formal", DimensionBounds::non_negative(Some(8)).unwrap());
        let actual = DimensionType::new("actual", DimensionBounds::non_negative(Some(8)).unwrap());
        let refinements = DimensionTypeRefinements::from_local_declarations([&formal], [&actual]).unwrap();
        assert_eq!(
            refinements.identity_classes().collect::<Vec<_>>(),
            vec![TypeIdentityClass::new(
                std::slice::from_ref(formal.variable().unwrap()),
                std::slice::from_ref(actual.variable().unwrap()),
                false,
            )],
        );
    }

    #[test]
    fn test_dimension_refinements_from_refinement_facts() {
        let empty = std::iter::empty::<Result<DimensionRefinementFact<'_>, TypeError>>();
        assert_eq!(DimensionRefinements::from_refinement_facts(empty), Ok(DimensionRefinements::default()));

        let error = TypeError::invalid("dimension comparison failed");
        let facts = std::iter::once(Err(error.clone()))
            .chain(std::iter::from_fn(|| panic!("facts after a comparison error must not be requested")));
        assert_eq!(DimensionRefinements::from_refinement_facts(facts), Err(error));

        let variable = DimensionVariable::new("length", DimensionBounds::non_negative(Some(8)).unwrap());
        let facts =
            [Ok(DimensionRefinementFact::Declared(&variable, 2)), Ok(DimensionRefinementFact::Declared(&variable, 3))]
                .into_iter()
                .chain(std::iter::from_fn(|| panic!("facts after a recording error must not be requested")));
        let error = DimensionRefinements::from_refinement_facts(facts).unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::InputDimensionMismatch { dimension: "length".to_string(), expected: 2, actual: 3 }),
        );
    }

    #[test]
    fn test_dimension_refinements() {
        let first = DimensionVariable::new("first", DimensionBounds::non_negative(Some(8)).unwrap());
        let mut isolated = DimensionRefinements::default();
        isolated.bind(&first, 3).unwrap();
        isolated.bind(&first, 3).unwrap();
        let error = isolated.bind(&first, 4).unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::InputDimensionMismatch { dimension: "first".to_string(), expected: 3, actual: 4 }),
        );
        assert_eq!(
            isolated.identity_classes().collect::<Vec<_>>(),
            vec![TypeIdentityClass::new(std::slice::from_ref(&first), &[], true)],
        );
        assert_eq!(isolated.refine_declared_dimension(&first.clone().into(), &[]), Dimension::Static(3));

        // Binding a declared identity does not reuse a fact for the same identity object in the actual scope.
        let mut scoped =
            DimensionRefinements::from_refinement_facts([Ok(DimensionRefinementFact::Actual(&first, 2))]).unwrap();
        scoped.bind(&first, 3).unwrap();
        assert_eq!(scoped.refine_declared_dimension(&first.clone().into(), &[]), Dimension::Static(3));
        assert_eq!(scoped.refine_actual_dimension(&first.into(), &[]), Dimension::Static(2));
    }

    #[test]
    fn test_dimension_refinements_identity_classes_validate_later_facts() {
        let formal = DimensionVariable::new("formal", DimensionBounds::non_negative(Some(8)).unwrap());
        let actual = DimensionVariable::new("actual", DimensionBounds::non_negative(Some(2)).unwrap());
        let mut refinements = DimensionRefinements::from_refinement_facts([Ok(
            DimensionRefinementFact::Correspondence(&formal, &actual),
        )])
        .unwrap();
        let error = refinements.bind(&formal, 2).unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::BindingOutOfBounds {
                variable: "actual".to_string(),
                value: 2,
                bounds: actual.bounds(),
            }),
        );
        assert_eq!(
            refinements.identity_classes().collect::<Vec<_>>(),
            vec![TypeIdentityClass::new(std::slice::from_ref(&formal), std::slice::from_ref(&actual), false)],
        );
    }

    #[test]
    fn test_shape_rank() {
        let s0 = Shape::scalar();
        let s1 = Shape::new(vec![Dimension::Static(42)]);
        let s2 = Shape::new(vec![
            Dimension::Static(4),
            Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
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
                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
            ])
            .element_count(),
            Ok(None)
        );
        assert_eq!(
            Shape::new(vec![
                Dimension::Static(42),
                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::non_negative(Some(8)).unwrap())),
            ])
            .element_count(),
            Ok(None)
        );
        assert_eq!(
            Shape::new(vec![
                Dimension::Static(0),
                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
            ])
            .element_count(),
            Ok(Some(0))
        );
        assert_eq!(
            Shape::new(vec![Dimension::Static(usize::MAX), Dimension::Static(2)]).element_count(),
            Err(TypeError::invalid(format!("shape [{}, 2] element count does not fit in `usize`", usize::MAX))),
        );
    }

    #[test]
    fn test_shape_has_equal_element_count() {
        let n = DimensionVariable::new("n", DimensionBounds::new(1, Some(9)).unwrap());
        let m = DimensionVariable::new("m", DimensionBounds::new(1, Some(9)).unwrap());

        // Static products are compared directly, and a static zero proves a zero count without multiplying through
        // the remaining extents, even when they would overflow.
        assert_eq!(Shape::new(vec![2.into(), 3.into()]).has_equal_element_count(&Shape::new(vec![6.into()])), Ok(true));
        assert_eq!(
            Shape::new(vec![2.into(), 3.into()]).has_equal_element_count(&Shape::new(vec![5.into()])),
            Ok(false),
        );
        assert_eq!(
            Shape::new(vec![0.into(), usize::MAX.into(), 2.into()])
                .has_equal_element_count(&Shape::new(vec![0.into()])),
            Ok(true),
        );
        assert_eq!(
            Shape::new(vec![0.into(), Dimension::Dynamic(n.clone())])
                .has_equal_element_count(&Shape::new(vec![Dimension::Dynamic(m.clone()), 0.into()])),
            Ok(true),
        );

        // Dynamic identities must match with multiplicity once the static coefficients agree.
        assert_eq!(
            Shape::new(vec![Dimension::Dynamic(n.clone()), 2.into()])
                .has_equal_element_count(&Shape::new(vec![2.into(), Dimension::Dynamic(n.clone())])),
            Ok(true),
        );
        assert_eq!(
            Shape::new(vec![Dimension::Dynamic(n.clone()), Dimension::Dynamic(n.clone())]).has_equal_element_count(
                &Shape::new(vec![Dimension::Dynamic(n.clone()), Dimension::Dynamic(n.clone())]),
            ),
            Ok(true),
        );
        assert_eq!(
            Shape::new(vec![Dimension::Dynamic(n.clone()), Dimension::Dynamic(n.clone())])
                .has_equal_element_count(&Shape::new(vec![Dimension::Dynamic(n.clone())])),
            Ok(false),
        );
        assert_eq!(
            Shape::new(vec![Dimension::Dynamic(n.clone())])
                .has_equal_element_count(&Shape::new(vec![Dimension::Dynamic(m)])),
            Ok(false),
        );
        assert_eq!(
            Shape::new(vec![Dimension::Dynamic(n.clone()), 2.into()])
                .has_equal_element_count(&Shape::new(vec![2.into()])),
            Ok(false),
        );

        // Static coefficients that do not fit are reported rather than wrapped.
        assert_eq!(
            Shape::new(vec![usize::MAX.into(), 2.into()])
                .has_equal_element_count(&Shape::new(vec![Dimension::Dynamic(n)])),
            Err(TypeError::invalid(format!(
                "shape `[{}, 2]` static element-count coefficient does not fit in `usize`",
                usize::MAX,
            ))),
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
    fn test_shape_from_sizes() {
        let expected = Shape::new(vec![Dimension::Static(2), Dimension::Static(0), Dimension::Static(3)]);
        assert_eq!(Shape::from([2, 0, 3]), expected);
        assert_eq!(Shape::from(&[2, 0, 3]), expected);
        assert_eq!(Shape::from([2, 0, 3].as_slice()), expected);
        assert_eq!(Shape::from(vec![2, 0, 3]), expected);
        assert_eq!(Shape::from(&vec![2, 0, 3]), expected);

        // Empty collections describe scalars, rather than a zero-sized axis.
        assert_eq!(Shape::from([]), Shape::scalar());
        assert_eq!(Shape::from(&[]), Shape::scalar());
        assert_eq!(Shape::from([].as_slice()), Shape::scalar());
        assert_eq!(Shape::from(Vec::<usize>::new()), Shape::scalar());
        assert_eq!(Shape::from(&Vec::<usize>::new()), Shape::scalar());
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
