use std::borrow::{Borrow, Cow};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::sync::Arc;

use thiserror::Error;

use crate::errors::CustomError;
use crate::parameters::Parameter;
use crate::programs::identities::{TypeIdentity, TypeIdentityPosition, TypeIdentityRenaming};

/// Represents errors produced while inferring or validating [`Type`]s.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
pub enum TypeError {
    /// A generic type contract was invalid.
    #[error("{message}")]
    Invalid { message: String },

    /// A type family produced an error with a concrete, recoverable type.
    #[error("{0}")]
    Custom(Arc<dyn CustomError>),
}

impl TypeError {
    /// Creates a new [`TypeError::Invalid`] error with the provided message.
    #[inline]
    pub fn invalid<M: Into<String>>(message: M) -> Self {
        Self::Invalid { message: message.into() }
    }

    /// Wraps a type-family-specific error in a [`Custom`](TypeError::Custom) variant. The concrete error can later be
    /// recovered using [`TypeError::downcast_custom`].
    #[inline]
    pub fn custom<T: CustomError>(error: T) -> Self {
        Self::Custom(Arc::new(error))
    }

    /// Returns the wrapped custom error downcast to `T` when this is a [`Custom`](TypeError::Custom) variant holding a
    /// `T`, and [`None`] otherwise.
    #[inline]
    pub fn downcast_custom<T: CustomError>(&self) -> Option<&T> {
        match self {
            // Deref through the `Arc` to the `dyn CustomError`, upcast to `&dyn std::error::Error`, and then use the
            // standard error downcast. Going through the `Arc` directly would downcast the `Arc` instead of the error.
            Self::Custom(custom) => (&**custom as &dyn std::error::Error).downcast_ref::<T>(),
            _ => None,
        }
    }
}

/// Lightweight type-level description of a family of runtime values. A [`Type`] captures the structural metadata that
/// Ryft needs to reason about values without inspecting the values themselves. Examples include scalar data types such
/// as [`DataType`](crate::DataType), array-like types that combine an element [`DataType`](crate::DataType) with shape
/// information, and richer types for traced values.
///
/// Note that [`Type`] requires [`Clone`] so that types can be duplicated into staged [`Program`](crate::Program)s
/// returned via [`Cow`] using the [`Typed`] trait, and stored in tracing data structures. It requires [`Debug`] and
/// [`Display`] so diagnostics and rendered programs can show types consistently without forcing every call site to
/// repeat those bounds. It also requires [`PartialEq`] because type equality is fundamental to type inference and
/// validation, and so generic code bounded on [`Type`] can compare types without repeating that bound. Finally, it
/// requires [`Parameter`] so that types can be used as leaves in [`Parameterized`](crate::Parameterized) data
/// structures.
pub trait Type: Clone + Debug + Display + PartialEq + Parameter {
    /// Nominal identity carried by this type family's metadata. A [`TypeIdentity`] represents a declared equality
    /// relationship between otherwise dynamic parts of types, rather than a runtime value or a Single Static
    /// Assignment (SSA) atom. Repeating the same identity means those occurrences denote the same runtime quantity.
    /// Type families without such relationships use [`NoIdentity`](crate::NoIdentity).
    type Identity: TypeIdentity;

    /// Facts that one invocation of a [`Program`](crate::Program) or [`Region`](crate::Region) boundary establishes by
    /// validating its complete input signature, ensuring that repeated identity occurrences remain consistent across
    /// boundary leaves and outputs. The facts about the declared (i.e., formal) [`TypeIdentity`]s and those about the
    /// identities of the supplied (i.e., actual) types are kept in separate scopes. For example, an array refinement
    /// environment remembers the concrete extent observed for a dynamic dimension, carries it across a first-class
    /// dimension input to the actual identity that enters the same declared identity, and rejects a conflicting extent
    /// elsewhere in the same signature. Type families that need no such facts use `()`. Also, type refinements never
    /// inspect or retain runtime value payloads.
    type Refinements: TypeRefinements<Self>;

    /// Validates that this type admits a zero value, independently of its shape or runtime allocation. Constructor
    /// inference uses this before staging. The default accepts the type, but families with restrictions override it
    /// so even empty values follow the same representability rules as non-empty values.
    #[inline]
    fn validate_zero(&self) -> Result<(), TypeError> {
        Ok(())
    }

    /// Validates that this type admits a one value, independently of its shape or runtime allocation. Like
    /// [`Type::validate_zero`], this validates representability without constructing a value. The default accepts
    /// the type, but families whose values cannot represent one override it.
    #[inline]
    fn validate_one(&self) -> Result<(), TypeError> {
        Ok(())
    }

    /// Returns the [`TypeIdentity`]s carried by this [`Type`] in deterministic positional order. Each item contains
    /// the occurrence's [`TypeIdentityPosition`] and a borrowed identity. Types without identities return an empty
    /// iterator.
    #[inline]
    fn identities(&self) -> impl Iterator<Item = (TypeIdentityPosition, &Self::Identity)> {
        std::iter::empty()
    }

    /// Derives the simultaneous [`TypeIdentityRenaming`] implied by matching a complete declared type signature
    /// against an actual type signature. Matching is directional: `declared` contains the formal types of a
    /// [`Program`](crate::Program) or [`Region`](crate::Region) boundary, while `actual` contains the corresponding
    /// types supplied at an instantiation site. The method validates that the actual signature can instantiate the
    /// declared signature and derives every declared-identity-to-actual-identity correspondence. For example:
    ///
    /// ```text
    /// declared: [array<n>, array<m>]
    /// actual:   [array<batch>, array<sequence>]
    /// renaming: n -> batch, m -> sequence
    /// ```
    ///
    /// The renaming is derived once from the complete boundary signature and then reused with
    /// [`Type::rename_identities`] to update all metadata related to that boundary, including intermediate and output
    /// types. Values, operations, and nested regions that store such types must apply the same renaming. Those internal
    /// types do not generally have corresponding actual types from which they could independently derive the mapping.
    ///
    /// Note that the returned [`TypeIdentityRenaming`] is simultaneous meaning that applying it does not recursively
    /// rename a target merely because that target is also a source. This makes swaps and permutations well-defined.
    ///
    /// When a declared identity is instantiated by a static component, there is no actual identity to place in the
    /// renaming. The implementation must still validate bounds and ensure that repeated occurrences of the declared
    /// identity observe consistent static refinements.
    ///
    /// Returns a [`TypeError`] when the signatures are structurally incompatible, an actual type cannot instantiate its
    /// declared type, bounds are violated, or repeated occurrences imply conflicting identity mappings or refinements.
    ///
    /// # Parameters
    ///
    ///   - `declared`: Borrowed types of the complete signature declared by the program or region boundary.
    ///   - `actual`: Corresponding borrowed types supplied at the instantiation site.
    #[inline]
    fn derive_identity_renaming<
        't,
        D: IntoIterator<Item = &'t Self, IntoIter: ExactSizeIterator>,
        A: IntoIterator<Item = &'t Self, IntoIter: ExactSizeIterator>,
    >(
        declared: D,
        actual: A,
    ) -> Result<TypeIdentityRenaming<Self::Identity>, TypeError>
    where
        Self: 't,
    {
        Self::Refinements::establish(declared, actual)?;
        Ok(TypeIdentityRenaming::new())
    }

    /// Returns this [`Type`] after simultaneously renaming all of its live [`TypeIdentity`]s.
    #[inline]
    fn rename_identities(&self, renaming: &TypeIdentityRenaming<Self::Identity>) -> Result<Self, TypeError> {
        let _ = renaming;
        Ok(self.clone())
    }

    /// Returns `true` if values described by this [`Type`] are compatible with the provided [`Type`]. The precise
    /// notion of compatibility is type-specific. For example, scalar data types may treat compatibility as promotion
    /// while array-like types may account for broadcasting and nested structure.
    ///
    /// This relation describes *implicit convertibility*: a value of the receiver's type is not itself a value of
    /// `other`, but it can be turned into one (e.g., by promoting its [`DataType`](crate::DataType) or broadcasting its
    /// [`Shape`](crate::Shape)), possibly losing information along the way. Contrast this with [`Self::is_refined_by`],
    /// which holds only when a value already *is* a value of the other type, with no conversion involved. For example,
    /// `f16` is compatible with (i.e., promotable to) `f32` but does not refine it, while a dynamically shaped `f32`
    /// array type of shape `[?, 3]` is refined by an `f32` array type of shape `[2, 3]` with no conversion involved
    /// at all.
    fn is_compatible_with(&self, other: &Self) -> bool;

    /// Returns `true` if every value described by `other` is also described by this [`Type`]. The receiver is the
    /// more general type (e.g., a declared or staged type) and the argument is the more precise one (e.g., the actual
    /// type carried by a runtime value), and so the relation is directional: `declared.is_refined_by(&actual)`.
    /// Interpretation entry points such as [`Program::interpret`](crate::Program::interpret) use this relation to
    /// validate runtime input values against declared program input types.
    ///
    /// For fully static types this is type equality (e.g., [`DataType`](crate::DataType) requires equal data types).
    /// Types that can carry unknown components additionally admit every more precise instantiation of those components.
    /// For example, [`ArrayType`](crate::ArrayType)s with [`Dimension::Dynamic`](crate::Dimension::Dynamic) dimensions
    /// are refined by otherwise-equal [`ArrayType`](crate::ArrayType)s whose corresponding dimensions are static, per
    /// [`Dimension::is_refined_by`](crate::Dimension::is_refined_by).
    ///
    /// Reading each [`Type`] as the set of values it describes, this relation is equivalent to set inclusion
    /// (i.e., argument ⊆ receiver) and forms a partial ordering (i.e., semantic subtyping where `other` is a subtype of
    /// the receiver, with no conversion involved). This is what distinguishes it from [`Self::is_compatible_with`],
    /// whose promotion- and broadcasting-based notions describe *implicit convertibility* between values of different
    /// types rather than containment. This is also consistent with the notion of refinement in
    /// [StableHLO](https://openxla.org/stablehlo/dynamism) and MLIR.
    fn is_refined_by(&self, other: &Self) -> bool;

    /// Returns `true` if this [`Type`] describes a single scalar (i.e., a rank-`0` array/tensor) value. This predicate
    /// exists to let reverse-mode differentiation enforce scalar-output functions. Reverse-mode differentiation seeds
    /// the output cotangent with the multiplicative identity (i.e., a value of one) and pulls it back to the inputs.
    /// That seed represents the derivative of the output with respect to itself and is only meaningful when the output
    /// is a scalar for simple gradients (i.e., not Jacobians).
    fn is_scalar(&self) -> bool;

    /// Returns `true` if this [`Type`] describes complex-valued (e.g., [`DataType::C64`](crate::DataType::C64) or
    /// [`DataType::C128`](crate::DataType::C128)) numeric values. Like [`Self::is_scalar`], this predicate primarily
    /// exists to serve reverse-mode differentiation. A single reverse-mode seed recovers the derivative of a
    /// complex-output function only when the function is _holomorphic_ (i.e., complex-differentiable), so the gradient
    /// entry points route complex scalar outputs through their `*_holomorphic` variants. The plain entry points reject
    /// output types for which this returns `true`, and the holomorphic ones reject output types for which it returns
    /// `false`.
    fn is_complex(&self) -> bool;

    /// Returns `true` if this [`Type`] describes a [`ReferenceType`](crate::ReferenceType). Transform boundaries use
    /// this predicate to reject unresolved references even when no stateful operation is present, such as a pure
    /// reference passthrough. The predicate is all that the core [`Type`] contract says about references. Type
    /// universes that also need the referent itself implement the [`ReferenceMemberType`](crate::ReferenceMemberType)
    /// extension, whose projections are required to agree with this predicate.
    #[inline]
    fn is_reference(&self) -> bool {
        false
    }
}

/// A borrowed view of a group of [`TypeIdentity`]s that must describe the same value for an invocation's entry
/// contract to hold. The group contains identities from the declared input types and from the actual supplied types.
/// [`TypeRefinements::identity_classes`] exposes these groups so that consumers can preserve relationships between
/// identities even when their exact values are unknown.
///
/// For example, entering declared types `[dimension<n>, dimension<p>]` with actual types `[dimension<m>,
/// dimension<m>]` relates both `n` and `p` to `m`. The resulting class contains declared identities `[n, p]` and actual
/// identities `[m]`. None of their extents needs to be known. If another input establishes that `n` has extent `2`, the
/// entire class has that extent and [`Self::is_exact`] returns `true`.
///
/// Declared and actual identities belong to separate scopes, even when the same identity object occurs in both.
/// Members are in first-observation order within each scope. Groups connect through correspondences admitted by the
/// type family; unrelated identities do not join merely because they have equal exact values. The relationships
/// describe what the entry contract requires (i.e., this view does not perform or prove the runtime entry check).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TypeIdentityClass<'o, I: TypeIdentity> {
    /// Identities of the declared signature in this class.
    declared: &'o [I],

    /// Identities of the supplied signature in this class.
    actual: &'o [I],

    /// Whether the invocation fixes this class to an exact refinement.
    exact: bool,
}

// TODO(eaplatanios): Review this.
impl<'o, I: TypeIdentity> TypeIdentityClass<'o, I> {
    /// Creates a semantic view of one class with the provided scoped members and exactness.
    ///
    /// # Parameters
    ///
    ///   - `declared`: Declared identities, in first-observation order.
    ///   - `actual`: Supplied identities, in first-observation order.
    ///   - `exact`: Whether the invocation fixes this class to an exact refinement.
    pub fn new(declared: &'o [I], actual: &'o [I], exact: bool) -> Self {
        Self { declared, actual, exact }
    }

    /// Returns the declared identities in first-observation order.
    pub fn declared(&self) -> &'o [I] {
        self.declared
    }

    /// Returns the supplied identities in first-observation order.
    pub fn actual(&self) -> &'o [I] {
        self.actual
    }

    /// Returns whether the invocation determines an exact refinement for this class.
    pub fn is_exact(&self) -> bool {
        self.exact
    }
}

// TODO(eaplatanios): Review this.
/// Facts that one invocation establishes by validating a complete type signature at a [`Program`](crate::Program) or
/// [`Region`](crate::Region) boundary, ensuring that repeated identity occurrences remain consistent across boundary
/// leaves and outputs.
///
/// [`Type::is_refined_by`] checks one type pair. This companion contract checks a complete boundary so relationships
/// repeated across several inputs or outputs remain consistent. Establishment via [`Self::establish`] is transactional
/// across the complete input signature and is the one place that reasons about the complete invocation. Its facts live
/// in two scopes that are never merged, even when the same identity object occurs on both sides: _formal_ facts about
/// the declared [`TypeIdentity`]s, which [`Self::refine_declared`] applies to declared, intermediate, and output types
/// and against which [`Self::validate`] checks outputs, and _actual_ facts about the identities of the supplied types,
/// which [`Self::refine_actual`] applies to those supplied input types. [`TypeIdentity`]s produced inside the
/// [`Program`](crate::Program) or [`Region`](crate::Region) may establish additional output facts, while unrelated,
/// unbound output identities are rejected.
///
/// The facts hold for one invocation only, conditional on its entry check. They never mutate a caller identity and
/// never justify dropping an entry check; relationships that the actual types leave unproven still need the boundary or
/// an [`GuardOperation`](crate::GuardOperation).
pub trait TypeRefinements<T: Type>: Clone + Debug + Default {
    /// Establishes the refinement facts of one invocation from the complete `actual` input signature relative to the
    /// corresponding `declared` signature. Each pair must first be admissible on its own (i.e., `actual` must refine
    /// `declared` up to the identity correspondences that the family admits). The facts that the pairs observe directly
    /// are shared within each connected class of formal/actual identity correspondences, so that, for
    /// example, entering `[dimension<n>, dimension<p>, f32[n]]` with `[dimension<m>, dimension<m>, f32[2]]` fixes the
    /// actual `m` and the formal `p` to 2 as well. A class fact that a receiving identity's bounds exclude, or
    /// that conflicts with another fact about the same identity, is rejected. Correspondences without an exact fact
    /// (e.g., a repeated formal identity entered by two distinct symbolic actual identities) add no fact; their runtime
    /// relationship stays the responsibility of the entry check.
    fn establish<D: IntoIterator, A: IntoIterator>(declared: D, actual: A) -> Result<Self, TypeError>
    where
        D::IntoIter: ExactSizeIterator,
        A::IntoIter: ExactSizeIterator,
        D::Item: Borrow<T>,
        A::Item: Borrow<T>;

    /// Establishes the refinement facts of `actual` relative to `declared` types whose identities are _local_ to the
    /// consumer that declares them, such as the declared types of an
    /// [`GuardOperation`](crate::GuardOperation), which name relationships rather than identities of the
    /// surrounding program. Unlike [`Self::establish`], any actual identity may then enter a declared identity, which
    /// links the two as first-class dimension members are linked, provided that the declared bounds contain the
    /// actual bounds, so that a consistent renaming of the surrounding program (e.g., from `n` to `m`) never changes
    /// what the declaration accepts. Every other rule of [`Self::establish`] applies, including shared class facts.
    fn establish_scoped<D: IntoIterator, A: IntoIterator>(declared: D, actual: A) -> Result<Self, TypeError>
    where
        D::IntoIter: ExactSizeIterator,
        A::IntoIter: ExactSizeIterator,
        D::Item: Borrow<T>,
        A::Item: Borrow<T>;

    /// Validates `actual` against `declared` using the refinement facts previously established from the corresponding
    /// input signature. Facts already stored in `self` must remain consistent at every occurrence of the corresponding
    /// identity. An identity in `closed_identities` (i.e., the validated boundary's complete closed identity set,
    /// covering both identities established by the formal input signature and identities defined by instructions inside
    /// the program or region) may establish a new refinement fact at its first concrete occurrence in the validated
    /// signature, and subsequent occurrences must agree with that fact. This is sound without inspecting runtime
    /// payloads (structural region closure already rejects an instruction result reference unless the producing
    /// instruction consumes or defines the identity, so validation only has to enforce cross-observation consistency).
    /// An unbound identity outside the closed set is rejected.
    ///
    /// `closed_identities` only identifies which identities may establish facts during validation. It does not contain
    /// concrete extents or inspect runtime value payloads.
    ///
    /// # Parameters
    ///
    ///   - `declared`: Complete declared type signature being validated, normally a program or region output signature.
    ///   - `actual`: Corresponding refining type signature supplied at the validation site.
    ///   - `closed_identities`: The validated boundary's complete closed identity set
    ///     (e.g., a [`TypeIdentitySignature`](crate::TypeIdentitySignature)'s
    ///     [`identities`](crate::TypeIdentitySignature::identities) slice), whose members are permitted to establish
    ///     refinement facts when first encountered in `actual`.
    fn validate<D: IntoIterator, A: IntoIterator>(
        &self,
        declared: D,
        actual: A,
        closed_identities: &[T::Identity],
    ) -> Result<(), TypeError>
    where
        D::IntoIter: ExactSizeIterator,
        A::IntoIter: ExactSizeIterator,
        D::Item: Borrow<T>,
        A::Item: Borrow<T>;

    /// Returns the invocation's semantic identity classes in first-observation order. Every observed identity
    /// belongs to one class in its own scope, including isolated observations and exact facts. The same identity
    /// object in the formal and actual scopes denotes two nodes, joined only when the invocation observes a
    /// correspondence. Equal exact refinements do not connect otherwise unrelated classes. The classes describe
    /// conditional entry relationships, so consuming them never replaces the corresponding runtime entry check.
    fn identity_classes<'o>(&'o self) -> impl Iterator<Item = TypeIdentityClass<'o, T::Identity>>
    where
        T::Identity: 'o;

    /// Returns `declared` with every identity that these refinement facts determine replaced by its refinement, except
    /// the identities in `symbolic`, which stay symbolic. This applies the facts that [`Self::establish`] derived from
    /// a complete input signature to a declared type (e.g., a declared output type of an operation whose inputs refine
    /// its declared input types). Because the facts are established from the complete signature first, an identity in
    /// `symbolic` still takes part in conflict detection and is only kept symbolic when the facts are applied.
    /// Exclusion applies per identity, so a type that mentions both an excluded identity and a refinable one is still
    /// refined at the latter. Refinements never add metadata that is not a fact about an identity (e.g., an array's
    /// sharding or layout).
    ///
    /// # Example
    ///
    /// A region that declares its input as `f32[rows, columns]` and receives an `f32[3, 4]` input establishes the
    /// facts `rows = 3` and `columns = 4`, which then refine the region's declared output types:
    ///
    /// ```rust
    /// # use ryft_core::{
    /// #     ArrayType, ArrayTypeRefinements, DataType, DimensionBounds, DimensionVariable, Shape, TypeRefinements,
    /// # };
    /// let rows = DimensionVariable::new("rows", DimensionBounds::positive(Some(8)).unwrap());
    /// let columns = DimensionVariable::new("columns", DimensionBounds::positive(Some(8)).unwrap());
    /// let declared_input_type =
    ///     ArrayType::new(DataType::F32, Shape::new(vec![rows.clone().into(), columns.clone().into()]));
    /// let input_type = ArrayType::new_static(DataType::F32, [3, 4]);
    /// let refinements = ArrayTypeRefinements::establish([&declared_input_type], [&input_type]).unwrap();
    ///
    /// // Every identity that has a fact is replaced by it, wherever it appears in the declared type.
    /// let declared_output_type =
    ///     ArrayType::new(DataType::F32, Shape::new(vec![columns.clone().into(), rows.clone().into()]));
    /// assert_eq!(refinements.refine_declared(&declared_output_type, &[]).unwrap().to_string(), "f32[4, 3]");
    ///
    /// // Identities in `symbolic` (e.g., ones that an output defines and that may therefore change, such as the extent
    /// // of a loop's dimension carry) are kept, while the remaining identities of the same type are still refined.
    /// assert_eq!(
    ///     refinements.refine_declared(&declared_output_type, &[rows]).unwrap().to_string(),
    ///     "f32[4, rows]",
    /// );
    /// ```
    ///
    /// # Parameters
    ///
    ///   - `declared`: Declared type to refine.
    ///   - `symbolic`: Identities that must stay symbolic in the returned type.
    fn refine_declared(&self, declared: &T, symbolic: &[T::Identity]) -> Result<T, TypeError>;

    /// Returns `actual`, one type of the supplied input signature from which these facts were established, with every
    /// identity that the actual facts fix replaced by its refinement, except for the identities in `symbolic`,
    /// preserving its layout, placement, and other non-identity metadata. For example, entering `[dimension<n>,
    /// f32[n]]` with `[dimension<m>, f32[2]]` refines the supplied `dimension<m>` to `dimension<2>`, whereas a formal
    /// fact never applies to an actual identity merely because the two share a name. Identities in `symbolic` are kept
    /// even when the facts fix them (e.g., ones that another value which cannot be refined, such as a reference, also
    /// carries), while the remaining identities of the same type are still refined.
    ///
    /// # Parameters
    ///
    ///   - `actual`: Supplied input type to refine.
    ///   - `symbolic`: Actual identities that must stay symbolic in the returned type.
    fn refine_actual(&self, actual: &T, symbolic: &[T::Identity]) -> Result<T, TypeError>;

    /// Returns `actual`, one type of the supplied input signature from which these facts were established, entering
    /// the `declared` type, refined like [`Self::refine_actual`] with the `representatives` kept symbolic, except that
    /// each component whose declared identity is linked (directly, or through other formal and actual identities that
    /// denote the same runtime extent) to one of the `representatives` takes the first compatible representative
    /// instead, even where `actual` is exact or carries another identity. For bounded identities, compatibility requires
    /// the declared component's bounds to contain the representative's bounds. An actual identity already in
    /// `representatives` and in that declared class keeps its own compatible name, so immutable handles may preserve
    /// their different identities within one checked class. When an unfixed class has requested representatives but
    /// none fits a component, that component uses its declared identity if an actual member carries it, otherwise the
    /// first compatible actual member. This keeps the body's representation within the declared component's bounds;
    /// exact classes instead apply their exact fact. For example, scoped establishment of `[f32[m], f32[m]]` with
    /// `[f32[a], f32[3]]` fixes `a = 3`, and representing the second input with a compatible `a` returns `f32[a]` rather
    /// than `f32[3]`, while entering `[dimension<m>, dimension<m>]` with `[dimension<a>, dimension<b>]` relates `a` and
    /// `b` without an extent, and representing the second input with a compatible `a` returns `dimension<a>`. Consumers
    /// use this to give related values compatible representations while respecting immutable reference geometry. A
    /// changed known representation may require a residual checked value. The result is a representation justified by
    /// the complete entry check, rather than a runtime check performed by this type-level function.
    ///
    /// # Parameters
    ///
    ///   - `declared`: Declared type that `actual` enters.
    ///   - `actual`: Supplied input type to refine.
    ///   - `representatives`: Actual identities that represent their classes, in order of preference.
    fn retype_actual(&self, declared: &T, actual: &T, representatives: &[T::Identity]) -> Result<T, TypeError>;
}

impl<T: Type> TypeRefinements<T> for () {
    #[inline]
    fn establish<D: IntoIterator, A: IntoIterator>(declared: D, actual: A) -> Result<Self, TypeError>
    where
        D::IntoIter: ExactSizeIterator,
        A::IntoIter: ExactSizeIterator,
        D::Item: Borrow<T>,
        A::Item: Borrow<T>,
    {
        Self::validate(&(), declared, actual, &[])
    }

    #[inline]
    fn establish_scoped<D: IntoIterator, A: IntoIterator>(declared: D, actual: A) -> Result<Self, TypeError>
    where
        D::IntoIter: ExactSizeIterator,
        A::IntoIter: ExactSizeIterator,
        D::Item: Borrow<T>,
        A::Item: Borrow<T>,
    {
        // Without identities, there is nothing to scope.
        Self::establish(declared, actual)
    }

    fn validate<D: IntoIterator, A: IntoIterator>(
        &self,
        declared: D,
        actual: A,
        _closed_identities: &[T::Identity],
    ) -> Result<(), TypeError>
    where
        D::IntoIter: ExactSizeIterator,
        A::IntoIter: ExactSizeIterator,
        D::Item: Borrow<T>,
        A::Item: Borrow<T>,
    {
        visit_type_signature_pairs(declared, actual, |declared, actual| {
            if !declared.is_refined_by(actual) {
                return Err(TypeError::invalid(format!("type {actual} does not refine declared type {declared}")));
            }
            Ok(())
        })
    }

    #[inline]
    fn identity_classes<'o>(&'o self) -> impl Iterator<Item = TypeIdentityClass<'o, T::Identity>>
    where
        T::Identity: 'o,
    {
        std::iter::empty()
    }

    #[inline]
    fn refine_declared(&self, declared: &T, _symbolic: &[T::Identity]) -> Result<T, TypeError> {
        Ok(declared.clone())
    }

    #[inline]
    fn refine_actual(&self, actual: &T, _symbolic: &[T::Identity]) -> Result<T, TypeError> {
        Ok(actual.clone())
    }

    #[inline]
    fn retype_actual(&self, _declared: &T, actual: &T, _representatives: &[T::Identity]) -> Result<T, TypeError> {
        Ok(actual.clone())
    }
}

/// Associates a runtime value with the abstract [`Type`] that Ryft should use to reason about it. [`Typed`] is the
/// value-level counterpart to [`Type`]. While [`Type`] models relationships between abstract types, [`Typed`] lets a
/// concrete value produce the type that should represent it during tracing, staging, type checking, and other forms of
/// abstract reasoning.
pub trait Typed {
    /// [`Type`] family this value is typed against (e.g., [`DataType`](crate::DataType) for scalars,
    /// [`ArrayType`](crate::ArrayType) for arrays, etc.).
    type Type: Type;

    /// Returns the [`Type`] description of this value. The returned [`Type`] should capture the structural information
    /// that Ryft needs to reason about the value without having to inspect its contents. Note that returning a [`Cow`]
    /// lets implementors lend out a stored [`Type`] by borrow when one is cached on the value, while still supporting
    /// values that compute their [`Type`] on the fly (and return [`Cow::Owned`]). Callers that need ownership can call
    /// [`Cow::into_owned`] to clone on demand.
    fn r#type(&self) -> Cow<'_, Self::Type>;
}

impl<V: Typed + ?Sized> Typed for &V {
    type Type = V::Type;

    #[inline]
    fn r#type(&self) -> Cow<'_, Self::Type> {
        (*self).r#type()
    }
}

/// Returns the declared type identities whose relationships are not proven by the supplied input types. For example,
/// `[f32[n], f32[n]]` declares that two arrays have the same length. Actual types `[f32[3], f32[3]]` prove that
/// relationship, while `[f32[n], f32[3]]` leave the first array's length unknown. A transform that removes the
/// boundary must preserve its runtime check, either by retaining the boundary or by emitting a
/// [`GuardOperation`](crate::GuardOperation).
///
/// A relationship is proven when one consistent structural substitution, as determined by
/// [`Type::derive_identity_renaming`], instantiates every occurrence of its identity. `actual` must align with
/// `declared` and use [`Some`] with a borrowed type for available types, including symbolic types, and [`None`] for
/// unavailable types. An unavailable type leaves every identity carried by its corresponding declared type unproven.
///
/// The result contains each unproven identity once, in the order it first appears in `declared`. The analysis is
/// _conservative_ meaning that a member carrying several identities can cause one to be reported because another is
/// unproven. This function neither validates input admissibility nor performs runtime checks. In particular, an empty
/// result does not establish that the actual types satisfy the boundary's complete type contract.
///
/// # Errors
///
/// Returns [`TypeError::Invalid`] when the signatures have different lengths. Other structural comparison failures
/// leave the affected identities unproven instead of returning an error.
///
/// # Examples
///
/// ```rust
/// # use ryft_core::{ArrayType, DataType, DimensionBounds, DimensionVariable, Shape, unproven_type_identities};
///
/// let length = DimensionVariable::new("length", DimensionBounds::positive(Some(8)).unwrap());
/// let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![length.clone().into()]));
/// let exact = ArrayType::new_static(DataType::F32, [3]);
/// let declared = [symbolic.clone(), symbolic.clone()];
///
/// // The first input may have a different length, so removing the boundary would lose a runtime check.
/// let mixed = [Some(&symbolic), Some(&exact)];
/// assert_eq!(unproven_type_identities(&declared, &mixed).unwrap(), vec![length]);
///
/// // Both types now prove length three, so this relationship needs no runtime boundary check.
/// let proven = [Some(&exact), Some(&exact)];
/// assert!(unproven_type_identities(&declared, &proven).unwrap().is_empty());
/// ```
pub fn unproven_type_identities<T: Type>(declared: &[T], actual: &[Option<&T>]) -> Result<Vec<T::Identity>, TypeError> {
    if declared.len() != actual.len() {
        return Err(TypeError::invalid(format!(
            "declared type count {} does not match actual type count {}",
            declared.len(),
            actual.len(),
        )));
    }

    if declared.iter().all(|r#type| r#type.identities().next().is_none()) {
        return Ok(Vec::new());
    }

    // A successful complete match proves every identity without constructing per-identity member groups.
    let is_complete_match = actual
        .iter()
        .copied()
        .collect::<Option<Vec<_>>>()
        .map(|actual| T::derive_identity_renaming(declared, actual).is_ok());
    if is_complete_match == Some(true) {
        return Ok(Vec::new());
    }

    // Record each identity's member positions once, preserving declaration order and repeated axes within a member.
    let mut identities = Vec::<(&T::Identity, Vec<usize>)>::new();
    for (index, r#type) in declared.iter().enumerate() {
        for (_, identity) in r#type.identities() {
            match identities.iter_mut().find(|(carried, _)| *carried == identity) {
                Some((_, members)) => {
                    if members.last() != Some(&index) {
                        members.push(index);
                    }
                }
                None => identities.push((identity, vec![index])),
            }
        }
    }

    // Identities carried by the same members have the same conservative result. Compare those members only once.
    let mut proven_members = HashMap::new();
    Ok(identities
        .into_iter()
        .filter_map(|(identity, members)| {
            let proven = *proven_members.entry(members).or_insert_with_key(|members| {
                // This group is the complete signature, so its failed comparison has already been performed.
                if is_complete_match == Some(false) && members.len() == declared.len() {
                    return false;
                }
                members.iter().all(|&index| actual[index].is_some())
                    && T::derive_identity_renaming(
                        members.iter().map(|&index| &declared[index]),
                        members.iter().map(|&index| actual[index].unwrap()),
                    )
                    .is_ok()
            });
            (!proven).then(|| identity.clone())
        })
        .collect())
}

/// Zips one complete declared/actual [`Type`] signature and applies `visitor` to each corresponding pair, rejecting
/// signatures of different lengths with the canonical count diagnostic. This is the shared driver behind complete
/// signature walks such as [`TypeRefinements::establish`], [`TypeRefinements::validate`], and
/// [`Type::derive_identity_renaming`], so that every such walk agrees on the length contract and its diagnostic.
pub(crate) fn visit_type_signature_pairs<
    T: Type,
    D: IntoIterator<Item: Borrow<T>, IntoIter: ExactSizeIterator>,
    A: IntoIterator<Item: Borrow<T>, IntoIter: ExactSizeIterator>,
>(
    declared: D,
    actual: A,
    mut visitor: impl FnMut(&T, &T) -> Result<(), TypeError>,
) -> Result<(), TypeError> {
    let declared = declared.into_iter();
    let actual = actual.into_iter();
    if declared.len() != actual.len() {
        return Err(TypeError::invalid(format!(
            "declared type count {} does not match actual type count {}",
            declared.len(),
            actual.len(),
        )));
    }
    declared.zip(actual).try_for_each(|(declared, actual)| visitor(declared.borrow(), actual.borrow()))
}

#[allow(clippy::let_unit_value)]
#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::arrays::{ArrayType, DataType, Dimension, DimensionBounds, DimensionVariable, Shape};

    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
    #[error("custom type error {code}")]
    struct CustomTypeError {
        code: usize,
    }

    #[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
    #[error("other custom type error")]
    struct OtherCustomTypeError;

    #[test]
    fn test_type_error() {
        let invalid = TypeError::invalid("invalid type");
        assert_eq!(invalid.to_string(), "invalid type");
        assert_eq!(invalid.downcast_custom::<CustomTypeError>(), None);

        let custom = TypeError::custom(CustomTypeError { code: 1 });
        let equal = TypeError::custom(CustomTypeError { code: 1 });
        assert_eq!(custom, equal);
        assert_ne!(custom, TypeError::custom(CustomTypeError { code: 2 }));
        assert_ne!(custom, TypeError::custom(OtherCustomTypeError));

        let cloned = custom.clone();
        assert_eq!(cloned.downcast_custom::<CustomTypeError>(), Some(&CustomTypeError { code: 1 }));
        assert_eq!(cloned.downcast_custom::<OtherCustomTypeError>(), None);

        let mut errors = HashSet::new();
        errors.insert(custom);
        assert!(errors.contains(&equal));
    }

    #[test]
    fn test_type_identity_class() {
        let declared = [DimensionVariable::new("declared", DimensionBounds::non_negative(Some(8)).unwrap())];
        let actual = [DimensionVariable::new("actual", DimensionBounds::non_negative(Some(8)).unwrap())];
        let class = TypeIdentityClass::new(&declared, &actual, true);
        assert_eq!(class.declared(), &declared);
        assert_eq!(class.actual(), &actual);
        assert!(class.is_exact());
        assert!(!TypeIdentityClass::new(&declared, &[], false).is_exact());
        assert_eq!(<() as TypeRefinements<DataType>>::identity_classes(&()).count(), 0);
    }

    #[test]
    fn test_type_refinements() {
        let refinements = <() as TypeRefinements<DataType>>::establish(
            [DataType::F32, DataType::I32].iter(),
            [DataType::F32, DataType::I32].iter(),
        )
        .unwrap();
        assert_eq!(
            <() as TypeRefinements<DataType>>::validate(
                &refinements,
                [DataType::F32, DataType::I32].iter(),
                [DataType::F32, DataType::I32].iter(),
                &[],
            ),
            Ok(()),
        );
        assert_eq!(
            <() as TypeRefinements<DataType>>::establish([DataType::F32].iter(), [DataType::F32, DataType::I32].iter(),),
            Err(TypeError::invalid("declared type count 1 does not match actual type count 2")),
        );
        assert_eq!(
            <() as TypeRefinements<DataType>>::validate(
                &refinements,
                [DataType::F32].iter(),
                [DataType::F64].iter(),
                &[],
            ),
            Err(TypeError::invalid("type f64 does not refine declared type f32")),
        );
        assert_eq!(
            <() as TypeRefinements<DataType>>::refine_declared(&refinements, &DataType::F32, &[]),
            Ok(DataType::F32)
        );
    }

    #[test]
    fn test_typed_reference_delegates_to_referent() {
        let value = ArrayType::scalar(DataType::F32);
        let reference = &value;
        assert_eq!(reference.r#type(), value.r#type());
    }

    #[test]
    fn test_unproven_type_identities() {
        let vector = |dimension: Dimension| ArrayType::new(DataType::F32, Shape::new(vec![dimension]));
        let n = DimensionVariable::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let m = DimensionVariable::new("m", DimensionBounds::positive(Some(8)).unwrap());
        let p = DimensionVariable::new("p", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = |variable: &DimensionVariable| vector(Dimension::Dynamic(variable.clone()));
        let exact = vector(Dimension::Static(3));
        let declared = [symbolic(&n), symbolic(&n), symbolic(&p)];
        let actual_n = symbolic(&n);
        let actual_m = symbolic(&m);
        let actual_p = symbolic(&p);

        // One structural substitution proves `n`, whether it renames `n` or fixes it to a static extent.
        let renamed = [Some(&actual_m), Some(&actual_m), Some(&actual_p)];
        assert_eq!(unproven_type_identities(&declared, &renamed), Ok(Vec::<DimensionVariable>::new()));
        let fixed = [Some(&exact), Some(&exact), Some(&actual_p)];
        assert_eq!(unproven_type_identities(&declared, &fixed), Ok(Vec::<DimensionVariable>::new()));

        // Mixed or conflicting occurrences, and unknown entries, leave `n` to the boundary, while `p` stays proven.
        let mixed = [Some(&actual_n), Some(&exact), Some(&actual_p)];
        assert_eq!(unproven_type_identities(&declared, &mixed), Ok(vec![n.clone()]));
        let conflicting = [Some(&actual_m), Some(&actual_n), Some(&actual_p)];
        assert_eq!(unproven_type_identities(&declared, &conflicting), Ok(vec![n.clone()]));
        let unknown = [Some(&actual_n), None, Some(&actual_p)];
        assert_eq!(unproven_type_identities(&declared, &unknown), Ok(vec![n.clone()]));

        // Unavailable types preserve the distinct identities' declaration order; empty signatures have none.
        assert_eq!(unproven_type_identities(&declared, &[None, None, None]), Ok(vec![n, p]));
        assert_eq!(unproven_type_identities::<ArrayType>(&[], &[]), Ok(Vec::new()));

        // This query does not validate scalar admissibility, even when it finds no identity relationship to preserve.
        assert_eq!(unproven_type_identities(&[DataType::F32], &[Some(&DataType::I32)]), Ok(Vec::new()));

        // Reject both missing and extra actual entries instead of silently truncating the signature walk.
        assert_eq!(
            unproven_type_identities(&declared, &renamed[..2]),
            Err(TypeError::invalid("declared type count 3 does not match actual type count 2")),
        );
        assert_eq!(
            unproven_type_identities(&declared, &[None, None, None, None]),
            Err(TypeError::invalid("declared type count 3 does not match actual type count 4")),
        );
    }
}
