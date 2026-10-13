use std::borrow::{Borrow, Cow};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::sync::Arc;

use thiserror::Error;

use crate::errors::CustomError;
use crate::parameters::Parameter;
use crate::programs::identities::{TypeIdentity, TypeIdentityClass, TypeIdentityPosition, TypeIdentityRenaming};

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
    /// The default implementation requires each actual type to refine its declared type and preserve all identity
    /// occurrences unchanged. It therefore returns an empty renaming. Families supporting identity substitutions
    /// override this method and verify that those substitutions preserve the complete signature's relationships without
    /// requiring additional runtime checks.
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
        // The default supports unchanged identities. Substitutions require a family-specific structural rule;
        // conditional establishment cannot prove that metadata may be rewritten without a runtime check.
        visit_type_signature_pairs::<Self, _, _>(declared, actual, |declared, actual| {
            if !declared.is_refined_by(actual) {
                return Err(TypeError::invalid(format!("type {actual} does not refine declared type {declared}")));
            }
            if !declared.identities().eq(actual.identities()) {
                return Err(TypeError::invalid(format!(
                    "type {actual} cannot structurally instantiate declared type {declared}",
                )));
            }
            Ok(())
        })?;
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

    /// Returns `true` if every value described by `other` is also described by this [`Type`]. The receiver is the more
    /// general type (e.g., a declared or staged type) and the argument is the more precise one (e.g., the actual type
    /// carried by a runtime value), and so the relation is directional: `declared.is_refined_by(&actual)`. Note that
    /// this is a proof about the types alone. [`TypeRefinements::new`] can also accept input types whose
    /// shared identities require checking at runtime. Such input types need not satisfy this predicate.
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

/// Facts that one invocation establishes by validating a complete type signature at a [`Program`](crate::Program) or
/// [`Region`](crate::Region) boundary, ensuring that repeated identity occurrences remain consistent across boundary
/// leaves and outputs.
///
/// A _guard_, represented by [`GuardOperation`](crate::GuardOperation), checks required relationships between its
/// inputs' type identities at runtime. It preserves these checks when a transform removes the program or region
/// invocation that would otherwise perform them.
///
/// Use [`Self::new`] to construct refinement facts for program and region inputs under ordinary input-admission rules.
/// Use [`Self::from_local_declarations`] when declared identities are local names for the comparison, such as a guard's
/// own signature. Both constructors return facts about the complete invocation, keeping identities repeated within a
/// type or across several inputs consistent. Their facts live in two scopes that are never merged, even when the same
/// identity object occurs on both sides: _declared_ facts about the declared [`TypeIdentity`]s, which
/// [`Self::refine_declared`] applies to declared, intermediate, and output types and against which
/// [`Self::validate_refinement`] checks outputs, and _actual_ facts about the identities of the supplied types,
/// which [`Self::refine_actual`] applies to those supplied input types. [`TypeIdentity`]s produced inside
/// [`Program`](crate::Program)s or [`Region`](crate::Region)s may establish additional output facts, while
/// unrelated, unbound output identities are rejected.
///
/// The facts hold for one invocation only, provided its input relationships are satisfied. They never mutate a caller
/// identity or replace runtime validation: relationships that the actual types leave unproven must still be checked by
/// the program or region, or by a [`GuardOperation`](crate::GuardOperation).
///
/// # Examples
///
/// These examples use consistent type signatures and identity names. All symbolic identities have bounds `[1, 8)`.
/// The abbreviated calls use type notation rather than Rust syntax. A complete runnable example follows.
///
/// ## Constructing Refinement Facts
///
/// Both constructors take corresponding declared and actual input types. They differ in how they interpret declared
/// identities: [`Self::new`] enforces the type family's ordinary input rules. [`Self::from_local_declarations`] permits
/// compatible actual identities to instantiate different declared names that are local to the comparison. Both reject
/// incompatible structure, bounds, and conflicting facts. Neither performs runtime checks.
///
/// For arrays with the bounds used here, `new` rejects declared `f32[n]` supplied as `f32[m]`, because the symbolic
/// axes have different nominal identities. `from_local_declarations` accepts that comparison and relates declared `n`
/// to actual `m`. The examples below use `new` for ordinary inputs and `from_local_declarations` for a guard's types.
///
/// ## Validating Supplied Types
///
/// Consider the following type signatures:
///
/// ```text
/// declared = [dimension<n>, f32[n]]
/// actual   = [dimension<m>, f32[2]]
/// ```
///
/// The dimension input relates the declared `n` to the supplied `m` and the array input supplies extent `2`. The
/// resulting class contains declared `[n]` and actual `[m]`, with conditional facts `n = 2` and `m = 2`. The types
/// alone do not prove that the supplied dimension has extent `2`, so the invocation must still check that relationship
/// at runtime. In the runnable example below, [`Self::new`] returns an object containing these facts, assigned
/// to a variable named `refinements`.
///
/// Calling [`Self::validate_refinement`] on that object checks output types against the stored input facts: a declared
/// output `f32[n]` accepts `f32[2]` and rejects `f32[3]`. Calling [`Self::new`] again with declared `f32[n]` and actual
/// `f32[3]` would instead create a new object with the fact `n = 3`. That separate comparison would not check
/// consistency with the original input fact `n = 2`.
///
/// Repeated identities must also agree within and across inputs. Declared `[f32[n], f32[n]]` rejects supplied
/// `[f32[2], f32[3]]`, and declared `f32[n, n]` rejects supplied `f32[2, 3]`. Supplied `f32[n, 3]` is accepted
/// for the latter declared type, conditional on checking that its symbolic dimension has extent 3 at runtime.
///
/// ## Applying Facts and Selecting Identities
///
/// To see how several symbolic inputs interact, establish a guard's local comparison:
///
/// ```text
/// local_declared = [f32[n], f32[n], f32[n]]
/// local_actual   = [f32[m], f32[k], f32[2]]
/// ```
/// All three declared arrays use `n`, so their supplied lengths must agree. The first pair relates declared `n` to
/// actual `m`, the second relates the same declared `n` to actual `k`, and the third supplies extent `2`. In the
/// runnable example below, [`Self::from_local_declarations`] compares the signatures and returns an object named
/// `guard_refinements`. It records one class with declared `[n]`, actual `[m, k]`, and conditional extent `2`. This
/// extent comes from the third input, not from a later call's `preserved_identities` or `preferred_identities`
/// argument. Runtime checks must establish that the arrays carrying `m` and `k` also have length `2`.
///
/// The `preserved_identities` argument to [`Self::refine_declared`] and [`Self::refine_actual`] lists existing names
/// to keep in the returned type, even when their exact extents are known. Preserving `m` does not preserve every
/// related identity: `k` still becomes `2` unless it is also listed. The `preferred_identities` argument to
/// [`Self::retype_actual`] selects which related actual identities to use in the returned type; it can
/// replace `k` with `m`.
///
/// The example also calls [`Self::from_local_declarations`] without the third input pair and names the returned object
/// `without_exact_extent`. This object relates `n`, `m`, and `k` but does not determine their exact extent. Refinement
/// then leaves `k` unchanged, while representation selection can still replace it with `m`, provided the two supplied
/// arrays have equal lengths. Ordinary input validation with [`Self::new`] rejects these signatures
/// because their symbolic array axes have different nominal identities.
///
/// The following calls use the facts stored in these objects. An empty `preserved_identities` list preserves no names;
/// a non-empty list preserves only the existing names it contains. `preferred_identities` instead selects compatible
/// actual identities from the corresponding declared class. An actual type already carrying a compatible preferred
/// identity keeps its own name, even if another requested member comes first.
///
/// |                                       Call                                       |  Result  |
/// | -------------------------------------------------------------------------------- | -------- |
/// | `guard_refinements.refine_declared(f32[n], preserved_identities = [])`           | `f32[2]` |
/// | `guard_refinements.refine_declared(f32[n], preserved_identities = [n])`          | `f32[n]` |
/// | `guard_refinements.refine_actual(f32[m], preserved_identities = [m])`            | `f32[m]` |
/// | `guard_refinements.refine_actual(f32[k], preserved_identities = [m])`            | `f32[2]` |
/// | `guard_refinements.refine_actual(f32[k], preserved_identities = [m, k])`         | `f32[k]` |
/// | `guard_refinements.refine_actual(f32[2], preserved_identities = [m])`            | `f32[2]` |
/// | `guard_refinements.retype_actual(f32[n], f32[k], preferred_identities = [m])`    | `f32[m]` |
/// | `guard_refinements.retype_actual(f32[n], f32[2], preferred_identities = [m])`    | `f32[m]` |
/// | `guard_refinements.retype_actual(f32[n], f32[k], preferred_identities = [m, k])` | `f32[k]` |
/// | `without_exact_extent.refine_declared(f32[n], preserved_identities = [])`        | `f32[n]` |
/// | `without_exact_extent.refine_actual(f32[k], preserved_identities = [m])`         | `f32[k]` |
/// | `without_exact_extent.retype_actual(f32[n], f32[k], preferred_identities = [m])` | `f32[m]` |
///
/// Refinement never renames `n` or `k` to `m`, or introduces `m` into a static type. Representation selection can
/// do that when the identities belong to the same class. Equal numeric extents in unrelated classes do not justify
/// choosing the same identity. Selecting `m` is useful when a related value, such as a reference, must keep that name
/// in its type. The resulting representations are justified only when the corresponding runtime checks succeed.
///
/// ## Identity Classes and Runtime Checks
///
/// [`Self::identity_classes`] returns the groups of related declared and actual identities.
/// [`TypeIdentityClass::is_exact`] reports whether the stored facts determine an exact extent for a group, assuming
/// the input relationships hold. [`Self::identities_with_runtime_checks`] reports which relationships still
/// require runtime checks, whether or not an exact extent is known:
///
/// |   Refinement Object    | Declared Members | Actual Members | Exact Extent | Required Check |
/// | ---------------------- | ---------------- | -------------- | ------------ | -------------- |
/// | `refinements`          | `[n]`            | `[m]`          | 2            | `m = 2`        |
/// | `guard_refinements`    | `[n]`            | `[m, k]`       | 2            | `m = k = 2`    |
/// | `without_exact_extent` | `[n]`            | `[m, k]`       | Unknown      | `m = k`        |
///
/// All three objects report `[n]` as requiring runtime checks. In the first comparison, supplying `dimension<2>`
/// instead of `dimension<m>` makes the complete supplied signature prove the relationship, so no check is needed.
///
/// ## Runnable Example
///
/// ```rust
/// # use ryft_core::{
/// #     ArrayIrType, ArrayIrTypeRefinements, ArrayType, DataType, Dimension, DimensionBounds, DimensionType,
/// #     DimensionVariable, Shape, TypeRefinements,
/// # };
///
/// // Shared identities and types: `n`, `m`, and `k` all have bounds `[1, 8)`.
/// let bounds = DimensionBounds::positive(Some(8)).unwrap();
/// let n = DimensionVariable::new("n", bounds);
/// let m = DimensionVariable::new("m", bounds);
/// let k = DimensionVariable::new("k", bounds);
///
/// let dimension_n: ArrayIrType = DimensionType::from(n.clone()).into();
/// let dimension_m: ArrayIrType = DimensionType::from(m.clone()).into();
/// let dimension_two: ArrayIrType = DimensionType::from(Dimension::Static(2)).into();
/// let array = |dimension: Dimension| {
///     ArrayIrType::from(ArrayType::new(DataType::F32, Shape::new(vec![dimension])))
/// };
/// let array_n = array(n.clone().into());
/// let array_m = array(m.clone().into());
/// let array_k = array(k.clone().into());
/// let array_two = array(2.into());
/// let array_three = array(3.into());
///
/// // Ordinary inputs: `[dimension<n>, f32[n]]` against `[dimension<m>, f32[2]]`.
/// let declared = [dimension_n, array_n.clone()];
/// let actual = [dimension_m.clone(), array_two.clone()];
/// let refinements = ArrayIrTypeRefinements::new(&declared, &actual).unwrap();
///
/// // Apply the stored facts `n = 2` and `m = 2`, or keep the listed identities symbolic.
/// let keep_n = std::slice::from_ref(&n);
/// let keep_m = std::slice::from_ref(&m);
/// assert_eq!(refinements.refine_declared(&array_n, &[]).unwrap(), array_two);
/// assert_eq!(refinements.refine_actual(&dimension_m, &[]).unwrap(), dimension_two);
/// assert_eq!(refinements.refine_declared(&array_n, keep_n).unwrap(), array_n);
/// assert_eq!(refinements.refine_actual(&dimension_m, keep_m).unwrap(), dimension_m);
///
/// // Keeping m symbolic does not introduce it into `f32[2]`; choosing m as the preferred identity does.
/// let preferred_identities = keep_m;
/// assert_eq!(refinements.refine_actual(&array_two, keep_m).unwrap(), array_two);
/// assert_eq!(refinements.retype_actual(&array_n, &array_two, preferred_identities).unwrap(), array_m);
/// assert_eq!(refinements.retype_actual(&array_n, &array_two, &[]).unwrap(), array_two);
///
/// // These exact facts are conditional: the dimension input still needs a runtime check.
/// let classes = refinements.identity_classes().collect::<Vec<_>>();
/// assert_eq!(classes.len(), 1);
/// assert_eq!(classes[0].declared(), keep_n);
/// assert_eq!(classes[0].actual(), keep_m);
/// assert!(classes[0].is_exact());
/// assert_eq!(refinements.identities_with_runtime_checks(), keep_n);
///
/// // Output validation uses the existing `n = 2` fact. A new comparison can instead establish `n = 3`.
/// assert!(refinements.validate_refinement([&array_n], [&array_two], &[]).is_ok());
/// assert!(refinements.validate_refinement([&array_n], [&array_three], &[]).is_err());
/// assert!(ArrayIrTypeRefinements::new([&array_n], [&array_three]).is_ok());
///
/// // Fully exact inputs prove the relationship, so no runtime check is required.
/// let exact_actual = [dimension_two, array_two.clone()];
/// let exact = ArrayIrTypeRefinements::new(&declared, &exact_actual).unwrap();
/// assert!(exact.identities_with_runtime_checks().is_empty());
///
/// // Repeating `n` within `f32[n, n]` also requires consistent extents.
/// let repeated_n: ArrayIrType =
///     ArrayType::new(DataType::F32, Shape::new(vec![n.clone().into(), n.clone().into()])).into();
/// let mismatched_extents: ArrayIrType = ArrayType::new_static(DataType::F32, [2, 3]).into();
/// let partially_symbolic: ArrayIrType =
///     ArrayType::new(DataType::F32, Shape::new(vec![n.clone().into(), 3.into()])).into();
/// assert!(ArrayIrTypeRefinements::new([&repeated_n], [&mismatched_extents]).is_err());
/// let repeated_refinements = ArrayIrTypeRefinements::new([&repeated_n], [&partially_symbolic]).unwrap();
/// assert_eq!(repeated_refinements.identities_with_runtime_checks(), keep_n);
///
/// // A guard's local declarations: `[f32[n], f32[n], f32[n]]` against `[f32[m], f32[k], f32[2]]`.
/// let local_declared = [array_n.clone(), array_n.clone(), array_n.clone()];
/// let local_actual = [array_m.clone(), array_k.clone(), array_two.clone()];
/// assert!(ArrayIrTypeRefinements::new(&local_declared, &local_actual).is_err());
/// assert!(ArrayIrTypeRefinements::new(&local_declared[..2], &[array_two.clone(), array_three]).is_err());
/// let guard_refinements = ArrayIrTypeRefinements::from_local_declarations(&local_declared, &local_actual).unwrap();
///
/// // Keeping `m` symbolic leaves `m` unchanged but still lets `k` refine to `2`.
/// let keep_m_and_k = [m.clone(), k.clone()];
/// assert_eq!(guard_refinements.refine_declared(&array_n, &[]).unwrap(), array_two);
/// assert_eq!(guard_refinements.refine_declared(&array_n, keep_n).unwrap(), array_n);
/// assert_eq!(guard_refinements.refine_actual(&array_m, keep_m).unwrap(), array_m);
/// assert_eq!(guard_refinements.refine_actual(&array_k, keep_m).unwrap(), array_two);
/// assert_eq!(guard_refinements.refine_actual(&array_k, &keep_m_and_k).unwrap(), array_k);
/// assert_eq!(guard_refinements.refine_actual(&array_two, keep_m).unwrap(), array_two);
///
/// // Choosing `m` as the preferred identity can rename `k` or introduce `m` into the static array type.
/// assert_eq!(guard_refinements.retype_actual(&array_n, &array_k, preferred_identities).unwrap(), array_m);
/// assert_eq!(guard_refinements.retype_actual(&array_n, &array_two, preferred_identities).unwrap(), array_m);
///
/// // If the actual type already carries a compatible preferred identity, it keeps its own name.
/// assert_eq!(guard_refinements.retype_actual(&array_n, &array_k, &keep_m_and_k).unwrap(), array_k);
///
/// let classes = guard_refinements.identity_classes().collect::<Vec<_>>();
/// assert_eq!(classes.len(), 1);
/// assert_eq!(classes[0].declared(), keep_n);
/// assert_eq!(classes[0].actual(), &keep_m_and_k);
/// assert!(classes[0].is_exact());
/// assert_eq!(guard_refinements.identities_with_runtime_checks(), keep_n);
///
/// // Removing the third pair leaves `n`, `m`, and `k` related, without an exact extent.
/// let without_exact_extent =
///     ArrayIrTypeRefinements::from_local_declarations(&local_declared[..2], &local_actual[..2]).unwrap();
/// assert_eq!(without_exact_extent.refine_declared(&array_n, &[]).unwrap(), array_n);
/// assert_eq!(without_exact_extent.refine_actual(&array_k, keep_m).unwrap(), array_k);
///
/// // The relationship alone still permits choosing `m` as the preferred identity, provided `m = k` at runtime.
/// assert_eq!(without_exact_extent.retype_actual(&array_n, &array_k, preferred_identities).unwrap(), array_m);
/// assert!(!without_exact_extent.identity_classes().next().unwrap().is_exact());
/// assert_eq!(without_exact_extent.identities_with_runtime_checks(), keep_n);
/// ```
///
/// These functions inspect types only. Their conditional results do not replace the required runtime input checks.
pub trait TypeRefinements<T: Type>: Clone + Debug + Default {
    /// Creates a new [`TypeRefinements`] instance by validating the complete `actual` input signature against
    /// `declared` under the type family's ordinary input-admission rules. Use this constructor for program and region
    /// inputs. This checks ordinary input admission and every repeated identity relationship, including repetitions
    /// within one type. Relationships not guaranteed by the supplied types must be checked at runtime. Such obligations
    /// are recorded in [`Self::identities_with_runtime_checks`]; success does not mean that a guard can be removed.
    ///
    /// Unlike [`Self::from_local_declarations`], which treats declared identities as local names, it enforces
    /// ordinary nominal input rules. The default requires each actual type to satisfy [`Type::is_refined_by`]
    /// before establishing relationships. Families that accept inputs requiring runtime checks or equivalent exact
    /// representations override this function, keeping complete-signature consistency and guard obligations in the
    /// returned facts. Neither input validation nor establishment converts values or performs runtime checks.
    ///
    /// Refer to the [examples on `TypeRefinements`](TypeRefinements#examples) for more information.
    ///
    /// # Parameters
    ///
    ///   - `declared`: Complete input type signature declared by the program or region.
    ///   - `actual`: Corresponding supplied input types.
    fn new<D: IntoIterator, A: IntoIterator>(declared: D, actual: A) -> Result<Self, TypeError>
    where
        D::IntoIter: ExactSizeIterator,
        A::IntoIter: ExactSizeIterator,
        D::Item: Borrow<T>,
        A::Item: Borrow<T>,
    {
        let declared = declared.into_iter().collect::<Vec<_>>();
        let actual = actual.into_iter().collect::<Vec<_>>();
        visit_type_signature_pairs::<T, _, _>(
            declared.iter().map(Borrow::borrow),
            actual.iter().map(Borrow::borrow),
            |declared, actual| {
                if !declared.is_refined_by(actual) {
                    return Err(TypeError::invalid(format!("type {actual} does not refine declared type {declared}")));
                }
                Ok(())
            },
        )?;
        Self::from_local_declarations(declared, actual)
    }

    /// Creates a new [`TypeRefinements`] instance by comparing the complete `actual` signature with `declared`,
    /// treating declared identities as local names for this comparison. Bounds-compatible actual identities may
    /// instantiate different declared identities. Programs and regions construct their input refinements with
    /// [`Self::new`]; guards and code replaying a body with prepared inputs call this function directly. Identities
    /// connected through matched input pairs form a class, which may contain several identities in either scope. An
    /// exact fact observed in any pair applies to every identity in its class. A fact outside an identity's bounds,
    /// or inconsistent with another fact about the same identity, is rejected. Correspondences without an exact fact
    /// do not determine an exact value. Relationships that the supplied types do not prove are recorded by
    /// [`Self::identities_with_runtime_checks`]. They require the original program's or region's input checks,
    /// or a guard, before the facts may be used to execute a specialized body. This constructor never performs
    /// a runtime check.
    ///
    /// Refer to the [examples on `TypeRefinements`](TypeRefinements#examples) for more information.
    ///
    /// # Parameters
    ///
    ///   - `declared`: Complete declared type signature whose identities are local to this comparison.
    ///   - `actual`: Corresponding supplied types to compare with those declarations.
    fn from_local_declarations<D: IntoIterator, A: IntoIterator>(declared: D, actual: A) -> Result<Self, TypeError>
    where
        D::IntoIter: ExactSizeIterator,
        A::IntoIter: ExactSizeIterator,
        D::Item: Borrow<T>,
        A::Item: Borrow<T>;

    /// Returns the invocation's groups of related declared and actual identities, in first-observation order. Every
    /// observed identity belongs to one class in its own scope, including isolated observations and exact facts. The
    /// same identity object is tracked separately in the declared and actual scopes and related only through matched
    /// input types. Equal exact refinements do not connect otherwise unrelated classes. The classes describe
    /// relationships between the supplied inputs, conditional on those relationships being satisfied. Using
    /// the classes never replaces the runtime checks that the inputs require.
    ///
    /// Refer to the [examples on `TypeRefinements`](TypeRefinements#examples) for more information.
    fn identity_classes<'o>(&'o self) -> impl Iterator<Item = TypeIdentityClass<'o, T::Identity>>
    where
        T::Identity: 'o;

    /// Returns the declared type identities whose equality relationships are not guaranteed by the supplied input
    /// types, in first-occurrence order. Those relationships still require runtime checking. These obligations are
    /// independent of exact conditional facts. Implementations may also include identities whose relationships are
    /// already guaranteed if they occur in the same input type as an identity that still requires a runtime check.
    ///
    /// Refer to the [examples on `TypeRefinements`](TypeRefinements#examples) for more information.
    fn identities_with_runtime_checks(&self) -> &[T::Identity];

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
    /// `closed_identities` only identifies which identities may establish facts during validation.
    /// It does not contain concrete extents or inspect runtime value payloads.
    ///
    /// Refer to the [examples on `TypeRefinements`](TypeRefinements#examples) for more information.
    ///
    /// # Parameters
    ///
    ///   - `declared`: Complete declared type signature being validated, normally a program or region output signature.
    ///   - `actual`: Corresponding refining type signature supplied at the validation site.
    ///   - `closed_identities`: The validated boundary's complete closed identity set
    ///     (e.g., a [`TypeIdentitySignature`](crate::TypeIdentitySignature)'s
    ///     [`identities`](crate::TypeIdentitySignature::identities) slice), whose members are permitted to establish
    ///     refinement facts when first encountered in `actual`.
    fn validate_refinement<D: IntoIterator, A: IntoIterator>(
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

    /// Returns `r#type` with the stored declared-scope facts applied, except for identities in `preserved_identities`.
    /// The type can be a declared input, intermediate, or output type. It need not be one of the input types used to
    /// construct these facts. The argument supplies the type to refine, while `self` supplies the facts to apply.
    /// Because the facts are established from the complete signature first, a preserved identity still takes part in
    /// conflict detection and is only kept symbolic when the facts are applied. Preservation applies per identity, so
    /// a type mentioning both a preserved and a refinable identity is still refined at the latter. Refinements never
    /// add metadata that is not a fact about an identity (e.g., an array's sharding or layout).
    ///
    /// Refer to the [examples on `TypeRefinements`](TypeRefinements#examples) for more information.
    ///
    /// # Parameters
    ///
    ///   - `r#type`: Type to refine using declared-scope facts.
    ///   - `preserved_identities`: Declared identities to preserve instead of replacing them with their refinements.
    fn refine_declared(&self, r#type: &T, preserved_identities: &[T::Identity]) -> Result<T, TypeError>;

    /// Returns the supplied input `r#type` with the stored actual-scope facts applied, except for preserved identities.
    /// It preserves the identities in `preserved_identities` and the type's layout, placement, and other metadata.
    /// The argument supplies the type to refine, while `self` supplies the facts from the complete input signature.
    /// A declared fact never applies to an actual identity merely because the two share a name. Preserved identities
    /// are kept even when their exact values are known. This lets a supplied value preserve identities also carried
    /// by a reference whose type cannot change. Other identities in the same supplied type are still refined.
    ///
    /// Refer to the [examples on `TypeRefinements`](TypeRefinements#examples) for more information.
    ///
    /// # Parameters
    ///
    ///   - `r#type`: Supplied input type to refine using actual-scope facts.
    ///   - `preserved_identities`: Actual identities to preserve instead of replacing them with their refinements.
    fn refine_actual(&self, r#type: &T, preserved_identities: &[T::Identity]) -> Result<T, TypeError>;

    /// Selects a type representation for the supplied `actual` input using compatible identities from its declared
    /// classes. Unlike [`Self::refine_actual`], this can introduce a requested identity where `actual` has an exact
    /// value or a different identity. `declared` identifies the relevant class. equal extents alone do not relate
    /// otherwise independent identities. The function chooses a type, without performing a runtime check or conversion.
    ///
    /// The function applies actual-scope facts as [`Self::refine_actual`] does, preserving compatible identities in
    /// `preferred_identities`. It can also select a preferred identity from the corresponding declared class for each
    /// part of the type. For arrays these parts are shape dimensions while for a first-class dimension it is the
    /// dimension itself. Relationships may pass through other declared and actual identities in the same class. An
    /// actual type already carrying a compatible preferred identity in that class keeps its own name. Otherwise, the
    /// first compatible preferred identity is selected. This lets immutable handles preserve their different names
    /// in one checked class. Compatibility requires the declared identity's bounds to contain the chosen identity's
    /// bounds. When a class without an exact value has preferred identities but none fits a part of the declared type,
    /// that part uses its declared identity if an actual member carries it, otherwise the first compatible actual
    /// member. This keeps the body's representation within the declared identity's bounds; exact classes instead apply
    /// their exact fact. Classes without an exact extent can also select preferred identities. Consumers use this to
    /// give related values compatible representations while respecting immutable reference geometry. A changed known
    /// representation may require a residual checked value. The result is a representation justified by checking the
    /// complete input signature, rather than a runtime check performed by this type-level function.
    ///
    /// Refer to the [examples on `TypeRefinements`](TypeRefinements#examples) for more information.
    ///
    /// # Parameters
    ///
    ///   - `declared`: Declared type corresponding to this supplied input, identifying its identity classes.
    ///   - `actual`: Supplied input type to refine.
    ///   - `preferred_identities`: Actual identities to use in the returned type where compatible, in preference order.
    ///     A compatible listed identity already present in the actual type takes precedence over earlier choices.
    fn retype_actual(&self, declared: &T, actual: &T, preferred_identities: &[T::Identity]) -> Result<T, TypeError>;
}

impl<T: Type> TypeRefinements<T> for () {
    #[inline]
    fn from_local_declarations<D: IntoIterator, A: IntoIterator>(declared: D, actual: A) -> Result<Self, TypeError>
    where
        D::IntoIter: ExactSizeIterator,
        A::IntoIter: ExactSizeIterator,
        D::Item: Borrow<T>,
        A::Item: Borrow<T>,
    {
        Self::validate_refinement(&(), declared, actual, &[])
    }

    #[inline]
    fn identity_classes<'o>(&'o self) -> impl Iterator<Item = TypeIdentityClass<'o, T::Identity>>
    where
        T::Identity: 'o,
    {
        std::iter::empty()
    }

    #[inline]
    fn identities_with_runtime_checks(&self) -> &[T::Identity] {
        &[]
    }

    #[inline]
    fn validate_refinement<D: IntoIterator, A: IntoIterator>(
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
    fn refine_declared(&self, r#type: &T, _preserved_identities: &[T::Identity]) -> Result<T, TypeError> {
        Ok(r#type.clone())
    }

    #[inline]
    fn refine_actual(&self, r#type: &T, _preserved_identities: &[T::Identity]) -> Result<T, TypeError> {
        Ok(r#type.clone())
    }

    #[inline]
    fn retype_actual(&self, _declared: &T, actual: &T, _preferred_identities: &[T::Identity]) -> Result<T, TypeError> {
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

// TODO(eaplatanios): Review this.
/// Returns the declared type identities whose equality relationships are not guaranteed by the supplied input types.
/// For example, `[f32[n], f32[n]]` declares that two arrays have the same length. Actual types `[f32[3], f32[3]]`
/// prove that relationship, while `[f32[n], f32[3]]` leave the first array's length unknown. A transform removing the
/// boundary must preserve its runtime check, either by retaining the boundary or by emitting a
/// [`GuardOperation`](crate::GuardOperation).
///
/// A relationship is proven when one consistent structural substitution, as determined by
/// [`Type::derive_identity_renaming`], instantiates every occurrence of its identity. `actual` must align with
/// `declared` and use [`Some`] with a borrowed type for available types, including symbolic types, and [`None`] for
/// unavailable types. An unavailable type leaves every identity carried by its corresponding declared type unproven.
/// Declared members may be owned types or values that borrow a type, such as references or [`Cow`]s; the analysis
/// borrows their types without cloning them. An empty declared slice needs an explicit element type.
///
/// Each reported identity appears once, in the order it first occurs in `declared`. The analysis is conservative:
/// when one input type contains several identities, failure to prove one relationship can also cause another identity
/// to be reported. This function neither validates input admissibility nor performs runtime checks. In particular,
/// an empty result does not establish that the actual types satisfy the boundary's complete type contract.
///
/// # Errors
///
/// Returns [`TypeError::Invalid`] when the signatures have different lengths. Other structural comparison failures
/// leave the affected identities unproven instead of returning an error.
///
/// # Examples
///
/// ```rust
/// # use ryft_core::{
/// #     ArrayType, DataType, DimensionBounds, DimensionVariable, Shape, identities_with_runtime_checks,
/// # };
///
/// let length = DimensionVariable::new("length", DimensionBounds::positive(Some(8)).unwrap());
/// let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![length.clone().into()]));
/// let exact = ArrayType::new_static(DataType::F32, [3]);
/// let declared = [symbolic.clone(), symbolic.clone()];
///
/// // The first input may have a different length, so removing the boundary would lose a runtime check.
/// let mixed = [Some(&symbolic), Some(&exact)];
/// assert_eq!(identities_with_runtime_checks(&declared, &mixed).unwrap(), vec![length]);
///
/// // Both types now prove length three, so this relationship needs no runtime boundary check.
/// let proven = [Some(&exact), Some(&exact)];
/// assert!(identities_with_runtime_checks(&declared, &proven).unwrap().is_empty());
/// ```
pub fn identities_with_runtime_checks<T: Type, D: Borrow<T>>(
    declared: &[D],
    actual: &[Option<&T>],
) -> Result<Vec<T::Identity>, TypeError> {
    if declared.len() != actual.len() {
        return Err(TypeError::invalid(format!(
            "declared type count {} does not match actual type count {}",
            declared.len(),
            actual.len(),
        )));
    }

    if declared.iter().all(|r#type| r#type.borrow().identities().next().is_none()) {
        return Ok(Vec::new());
    }

    // A successful complete match proves every identity without constructing per-identity member groups.
    let is_complete_match = actual
        .iter()
        .copied()
        .collect::<Option<Vec<_>>>()
        .map(|actual| T::derive_identity_renaming(declared.iter().map(Borrow::borrow), actual).is_ok());
    if is_complete_match == Some(true) {
        return Ok(Vec::new());
    }

    // Record each identity's member positions once, preserving declaration order and repeated axes within a member.
    let mut identities = Vec::<(&T::Identity, Vec<usize>)>::new();
    for (index, r#type) in declared.iter().enumerate() {
        for (_, identity) in r#type.borrow().identities() {
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
                        members.iter().map(|&index| declared[index].borrow()),
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
/// signature walks such as [`TypeRefinements::from_local_declarations`], [`TypeRefinements::validate_refinement`],
/// and [`Type::derive_identity_renaming`], so that every such walk agrees on the length contract and its diagnostic.
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

    use crate::arrays::{
        ArrayType, ArrayTypeRefinements, DataType, Dimension, DimensionBounds, DimensionVariable, Shape,
    };

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
    fn test_type_refinements_new() {
        assert_eq!(<() as TypeRefinements<DataType>>::new([DataType::F32], [DataType::F32]), Ok(()));
        assert_eq!(
            <() as TypeRefinements<DataType>>::new([DataType::F32], [DataType::F32, DataType::F32]),
            Err(TypeError::invalid("declared type count 1 does not match actual type count 2")),
        );

        // Implicit promotion does not make a differently typed value a valid input to an f32 program.
        assert!(DataType::F16.is_compatible_with(&DataType::F32));
        assert_eq!(
            <() as TypeRefinements<DataType>>::new([DataType::F32], [DataType::F16]),
            Err(TypeError::invalid("type f16 does not refine declared type f32")),
        );

        // Broadcasting is likewise a conversion, not ordinary entry admission.
        let declared = ArrayType::new_static(DataType::F32, [2, 3]);
        let actual = ArrayType::new_static(DataType::F32, [1, 3]);
        assert!(actual.is_compatible_with(&declared));
        assert!(ArrayTypeRefinements::new([declared], [actual]).is_err());
    }

    #[test]
    fn test_type_refinements_from_local_declarations() {
        let refinements = <() as TypeRefinements<DataType>>::from_local_declarations(
            [DataType::F32, DataType::I32].iter(),
            [DataType::F32, DataType::I32].iter(),
        )
        .unwrap();
        assert_eq!(<() as TypeRefinements<DataType>>::identity_classes(&refinements).count(), 0);
        assert!(<() as TypeRefinements<DataType>>::identities_with_runtime_checks(&refinements).is_empty());
        assert_eq!(
            <() as TypeRefinements<DataType>>::validate_refinement(
                &refinements,
                [DataType::F32, DataType::I32].iter(),
                [DataType::F32, DataType::I32].iter(),
                &[],
            ),
            Ok(()),
        );
        assert_eq!(
            <() as TypeRefinements<DataType>>::from_local_declarations(
                [DataType::F32].iter(),
                [DataType::F32, DataType::I32].iter(),
            ),
            Err(TypeError::invalid("declared type count 1 does not match actual type count 2")),
        );
        assert_eq!(
            <() as TypeRefinements<DataType>>::validate_refinement(
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
    fn test_type_refinements_admission_and_guard_obligations() {
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let n = DimensionVariable::new("n", bounds);
        let m = DimensionVariable::new("m", bounds);
        let vector =
            |variable: &DimensionVariable| ArrayType::new(DataType::F32, Shape::new(vec![variable.clone().into()]));
        let exact = ArrayType::new_static(DataType::F32, [2]);
        let declared = [vector(&n), vector(&n)];
        let actual = [vector(&n), exact.clone()];
        let refinements = ArrayTypeRefinements::new(&declared, &actual).unwrap();

        // Exact conditional facts do not prove the relationship: the symbolic input can still have another extent.
        assert!(refinements.identity_classes().next().unwrap().is_exact());
        assert_eq!(refinements.identities_with_runtime_checks(), &[n.clone()]);
        assert_eq!(refinements.refine_declared(&declared[0], &[]).unwrap(), exact);
        assert!(ArrayType::derive_identity_renaming(&declared, &actual).is_err());

        // A local declaration can be structurally instantiated without ordinary nominal admission permitting it.
        let renamed = [vector(&m), vector(&m)];
        assert!(ArrayTypeRefinements::new(&declared, &renamed).is_err());
        let refinements = ArrayTypeRefinements::from_local_declarations(&declared, &renamed).unwrap();
        assert!(refinements.identities_with_runtime_checks().is_empty());

        // A mixed local instantiation records the guard obligation, while unrelated extents remain invalid.
        let mixed = [vector(&m), exact.clone()];
        let refinements = ArrayTypeRefinements::from_local_declarations(&declared, &mixed).unwrap();
        assert_eq!(refinements.identities_with_runtime_checks(), &[n]);
        assert!(
            ArrayTypeRefinements::from_local_declarations(
                &declared,
                &[exact, ArrayType::new_static(DataType::F32, [3])]
            )
            .is_err()
        );
    }

    #[test]
    fn test_typed_reference_delegates_to_referent() {
        let value = ArrayType::scalar(DataType::F32);
        let reference = &value;
        assert_eq!(reference.r#type(), value.r#type());
    }

    #[test]
    fn test_identities_with_runtime_checks() {
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
        assert_eq!(identities_with_runtime_checks(&declared, &renamed), Ok(Vec::<DimensionVariable>::new()));
        let borrowed = declared.each_ref();
        assert_eq!(identities_with_runtime_checks(&borrowed, &renamed), Ok(Vec::<DimensionVariable>::new()));
        assert_eq!(
            identities_with_runtime_checks::<ArrayType>(&borrowed, &[None, None, None]),
            Ok(vec![n.clone(), p.clone()]),
        );
        let fixed = [Some(&exact), Some(&exact), Some(&actual_p)];
        assert_eq!(identities_with_runtime_checks(&declared, &fixed), Ok(Vec::<DimensionVariable>::new()));

        // Mixed or conflicting occurrences, and unknown entries, leave `n` to the boundary, while `p` stays proven.
        let mixed = [Some(&actual_n), Some(&exact), Some(&actual_p)];
        assert_eq!(identities_with_runtime_checks(&declared, &mixed), Ok(vec![n.clone()]));
        let borrowed = declared.each_ref().map(Cow::Borrowed);
        assert_eq!(identities_with_runtime_checks(&borrowed, &mixed), Ok(vec![n.clone()]));
        let conflicting = [Some(&actual_m), Some(&actual_n), Some(&actual_p)];
        assert_eq!(identities_with_runtime_checks(&declared, &conflicting), Ok(vec![n.clone()]));
        let unknown = [Some(&actual_n), None, Some(&actual_p)];
        assert_eq!(identities_with_runtime_checks(&declared, &unknown), Ok(vec![n.clone()]));

        // Unavailable types preserve the distinct identities' declaration order; empty signatures have none.
        assert_eq!(identities_with_runtime_checks(&declared, &[None, None, None]), Ok(vec![n, p]));
        assert_eq!(identities_with_runtime_checks::<ArrayType>(&[] as &[ArrayType], &[]), Ok(Vec::new()));
        assert_eq!(identities_with_runtime_checks::<ArrayType>(&[] as &[&ArrayType], &[]), Ok(Vec::new()));

        // This query does not validate scalar admissibility, even when it finds no identity relationship to preserve.
        assert_eq!(identities_with_runtime_checks(&[DataType::F32], &[Some(&DataType::I32)]), Ok(Vec::new()));

        // Reject both missing and extra actual entries instead of silently truncating the signature walk.
        assert_eq!(
            identities_with_runtime_checks(&declared, &renamed[..2]),
            Err(TypeError::invalid("declared type count 3 does not match actual type count 2")),
        );
        assert_eq!(
            identities_with_runtime_checks(&declared, &[None, None, None, None]),
            Err(TypeError::invalid("declared type count 3 does not match actual type count 4")),
        );
    }
}
