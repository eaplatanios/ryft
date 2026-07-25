use std::borrow::Cow;
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

    /// Cross-occurrence facts accumulated while validating complete type signatures at [`Program`](crate::Program) and
    /// [`Region`](crate::Region) boundaries. These facts relate declared [`TypeIdentity`]s to refinements observed in
    /// corresponding actual types, ensuring that repeated identity occurrences remain consistent across boundary leaves
    /// and outputs. For example, an array refinement environment remembers the concrete extent first observed for a
    /// dynamic dimension and rejects a conflicting extent elsewhere in the same signature. Type families that need no
    /// cross-value facts use `()`. Also, type refinements never inspect or retain runtime value payloads.
    type Refinements: TypeRefinements<Self>;

    /// Visits the [`TypeIdentity`]s carried by this [`Type`] in deterministic positional order.
    #[inline]
    fn visit_identities(&self, visitor: &mut impl FnMut(TypeIdentityPosition, &Self::Identity)) {
        let _ = visitor;
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
    ///   - `declared`: Complete formal type signature declared by the program or region boundary.
    ///   - `actual`: Corresponding type signature supplied at the instantiation site.
    #[inline]
    fn derive_identity_renaming(
        declared: &[Self],
        actual: &[Self],
    ) -> Result<TypeIdentityRenaming<Self::Identity>, TypeError> {
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
    /// `other`, but it can be turned into one (e.g., by promoting its [`DataType`](crate::types::DataType) or
    /// broadcasting its [`Shape`](crate::types::Shape)), possibly losing information along the way. Contrast this with
    /// [`Self::is_refined_by`], which holds only when a value already *is* a value of the other type, with no conversion
    /// involved. For example, `f16` is compatible with (i.e., promotable to) `f32` but does not refine it, while a
    /// dynamically shaped `f32` array type of shape `[?, 3]` is refined by an `f32` array type of shape `[2, 3]` with no
    /// conversion involved at all.
    fn is_compatible_with(&self, other: &Self) -> bool;

    /// Returns `true` if every value described by `other` is also described by this [`Type`]. The receiver is the
    /// more general type (e.g., a declared or staged type) and the argument is the more precise one (e.g., the actual
    /// type carried by a runtime value), and so the relation is directional: `declared.is_refined_by(&actual)`.
    /// Interpretation entry points such as [`Program::interpret`](crate::Program::interpret) use this relation to
    /// validate runtime input values against declared program input types.
    ///
    /// For fully static types this is type equality (e.g., [`DataType`](crate::types::DataType) requires equal data
    /// types). Types that can carry unknown components additionally admit every more precise instantiation of those
    /// components. For example, [`ArrayType`](crate::types::ArrayType)s with
    /// [`Dimension::Dynamic`](crate::Dimension::Dynamic) dimensions are refined by otherwise-equal
    /// [`ArrayType`](crate::ArrayType)s whose corresponding dimensions are static,
    /// per [`Dimension::is_refined_by`](crate::Dimension::is_refined_by).
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

    /// Returns `true` if this [`Type`] describes complex-valued (e.g.,
    /// [`DataType::C64`](crate::types::DataType::C64) or [`DataType::C128`](crate::types::DataType::C128)) numeric
    /// values. Like [`Self::is_scalar`], this predicate primarily exists to serve reverse-mode differentiation. A single
    /// reverse-mode seed recovers the derivative of a complex-output function only when the function is _holomorphic_
    /// (i.e., complex-differentiable), so the gradient entry points route complex scalar outputs through their
    /// `*_holomorphic` variants. The plain entry points reject output types for which this returns `true`, and the
    /// holomorphic ones reject output types for which it returns `false`.
    fn is_complex(&self) -> bool;
}

/// Cross-occurrence established while validating one complete type signature at a [`Program`](crate::Program) or
/// [`Region`](crate::Region) boundary. These facts relate declared [`TypeIdentity`]s to refinements observed in
/// corresponding actual types, ensuring that repeated identity occurrences remain consistent across boundary leaves
/// and outputs.
///
/// [`Type::is_refined_by`] checks one type pair. This companion contract checks a complete boundary so relationships
/// repeated across several inputs or outputs remain consistent. Establishment via [`Self::establish`] is transactional
/// across the complete input signature, and the resulting environment validates outputs against those same facts.
/// [`TypeIdentity`]s produced inside the [`Program`](crate::Program) or [`Region`](crate::Region) may establish
/// additional output facts, while unrelated, unbound output identities are rejected.
pub trait TypeRefinements<T: Type>: Clone + Debug + Default {
    /// Establishes refinement facts from `actual` relative to `declared`.
    fn establish(declared: &[T], actual: &[T]) -> Result<Self, TypeError>;

    /// Validates `actual` against `declared` using the already-established facts. Identities in `internal_identities`
    /// may establish new facts at this boundary. All other identities must already be bound.
    fn validate(&self, declared: &[T], actual: &[T], internal_identities: &[T::Identity]) -> Result<(), TypeError>;
}

impl<T: Type> TypeRefinements<T> for () {
    #[inline]
    fn establish(declared: &[T], actual: &[T]) -> Result<Self, TypeError> {
        Self::validate(&(), declared, actual, &[])
    }

    fn validate(&self, declared: &[T], actual: &[T], _internal_identities: &[T::Identity]) -> Result<(), TypeError> {
        if declared.len() != actual.len() {
            return Err(TypeError::invalid(format!(
                "declared type count {} does not match actual type count {}",
                declared.len(),
                actual.len(),
            )));
        }
        for (declared, actual) in declared.iter().zip(actual) {
            if !declared.is_refined_by(actual) {
                return Err(TypeError::invalid(format!("type {actual} does not refine declared type {declared}")));
            }
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

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
}
