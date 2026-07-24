use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::sync::Arc;

use thiserror::Error;

use crate::errors::CustomError;
use crate::parameters::Parameter;
use crate::programs::ProgramError;

/// Represents errors produced while inferring or validating types.
#[derive(Clone, Debug, Error, PartialEq, Eq, Hash)]
pub enum TypeError {
    /// A generic type contract was invalid.
    #[error("{0}")]
    Invalid(String),

    /// A heterogeneous type projection encountered a different type kind than the requested one.
    #[error("expected {expected} type but got {actual} type")]
    WrongKind { expected: &'static str, actual: &'static str },

    /// A type family produced an error with a concrete, recoverable type.
    #[error("{0}")]
    Custom(Arc<dyn CustomError>),
}

impl TypeError {
    /// Wraps a type-family-specific error in a [`Custom`](TypeError::Custom) variant. The concrete error can later be
    /// recovered using [`TypeError::downcast_custom`].
    #[inline]
    pub fn custom(error: impl CustomError) -> Self {
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

    /// Converts `error` into a type error, preserving an existing [`TypeError`] and rendering every other program
    /// error as a generic invalid-type diagnostic.
    #[inline]
    pub fn from_program(error: ProgramError) -> Self {
        match error {
            ProgramError::Type(error) => error,
            error => Self::Invalid(error.to_string()),
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
    /// [`Size::Dynamic`](crate::types::Size::Dynamic) dimensions are refined by otherwise-equal
    /// [`ArrayType`](crate::types::ArrayType)s whose corresponding dimensions are static, per
    /// [`Size::is_refined_by`](crate::types::Size::is_refined_by).
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

    use crate::programs::ProgramError;

    use super::TypeError;

    #[derive(Clone, Debug, thiserror::Error, PartialEq, Eq, Hash)]
    #[error("custom type error {code}")]
    struct CustomTypeError {
        code: usize,
    }

    #[derive(Clone, Debug, thiserror::Error, PartialEq, Eq, Hash)]
    #[error("other custom type error")]
    struct OtherCustomTypeError;

    #[test]
    fn test_type_error_variants() {
        let invalid = TypeError::Invalid("invalid type".to_owned());
        assert_eq!(invalid.to_string(), "invalid type");
        assert_eq!(invalid.downcast_custom::<CustomTypeError>(), None);

        let wrong_kind = TypeError::WrongKind { expected: "array", actual: "dimension" };
        assert_eq!(wrong_kind.to_string(), "expected array type but got dimension type");
        assert_eq!(wrong_kind.downcast_custom::<CustomTypeError>(), None);

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
    fn test_type_error_from_program() {
        let type_error = TypeError::custom(CustomTypeError { code: 7 });
        assert_eq!(TypeError::from_program(ProgramError::Type(type_error.clone())), type_error);
        assert_eq!(
            TypeError::from_program(ProgramError::InvalidArgument { message: "invalid argument".to_owned() }),
            TypeError::Invalid("invalid argument".to_owned()),
        );
    }
}
