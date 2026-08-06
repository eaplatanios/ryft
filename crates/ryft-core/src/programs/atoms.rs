use std::borrow::Cow;
use std::fmt::Display;

use ryft_macros::Parameter;

use crate::contexts::Context;
use crate::operations::constants::Zero;
use crate::parameters::Parameter;
use crate::programs::ProgramError;
use crate::programs::types::Typed;

/// Represents either a [`Typed`] value or a _structural zero_ that carries only its [`Type`](crate::Type).
/// [`MaybeZero`] is the symbolic zero representation shared by transforms like forward-mode and reverse-mode
/// differentiation, where it is the tangent type carried by [`DifferentiationTracer`](crate::DifferentiationTracer)s
/// and the cotangent type that transposition rules consume and produce. A [`MaybeZero::Zero`] means that no value
/// exists and nothing has been staged or computed for it. In the context of differentiation, it means that the
/// corresponding derivative is zero *by construction* (e.g., a disconnected input, a severed tangent, an unused output,
/// etc.), and is not a runtime value that happens to contain zeros. Differentiation rules branch on the variant to skip
/// work entirely. A rule that sees a zero tangent or cotangent emits no operations for it, and "zero-ness" propagates
/// transitively through rules without ever inspecting a program or materializing a buffer. A zero is _materialized_
/// into a real value only at boundaries where one is structurally required (e.g., a nested sub-program operand, a
/// program output, or an eagerly returned tangent), which is also where its carried [`Type`](crate::Type) is consumed.
#[derive(Clone, Debug)]
pub enum MaybeZero<V: Typed> {
    /// Structural zero of the carried [`Type`](crate::Type) (i.e., no value exists and nothing has been staged or
    /// computed for it).
    Zero(V::Type),

    /// Value that is not known to be structurally equal to zero.
    Value(V),
}

impl<V: Typed> MaybeZero<V> {
    /// Returns `true` if this is a [`MaybeZero::Zero`].
    #[inline]
    pub const fn is_zero(&self) -> bool {
        matches!(self, Self::Zero(_))
    }

    /// Returns the value stored in this [`MaybeZero`], if it is a [`MaybeZero::Value`], and [`None`] otherwise.
    #[inline]
    pub const fn as_value(&self) -> Option<&V> {
        match self {
            Self::Zero(_) => None,
            Self::Value(value) => Some(value),
        }
    }

    /// Maps the value stored in this [`MaybeZero`] using the provided function, leaving a [`MaybeZero::Zero`] and
    /// its carried [`Type`](crate::Type) unchanged. If this is [`MaybeZero::Zero`], then [`MaybeZero::Zero`] will be
    /// returned irrespective of what `function` is provided.
    #[inline]
    pub fn map<W: Typed<Type = V::Type>, F: FnOnce(V) -> W>(self, function: F) -> MaybeZero<W> {
        match self {
            Self::Zero(r#type) => MaybeZero::Zero(r#type),
            Self::Value(value) => MaybeZero::Value(function(value)),
        }
    }

    /// Returns the value inside this [`MaybeZero`], materializing a structural [`MaybeZero::Zero`] as a real typed
    /// zero value in the provided [`Context`] through its [`Zero`] capability (a staging context stages a typed
    /// [`ZeroOperation`](crate::ZeroOperation) instruction, while an eager context constructs a concrete zero value).
    #[inline]
    pub fn materialize<C: Context<Value = V> + Zero<V>>(self, context: &C) -> Result<V, ProgramError> {
        match self {
            Self::Value(value) => Ok(value),
            Self::Zero(r#type) => context.zero(&r#type),
        }
    }
}

impl<V: Typed> Typed for MaybeZero<V> {
    type Type = V::Type;

    #[inline]
    fn r#type(&self) -> Cow<'_, V::Type> {
        match self {
            Self::Zero(r#type) => Cow::Borrowed(r#type),
            Self::Value(value) => value.r#type(),
        }
    }
}

impl<V: Typed> From<V> for MaybeZero<V> {
    #[inline]
    fn from(value: V) -> Self {
        Self::Value(value)
    }
}

/// Unique identifier for an [`Atom`] within one [`Region`](crate::Region) of a [`Program`](crate::Program). [`AtomId`]s
/// are stable indexes into the containing [`Region`](crate::Region)'s atom table (every region owns its own table, so
/// an [`AtomId`] is meaningful only together with its region). [`Instruction`](crate::Instruction)s refer to their
/// inputs and outputs by these IDs, which keeps the intermediate representation compact and easy to clone.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Parameter)]
pub struct AtomId {
    /// Zero-based index of the corresponding [`Atom`] inside the containing [`Region`](crate::Region)'s atom table.
    index: usize,
}

impl AtomId {
    /// Creates a new [`AtomId`] from the provided zero-based atom-table index.
    #[inline]
    pub fn new(index: usize) -> Self {
        Self { index }
    }

    /// Returns the zero-based index of the corresponding [`Atom`] inside the owning [`Program`](crate::Program)'s
    /// atom table.
    #[inline]
    pub fn index(self) -> usize {
        self.index
    }
}

impl Display for AtomId {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "%{}", self.index)
    }
}

/// [`Atom`]s represent nodes in the [`Region`](crate::Region)s of [`Program`](crate::Program)s that represent either
/// concrete values or variables of specific [`Type`](crate::Type)s.
#[derive(Clone, Debug, Parameter)]
pub enum Atom<V: Typed> {
    /// Literal constant value that appears in a [`Program`](crate::Program).
    Constant(V),

    /// Non-constant variable of a specific [`Type`](crate::Type) that appears in a [`Program`](crate::Program).
    Variable(V::Type),
}

impl<V: Typed> Atom<V> {
    /// Returns `true` if this [`Atom`] is an [`Atom::Constant`].
    #[inline]
    pub fn is_constant(&self) -> bool {
        matches!(self, Self::Constant(_))
    }

    /// Returns `true` if this [`Atom`] is an [`Atom::Variable`].
    #[inline]
    pub fn is_variable(&self) -> bool {
        matches!(self, Self::Variable(_))
    }

    /// Returns the underlying constant value if this atom is an [`Atom::Constant`] and [`None`] otherwise.
    #[inline]
    pub fn as_constant(&self) -> Option<&V> {
        match self {
            Self::Constant(value) => Some(value),
            Self::Variable(_) => None,
        }
    }
}

impl<V: Typed> Typed for Atom<V> {
    type Type = V::Type;

    fn r#type(&self) -> Cow<'_, V::Type> {
        match self {
            Self::Constant(value) => value.r#type(),
            Self::Variable(r#type) => Cow::Borrowed(r#type),
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::Array;
    use crate::types::{ArrayType, DataType};

    use super::*;

    #[test]
    fn test_atom_id() {
        assert_eq!(AtomId::new(1).to_string(), "%1");
    }

    #[test]
    fn test_atom() {
        let constant = Atom::<Array>::Constant(Array::scalar(3.0));
        let variable = Atom::<Array>::Variable(ArrayType::scalar(DataType::F64));

        assert!(constant.is_constant());
        assert!(!constant.is_variable());
        assert_eq!(constant.as_constant(), Some(&Array::scalar(3.0)));
        assert_eq!(constant.r#type().into_owned(), ArrayType::scalar(DataType::F64));

        assert!(variable.is_variable());
        assert_eq!(variable.as_constant(), None);
        assert_eq!(variable.r#type().into_owned(), ArrayType::scalar(DataType::F64));
    }
}
