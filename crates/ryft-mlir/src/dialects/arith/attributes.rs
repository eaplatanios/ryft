use crate::{Context, IntegerAttributeRef};

/// Atomic read-modify-write reduction kind used by MLIR arithmetic operations and affine parallel reductions.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AtomicRmwKind {
    /// Floating-point addition reduction.
    AddFloat,

    /// Integer addition reduction.
    AddInteger,

    /// Integer bitwise-and reduction.
    AndInteger,

    /// Assignment reduction.
    Assign,

    /// Floating-point maximum reduction.
    MaximumFloat,

    /// Floating-point maxnum reduction.
    MaxNumFloat,

    /// Signed integer maximum reduction.
    MaxSigned,

    /// Unsigned integer maximum reduction.
    MaxUnsigned,

    /// Floating-point minimum reduction.
    MinimumFloat,

    /// Floating-point minnum reduction.
    MinNumFloat,

    /// Signed integer minimum reduction.
    MinSigned,

    /// Unsigned integer minimum reduction.
    MinUnsigned,

    /// Floating-point multiplication reduction.
    MulFloat,

    /// Integer multiplication reduction.
    MulInteger,

    /// Integer bitwise-or reduction.
    OrInteger,

    /// Integer bitwise-xor reduction.
    XorInteger,
}

impl AtomicRmwKind {
    /// Returns the integer representation used by MLIR for this atomic read-modify-write kind.
    pub fn value(&self) -> i64 {
        match self {
            Self::AddFloat => 0,
            Self::AddInteger => 1,
            Self::AndInteger => 2,
            Self::Assign => 3,
            Self::MaximumFloat => 4,
            Self::MaxNumFloat => 5,
            Self::MaxSigned => 6,
            Self::MaxUnsigned => 7,
            Self::MinimumFloat => 8,
            Self::MinNumFloat => 9,
            Self::MinSigned => 10,
            Self::MinUnsigned => 11,
            Self::MulFloat => 12,
            Self::MulInteger => 13,
            Self::OrInteger => 14,
            Self::XorInteger => 15,
        }
    }

    /// Creates an [`AtomicRmwKind`] from the integer representation used by MLIR.
    pub fn from_value(value: i64) -> Option<Self> {
        match value {
            0 => Some(Self::AddFloat),
            1 => Some(Self::AddInteger),
            2 => Some(Self::AndInteger),
            3 => Some(Self::Assign),
            4 => Some(Self::MaximumFloat),
            5 => Some(Self::MaxNumFloat),
            6 => Some(Self::MaxSigned),
            7 => Some(Self::MaxUnsigned),
            8 => Some(Self::MinimumFloat),
            9 => Some(Self::MinNumFloat),
            10 => Some(Self::MinSigned),
            11 => Some(Self::MinUnsigned),
            12 => Some(Self::MulFloat),
            13 => Some(Self::MulInteger),
            14 => Some(Self::OrInteger),
            15 => Some(Self::XorInteger),
            _ => None,
        }
    }
}

/// Creates the integer attribute representation used by MLIR for an [`AtomicRmwKind`].
pub(crate) fn atomic_rmw_kind_attribute<'c, 't>(
    context: &'c Context<'t>,
    kind: AtomicRmwKind,
) -> IntegerAttributeRef<'c, 't> {
    context.integer_attribute(context.signless_integer_type(64), kind.value())
}

/// Converts an integer attribute into an [`AtomicRmwKind`], if it stores a valid MLIR enum value.
pub(crate) fn atomic_rmw_kind_from_attribute(attribute: IntegerAttributeRef<'_, '_>) -> Option<AtomicRmwKind> {
    AtomicRmwKind::from_value(attribute.signless_value())
}
