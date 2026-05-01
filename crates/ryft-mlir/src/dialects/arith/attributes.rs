use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

use ryft_xla_sys::bindings::MlirAttribute;
use ryft_xla_sys::mlir::dialects::arith::{
    mlirArithAtomicRmwKindAttrGet, mlirArithAtomicRmwKindAttrGetValue, mlirArithFastMathFlagsAttrGet,
    mlirArithFastMathFlagsAttrGetValue, mlirArithIntegerOverflowFlagsAttrGet,
    mlirArithIntegerOverflowFlagsAttrGetValue, mlirArithRoundingModeAttrGet, mlirArithRoundingModeAttrGetValue,
    mlirAttributeIsAArithAtomicRmwKindAttr, mlirAttributeIsAArithFastMathFlagsAttr,
    mlirAttributeIsAArithIntegerOverflowFlagsAttr, mlirAttributeIsAArithRoundingModeAttr,
};

use crate::{Attribute, Context, DialectHandle, FromWithContext, mlir_subtype_trait_impls};

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

/// MLIR `arith` atomic read-modify-write kind [`Attribute`].
#[derive(Copy, Clone)]
pub struct AtomicRmwKindAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl AtomicRmwKindAttributeRef<'_, '_> {
    /// Returns the atomic read-modify-write kind stored in this attribute.
    pub fn value(&self) -> AtomicRmwKind {
        AtomicRmwKind::from_value(unsafe { mlirArithAtomicRmwKindAttrGetValue(self.handle) as i64 })
            .expect("invalid `arith::AtomicRmwKind` attribute")
    }
}

impl<'c, 't> Attribute<'c, 't> for AtomicRmwKindAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirAttributeIsAArithAtomicRmwKindAttr(handle) } {
            Some(Self { handle, context })
        } else {
            None
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(AtomicRmwKindAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'c, 't> FromWithContext<'c, 't, AtomicRmwKind> for AtomicRmwKindAttributeRef<'c, 't> {
    fn from_with_context(value: AtomicRmwKind, context: &'c Context<'t>) -> Self {
        context.arith_atomic_rmw_kind_attribute(value)
    }
}

/// Floating-point fast-math flags used by MLIR arithmetic operations.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FastMathFlags {
    /// Bit mask storing the MLIR fast-math flags.
    bits: u32,
}

impl FastMathFlags {
    /// No fast-math assumptions.
    pub const NONE: Self = Self { bits: 0 };

    /// Floating-point reassociation is allowed.
    pub const REASSOCIATE: Self = Self { bits: 1 };

    /// Operands and results are assumed not to be NaN.
    pub const NO_NANS: Self = Self { bits: 2 };

    /// Operands and results are assumed not to be infinite.
    pub const NO_INFINITIES: Self = Self { bits: 4 };

    /// Signed zeroes may be treated as unsigned zeroes.
    pub const NO_SIGNED_ZEROES: Self = Self { bits: 8 };

    /// Reciprocal transformations are allowed.
    pub const ALLOW_RECIPROCAL: Self = Self { bits: 16 };

    /// Floating-point contraction is allowed.
    pub const ALLOW_CONTRACT: Self = Self { bits: 32 };

    /// Approximate functions are allowed.
    pub const APPROXIMATE_FUNCTIONS: Self = Self { bits: 64 };

    /// All fast-math transformations are allowed.
    pub const FAST: Self = Self { bits: 127 };

    /// Returns the raw MLIR bit representation of these flags.
    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// Creates [`FastMathFlags`] from a raw MLIR bit representation.
    pub fn from_bits(bits: u32) -> Option<Self> {
        if bits & !Self::FAST.bits == 0 { Some(Self { bits }) } else { None }
    }

    /// Returns `true` if all `flags` are present in this flag set.
    pub fn contains(&self, flags: Self) -> bool {
        self.bits & flags.bits == flags.bits
    }
}

impl BitOr for FastMathFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self { bits: self.bits | rhs.bits }
    }
}

impl BitOrAssign for FastMathFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl BitAnd for FastMathFlags {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self { bits: self.bits & rhs.bits }
    }
}

impl BitAndAssign for FastMathFlags {
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl BitXor for FastMathFlags {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self { bits: self.bits ^ rhs.bits }
    }
}

impl BitXorAssign for FastMathFlags {
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

impl Not for FastMathFlags {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self { bits: !self.bits & Self::FAST.bits }
    }
}

/// MLIR `arith` fast-math flags [`Attribute`].
#[derive(Copy, Clone)]
pub struct FastMathFlagsAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl FastMathFlagsAttributeRef<'_, '_> {
    /// Returns the fast-math flags stored in this attribute.
    pub fn value(&self) -> FastMathFlags {
        FastMathFlags::from_bits(unsafe { mlirArithFastMathFlagsAttrGetValue(self.handle) })
            .expect("invalid `arith::FastMathFlags` attribute")
    }
}

impl<'c, 't> Attribute<'c, 't> for FastMathFlagsAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirAttributeIsAArithFastMathFlagsAttr(handle) } {
            Some(Self { handle, context })
        } else {
            None
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(FastMathFlagsAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'c, 't> FromWithContext<'c, 't, FastMathFlags> for FastMathFlagsAttributeRef<'c, 't> {
    fn from_with_context(value: FastMathFlags, context: &'c Context<'t>) -> Self {
        context.arith_fast_math_flags_attribute(value)
    }
}

/// Integer overflow flags used by MLIR arithmetic operations.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IntegerOverflowFlags {
    /// Bit mask storing the MLIR integer overflow flags.
    bits: u32,
}

impl IntegerOverflowFlags {
    /// No overflow assumptions.
    pub const NONE: Self = Self { bits: 0 };

    /// Signed integer operations do not overflow.
    pub const NO_SIGNED_WRAP: Self = Self { bits: 1 };

    /// Unsigned integer operations do not overflow.
    pub const NO_UNSIGNED_WRAP: Self = Self { bits: 2 };

    /// Returns the raw MLIR bit representation of these flags.
    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// Creates [`IntegerOverflowFlags`] from a raw MLIR bit representation.
    pub fn from_bits(bits: u32) -> Option<Self> {
        if bits & !(Self::NO_SIGNED_WRAP.bits | Self::NO_UNSIGNED_WRAP.bits) == 0 { Some(Self { bits }) } else { None }
    }

    /// Returns `true` if all `flags` are present in this flag set.
    pub fn contains(&self, flags: Self) -> bool {
        self.bits & flags.bits == flags.bits
    }
}

impl BitOr for IntegerOverflowFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self { bits: self.bits | rhs.bits }
    }
}

impl BitOrAssign for IntegerOverflowFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl BitAnd for IntegerOverflowFlags {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self { bits: self.bits & rhs.bits }
    }
}

impl BitAndAssign for IntegerOverflowFlags {
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl BitXor for IntegerOverflowFlags {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self { bits: self.bits ^ rhs.bits }
    }
}

impl BitXorAssign for IntegerOverflowFlags {
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

impl Not for IntegerOverflowFlags {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self { bits: !self.bits & (Self::NO_SIGNED_WRAP.bits | Self::NO_UNSIGNED_WRAP.bits) }
    }
}

/// MLIR `arith` integer overflow flags [`Attribute`].
#[derive(Copy, Clone)]
pub struct IntegerOverflowFlagsAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl IntegerOverflowFlagsAttributeRef<'_, '_> {
    /// Returns the integer overflow flags stored in this attribute.
    pub fn value(&self) -> IntegerOverflowFlags {
        IntegerOverflowFlags::from_bits(unsafe { mlirArithIntegerOverflowFlagsAttrGetValue(self.handle) })
            .expect("invalid `arith::IntegerOverflowFlags` attribute")
    }
}

impl<'c, 't> Attribute<'c, 't> for IntegerOverflowFlagsAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirAttributeIsAArithIntegerOverflowFlagsAttr(handle) } {
            Some(Self { handle, context })
        } else {
            None
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(IntegerOverflowFlagsAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'c, 't> FromWithContext<'c, 't, IntegerOverflowFlags> for IntegerOverflowFlagsAttributeRef<'c, 't> {
    fn from_with_context(value: IntegerOverflowFlags, context: &'c Context<'t>) -> Self {
        context.arith_integer_overflow_flags_attribute(value)
    }
}

/// Floating-point rounding mode used by MLIR arithmetic cast operations.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RoundingMode {
    /// Rounds to the nearest value, with ties rounded to the even significand.
    ToNearestEven,

    /// Rounds toward negative infinity.
    Downward,

    /// Rounds toward positive infinity.
    Upward,

    /// Rounds toward zero.
    TowardZero,

    /// Rounds to the nearest value, with ties rounded away from zero.
    ToNearestAwayFromZero,
}

impl RoundingMode {
    /// Returns the integer representation used by MLIR for this rounding mode.
    pub fn value(&self) -> u32 {
        match self {
            Self::ToNearestEven => 0,
            Self::Downward => 1,
            Self::Upward => 2,
            Self::TowardZero => 3,
            Self::ToNearestAwayFromZero => 4,
        }
    }

    /// Creates a [`RoundingMode`] from the integer representation used by MLIR.
    pub fn from_value(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::ToNearestEven),
            1 => Some(Self::Downward),
            2 => Some(Self::Upward),
            3 => Some(Self::TowardZero),
            4 => Some(Self::ToNearestAwayFromZero),
            _ => None,
        }
    }
}

/// MLIR `arith` rounding mode [`Attribute`].
#[derive(Copy, Clone)]
pub struct RoundingModeAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl RoundingModeAttributeRef<'_, '_> {
    /// Returns the rounding mode stored in this attribute.
    pub fn value(&self) -> RoundingMode {
        RoundingMode::from_value(unsafe { mlirArithRoundingModeAttrGetValue(self.handle) })
            .expect("invalid `arith::RoundingMode` attribute")
    }
}

impl<'c, 't> Attribute<'c, 't> for RoundingModeAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirAttributeIsAArithRoundingModeAttr(handle) } {
            Some(Self { handle, context })
        } else {
            None
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(RoundingModeAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'c, 't> FromWithContext<'c, 't, RoundingMode> for RoundingModeAttributeRef<'c, 't> {
    fn from_with_context(value: RoundingMode, context: &'c Context<'t>) -> Self {
        context.arith_rounding_mode_attribute(value)
    }
}

impl<'t> Context<'t> {
    /// Creates an `arith` atomic read-modify-write kind attribute owned by this [`Context`].
    pub fn arith_atomic_rmw_kind_attribute<'c>(&'c self, value: AtomicRmwKind) -> AtomicRmwKindAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::arith());
        unsafe {
            AtomicRmwKindAttributeRef::from_c_api(
                mlirArithAtomicRmwKindAttrGet(*self.handle.borrow_mut(), value.value() as u64),
                self,
            )
            .expect("invalid arguments to `Context::arith_atomic_rmw_kind_attribute`")
        }
    }

    /// Creates an `arith` fast-math flags attribute owned by this [`Context`].
    pub fn arith_fast_math_flags_attribute<'c>(&'c self, value: FastMathFlags) -> FastMathFlagsAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::arith());
        unsafe {
            FastMathFlagsAttributeRef::from_c_api(
                mlirArithFastMathFlagsAttrGet(*self.handle.borrow_mut(), value.bits()),
                self,
            )
            .expect("invalid arguments to `Context::arith_fast_math_flags_attribute`")
        }
    }

    /// Creates an `arith` integer overflow flags attribute owned by this [`Context`].
    pub fn arith_integer_overflow_flags_attribute<'c>(
        &'c self,
        value: IntegerOverflowFlags,
    ) -> IntegerOverflowFlagsAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::arith());
        unsafe {
            IntegerOverflowFlagsAttributeRef::from_c_api(
                mlirArithIntegerOverflowFlagsAttrGet(*self.handle.borrow_mut(), value.bits()),
                self,
            )
            .expect("invalid arguments to `Context::arith_integer_overflow_flags_attribute`")
        }
    }

    /// Creates an `arith` rounding mode attribute owned by this [`Context`].
    pub fn arith_rounding_mode_attribute<'c>(&'c self, value: RoundingMode) -> RoundingModeAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::arith());
        unsafe {
            RoundingModeAttributeRef::from_c_api(
                mlirArithRoundingModeAttrGet(*self.handle.borrow_mut(), value.value()),
                self,
            )
            .expect("invalid arguments to `Context::arith_rounding_mode_attribute`")
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    #[test]
    fn test_atomic_rmw_kind_attribute() {
        let context = Context::new();
        let attribute = context.arith_atomic_rmw_kind_attribute(AtomicRmwKind::AddInteger);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value(), AtomicRmwKind::AddInteger);
        assert_eq!(AtomicRmwKind::from_value(attribute.value().value()), Some(AtomicRmwKind::AddInteger));
    }

    #[test]
    fn test_atomic_rmw_kind_attribute_equality() {
        let context = Context::new();
        let attribute_1 = context.arith_atomic_rmw_kind_attribute(AtomicRmwKind::AddInteger);
        let attribute_2 = context.arith_atomic_rmw_kind_attribute(AtomicRmwKind::AddInteger);
        assert_eq!(attribute_1, attribute_2);

        let attribute_2 = context.arith_atomic_rmw_kind_attribute(AtomicRmwKind::MaxSigned);
        assert_ne!(attribute_1, attribute_2);

        let context = Context::new();
        let attribute_2 = context.arith_atomic_rmw_kind_attribute(AtomicRmwKind::AddInteger);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_atomic_rmw_kind_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.arith_atomic_rmw_kind_attribute(AtomicRmwKind::AddInteger);
        test_attribute_display_and_debug(attribute, "1 : i64");
    }

    #[test]
    fn test_atomic_rmw_kind_attribute_casting() {
        let context = Context::new();
        let attribute = context.arith_atomic_rmw_kind_attribute(AtomicRmwKind::AddInteger);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_fast_math_flags_attribute() {
        let context = Context::new();
        let flags = FastMathFlags::NO_NANS | FastMathFlags::NO_INFINITIES;
        let attribute = context.arith_fast_math_flags_attribute(flags);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value(), flags);
        assert!(attribute.value().contains(FastMathFlags::NO_NANS));
        assert!(attribute.value().contains(FastMathFlags::NO_INFINITIES));
        assert!(!attribute.value().contains(FastMathFlags::ALLOW_CONTRACT));
        assert_eq!(FastMathFlags::from_bits(flags.bits()), Some(flags));
        assert_eq!(FastMathFlags::from_bits(FastMathFlags::FAST.bits() + 1), None);
    }

    #[test]
    fn test_fast_math_flags_attribute_equality() {
        let context = Context::new();
        let attribute_1 = context.arith_fast_math_flags_attribute(FastMathFlags::NO_NANS);
        let attribute_2 = context.arith_fast_math_flags_attribute(FastMathFlags::NO_NANS);
        assert_eq!(attribute_1, attribute_2);

        let attribute_2 = context.arith_fast_math_flags_attribute(FastMathFlags::NO_INFINITIES);
        assert_ne!(attribute_1, attribute_2);

        let context = Context::new();
        let attribute_2 = context.arith_fast_math_flags_attribute(FastMathFlags::NO_NANS);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_fast_math_flags_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.arith_fast_math_flags_attribute(FastMathFlags::NO_NANS);
        test_attribute_display_and_debug(attribute, "#arith.fastmath<nnan>");
    }

    #[test]
    fn test_fast_math_flags_attribute_casting() {
        let context = Context::new();
        let attribute = context.arith_fast_math_flags_attribute(FastMathFlags::NO_NANS);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_integer_overflow_flags_attribute() {
        let context = Context::new();
        let flags = IntegerOverflowFlags::NO_SIGNED_WRAP | IntegerOverflowFlags::NO_UNSIGNED_WRAP;
        let attribute = context.arith_integer_overflow_flags_attribute(flags);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value(), flags);
        assert!(attribute.value().contains(IntegerOverflowFlags::NO_SIGNED_WRAP));
        assert!(attribute.value().contains(IntegerOverflowFlags::NO_UNSIGNED_WRAP));
        assert_eq!(IntegerOverflowFlags::from_bits(flags.bits()), Some(flags));
        assert_eq!(IntegerOverflowFlags::from_bits(flags.bits() + 1), None);
    }

    #[test]
    fn test_integer_overflow_flags_attribute_equality() {
        let context = Context::new();
        let attribute_1 = context.arith_integer_overflow_flags_attribute(IntegerOverflowFlags::NO_SIGNED_WRAP);
        let attribute_2 = context.arith_integer_overflow_flags_attribute(IntegerOverflowFlags::NO_SIGNED_WRAP);
        assert_eq!(attribute_1, attribute_2);

        let attribute_2 = context.arith_integer_overflow_flags_attribute(IntegerOverflowFlags::NO_UNSIGNED_WRAP);
        assert_ne!(attribute_1, attribute_2);

        let context = Context::new();
        let attribute_2 = context.arith_integer_overflow_flags_attribute(IntegerOverflowFlags::NO_SIGNED_WRAP);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_integer_overflow_flags_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.arith_integer_overflow_flags_attribute(IntegerOverflowFlags::NO_SIGNED_WRAP);
        test_attribute_display_and_debug(attribute, "#arith.overflow<nsw>");
    }

    #[test]
    fn test_integer_overflow_flags_attribute_casting() {
        let context = Context::new();
        let attribute = context.arith_integer_overflow_flags_attribute(IntegerOverflowFlags::NO_SIGNED_WRAP);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_rounding_mode_attribute() {
        let context = Context::new();
        let attribute = context.arith_rounding_mode_attribute(RoundingMode::TowardZero);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value(), RoundingMode::TowardZero);
        assert_eq!(RoundingMode::from_value(attribute.value().value()), Some(RoundingMode::TowardZero));
    }

    #[test]
    fn test_rounding_mode_attribute_equality() {
        let context = Context::new();
        let attribute_1 = context.arith_rounding_mode_attribute(RoundingMode::TowardZero);
        let attribute_2 = context.arith_rounding_mode_attribute(RoundingMode::TowardZero);
        assert_eq!(attribute_1, attribute_2);

        let attribute_2 = context.arith_rounding_mode_attribute(RoundingMode::Downward);
        assert_ne!(attribute_1, attribute_2);

        let context = Context::new();
        let attribute_2 = context.arith_rounding_mode_attribute(RoundingMode::TowardZero);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_rounding_mode_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.arith_rounding_mode_attribute(RoundingMode::TowardZero);
        test_attribute_display_and_debug(attribute, "3 : i32");
    }

    #[test]
    fn test_rounding_mode_attribute_casting() {
        let context = Context::new();
        let attribute = context.arith_rounding_mode_attribute(RoundingMode::TowardZero);
        test_attribute_casting(attribute);
    }
}
