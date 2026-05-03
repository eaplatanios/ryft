use ryft_xla_sys::bindings::{
    MlirType, mlirIntegerTypeGet, mlirIntegerTypeGetTypeID, mlirIntegerTypeGetWidth, mlirIntegerTypeIsSigned,
    mlirIntegerTypeIsSignless, mlirIntegerTypeIsUnsigned, mlirIntegerTypeSignedGet, mlirIntegerTypeUnsignedGet,
};

use crate::{Context, Type, TypeId, mlir_subtype_trait_impls};

/// Built-in MLIR [`Type`] that represents an integer type. Refer to the
/// [MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#integertype) for more information.
#[derive(Copy, Clone)]
pub struct IntegerTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> IntegerTypeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`IntegerTypeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirIntegerTypeGetTypeID()).unwrap() }
    }

    /// Returns the bit width of this [`IntegerTypeRef`]. The bit width of an integer type is defined as the
    /// number of bits that each value that belongs to that type occupies.
    pub fn bit_width(&self) -> usize {
        unsafe { mlirIntegerTypeGetWidth(self.handle) as usize }
    }

    /// Returns `true` if this [`IntegerTypeRef`] is signless.
    pub fn is_signless(&self) -> bool {
        unsafe { mlirIntegerTypeIsSignless(self.handle) }
    }

    /// Returns `true` if this [`IntegerTypeRef`] is signed.
    pub fn is_signed(&self) -> bool {
        unsafe { mlirIntegerTypeIsSigned(self.handle) }
    }

    /// Returns `true` if this [`IntegerTypeRef`] is unsigned.
    pub fn is_unsigned(&self) -> bool {
        unsafe { mlirIntegerTypeIsUnsigned(self.handle) }
    }
}

mlir_subtype_trait_impls!(IntegerTypeRef<'c, 't> as Type, mlir_type = Type, mlir_subtype = Integer);

impl<'t> Context<'t> {
    /// Creates a new signless [`IntegerTypeRef`] with the provided bit width owned by this [`Context`].
    pub fn signless_integer_type<'c>(&'c self, bit_width: usize) -> IntegerTypeRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe {
            IntegerTypeRef::from_c_api(mlirIntegerTypeGet(*self.handle.borrow(), bit_width as u32), self).unwrap()
        }
    }

    /// Creates a new signed [`IntegerTypeRef`] with the provided bit width owned by this [`Context`].
    pub fn signed_integer_type<'c>(&'c self, bit_width: usize) -> IntegerTypeRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe {
            IntegerTypeRef::from_c_api(mlirIntegerTypeSignedGet(*self.handle.borrow(), bit_width as u32), self).unwrap()
        }
    }

    /// Creates a new unsigned [`IntegerTypeRef`] with the provided bit width owned by this [`Context`].
    pub fn unsigned_integer_type<'c>(&'c self, bit_width: usize) -> IntegerTypeRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe {
            IntegerTypeRef::from_c_api(mlirIntegerTypeUnsignedGet(*self.handle.borrow(), bit_width as u32), self)
                .unwrap()
        }
    }

    /// Creates a new signless 1-bit [`IntegerTypeRef`] owned by this [`Context`].
    #[inline]
    pub fn i1_type<'c>(&'c self) -> IntegerTypeRef<'c, 't> {
        self.signless_integer_type(1)
    }

    /// Creates a new signless 1-bit [`IntegerTypeRef`] owned by this [`Context`].
    ///
    /// This is an alias for [`Context::i1_type`] for APIs that model booleans as MLIR `i1` values.
    #[inline]
    pub fn bool_type<'c>(&'c self) -> IntegerTypeRef<'c, 't> {
        self.i1_type()
    }

    /// Creates a new signless 2-bit [`IntegerTypeRef`] owned by this [`Context`].
    #[inline]
    pub fn i2_type<'c>(&'c self) -> IntegerTypeRef<'c, 't> {
        self.signless_integer_type(2)
    }

    /// Creates a new signless 4-bit [`IntegerTypeRef`] owned by this [`Context`].
    #[inline]
    pub fn i4_type<'c>(&'c self) -> IntegerTypeRef<'c, 't> {
        self.signless_integer_type(4)
    }

    /// Creates a new signless 8-bit [`IntegerTypeRef`] owned by this [`Context`].
    #[inline]
    pub fn i8_type<'c>(&'c self) -> IntegerTypeRef<'c, 't> {
        self.signless_integer_type(8)
    }

    /// Creates a new signless 16-bit [`IntegerTypeRef`] owned by this [`Context`].
    #[inline]
    pub fn i16_type<'c>(&'c self) -> IntegerTypeRef<'c, 't> {
        self.signless_integer_type(16)
    }

    /// Creates a new signless 32-bit [`IntegerTypeRef`] owned by this [`Context`].
    #[inline]
    pub fn i32_type<'c>(&'c self) -> IntegerTypeRef<'c, 't> {
        self.signless_integer_type(32)
    }

    /// Creates a new signless 64-bit [`IntegerTypeRef`] owned by this [`Context`].
    #[inline]
    pub fn i64_type<'c>(&'c self) -> IntegerTypeRef<'c, 't> {
        self.signless_integer_type(64)
    }

    /// Creates a new signless 128-bit [`IntegerTypeRef`] owned by this [`Context`].
    #[inline]
    pub fn i128_type<'c>(&'c self) -> IntegerTypeRef<'c, 't> {
        self.signless_integer_type(128)
    }

    /// Creates a new unsigned 1-bit [`IntegerTypeRef`] owned by this [`Context`].
    #[inline]
    pub fn u1_type<'c>(&'c self) -> IntegerTypeRef<'c, 't> {
        self.unsigned_integer_type(1)
    }

    /// Creates a new unsigned 2-bit [`IntegerTypeRef`] owned by this [`Context`].
    #[inline]
    pub fn u2_type<'c>(&'c self) -> IntegerTypeRef<'c, 't> {
        self.unsigned_integer_type(2)
    }

    /// Creates a new unsigned 4-bit [`IntegerTypeRef`] owned by this [`Context`].
    #[inline]
    pub fn u4_type<'c>(&'c self) -> IntegerTypeRef<'c, 't> {
        self.unsigned_integer_type(4)
    }

    /// Creates a new unsigned 8-bit [`IntegerTypeRef`] owned by this [`Context`].
    #[inline]
    pub fn u8_type<'c>(&'c self) -> IntegerTypeRef<'c, 't> {
        self.unsigned_integer_type(8)
    }

    /// Creates a new unsigned 16-bit [`IntegerTypeRef`] owned by this [`Context`].
    #[inline]
    pub fn u16_type<'c>(&'c self) -> IntegerTypeRef<'c, 't> {
        self.unsigned_integer_type(16)
    }

    /// Creates a new unsigned 32-bit [`IntegerTypeRef`] owned by this [`Context`].
    #[inline]
    pub fn u32_type<'c>(&'c self) -> IntegerTypeRef<'c, 't> {
        self.unsigned_integer_type(32)
    }

    /// Creates a new unsigned 64-bit [`IntegerTypeRef`] owned by this [`Context`].
    #[inline]
    pub fn u64_type<'c>(&'c self) -> IntegerTypeRef<'c, 't> {
        self.unsigned_integer_type(64)
    }

    /// Creates a new unsigned 128-bit [`IntegerTypeRef`] owned by this [`Context`].
    #[inline]
    pub fn u128_type<'c>(&'c self) -> IntegerTypeRef<'c, 't> {
        self.unsigned_integer_type(128)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::types::tests::{test_type_casting, test_type_display_and_debug};

    use super::*;

    #[test]
    fn test_integer_type_ids() {
        let context = Context::new();
        let integer_type = IntegerTypeRef::type_id();
        let signless_integer_2_type = context.signless_integer_type(2);
        let signless_integer_4_type = context.signless_integer_type(4);
        assert_eq!(signless_integer_2_type.type_id(), signless_integer_4_type.type_id());
        assert_eq!(integer_type, signless_integer_2_type.type_id());
    }

    #[test]
    fn test_signless_integer_type() {
        let context = Context::new();

        // Signless integer type.
        let r#type = context.signless_integer_type(32);
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.bit_width(), 32);
        assert!(r#type.is_signless());
        assert!(!r#type.is_signed());
        assert!(!r#type.is_unsigned());

        // Signed integer type.
        let r#type = context.signed_integer_type(64);
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.bit_width(), 64);
        assert!(!r#type.is_signless());
        assert!(r#type.is_signed());
        assert!(!r#type.is_unsigned());

        // Unsigned integer type.
        let r#type = context.unsigned_integer_type(16);
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.bit_width(), 16);
        assert!(!r#type.is_signless());
        assert!(!r#type.is_signed());
        assert!(r#type.is_unsigned());
    }

    #[test]
    fn test_signless_integer_type_convenience_constructors() {
        let context = Context::new();

        assert_eq!(context.i1_type(), context.signless_integer_type(1));
        assert_eq!(context.bool_type(), context.i1_type());
        assert_eq!(context.i2_type(), context.signless_integer_type(2));
        assert_eq!(context.i4_type(), context.signless_integer_type(4));
        assert_eq!(context.i8_type(), context.signless_integer_type(8));
        assert_eq!(context.i16_type(), context.signless_integer_type(16));
        assert_eq!(context.i32_type(), context.signless_integer_type(32));
        assert_eq!(context.i64_type(), context.signless_integer_type(64));
        assert_eq!(context.i128_type(), context.signless_integer_type(128));

        assert!(context.bool_type().is_signless());
        assert_eq!(context.i2_type().to_string(), "i2");
        assert_eq!(context.i32_type().bit_width(), 32);
        assert_eq!(context.i32_type().to_string(), "i32");
    }

    #[test]
    fn test_unsigned_integer_type_convenience_constructors() {
        let context = Context::new();

        assert_eq!(context.u1_type(), context.unsigned_integer_type(1));
        assert_eq!(context.u2_type(), context.unsigned_integer_type(2));
        assert_eq!(context.u4_type(), context.unsigned_integer_type(4));
        assert_eq!(context.u8_type(), context.unsigned_integer_type(8));
        assert_eq!(context.u16_type(), context.unsigned_integer_type(16));
        assert_eq!(context.u32_type(), context.unsigned_integer_type(32));
        assert_eq!(context.u64_type(), context.unsigned_integer_type(64));
        assert_eq!(context.u128_type(), context.unsigned_integer_type(128));

        assert!(context.u32_type().is_unsigned());
        assert_eq!(context.u4_type().to_string(), "ui4");
        assert_eq!(context.u32_type().bit_width(), 32);
        assert_eq!(context.u32_type().to_string(), "ui32");
    }

    #[test]
    fn test_integer_type_equality() {
        let context = Context::new();

        // Same types from the same context must be equal because they are "uniqued".
        let type_1 = context.signless_integer_type(32);
        let type_2 = context.signless_integer_type(32);
        assert_eq!(type_1, type_2);

        // Different types from the same context must not be equal.
        let type_2 = context.signless_integer_type(64);
        assert_ne!(type_1, type_2);

        // Different signedness from the same context must not be equal.
        let type_2 = context.signed_integer_type(32);
        assert_ne!(type_1, type_2);

        // Same types from different contexts must not be equal.
        let context = Context::new();
        let type_2 = context.signless_integer_type(32);
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_integer_type_display_and_debug() {
        let context = Context::new();

        let r#type = context.signless_integer_type(32);
        test_type_display_and_debug(r#type, "i32");

        let r#type = context.signed_integer_type(64);
        test_type_display_and_debug(r#type, "si64");

        let r#type = context.unsigned_integer_type(16);
        test_type_display_and_debug(r#type, "ui16");
    }

    #[test]
    fn test_integer_type_parsing() {
        let context = Context::new();
        assert_eq!(context.parse_type("i32").unwrap(), context.signless_integer_type(32));
        assert_eq!(context.parse_type("si64").unwrap(), context.signed_integer_type(64));
        assert_eq!(context.parse_type("ui16").unwrap(), context.unsigned_integer_type(16));
    }

    #[test]
    fn test_integer_type_casting() {
        let context = Context::new();
        let r#type = context.signless_integer_type(32);
        test_type_casting(r#type);
    }
}
