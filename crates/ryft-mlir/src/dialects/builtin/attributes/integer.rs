use ryft_xla_sys::bindings::{
    MlirAttribute, mlirIntegerAttrGet, mlirIntegerAttrGetTypeID, mlirIntegerAttrGetValueInt,
    mlirIntegerAttrGetValueSInt, mlirIntegerAttrGetValueUInt,
};

use crate::{
    Attribute, Context, FromWithContext, IndexTypeRef, IntegerTypeRef, Type, TypeId, mlir_subtype_trait_impls,
};

/// Built-in MLIR [`Attribute`] that stores an integral value of a specific [`IntegerTypeRef`] or [`IndexTypeRef`].
/// `i1` [`IntegerAttributeRef`] are treated as [`BooleanAttributeRef`](crate::BooleanAttributeRef), and use a unique
/// assembly format of either `true` or `false` depending on the underlying value. The default type for non-boolean
/// [`IntegerAttributeRef`]s, if a type is not specified, is a signless 64-bit [`IntegerTypeRef`].
///
/// # Examples
///
/// The following are examples of [`IntegerAttributeRef`]s represented using their
/// [`Display`](std::fmt::Display) rendering:
///
/// ```text
/// 10 : i32
/// 10    // : i64 is implied here.
/// true  // A bool, i.e. i1, value.
/// false // A bool, i.e. i1, value.
/// ```
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#integerattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct IntegerAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> IntegerAttributeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`IntegerAttributeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirIntegerAttrGetTypeID()).unwrap() }
    }

    /// Returns the value stored in this [`IntegerAttributeRef`], assuming that it is signless
    /// and fits into a signed 64-bit integer.
    pub fn signless_value(&self) -> i64 {
        unsafe { mlirIntegerAttrGetValueInt(self.handle) }
    }

    /// Returns the value stored in this [`IntegerAttributeRef`], assuming that it is signed
    /// and fits into a signed 64-bit integer.
    pub fn signed_value(&self) -> i64 {
        unsafe { mlirIntegerAttrGetValueSInt(self.handle) }
    }

    /// Returns the value stored in this [`IntegerAttributeRef`], assuming that it is unsigned
    /// and fits into an unsigned 64-bit integer.
    pub fn unsigned_value(&self) -> u64 {
        unsafe { mlirIntegerAttrGetValueUInt(self.handle) }
    }
}

mlir_subtype_trait_impls!(IntegerAttributeRef<'c, 't> as Attribute, mlir_type = Attribute, mlir_subtype = Integer);

/// Helper trait for representing [`Type`]s that are supported by [`IntegerAttributeRef`]s.
pub trait IntegerAttributeType<'c, 't: 'c>: Type<'c, 't> {}
impl<'c, 't> IntegerAttributeType<'c, 't> for IndexTypeRef<'c, 't> {}
impl<'c, 't> IntegerAttributeType<'c, 't> for IntegerTypeRef<'c, 't> {}

impl<'c, 't> FromWithContext<'c, 't, usize> for IntegerAttributeRef<'c, 't> {
    fn from_with_context(value: usize, context: &'c Context<'t>) -> Self {
        context.integer_attribute(context.index_type(), value as i64)
    }
}

impl<'c, 't> FromWithContext<'c, 't, u8> for IntegerAttributeRef<'c, 't> {
    fn from_with_context(value: u8, context: &'c Context<'t>) -> Self {
        context.integer_attribute(context.unsigned_integer_type(8), value as i64)
    }
}

impl<'c, 't> FromWithContext<'c, 't, u16> for IntegerAttributeRef<'c, 't> {
    fn from_with_context(value: u16, context: &'c Context<'t>) -> Self {
        context.integer_attribute(context.unsigned_integer_type(16), value as i64)
    }
}

impl<'c, 't> FromWithContext<'c, 't, u32> for IntegerAttributeRef<'c, 't> {
    fn from_with_context(value: u32, context: &'c Context<'t>) -> Self {
        context.integer_attribute(context.unsigned_integer_type(32), value as i64)
    }
}

impl<'c, 't> FromWithContext<'c, 't, u64> for IntegerAttributeRef<'c, 't> {
    fn from_with_context(value: u64, context: &'c Context<'t>) -> Self {
        context.integer_attribute(context.unsigned_integer_type(64), i64::from_ne_bytes(value.to_ne_bytes()))
    }
}

impl<'c, 't> FromWithContext<'c, 't, i8> for IntegerAttributeRef<'c, 't> {
    fn from_with_context(value: i8, context: &'c Context<'t>) -> Self {
        context.integer_attribute(context.signed_integer_type(8), value as i64)
    }
}

impl<'c, 't> FromWithContext<'c, 't, i16> for IntegerAttributeRef<'c, 't> {
    fn from_with_context(value: i16, context: &'c Context<'t>) -> Self {
        context.integer_attribute(context.signed_integer_type(16), value as i64)
    }
}

impl<'c, 't> FromWithContext<'c, 't, i32> for IntegerAttributeRef<'c, 't> {
    fn from_with_context(value: i32, context: &'c Context<'t>) -> Self {
        context.integer_attribute(context.signed_integer_type(32), value as i64)
    }
}

impl<'c, 't> FromWithContext<'c, 't, i64> for IntegerAttributeRef<'c, 't> {
    fn from_with_context(value: i64, context: &'c Context<'t>) -> Self {
        context.integer_attribute(context.signed_integer_type(64), value)
    }
}

impl<'t> Context<'t> {
    /// Creates a new [`IntegerAttributeRef`] owned by this [`Context`].
    pub fn integer_attribute<'c, T: IntegerAttributeType<'c, 't>>(
        &'c self,
        r#type: T,
        value: i64,
    ) -> IntegerAttributeRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        let _guard = self.borrow();
        unsafe { IntegerAttributeRef::from_c_api(mlirIntegerAttrGet(r#type.to_c_api(), value), &self).unwrap() }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::IntoWithContext;
    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    #[test]
    fn test_integer_attribute_type_id() {
        let context = Context::new();
        let integer_attribute_id = IntegerAttributeRef::type_id();
        let integer_attribute_1: IntegerAttributeRef<'_, '_> = 42u32.into_with_context(&context);
        let integer_attribute_2: IntegerAttributeRef<'_, '_> = 42u64.into_with_context(&context);
        assert_eq!(integer_attribute_1.type_id(), integer_attribute_2.type_id());
        assert_eq!(integer_attribute_id, integer_attribute_1.type_id());
    }

    #[test]
    fn test_integer_attribute() {
        let context = Context::new();

        // Test with signless integer type.
        let attribute: IntegerAttributeRef<'_, '_> = 42u8.into_with_context(&context);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.signless_value(), 42);

        // Test with signed integer type.
        let attribute: IntegerAttributeRef<'_, '_> = (-100i8).into_with_context(&context);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.signed_value(), -100);

        // Test with unsigned integer type.
        let attribute: IntegerAttributeRef<'_, '_> = 65_535u16.into_with_context(&context);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.unsigned_value(), 65_535);

        // Test with index type.
        let attribute: IntegerAttributeRef<'_, '_> = 1000usize.into_with_context(&context);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.signless_value(), 1000);
    }

    #[test]
    fn test_integer_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1: IntegerAttributeRef<'_, '_> = 42i16.into_with_context(&context);
        let attribute_2: IntegerAttributeRef<'_, '_> = 42i16.into_with_context(&context);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2: IntegerAttributeRef<'_, '_> = 42i64.into_with_context(&context);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2: IntegerAttributeRef<'_, '_> = 42i16.into_with_context(&context);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_integer_attribute_display_and_debug() {
        let context = Context::new();

        let attribute: IntegerAttributeRef<'_, '_> = 42i32.into_with_context(&context);
        test_attribute_display_and_debug(attribute, "42 : si32");

        let attribute = context.integer_attribute(context.signless_integer_type(64), 100);
        test_attribute_display_and_debug(attribute, "100 : i64");
    }

    #[test]
    fn test_integer_attribute_parsing() {
        let context = Context::new();
        assert_eq!(
            context.parse_attribute("42 : i32").unwrap(),
            context.integer_attribute(context.signless_integer_type(32), 42)
        );
        assert_eq!(
            context.parse_attribute("100 : i64").unwrap(),
            context.integer_attribute(context.signless_integer_type(64), 100)
        );
    }

    #[test]
    fn test_integer_attribute_casting() {
        let context = Context::new();
        let attribute = context.integer_attribute(context.signless_integer_type(64), 42);
        test_attribute_casting(attribute);
    }
}
