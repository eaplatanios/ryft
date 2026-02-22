use ryft_xla_sys::bindings::{MlirAttribute, mlirTypeAttrGet, mlirTypeAttrGetTypeID, mlirTypeAttrGetValue};

use crate::{Attribute, Context, FromWithContext, Type, TypeId, TypeRef, mlir_subtype_trait_impls};

/// Built-in MLIR [`Attribute`] that stores a [`Type`].
///
/// # Examples
///
/// The following are examples of [`TypeAttributeRef`]s represented using their
/// [`Display`](std::fmt::Display) rendering:
///
/// ```text
/// i32
/// !dialect.type
/// ```
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#typeattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct TypeAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> TypeAttributeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`TypeAttributeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirTypeAttrGetTypeID()).unwrap() }
    }

    /// Returns the [`Type`] that is stored in this [`TypeAttributeRef`].
    pub fn r#type(&self) -> TypeRef<'c, 't> {
        unsafe { TypeRef::from_c_api(mlirTypeAttrGetValue(self.handle), self.context).unwrap() }
    }
}

mlir_subtype_trait_impls!(TypeAttributeRef<'c, 't> as Attribute, mlir_type = Attribute, mlir_subtype = Type);

impl<'c, 't, T: Type<'c, 't>> FromWithContext<'c, 't, T> for TypeAttributeRef<'c, 't> {
    fn from_with_context(value: T, context: &'c Context<'t>) -> Self {
        context.type_attribute(value)
    }
}

impl<'t> Context<'t> {
    /// Creates a new [`TypeAttributeRef`] owned by this [`Context`].
    pub fn type_attribute<'c, T: Type<'c, 't>>(&'c self, r#type: T) -> TypeAttributeRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        let _guard = self.borrow();
        unsafe { TypeAttributeRef::from_c_api(mlirTypeAttrGet(r#type.to_c_api()), self).unwrap() }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::IntoWithContext;
    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    #[test]
    fn test_type_attribute_type_id() {
        let context = Context::new();
        let type_attribute_id = TypeAttributeRef::type_id();
        let type_attribute_1 = context.type_attribute(context.index_type());
        let type_attribute_2: TypeAttributeRef<'_, '_> = context.index_type().into_with_context(&context);
        assert_eq!(type_attribute_1.type_id(), type_attribute_2.type_id());
        assert_eq!(type_attribute_id, type_attribute_1.type_id());
    }

    #[test]
    fn test_type_attribute() {
        let context = Context::new();

        // Test with index type.
        let index_type = context.index_type();
        let attribute = context.type_attribute(index_type);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.r#type(), index_type.as_ref());

        // Test with integer type.
        let i64_type = context.signless_integer_type(64);
        let attribute = context.type_attribute(i64_type);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.r#type(), i64_type.as_ref());

        // Test with float type.
        let f32_type = context.float32_type();
        let attribute = context.type_attribute(f32_type);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.r#type(), f32_type.as_ref());
    }

    #[test]
    fn test_type_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.type_attribute(context.index_type());
        let attribute_2 = context.type_attribute(context.index_type());
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.type_attribute(context.signless_integer_type(32));
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.type_attribute(context.index_type());
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_type_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.type_attribute(context.signless_integer_type(32));
        test_attribute_display_and_debug(attribute, "i32");
    }

    #[test]
    fn test_type_attribute_parsing() {
        let context = Context::new();
        assert_eq!(context.parse_attribute("i32").unwrap(), context.type_attribute(context.signless_integer_type(32)));
    }

    #[test]
    fn test_type_attribute_casting() {
        let context = Context::new();
        let attribute = context.type_attribute(context.index_type());
        test_attribute_casting(attribute);
    }
}
