use ryft_xla_sys::bindings::{
    MlirAttribute, mlirStringAttrGet, mlirStringAttrGetTypeID, mlirStringAttrGetValue, mlirStringAttrTypedGet,
};

use crate::{Attribute, Context, FromWithContext, StringRef, Type, TypeId, mlir_subtype_trait_impls};

/// Built-in MLIR [`Attribute`] that stores a string literal value.
///
/// # Examples
///
/// The following are examples of [`StringAttributeRef`]s represented using their
/// [`Display`](std::fmt::Display) rendering:
///
/// ```text
/// "An important string"
/// "string with a type" : !dialect.string
/// ```
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#stringattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct StringAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> StringAttributeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`StringAttributeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirStringAttrGetTypeID()).unwrap() }
    }

    /// Returns a reference to the string value that is stored in this [`StringAttributeRef`].
    pub fn string(&self) -> StringRef<'c> {
        unsafe { StringRef::from_c_api(mlirStringAttrGetValue(self.handle)) }
    }
}

mlir_subtype_trait_impls!(StringAttributeRef<'c, 't> as Attribute, mlir_type = Attribute, mlir_subtype = String);

impl<'c, 't, 's, S: Into<StringRef<'s>>> FromWithContext<'c, 't, S> for StringAttributeRef<'c, 't> {
    fn from_with_context(value: S, context: &'c Context<'t>) -> Self {
        context.string_attribute(value)
    }
}

impl<'t> Context<'t> {
    /// Creates a new [`StringAttributeRef`] owned by this [`Context`].
    pub fn string_attribute<'c, 's, S: Into<StringRef<'s>>>(&'c self, string: S) -> StringAttributeRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe {
            StringAttributeRef::from_c_api(mlirStringAttrGet(*self.handle.borrow(), string.into().to_c_api()), &self)
                .unwrap()
        }
    }

    /// Creates a new typed [`StringAttributeRef`] owned by this [`Context`].
    pub fn typed_string_attribute<'c, 's, S: Into<StringRef<'s>>, T: Type<'c, 't>>(
        &'c self,
        string: S,
        r#type: T,
    ) -> StringAttributeRef<'c, 't> {
        unsafe {
            StringAttributeRef::from_c_api(mlirStringAttrTypedGet(r#type.to_c_api(), string.into().to_c_api()), &self)
                .unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    #[test]
    fn test_string_attribute_type_id() {
        let context = Context::new();
        let string_attribute_id = StringAttributeRef::type_id();
        let string_attribute_1 = context.string_attribute("test");
        let string_attribute_2 = context.typed_string_attribute("test", context.none_type());
        assert_eq!(string_attribute_1.type_id(), string_attribute_2.type_id());
        assert_eq!(string_attribute_id, string_attribute_1.type_id());
    }

    #[test]
    fn test_string_attribute() {
        let context = Context::new();

        // Test simple string.
        let attribute = context.string_attribute("hello");
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.string().as_str().unwrap(), "hello");

        // Test string with special characters.
        let attribute = context.string_attribute("test\nwith\nnewlines");
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.string().as_str().unwrap(), "test\nwith\nnewlines");

        // Test another simple string.
        let attribute = context.string_attribute("foo bar");
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.string().as_str().unwrap(), "foo bar");
    }

    #[test]
    fn test_string_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.string_attribute("foo");
        let attribute_2 = context.string_attribute("foo");
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.string_attribute("bar");
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.string_attribute("foo");
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_string_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.string_attribute("test string");
        test_attribute_display_and_debug(attribute, "\"test string\"");
    }

    #[test]
    fn test_string_attribute_parsing() {
        let context = Context::new();
        assert_eq!(context.parse_attribute("\"test string\"").unwrap(), context.string_attribute("test string"));
    }

    #[test]
    fn test_string_attribute_casting() {
        let context = Context::new();
        let attribute = context.string_attribute("foo");
        test_attribute_casting(attribute);
    }
}
