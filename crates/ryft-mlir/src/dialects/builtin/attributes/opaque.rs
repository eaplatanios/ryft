use ryft_xla_sys::bindings::{
    MlirAttribute, mlirOpaqueAttrGet, mlirOpaqueAttrGetData, mlirOpaqueAttrGetDialectNamespace, mlirOpaqueAttrGetTypeID,
};

use crate::{Attribute, Context, StringRef, Type, TypeId, mlir_subtype_trait_impls};

/// Built-in MLIR [`Attribute`] that stores an opaque representation of another [`Attribute`]. Opaque attributes
/// represent attributes of non-registered dialects. These attributes are represented in their raw string form,
/// and can only usefully be tested for attribute equality.
///
/// # Examples
///
/// The following is an example of an [`OpaqueAttributeRef`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```text
/// #dialect<"opaque attribute data">
/// ```
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#integersetattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct OpaqueAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> OpaqueAttributeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`OpaqueAttributeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirOpaqueAttrGetTypeID()).unwrap() }
    }

    /// Returns the namespace of the dialect with which this [`OpaqueAttributeRef`] is associated. The returned string
    /// is owned by the underlying [`Context`] which also owns this [`OpaqueAttributeRef`].
    pub fn dialect_namespace(&'c self) -> Result<&'c str, std::str::Utf8Error> {
        unsafe { StringRef::from_c_api(mlirOpaqueAttrGetDialectNamespace(self.handle)).as_str() }
    }

    /// Returns the raw underlying data of this [`OpaqueAttributeRef`] as a string reference. The returned string is
    /// owned by the underlying [`Context`] which also owns this [`OpaqueAttributeRef`].
    pub fn data(&'c self) -> Result<&'c str, std::str::Utf8Error> {
        unsafe { StringRef::from_c_api(mlirOpaqueAttrGetData(self.handle)).as_str() }
    }
}

mlir_subtype_trait_impls!(OpaqueAttributeRef<'c, 't> as Attribute, mlir_type = Attribute, mlir_subtype = Opaque);

impl<'t> Context<'t> {
    /// Creates a new [`OpaqueAttributeRef`] associated with the dialect that corresponds to the provided namespace,
    /// which is owned by this [`Context`].
    pub fn opaque_attribute<'c, S: AsRef<str>, D: AsRef<str>, T: Type<'c, 't>>(
        &'c self,
        dialect_namespace: S,
        data: D,
        r#type: T,
    ) -> OpaqueAttributeRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        let data = data.as_ref();
        unsafe {
            OpaqueAttributeRef::from_c_api(
                mlirOpaqueAttrGet(
                    *self.handle.borrow(),
                    StringRef::from(dialect_namespace.as_ref()).to_c_api(),
                    data.len().cast_signed(),
                    data.as_ptr() as *const _,
                    r#type.to_c_api(),
                ),
                &self,
            )
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
    fn test_opaque_attribute_type_id() {
        let context = Context::new();
        let opaque_attribute_id = OpaqueAttributeRef::type_id();
        let opaque_attribute_1 = context.opaque_attribute("test_dialect", "test_data", context.index_type());
        let opaque_attribute_2 = context.opaque_attribute("test_dialect", "test_data", context.index_type());
        assert_eq!(opaque_attribute_1.type_id(), opaque_attribute_2.type_id());
        assert_eq!(opaque_attribute_id, opaque_attribute_1.type_id());
    }

    #[test]
    fn test_opaque_attribute() {
        let context = Context::new();

        // Test opaque attribute with index type.
        let index_type = context.index_type();
        let attribute = context.opaque_attribute("test_dialect", "opaque_data", index_type);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.dialect_namespace().unwrap(), "test_dialect");
        assert_eq!(attribute.data().unwrap(), "opaque_data");

        // Test with another dialect and data.
        let i32_type = context.signless_integer_type(32);
        let attribute = context.opaque_attribute("my_dialect", "some data", i32_type);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.dialect_namespace().unwrap(), "my_dialect");
        assert_eq!(attribute.data().unwrap(), "some data");
    }

    #[test]
    fn test_opaque_attribute_equality() {
        let context = Context::new();
        let index_type = context.index_type();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.opaque_attribute("test", "data", index_type);
        let attribute_2 = context.opaque_attribute("test", "data", index_type);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.opaque_attribute("test", "other_data", index_type);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let other_index_type = context.index_type();
        let attribute_2 = context.opaque_attribute("test", "data", other_index_type);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_opaque_attribute_display_and_debug() {
        let context = Context::new();
        let index_type = context.index_type();
        let attribute = context.opaque_attribute("test", "data", index_type);
        test_attribute_display_and_debug(attribute, "#test.data : index");
    }

    #[test]
    fn test_opaque_attribute_parsing() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        assert_eq!(
            context.parse_attribute("#test.data : index").unwrap(),
            context.opaque_attribute("test", "data", context.index_type()),
        );
    }

    #[test]
    fn test_opaque_attribute_casting() {
        let context = Context::new();
        let index_type = context.index_type();
        let attribute = context.opaque_attribute("test", "data", index_type);
        test_attribute_casting(attribute);
    }
}
