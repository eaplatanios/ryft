use ryft_xla_sys::bindings::{
    MlirType, mlirOpaqueTypeGet, mlirOpaqueTypeGetData, mlirOpaqueTypeGetDialectNamespace, mlirOpaqueTypeGetTypeID,
};

use crate::{Context, StringRef, Type, TypeId, mlir_subtype_trait_impls};

/// Built-in MLIR [`Type`] that represents a type associated with a dialect that has not been registered with the
/// owning context. [`OpaqueTypeRef`] [`Display`](std::fmt::Display) renderings consist of the namespace of the dialect
/// associated with the type, combined with the underlying type data (e.g., `opaque<"llvm", "struct<(i32, float)>">` or
/// `opaque<"pdl", "value">`).
///
/// Refer to the [MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#opaquetype) for more information.
#[derive(Copy, Clone)]
pub struct OpaqueTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> OpaqueTypeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`OpaqueTypeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirOpaqueTypeGetTypeID()).unwrap() }
    }

    /// Returns the namespace of the dialect with which this [`OpaqueTypeRef`] is associated. The returned string is
    /// owned by the underlying [`Context`] which also owns this [`OpaqueTypeRef`].
    pub fn dialect_namespace(&'c self) -> Result<&'c str, std::str::Utf8Error> {
        unsafe { StringRef::from_c_api(mlirOpaqueTypeGetDialectNamespace(self.handle)).as_str() }
    }

    /// Returns the raw underlying data of this [`OpaqueTypeRef`] as a string reference. The returned string is
    /// owned by the underlying [`Context`] which also owns this [`OpaqueTypeRef`].
    pub fn data(&'c self) -> Result<&'c str, std::str::Utf8Error> {
        unsafe { StringRef::from_c_api(mlirOpaqueTypeGetData(self.handle)).as_str() }
    }
}

mlir_subtype_trait_impls!(OpaqueTypeRef<'c, 't> as Type, mlir_type = Type, mlir_subtype = Opaque);

impl<'t> Context<'t> {
    /// Creates a new [`OpaqueTypeRef`] associated with the dialect that corresponds to the provided namespace,
    /// which is owned by this [`Context`].
    pub fn opaque_type<'c, S: AsRef<str>, D: AsRef<str>>(
        &'c self,
        dialect_namespace: S,
        data: D,
    ) -> OpaqueTypeRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe {
            OpaqueTypeRef::from_c_api(
                mlirOpaqueTypeGet(
                    *self.handle.borrow(),
                    StringRef::from(dialect_namespace.as_ref()).to_c_api(),
                    StringRef::from(data.as_ref()).to_c_api(),
                ),
                self,
            )
            .unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::types::tests::{test_type_casting, test_type_display_and_debug};

    use super::*;

    #[test]
    fn test_opaque_type_type_id() {
        let context = Context::new();
        let opaque_type = OpaqueTypeRef::type_id();
        let opaque_type_1 = context.opaque_type("test_dialect", "test_data");
        let opaque_type_2 = context.opaque_type("test_dialect", "test_data");
        assert_eq!(opaque_type_1.type_id(), opaque_type_2.type_id());
        assert_eq!(opaque_type, opaque_type_1.type_id());
    }

    #[test]
    fn test_opaque_type() {
        let context = Context::new();
        let r#type = context.opaque_type("test_dialect", "test_data");
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.dialect_namespace().unwrap(), "test_dialect");
        assert_eq!(r#type.data().unwrap(), "test_data");
    }

    #[test]
    fn test_opaque_type_equality() {
        let context = Context::new();

        // Same types from the same context must be equal because they are "uniqued".
        let type_1 = context.opaque_type("test_dialect", "test_data");
        let type_2 = context.opaque_type("test_dialect", "test_data");
        assert_eq!(type_1, type_2);

        // Different data from the same context must not be equal.
        let type_2 = context.opaque_type("test_dialect", "other_data");
        assert_ne!(type_1, type_2);

        // Different namespace from the same context must not be equal.
        let type_2 = context.opaque_type("other_dialect", "test_data");
        assert_ne!(type_1, type_2);

        // Same types from different contexts must not be equal.
        let context = Context::new();
        let type_2 = context.opaque_type("test_dialect", "test_data");
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_opaque_type_display_and_debug() {
        let context = Context::new();
        let r#type = context.opaque_type("llvm", "struct<(i32, float)>");
        test_type_display_and_debug(r#type, "!llvm.struct<(i32, float)>");
    }

    #[test]
    fn test_opaque_type_parsing() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        assert_eq!(
            context.parse_type("!llvm.struct<(i32, float)>").unwrap(),
            context.opaque_type("llvm", "struct<(i32, float)>"),
        );
    }

    #[test]
    fn test_opaque_type_casting() {
        let context = Context::new();
        let r#type = context.opaque_type("test_dialect", "test_data");
        test_type_casting(r#type);
    }
}
