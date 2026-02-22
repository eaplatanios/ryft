use ryft_xla_sys::bindings::{MlirType, mlirIndexTypeGet, mlirIndexTypeGetTypeID};

use crate::{Context, Type, TypeId, mlir_subtype_trait_impls};

/// Built-in MLIR [`Type`] that represents an index type (i.e., an integer type with an unknown platform-dependent
/// bit width). Refer to the [MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#indextype)
/// for more information.
#[derive(Copy, Clone)]
pub struct IndexTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> IndexTypeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`IndexTypeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirIndexTypeGetTypeID()).unwrap() }
    }
}

mlir_subtype_trait_impls!(IndexTypeRef<'c, 't> as Type, mlir_type = Type, mlir_subtype = Index);

impl<'t> Context<'t> {
    /// Creates a new [`IndexTypeRef`] owned by this [`Context`].
    pub fn index_type<'c>(&'c self) -> IndexTypeRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe { IndexTypeRef::from_c_api(mlirIndexTypeGet(*self.handle.borrow()), self).unwrap() }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::types::tests::{test_type_casting, test_type_display_and_debug};

    use super::*;

    #[test]
    fn test_index_type_type_id() {
        let context = Context::new();
        let index_type_id = IndexTypeRef::type_id();
        let index_type_1 = context.index_type();
        let index_type_2 = context.index_type();
        assert_eq!(index_type_1.type_id(), index_type_2.type_id());
        assert_eq!(index_type_id, index_type_1.type_id());
    }

    #[test]
    fn test_index_type() {
        let context = Context::new();
        let r#type = context.index_type();
        assert_eq!(&context, r#type.context());
    }

    #[test]
    fn test_index_type_equality() {
        let context = Context::new();

        // Same types from the same context must be equal because they are "uniqued".
        let type_1 = context.index_type();
        let type_2 = context.index_type();
        assert_eq!(type_1, type_2);

        // Same types from different contexts must not be equal.
        let context = Context::new();
        let type_2 = context.index_type();
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_index_type_display_and_debug() {
        let context = Context::new();
        let r#type = context.index_type();
        test_type_display_and_debug(r#type, "index");
    }

    #[test]
    fn test_index_type_parsing() {
        let context = Context::new();
        assert_eq!(context.parse_type("index").unwrap(), context.index_type());
    }

    #[test]
    fn test_index_type_casting() {
        let context = Context::new();
        let r#type = context.index_type();
        test_type_casting(r#type);
    }
}
