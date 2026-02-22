use ryft_xla_sys::bindings::{MlirType, mlirNoneTypeGet, mlirNoneTypeGetTypeID};

use crate::{Context, Type, TypeId, mlir_subtype_trait_impls};

/// Built-in MLIR [`Type`] that represents a unit type (i.e., a type with exactly one
/// possible value where its value does not have a defined dynamic representation). Refer to the
/// [MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#nonetype) for more information.
#[derive(Copy, Clone)]
pub struct NoneTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> NoneTypeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`NoneTypeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirNoneTypeGetTypeID()).unwrap() }
    }
}

mlir_subtype_trait_impls!(NoneTypeRef<'c, 't> as Type, mlir_type = Type, mlir_subtype = None);

impl<'t> Context<'t> {
    /// Creates a new [`NoneTypeRef`] owned by this [`Context`].
    pub fn none_type<'c>(&'c self) -> NoneTypeRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe { NoneTypeRef::from_c_api(mlirNoneTypeGet(*self.handle.borrow()), &self).unwrap() }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::types::tests::test_type_display_and_debug;

    use super::*;

    #[test]
    fn test_none_type_type_id() {
        let context = Context::new();
        let none_type_id = NoneTypeRef::type_id();
        let none_type_1 = context.none_type();
        let none_type_2 = context.none_type();
        assert_eq!(none_type_1.type_id(), none_type_2.type_id());
        assert_eq!(none_type_id, none_type_1.type_id());
    }

    #[test]
    fn test_none_type() {
        let context = Context::new();
        let r#type = context.none_type();
        assert_eq!(&context, r#type.context());
    }

    #[test]
    fn test_none_type_equality() {
        let context = Context::new();

        // Same types from the same context must be equal because they are "uniqued".
        let type_1 = context.none_type();
        let type_2 = context.none_type();
        assert_eq!(type_1, type_2);

        // Same types from different contexts must not be equal.
        let context = Context::new();
        let type_2 = context.none_type();
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_none_type_display_and_debug() {
        let context = Context::new();
        let r#type = context.none_type();
        test_type_display_and_debug(r#type, "none");
    }

    #[test]
    fn test_none_type_parsing() {
        let context = Context::new();
        assert_eq!(context.parse_type("none").unwrap(), context.none_type());
    }

    #[test]
    fn test_none_type_casting() {
        let context = Context::new();
        let r#type = context.none_type();
        let rendered_type = r#type.to_string();

        // Test upcasting.
        let r#type = r#type.as_type_ref();
        assert!(r#type.is::<NoneTypeRef>());
        assert_eq!(r#type.to_string(), rendered_type);

        // Test downcasting.
        let r#type = r#type.cast::<NoneTypeRef>().unwrap();
        assert!(r#type.is::<NoneTypeRef>());
        assert_eq!(r#type.to_string(), rendered_type);

        // Invalid cast from specific type.
        let r#type = context.index_type();
        assert!(!r#type.is::<NoneTypeRef>());
        assert_eq!(r#type.cast::<NoneTypeRef>(), None);

        // Invalid cast from a generic type reference.
        let r#type = r#type.as_type_ref();
        assert!(!r#type.is::<NoneTypeRef>());
        assert_eq!(r#type.cast::<NoneTypeRef>(), None);
    }
}
