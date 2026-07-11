use ryft_xla_sys::bindings::MlirType;
use ryft_xla_sys::mlir::dialects::builtin::{mlirTokenTypeGet, mlirTypeIsAToken};

use crate::{Context, Error, Type, mlir_subtype_trait_impls};

/// Built-in MLIR [`Type`] that represents a token (i.e., an opaque value that carries no runtime data and that is
/// produced and consumed by operations to establish ordering between them). Refer to the
/// [MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#tokentype) for more information.
#[derive(Copy, Clone)]
pub struct TokenTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> Type<'c, 't> for TokenTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Result<Self, Error> {
        if handle.ptr.is_null() {
            Err(Error::internal("expected non-null MLIR type handle"))
        } else if unsafe { mlirTypeIsAToken(handle) } {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected MLIR token type handle"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirType {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(TokenTypeRef<'c, 't> as Type, mlir_type = Type);

impl<'t> Context<'t> {
    /// Creates a new [`TokenTypeRef`] owned by this [`Context`].
    pub fn token_type<'c>(&'c self) -> TokenTypeRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        let handle = unsafe { mlirTokenTypeGet(*self.handle.borrow()) };
        TokenTypeRef { handle, context: self }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::types::tests::test_type_display_and_debug;

    use super::*;

    #[test]
    fn test_token_type() {
        let context = Context::new();
        let r#type = context.token_type();
        assert_eq!(&context, r#type.context());
    }

    #[test]
    fn test_token_type_equality() {
        let context = Context::new();

        // Same types from the same context must be equal because they are "uniqued".
        let type_1 = context.token_type();
        let type_2 = context.token_type();
        assert_eq!(type_1, type_2);

        // Same types from different contexts must not be equal.
        let context = Context::new();
        let type_2 = context.token_type();
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_token_type_display_and_debug() {
        let context = Context::new();
        let r#type = context.token_type();
        test_type_display_and_debug(r#type, "token");
    }

    #[test]
    fn test_token_type_parsing() {
        let context = Context::new();
        assert_eq!(context.parse_type("token").unwrap(), context.token_type());
    }

    #[test]
    fn test_token_type_casting() {
        let context = Context::new();
        let r#type = context.token_type();
        let rendered_type = r#type.to_string();

        // Test upcasting.
        let r#type = r#type.as_ref();
        assert!(r#type.is::<TokenTypeRef>());
        assert_eq!(r#type.to_string(), rendered_type);

        // Test downcasting.
        let r#type = r#type.cast::<TokenTypeRef>().unwrap();
        assert!(r#type.is::<TokenTypeRef>());
        assert_eq!(r#type.to_string(), rendered_type);

        // Invalid cast from specific type.
        let r#type = context.index_type();
        assert!(!r#type.is::<TokenTypeRef>());
        assert_eq!(r#type.cast::<TokenTypeRef>(), None);

        // Invalid cast from a generic type reference.
        let r#type = r#type.as_ref();
        assert!(!r#type.is::<TokenTypeRef>());
        assert_eq!(r#type.cast::<TokenTypeRef>(), None);
    }
}
