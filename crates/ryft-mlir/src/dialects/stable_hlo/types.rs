use ryft_xla_sys::bindings::{MlirType, stablehloTokenTypeGet};

use crate::{Context, DialectHandle, Type, mlir_subtype_trait_impls};

/// StableHLO token [`Type`]. In StableHLO, [`TokenTypeRef`]s represent tokens (i.e., opaque values produced and
/// consumed by some operations). Tokens are used for imposing execution order on operations as described in the
/// [Execution](https://openxla.org/stablehlo/spec#execution) section of the official StableHLO documentation.
#[derive(Copy, Clone)]
pub struct TokenTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

mlir_subtype_trait_impls!(
    TokenTypeRef<'c, 't> as Type,
    mlir_type = Type,
    mlir_subtype = Token,
    mlir_prefix = stablehlo,
);

impl<'t> Context<'t> {
    /// Creates a new StableHLO [`TokenTypeRef`] owned by this [`Context`].
    pub fn stable_hlo_token_type<'c>(&'c self) -> TokenTypeRef<'c, 't> {
        // Make sure that the StableHLO dialect is loaded into the current context to prevent segmentation faults.
        self.load_dialect(DialectHandle::stable_hlo());
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe { TokenTypeRef::from_c_api(stablehloTokenTypeGet(*self.handle.borrow()), &self).unwrap() }
    }
}

#[cfg(test)]
mod tests {
    use crate::Type;
    use crate::types::tests::{test_type_casting, test_type_display_and_debug};

    use super::*;

    #[test]
    fn test_token_type() {
        let context = Context::new();
        let token_type = context.stable_hlo_token_type();
        assert_eq!(&context, token_type.context());
        assert_eq!(token_type.dialect().namespace().unwrap(), "stablehlo");
    }

    #[test]
    fn test_token_type_equality() {
        let context = Context::new();

        // Token types from the same context must be equal because they are "uniqued".
        let token_type_1 = context.stable_hlo_token_type();
        let token_type_2 = context.stable_hlo_token_type();
        assert_eq!(token_type_1, token_type_2);

        // Token types from different contexts must not be equal.
        let context = Context::new();
        let token_type_2 = context.stable_hlo_token_type();
        assert_ne!(token_type_1, token_type_2);
    }

    #[test]
    fn test_token_type_display_and_debug() {
        let context = Context::new();
        let token_type = context.stable_hlo_token_type();
        test_type_display_and_debug(token_type, "!stablehlo.token");
    }

    #[test]
    fn test_token_type_parsing() {
        let context = Context::new();
        let token_type = context.stable_hlo_token_type();
        assert_eq!(context.parse_type("!stablehlo.token").unwrap(), token_type);
    }

    #[test]
    fn test_token_type_casting() {
        let context = Context::new();
        let token_type = context.stable_hlo_token_type();
        test_type_casting(token_type);
    }
}
