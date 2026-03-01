use proc_macro2::TokenStream;
use quote::quote;

/// Wraps the provided [`TokenStream`] with a `const _: () = { ... }` block, returning a new [`TokenStream`].
///
/// This is meant to make our macros [hygienic](https://en.wikipedia.org/wiki/Hygienic_macro) by making sure that
/// any symbols that we generate do not pollute the scope in which the macro is invoked. Note that if the wrapped
/// code contains any `impl` blocks, then the functions defined in those blocks will be made available to the outer
/// scope for the relevant types.
///
/// Note also that you can use this function to wrap code that performs compile-time checks and which you do not
/// want to keep around during runtime. That is because the whole `const` block will be executed by the compiler and
/// the result is always `()` and thus will be thrown away (except for any functions defined in `impl` blocks, as
/// discussed earlier).
///
/// # Parameters
///
///   * `ryft_path` - Optional [`syn::Path`] specifying the path in which the `ryft` library should be imported from.
///     This does not generally need to be provided as `ryft` will always be used by default. However, you may want to
///     use this if you have wrapped `ryft` in some other library and would like to import it from there instead. This
///     function will include a `use ryft as _ryft;` expression in the beginning of the wrapped block if you do not
///     provide a custom `ryft_path`. If you do, `ryft` in this expression will be replaced with the provided path.
///   * `code` - [`TokenStream`] to wrap in a `const _: () = { ... }` block.
pub fn const_block(code: TokenStream) -> TokenStream {
    quote! {
        #[doc(hidden)]
        #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
        const _: () = {
            #code
        };
    }
}

#[cfg(test)]
mod tests {
    use proc_macro2::TokenStream;
    use quote::{ToTokens, quote};

    use super::const_block;

    #[test]
    fn test_const_block() {
        // Test with an empty const block.
        let block = const_block(TokenStream::new());
        let item_const: syn::ItemConst = syn::parse2(block).expect("failed to parse empty const block");
        let expr = match &*item_const.expr {
            syn::Expr::Block(expr) => expr,
            other => panic!("expected a block expression but found: {other:?}"),
        };
        assert!(expr.block.stmts.is_empty());

        // Test with a non-empty const block.
        let block = const_block(quote! {
            type Alias = u32;
            fn helper() -> u32 {
                7
            }
        });
        let item_const: syn::ItemConst = syn::parse2(block).expect("failed to parse const block");
        assert_eq!(item_const.ident.to_string(), "_");
        assert!(
            item_const
                .attrs
                .iter()
                .any(|attr| attr.path().is_ident("doc") && attr.to_token_stream().to_string().contains("hidden"))
        );
        assert!(item_const.attrs.iter().any(|attr| attr.path().is_ident("allow")));
        let expr = match &*item_const.expr {
            syn::Expr::Block(expr) => expr,
            other => panic!("expected a block expression but found: {other:?}"),
        };
        assert_eq!(expr.block.stmts.len(), 2);
    }
}
