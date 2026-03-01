use proc_macro2::{Group, Span, TokenStream, TokenTree};

/// Constructs a new [`TokenStream`] matching the provided [`TokenStream`] but using the provided [`Span`] instead of
/// its current one. This is useful in procedural macros to ensure that generated error messages point to the correct
/// locations in the original source code when constructing copies of a specific [`TokenStream`] in different locations,
/// for example.
///
/// # Parameters
///
///   * `stream` - [`TokenStream`] to copy.
///   * `span` - [`Span`] to use for the new [`TokenStream`].
pub fn with_span(stream: TokenStream, span: Span) -> TokenStream {
    stream
        .into_iter()
        .map(|mut token| {
            if let TokenTree::Group(group) = &mut token {
                *group = Group::new(group.delimiter(), with_span(group.stream(), span));
            }
            token.set_span(span);
            token
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use proc_macro2::{Delimiter, Span, TokenStream, TokenTree};
    use quote::quote;

    use super::with_span;

    #[test]
    fn test_with_span() {
        // Test whether [`with_span`] preserves group structure.
        fn group_count(stream: TokenStream) -> usize {
            stream
                .into_iter()
                .map(|token| match token {
                    TokenTree::Group(group) => 1 + group_count(group.stream()),
                    _ => 0,
                })
                .sum()
        }

        let stream = quote!(outer(inner(a), { b(c) }, [d]));
        let transformed = with_span(stream.clone(), Span::call_site());
        assert_eq!(transformed.to_string(), stream.to_string());
        assert_eq!(group_count(transformed), group_count(stream));

        // Test whether [`with_span`] preserves group delimiters.
        let transformed = with_span(quote!((value)), Span::call_site());
        let token = transformed.into_iter().next().expect("expected one token");
        let group = match token {
            TokenTree::Group(group) => group,
            other => panic!("expected a group token but found: {other:?}"),
        };
        assert_eq!(group.delimiter(), Delimiter::Parenthesis);

        // Test whether [`with_span`] handles empty streams correctly.
        assert!(with_span(TokenStream::new(), Span::call_site()).is_empty());
    }
}
