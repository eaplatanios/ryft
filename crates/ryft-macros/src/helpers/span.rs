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
