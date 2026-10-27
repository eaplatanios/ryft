//! Expansion of the `#[mermaid]` attribute, which renders the Mermaid diagrams of rustdoc comments in the browser.

use std::path::PathBuf;
use std::{env, fs, io};

use proc_macro2::TokenStream;
use quote::{ToTokens, quote_spanned};
use syn::parse::{Parse, ParseStream};
use syn::spanned::Spanned;
use syn::{Attribute, Expr, ExprLit, Lit, LitStr, Meta};

/// Mermaid bundle that [`install_bundle`] copies into the rustdoc output directory.
const BUNDLE: &[u8] = include_bytes!("../static/mermaid.min.js");

/// Directory, relative to the rustdoc output directory, into which [`install_bundle`] writes [`BUNDLE`].
const BUNDLE_DIRECTORY: &str = "static.files.mermaid";

/// File name of [`BUNDLE`] within [`BUNDLE_DIRECTORY`].
const BUNDLE_FILE: &str = "mermaid.min.js";

/// URL from which the loader requests Mermaid when the local bundle cannot be loaded.
const BUNDLE_CDN_URL: &str = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js";

/// Script that is inserted before the first diagram of every annotated item and that loads Mermaid and renders the
/// `pre.mermaid` blocks of the page, where `{local}` stands for the bundle path relative to the rustdoc root path and
/// `{remote}` stands for [`BUNDLE_CDN_URL`].
///
/// The bundle is a UMD script rather than an ES module because browsers block module imports from `file://` pages, so
/// the loader injects a `script` element and awaits it instead of importing the bundle. The whole procedure is memoized
/// in a page-global promise, so that pages documenting several annotated items load the bundle and render only once.
/// The rustdoc root path is read from the `rustdoc-vars` element, which current rustdoc versions emit as a `meta`
/// element and older versions as an element with that identifier.
const LOADER: &str = r#"<script type="module">
window.ryftMermaid ??= (async () => {
  const load = (source) => new Promise((resolve, reject) => {
    document.head.append(Object.assign(document.createElement('script'), {
      src: source,
      onload: resolve,
      onerror: reject,
    }));
  });
  const variables = document.querySelector('meta[name="rustdoc-vars"], #rustdoc-vars');
  const root = variables?.dataset.rootPath ?? './';
  try {
    await load(root + '{local}').catch(() => load('{remote}'));
  } catch {
    for (const diagram of document.querySelectorAll('pre.mermaid')) {
      diagram.textContent = 'This diagram could not be rendered because Mermaid failed to load (locally built '
        + 'documentation needs cargo doc to have written its bundle, or network access).';
    }
    return;
  }
  const dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  mermaid.initialize({ startOnLoad: false, theme: dark ? 'dark' : 'default' });
  await mermaid.run({ querySelector: 'pre.mermaid' });
})();
</script>"#;

/// Item annotated with `#[mermaid]`, split into its outer attributes and its remaining tokens.
struct AnnotatedItem {
    /// Outer attributes of the item, which include its documentation comments.
    attributes: Vec<Attribute>,

    /// Tokens of the item that follow its outer attributes.
    rest: TokenStream,
}

impl Parse for AnnotatedItem {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(Self { attributes: input.call(Attribute::parse_outer)?, rest: input.parse()? })
    }
}

/// Code fence that encloses a documentation line.
enum Fence {
    /// The line is not inside a code fence.
    None,

    /// The line is inside a non-Mermaid code fence opened by `length` repetitions of `marker`, whose lines are
    /// preserved verbatim.
    Other { marker: char, length: usize },

    /// The line is inside a Mermaid diagram opened by `length` repetitions of `marker` in the attribute whose tokens
    /// are `opening` (kept for diagnostics), and the HTML lines generated for that diagram start with `indentation`.
    Mermaid { marker: char, length: usize, opening: TokenStream, indentation: String },
}

/// Expands one `#[mermaid]` attribute: rewrites every Mermaid code fence in the documentation of `item` into a
/// `<pre class="mermaid">` block, inserts the loader script before the first diagram, and writes the Mermaid bundle
/// into the rustdoc output directory. Items without Mermaid code fences are returned unchanged.
pub(crate) fn expand(attributes: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
    if !attributes.is_empty() {
        return Err(syn::Error::new_spanned(attributes, "the `mermaid` attribute does not accept arguments"));
    }
    let AnnotatedItem { attributes, rest } = syn::parse2(item.clone())?;
    let mut expansion = TokenStream::new();
    let mut fence = Fence::None;
    let mut has_diagrams = false;
    for attribute in attributes {
        let Some(documentation) = documentation(&attribute) else {
            if matches!(fence, Fence::Mermaid { .. }) {
                return Err(syn::Error::new_spanned(
                    attribute,
                    "only literal documentation attributes may appear inside a `mermaid` diagram",
                ));
            }
            attribute.to_tokens(&mut expansion);
            continue;
        };

        // Attributes that neither belong to nor delimit a diagram are forwarded unchanged, which preserves their spans
        // and doc-comment form. The others are split into one documentation attribute per line.
        let mut lines = Vec::new();
        let mut rewritten = false;
        for line in documentation.split('\n') {
            match (&fence, code_fence(line)) {
                (Fence::None, Some((marker, length, "mermaid"))) => {
                    let indentation = &line[..line.len() - line.trim_start().len()];
                    if !has_diagrams {
                        let loader = LOADER
                            .replace("{local}", &format!("{BUNDLE_DIRECTORY}/{BUNDLE_FILE}"))
                            .replace("{remote}", BUNDLE_CDN_URL);
                        lines.extend(loader.lines().map(|loader_line| format!("{indentation}{loader_line}")));
                    }
                    lines.push(format!("{indentation}<pre class=\"mermaid\">"));
                    fence = Fence::Mermaid {
                        marker,
                        length,
                        opening: attribute.to_token_stream(),
                        indentation: indentation.to_string(),
                    };
                    has_diagrams = true;
                    rewritten = true;
                }
                (Fence::None, Some((marker, length, _))) => {
                    lines.push(line.to_string());
                    fence = Fence::Other { marker, length };
                }
                (Fence::None, None) => lines.push(line.to_string()),
                (Fence::Other { marker, length }, Some((closing_marker, closing_length, "")))
                    if closing_marker == *marker && closing_length >= *length =>
                {
                    lines.push(line.to_string());
                    fence = Fence::None;
                }
                (Fence::Other { .. }, _) => lines.push(line.to_string()),
                (Fence::Mermaid { marker, length, indentation, .. }, Some((closing_marker, closing_length, "")))
                    if closing_marker == *marker && closing_length >= *length =>
                {
                    lines.push(format!("{indentation}</pre>"));
                    fence = Fence::None;
                    rewritten = true;
                }
                (Fence::Mermaid { .. }, Some(_)) => {
                    return Err(syn::Error::new_spanned(
                        attribute,
                        "code fences cannot be nested inside a `mermaid` diagram",
                    ));
                }
                (Fence::Mermaid { .. }, None) => {
                    // The browser decodes the character references of each diagram, so escaping `<` and `>` keeps them
                    // from opening HTML tags and passes them to Mermaid literally, while references that the diagram
                    // spells out (e.g., `&lt;code&gt;` for an HTML label) reach Mermaid decoded.
                    lines.push(line.replace('<', "&lt;").replace('>', "&gt;"));
                    rewritten = true;
                }
            }
        }
        if rewritten {
            let span = attribute.span();
            for line in lines {
                let line = LitStr::new(&line, span);
                expansion.extend(quote_spanned!(span=> #[doc = #line]));
            }
        } else {
            attribute.to_tokens(&mut expansion);
        }
    }
    if let Fence::Mermaid { opening, .. } = fence {
        return Err(syn::Error::new_spanned(opening, "unterminated `mermaid` code fence"));
    }
    if !has_diagrams {
        return Ok(item);
    }

    // Documentation builds must not fail when the bundle cannot be written (e.g., because the target directory is read
    // only), because the loader then falls back to the CDN.
    install_bundle().unwrap_or_default();
    expansion.extend(rest);
    Ok(expansion)
}

/// Returns the text of `attribute` if it is a documentation attribute with a string literal value, which includes
/// `///` comments, and `None` otherwise.
fn documentation(attribute: &Attribute) -> Option<String> {
    match &attribute.meta {
        Meta::NameValue(meta) if meta.path.is_ident("doc") => match &meta.value {
            Expr::Lit(ExprLit { lit: Lit::Str(literal), .. }) => Some(literal.value()),
            _ => None,
        },
        _ => None,
    }
}

/// Parses `line` as a Markdown code fence line, returning its marker character (i.e., a backtick or a tilde), the
/// length of its marker run (at least three), and its trimmed info string, or `None` if `line` is not a fence line.
fn code_fence(line: &str) -> Option<(char, usize, &str)> {
    let line = line.trim();
    let marker = line.chars().next().filter(|marker| matches!(marker, '`' | '~'))?;
    let length = line.len() - line.trim_start_matches(marker).len();
    (length >= 3).then(|| (marker, length, line[length..].trim()))
}

/// Writes [`BUNDLE`] to `<target>/doc/static.files.mermaid/mermaid.min.js`, where `<target>` is `$CARGO_TARGET_DIR`
/// or, when that variable is unset, `./target` relative to the working directory of the compiler. The bundle is only
/// written when the rustdoc output directory `<target>/doc` already exists (i.e., during documentation builds) and the
/// file is missing or its size differs from that of [`BUNDLE`].
fn install_bundle() -> io::Result<()> {
    let target = env::var_os("CARGO_TARGET_DIR").map_or_else(|| PathBuf::from("target"), PathBuf::from);
    let documentation = target.join("doc");
    if !documentation.is_dir() {
        return Ok(());
    }
    let directory = documentation.join(BUNDLE_DIRECTORY);
    let path = directory.join(BUNDLE_FILE);
    if fs::metadata(&path).is_ok_and(|metadata| metadata.len() == BUNDLE.len() as u64) {
        return Ok(());
    }
    fs::create_dir_all(&directory)?;
    fs::write(path, BUNDLE)
}

#[cfg(test)]
mod tests {
    use indoc::formatdoc;
    use pretty_assertions::assert_eq;
    use quote::quote;

    use super::*;

    /// Expands `item` and renders the result with one line per documentation line, followed by the remaining
    /// attributes and item tokens.
    fn render(item: TokenStream) -> String {
        let AnnotatedItem { attributes, rest } = syn::parse2(expand(TokenStream::new(), item).unwrap()).unwrap();
        attributes
            .iter()
            .map(|attribute| documentation(attribute).unwrap_or_else(|| attribute.to_token_stream().to_string()))
            .chain([rest.to_string()])
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Returns the loader script with each line prefixed by `indentation`, using independently spelled bundle URLs.
    fn loader(indentation: &str) -> String {
        LOADER
            .replace("{local}", "static.files.mermaid/mermaid.min.js")
            .replace("{remote}", "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js")
            .lines()
            .map(|line| format!("{indentation}{line}"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Expands `item` with `attributes` and returns the rendered error message.
    fn error(attributes: TokenStream, item: TokenStream) -> String {
        expand(attributes, item).unwrap_err().to_string()
    }

    #[test]
    fn test_expand() {
        let item = quote! {
            /// Before.
            ///
            /// ```mermaid
            /// flowchart LR
            ///   source --> target
            /// ```
            ///
            /// After.
            #[derive(Debug)]
            struct Graph;
        };
        assert_eq!(
            render(item),
            formatdoc! {r#"
                 Before.

                {loader}
                 <pre class="mermaid">
                 flowchart LR
                   source --&gt; target
                 </pre>

                 After.
                # [derive (Debug)]
                struct Graph ;"#,
                loader = loader(" "),
            },
        );
    }

    #[test]
    fn test_expand_with_multiple_diagrams() {
        // The loader is emitted once per item, before its first diagram.
        let item = quote! {
            /// ```mermaid
            /// flowchart LR
            /// ```
            /// Between.
            /// ```mermaid
            /// flowchart TD
            /// ```
            struct Graph;
        };
        assert_eq!(
            render(item),
            formatdoc! {r#"
                {loader}
                 <pre class="mermaid">
                 flowchart LR
                 </pre>
                 Between.
                 <pre class="mermaid">
                 flowchart TD
                 </pre>
                struct Graph ;"#,
                loader = loader(" "),
            },
        );
    }

    #[test]
    fn test_expand_with_other_code_fences() {
        // Other code fences, including one whose content looks like a Mermaid fence, are preserved verbatim.
        let item = quote! {
            /// ````text
            /// ```mermaid
            /// ```
            /// ````
            /// ```rust
            /// let x = 1;
            /// ```
            /// ~~~mermaid
            /// flowchart LR
            /// ~~~
            struct Graph;
        };
        assert_eq!(
            render(item),
            formatdoc! {r#"
                 ````text
                 ```mermaid
                 ```
                 ````
                 ```rust
                 let x = 1;
                 ```
                {loader}
                 <pre class="mermaid">
                 flowchart LR
                 </pre>
                struct Graph ;"#,
                loader = loader(" "),
            },
        );
    }

    #[test]
    fn test_expand_with_multiline_documentation_attribute() {
        // Multiline documentation attributes are split into lines, and the generated lines take the fence indentation.
        let item = quote! {
            #[doc = "Before.\n  ```mermaid\n  flowchart LR\n  ```\nAfter."]
            struct Graph;
        };
        assert_eq!(
            render(item),
            formatdoc! {r#"
                Before.
                {loader}
                  <pre class="mermaid">
                  flowchart LR
                  </pre>
                After.
                struct Graph ;"#,
                loader = loader("  "),
            },
        );
    }

    #[test]
    fn test_expand_escaping() {
        // `<` and `>` are escaped, while character references that the diagram spells out are kept as written.
        let item = quote! {
            /// ```mermaid
            /// flowchart LR
            ///   map["&lt;code&gt;Map&lt;/code&gt;"] --> compare["x < y & y > z"]
            /// ```
            struct Graph;
        };
        assert_eq!(
            render(item),
            formatdoc! {r#"
                {loader}
                 <pre class="mermaid">
                 flowchart LR
                   map["&lt;code&gt;Map&lt;/code&gt;"] --&gt; compare["x &lt; y & y &gt; z"]
                 </pre>
                struct Graph ;"#,
                loader = loader(" "),
            },
        );
    }

    #[test]
    fn test_expand_without_diagrams() {
        let item = quote! {
            /// Documentation with a code fence.
            /// ```rust
            /// let x = 1;
            /// ```
            #[derive(Debug)]
            struct Graph;
        };
        assert_eq!(expand(TokenStream::new(), item.clone()).unwrap().to_string(), item.to_string());
    }

    #[test]
    fn test_expand_validation() {
        assert_eq!(
            error(
                quote!(theme = "dark"),
                quote!(
                    struct Graph;
                )
            ),
            "the `mermaid` attribute does not accept arguments",
        );
        assert_eq!(
            error(
                quote!(),
                quote! {
                    /// ```mermaid
                    /// flowchart LR
                    struct Graph;
                },
            ),
            "unterminated `mermaid` code fence",
        );
        assert_eq!(
            error(
                quote!(),
                quote! {
                    /// ```mermaid
                    /// ```text
                    /// ```
                    struct Graph;
                },
            ),
            "code fences cannot be nested inside a `mermaid` diagram",
        );
        assert_eq!(
            error(
                quote!(),
                quote! {
                    /// ```mermaid
                    #[derive(Debug)]
                    /// ```
                    struct Graph;
                },
            ),
            "only literal documentation attributes may appear inside a `mermaid` diagram",
        );
    }
}
