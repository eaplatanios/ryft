use std::path::{Path, PathBuf};
use std::{env, fs, io};

use proc_macro2::TokenStream;
use quote::{ToTokens, quote, quote_spanned};
use syn::parse::{Parse, ParseStream};
use syn::spanned::Spanned;

/// Mermaid bundle that [`DiagramRenderer::install_bundle`] copies into the rustdoc output directory.
const BUNDLE: &[u8] = include_bytes!("../static/mermaid.min.js");

/// Directory, relative to the rustdoc output directory, into which [`DiagramRenderer::install_bundle`] writes
/// [`BUNDLE`].
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
/// element and older versions as an element with that identifier. The diagrams use the dark Mermaid theme when the page
/// uses one of the dark rustdoc themes (i.e., `dark` or `ayu`, which rustdoc records in the `data-theme` attribute of
/// the root element), or when the page uses no other known rustdoc theme and the browser prefers a dark color scheme.
/// The theme is selected once, when the diagrams are rendered.
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
  const theme = document.documentElement.dataset.theme;
  const dark = ['dark', 'ayu'].includes(theme)
    || (theme !== 'light' && window.matchMedia('(prefers-color-scheme: dark)').matches);
  mermaid.initialize({ startOnLoad: false, theme: dark ? 'dark' : 'default' });
  await mermaid.run({ querySelector: 'pre.mermaid' });
})();
</script>"#;

/// Renderer for the `#[mermaid]` attribute, which rewrites the Mermaid code fences in the documentation of the
/// annotated item into HTML blocks that Mermaid renders in the browser. The renderer processes the documentation lines
/// of the item in order, tracking the code block that encloses each line. Refer to the documentation of the `mermaid`
/// attribute in the crate root for information on how to use it.
pub(crate) struct DiagramRenderer {
    /// Block that encloses the next documentation line of the annotated item.
    block: EnclosingBlock,

    /// Indicates whether the documentation that has been rendered so far contains a diagram, in which case the loader
    /// script has already been inserted before the first one.
    has_diagrams: bool,
}

impl DiagramRenderer {
    /// Renders the Mermaid diagrams in the documentation of the provided item. Every ```` ```mermaid ```` (or
    /// `~~~mermaid`) code fence is rewritten into a `<pre class="mermaid">` HTML block, the loader script is inserted
    /// before the first diagram, and the Mermaid bundle is written into the rustdoc output directory. Items without
    /// diagrams are returned unchanged. If any errors are encountered, the item is also emitted unchanged, followed by
    /// a [`compile_error!`] invocation, so that its uses elsewhere do not report additional errors about a missing
    /// item.
    ///
    /// For example, the following documentation lines:
    ///
    /// ````text
    /// ```mermaid
    /// flowchart LR
    ///   source --> target
    /// ```
    /// ````
    ///
    /// are rendered as the following lines, where the body of the loader script is elided:
    ///
    /// ```text
    /// <script type="module">
    /// ...
    /// </script>
    /// <pre class="mermaid">
    /// flowchart LR
    ///   source --&gt; target
    /// </pre>
    /// ```
    pub(crate) fn render_diagrams(attributes: TokenStream, item: TokenStream) -> TokenStream {
        Self::new().render_item(attributes, item.clone()).unwrap_or_else(|error| {
            let error = error.into_compile_error();
            quote!(#item #error)
        })
    }

    /// Creates a new [`DiagramRenderer`] that has not rendered any documentation lines yet.
    fn new() -> Self {
        Self { block: EnclosingBlock::None, has_diagrams: false }
    }

    /// Renders the Mermaid diagrams in the documentation of the provided item, returning the rendered item or the
    /// first error that is encountered. Refer to [`DiagramRenderer::render_diagrams`] for more information.
    fn render_item(&mut self, attributes: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
        if !attributes.is_empty() {
            return Err(syn::Error::new_spanned(attributes, "the `mermaid` attribute does not accept arguments"));
        }

        let AnnotatedItem { attributes, rest } = syn::parse2(item.clone())?;
        let mut rendered = TokenStream::new();
        for attribute in attributes {
            match self.render_attribute(&attribute)? {
                // Rewritten attributes keep the span of the original attribute, and all other attributes are forwarded
                // unchanged, so that lints such as broken intra-doc links still report the corresponding source lines.
                Some(documentation) => {
                    let span = attribute.span();
                    let documentation = syn::LitStr::new(&documentation, span);
                    rendered.extend(quote_spanned!(span=> #[doc = #documentation]));
                }
                None => attribute.to_tokens(&mut rendered),
            }
        }

        if let EnclosingBlock::Diagram { attribute, .. } = &self.block {
            return Err(syn::Error::new_spanned(attribute, "unterminated `mermaid` code fence"));
        }

        if !self.has_diagrams {
            return Ok(item);
        }

        // Documentation builds must not fail when the bundle cannot be written (e.g., because the target directory is
        // read only), because the loader then falls back to the CDN.
        let target = env::var_os("CARGO_TARGET_DIR").map_or_else(|| PathBuf::from("target"), PathBuf::from);
        Self::install_bundle(&target).unwrap_or_default();
        rendered.extend(rest);
        Ok(rendered)
    }

    /// Renders the documentation of the provided attribute, returning [`None`] when the attribute stays unchanged
    /// (i.e., when it is not a literal documentation attribute or when none of its documentation lines is rewritten).
    fn render_attribute(&mut self, attribute: &syn::Attribute) -> syn::Result<Option<String>> {
        let documentation = if let syn::Meta::NameValue(syn::MetaNameValue { path, value, .. }) = &attribute.meta
            && path.is_ident("doc")
            && let syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Str(literal), .. }) = value
        {
            literal.value()
        } else if matches!(self.block, EnclosingBlock::Diagram { .. }) {
            return Err(syn::Error::new_spanned(
                attribute,
                "only literal documentation attributes may appear inside a `mermaid` diagram",
            ));
        } else {
            return Ok(None);
        };
        let rendered = documentation
            .split('\n')
            .map(|line| self.render_line(line, attribute))
            .collect::<syn::Result<Vec<_>>>()?
            .join("\n");
        Ok((rendered != documentation).then_some(rendered))
    }

    /// Renders one documentation line of the provided attribute and advances [`DiagramRenderer::block`] past it.
    /// The returned text spans multiple lines when it includes the loader script.
    fn render_line(&mut self, line: &str, attribute: &syn::Attribute) -> syn::Result<String> {
        match (&self.block, CodeFence::parse(line)) {
            (EnclosingBlock::None, Some((opening, "mermaid"))) => {
                let indentation = &line[..line.len() - line.trim_start().len()];
                let loader = if self.has_diagrams { String::new() } else { Self::loader(indentation) + "\n" };
                self.block = EnclosingBlock::Diagram {
                    opening,
                    attribute: attribute.to_token_stream(),
                    indentation: indentation.to_string(),
                };
                self.has_diagrams = true;
                Ok(format!("{loader}{indentation}<pre class=\"mermaid\">"))
            }
            (EnclosingBlock::None, Some((opening, _))) => {
                self.block = EnclosingBlock::Code { opening };
                Ok(line.to_string())
            }
            (EnclosingBlock::Code { opening }, Some((closing, ""))) if closing.closes(*opening) => {
                self.block = EnclosingBlock::None;
                Ok(line.to_string())
            }
            (EnclosingBlock::None | EnclosingBlock::Code { .. }, _) => Ok(line.to_string()),
            (EnclosingBlock::Diagram { opening, indentation, .. }, Some((closing, ""))) if closing.closes(*opening) => {
                let line = format!("{indentation}</pre>");
                self.block = EnclosingBlock::None;
                Ok(line)
            }
            (EnclosingBlock::Diagram { .. }, Some(_)) => {
                Err(syn::Error::new_spanned(attribute, "code fences cannot be nested inside a `mermaid` diagram"))
            }

            // The browser decodes the character references of each diagram, so escaping `<` and `>` keeps them from
            // opening HTML tags and passes them to Mermaid literally, while references that the diagram spells out
            // (e.g., `&lt;code&gt;` for an HTML label) reach Mermaid decoded.
            (EnclosingBlock::Diagram { .. }, None) => Ok(line.replace('<', "&lt;").replace('>', "&gt;")),
        }
    }

    /// Returns [`LOADER`] with its bundle URLs filled in and each of its lines prefixed by `indentation`.
    fn loader(indentation: &str) -> String {
        LOADER
            .replace("{local}", &format!("{BUNDLE_DIRECTORY}/{BUNDLE_FILE}"))
            .replace("{remote}", BUNDLE_CDN_URL)
            .lines()
            .map(|line| format!("{indentation}{line}"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Writes [`BUNDLE`] to `<target>/doc/static.files.mermaid/mermaid.min.js`, where `<target>` is the provided
    /// target directory (i.e., `$CARGO_TARGET_DIR` or, when that variable is unset, `./target` relative to the working
    /// directory of the compiler). The bundle is only written when the rustdoc output directory `<target>/doc` already
    /// exists (i.e., during documentation builds) and the file is missing or its size differs from that of [`BUNDLE`].
    fn install_bundle(target: &Path) -> io::Result<()> {
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
}

/// Item annotated with `#[mermaid]`, split into its outer attributes and its remaining tokens.
struct AnnotatedItem {
    /// Outer attributes of the item, which include its documentation comments.
    attributes: Vec<syn::Attribute>,

    /// Tokens of the item that follow its outer attributes.
    rest: TokenStream,
}

impl Parse for AnnotatedItem {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(Self { attributes: input.call(syn::Attribute::parse_outer)?, rest: input.parse()? })
    }
}

/// Block of Markdown documentation that encloses a documentation line.
enum EnclosingBlock {
    /// The line is not inside a fenced code block.
    None,

    /// The line is inside a fenced code block that `opening` opened and whose lines are preserved verbatim.
    Code { opening: CodeFence },

    /// The line is inside a Mermaid diagram that `opening` opened in the attribute whose tokens are `attribute`
    /// (kept for diagnostics), and the HTML lines that are generated for that diagram start with `indentation`.
    Diagram { opening: CodeFence, attribute: TokenStream, indentation: String },
}

/// Line that opens or closes a fenced Markdown code block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CodeFence {
    /// Character that the fence consists of (i.e., a backtick or a tilde).
    marker: char,

    /// Number of repetitions of [`CodeFence::marker`] that the fence starts with, which is at least three.
    length: usize,
}

impl CodeFence {
    /// Parses `line` as a code fence line, returning the fence along with its trimmed info string (e.g., `mermaid`), or
    /// [`None`] if `line` is not a code fence line. Fence lines are recognized regardless of their indentation, because
    /// rustdoc removes the indentation that all documentation lines share only after this attribute is expanded and
    /// because fences inside list items are indented further. As in CommonMark, the info string of a backtick fence
    /// cannot contain backticks.
    fn parse(line: &str) -> Option<(Self, &str)> {
        let line = line.trim();
        let marker = line.chars().next().filter(|marker| matches!(marker, '`' | '~'))?;
        let length = line.len() - line.trim_start_matches(marker).len();
        let info = line[length..].trim();
        (length >= 3 && !(marker == '`' && info.contains('`'))).then_some((Self { marker, length }, info))
    }

    /// Returns `true` if this fence closes the code block that `opening` opened, which requires the same marker and at
    /// least as many repetitions of it. The info string of a closing fence must also be empty, which callers check on
    /// the result of [`CodeFence::parse`].
    fn closes(self, opening: Self) -> bool {
        self.marker == opening.marker && self.length >= opening.length
    }
}

#[cfg(test)]
mod tests {
    use std::process;

    use indoc::formatdoc;
    use pretty_assertions::assert_eq;

    use super::*;

    /// Renders the diagrams of `item` and returns the result with one line per documentation line, followed by the
    /// remaining attributes and item tokens.
    fn render(item: TokenStream) -> String {
        let rendered = DiagramRenderer::render_diagrams(TokenStream::new(), item);
        let AnnotatedItem { attributes, rest } = syn::parse2(rendered).unwrap();
        attributes
            .iter()
            .map(|attribute| match &attribute.meta {
                syn::Meta::NameValue(syn::MetaNameValue {
                    value: syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Str(literal), .. }),
                    ..
                }) => literal.value(),
                _ => attribute.to_token_stream().to_string(),
            })
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

    /// Renders the diagrams of `item` with the provided attribute arguments and returns the resulting error message.
    fn error(attributes: TokenStream, item: TokenStream) -> String {
        DiagramRenderer::new().render_item(attributes, item).unwrap_err().to_string()
    }

    #[test]
    fn test_diagram_renderer_render_diagrams() {
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
    fn test_diagram_renderer_render_diagrams_with_multiple_diagrams() {
        // The loader is inserted once per item, before its first diagram.
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
    fn test_diagram_renderer_render_diagrams_with_other_code_blocks() {
        // Other code blocks, including one whose content looks like a Mermaid fence, are preserved verbatim,
        // and diagrams may also be fenced with tildes and closed by longer fences.
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
            /// ~~~~
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
    fn test_diagram_renderer_render_diagrams_with_multiline_documentation_attributes() {
        // The lines of multiline documentation attributes are rendered individually, and the generated lines take the
        // indentation of the opening fence.
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
    fn test_diagram_renderer_render_diagrams_escaping() {
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
    fn test_diagram_renderer_render_diagrams_without_diagrams() {
        // Mermaid fences inside other code blocks, including blocks that are never closed, are not diagrams.
        let item = quote! {
            /// Documentation with a code block.
            /// ```rust
            /// let x = 1;
            /// ```
            /// ```text
            /// ```mermaid
            /// flowchart LR
            #[derive(Debug)]
            struct Graph;
        };
        assert_eq!(DiagramRenderer::render_diagrams(TokenStream::new(), item.clone()).to_string(), item.to_string());
    }

    #[test]
    fn test_diagram_renderer_render_diagrams_errors() {
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
        assert_eq!(
            error(
                quote!(),
                quote! {
                    /// ```mermaid
                    #[doc = concat!("flowchart LR")]
                    /// ```
                    struct Graph;
                },
            ),
            "only literal documentation attributes may appear inside a `mermaid` diagram",
        );

        // The item is still emitted unchanged when an error is reported.
        let item = quote! {
            /// ```mermaid
            struct Graph;
        };
        assert_eq!(
            DiagramRenderer::render_diagrams(TokenStream::new(), item.clone()).to_string(),
            quote!(#item ::core::compile_error! { "unterminated `mermaid` code fence" }).to_string(),
        );
    }

    #[test]
    fn test_diagram_renderer_install_bundle() {
        let target = env::temp_dir().join(format!("ryft-macros-install-bundle-{}", process::id()));
        let path = target.join("doc").join(BUNDLE_DIRECTORY).join(BUNDLE_FILE);
        if target.exists() {
            fs::remove_dir_all(&target).unwrap();
        }
        fs::create_dir(&target).unwrap();

        // Nothing is written when there is no rustdoc output directory.
        DiagramRenderer::install_bundle(&target).unwrap();
        assert!(!target.join("doc").exists());

        // The bundle is written into an existing rustdoc output directory.
        fs::create_dir(target.join("doc")).unwrap();
        DiagramRenderer::install_bundle(&target).unwrap();
        assert_eq!(fs::read(&path).unwrap(), BUNDLE);

        // An existing file is only replaced when its size differs from that of the bundle.
        let placeholder = vec![0u8; BUNDLE.len()];
        fs::write(&path, &placeholder).unwrap();
        DiagramRenderer::install_bundle(&target).unwrap();
        assert_eq!(fs::read(&path).unwrap(), placeholder);
        fs::write(&path, "stale").unwrap();
        DiagramRenderer::install_bundle(&target).unwrap();
        assert_eq!(fs::read(&path).unwrap(), BUNDLE);

        fs::remove_dir_all(&target).unwrap();
    }

    #[test]
    fn test_code_fence_parse() {
        assert_eq!(CodeFence::parse("```mermaid"), Some((CodeFence { marker: '`', length: 3 }, "mermaid")));
        assert_eq!(
            CodeFence::parse("  ~~~~ rust,ignore  "),
            Some((CodeFence { marker: '~', length: 4 }, "rust,ignore"))
        );
        assert_eq!(CodeFence::parse("```"), Some((CodeFence { marker: '`', length: 3 }, "")));

        // Unlike tilde fences, backtick fences cannot have backticks in their info strings.
        assert_eq!(CodeFence::parse("~~~a`b"), Some((CodeFence { marker: '~', length: 3 }, "a`b")));
        assert_eq!(CodeFence::parse("```a`b"), None);
        assert_eq!(CodeFence::parse("``mermaid"), None);
        assert_eq!(CodeFence::parse("text ```"), None);
        assert_eq!(CodeFence::parse(""), None);
    }

    #[test]
    fn test_code_fence_closes() {
        let opening = CodeFence { marker: '`', length: 4 };
        assert!(CodeFence { marker: '`', length: 4 }.closes(opening));
        assert!(CodeFence { marker: '`', length: 5 }.closes(opening));
        assert!(!CodeFence { marker: '`', length: 3 }.closes(opening));
        assert!(!CodeFence { marker: '~', length: 4 }.closes(opening));
    }
}
