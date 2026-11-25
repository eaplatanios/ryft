//! Tokenizers, which convert between text and the token identifiers that language models consume and produce.
//!
//! A [`Tokenizer`] owns a vocabulary that assigns a [`TokenId`] to every token string. [`Tokenizer::encode`] splits
//! text into tokens and [`Tokenizer::decode`] concatenates the strings of tokens back into text, so decoding the
//! encoding of a text returns that text. Some tokens are _special_ (e.g., the tokens that delimit the messages of a
//! conversation). Their strings never come out of encoding ordinary text, and [`SpecialTokenHandling`] selects
//! whether occurrences of those strings in the input of [`Tokenizer::encode`] become the special tokens or are
//! treated as ordinary text. Renderers (i.e., [`Renderer`](crate::renderers::Renderer) implementations) are generic
//! over this trait, which keeps them independent of the vocabulary file format.
//!
//! [`HuggingFaceTokenizer`] is the implementation for the `tokenizer.json` files of Hugging Face model releases. It
//! reads them natively and supports the byte level BPE configuration that those releases commonly use, as described in
//! its documentation, and it rejects every other configuration with a [`TokenizerError`] that names the unsupported
//! component.
//!
//! # Release Parity Tests
//!
//! The unit tests of this module use small synthetic tokenizer definitions. An additional ignored test checks
//! [`HuggingFaceTokenizer`] against token sequences that the reference Hugging Face `tokenizers` library produced for
//! the Inkling 2 release. It reads `tokenizer.json` from the directory named by the
//! `RYFT_MODELS_INKLING2_RELEASE_DIRECTORY` environment variable and runs with `cargo test --release -- --ignored`.

use thiserror::Error;

pub mod hugging_face;

pub use hugging_face::HuggingFaceTokenizer;

// TODO(eaplatanios): Review this module.

/// Error produced when loading a tokenizer definition, encoding text, or decoding tokens.
#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum TokenizerError {
    #[error("failed to read the tokenizer definition: {message}")]
    Io { message: String },

    #[error("invalid tokenizer definition: {message}")]
    InvalidDefinition { message: String },

    #[error("unsupported tokenizer definition: {message}")]
    UnsupportedDefinition { message: String },

    #[error("failed to encode text: {message}")]
    Encoding { message: String },

    #[error("failed to decode tokens: {message}")]
    Decoding { message: String },
}

/// Identifier of a token in a tokenizer vocabulary.
pub type TokenId = u32;

/// Treatment of special token strings that occur in the text passed to [`Tokenizer::encode`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SpecialTokenHandling {
    /// Occurrences of special token strings in the text become their special token identifiers.
    Parse,

    /// The text is ordinary text, so special token strings in it are tokenized like any other characters.
    Literal,
}

/// Bidirectional mapping between text and sequences of [`TokenId`]s, backed by a vocabulary of token strings.
pub trait Tokenizer {
    /// Encodes `text` into a sequence of token identifiers.
    ///
    /// # Parameters
    ///
    ///   - `text`: Text to encode.
    ///   - `special_tokens`: Whether occurrences of special token strings in `text` become special tokens (i.e.,
    ///     [`SpecialTokenHandling::Parse`]) or are encoded as ordinary text (i.e., [`SpecialTokenHandling::Literal`]).
    fn encode(&self, text: &str, special_tokens: SpecialTokenHandling) -> Result<Vec<TokenId>, TokenizerError>;

    /// Decodes `tokens` into text. Decoding fails for identifiers that are not in the vocabulary.
    ///
    /// # Parameters
    ///
    ///   - `tokens`: Token identifiers to decode.
    ///   - `skip_special_tokens`: Whether special tokens are dropped instead of contributing their strings.
    fn decode(&self, tokens: &[TokenId], skip_special_tokens: bool) -> Result<String, TokenizerError>;

    /// Returns the number of token identifiers of this tokenizer (i.e., its largest identifier plus one).
    fn vocabulary_size(&self) -> usize;

    /// Returns the identifier of the token whose vocabulary string is `token`, including special tokens, or [`None`]
    /// if the vocabulary has no such token.
    fn token_id(&self, token: &str) -> Option<TokenId>;

    /// Returns the vocabulary string of the token with identifier `id`, including special tokens, or [`None`] if the
    /// vocabulary has no such token.
    fn token(&self, id: TokenId) -> Option<&str>;
}
