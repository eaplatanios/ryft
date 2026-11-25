//! Model definitions for Ryft, together with the components that surround a model, namely checkpoint readers and
//! importers, host codecs for packed low precision weights, tokenizers, and conversation renderers.
//!
//! Everything in this crate is written against the backend independent capabilities of `ryft-core`. A model is a
//! parameter structure (i.e., a type that implements [`Parameterized`](ryft_core::Parameterized)) whose functions are
//! generic over the [`ArrayOperations`](ryft_core::ArrayOperations) bundle, so the same definition is evaluated
//! eagerly on host arrays, differentiated, and compiled by a backend such as `ryft-xla`. This crate depends on no
//! backend.
//!
//! # Modules
//!
//!   - [`tokenizers`] converts between text and token identifiers ([`Tokenizer`], [`HuggingFaceTokenizer`]).
//!   - [`renderers`] converts between structured conversations and token sequences ([`Renderer`],
//!     [`Inkling2Renderer`]).
//!   - [`layers`] holds numerical building blocks that model definitions share.
//!   - [`experts`] abstracts the projections of routed experts ([`ExpertProjector`]), including the mixed precision
//!     training recipe of the Inkling 2 models ([`BlockScaledExpertProjector`]).
//!   - [`inkling2`] is the Inkling 2 model family ([`Inkling2Decoder`]) and the importer of its release checkpoints
//!     ([`Inkling2Importer`]), which implements the `Importer` trait of `ryft-checkpoints`.
//!   - [`json`] is an insertion ordered JSON value with exact number parsing, which renderers and configuration
//!     readers share.
//!
//! # From a Release to a Training Step
//!
//! The following outlines how the pieces fit together for the Inkling 2 Mini release.
//!
//! ```text
//!  `config.json` ───▶ `Inkling2Configuration` ─────────────────────────────┐
//!                                                                          ▼
//!  `model.safetensors` ──▶ `SafetensorsCheckpoint` ──▶ `Inkling2Importer` ──▶ `Inkling2Decoder<P>`
//!                               ▲                                          │
//!  `Manifest` ┄┄ validates ┄┄┄┄┄┘                                          ▼
//!                                                          `Inkling2Decoder::loss` ◀── `ExpertProjector`
//!  `tokenizer.json` ──▶ `HuggingFaceTokenizer` ──▶ `Inkling2Renderer` ──▶ tokens and supervision
//! ```
//!
//! The importer hands every decoded leaf to a function of the caller, which places it (e.g., on an accelerator), so
//! the leaf type `P` of the decoder is the value type of the backend that evaluates it.

pub mod errors;
pub mod experts;
pub mod inkling2;
pub mod json;
pub mod layers;
pub mod renderers;
pub mod tokenizers;

pub use errors::Error;
pub use experts::{
    BlockScaledExpertProjector, BlockScaledFormat, BlockScaledProjections, ExpertProjector, FloatingExpertProjector,
};
pub use inkling2::{
    Inkling2Configuration, Inkling2ConfigurationError, Inkling2Decoder, Inkling2Importer, Inkling2Inputs,
    Inkling2LoopConfiguration, export_inkling2_decoder,
};
pub use json::{JsonError, JsonValue};
pub use renderers::{
    ContentPart, Conversation, Inkling2Renderer, Message, ReasoningEffort, RenderedConversation, Renderer,
    RendererError, Role, Tool, ToolCall,
};
pub use tokenizers::{HuggingFaceTokenizer, SpecialTokenHandling, TokenId, Tokenizer, TokenizerError};

// TODO(eaplatanios): Review this module.
