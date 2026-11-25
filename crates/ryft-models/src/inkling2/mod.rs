//! The Inkling 2 model family of Thinking Machines.
//!
//! Inkling 2 models are multimodal decoders with attention residuals, a fixed loop over their middle layers, hybrid
//! local and global attention, and routed experts whose projections are stored and trained in block-scaled low
//! precision formats. This module currently covers the text decoder.
//!
//!   - [`Inkling2Configuration`] describes a decoder, with presets for the Inkling 2 Mini release and for a reduced
//!     model with the same layer layout.
//!   - [`Inkling2Decoder`] is the parameter structure of the decoder together with its forward computation and its
//!     training loss. Its numerics follow the pinned floating reference implementation, and the precision of its
//!     expert projections is selected by the [`ExpertProjector`](crate::experts::ExpertProjector) that it is given.
//!   - [`Inkling2Importer`] maps the checkpoint of a release to the parameters of a decoder, decoding its packed
//!     NVFP4 and MXFP8 arrays, and [`export_inkling2_decoder`] writes parameters back under the release array
//!     names.
//!
//! The conversation format of these models is implemented by
//! [`Inkling2Renderer`](crate::renderers::Inkling2Renderer). The vision encoder and the speculative decoding drafter
//! of the releases are not implemented, and the importer leaves their arrays unread.

pub mod configuration;
pub mod decoder;
pub mod importer;

pub use configuration::{Inkling2Configuration, Inkling2ConfigurationError, Inkling2LoopConfiguration};
pub use decoder::{
    Inkling2Attention, Inkling2Convolution, Inkling2Decoder, Inkling2Diagnostics, Inkling2Experts, Inkling2Inputs,
    Inkling2Layer, Inkling2LoopGate, Inkling2LoopGates,
};
pub use importer::{
    INKLING_2_MINI_ARRAY_COUNT, INKLING_2_MINI_CHECKPOINT_FILE, Inkling2Importer, export_inkling2_decoder,
    inkling_2_mini_manifest,
};

// TODO(eaplatanios): Review this module.
