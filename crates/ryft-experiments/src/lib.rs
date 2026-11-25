//! End-to-end experiments that combine the models of `ryft-models` and the optimizers of `ryft-optimizers` with the
//! XLA backend of `ryft-xla`.
//!
//! Every experiment is a module of this library together with a binary that runs it from the command line. The
//! library part holds the pieces that tests and other experiments reuse (e.g., a compiled training step), and the
//! binary handles arguments and reporting.
//!
//!   - [`inkling2`] trains the Inkling 2 text decoder. Its [`Inkling2Trainer`](inkling2::Inkling2Trainer) compiles
//!     one training step that computes the loss, its gradients, and the optimizer update on the device, with either
//!     floating-point or mixed precision expert projections, and saves and loads its training state through
//!     the training checkpointers of `ryft-checkpoints`. The `inkling2_training` binary times such steps on a selected
//!     platform, optionally checkpointing and resuming the run, and the `inkling2_checkpoint` binary writes
//!     synthetic checkpoints in the release layout.

// TODO(eaplatanios): Review this module.

pub mod inkling2;
