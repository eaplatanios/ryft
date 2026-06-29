//! Backend-agnostic JIT compilation infrastructure.
//!
//! This module provides the pieces that every compilation backend can reuse:
//!
//!   - [`CompilationDomain`] — backend interface for the trace → compile → execute pipeline.
//!   - [`FunctionFingerprint`] — call-site fingerprint used as part of the cache key.
//!   - [`CompilationContext`] — process-local LRU cache of compiled programs, optionally
//!     paired with a disk-backed second tier ([`DiskCache`]).
//!   - [`CompilationOptions`] — thin wrapper around the domain's own
//!     [`CompilationDomain::Options`].
//!   - [`CompiledFunction`] — handle returned by [`compile_with_options`].
//!   - [`compile_with_options`], [`compile`] — the user-facing entry points.
//!
//! Backends implement [`CompilationDomain`] on their tracing/execution domain (e.g.
//! `ryft_xla::XlaDomain`) and the rest of the pipeline composes for free.

pub mod captures;
pub mod context;
pub mod disk_cache;
pub mod domain;
pub mod fingerprint;
pub mod function;
pub mod options;

pub use captures::{CaptureReference, ClosedProgram};
pub use context::{CapturingContext, CompilationContext};
pub use disk_cache::DiskCache;
pub use domain::CompilationDomain;
pub use fingerprint::FunctionFingerprint;
pub use function::{CompiledFunction, compile, compile_with_options};
pub use options::CompilationOptions;
