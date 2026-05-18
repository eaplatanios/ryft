//! Backend-agnostic JIT compilation infrastructure.
//!
//! This module provides the pieces that every compilation backend can reuse:
//!
//!   - [`CompilationDomain`] — backend interface for the trace → compile → execute pipeline.
//!   - [`FunctionFingerprint`] — call-site fingerprint used as part of the cache key.
//!   - [`CompilationContext`] — process-local LRU cache of compiled programs, optionally
//!     paired with a disk-backed second tier ([`DiskCache`]).
//!   - [`CompilationOptions`] — thin wrapper around the engine's own
//!     [`CompilationDomain::Options`].
//!   - [`CompiledFunction`] — handle returned by [`compile_and_execute_with_options`].
//!   - [`CompilationError`] — error type returned by the core pipeline; wraps the engine's
//!     [`CompilationDomain::Error`].
//!   - [`compile_and_execute_with_options`], [`compile_and_execute`], [`eval_shape`] — the
//!     user-facing entry points.
//!
//! Backends implement [`CompilationDomain`] on their tracing/execution domain (e.g.
//! `ryft_xla::XlaDomain`) and the rest of the pipeline composes for free.

pub mod context;
pub mod disk_cache;
pub mod domain;
pub mod error;
pub mod fingerprint;
pub mod function;
pub mod options;

pub use context::CompilationContext;
pub use disk_cache::DiskCache;
pub use domain::CompilationDomain;
pub use error::CompilationError;
pub use fingerprint::FunctionFingerprint;
pub use function::{CompiledFunction, compile_and_execute, compile_and_execute_with_options, eval_shape};
pub use options::CompilationOptions;
