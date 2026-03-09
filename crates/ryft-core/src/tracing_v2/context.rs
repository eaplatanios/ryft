//! Runtime and transform-local context types for `tracing_v2`.
//!
//! # Why Does `tracing_v2` Need Contexts?
//!
//! A tracing transform often needs two qualitatively different kinds of state at the same time:
//!
//! 1. **Long-lived runtime or backend state.** This includes resources that should outlive any single trace,
//!    such as an MLIR context, a PJRT client, executable caches, or monotonically increasing identifiers for
//!    compiled artifacts.
//! 2. **Short-lived transform-local state.** This includes data structures that exist only while a particular
//!    transformation is active, such as a graph builder used by `jit`, a linear builder used by `jvp`, or the
//!    current batch axis size used by `vmap`.
//!
//! Keeping those two roles in one monolithic context type would make composition awkward. For example, a JVP trace
//! does not conceptually *own* the global compilation state of the runtime, and a JIT trace does not conceptually
//! *own* the linearization builder of an enclosing differentiation transform. Instead, each transform should carry
//! only the local state it is responsible for, while *borrowing* whatever broader runtime or parent transform state
//! already exists.
//!
//! This module therefore separates context types into two layers:
//!
//! - a **top-level context**, such as [`PrototypeContext`], which represents long-lived runtime state;
//! - **transform contexts**, such as [`JvpContext`], [`BatchingContext`], and [`JitContext`], which wrap an
//!   underlying context and add temporary state needed by a specific transformation.
//!
//! # Why Do Some Contexts Wrap Another `Context`?
//!
//! The generic `Context` parameter inside [`JvpContext`], [`BatchingContext`], and [`JitContext`] represents the
//! **underlying context** on which the current transformation is running. In the simplest case, that underlying
//! context is the top-level runtime context. However, when transformations are composed, it can itself be another
//! transform context.
//!
//! For example, if a `jit` trace is started while a JVP trace is already active, then the resulting structure is
//! conceptually similar to a stack:
//!
//! ```text
//! JitContext
//!   -> JvpContext
//!        -> PrototypeContext
//! ```
//!
//! In that situation:
//!
//! - the [`JitContext`] owns the graph builder for the JIT-staged program;
//! - the enclosing [`JvpContext`] owns the linear builder for forward-mode differentiation;
//! - the root [`PrototypeContext`] owns long-lived runtime state such as executable identifiers.
//!
//! This is indeed a parent/child relationship between contexts. The nested context stored inside a transform context
//! is the **parent context in the current transform stack**. It is not required to be the ultimate root context.
//! That distinction is important because it lets transforms compose without flattening all state into one giant
//! structure.
//!
//! # Why Is There Both `JitContext` and `CompilationContext`?
//!
//! [`JitContext`] and [`CompilationContext`] solve different problems.
//!
//! [`JitContext`] is a **concrete transform-local context**. It exists only while staging a JIT trace and carries
//! the graph builder for that trace.
//!
//! [`CompilationContext`] is a **small capability trait**. It answers a much narrower question: "can this context
//! allocate identifiers for compiled executables?" Any context that can answer that question may implement the trait,
//! including:
//!
//! - a root runtime context like [`PrototypeContext`];
//! - a transform context like [`JvpContext`] or [`BatchingContext`] that simply forwards the request to its parent;
//! - a [`JitContext`] that likewise forwards to its parent.
//!
//! This separation matters because many APIs should depend only on a *capability*, not on a specific concrete context
//! type. In particular, the public [`crate::tracing_v2::jit`] API needs only the ability to allocate an executable id
//! for the staged program it returns. It does **not** need to know whether it is running directly against the root
//! runtime context or inside some enclosing transformation. By constraining that API with [`CompilationContext`]
//! instead of a concrete context type, `jit` can be invoked in both situations.
//!
//! Put differently:
//!
//! - [`JitContext`] provides the **local state for running the JIT transform**.
//! - [`CompilationContext`] provides the **minimum shared capability needed to register the result of compilation**.
//!
//! That is why transform contexts implement [`CompilationContext`] by delegation: it preserves the ability to compile
//! from inside nested transform stacks without forcing every transform to manually thread executable-id state.
//!
//! # Why Does [`TransposeContext`] Look Different?
//!
//! [`TransposeContext`] currently does not wrap a generic underlying context. The present transpose pass only needs
//! mutable access to the linear graph builder used to construct the transposed program; it does not currently require
//! any runtime capability from a parent context. If that changes later, this type can be extended in the same style
//! as the other transform contexts.

use std::{cell::RefCell, rc::Rc};

use crate::tracing_v2::{
    TraceValue,
    graph::GraphBuilder,
    ops::{LinearOpRef, StagedOpRef},
};

/// Capability trait for contexts that can assign identifiers to compiled executables.
///
/// This trait is intentionally tiny. It does **not** describe everything a runtime or backend context might know how
/// to do. Instead, it captures only the specific capability currently needed by the JIT staging pipeline: allocating
/// a stable identifier for each compiled program.
///
/// Using a trait here instead of hard-coding a concrete root context serves two purposes:
///
/// 1. It keeps APIs like [`crate::tracing_v2::jit`] generic over the *capability they need* rather than over a
///    specific context implementation.
/// 2. It lets transform contexts participate in compilation simply by forwarding the request to the underlying parent
///    context they wrap.
///
/// That forwarding behavior is what allows nested transform stacks to continue supporting JIT staging. For example,
/// a [`JvpContext`] can implement [`CompilationContext`] by delegating to its parent context, which means `jit` can be
/// called even while a JVP trace is active.
pub trait CompilationContext {
    /// Allocates the next executable identifier.
    ///
    /// The returned identifier is expected to be unique within the lifetime of the underlying runtime context.
    /// In the current prototype this is used only for staged program bookkeeping, but later it may also index caches,
    /// backend-specific executable registries, or compiled artifact tables.
    fn allocate_executable_id(&mut self) -> usize;
}

/// Small in-memory top-level context used by tests and examples.
///
/// [`PrototypeContext`] plays the role of the root runtime context in the current prototype. It intentionally keeps
/// the state minimal so that the tracing design can be exercised without tying it to MLIR, PJRT, or a particular
/// backend implementation.
///
/// Conceptually, this type is where future long-lived backend state would live. For example, a production runtime
/// context might store:
///
/// - an MLIR context or module arena;
/// - a PJRT client and device topology;
/// - executable caches;
/// - backend handles for constant interning or sharded array management.
///
/// The current implementation stores only a monotonically increasing executable counter because that is the only
/// long-lived capability needed by the present prototype.
#[derive(Clone, Debug, Default)]
pub struct PrototypeContext {
    next_executable_id: usize,
}

impl PrototypeContext {
    /// Returns the number of compiled programs allocated so far.
    ///
    /// Because [`PrototypeContext`] currently allocates executable identifiers sequentially starting from zero, this
    /// is also equal to the next identifier that would be returned by [`CompilationContext::allocate_executable_id`].
    #[inline]
    pub fn compiled_program_count(&self) -> usize {
        self.next_executable_id
    }
}

impl CompilationContext for PrototypeContext {
    #[inline]
    fn allocate_executable_id(&mut self) -> usize {
        let id = self.next_executable_id;
        self.next_executable_id += 1;
        id
    }
}

/// Context active while staging a JVP / linearization trace.
///
/// [`JvpContext`] is the transform-local context for forward-mode differentiation. It owns the temporary builder used
/// to stage the linearized program while also borrowing an underlying parent context of type `Context`.
///
/// The generic `Context` parameter is deliberately unconstrained here. It may be:
///
/// - the root runtime context;
/// - another transform context higher in the active stack; or
/// - any other context-like object that the caller wishes to make available while the JVP trace is running.
///
/// This design makes transform composition explicit in the type system. A nested `jit` inside a JVP trace can stage
/// its own graph while still having access to the enclosing JVP context via [`Self::top_level_context`], and any
/// compilation-related requests can continue to flow outward through delegated trait implementations.
pub struct JvpContext<'a, Context, V>
where
    V: TraceValue,
{
    context: &'a mut Context,
    linear_builder: Rc<RefCell<GraphBuilder<LinearOpRef<V>, V>>>,
}

impl<'a, Context, V> JvpContext<'a, Context, V>
where
    V: TraceValue,
{
    /// Creates a fresh JVP context over `context`.
    ///
    /// The newly created context starts with an empty linear graph builder that will accumulate the pushforward traced
    /// by the active forward-mode transformation.
    #[inline]
    pub(crate) fn new(context: &'a mut Context) -> Self {
        Self { context, linear_builder: Rc::new(RefCell::new(GraphBuilder::new())) }
    }

    /// Returns the underlying context borrowed by this JVP context.
    ///
    /// Despite the current method name, this returns the **immediate parent context**, not necessarily the ultimate
    /// root runtime context. When no other transformation is active, those two are the same. When transforms are
    /// nested, the returned value is the next context outward in the stack.
    #[inline]
    pub fn top_level_context(&mut self) -> &mut Context {
        self.context
    }

    /// Returns shared access to the linear graph builder owned by this JVP context.
    ///
    /// The builder is wrapped in [`Rc<RefCell<_>>`] because multiple traced tangent values created during the same JVP
    /// trace need to keep appending equations to the same linearized program.
    #[inline]
    pub(crate) fn linear_builder(&self) -> Rc<RefCell<GraphBuilder<LinearOpRef<V>, V>>> {
        self.linear_builder.clone()
    }

    /// Finishes the JVP trace and returns both the parent context and the staged linear builder.
    ///
    /// This is used internally once the trace has produced all of its outputs and the caller is ready to finalize the
    /// accumulated builder into a [`crate::tracing_v2::LinearProgram`].
    #[inline]
    pub(crate) fn finish(self) -> (&'a mut Context, Rc<RefCell<GraphBuilder<LinearOpRef<V>, V>>>) {
        (self.context, self.linear_builder)
    }
}

impl<Context, V> CompilationContext for JvpContext<'_, Context, V>
where
    Context: CompilationContext,
    V: TraceValue,
{
    #[inline]
    fn allocate_executable_id(&mut self) -> usize {
        self.context.allocate_executable_id()
    }
}

/// Context active while batching a computation with `vmap`-style semantics.
///
/// [`BatchingContext`] carries the batch-axis metadata needed by the current batching transformation while borrowing
/// an underlying parent context of type `Context`.
///
/// Like [`JvpContext`], the stored `Context` represents the next context outward in the active transform stack. This
/// lets batching coexist cleanly with other transforms without forcing all transform state into one structure.
pub struct BatchingContext<'a, Context> {
    context: &'a mut Context,
    axis_size: usize,
}

impl<'a, Context> BatchingContext<'a, Context> {
    /// Creates a fresh batching context over `context` for a batch axis of length `axis_size`.
    #[inline]
    pub(crate) fn new(context: &'a mut Context, axis_size: usize) -> Self {
        Self { context, axis_size }
    }

    /// Returns the underlying context borrowed by this batching context.
    ///
    /// As with [`JvpContext::top_level_context`], this is the immediate parent context, which may or may not be the
    /// root runtime context depending on how transforms are composed.
    #[inline]
    pub fn top_level_context(&mut self) -> &mut Context {
        self.context
    }

    /// Returns the current batch axis size.
    ///
    /// This is the number of logical lanes being batched together by the active `vmap`-style transformation.
    #[inline]
    pub fn axis_size(&self) -> usize {
        self.axis_size
    }

    /// Finishes the batching transform and returns the borrowed parent context.
    #[inline]
    pub(crate) fn finish(self) -> &'a mut Context {
        self.context
    }
}

impl<Context> CompilationContext for BatchingContext<'_, Context>
where
    Context: CompilationContext,
{
    #[inline]
    fn allocate_executable_id(&mut self) -> usize {
        self.context.allocate_executable_id()
    }
}

/// Context active while staging a JIT graph.
///
/// [`JitContext`] is the transform-local context for JIT staging. It owns the graph builder used to record primitive
/// applications encountered while tracing the staged computation, while borrowing an underlying parent context of type
/// `Context`.
///
/// It is important to distinguish this type from [`CompilationContext`]:
///
/// - [`JitContext`] stores the **temporary state needed to *perform* JIT tracing**.
/// - [`CompilationContext`] exposes the **longer-lived capability needed to *register the result* of JIT tracing**.
///
/// That distinction is what allows the public `jit` API to work both at the root level and inside nested transform
/// stacks.
pub struct JitContext<'a, Context, V>
where
    V: TraceValue,
{
    context: &'a mut Context,
    staged_builder: Rc<RefCell<GraphBuilder<StagedOpRef<V>, V>>>,
}

impl<'a, Context, V> JitContext<'a, Context, V>
where
    V: TraceValue,
{
    /// Creates a fresh JIT context over `context`.
    ///
    /// The new context starts with an empty staged graph builder that will record the primitive equations of the JIT
    /// trace as it executes eagerly.
    #[inline]
    pub(crate) fn new(context: &'a mut Context) -> Self {
        Self { context, staged_builder: Rc::new(RefCell::new(GraphBuilder::new())) }
    }

    /// Returns the underlying context borrowed by this JIT context.
    ///
    /// When transforms are nested, this is the immediate parent context in the active transform stack.
    #[inline]
    pub fn top_level_context(&mut self) -> &mut Context {
        self.context
    }

    /// Returns shared access to the staged graph builder owned by this JIT context.
    ///
    /// The builder is wrapped in [`Rc<RefCell<_>>`] because multiple traced values created during the same JIT trace
    /// must all append equations to the same staged graph.
    #[inline]
    pub(crate) fn staged_builder(&self) -> Rc<RefCell<GraphBuilder<StagedOpRef<V>, V>>> {
        self.staged_builder.clone()
    }

    /// Finishes the JIT trace and returns both the parent context and the staged graph builder.
    ///
    /// This is used internally once tracing is complete and the accumulated builder is ready to be finalized into a
    /// [`crate::tracing_v2::CompiledFunction`].
    #[inline]
    pub(crate) fn finish(self) -> (&'a mut Context, Rc<RefCell<GraphBuilder<StagedOpRef<V>, V>>>) {
        (self.context, self.staged_builder)
    }
}

impl<Context, V> CompilationContext for JitContext<'_, Context, V>
where
    Context: CompilationContext,
    V: TraceValue,
{
    #[inline]
    fn allocate_executable_id(&mut self) -> usize {
        self.context.allocate_executable_id()
    }
}

/// Context used while transposing a linear program.
///
/// [`TransposeContext`] currently contains only the builder into which the transposed linear program is emitted. In
/// contrast to [`JvpContext`] and [`JitContext`], it does not yet wrap a generic parent context because the current
/// transpose pass requires only local staging state and no runtime capability.
///
/// If future transpose rules need access to backend state, residual environments, or other outer-transform metadata,
/// this type can be extended to wrap an underlying parent context in the same style as the other transform contexts.
pub struct TransposeContext<'a, V>
where
    V: TraceValue,
{
    graph_builder: &'a mut GraphBuilder<LinearOpRef<V>, V>,
}

impl<'a, V> TransposeContext<'a, V>
where
    V: TraceValue,
{
    /// Creates a transpose context over `graph_builder`.
    #[inline]
    pub(crate) fn new(graph_builder: &'a mut GraphBuilder<LinearOpRef<V>, V>) -> Self {
        Self { graph_builder }
    }

    /// Returns the mutable linear graph builder used to emit the transposed program.
    #[inline]
    pub(crate) fn graph_builder(&mut self) -> &mut GraphBuilder<LinearOpRef<V>, V> {
        self.graph_builder
    }
}

#[cfg(test)]
mod tests {
    use crate::tracing_v2::{
        AddOp,
        graph::GraphBuilder,
        ops::{LinearOpRef, StagedOpRef},
    };

    use super::*;

    #[test]
    fn prototype_context_allocates_monotonic_ids() {
        let mut context = PrototypeContext::default();
        assert_eq!(context.allocate_executable_id(), 0);
        assert_eq!(context.allocate_executable_id(), 1);
        assert_eq!(context.compiled_program_count(), 2);
    }

    #[test]
    fn transform_contexts_forward_compilation_context_calls() {
        let mut top_level = PrototypeContext::default();
        {
            let mut jvp_context = JvpContext::<_, f64>::new(&mut top_level);
            assert_eq!(jvp_context.allocate_executable_id(), 0);
            let _ = jvp_context.top_level_context();
        }
        {
            let mut batching_context = BatchingContext::new(&mut top_level, 3);
            assert_eq!(batching_context.axis_size(), 3);
            assert_eq!(batching_context.allocate_executable_id(), 1);
        }
        {
            let mut jit_context = JitContext::<_, f64>::new(&mut top_level);
            assert_eq!(jit_context.allocate_executable_id(), 2);
        }
        assert_eq!(top_level.compiled_program_count(), 3);
    }

    #[test]
    fn nested_transform_contexts_delegate_compilation_to_the_root_context() {
        let mut root = PrototypeContext::default();
        let mut jvp_context = JvpContext::<_, f64>::new(&mut root);
        {
            let mut jit_context = JitContext::<_, f64>::new(&mut jvp_context);
            assert_eq!(jit_context.allocate_executable_id(), 0);
            assert_eq!(jit_context.top_level_context().allocate_executable_id(), 1);
        }
        assert_eq!(jvp_context.allocate_executable_id(), 2);
        let (root, _) = jvp_context.finish();
        assert_eq!(root.compiled_program_count(), 3);
    }

    #[test]
    fn transpose_context_exposes_the_underlying_builder() {
        let mut builder = GraphBuilder::<LinearOpRef<f64>, f64>::new();
        let input = builder.add_input(&1.0);
        let mut context = TransposeContext::new(&mut builder);
        let output = context.graph_builder().add_equation(std::sync::Arc::new(AddOp), vec![input, input]).unwrap();
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn jit_context_builder_tracks_inputs() {
        let mut top_level = PrototypeContext::default();
        let jit_context = JitContext::<_, f64>::new(&mut top_level);
        let builder: Rc<RefCell<GraphBuilder<StagedOpRef<f64>, f64>>> = jit_context.staged_builder();
        let input = builder.borrow_mut().add_input(&2.0);
        assert_eq!(input, 0);
    }
}
