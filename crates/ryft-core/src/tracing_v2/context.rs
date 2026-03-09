//! Runtime and transform-local context types for `tracing_v2`.
//!
//! # Why Does `tracing_v2` Need Contexts?
//!
//! A tracing transform often needs two qualitatively different kinds of state at the same time:
//!
//! 1. **Long-lived runtime or backend state.** This includes resources that should outlive any single trace,
//!    such as an MLIR context, a PJRT client, executable caches, device topology information, or any other state
//!    that conceptually belongs to the runtime as a whole.
//! 2. **Short-lived transform-local state.** This includes data structures that exist only while a particular
//!    transformation is active, such as a graph builder used by `jit`, a linear builder used by `jvp`, or the
//!    current batch axis size used by `vmap`.
//!
//! Keeping those two roles in one monolithic context type would make composition awkward. For example, a JVP trace
//! does not conceptually *own* the global runtime state of the backend, and a JIT trace does not conceptually *own*
//! the linearization builder of an enclosing differentiation transform. Instead, each transform should carry only the
//! local state it is responsible for, while *borrowing* whatever broader runtime or parent-transform state already
//! exists.
//!
//! This module therefore separates context types into two layers:
//!
//! - a **top-level context**, which represents long-lived runtime state;
//! - **transform contexts**, such as [`JvpContext`], [`BatchingContext`], and [`JitContext`], which wrap an
//!   underlying context and add temporary state needed by a specific transformation.
//!
//! When no top-level state is needed, the unit type `()` serves as the root context. This is the current default for
//! tests and examples.
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
//!        -> ()
//! ```
//!
//! In that situation:
//!
//! - the [`JitContext`] owns the graph builder for the JIT-staged program;
//! - the enclosing [`JvpContext`] owns the linear builder for forward-mode differentiation;
//! - the root `()` value carries no runtime state at all.
//!
//! This is indeed a parent/child relationship between contexts. The nested context stored inside a transform context
//! is the **parent context in the current transform stack**. It is not required to be the ultimate root context.
//! That distinction is important because it lets transforms compose without flattening all state into one giant
//! structure.
//!
//! # Why `()` Instead Of `None`?
//!
//! If the root context carries no information, the natural Rust representation is the unit type `()`, not `None`.
//! That is because:
//!
//! - `None` is a value, not a type;
//! - using `None` would force the API to mention an `Option<T>` type even though there is no meaningful optionality in
//!   the context model itself;
//! - `()` cleanly expresses "there is a context slot here, but it carries no data"; and
//! - `&mut ()` fits naturally into the existing generic APIs without introducing extra enum structure.
//!
//! So the intended "no root state" story is: pass `&mut ()`.

use std::{cell::RefCell, rc::Rc};

use crate::tracing_v2::{
    TraceValue,
    graph::GraphBuilder,
    ops::{LinearOpRef, StagedOpRef},
};

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
/// its own graph while still having access to the enclosing JVP context via [`Self::top_level_context`].
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

/// Context active while staging a JIT graph.
///
/// [`JitContext`] is the transform-local context for JIT staging. It owns the graph builder used to record primitive
/// applications encountered while tracing the staged computation, while borrowing an underlying parent context of type
/// `Context`.
///
/// The wrapped `Context` is intentionally left generic and unconstrained. At the moment JIT staging does not require
/// any special runtime capability beyond whatever state the caller may already be threading through the transform
/// stack. Later, if a concrete backend such as PJRT requires additional runtime services, those requirements can be
/// introduced where they are actually needed.
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

#[cfg(test)]
mod tests {
    use crate::tracing_v2::{
        AddOp,
        graph::GraphBuilder,
        ops::{LinearOpRef, StagedOpRef},
    };

    use super::*;

    #[derive(Default)]
    struct RootContext {
        visits: usize,
    }

    #[test]
    fn unit_type_serves_as_the_empty_root_context() {
        let _context = ();
        let _also_context: () = Default::default();
    }

    #[test]
    fn transform_contexts_expose_parent_state() {
        let mut root = RootContext::default();
        {
            let mut jvp_context = JvpContext::<_, f64>::new(&mut root);
            jvp_context.top_level_context().visits += 1;
        }
        {
            let mut batching_context = BatchingContext::new(&mut root, 3);
            assert_eq!(batching_context.axis_size(), 3);
            batching_context.top_level_context().visits += 1;
        }
        {
            let mut jit_context = JitContext::<_, f64>::new(&mut root);
            jit_context.top_level_context().visits += 1;
        }
        assert_eq!(root.visits, 3);
    }

    #[test]
    fn nested_transform_contexts_form_a_parent_stack() {
        let mut root = RootContext::default();
        let mut jvp_context = JvpContext::<_, f64>::new(&mut root);
        {
            let mut jit_context = JitContext::<_, f64>::new(&mut jvp_context);
            jit_context.top_level_context().top_level_context().visits += 1;
        }
        jvp_context.top_level_context().visits += 1;
        let (root, _) = jvp_context.finish();
        assert_eq!(root.visits, 2);
    }

    #[test]
    fn direct_transpose_builders_support_local_rewrites() {
        let mut builder = GraphBuilder::<LinearOpRef<f64>, f64>::new();
        let input = builder.add_input(&1.0);
        let output = builder.add_equation(std::sync::Arc::new(AddOp), vec![input, input]).unwrap();
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn jit_context_builder_tracks_inputs() {
        let mut top_level = ();
        let jit_context = JitContext::<_, f64>::new(&mut top_level);
        let builder: Rc<RefCell<GraphBuilder<StagedOpRef<f64>, f64>>> = jit_context.staged_builder();
        let input = builder.borrow_mut().add_input(&2.0);
        assert_eq!(input, 0);
    }
}
