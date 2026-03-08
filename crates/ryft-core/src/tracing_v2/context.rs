//! Runtime and transform-local context types for `tracing_v2`.
//!
//! The current design separates long-lived runtime state from short-lived tracing state:
//!
//! - a top-level context owns backend/runtime resources such as executable identifiers;
//! - transform-specific contexts borrow that top-level context while also carrying builders or transform metadata.
//!
//! This keeps the tracing APIs explicit about which transformation is active while still allowing all transforms to
//! share a common backend state object.

use std::{cell::RefCell, rc::Rc};

use crate::tracing_v2::{
    TraceValue,
    graph::GraphBuilder,
    ops::{LinearOpRef, StagedOpRef},
};

/// Capability trait for contexts that can assign identifiers to staged executables.
pub trait CompilationContext {
    /// Allocates the next executable identifier.
    fn allocate_executable_id(&mut self) -> usize;
}

/// Small in-memory context used by tests and examples.
#[derive(Clone, Debug, Default)]
pub struct PrototypeContext {
    next_executable_id: usize,
}

impl PrototypeContext {
    /// Returns the number of compiled programs allocated so far.
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
    #[inline]
    pub(crate) fn new(context: &'a mut Context) -> Self {
        Self { context, linear_builder: Rc::new(RefCell::new(GraphBuilder::new())) }
    }

    /// Returns the borrowed top-level context.
    #[inline]
    pub fn top_level_context(&mut self) -> &mut Context {
        self.context
    }

    #[inline]
    pub(crate) fn linear_builder(&self) -> Rc<RefCell<GraphBuilder<LinearOpRef<V>, V>>> {
        self.linear_builder.clone()
    }

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
pub struct BatchingContext<'a, Context> {
    context: &'a mut Context,
    axis_size: usize,
}

impl<'a, Context> BatchingContext<'a, Context> {
    #[inline]
    pub(crate) fn new(context: &'a mut Context, axis_size: usize) -> Self {
        Self { context, axis_size }
    }

    /// Returns the borrowed top-level context.
    #[inline]
    pub fn top_level_context(&mut self) -> &mut Context {
        self.context
    }

    /// Returns the current batch axis size.
    #[inline]
    pub fn axis_size(&self) -> usize {
        self.axis_size
    }

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
    #[inline]
    pub(crate) fn new(context: &'a mut Context) -> Self {
        Self { context, staged_builder: Rc::new(RefCell::new(GraphBuilder::new())) }
    }

    /// Returns the borrowed top-level context.
    #[inline]
    pub fn top_level_context(&mut self) -> &mut Context {
        self.context
    }

    #[inline]
    pub(crate) fn staged_builder(&self) -> Rc<RefCell<GraphBuilder<StagedOpRef<V>, V>>> {
        self.staged_builder.clone()
    }

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
    #[inline]
    pub(crate) fn new(graph_builder: &'a mut GraphBuilder<LinearOpRef<V>, V>) -> Self {
        Self { graph_builder }
    }

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
