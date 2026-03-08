use std::{cell::RefCell, rc::Rc};

use crate::tracing_v2::{
    TraceValue,
    graph::GraphBuilder,
    ops::{LinearOpRef, StagedOpRef},
};

pub trait CompilationContext {
    fn allocate_executable_id(&mut self) -> usize;
}

#[derive(Clone, Debug, Default)]
pub struct PrototypeContext {
    next_executable_id: usize,
}

impl PrototypeContext {
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

pub struct BatchingContext<'a, Context> {
    context: &'a mut Context,
    axis_size: usize,
}

impl<'a, Context> BatchingContext<'a, Context> {
    #[inline]
    pub(crate) fn new(context: &'a mut Context, axis_size: usize) -> Self {
        Self { context, axis_size }
    }

    #[inline]
    pub fn top_level_context(&mut self) -> &mut Context {
        self.context
    }

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
