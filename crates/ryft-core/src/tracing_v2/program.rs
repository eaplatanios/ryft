//! Shared staged program wrapper for `tracing_v2`.
//!
//! A [`Program`] is a thin semantic wrapper around [`Graph`](crate::tracing_v2::Graph) using the
//! common operation universe for staged tracing. It is the canonical internal IR that later
//! transforms can replay or rewrite without retracing the original Rust closure.

use std::{fmt::Display, marker::PhantomData, sync::Arc};

use crate::{
    parameters::Parameterized,
    tracing_v2::{Graph, GraphBuilder, Op, TraceError, TraceValue},
};

/// Shared operation reference used by the canonical staged program IR.
pub(crate) type ProgramOpRef<V> = Arc<dyn Op<V>>;

/// Shared builder used by the canonical staged program IR.
pub(crate) type ProgramBuilder<V> = GraphBuilder<ProgramOpRef<V>, V>;

/// Canonical staged program used by `tracing_v2`.
pub(crate) struct Program<V, Input, Output>
where
    V: TraceValue,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    graph: Graph<ProgramOpRef<V>, V, Input, Output>,
    marker: PhantomData<fn(Input) -> Output>,
}

impl<V, Input, Output> Clone for Program<V, Input, Output>
where
    V: TraceValue,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
{
    fn clone(&self) -> Self {
        Self { graph: self.graph.clone(), marker: PhantomData }
    }
}

impl<V, Input, Output> Program<V, Input, Output>
where
    V: TraceValue,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    /// Creates a program from an existing staged graph.
    #[inline]
    pub(crate) fn from_graph(graph: Graph<ProgramOpRef<V>, V, Input, Output>) -> Self {
        Self { graph, marker: PhantomData }
    }

    /// Returns the underlying staged graph.
    #[inline]
    pub(crate) fn graph(&self) -> &Graph<ProgramOpRef<V>, V, Input, Output> {
        &self.graph
    }

    /// Replays the staged program on concrete input values.
    #[inline]
    pub(crate) fn call(&self, input: Input) -> Result<Output, TraceError>
    where
        Input::ParameterStructure: PartialEq,
        Output::ParameterStructure: Clone,
    {
        self.graph.call(input)
    }

    /// Eliminates dead constants and equations that do not contribute to the program outputs.
    pub(crate) fn simplify(&self) -> Result<Self, TraceError>
    where
        Input::ParameterStructure: Clone,
        Output::ParameterStructure: Clone,
    {
        Ok(Self::from_graph(self.graph.simplify()?))
    }
}

impl<V, Input, Output> Display for Program<V, Input, Output>
where
    V: TraceValue,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.graph, formatter)
    }
}
