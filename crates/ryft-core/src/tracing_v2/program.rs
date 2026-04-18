//! Shared staged program wrapper for `tracing_v2`.
//!
//! A [`Program`] is a thin semantic wrapper around [`Graph`](crate::tracing_v2::Graph) using the
//! common operation universe for staged tracing. It is the canonical internal IR that later
//! transforms can replay or rewrite without retracing the original Rust closure.

use std::{fmt::Display, marker::PhantomData};

use crate::{
    parameters::{Parameter, Parameterized},
    tracing_v2::{Graph, GraphBuilder, InterpretableOp, LinearPrimitiveOp, Op, PrimitiveOp, TraceError, Traceable},
    types::{ArrayType, Type, Typed},
};

/// Operation type used by the staged program IR.
pub type ProgramOpFor<O> = O;

/// Canonical operation type used by the staged program IR.
pub type ProgramOpRef<V> = PrimitiveOp<ArrayType, V>;

/// Shared builder used by the staged program IR.
pub type ProgramBuilderFor<V, O> = GraphBuilder<ProgramOpFor<O>, ArrayType, V>;

/// Shared builder used by the canonical staged program IR.
pub type ProgramBuilder<V> = ProgramBuilderFor<V, ProgramOpRef<V>>;

/// Operation type used by the staged linear-program IR.
pub type LinearProgramOpFor<O> = O;

/// Canonical operation type used by the staged linear-program IR.
pub type LinearProgramOpRef<V> = LinearPrimitiveOp<ArrayType, V>;

/// Shared builder used by the staged linear-program IR.
pub type LinearProgramBuilderFor<V, O> = GraphBuilder<LinearProgramOpFor<O>, ArrayType, V>;

/// Shared builder used by the staged linear-program IR.
pub type LinearProgramBuilder<V> = LinearProgramBuilderFor<V, LinearProgramOpRef<V>>;

/// Canonical staged program used by `tracing_v2`.
pub struct Program<
    T: Type,
    V: Typed<T> + Parameter,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
    O = ProgramOpRef<V>,
> {
    graph: Graph<O, T, V, Input, Output>,
    marker: PhantomData<fn(Input) -> Output>,
}

impl<
    T: Type,
    V: Traceable<T>,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    O: Clone,
> Clone for Program<T, V, Input, Output, O>
{
    fn clone(&self) -> Self {
        Self { graph: self.graph.clone(), marker: PhantomData }
    }
}

impl<T: Type, V: Traceable<T>, Input: Parameterized<V>, Output: Parameterized<V>, O: Clone>
    Program<T, V, Input, Output, O>
{
    /// Creates a program from an existing staged graph.
    #[inline]
    pub fn from_graph(graph: Graph<O, T, V, Input, Output>) -> Self {
        Self { graph, marker: PhantomData }
    }

    /// Returns the underlying staged graph.
    #[inline]
    pub fn graph(&self) -> &Graph<O, T, V, Input, Output> {
        &self.graph
    }

    /// Replays the staged program on concrete input values.
    #[inline]
    pub fn call(&self, input: Input) -> Result<Output, TraceError>
    where
        O: InterpretableOp<T, V>,
        Input::ParameterStructure: PartialEq,
        Output::ParameterStructure: Clone,
    {
        self.graph.call(input)
    }

    /// Eliminates dead constants and equations that do not contribute to the program outputs.
    pub fn simplify(&self) -> Result<Self, TraceError>
    where
        O: Op<T>,
        Input::ParameterStructure: Clone,
        Output::ParameterStructure: Clone,
    {
        Ok(Self::from_graph(self.graph.simplify()?))
    }
}

impl<T: Type + Display, V: Traceable<T>, Input: Parameterized<V>, Output: Parameterized<V>, O: Clone + Display> Display
    for Program<T, V, Input, Output, O>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.graph, formatter)
    }
}
