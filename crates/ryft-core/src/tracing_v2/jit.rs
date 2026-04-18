//! Just-in-time staging support for `tracing_v2`.
//!
//! The current `jit` transform captures a graph of staged primitive applications and replays that graph with the
//! built-in interpreter. This keeps the API shape close to the eventual compiled-backend design while remaining easy
//! to test in pure Rust.

use std::{
    borrow::Cow,
    cell::RefCell,
    fmt::Display,
    marker::PhantomData,
    ops::{Add, Mul, Neg},
    rc::Rc,
};

use crate::{
    parameters::{Parameter, Parameterized, ParameterizedFamily},
    tracing_v2::{
        GraphBuilder, InterpretableOp, OneLike, TraceError, Traceable, ZeroLike,
        engine::Engine,
        graph::AtomId,
        ops::{AddTracingOperation, MulTracingOperation, NegTracingOperation, Op},
        program::{Program, ProgramBuilder, ProgramOpRef},
    },
    types::{ArrayType, Type, Typed},
};

/// Input family that can be rebuilt with traced leaves for one staged carrier pair.
#[doc(hidden)]
pub trait TraceInput<V: Traceable<ArrayType>, O: Clone + 'static, L: Clone + 'static>:
    Parameterized<V, ParameterStructure: Clone>
{
    /// Traced version of this input family for one staged carrier pair.
    type Traced: Parameterized<JitTracer<ArrayType, V, O, L>, ParameterStructure = Self::ParameterStructure>;

    /// Rebuilds `self` with traced leaves owned by `builder`.
    fn into_traced(
        self,
        builder: Rc<RefCell<GraphBuilder<O, ArrayType, V>>>,
        staging_error: Rc<RefCell<Option<TraceError>>>,
        engine: &dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L>,
    ) -> Result<Self::Traced, TraceError>;
}

impl<T, V, O, L> TraceInput<V, O, L> for T
where
    T: Parameterized<V, ParameterStructure: Clone>,
    V: Traceable<ArrayType>,
    O: Clone + 'static,
    L: Clone + 'static,
    T::Family: ParameterizedFamily<JitTracer<ArrayType, V, O, L>>,
{
    type Traced = T::To<JitTracer<ArrayType, V, O, L>>;

    fn into_traced(
        self,
        builder: Rc<RefCell<GraphBuilder<O, ArrayType, V>>>,
        staging_error: Rc<RefCell<Option<TraceError>>>,
        engine: &dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L>,
    ) -> Result<Self::Traced, TraceError> {
        let structure = self.parameter_structure();
        Self::Traced::from_parameters(
            structure,
            self.into_parameters().map(|value| {
                let atom = builder.borrow_mut().add_input(&value);
                JitTracer::<ArrayType, V, O, L>::from_engine(atom, builder.clone(), staging_error.clone(), engine)
            }),
        )
        .map_err(TraceError::from)
    }
}

/// Output family that can be lowered back to concrete leaves after tracing.
#[doc(hidden)]
pub trait TraceOutput<V: Traceable<ArrayType>, O: Clone + 'static, L: Clone + 'static>:
    Parameterized<V, ParameterStructure: Clone>
{
    /// Traced version of this output family for one staged carrier pair.
    type Traced: Parameterized<JitTracer<ArrayType, V, O, L>, ParameterStructure = Self::ParameterStructure>;

    /// Lowers one traced output to its parameter structure and the corresponding staged output atoms.
    fn from_traced(traced_output: Self::Traced) -> Result<(Self::ParameterStructure, Vec<AtomId>), TraceError>;
}

impl<T, V, O, L> TraceOutput<V, O, L> for T
where
    T: Parameterized<V, ParameterStructure: Clone>,
    V: Traceable<ArrayType>,
    O: Clone + 'static,
    L: Clone + 'static,
    T::Family: ParameterizedFamily<JitTracer<ArrayType, V, O, L>>,
{
    type Traced = T::To<JitTracer<ArrayType, V, O, L>>;

    fn from_traced(traced_output: Self::Traced) -> Result<(Self::ParameterStructure, Vec<AtomId>), TraceError> {
        let output_structure = traced_output.parameter_structure();
        let output_atoms = traced_output.into_parameters().map(|output| output.atom).collect::<Vec<_>>();
        Ok((output_structure, output_atoms))
    }
}

/// Type-directed tracing family for one staged carrier pair.
///
/// Unifies the input side (rebuilding a type exemplar with type-directed traced leaves via
/// [`into_type_traced`](Self::into_type_traced)) and the output side (lowering a type-traced output back to type
/// metadata and staged atoms via [`from_type_traced`](Self::from_type_traced)) into a single trait. Both
/// directions share the same [`Staged`](Self::Staged) and [`Traced`](Self::Traced) projections.
#[doc(hidden)]
pub trait TypeTracing<T: Type + Display + Parameter, V: Traceable<T>, O: Clone + 'static, L: Clone + 'static>:
    Parameterized<T, ParameterStructure: Clone>
{
    /// Staged concrete leaf family used by the traced program.
    type Staged: Parameterized<V, ParameterStructure = Self::ParameterStructure>;

    /// Type-traced version of this family for one staged carrier pair.
    type Traced: Parameterized<JitTracer<T, V, O, L>, ParameterStructure = Self::ParameterStructure>;

    /// Rebuilds `self` with type-directed traced leaves owned by `builder`.
    fn into_type_traced(
        self,
        builder: Rc<RefCell<GraphBuilder<O, T, V>>>,
        staging_error: Rc<RefCell<Option<TraceError>>>,
        engine: &dyn Engine<Type = T, Value = V, TracingOperation = O, LinearOperation = L>,
    ) -> Result<Self::Traced, TraceError>;

    /// Lowers one type-traced output back to type metadata and output atoms.
    fn from_type_traced(traced_output: Self::Traced) -> Result<(Self, Vec<AtomId>), TraceError>;
}

impl<Value, T, V, O, L> TypeTracing<T, V, O, L> for Value
where
    Value: Parameterized<T, ParameterStructure: Clone>,
    T: Type + Display + Parameter,
    V: Traceable<T>,
    O: Clone + 'static,
    L: Clone + 'static,
    Value::Family: ParameterizedFamily<V> + ParameterizedFamily<JitTracer<T, V, O, L>>,
{
    type Staged = Value::To<V>;
    type Traced = Value::To<JitTracer<T, V, O, L>>;

    fn into_type_traced(
        self,
        builder: Rc<RefCell<GraphBuilder<O, T, V>>>,
        staging_error: Rc<RefCell<Option<TraceError>>>,
        engine: &dyn Engine<Type = T, Value = V, TracingOperation = O, LinearOperation = L>,
    ) -> Result<Self::Traced, TraceError> {
        let structure = self.parameter_structure();
        Self::Traced::from_parameters(
            structure,
            self.into_parameters().map(|r#type| {
                let atom = builder.borrow_mut().add_input_abstract(r#type);
                JitTracer::<T, V, O, L>::from_engine(atom, builder.clone(), staging_error.clone(), engine)
            }),
        )
        .map_err(TraceError::from)
    }

    fn from_type_traced(traced_output: Self::Traced) -> Result<(Self, Vec<AtomId>), TraceError> {
        let output_structure = traced_output.parameter_structure();
        let traced_outputs = traced_output.into_parameters().collect::<Vec<_>>();
        let output_types = Value::from_parameters(
            output_structure,
            traced_outputs.iter().map(|output| output.tpe().into_owned()).collect::<Vec<_>>(),
        )?;
        let output_atoms = traced_outputs.into_iter().map(|output| output.atom).collect::<Vec<_>>();
        Ok((output_types, output_atoms))
    }
}

/// Tracer used while staging JIT programs.
#[derive(Clone)]
pub struct JitTracer<
    T: Type + Display,
    V: Traceable<T> + Parameter,
    O: Clone + 'static = ProgramOpRef<V>,
    L: Clone + 'static = crate::tracing_v2::LinearProgramOpRef<V>,
> {
    atom: AtomId,
    builder: Rc<RefCell<GraphBuilder<O, T, V>>>,
    staging_error: Rc<RefCell<Option<TraceError>>>,
    engine: *const dyn Engine<Type = T, Value = V, TracingOperation = O, LinearOperation = L>,
    marker: PhantomData<fn() -> L>,
}

impl<T: Type + Display, V: Traceable<T>, O: Clone + 'static, L: Clone + 'static> Parameter for JitTracer<T, V, O, L> {}

impl<T: Type + Display, V: Traceable<T>, O: Clone + 'static, L: Clone + 'static> std::fmt::Debug
    for JitTracer<T, V, O, L>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("JitTracer").field("atom", &self.atom).finish_non_exhaustive()
    }
}

impl<T: Type + Display, V: Traceable<T>, O: Clone + 'static, L: Clone + 'static> JitTracer<T, V, O, L> {
    #[doc(hidden)]
    #[inline]
    pub fn atom(&self) -> AtomId {
        self.atom
    }

    #[inline]
    pub fn builder_handle(&self) -> Rc<RefCell<GraphBuilder<O, T, V>>> {
        self.builder.clone()
    }

    #[inline]
    pub fn staging_error_handle(&self) -> Rc<RefCell<Option<TraceError>>> {
        self.staging_error.clone()
    }

    #[inline]
    pub fn engine(&self) -> &dyn Engine<Type = T, Value = V, TracingOperation = O, LinearOperation = L> {
        // Safe because traced values are confined to the tracing scope: all public tracing entry
        // points require the shared builder to be uniquely reclaimed before they return, so no
        // tracer can outlive the borrowed engine captured here.
        unsafe { &*self.engine }
    }

    #[inline]
    pub fn from_engine(
        atom: AtomId,
        builder: Rc<RefCell<GraphBuilder<O, T, V>>>,
        staging_error: Rc<RefCell<Option<TraceError>>>,
        engine: &dyn Engine<Type = T, Value = V, TracingOperation = O, LinearOperation = L>,
    ) -> Self {
        Self::from_staged_parts(atom, builder, staging_error, engine)
    }

    #[inline]
    pub fn from_staged_parts(
        atom: AtomId,
        builder: Rc<RefCell<GraphBuilder<O, T, V>>>,
        staging_error: Rc<RefCell<Option<TraceError>>>,
        engine: &dyn Engine<Type = T, Value = V, TracingOperation = O, LinearOperation = L>,
    ) -> Self {
        // Safe because traced values are confined to the tracing scope and all public tracing
        // entry points require reclaiming the shared builder before they return, so no staged
        // tracer can outlive the engine reference captured here.
        let engine = unsafe {
            std::mem::transmute::<
                &dyn Engine<Type = T, Value = V, TracingOperation = O, LinearOperation = L>,
                *const dyn Engine<Type = T, Value = V, TracingOperation = O, LinearOperation = L>,
            >(engine)
        };
        Self { atom, builder, staging_error, engine, marker: PhantomData }
    }

    pub fn apply_staged_op(inputs: &[Self], op: O) -> Result<Vec<Self>, TraceError>
    where
        O: Op<T>,
    {
        if inputs.is_empty() {
            return Err(TraceError::EmptyParameterizedValue);
        }

        let builder = inputs[0].builder.clone();
        let staging_error = inputs[0].staging_error.clone();
        if inputs.iter().skip(1).any(|input| !Rc::ptr_eq(&builder, &input.builder)) {
            return Err(TraceError::InternalInvariantViolation(
                "jit tracer inputs for one staged op must share the same builder",
            ));
        }
        if inputs.iter().skip(1).any(|input| !Rc::ptr_eq(&staging_error, &input.staging_error)) {
            return Err(TraceError::InternalInvariantViolation(
                "jit tracer inputs for one staged op must share the same staging error handle",
            ));
        }

        let input_atoms = inputs.iter().map(|input| input.atom).collect::<Vec<_>>();
        let output_count = {
            let builder_borrow = builder.borrow();
            match op.abstract_eval(
                input_atoms
                    .iter()
                    .map(|input| {
                        builder_borrow.atom(*input).expect("jit tracer input atoms should exist").tpe().into_owned()
                    })
                    .collect::<Vec<_>>()
                    .as_slice(),
            ) {
                Ok(outputs) => outputs.len(),
                Err(error) => {
                    if staging_error.borrow().is_none() {
                        *staging_error.borrow_mut() = Some(error);
                    }
                    1
                }
            }
        };
        let output_atoms = if staging_error.borrow().is_some() {
            vec![inputs[0].atom; output_count]
        } else {
            match builder.borrow_mut().add_equation_abstract(op, input_atoms) {
                Ok(outputs) => outputs,
                Err(error) => {
                    *staging_error.borrow_mut() = Some(error);
                    vec![inputs[0].atom; output_count]
                }
            }
        };

        Ok(output_atoms
            .into_iter()
            .map(|atom| Self {
                atom,
                builder: builder.clone(),
                staging_error: staging_error.clone(),
                engine: inputs[0].engine,
                marker: PhantomData,
            })
            .collect())
    }

    pub fn unary(self, op: O) -> Self
    where
        O: Op<T>,
    {
        Self::apply_staged_op(std::slice::from_ref(&self), op)
            .expect("unary traced staging should preserve non-empty inputs")
            .into_iter()
            .next()
            .expect("unary traced staging should produce one output")
    }

    pub fn binary(self, rhs: Self, op: O) -> Self
    where
        O: Op<T>,
    {
        debug_assert!(Rc::ptr_eq(&self.builder, &rhs.builder));
        debug_assert!(Rc::ptr_eq(&self.staging_error, &rhs.staging_error));
        Self::apply_staged_op(&[self, rhs], op)
            .expect("binary traced staging should preserve non-empty inputs")
            .into_iter()
            .next()
            .expect("binary traced staging should produce one output")
    }
}

impl<T: Type + Display, V: Traceable<T>, O: Clone + 'static, L: Clone + 'static> Typed<T> for JitTracer<T, V, O, L> {
    #[inline]
    fn tpe(&self) -> Cow<'_, T> {
        Cow::Owned(
            self.builder
                .borrow()
                .atom(self.atom)
                .expect("jit tracer atom should exist in its staging builder")
                .tpe()
                .into_owned(),
        )
    }
}

impl<T: Type + Display + 'static, V: Traceable<T>, O: Clone + 'static, L: Clone + 'static> Traceable<T>
    for JitTracer<T, V, O, L>
{
}

impl<V: Traceable<ArrayType> + ZeroLike, O: Clone + 'static, L: Clone + 'static> ZeroLike
    for JitTracer<ArrayType, V, O, L>
{
    #[inline]
    fn zero_like(&self) -> Self {
        let value = self.engine().zero(&self.tpe().into_owned());
        let atom = self.builder.borrow_mut().add_constant(value.clone());
        Self {
            atom,
            builder: self.builder.clone(),
            staging_error: self.staging_error.clone(),
            engine: self.engine,
            marker: PhantomData,
        }
    }
}

impl<V: Traceable<ArrayType> + OneLike, O: Clone + 'static, L: Clone + 'static> OneLike
    for JitTracer<ArrayType, V, O, L>
{
    #[inline]
    fn one_like(&self) -> Self {
        let value = self.engine().one(&self.tpe().into_owned());
        let atom = self.builder.borrow_mut().add_constant(value.clone());
        Self {
            atom,
            builder: self.builder.clone(),
            staging_error: self.staging_error.clone(),
            engine: self.engine,
            marker: PhantomData,
        }
    }
}

impl<V: Traceable<ArrayType>, O: AddTracingOperation<ArrayType, V> + 'static, L: Clone + 'static> Add
    for JitTracer<ArrayType, V, O, L>
where
    O: Op<ArrayType>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.binary(rhs, O::add_op())
    }
}

impl<V: Traceable<ArrayType>, O: MulTracingOperation<ArrayType, V> + 'static, L: Clone + 'static> Mul
    for JitTracer<ArrayType, V, O, L>
where
    O: Op<ArrayType>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.binary(rhs, O::mul_op())
    }
}

impl<V: Traceable<ArrayType>, O: NegTracingOperation<ArrayType, V> + 'static, L: Clone + 'static> Neg
    for JitTracer<ArrayType, V, O, L>
where
    O: Op<ArrayType>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self.unary(O::neg_op())
    }
}

/// Staged function returned by [`jit`].
///
/// In the current prototype this type stores only the staged graph and replays it with the built-in interpreter.
/// Later, once a concrete backend exists, it can grow additional fields that hold backend-specific compiled artifacts
/// while keeping the same high-level API shape.
pub struct CompiledFunction<
    T: Type,
    V: Typed<T> + Parameter,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
    O = ProgramOpRef<V>,
> {
    program: Program<T, V, Input, Output, O>,
    marker: PhantomData<fn(Input) -> Output>,
}

impl<
    T: Type,
    V: Traceable<T>,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    O: Clone,
> Clone for CompiledFunction<T, V, Input, Output, O>
{
    fn clone(&self) -> Self {
        Self { program: self.program.clone(), marker: PhantomData }
    }
}

impl<T: Type, V: Traceable<T>, Input: Parameterized<V>, Output: Parameterized<V>, O: Clone>
    CompiledFunction<T, V, Input, Output, O>
{
    #[inline]
    pub fn from_graph(graph: crate::tracing_v2::Graph<O, T, V, Input, Output>) -> Self {
        Self::from_program(Program::from_graph(graph))
    }

    #[inline]
    pub fn from_program(program: Program<T, V, Input, Output, O>) -> Self {
        Self { program, marker: PhantomData }
    }

    /// Returns the staged graph backing this compiled function.
    #[inline]
    pub fn graph(&self) -> &crate::tracing_v2::Graph<O, T, V, Input, Output> {
        self.program.graph()
    }

    /// Returns the staged program backing this compiled function.
    #[inline]
    pub fn program(&self) -> &Program<T, V, Input, Output, O> {
        &self.program
    }

    /// Replays the staged graph on concrete input values.
    pub fn call(&self, input: Input) -> Result<Output, TraceError>
    where
        O: InterpretableOp<T, V>,
        Input::ParameterStructure: PartialEq,
        Output::ParameterStructure: Clone,
    {
        self.program.call(input)
    }
}

impl<T: Type + Display, V: Traceable<T>, Input: Parameterized<V>, Output: Parameterized<V>, O: Clone + Display> Display
    for CompiledFunction<T, V, Input, Output, O>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.program, formatter)
    }
}

fn try_trace_program_with_options<E: Engine<Type = ArrayType, Value = V>, F, Input, Output, V>(
    engine: &E,
    function: F,
    input: Input,
    simplify_program: bool,
) -> Result<(Output, Program<ArrayType, V, Input, Output, E::TracingOperation>), TraceError>
where
    V: Traceable<ArrayType>,
    Input: TraceInput<V, E::TracingOperation, E::LinearOperation>,
    Output: TraceOutput<V, E::TracingOperation, E::LinearOperation>,
    F: FnOnce(Input::Traced) -> Result<Output::Traced, TraceError>,
    E::TracingOperation: InterpretableOp<ArrayType, V>,
    Input::ParameterStructure: PartialEq,
    Output::ParameterStructure: Clone,
{
    let input_structure = input.parameter_structure();
    let input_values = input.into_parameters().collect::<Vec<_>>();
    let builder = Rc::new(RefCell::new(ProgramBuilder::<V, E::TracingOperation>::new()));
    let staging_error = Rc::new(RefCell::new(None));
    let concrete_input = Input::from_parameters(input_structure.clone(), input_values.clone())?;
    let traced_input = concrete_input.into_traced(builder.clone(), staging_error.clone(), engine)?;

    let (output_structure, outputs) = {
        let traced_output = function(traced_input)?;
        Output::from_traced(traced_output)?
    };

    if let Some(error) = staging_error.borrow_mut().take() {
        return Err(error);
    }
    let builder = match Rc::try_unwrap(builder) {
        Ok(builder) => builder.into_inner(),
        Err(_) => return Err(TraceError::InternalInvariantViolation("jit builder escaped the tracing scope")),
    };
    let program = Program::from_graph(builder.build::<Input, Output>(outputs, input_structure, output_structure));
    let program = if simplify_program { program.simplify()? } else { program };
    let concrete_input = Input::from_parameters(program.graph().input_structure().clone(), input_values)?;
    Ok((program.call(concrete_input)?, program))
}

fn try_trace_program_with_operation_options<F, Input, Output, V, O, L>(
    engine: &dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L>,
    function: F,
    input: Input,
    simplify_program: bool,
) -> Result<(Output, Program<ArrayType, V, Input, Output, O>), TraceError>
where
    V: Traceable<ArrayType>,
    O: Clone + 'static + InterpretableOp<ArrayType, V>,
    L: Clone + 'static,
    Input: TraceInput<V, O, L>,
    Output: TraceOutput<V, O, L>,
    F: FnOnce(Input::Traced) -> Result<Output::Traced, TraceError>,
    Input::ParameterStructure: PartialEq,
    Output::ParameterStructure: Clone,
{
    let input_structure = input.parameter_structure();
    let input_values = input.into_parameters().collect::<Vec<_>>();
    let builder = Rc::new(RefCell::new(GraphBuilder::<O, ArrayType, V>::new()));
    let staging_error = Rc::new(RefCell::new(None));
    let concrete_input = Input::from_parameters(input_structure.clone(), input_values.clone())?;
    let traced_input = concrete_input.into_traced(builder.clone(), staging_error.clone(), engine)?;

    let (output_structure, outputs) = {
        let traced_output = function(traced_input)?;
        Output::from_traced(traced_output)?
    };

    if let Some(error) = staging_error.borrow_mut().take() {
        return Err(error);
    }
    let builder = match Rc::try_unwrap(builder) {
        Ok(builder) => builder.into_inner(),
        Err(_) => return Err(TraceError::InternalInvariantViolation("jit builder escaped the tracing scope")),
    };
    let program = Program::from_graph(builder.build::<Input, Output>(outputs, input_structure, output_structure));
    let program = if simplify_program { program.simplify()? } else { program };
    let concrete_input = Input::from_parameters(program.graph().input_structure().clone(), input_values)?;
    Ok((program.call(concrete_input)?, program))
}

fn try_trace_program_from_types_with_options<E: Engine<Type = T, Value = V>, F, Input, Output, T, V>(
    engine: &E,
    function: F,
    input_types: Input,
    simplify_program: bool,
) -> Result<(Output, Program<T, V, Input::Staged, Output::Staged, E::TracingOperation>), TraceError>
where
    T: Type + Display + Parameter,
    V: Traceable<T>,
    Input: TypeTracing<T, V, E::TracingOperation, E::LinearOperation>,
    Output: TypeTracing<T, V, E::TracingOperation, E::LinearOperation>,
    F: FnOnce(Input::Traced) -> Result<Output::Traced, TraceError>,
    E::TracingOperation: Op<T>,
{
    let input_structure = input_types.parameter_structure();
    let builder = Rc::new(RefCell::new(GraphBuilder::<E::TracingOperation, T, V>::new()));
    let staging_error = Rc::new(RefCell::new(None));
    let traced_input = input_types.into_type_traced(builder.clone(), staging_error.clone(), engine)?;

    let (output_structure, output_types, outputs) = {
        let traced_output = function(traced_input)?;
        let (output_types, outputs) = Output::from_type_traced(traced_output)?;
        let output_structure = output_types.parameter_structure();
        (output_structure, output_types, outputs)
    };

    if let Some(error) = staging_error.borrow_mut().take() {
        return Err(error);
    }
    let builder = match Rc::try_unwrap(builder) {
        Ok(builder) => builder.into_inner(),
        Err(_) => return Err(TraceError::InternalInvariantViolation("jit builder escaped the tracing scope")),
    };
    let program =
        Program::from_graph(builder.build::<Input::Staged, Output::Staged>(outputs, input_structure, output_structure));
    let program = if simplify_program { program.simplify()? } else { program };
    Ok((output_types, program))
}

fn try_trace_program_from_types_with_operation_options<F, Input, Output, T, V, O, L>(
    engine: &dyn Engine<Type = T, Value = V, TracingOperation = O, LinearOperation = L>,
    function: F,
    input_types: Input,
    simplify_program: bool,
) -> Result<(Output, Program<T, V, Input::Staged, Output::Staged, O>), TraceError>
where
    T: Type + Display + Parameter,
    V: Traceable<T>,
    O: Clone + 'static + Op<T>,
    L: Clone + 'static,
    Input: TypeTracing<T, V, O, L>,
    Output: TypeTracing<T, V, O, L>,
    F: FnOnce(Input::Traced) -> Result<Output::Traced, TraceError>,
{
    let input_structure = input_types.parameter_structure();
    let builder = Rc::new(RefCell::new(GraphBuilder::<O, T, V>::new()));
    let staging_error = Rc::new(RefCell::new(None));
    let traced_input = input_types.into_type_traced(builder.clone(), staging_error.clone(), engine)?;

    let (output_structure, output_types, outputs) = {
        let traced_output = function(traced_input)?;
        let (output_types, outputs) = Output::from_type_traced(traced_output)?;
        let output_structure = output_types.parameter_structure();
        (output_structure, output_types, outputs)
    };

    if let Some(error) = staging_error.borrow_mut().take() {
        return Err(error);
    }
    let builder = match Rc::try_unwrap(builder) {
        Ok(builder) => builder.into_inner(),
        Err(_) => return Err(TraceError::InternalInvariantViolation("jit builder escaped the tracing scope")),
    };
    let program =
        Program::from_graph(builder.build::<Input::Staged, Output::Staged>(outputs, input_structure, output_structure));
    let program = if simplify_program { program.simplify()? } else { program };
    Ok((output_types, program))
}

/// Stages `function` using the staged op set selected by `engine`.
pub fn try_trace_program<E, F, Input, Output, V>(
    engine: &E,
    function: F,
    input: Input,
) -> Result<(Output, Program<ArrayType, V, Input, Output, E::TracingOperation>), TraceError>
where
    E: Engine<Type = ArrayType, Value = V>,
    V: Traceable<ArrayType>,
    Input: TraceInput<V, E::TracingOperation, E::LinearOperation>,
    Output: TraceOutput<V, E::TracingOperation, E::LinearOperation>,
    F: FnOnce(Input::Traced) -> Result<Output::Traced, TraceError>,
    E::TracingOperation: InterpretableOp<ArrayType, V>,
    Input::ParameterStructure: PartialEq,
    Output::ParameterStructure: Clone,
{
    try_trace_program_with_operation_options(engine, function, input, true)
}

/// Stages `function` using one explicit staged ordinary operation type.
pub(crate) fn try_trace_program_for_operation<F, Input, Output, V, O, L>(
    engine: &dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L>,
    function: F,
    input: Input,
) -> Result<(Output, Program<ArrayType, V, Input, Output, O>), TraceError>
where
    V: Traceable<ArrayType>,
    O: Clone + 'static,
    L: Clone + 'static,
    Input: TraceInput<V, O, L>,
    Output: TraceOutput<V, O, L>,
    F: FnOnce(Input::Traced) -> Result<Output::Traced, TraceError>,
    O: InterpretableOp<ArrayType, V>,
    Input::ParameterStructure: PartialEq,
    Output::ParameterStructure: Clone,
{
    try_trace_program_with_operation_options(engine, function, input, true)
}

/// Stages `function` directly from type metadata using one explicit staged ordinary operation type.
pub fn try_trace_program_from_types<E, F, Input, Output, T, V>(
    engine: &E,
    function: F,
    input_types: Input,
) -> Result<(Output, Program<T, V, Input::Staged, Output::Staged, E::TracingOperation>), TraceError>
where
    E: Engine<Type = T, Value = V>,
    T: Type + Display + Parameter,
    V: Traceable<T>,
    Input: TypeTracing<T, V, E::TracingOperation, E::LinearOperation>,
    Output: TypeTracing<T, V, E::TracingOperation, E::LinearOperation>,
    F: FnOnce(Input::Traced) -> Result<Output::Traced, TraceError>,
    E::TracingOperation: Op<T>,
{
    try_trace_program_from_types_with_options(engine, function, input_types, true)
}

/// Stages `function` directly from type metadata using one explicit staged ordinary operation type.
pub(crate) fn try_trace_program_from_types_for_operation<F, Input, Output, T, V, O, L>(
    engine: &dyn Engine<Type = T, Value = V, TracingOperation = O, LinearOperation = L>,
    function: F,
    input_types: Input,
) -> Result<(Output, Program<T, V, Input::Staged, Output::Staged, O>), TraceError>
where
    T: Type + Display + Parameter,
    V: Traceable<T>,
    O: Clone + 'static + Op<T>,
    L: Clone + 'static,
    Input: TypeTracing<T, V, O, L>,
    Output: TypeTracing<T, V, O, L>,
    F: FnOnce(Input::Traced) -> Result<Output::Traced, TraceError>,
{
    try_trace_program_from_types_with_operation_options(engine, function, input_types, true)
}

/// Stages `function` as a graph using the staged op set selected by `engine`.
pub fn try_jit<E, F, Input, Output, V>(
    engine: &E,
    function: F,
    input: Input,
) -> Result<(Output, CompiledFunction<ArrayType, V, Input, Output, E::TracingOperation>), TraceError>
where
    E: Engine<Type = ArrayType, Value = V>,
    V: Traceable<ArrayType>,
    Input: TraceInput<V, E::TracingOperation, E::LinearOperation>,
    Output: TraceOutput<V, E::TracingOperation, E::LinearOperation>,
    F: FnOnce(Input::Traced) -> Result<Output::Traced, TraceError>,
    E::TracingOperation: InterpretableOp<ArrayType, V>,
    Input::ParameterStructure: PartialEq,
    Output::ParameterStructure: Clone,
{
    let (output, program) = try_trace_program_with_options(engine, function, input, true)?;
    Ok((output, CompiledFunction::from_program(program)))
}

/// Stages `function` as a graph using one explicit staged ordinary operation type.
pub(crate) fn try_jit_for_operation<F, Input, Output, V, O, L>(
    engine: &dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L>,
    function: F,
    input: Input,
) -> Result<(Output, CompiledFunction<ArrayType, V, Input, Output, O>), TraceError>
where
    V: Traceable<ArrayType>,
    O: Clone + 'static,
    L: Clone + 'static,
    Input: TraceInput<V, O, L>,
    Output: TraceOutput<V, O, L>,
    F: FnOnce(Input::Traced) -> Result<Output::Traced, TraceError>,
    O: InterpretableOp<ArrayType, V>,
    Input::ParameterStructure: PartialEq,
    Output::ParameterStructure: Clone,
{
    let (output, program) = try_trace_program_for_operation(engine, function, input)?;
    Ok((output, CompiledFunction::from_program(program)))
}

/// Stages `function` directly from type metadata as one graph.
pub fn try_jit_from_types<E, F, Input, Output, T, V>(
    engine: &E,
    function: F,
    input_types: Input,
) -> Result<(Output, CompiledFunction<T, V, Input::Staged, Output::Staged, E::TracingOperation>), TraceError>
where
    E: Engine<Type = T, Value = V>,
    T: Type + Display + Parameter,
    V: Traceable<T>,
    Input: TypeTracing<T, V, E::TracingOperation, E::LinearOperation>,
    Output: TypeTracing<T, V, E::TracingOperation, E::LinearOperation>,
    F: FnOnce(Input::Traced) -> Result<Output::Traced, TraceError>,
    E::TracingOperation: Op<T>,
{
    let (output_types, program) = try_trace_program_from_types(engine, function, input_types)?;
    Ok((output_types, CompiledFunction::from_program(program)))
}

/// Stages `function` as a graph and returns both the eager output and the staged program selected
/// by `engine`.
///
/// The returned [`CompiledFunction`] currently stores only the staged graph. Later, once a concrete backend exists,
/// this type can be extended to carry backend-specific compilation artifacts alongside that graph.
pub fn jit<E, F, Input, Output, V>(
    engine: &E,
    function: F,
    input: Input,
) -> Result<(Output, CompiledFunction<ArrayType, V, Input, Output, E::TracingOperation>), TraceError>
where
    E: Engine<Type = ArrayType, Value = V>,
    V: Traceable<ArrayType>,
    Input: TraceInput<V, E::TracingOperation, E::LinearOperation>,
    Output: TraceOutput<V, E::TracingOperation, E::LinearOperation>,
    F: FnOnce(Input::Traced) -> Output::Traced,
    E::TracingOperation: InterpretableOp<ArrayType, V>,
    Input::ParameterStructure: PartialEq,
    Output::ParameterStructure: Clone,
{
    try_jit(engine, |traced_input| Ok(function(traced_input)), input)
}

/// Stages `function` directly from type metadata and returns both the inferred output types and the
/// staged program.
pub fn jit_from_types<E, F, Input, Output, T, V>(
    engine: &E,
    function: F,
    input_types: Input,
) -> Result<(Output, CompiledFunction<T, V, Input::Staged, Output::Staged, E::TracingOperation>), TraceError>
where
    E: Engine<Type = T, Value = V>,
    T: Type + Display + Parameter,
    V: Traceable<T>,
    Input: TypeTracing<T, V, E::TracingOperation, E::LinearOperation>,
    Output: TypeTracing<T, V, E::TracingOperation, E::LinearOperation>,
    F: FnOnce(Input::Traced) -> Output::Traced,
    E::TracingOperation: Op<T>,
{
    try_jit_from_types(engine, |traced_input| Ok(function(traced_input)), input_types)
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use indoc::indoc;

    use crate::{
        parameters::Placeholder,
        tracing_v2::{ProgramBuilder, Sin, engine::ArrayScalarEngine, test_support},
    };

    use super::*;

    #[test]
    fn jit_tracer_zero_like_adds_constant_atoms() {
        let builder = Rc::new(RefCell::new(ProgramBuilder::<f64>::new()));
        let staging_error = Rc::new(RefCell::new(None));
        let atom = builder.borrow_mut().add_input(&3.0f64);
        let engine = ArrayScalarEngine::<f64>::new();
        let tracer: JitTracer<ArrayType, f64> = JitTracer::from_engine(atom, builder, staging_error, &engine);
        let zero = tracer.zero_like();
        assert_eq!(zero.tpe().into_owned(), ArrayType::scalar(crate::types::DataType::F64));
        assert!(zero.atom > atom);

        let graph = zero.builder.borrow().clone().build::<f64, f64>(vec![zero.atom], Placeholder, Placeholder);
        assert_eq!(
            graph.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = const
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn compiled_function_replays_staged_graphs() {
        let engine = ArrayScalarEngine::<f64>::new();
        let (output, compiled): (f64, CompiledFunction<ArrayType, f64, f64, f64>) = jit(
            &engine,
            |x: JitTracer<ArrayType, f64>| {
                let squared = x.clone() * x.clone();
                squared + x.sin()
            },
            2.0f64,
        )
        .unwrap();

        assert_eq!(output, 2.0f64 * 2.0f64 + 2.0f64.sin());
        assert_eq!(compiled.call(0.5f64).unwrap(), 0.5f64 * 0.5f64 + 0.5f64.sin());
        assert_eq!(compiled.graph().input_atoms().len(), 1);
        assert_eq!(
            compiled.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = mul %0 %0
                    %2:f64[] = sin %0
                    %3:f64[] = add %1 %2
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn jit_returns_abstract_eval_errors_instead_of_panicking() {
        use ryft_macros::Parameter;

        use crate::{
            tracing_v2::{Cos, MatrixOps, OneLike, Sin, ZeroLike, operations::reshape::ReshapeOps},
            types::{ArrayType, DataType, Typed},
        };

        #[derive(Clone, Debug, Parameter)]
        struct TestAbstractValue {
            r#type: ArrayType,
        }

        impl Typed<ArrayType> for TestAbstractValue {
            fn tpe(&self) -> Cow<'_, ArrayType> {
                Cow::Borrowed(&self.r#type)
            }
        }

        impl Traceable<ArrayType> for TestAbstractValue {
            fn is_zero(&self) -> bool {
                false
            }

            fn is_one(&self) -> bool {
                false
            }
        }

        impl crate::tracing_v2::Value<ArrayType> for TestAbstractValue {}

        impl Add for TestAbstractValue {
            type Output = Self;

            fn add(self, _rhs: Self) -> Self::Output {
                self
            }
        }

        impl Mul for TestAbstractValue {
            type Output = Self;

            fn mul(self, _rhs: Self) -> Self::Output {
                self
            }
        }

        impl Neg for TestAbstractValue {
            type Output = Self;

            fn neg(self) -> Self::Output {
                self
            }
        }

        impl Sin for TestAbstractValue {
            fn sin(self) -> Self {
                self
            }
        }

        impl Cos for TestAbstractValue {
            fn cos(self) -> Self {
                self
            }
        }

        impl ZeroLike for TestAbstractValue {
            fn zero_like(&self) -> Self {
                self.clone()
            }
        }

        impl OneLike for TestAbstractValue {
            fn one_like(&self) -> Self {
                self.clone()
            }
        }

        impl MatrixOps for TestAbstractValue {
            fn matmul(self, _rhs: Self) -> Self {
                self
            }

            fn transpose_matrix(self) -> Self {
                self
            }
        }

        impl ReshapeOps for TestAbstractValue {
            fn reshape(self, _target_shape: crate::types::Shape) -> Result<Self, TraceError> {
                Ok(self)
            }
        }

        struct TestEngine;

        impl crate::tracing_v2::engine::Engine for TestEngine {
            type Type = ArrayType;
            type Value = TestAbstractValue;
            type TracingOperation = crate::tracing_v2::PrimitiveOp<ArrayType, TestAbstractValue>;
            type LinearOperation = crate::tracing_v2::LinearPrimitiveOp<ArrayType, TestAbstractValue>;

            fn zero(&self, r#type: &ArrayType) -> TestAbstractValue {
                TestAbstractValue { r#type: r#type.clone() }
            }

            fn one(&self, r#type: &ArrayType) -> TestAbstractValue {
                TestAbstractValue { r#type: r#type.clone() }
            }
        }

        let result: Result<
            (
                TestAbstractValue,
                CompiledFunction<
                    ArrayType,
                    TestAbstractValue,
                    (TestAbstractValue, TestAbstractValue),
                    TestAbstractValue,
                >,
            ),
            TraceError,
        > = jit(
            &TestEngine,
            |inputs: (JitTracer<ArrayType, TestAbstractValue>, JitTracer<ArrayType, TestAbstractValue>)| {
                inputs.0 + inputs.1
            },
            (
                TestAbstractValue { r#type: ArrayType::scalar(DataType::F32) },
                TestAbstractValue { r#type: ArrayType::scalar(DataType::F64) },
            ),
        );

        assert!(matches!(result, Err(TraceError::IncompatibleAbstractValues { op: "add" })));
    }

    #[test]
    fn compiled_function_display_delegates_to_the_underlying_graph() {
        let engine = ArrayScalarEngine::<f64>::new();
        let (_, compiled): (f64, CompiledFunction<ArrayType, f64, f64, f64>) =
            jit(&engine, |x: JitTracer<ArrayType, f64>| x.clone() * x.clone() + x.sin(), 2.0f64).unwrap();

        assert_eq!(
            compiled.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = mul %0 %0
                    %2:f64[] = sin %0
                    %3:f64[] = add %1 %2
                in (%3)
            "}
            .trim_end(),
        );
        assert_eq!(compiled.to_string(), compiled.graph().to_string());
        test_support::assert_bilinear_jit_rendering();
    }
}
