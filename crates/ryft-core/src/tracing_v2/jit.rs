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

use ryft_macros::Parameter;

use crate::{
    parameters::{Parameter, Parameterized, ParameterizedFamily},
    tracing_v2::{
        GraphBuilder, InterpretableOp, OneLike, TraceError, Traceable, ZeroLike,
        engine::Engine,
        graph::AtomId,
        ops::{CoreOpSet, Op, OpSet, SupportsAdd, SupportsMul, SupportsNeg},
        program::{Program, ProgramBuilderFor, ProgramOpRef},
    },
    types::{ArrayType, Type, Typed},
};

/// Input family that can be rebuilt with traced leaves for one op set.
#[doc(hidden)]
pub trait TraceInput<V: Traceable<ArrayType>, S: OpSet<ArrayType, V>>:
    Parameterized<V, ParameterStructure: Clone>
{
    /// Traced version of this input family for one op set.
    type Traced: Parameterized<JitTracer<ArrayType, V, S>, ParameterStructure = Self::ParameterStructure>;

    /// Rebuilds `self` with traced leaves owned by `builder`.
    fn into_traced(
        self,
        builder: Rc<RefCell<GraphBuilder<S::JitOp, ArrayType, V>>>,
        staging_error: Rc<RefCell<Option<TraceError>>>,
    ) -> Result<Self::Traced, TraceError>;
}

impl<T, V, S> TraceInput<V, S> for T
where
    T: Parameterized<V, ParameterStructure: Clone>,
    V: Traceable<ArrayType>,
    S: OpSet<ArrayType, V>,
    T::Family: ParameterizedFamily<JitTracer<ArrayType, V, S>>,
{
    type Traced = T::To<JitTracer<ArrayType, V, S>>;

    fn into_traced(
        self,
        builder: Rc<RefCell<GraphBuilder<S::JitOp, ArrayType, V>>>,
        staging_error: Rc<RefCell<Option<TraceError>>>,
    ) -> Result<Self::Traced, TraceError> {
        let structure = self.parameter_structure();
        Self::Traced::from_parameters(
            structure,
            self.into_parameters().map(|value| {
                let atom = builder.borrow_mut().add_input(&value);
                JitTracer::<ArrayType, V, S> {
                    value,
                    atom,
                    builder: builder.clone(),
                    staging_error: staging_error.clone(),
                }
            }),
        )
        .map_err(TraceError::from)
    }
}

/// Output family that can be lowered back to concrete leaves after tracing.
#[doc(hidden)]
pub trait TraceOutput<V: Traceable<ArrayType>, S: OpSet<ArrayType, V>>:
    Parameterized<V, ParameterStructure: Clone>
{
    /// Traced version of this output family for one op set.
    type Traced: Parameterized<JitTracer<ArrayType, V, S>, ParameterStructure = Self::ParameterStructure>;

    /// Lowers one traced output back to concrete values and the corresponding staged output atoms.
    fn from_traced(traced_output: Self::Traced) -> Result<(Self, Vec<AtomId>), TraceError>;
}

impl<T, V, S> TraceOutput<V, S> for T
where
    T: Parameterized<V, ParameterStructure: Clone>,
    V: Traceable<ArrayType>,
    S: OpSet<ArrayType, V>,
    T::Family: ParameterizedFamily<JitTracer<ArrayType, V, S>>,
{
    type Traced = T::To<JitTracer<ArrayType, V, S>>;

    fn from_traced(traced_output: Self::Traced) -> Result<(Self, Vec<AtomId>), TraceError> {
        let output_structure = traced_output.parameter_structure();
        let traced_outputs = traced_output.into_parameters().collect::<Vec<_>>();
        let output_value = T::from_parameters(
            output_structure.clone(),
            traced_outputs.iter().map(|output| output.value.clone()).collect::<Vec<_>>(),
        )?;
        let output_atoms = traced_outputs.into_iter().map(|output| output.atom).collect::<Vec<_>>();
        Ok((output_value, output_atoms))
    }
}

/// Tracer used while staging JIT programs.
#[derive(Clone, Parameter)]
pub struct JitTracer<T: Type + Display, V: Traceable<T> + Parameter, S: OpSet<T, V> = CoreOpSet> {
    /// Concrete value obtained during eager execution of the staged computation.
    pub value: V,
    atom: AtomId,
    builder: Rc<RefCell<GraphBuilder<S::JitOp, T, V>>>,
    staging_error: Rc<RefCell<Option<TraceError>>>,
}

impl<T: Type + Display, V: Traceable<T>, S: OpSet<T, V>> std::fmt::Debug for JitTracer<T, V, S> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("JitTracer").field("atom", &self.atom).finish_non_exhaustive()
    }
}

impl<T: Type + Display, V: Traceable<T>, S: OpSet<T, V>> JitTracer<T, V, S> {
    #[doc(hidden)]
    #[inline]
    pub fn atom(&self) -> AtomId {
        self.atom
    }

    #[inline]
    pub fn builder_handle(&self) -> Rc<RefCell<GraphBuilder<S::JitOp, T, V>>> {
        self.builder.clone()
    }

    #[inline]
    pub fn staging_error_handle(&self) -> Rc<RefCell<Option<TraceError>>> {
        self.staging_error.clone()
    }

    #[inline]
    pub fn from_staged_parts(
        value: V,
        atom: AtomId,
        builder: Rc<RefCell<GraphBuilder<S::JitOp, T, V>>>,
        staging_error: Rc<RefCell<Option<TraceError>>>,
    ) -> Self {
        Self { value, atom, builder, staging_error }
    }

    pub fn apply_staged_op(inputs: &[Self], op: S::JitOp, output_values: Vec<V>) -> Result<Vec<Self>, TraceError>
    where
        S::JitOp: Op<T>,
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
        let output_atoms = if staging_error.borrow().is_some() {
            vec![inputs[0].atom; output_values.len()]
        } else {
            match builder.borrow_mut().add_equation_with_output_values(op, input_atoms, output_values.clone()) {
                Ok(outputs) => outputs,
                Err(error) => {
                    *staging_error.borrow_mut() = Some(error);
                    vec![inputs[0].atom; output_values.len()]
                }
            }
        };

        Ok(output_values
            .into_iter()
            .zip(output_atoms)
            .map(|(value, atom)| Self { value, atom, builder: builder.clone(), staging_error: staging_error.clone() })
            .collect())
    }

    pub fn unary(self, op: S::JitOp, apply: impl FnOnce(V) -> V) -> Self
    where
        S::JitOp: Op<T>,
    {
        let value = apply(self.value);
        let atom = if self.staging_error.borrow().is_some() {
            self.atom
        } else {
            match self.builder.borrow_mut().add_equation_with_output_values(op, vec![self.atom], vec![value.clone()]) {
                Ok(outputs) => outputs[0],
                Err(error) => {
                    *self.staging_error.borrow_mut() = Some(error);
                    self.atom
                }
            }
        };
        Self { value, atom, builder: self.builder, staging_error: self.staging_error }
    }

    pub fn binary(self, rhs: Self, op: S::JitOp, apply: impl FnOnce(V, V) -> V) -> Self
    where
        S::JitOp: Op<T>,
    {
        debug_assert!(Rc::ptr_eq(&self.builder, &rhs.builder));
        debug_assert!(Rc::ptr_eq(&self.staging_error, &rhs.staging_error));
        let value = apply(self.value, rhs.value);
        let atom = if self.staging_error.borrow().is_some() {
            self.atom
        } else {
            match self.builder.borrow_mut().add_equation_with_output_values(
                op,
                vec![self.atom, rhs.atom],
                vec![value.clone()],
            ) {
                Ok(outputs) => outputs[0],
                Err(error) => {
                    *self.staging_error.borrow_mut() = Some(error);
                    self.atom
                }
            }
        };
        Self { value, atom, builder: self.builder, staging_error: self.staging_error }
    }
}

impl<V: Traceable<ArrayType>, S: OpSet<ArrayType, V>> Typed<ArrayType> for JitTracer<ArrayType, V, S> {
    #[inline]
    fn tpe(&self) -> Cow<'_, ArrayType> {
        <V as Typed<ArrayType>>::tpe(&self.value)
    }
}

impl<V: Traceable<ArrayType>, S: OpSet<ArrayType, V>> Traceable<ArrayType> for JitTracer<ArrayType, V, S> {}

impl<V: Traceable<ArrayType> + ZeroLike, S: OpSet<ArrayType, V>> ZeroLike for JitTracer<ArrayType, V, S> {
    #[inline]
    fn zero_like(&self) -> Self {
        let value = self.value.zero_like();
        let atom = self.builder.borrow_mut().add_constant(value.clone());
        Self { value, atom, builder: self.builder.clone(), staging_error: self.staging_error.clone() }
    }
}

impl<V: Traceable<ArrayType> + OneLike, S: OpSet<ArrayType, V>> OneLike for JitTracer<ArrayType, V, S> {
    #[inline]
    fn one_like(&self) -> Self {
        let value = self.value.one_like();
        let atom = self.builder.borrow_mut().add_constant(value.clone());
        Self { value, atom, builder: self.builder.clone(), staging_error: self.staging_error.clone() }
    }
}

impl<V: Traceable<ArrayType> + Add<Output = V>, S: SupportsAdd<ArrayType, V>> Add for JitTracer<ArrayType, V, S>
where
    S::JitOp: Op<ArrayType>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.binary(rhs, S::add_op(), |left, right| left + right)
    }
}

impl<V: Traceable<ArrayType> + Mul<Output = V>, S: SupportsMul<ArrayType, V>> Mul for JitTracer<ArrayType, V, S>
where
    S::JitOp: Op<ArrayType>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.binary(rhs, S::mul_op(), |left, right| left * right)
    }
}

impl<V: Traceable<ArrayType> + Neg<Output = V>, S: SupportsNeg<ArrayType, V>> Neg for JitTracer<ArrayType, V, S>
where
    S::JitOp: Op<ArrayType>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self.unary(S::neg_op(), |value| -value)
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

fn try_trace_program_with_options<F, Input, Output, V, S>(
    function: F,
    input: Input,
    simplify_program: bool,
) -> Result<(Output, Program<ArrayType, V, Input, Output, S::JitOp>), TraceError>
where
    V: Traceable<ArrayType>,
    S: OpSet<ArrayType, V>,
    Input: TraceInput<V, S>,
    Output: TraceOutput<V, S>,
    F: FnOnce(Input::Traced) -> Result<Output::Traced, TraceError>,
    S::JitOp: Op<ArrayType>,
{
    let input_structure = input.parameter_structure();
    let builder = Rc::new(RefCell::new(ProgramBuilderFor::<S, V>::new()));
    let staging_error = Rc::new(RefCell::new(None));
    let traced_input = input.into_traced(builder.clone(), staging_error.clone())?;

    let (output_structure, output_value, outputs) = {
        let traced_output = function(traced_input)?;
        let (output_value, outputs) = Output::from_traced(traced_output)?;
        let output_structure = output_value.parameter_structure();
        (output_structure, output_value, outputs)
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
    Ok((output_value, program))
}

/// Stages `function` using the staged op set selected by `engine`.
pub fn try_trace_program<E, F, Input, Output, V>(
    _engine: &E,
    function: F,
    input: Input,
) -> Result<
    (Output, Program<ArrayType, V, Input, Output, <<E as Engine>::OpSet as OpSet<ArrayType, V>>::JitOp>),
    TraceError,
>
where
    E: Engine<Type = ArrayType, Value = V>,
    V: Traceable<ArrayType>,
    Input: TraceInput<V, E::OpSet>,
    Output: TraceOutput<V, E::OpSet>,
    F: FnOnce(Input::Traced) -> Result<Output::Traced, TraceError>,
    E::OpSet: OpSet<ArrayType, V>,
    <E::OpSet as OpSet<ArrayType, V>>::JitOp: Op<ArrayType>,
{
    try_trace_program_for_op_set::<_, _, _, _, E::OpSet>(function, input)
}

/// Stages `function` using one explicit backend-owned op set.
pub(crate) fn try_trace_program_for_op_set<F, Input, Output, V, S>(
    function: F,
    input: Input,
) -> Result<(Output, Program<ArrayType, V, Input, Output, S::JitOp>), TraceError>
where
    V: Traceable<ArrayType>,
    S: OpSet<ArrayType, V>,
    Input: TraceInput<V, S>,
    Output: TraceOutput<V, S>,
    F: FnOnce(Input::Traced) -> Result<Output::Traced, TraceError>,
    S::JitOp: Op<ArrayType>,
{
    try_trace_program_with_options::<_, _, _, _, S>(function, input, true)
}

/// Stages `function` as a graph using the staged op set selected by `engine`.
pub fn try_jit<E, F, Input, Output, V>(
    _engine: &E,
    function: F,
    input: Input,
) -> Result<(Output, CompiledFunction<ArrayType, V, Input, Output, <E::OpSet as OpSet<ArrayType, V>>::JitOp>), TraceError>
where
    E: Engine<Type = ArrayType, Value = V>,
    V: Traceable<ArrayType>,
    Input: TraceInput<V, E::OpSet>,
    Output: TraceOutput<V, E::OpSet>,
    F: FnOnce(Input::Traced) -> Result<Output::Traced, TraceError>,
    E::OpSet: OpSet<ArrayType, V>,
    <E::OpSet as OpSet<ArrayType, V>>::JitOp: Op<ArrayType>,
{
    try_jit_for_op_set::<_, _, _, _, E::OpSet>(function, input)
}

/// Stages `function` as a graph using one explicit backend-owned op set.
pub(crate) fn try_jit_for_op_set<F, Input, Output, V, S>(
    function: F,
    input: Input,
) -> Result<(Output, CompiledFunction<ArrayType, V, Input, Output, S::JitOp>), TraceError>
where
    V: Traceable<ArrayType>,
    S: OpSet<ArrayType, V>,
    Input: TraceInput<V, S>,
    Output: TraceOutput<V, S>,
    F: FnOnce(Input::Traced) -> Result<Output::Traced, TraceError>,
    S::JitOp: Op<ArrayType>,
{
    let (output, program) = try_trace_program_for_op_set::<_, _, _, _, S>(function, input)?;
    Ok((output, CompiledFunction::from_program(program)))
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
) -> Result<(Output, CompiledFunction<ArrayType, V, Input, Output, <E::OpSet as OpSet<ArrayType, V>>::JitOp>), TraceError>
where
    E: Engine<Type = ArrayType, Value = V>,
    V: Traceable<ArrayType>,
    Input: TraceInput<V, E::OpSet>,
    Output: TraceOutput<V, E::OpSet>,
    F: FnOnce(Input::Traced) -> Output::Traced,
    E::OpSet: OpSet<ArrayType, V>,
    <E::OpSet as OpSet<ArrayType, V>>::JitOp: Op<ArrayType>,
{
    try_jit(engine, |traced_input| Ok(function(traced_input)), input)
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use indoc::indoc;

    use crate::{
        parameters::Placeholder,
        tracing_v2::{CoreOpSet, ProgramBuilder, Sin, engine::ArrayScalarEngine, test_support},
    };

    use super::*;

    #[test]
    fn jit_tracer_zero_like_adds_constant_atoms() {
        let builder = Rc::new(RefCell::new(ProgramBuilder::<f64>::new()));
        let staging_error = Rc::new(RefCell::new(None));
        let atom = builder.borrow_mut().add_input(&3.0f64);
        let tracer: JitTracer<ArrayType, f64, CoreOpSet> = JitTracer { value: 3.0, atom, builder, staging_error };
        let zero = tracer.zero_like();
        assert_eq!(zero.value, 0.0);
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
            type OpSet = CoreOpSet;

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
