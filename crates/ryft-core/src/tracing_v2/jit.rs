//! Just-in-time staging support for `tracing_v2`.
//!
//! The current `jit` transform captures a graph of staged primitive applications and replays that graph with the
//! built-in interpreter. This keeps the API shape close to the eventual compiled-backend design while remaining easy
//! to test in pure Rust.

use std::{
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
        FloatExt, MatrixOps, OneLike, TraceError, TraceValue, ZeroLike,
        graph::AtomId,
        operations::reshape::ReshapeOps,
        ops::PrimitiveOp,
        program::{Program, ProgramBuilder, ProgramOpRef},
    },
    types::{ArrayType, Typed},
};

/// Tracer used while staging JIT programs.
#[derive(Clone, Debug, Parameter)]
pub struct JitTracer<V: TraceValue> {
    /// Concrete value obtained during eager execution of the staged computation.
    pub value: V,
    atom: AtomId,
    builder: Rc<RefCell<ProgramBuilder<V>>>,
    staging_error: Rc<RefCell<Option<TraceError>>>,
}

impl<V: TraceValue> JitTracer<V> {
    #[inline]
    pub fn builder_handle(&self) -> Rc<RefCell<ProgramBuilder<V>>> {
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
        builder: Rc<RefCell<ProgramBuilder<V>>>,
        staging_error: Rc<RefCell<Option<TraceError>>>,
    ) -> Self {
        Self { value, atom, builder, staging_error }
    }

    pub fn apply_staged_op(
        inputs: &[Self],
        op: PrimitiveOp<V>,
        output_values: Vec<V>,
    ) -> Result<Vec<Self>, TraceError> {
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

    pub fn unary(self, op: PrimitiveOp<V>, apply: impl FnOnce(V) -> V) -> Self {
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

    pub fn binary(self, rhs: Self, op: PrimitiveOp<V>, apply: impl FnOnce(V, V) -> V) -> Self {
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

impl<V: TraceValue> Typed<ArrayType> for JitTracer<V> {
    #[inline]
    fn tpe(&self) -> ArrayType {
        <V as Typed<ArrayType>>::tpe(&self.value)
    }
}

impl<V: TraceValue> TraceValue for JitTracer<V> {}

impl<V: TraceValue + ZeroLike> ZeroLike for JitTracer<V> {
    #[inline]
    fn zero_like(&self) -> Self {
        let value = self.value.zero_like();
        let atom = self.builder.borrow_mut().add_constant(value.clone());
        Self { value, atom, builder: self.builder.clone(), staging_error: self.staging_error.clone() }
    }
}

impl<V: TraceValue + OneLike> OneLike for JitTracer<V> {
    #[inline]
    fn one_like(&self) -> Self {
        let value = self.value.one_like();
        let atom = self.builder.borrow_mut().add_constant(value.clone());
        Self { value, atom, builder: self.builder.clone(), staging_error: self.staging_error.clone() }
    }
}

impl<V: TraceValue + Add<Output = V>> Add for JitTracer<V> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.binary(rhs, PrimitiveOp::Add, |left, right| left + right)
    }
}

impl<V: TraceValue + Mul<Output = V>> Mul for JitTracer<V> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.binary(rhs, PrimitiveOp::Mul, |left, right| left * right)
    }
}

impl<V: TraceValue + Neg<Output = V>> Neg for JitTracer<V> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self.unary(PrimitiveOp::Neg, |value| -value)
    }
}

impl<V: TraceValue + FloatExt> FloatExt for JitTracer<V> {
    #[inline]
    fn sin(self) -> Self {
        self.unary(PrimitiveOp::Sin, FloatExt::sin)
    }

    #[inline]
    fn cos(self) -> Self {
        self.unary(PrimitiveOp::Cos, FloatExt::cos)
    }
}

/// Staged function returned by [`jit`].
///
/// In the current prototype this type stores only the staged graph and replays it with the built-in interpreter.
/// Later, once a concrete backend exists, it can grow additional fields that hold backend-specific compiled artifacts
/// while keeping the same high-level API shape.
pub struct CompiledFunction<V: TraceValue, Input: Parameterized<V>, Output: Parameterized<V>> {
    program: Program<V, Input, Output>,
    marker: PhantomData<fn(Input) -> Output>,
}

impl<
    V: TraceValue,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
> Clone for CompiledFunction<V, Input, Output>
{
    fn clone(&self) -> Self {
        Self { program: self.program.clone(), marker: PhantomData }
    }
}

impl<V: TraceValue, Input: Parameterized<V>, Output: Parameterized<V>> CompiledFunction<V, Input, Output> {
    #[inline]
    pub fn from_graph(graph: crate::tracing_v2::Graph<ProgramOpRef<V>, V, Input, Output>) -> Self {
        Self::from_program(Program::from_graph(graph))
    }

    #[inline]
    pub fn from_program(program: Program<V, Input, Output>) -> Self {
        Self { program, marker: PhantomData }
    }

    /// Returns the staged graph backing this compiled function.
    #[inline]
    pub fn graph(&self) -> &crate::tracing_v2::Graph<ProgramOpRef<V>, V, Input, Output> {
        self.program.graph()
    }

    /// Returns the staged program backing this compiled function.
    #[inline]
    pub fn program(&self) -> &Program<V, Input, Output> {
        &self.program
    }

    /// Replays the staged graph on concrete input values.
    pub fn call(&self, input: Input) -> Result<Output, TraceError>
    where
        V: FloatExt + ZeroLike + OneLike + MatrixOps + ReshapeOps,
        Input::ParameterStructure: PartialEq,
        Output::ParameterStructure: Clone,
    {
        self.program.call(input)
    }
}

impl<V: TraceValue, Input: Parameterized<V>, Output: Parameterized<V>> Display for CompiledFunction<V, Input, Output> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.program, formatter)
    }
}

fn try_trace_program_with_options<F, Input, Output, V>(
    function: F,
    input: Input,
    simplify_program: bool,
) -> Result<(Output, Program<V, Input, Output>), TraceError>
where
    V: TraceValue + FloatExt + ZeroLike + OneLike + MatrixOps + ReshapeOps,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<JitTracer<V>>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<JitTracer<V>>,
    F: FnOnce(Input::To<JitTracer<V>>) -> Result<Output::To<JitTracer<V>>, TraceError>,
{
    let input_structure = input.parameter_structure();
    let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
    let staging_error = Rc::new(RefCell::new(None));
    let traced_input = Input::To::<JitTracer<V>>::from_parameters(
        input_structure.clone(),
        input.into_parameters().map(|value| {
            let atom = builder.borrow_mut().add_input(&value);
            JitTracer { value, atom, builder: builder.clone(), staging_error: staging_error.clone() }
        }),
    )?;

    let (output_structure, output_value, outputs) = {
        let traced_output = function(traced_input)?;
        let output_structure = traced_output.parameter_structure();
        let traced_outputs = traced_output.into_parameters().collect::<Vec<_>>();
        let output_value = Output::from_parameters(
            output_structure.clone(),
            traced_outputs.iter().map(|output| output.value.clone()).collect::<Vec<_>>(),
        )?;
        let outputs = traced_outputs.into_iter().map(|output| output.atom).collect::<Vec<_>>();
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

pub fn try_trace_program<F, Input, Output, V>(
    function: F,
    input: Input,
) -> Result<(Output, Program<V, Input, Output>), TraceError>
where
    V: TraceValue + FloatExt + ZeroLike + OneLike + MatrixOps + ReshapeOps,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<JitTracer<V>>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<JitTracer<V>>,
    F: FnOnce(Input::To<JitTracer<V>>) -> Result<Output::To<JitTracer<V>>, TraceError>,
{
    try_trace_program_with_options(function, input, true)
}

pub fn try_jit<F, Input, Output, V>(
    function: F,
    input: Input,
) -> Result<(Output, CompiledFunction<V, Input, Output>), TraceError>
where
    V: TraceValue + FloatExt + ZeroLike + OneLike + MatrixOps + ReshapeOps,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<JitTracer<V>>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<JitTracer<V>>,
    F: FnOnce(Input::To<JitTracer<V>>) -> Result<Output::To<JitTracer<V>>, TraceError>,
{
    let (output, program) = try_trace_program(function, input)?;
    Ok((output, CompiledFunction::from_program(program)))
}

/// Stages `function` as a graph and returns both the eager output and the staged program.
///
/// The returned [`CompiledFunction`] currently stores only the staged graph. Later, once a concrete backend exists,
/// this type can be extended to carry backend-specific compilation artifacts alongside that graph.
pub fn jit<F, Input, Output, V>(
    function: F,
    input: Input,
) -> Result<(Output, CompiledFunction<V, Input, Output>), TraceError>
where
    V: TraceValue + FloatExt + ZeroLike + OneLike + MatrixOps + ReshapeOps,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<JitTracer<V>>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<JitTracer<V>>,
    F: FnOnce(Input::To<JitTracer<V>>) -> Output::To<JitTracer<V>>,
{
    try_jit(|traced_input| Ok(function(traced_input)), input)
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use indoc::indoc;

    use crate::{
        parameters::Placeholder,
        tracing_v2::{GraphBuilder, test_support},
    };

    use super::*;

    #[test]
    fn jit_tracer_zero_like_adds_constant_atoms() {
        let builder = Rc::new(RefCell::new(GraphBuilder::new()));
        let staging_error = Rc::new(RefCell::new(None));
        let atom = builder.borrow_mut().add_input(&3.0f64);
        let tracer = JitTracer { value: 3.0, atom, builder, staging_error };
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
        let (output, compiled): (f64, CompiledFunction<f64, f64, f64>) = jit(
            |x: JitTracer<f64>| {
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
            tracing_v2::{FloatExt, MatrixOps, OneLike, ZeroLike, operations::reshape::ReshapeOps},
            types::{ArrayType, DataType, Typed},
        };

        #[derive(Clone, Debug, Parameter)]
        struct TestAbstractValue {
            r#type: ArrayType,
        }

        impl Typed<ArrayType> for TestAbstractValue {
            fn tpe(&self) -> ArrayType {
                self.r#type.clone()
            }
        }

        impl TraceValue for TestAbstractValue {
            fn is_zero(&self) -> bool {
                false
            }

            fn is_one(&self) -> bool {
                false
            }
        }

        impl crate::tracing_v2::ConcreteTraceValue for TestAbstractValue {}

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

        impl FloatExt for TestAbstractValue {
            fn sin(self) -> Self {
                self
            }

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

        let result: Result<
            (
                TestAbstractValue,
                CompiledFunction<TestAbstractValue, (TestAbstractValue, TestAbstractValue), TestAbstractValue>,
            ),
            TraceError,
        > = jit(
            |inputs: (JitTracer<TestAbstractValue>, JitTracer<TestAbstractValue>)| inputs.0 + inputs.1,
            (
                TestAbstractValue { r#type: ArrayType::scalar(DataType::F32) },
                TestAbstractValue { r#type: ArrayType::scalar(DataType::F64) },
            ),
        );

        assert!(matches!(result, Err(TraceError::IncompatibleAbstractValues { op: "add" })));
    }

    #[test]
    fn compiled_function_display_delegates_to_the_underlying_graph() {
        let (_, compiled): (f64, CompiledFunction<f64, f64, f64>) =
            jit(|x: JitTracer<f64>| x.clone() * x.clone() + x.sin(), 2.0f64).unwrap();

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
