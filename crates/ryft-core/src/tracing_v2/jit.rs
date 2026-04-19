//! Just-in-time staging support for `tracing_v2`.
//!
//! The current `jit` transform captures a program of staged primitive applications and replays that program with the
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
        AtomId, InterpretableOp, OneLike, Program, ProgramBuilder, ProgramOpRef, TraceError, Traceable, ZeroLike,
        engine::Engine,
        operations::{AddTracingOperation, MulTracingOperation, NegTracingOperation, Op},
    },
    types::{ArrayType, Type, Typed},
};

/// Tracer used while staging JIT programs.
#[derive(Clone)]
pub struct JitTracer<
    T: Type + Display,
    V: Traceable<T> + Parameter,
    O: Clone + 'static = ProgramOpRef<V>,
    L: Clone + 'static = crate::tracing_v2::LinearProgramOpRef<V>,
> {
    atom: AtomId,
    builder: Rc<RefCell<ProgramBuilder<O, T, V>>>,
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
    pub fn builder_handle(&self) -> Rc<RefCell<ProgramBuilder<O, T, V>>> {
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
        builder: Rc<RefCell<ProgramBuilder<O, T, V>>>,
        staging_error: Rc<RefCell<Option<TraceError>>>,
        engine: &dyn Engine<Type = T, Value = V, TracingOperation = O, LinearOperation = L>,
    ) -> Self {
        Self::from_staged_parts(atom, builder, staging_error, engine)
    }

    #[inline]
    pub fn from_staged_parts(
        atom: AtomId,
        builder: Rc<RefCell<ProgramBuilder<O, T, V>>>,
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

/// Stages `function` using the staged op set selected by `engine`.
pub fn trace_program<F, Input, Output, V, O, L>(
    engine: &dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L>,
    function: F,
    input: Input,
) -> Result<(Output, Program<ArrayType, V, Input, Output, O>), TraceError>
where
    V: Traceable<ArrayType>,
    O: Clone + 'static + InterpretableOp<ArrayType, V>,
    L: Clone + 'static,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<JitTracer<ArrayType, V, O, L>>,
    Output::Family: ParameterizedFamily<JitTracer<ArrayType, V, O, L>>,
    F: FnOnce(
        Input::To<JitTracer<ArrayType, V, O, L>>,
    ) -> Result<Output::To<JitTracer<ArrayType, V, O, L>>, TraceError>,
    Input::ParameterStructure: PartialEq,
    Output::ParameterStructure: Clone,
{
    let input_structure = input.parameter_structure();
    let input_values = input.into_parameters().collect::<Vec<_>>();
    let builder = Rc::new(RefCell::new(ProgramBuilder::<O, ArrayType, V>::new()));
    let staging_error = Rc::new(RefCell::new(None));
    let concrete_input = Input::from_parameters(input_structure.clone(), input_values.clone())?;
    let traced_input = Input::To::<JitTracer<ArrayType, V, O, L>>::from_parameters(
        concrete_input.parameter_structure(),
        concrete_input.into_parameters().map(|value| {
            let atom = builder.borrow_mut().add_input(&value);
            JitTracer::<ArrayType, V, O, L>::from_engine(atom, builder.clone(), staging_error.clone(), engine)
        }),
    )
    .map_err(TraceError::from)?;

    let (output_structure, outputs) = {
        let traced_output = function(traced_input)?;
        let output_structure = traced_output.parameter_structure();
        let outputs = traced_output.into_parameters().map(|output| output.atom()).collect::<Vec<_>>();
        (output_structure, outputs)
    };

    if let Some(error) = staging_error.borrow_mut().take() {
        return Err(error);
    }
    let builder = match Rc::try_unwrap(builder) {
        Ok(builder) => builder.into_inner(),
        Err(_) => return Err(TraceError::InternalInvariantViolation("jit builder escaped the tracing scope")),
    };
    let program = builder.build::<Input, Output>(outputs, input_structure, output_structure);
    let program = program.simplify()?;
    let concrete_input = Input::from_parameters(program.input_structure().clone(), input_values)?;
    Ok((program.call(concrete_input)?, program))
}

fn trace_program_from_types_with_operation_options<F, Input, Output, T, V, O, L>(
    engine: &dyn Engine<Type = T, Value = V, TracingOperation = O, LinearOperation = L>,
    function: F,
    input_types: Input,
    simplify_program: bool,
) -> Result<(Output, Program<T, V, Input::To<V>, Output::To<V>, O>), TraceError>
where
    T: Type + Display + Parameter,
    V: Traceable<T>,
    O: Clone + 'static + Op<T>,
    L: Clone + 'static,
    Input: Parameterized<T, ParameterStructure: Clone>,
    Output: Parameterized<T, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<V> + ParameterizedFamily<JitTracer<T, V, O, L>>,
    Output::Family: ParameterizedFamily<V> + ParameterizedFamily<JitTracer<T, V, O, L>>,
    F: FnOnce(Input::To<JitTracer<T, V, O, L>>) -> Result<Output::To<JitTracer<T, V, O, L>>, TraceError>,
{
    let input_structure = input_types.parameter_structure();
    let builder = Rc::new(RefCell::new(ProgramBuilder::<O, T, V>::new()));
    let staging_error = Rc::new(RefCell::new(None));
    let traced_input = Input::To::<JitTracer<T, V, O, L>>::from_parameters(
        input_types.parameter_structure(),
        input_types.into_parameters().map(|r#type| {
            let atom = builder.borrow_mut().add_input_abstract(r#type);
            JitTracer::<T, V, O, L>::from_engine(atom, builder.clone(), staging_error.clone(), engine)
        }),
    )
    .map_err(TraceError::from)?;

    let (output_structure, output_types, outputs) = {
        let traced_output = function(traced_input)?;
        let output_structure = traced_output.parameter_structure();
        let traced_outputs = traced_output.into_parameters().collect::<Vec<_>>();
        let output_types = Output::from_parameters(
            output_structure.clone(),
            traced_outputs.iter().map(|output| output.tpe().into_owned()).collect::<Vec<_>>(),
        )?;
        let outputs = traced_outputs.into_iter().map(|output| output.atom()).collect::<Vec<_>>();
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
    let program = builder.build::<Input::To<V>, Output::To<V>>(outputs, input_structure, output_structure);
    let program = if simplify_program { program.simplify()? } else { program };
    Ok((output_types, program))
}

/// Stages `function` directly from type metadata using one explicit staged ordinary operation type.
pub fn trace_program_from_types<E, F, Input, Output, T, V>(
    engine: &E,
    function: F,
    input_types: Input,
) -> Result<(Output, Program<T, V, Input::To<V>, Output::To<V>, E::TracingOperation>), TraceError>
where
    E: Engine<Type = T, Value = V>,
    T: Type + Display + Parameter,
    V: Traceable<T>,
    Input: Parameterized<T, ParameterStructure: Clone>,
    Output: Parameterized<T, ParameterStructure: Clone>,
    Input::Family:
        ParameterizedFamily<V> + ParameterizedFamily<JitTracer<T, V, E::TracingOperation, E::LinearOperation>>,
    Output::Family:
        ParameterizedFamily<V> + ParameterizedFamily<JitTracer<T, V, E::TracingOperation, E::LinearOperation>>,
    F: FnOnce(
        Input::To<JitTracer<T, V, E::TracingOperation, E::LinearOperation>>,
    ) -> Result<Output::To<JitTracer<T, V, E::TracingOperation, E::LinearOperation>>, TraceError>,
    E::TracingOperation: Op<T>,
{
    trace_program_from_types_with_operation_options(engine, function, input_types, true)
}

/// Stages `function` directly from type metadata using one explicit staged ordinary operation type.
pub(crate) fn trace_program_from_types_for_operation<F, Input, Output, T, V, O, L>(
    engine: &dyn Engine<Type = T, Value = V, TracingOperation = O, LinearOperation = L>,
    function: F,
    input_types: Input,
) -> Result<(Output, Program<T, V, Input::To<V>, Output::To<V>, O>), TraceError>
where
    T: Type + Display + Parameter,
    V: Traceable<T>,
    O: Clone + 'static + Op<T>,
    L: Clone + 'static,
    Input: Parameterized<T, ParameterStructure: Clone>,
    Output: Parameterized<T, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<V> + ParameterizedFamily<JitTracer<T, V, O, L>>,
    Output::Family: ParameterizedFamily<V> + ParameterizedFamily<JitTracer<T, V, O, L>>,
    F: FnOnce(Input::To<JitTracer<T, V, O, L>>) -> Result<Output::To<JitTracer<T, V, O, L>>, TraceError>,
{
    trace_program_from_types_with_operation_options(engine, function, input_types, true)
}

/// Stages `function` as a program using the staged op set selected by `engine`.
pub fn jit<E, F, Input, Output, V>(
    engine: &E,
    function: F,
    input: Input,
) -> Result<(Output, Program<ArrayType, V, Input, Output, E::TracingOperation>), TraceError>
where
    E: Engine<Type = ArrayType, Value = V>,
    V: Traceable<ArrayType>,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
    Output::Family: ParameterizedFamily<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
    F: FnOnce(
        Input::To<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
    ) -> Result<Output::To<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>, TraceError>,
    E::TracingOperation: InterpretableOp<ArrayType, V>,
    Input::ParameterStructure: PartialEq,
    Output::ParameterStructure: Clone,
{
    trace_program(engine, function, input)
}

/// Stages `function` as a program using one explicit staged ordinary operation type.
pub(crate) fn jit_for_operation<F, Input, Output, V, O, L>(
    engine: &dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L>,
    function: F,
    input: Input,
) -> Result<(Output, Program<ArrayType, V, Input, Output, O>), TraceError>
where
    V: Traceable<ArrayType>,
    O: Clone + 'static + InterpretableOp<ArrayType, V>,
    L: Clone + 'static,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<JitTracer<ArrayType, V, O, L>>,
    Output::Family: ParameterizedFamily<JitTracer<ArrayType, V, O, L>>,
    F: FnOnce(
        Input::To<JitTracer<ArrayType, V, O, L>>,
    ) -> Result<Output::To<JitTracer<ArrayType, V, O, L>>, TraceError>,
    Input::ParameterStructure: PartialEq,
    Output::ParameterStructure: Clone,
{
    trace_program(engine, function, input)
}

/// Stages `function` directly from type metadata as one program.
pub fn jit_from_types<E, F, Input, Output, T, V>(
    engine: &E,
    function: F,
    input_types: Input,
) -> Result<(Output, Program<T, V, Input::To<V>, Output::To<V>, E::TracingOperation>), TraceError>
where
    E: Engine<Type = T, Value = V>,
    T: Type + Display + Parameter,
    V: Traceable<T>,
    Input: Parameterized<T, ParameterStructure: Clone>,
    Output: Parameterized<T, ParameterStructure: Clone>,
    Input::Family:
        ParameterizedFamily<V> + ParameterizedFamily<JitTracer<T, V, E::TracingOperation, E::LinearOperation>>,
    Output::Family:
        ParameterizedFamily<V> + ParameterizedFamily<JitTracer<T, V, E::TracingOperation, E::LinearOperation>>,
    F: FnOnce(
        Input::To<JitTracer<T, V, E::TracingOperation, E::LinearOperation>>,
    ) -> Result<Output::To<JitTracer<T, V, E::TracingOperation, E::LinearOperation>>, TraceError>,
    E::TracingOperation: Op<T>,
{
    trace_program_from_types(engine, function, input_types)
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
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ProgramOpRef<f64>, ArrayType, f64>::new()));
        let staging_error = Rc::new(RefCell::new(None));
        let atom = builder.borrow_mut().add_input(&3.0f64);
        let engine = ArrayScalarEngine::<f64>::new();
        let tracer: JitTracer<ArrayType, f64> = JitTracer::from_engine(atom, builder, staging_error, &engine);
        let zero = tracer.zero_like();
        assert_eq!(zero.tpe().into_owned(), ArrayType::scalar(crate::types::DataType::F64));
        assert!(zero.atom > atom);

        let program = zero.builder.borrow().clone().build::<f64, f64>(vec![zero.atom], Placeholder, Placeholder);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = const
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn staged_program_replays_graphs() {
        let engine = ArrayScalarEngine::<f64>::new();
        let (output, program): (f64, Program<ArrayType, f64, f64, f64>) = jit(
            &engine,
            |x: JitTracer<ArrayType, f64>| {
                let squared = x.clone() * x.clone();
                Ok(squared + x.sin())
            },
            2.0f64,
        )
        .unwrap();

        assert_eq!(output, 2.0f64 * 2.0f64 + 2.0f64.sin());
        assert_eq!(program.call(0.5f64).unwrap(), 0.5f64 * 0.5f64 + 0.5f64.sin());
        assert_eq!(program.input_atoms().len(), 1);
        assert_eq!(
            program.to_string(),
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
                Program<ArrayType, TestAbstractValue, (TestAbstractValue, TestAbstractValue), TestAbstractValue>,
            ),
            TraceError,
        > = jit(
            &TestEngine,
            |inputs: (JitTracer<ArrayType, TestAbstractValue>, JitTracer<ArrayType, TestAbstractValue>)| {
                Ok(inputs.0 + inputs.1)
            },
            (
                TestAbstractValue { r#type: ArrayType::scalar(DataType::F32) },
                TestAbstractValue { r#type: ArrayType::scalar(DataType::F64) },
            ),
        );

        assert!(matches!(result, Err(TraceError::IncompatibleAbstractValues { op: "add" })));
    }

    #[test]
    fn staged_program_display_renders_the_staged_program() {
        let engine = ArrayScalarEngine::<f64>::new();
        let (_, compiled): (f64, Program<ArrayType, f64, f64, f64>) =
            jit(&engine, |x: JitTracer<ArrayType, f64>| Ok(x.clone() * x.clone() + x.sin()), 2.0f64).unwrap();

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
        test_support::assert_bilinear_jit_rendering();
    }
}
