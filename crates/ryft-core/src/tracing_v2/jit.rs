//! Just-in-time staging support for `tracing_v2`.
//!
//! The current [`interpret_and_trace`] transform captures a program of staged primitive applications and replays that
//! program with the built-in interpreter. This keeps the API shape close to the eventual compiled-backend design while
//! remaining easy to test in pure Rust.

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
        AtomId, InterpretableOp, OneLike, Program, ProgramBuilder, TraceError, Traceable, ZeroLike,
        engine::Engine,
        operations::{AddTracingOperation, MulTracingOperation, NegTracingOperation, Op},
    },
    types::{Type, Typed},
};

/// Tracer used while staging JIT programs.
pub struct Tracer<
    T: Type + Display,
    V: Traceable<T> + Parameter,
    O: Clone + 'static,
    L: Clone + 'static,
    E: Engine<Type = T, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized,
> {
    atom: AtomId,
    builder: Rc<RefCell<ProgramBuilder<O, T, V>>>,
    staging_error: Rc<RefCell<Option<TraceError>>>,
    engine: *const E,
    marker: PhantomData<fn() -> L>,
}

impl<
    T: Type + Display,
    V: Traceable<T>,
    O: Clone + 'static,
    L: Clone + 'static,
    E: Engine<Type = T, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized,
> Clone for Tracer<T, V, O, L, E>
{
    fn clone(&self) -> Self {
        Self {
            atom: self.atom,
            builder: self.builder.clone(),
            staging_error: self.staging_error.clone(),
            engine: self.engine,
            marker: PhantomData,
        }
    }
}

impl<
    T: Type + Display,
    V: Traceable<T>,
    O: Clone + 'static,
    L: Clone + 'static,
    E: Engine<Type = T, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized,
> Parameter for Tracer<T, V, O, L, E>
{
}

impl<
    T: Type + Display,
    V: Traceable<T>,
    O: Clone + 'static,
    L: Clone + 'static,
    E: Engine<Type = T, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized,
> std::fmt::Debug for Tracer<T, V, O, L, E>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Tracer").field("atom", &self.atom).finish_non_exhaustive()
    }
}

impl<
    T: Type + Display,
    V: Traceable<T>,
    O: Clone + 'static,
    L: Clone + 'static,
    E: Engine<Type = T, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized,
> Tracer<T, V, O, L, E>
{
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
    pub fn engine(&self) -> &E {
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
        engine: &E,
    ) -> Self {
        Self::from_staged_parts(atom, builder, staging_error, engine)
    }

    #[inline]
    pub fn from_staged_parts(
        atom: AtomId,
        builder: Rc<RefCell<ProgramBuilder<O, T, V>>>,
        staging_error: Rc<RefCell<Option<TraceError>>>,
        engine: &E,
    ) -> Self {
        // Safe because traced values are confined to the tracing scope and all public tracing
        // entry points require reclaiming the shared builder before they return, so no staged
        // tracer can outlive the engine reference captured here.
        let engine = engine as *const E;
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
                "tracer inputs for one staged op must share the same builder",
            ));
        }
        if inputs.iter().skip(1).any(|input| !Rc::ptr_eq(&staging_error, &input.staging_error)) {
            return Err(TraceError::InternalInvariantViolation(
                "tracer inputs for one staged op must share the same staging error handle",
            ));
        }

        let input_atoms = inputs.iter().map(|input| input.atom).collect::<Vec<_>>();
        let output_count = {
            let builder_borrow = builder.borrow();
            match op.abstract_eval(
                input_atoms
                    .iter()
                    .map(|input| {
                        builder_borrow.atom(*input).expect("tracer input atoms should exist").tpe().into_owned()
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

impl<
    T: Type + Display,
    V: Traceable<T>,
    O: Clone + 'static,
    L: Clone + 'static,
    E: Engine<Type = T, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized,
> Typed<T> for Tracer<T, V, O, L, E>
{
    #[inline]
    fn tpe(&self) -> Cow<'_, T> {
        Cow::Owned(
            self.builder
                .borrow()
                .atom(self.atom)
                .expect("tracer atom should exist in its staging builder")
                .tpe()
                .into_owned(),
        )
    }
}

impl<
    T: Type + Display + 'static,
    V: Traceable<T>,
    O: Clone + 'static,
    L: Clone + 'static,
    E: Engine<Type = T, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized + 'static,
> Traceable<T> for Tracer<T, V, O, L, E>
{
}

impl<
    T: Type + Display,
    V: Traceable<T>,
    O: Clone + 'static,
    L: Clone + 'static,
    E: Engine<Type = T, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized,
> ZeroLike for Tracer<T, V, O, L, E>
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

impl<
    T: Type + Display,
    V: Traceable<T>,
    O: Clone + 'static,
    L: Clone + 'static,
    E: Engine<Type = T, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized,
> OneLike for Tracer<T, V, O, L, E>
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

impl<
    T: Type + Display,
    V: Traceable<T>,
    O: AddTracingOperation<T, V> + 'static,
    L: Clone + 'static,
    E: Engine<Type = T, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized,
> Add for Tracer<T, V, O, L, E>
where
    O: Op<T>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.binary(rhs, O::add_op())
    }
}

impl<
    T: Type + Display,
    V: Traceable<T>,
    O: MulTracingOperation<T, V> + 'static,
    L: Clone + 'static,
    E: Engine<Type = T, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized,
> Mul for Tracer<T, V, O, L, E>
where
    O: Op<T>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.binary(rhs, O::mul_op())
    }
}

impl<
    T: Type + Display,
    V: Traceable<T>,
    O: NegTracingOperation<T, V> + 'static,
    L: Clone + 'static,
    E: Engine<Type = T, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized,
> Neg for Tracer<T, V, O, L, E>
where
    O: Op<T>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self.unary(O::neg_op())
    }
}

pub(crate) type ConcreteTracer<E> = Tracer<
    <E as Engine>::Type,
    <E as Engine>::Value,
    <E as Engine>::TracingOperation,
    <E as Engine>::LinearOperation,
    E,
>;

/// Stages `function`, interprets the resulting program on the supplied concrete inputs, and returns
/// both the interpreted output and the staged program.
pub fn interpret_and_trace<E, F, Input, Output>(
    engine: &E,
    function: F,
    input: Input,
) -> Result<(Output, Program<E::Type, E::Value, Input, Output, E::TracingOperation>), TraceError>
where
    E: Engine + ?Sized + 'static,
    E::Type: Type + Display + Parameter,
    E::Value: Traceable<E::Type>,
    E::TracingOperation: Clone + 'static + InterpretableOp<E::Type, E::Value>,
    E::LinearOperation: Clone + 'static,
    Input: Parameterized<E::Value, ParameterStructure: Clone>,
    Output: Parameterized<E::Value, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<ConcreteTracer<E>>,
    Output::Family: ParameterizedFamily<ConcreteTracer<E>>,
    F: FnOnce(Input::To<ConcreteTracer<E>>) -> Result<Output::To<ConcreteTracer<E>>, TraceError>,
    Input::ParameterStructure: PartialEq,
    Output::ParameterStructure: Clone,
{
    let input_structure = input.parameter_structure();
    let input_values = input.into_parameters().collect::<Vec<_>>();
    let input_types = input_values.iter().map(|value| value.tpe().into_owned()).collect::<Vec<_>>();
    let mut output_structure = None;
    let (_, flat_program): (
        Vec<E::Type>,
        Program<E::Type, E::Value, Vec<E::Value>, Vec<E::Value>, E::TracingOperation>,
    ) = trace(
        engine,
        |flat_traced_input| {
            let traced_input =
                Input::To::<ConcreteTracer<E>>::from_parameters(input_structure.clone(), flat_traced_input)?;
            let traced_output = function(traced_input)?;
            output_structure = Some(traced_output.parameter_structure());
            Ok(traced_output.into_parameters().collect::<Vec<_>>())
        },
        input_types,
    )?;
    let output_structure = output_structure.ok_or(TraceError::InternalInvariantViolation(
        "interpret_and_trace did not record the staged output structure",
    ))?;
    let program = flat_program.clone_with_structures::<Input, Output>(input_structure, output_structure).simplify()?;
    let concrete_input = Input::from_parameters(program.input_structure().clone(), input_values)?;
    Ok((program.call(concrete_input)?, program))
}

/// Stages `function` directly from type metadata using the staged op set selected by `engine`.
///
/// This captures the raw staged program without applying post-trace simplification so callers can
/// decide whether to keep the unsimplified form or run [`Program::simplify`] themselves.
pub fn trace<E, F, Input, Output>(
    engine: &E,
    function: F,
    input_types: Input,
) -> Result<
    (Output, Program<E::Type, E::Value, Input::To<E::Value>, Output::To<E::Value>, E::TracingOperation>),
    TraceError,
>
where
    E: Engine + ?Sized + 'static,
    E::Type: Type + Display + Parameter,
    E::Value: Traceable<E::Type>,
    E::TracingOperation: Clone + 'static + Op<E::Type>,
    E::LinearOperation: Clone + 'static,
    Input: Parameterized<E::Type, ParameterStructure: Clone>,
    Output: Parameterized<E::Type, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<E::Value> + ParameterizedFamily<ConcreteTracer<E>>,
    Output::Family: ParameterizedFamily<E::Value> + ParameterizedFamily<ConcreteTracer<E>>,
    F: FnOnce(Input::To<ConcreteTracer<E>>) -> Result<Output::To<ConcreteTracer<E>>, TraceError>,
{
    let input_structure = input_types.parameter_structure();
    let builder = Rc::new(RefCell::new(ProgramBuilder::<E::TracingOperation, E::Type, E::Value>::new()));
    let staging_error = Rc::new(RefCell::new(None));
    let traced_input = Input::To::<ConcreteTracer<E>>::from_parameters(
        input_types.parameter_structure(),
        input_types.into_parameters().map(|r#type| {
            let atom = builder.borrow_mut().add_input_abstract(r#type);
            ConcreteTracer::<E>::from_engine(atom, builder.clone(), staging_error.clone(), engine)
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
    let program =
        builder.build::<Input::To<E::Value>, Output::To<E::Value>>(outputs, input_structure, output_structure);
    Ok((output_types, program))
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use indoc::indoc;

    use crate::{
        parameters::Placeholder,
        tracing_v2::{ProgramBuilder, ProgramOpRef, Sin, engine::ArrayScalarEngine, test_support},
        types::ArrayType,
    };

    use super::*;

    #[test]
    fn jit_tracer_zero_like_adds_constant_atoms() {
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ProgramOpRef<f64>, ArrayType, f64>::new()));
        let staging_error = Rc::new(RefCell::new(None));
        let atom = builder.borrow_mut().add_input(&3.0f64);
        let engine = ArrayScalarEngine::<f64>::new();
        let tracer: ConcreteTracer<ArrayScalarEngine<f64>> =
            ConcreteTracer::from_engine(atom, builder, staging_error, &engine);
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
        let (output, program): (f64, Program<ArrayType, f64, f64, f64>) = interpret_and_trace(
            &engine,
            |x: ConcreteTracer<ArrayScalarEngine<f64>>| {
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
    fn test_interpret_and_trace_supports_non_array_types() {
        use std::fmt;

        use ryft_macros::Parameter;

        use crate::types::Type;

        #[derive(Clone, Debug, Eq, PartialEq)]
        struct TestType(&'static str);

        impl Type for TestType {
            fn is_compatible_with(&self, other: &Self) -> bool {
                self == other
            }
        }

        impl Parameter for TestType {}

        impl fmt::Display for TestType {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(self.0)
            }
        }

        #[derive(Clone, Debug, Eq, Parameter, PartialEq)]
        struct TestValue {
            r#type: TestType,
            value: i32,
        }

        impl TestValue {
            fn new(r#type: TestType, value: i32) -> Self {
                Self { r#type, value }
            }
        }

        impl Typed<TestType> for TestValue {
            fn tpe(&self) -> Cow<'_, TestType> {
                Cow::Borrowed(&self.r#type)
            }
        }

        impl Traceable<TestType> for TestValue {
            fn is_zero(&self) -> bool {
                self.value == 0
            }

            fn is_one(&self) -> bool {
                self.value == 1
            }
        }

        impl crate::tracing_v2::Value<TestType> for TestValue {}

        impl Add for TestValue {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                assert_eq!(self.r#type, rhs.r#type);
                Self { r#type: self.r#type, value: self.value + rhs.value }
            }
        }

        #[derive(Clone, Debug)]
        struct TestAddOp;

        impl fmt::Display for TestAddOp {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("test_add")
            }
        }

        impl AddTracingOperation<TestType, TestValue> for TestAddOp {
            fn add_op() -> Self {
                Self
            }
        }

        impl Op<TestType> for TestAddOp {
            fn name(&self) -> &'static str {
                "test_add"
            }

            fn abstract_eval(&self, inputs: &[TestType]) -> Result<Vec<TestType>, TraceError> {
                if inputs.len() != 2 {
                    return Err(TraceError::InvalidInputCount { expected: 2, got: inputs.len() });
                }
                if !inputs[0].is_compatible_with(&inputs[1]) {
                    return Err(TraceError::IncompatibleAbstractValues { op: "test_add" });
                }
                Ok(vec![inputs[0].clone()])
            }

            fn try_simplify(
                &self,
                inputs: &[usize],
                is_zero_constant: &dyn Fn(usize) -> bool,
                _is_one_constant: &dyn Fn(usize) -> bool,
            ) -> Option<Vec<usize>> {
                if inputs.len() != 2 {
                    return None;
                }
                if is_zero_constant(inputs[0]) {
                    Some(vec![inputs[1]])
                } else if is_zero_constant(inputs[1]) {
                    Some(vec![inputs[0]])
                } else {
                    None
                }
            }
        }

        impl InterpretableOp<TestType, TestValue> for TestAddOp {
            fn interpret(&self, inputs: &[TestValue]) -> Result<Vec<TestValue>, TraceError> {
                if inputs.len() != 2 {
                    return Err(TraceError::InvalidInputCount { expected: 2, got: inputs.len() });
                }
                if !inputs[0].r#type.is_compatible_with(&inputs[1].r#type) {
                    return Err(TraceError::IncompatibleAbstractValues { op: "test_add" });
                }
                Ok(vec![inputs[0].clone() + inputs[1].clone()])
            }
        }

        struct TestEngine;

        impl Engine for TestEngine {
            type Type = TestType;
            type Value = TestValue;
            type TracingOperation = TestAddOp;
            type LinearOperation = TestAddOp;

            fn zero(&self, r#type: &TestType) -> TestValue {
                TestValue::new(r#type.clone(), 0)
            }

            fn one(&self, r#type: &TestType) -> TestValue {
                TestValue::new(r#type.clone(), 1)
            }
        }

        let scalar_type = TestType("test_scalar");
        let (output, program): (TestValue, Program<TestType, TestValue, (TestValue, TestValue), TestValue, TestAddOp>) =
            interpret_and_trace(
                &TestEngine,
                |inputs: (ConcreteTracer<TestEngine>, ConcreteTracer<TestEngine>)| {
                    let sum = inputs.0.clone() + inputs.1;
                    let stabilized = sum + inputs.0.zero_like();
                    Ok(stabilized + inputs.0.one_like())
                },
                (TestValue::new(scalar_type.clone(), 2), TestValue::new(scalar_type.clone(), 3)),
            )
            .unwrap();

        assert_eq!(output, TestValue::new(scalar_type.clone(), 6));
        assert_eq!(
            program
                .call((TestValue::new(scalar_type.clone(), 4), TestValue::new(scalar_type.clone(), 5)))
                .unwrap(),
            TestValue::new(scalar_type, 10),
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
        > = interpret_and_trace(
            &TestEngine,
            |inputs: (ConcreteTracer<TestEngine>, ConcreteTracer<TestEngine>)| Ok(inputs.0 + inputs.1),
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
        let (_, compiled): (f64, Program<ArrayType, f64, f64, f64>) = interpret_and_trace(
            &engine,
            |x: ConcreteTracer<ArrayScalarEngine<f64>>| Ok(x.clone() * x.clone() + x.sin()),
            2.0f64,
        )
        .unwrap();

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
