//! Just-in-time staging support for `tracing_v2`.
//!
//! This module is the entry point for turning ordinary Rust closures into staged programs. It owns
//! [`Tracer`], the symbolic leaf wrapper that records primitive applications into a shared
//! [`ProgramBuilder`](crate::tracing_v2::ProgramBuilder), plus the two main capture modes:
//!
//! - [`trace`] records a program from abstract input metadata alone.
//! - [`interpret_and_trace`] records the same program shape while also replaying it eagerly on
//!   concrete inputs so the caller gets both the runtime result and the staged artifact.
//!
//! The rest of `tracing_v2` builds on these same primitives. Forward-mode, reverse-mode,
//! rematerialization, and traced `vmap` all eventually stage through [`Tracer`].

use std::{
    borrow::Cow,
    cell::RefCell,
    ops::{Add, Mul, Neg},
    rc::Rc,
};

use ryft_macros::Parameter;

use crate::{
    parameters::{Parameter, Parameterized, ParameterizedFamily},
    tracing_v2::{
        AtomId, InterpretableOperation, OneLike, Program, ProgramBuilder, Traceable, TracingError, ZeroLike,
        engine::Engine,
        operations::{AddTracingOperation, MulTracingOperation, NegTracingOperation, Operation},
    },
    types::Typed,
};

/// Symbolic leaf used while staging ordinary traced programs.
///
/// A [`Tracer`] is the value-level facade for one staged atom. Primitive trait impls on
/// [`Tracer`] do not compute numerically; instead, they add instructions to a shared
/// [`ProgramBuilder`](crate::tracing_v2::ProgramBuilder) and return new tracers pointing at the
/// output atoms. This makes `Tracer` the central "big picture" type for symbolic execution in
/// `tracing_v2`: if a closure is being traced rather than eagerly evaluated, its leaves are almost
/// always instances of this type.
#[derive(Parameter)]
pub struct Tracer<'engine, E: Engine<Value: Traceable<E::Type>, TracingOperation: Operation<E::Type>> + ?Sized> {
    /// Atom id representing this traced leaf inside the shared staged program.
    pub atom: AtomId,

    /// Shared builder that owns the staged program currently being traced.
    pub builder: Rc<RefCell<ProgramBuilder<E::Type, E::Value, E::TracingOperation>>>,

    /// Engine borrowed by this tracing scope for metadata-driven value synthesis.
    pub engine: &'engine E,
}

impl<'engine, E: Engine<Value: Traceable<E::Type>, TracingOperation: Operation<E::Type>> + ?Sized> Clone
    for Tracer<'engine, E>
{
    fn clone(&self) -> Self {
        Self { atom: self.atom, builder: self.builder.clone(), engine: self.engine }
    }
}

impl<'engine, E: Engine<Value: Traceable<E::Type>, TracingOperation: Operation<E::Type>> + ?Sized> std::fmt::Debug
    for Tracer<'engine, E>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Tracer").field("atom", &self.atom).finish_non_exhaustive()
    }
}

impl<'engine, E: Engine<Value: Traceable<E::Type>, TracingOperation: Operation<E::Type>> + ?Sized> Tracer<'engine, E> {
    /// Constructs a traced leaf from an existing tracing scope.
    ///
    /// This is the standard constructor used by entry points such as [`trace`] when they turn one
    /// input leaf into the corresponding symbolic tracer.
    #[inline]
    pub fn from_engine(
        atom: AtomId,
        builder: Rc<RefCell<ProgramBuilder<E::Type, E::Value, E::TracingOperation>>>,
        engine: &'engine E,
    ) -> Self {
        Self { atom, builder, engine }
    }

    /// Stages one primitive application in the current trace and returns tracers for its outputs.
    ///
    /// This is the common helper behind both the arithmetic trait impls on [`Tracer`] and the
    /// higher-order transforms that need to inject backend-selected operations manually. The method
    /// validates that all inputs belong to the same tracing scope, runs abstract evaluation to
    /// determine the output arity, and records the instruction unless the scope has already failed.
    pub fn apply_staged_op(
        engine: &'engine E,
        builder: Rc<RefCell<ProgramBuilder<E::Type, E::Value, E::TracingOperation>>>,
        inputs: &[Self],
        op: E::TracingOperation,
    ) -> Result<Vec<Self>, TracingError>
    where
        E::TracingOperation: Operation<E::Type>,
    {
        if inputs.iter().skip(1).any(|input| !Rc::ptr_eq(&builder, &input.builder)) {
            return Err(TracingError::InternalInvariantViolation(
                "tracer inputs for one staged op must share the same builder",
            ));
        }
        if inputs.iter().any(|input| !std::ptr::eq(input.engine, engine)) {
            return Err(TracingError::InternalInvariantViolation(
                "tracer inputs for one staged op must share the same engine",
            ));
        }

        let input_atoms = inputs.iter().map(|input| input.atom).collect::<Vec<_>>();
        let input_types = {
            let builder_borrow = builder.borrow();
            input_atoms
                .iter()
                .map(|input| builder_borrow.atoms[input.index].r#type().into_owned())
                .collect::<Vec<_>>()
        };
        let fallback_atom = inputs
            .first()
            .map(|input| input.atom)
            .or_else(|| builder.borrow().atoms.first().map(|_| AtomId { index: 0 }));
        let output_count = match op.infer_output_types(input_types.as_slice()) {
            Ok(output_types) => output_types.len(),
            Err(error) => {
                if builder.borrow().error.is_none() {
                    builder.borrow_mut().error = Some(error.clone());
                }
                let fallback_atom = fallback_atom.ok_or(error)?;
                return Ok(vec![Self { atom: fallback_atom, builder, engine }]);
            }
        };
        let output_atoms = if builder.borrow().error.is_some() {
            let fallback_atom = fallback_atom.ok_or(TracingError::InternalInvariantViolation(
                "failed traced staging must have one fallback atom available",
            ))?;
            vec![fallback_atom; output_count]
        } else {
            match builder.borrow_mut().add_instruction(op, input_atoms) {
                Ok(outputs) => outputs,
                Err(error) => {
                    if builder.borrow().error.is_none() {
                        builder.borrow_mut().error = Some(error.clone());
                    }
                    let fallback_atom = fallback_atom.ok_or(error)?;
                    vec![fallback_atom; output_count]
                }
            }
        };

        Ok(output_atoms.into_iter().map(|atom| Self { atom, builder: builder.clone(), engine }).collect())
    }

    /// Stages a single-input primitive application and returns its unique output.
    pub fn unary(self, op: E::TracingOperation) -> Self
    where
        E::TracingOperation: Operation<E::Type>,
    {
        Self::apply_staged_op(self.engine, self.builder.clone(), std::slice::from_ref(&self), op)
            .expect("unary traced staging should preserve non-empty inputs")
            .into_iter()
            .next()
            .expect("unary traced staging should produce one output")
    }

    /// Stages a two-input primitive application and returns its unique output.
    pub fn binary(self, rhs: Self, op: E::TracingOperation) -> Self
    where
        E::TracingOperation: Operation<E::Type>,
    {
        debug_assert!(Rc::ptr_eq(&self.builder, &rhs.builder));
        Self::apply_staged_op(self.engine, self.builder.clone(), &[self, rhs], op)
            .expect("binary traced staging should preserve non-empty inputs")
            .into_iter()
            .next()
            .expect("binary traced staging should produce one output")
    }
}

impl<'engine, E: Engine<Value: Traceable<E::Type>, TracingOperation: Operation<E::Type>> + ?Sized> Typed<E::Type>
    for Tracer<'engine, E>
{
    #[inline]
    fn r#type(&self) -> Cow<'_, E::Type> {
        Cow::Owned(self.builder.borrow().atoms[self.atom.index].r#type().into_owned())
    }
}

impl<'engine, E: Engine<Value: Traceable<E::Type>, TracingOperation: Operation<E::Type>> + ?Sized> Traceable<E::Type>
    for Tracer<'engine, E>
{
}

impl<'engine, E: Engine<Value: Traceable<E::Type>, TracingOperation: Operation<E::Type>> + ?Sized> ZeroLike
    for Tracer<'engine, E>
{
    #[inline]
    fn zero_like(&self) -> Self {
        let value = self.engine.zero(&self.r#type().into_owned());
        let atom = self.builder.borrow_mut().add_constant(value.clone());
        Self { atom, builder: self.builder.clone(), engine: self.engine }
    }
}

impl<'engine, E: Engine<Value: Traceable<E::Type>, TracingOperation: Operation<E::Type>> + ?Sized> OneLike
    for Tracer<'engine, E>
{
    #[inline]
    fn one_like(&self) -> Self {
        let value = self.engine.one(&self.r#type().into_owned());
        let atom = self.builder.borrow_mut().add_constant(value.clone());
        Self { atom, builder: self.builder.clone(), engine: self.engine }
    }
}

impl<
    'engine,
    E: Engine<Value: Traceable<E::Type>, TracingOperation: AddTracingOperation<E::Type, E::Value> + Operation<E::Type>>
        + ?Sized,
> Add for Tracer<'engine, E>
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.binary(rhs, E::TracingOperation::add_op())
    }
}

impl<
    'engine,
    E: Engine<Value: Traceable<E::Type>, TracingOperation: MulTracingOperation<E::Type, E::Value> + Operation<E::Type>>
        + ?Sized,
> Mul for Tracer<'engine, E>
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.binary(rhs, E::TracingOperation::mul_op())
    }
}

impl<
    'engine,
    E: Engine<Value: Traceable<E::Type>, TracingOperation: NegTracingOperation<E::Type, E::Value> + Operation<E::Type>>
        + ?Sized,
> Neg for Tracer<'engine, E>
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self.unary(E::TracingOperation::neg_op())
    }
}

/// Stages `function` directly from type metadata using the staged op set selected by `engine`.
///
/// [`trace`] is the most "symbolic" entry point in the module: it never needs concrete runtime
/// inputs, only the parameterized input metadata. The closure is executed once on [`Tracer`] leaves
/// that stand in for those abstract inputs, and the resulting builder state is finalized into a
/// [`Program`].
///
/// The returned pair contains both the structured output metadata inferred during tracing and the
/// unsimplified staged program itself. Callers that want the canonical simplified form can invoke
/// [`Program::with_folded_constants`](crate::tracing_v2::Program::with_folded_constants) afterward.
pub fn trace<'engine, E, F, Input, Output>(
    engine: &'engine E,
    function: F,
    input_types: Input,
) -> Result<
    (Output, Program<E::Type, E::Value, E::TracingOperation, Input::To<E::Value>, Output::To<E::Value>>),
    TracingError,
>
where
    E: Engine<Type: Parameter, Value: Traceable<E::Type>, TracingOperation: Operation<E::Type>> + ?Sized,
    Input: Parameterized<
            E::Type,
            ParameterStructure: Clone,
            Family: ParameterizedFamily<E::Value> + ParameterizedFamily<Tracer<'engine, E>>,
        >,
    Output: Parameterized<
            E::Type,
            ParameterStructure: Clone,
            Family: ParameterizedFamily<E::Value> + ParameterizedFamily<Tracer<'engine, E>>,
        >,
    F: FnOnce(Input::To<Tracer<'engine, E>>) -> Result<Output::To<Tracer<'engine, E>>, TracingError>,
{
    let input_structure = input_types.parameter_structure();
    let builder = Rc::new(RefCell::new(ProgramBuilder::<E::Type, E::Value, E::TracingOperation>::new()));
    let traced_input = Input::To::<Tracer<'engine, E>>::from_parameters(
        input_types.parameter_structure(),
        input_types.into_parameters().map(|r#type| {
            let atom = builder.borrow_mut().add_input(r#type);
            Tracer::from_engine(atom, builder.clone(), engine)
        }),
    )
    .map_err(TracingError::from)?;

    let (output_structure, output_types, outputs) = {
        let traced_output = function(traced_input)?;
        let output_structure = traced_output.parameter_structure();
        let traced_outputs = traced_output.into_parameters().collect::<Vec<_>>();
        let output_types = Output::from_parameters(
            output_structure.clone(),
            traced_outputs.iter().map(|output| output.r#type().into_owned()).collect::<Vec<_>>(),
        )?;
        let outputs = traced_outputs.into_iter().map(|output| output.atom).collect::<Vec<_>>();
        let output_structure = output_types.parameter_structure();
        (output_structure, output_types, outputs)
    };

    if let Some(tracing_error) = builder.borrow_mut().error.take() {
        return Err(tracing_error);
    }
    let builder = match Rc::try_unwrap(builder) {
        Ok(builder) => builder.into_inner(),
        Err(_) => return Err(TracingError::InternalInvariantViolation("jit builder escaped the tracing scope")),
    };
    let program =
        builder.build::<Input::To<E::Value>, Output::To<E::Value>>(outputs, input_structure, output_structure);
    Ok((output_types, program))
}

/// Stages `function`, interprets the resulting program on the supplied concrete inputs, and returns
/// both the interpreted output and the staged program.
///
/// This is the main "trace what I just ran" API used throughout tests and higher-order transforms.
/// It first captures the symbolic program shape through [`trace`], then immediately re-tags that
/// flat trace with the caller's original structures, simplifies it, and replays it on the supplied
/// inputs. The result is a convenient pair:
///
/// - the concrete output that the caller would expect from eager execution, and
/// - the staged [`Program`] representing the same computation for later reuse.
pub fn interpret_and_trace<'engine, E, F, Input, Output>(
    engine: &'engine E,
    function: F,
    input: Input,
) -> Result<(Output, Program<E::Type, E::Value, E::TracingOperation, Input, Output>), TracingError>
where
    E: Engine<Type: Parameter, Value: Traceable<E::Type>, TracingOperation: InterpretableOperation<E::Type, E::Value>>
        + ?Sized,
    Input:
        Parameterized<E::Value, ParameterStructure: Clone + PartialEq, Family: ParameterizedFamily<Tracer<'engine, E>>>,
    Output: Parameterized<E::Value, ParameterStructure: Clone, Family: ParameterizedFamily<Tracer<'engine, E>>>,
    F: FnOnce(Input::To<Tracer<'engine, E>>) -> Result<Output::To<Tracer<'engine, E>>, TracingError>,
{
    let input_structure = input.parameter_structure();
    let input_values = input.into_parameters().collect::<Vec<_>>();
    let input_types = input_values.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
    let mut output_structure = None;
    let (_, flat_program): (
        Vec<E::Type>,
        Program<E::Type, E::Value, E::TracingOperation, Vec<E::Value>, Vec<E::Value>>,
    ) = trace(
        engine,
        |flat_traced_input| {
            let traced_input =
                Input::To::<Tracer<'engine, E>>::from_parameters(input_structure.clone(), flat_traced_input)?;
            let traced_output = function(traced_input)?;
            output_structure = Some(traced_output.parameter_structure());
            Ok(traced_output.into_parameters().collect::<Vec<_>>())
        },
        input_types,
    )?;
    let output_structure = output_structure.ok_or(TracingError::InternalInvariantViolation(
        "interpret_and_trace did not record the staged output structure",
    ))?;
    let program: Program<E::Type, E::Value, E::TracingOperation, Input, Output> = Program {
        atoms: flat_program.atoms.clone(),
        input_ids: flat_program.input_ids.clone(),
        output_ids: flat_program.output_ids.clone(),
        instructions: flat_program.instructions.clone(),
        input_structure,
        output_structure,
        marker: std::marker::PhantomData,
    }
    .with_folded_constants()?
    .simplified()?;
    let concrete_input = Input::from_parameters(program.input_structure.clone(), input_values)?;
    Ok((program.interpret(concrete_input)?, program))
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use indoc::indoc;

    use crate::{
        parameters::Placeholder,
        tracing_v2::{PrimitiveOperation, ProgramBuilder, Sin, engine::ArrayScalarEngine, test_support},
        types::ArrayType,
    };

    use super::*;

    #[test]
    fn jit_tracer_zero_like_adds_constant_atoms() {
        let builder =
            Rc::new(RefCell::new(ProgramBuilder::<ArrayType, f64, PrimitiveOperation<ArrayType, f64>>::new()));
        let atom = builder.borrow_mut().add_input(3.0f64.r#type().into_owned());
        let engine = ArrayScalarEngine::<f64>::new();
        let tracer: Tracer<ArrayScalarEngine<f64>> = Tracer::from_engine(atom, builder, &engine);
        let zero = tracer.zero_like();
        assert_eq!(zero.r#type().into_owned(), ArrayType::scalar(crate::types::DataType::F64));
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
        let (output, program): (f64, Program<ArrayType, f64, PrimitiveOperation<ArrayType, f64>, f64, f64>) =
            interpret_and_trace(
                &engine,
                |x: Tracer<ArrayScalarEngine<f64>>| {
                    let squared = x.clone() * x.clone();
                    Ok(squared + x.sin())
                },
                2.0f64,
            )
            .unwrap();

        assert_eq!(output, 2.0f64 * 2.0f64 + 2.0f64.sin());
        assert_eq!(program.interpret(0.5f64).unwrap(), 0.5f64 * 0.5f64 + 0.5f64.sin());
        assert_eq!(program.input_ids.len(), 1);
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
            fn r#type(&self) -> Cow<'_, TestType> {
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

        impl Operation<TestType> for TestAddOp {
            fn name(&self) -> &'static str {
                "test_add"
            }

            fn infer_output_types(&self, input_types: &[TestType]) -> Result<Vec<TestType>, TracingError> {
                if input_types.len() != 2 {
                    return Err(TracingError::InvalidInputCount { expected: 2, got: input_types.len() });
                }
                if !input_types[0].is_compatible_with(&input_types[1]) {
                    return Err(TracingError::IncompatibleAbstractValues { op: "test_add" });
                }
                Ok(vec![input_types[0].clone()])
            }

            fn try_simplify(
                &self,
                inputs: &[AtomId],
                is_zero_constant: &dyn Fn(AtomId) -> bool,
                _is_one_constant: &dyn Fn(AtomId) -> bool,
            ) -> Option<Vec<AtomId>> {
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

        impl InterpretableOperation<TestType, TestValue> for TestAddOp {
            fn interpret(&self, inputs: &[TestValue]) -> Result<Vec<TestValue>, TracingError> {
                if inputs.len() != 2 {
                    return Err(TracingError::InvalidInputCount { expected: 2, got: inputs.len() });
                }
                if !inputs[0].r#type.is_compatible_with(&inputs[1].r#type) {
                    return Err(TracingError::IncompatibleAbstractValues { op: "test_add" });
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
        let (output, program): (TestValue, Program<TestType, TestValue, TestAddOp, (TestValue, TestValue), TestValue>) =
            interpret_and_trace(
                &TestEngine,
                |inputs: (Tracer<TestEngine>, Tracer<TestEngine>)| {
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
                .interpret((TestValue::new(scalar_type.clone(), 4), TestValue::new(scalar_type.clone(), 5)))
                .unwrap(),
            TestValue::new(scalar_type, 10),
        );
    }

    #[test]
    fn jit_returns_abstract_eval_errors_instead_of_panicking() {
        use ryft_macros::Parameter;

        use crate::{
            tracing_v2::{Cos, MatrixOps, OneLike, Sin, ZeroLike, operations::reshape::ReshapeOps},
            types::{ArrayType, DataType, Shape, Size, Typed},
        };

        #[derive(Clone, Debug, Parameter)]
        struct TestAbstractValue {
            r#type: ArrayType,
        }

        impl Typed<ArrayType> for TestAbstractValue {
            fn r#type(&self) -> Cow<'_, ArrayType> {
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
            fn reshape(self, _target_shape: crate::types::Shape) -> Result<Self, TracingError> {
                Ok(self)
            }
        }

        struct TestEngine;

        impl crate::tracing_v2::engine::Engine for TestEngine {
            type Type = ArrayType;
            type Value = TestAbstractValue;
            type TracingOperation = crate::tracing_v2::PrimitiveOperation<ArrayType, TestAbstractValue>;
            type LinearOperation = crate::tracing_v2::LinearPrimitiveOperation<ArrayType, TestAbstractValue>;

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
                Program<
                    ArrayType,
                    TestAbstractValue,
                    crate::tracing_v2::PrimitiveOperation<ArrayType, TestAbstractValue>,
                    (TestAbstractValue, TestAbstractValue),
                    TestAbstractValue,
                >,
            ),
            TracingError,
        > = interpret_and_trace(
            &TestEngine,
            |inputs: (Tracer<TestEngine>, Tracer<TestEngine>)| Ok(inputs.0 + inputs.1),
            (
                TestAbstractValue {
                    r#type: ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)]), None, None).unwrap(),
                },
                TestAbstractValue {
                    r#type: ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3)]), None, None).unwrap(),
                },
            ),
        );

        assert!(matches!(result, Err(TracingError::IncompatibleAbstractValues { op: "add" })));
    }

    #[test]
    fn staged_program_display_renders_the_staged_program() {
        let engine = ArrayScalarEngine::<f64>::new();
        let (_, compiled): (f64, Program<ArrayType, f64, PrimitiveOperation<ArrayType, f64>, f64, f64>) =
            interpret_and_trace(
                &engine,
                |x: Tracer<ArrayScalarEngine<f64>>| Ok(x.clone() * x.clone() + x.sin()),
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
