// TODO(eaplatanios): Review this module.

//! Tests for the `#[derive(Operation)]` procedural macro.
//!
//! These tests define local stand-in traits and types that mirror the shapes the derive emits against. That keeps the
//! macro test focused on generated code rather than on the current `ryft-core` implementation details.

#![allow(private_interfaces, dead_code)]

use std::marker::PhantomData;

use self::partial::PartialValue;

/// Stand-in for `ryft_core::Type`.
trait Type {}

/// Stand-in for `ryft_core::DifferentiableType`.
trait DifferentiableType: Type {}

/// Stand-in for `ryft_core::TypeError`.
#[derive(Debug, PartialEq, Eq)]
struct TypeError;

/// Stand-in for `ryft_core::ProgramError`.
#[derive(Debug, PartialEq, Eq)]
struct ProgramError;

/// Stand-in for `ryft_core::Domain`.
trait Domain {
    type Type: Type;
    type Value: Value<Self::Type>;
    type Constant: Value<Self::Type>;
    type Operation: Operation<Self::Type>;
}

/// Stand-in for `ryft_core::Context`.
trait Context: Domain {
    fn lift(&self, constant: Self::Constant) -> Result<Self::Value, ProgramError>;
}

/// Stand-in minimal operation family for value-only contexts.
#[derive(Clone, Debug)]
struct NoOperation;

impl<T: Type> Operation<T> for NoOperation {
    fn name(&self) -> &'static str {
        "no_operation"
    }

    fn infer_output_types(&self, _input_types: &[T]) -> Result<Vec<T>, TypeError> {
        Ok(Vec::new())
    }
}

/// Stand-in interpretation context that lifts constants by cloning them.
struct TestContext<T: Type, V: Value<T>, O: Operation<T> = NoOperation> {
    marker: PhantomData<(T, V, O)>,
}

impl<T: Type, V: Value<T>, O: Operation<T>> Domain for TestContext<T, V, O> {
    type Type = T;
    type Value = V;
    type Constant = V;
    type Operation = O;
}

impl<T: Type, V: Value<T>, O: Operation<T>> Context for TestContext<T, V, O> {
    fn lift(&self, constant: Self::Constant) -> Result<Self::Value, ProgramError> {
        Ok(constant)
    }
}

/// Stand-in for `ryft_core::Value`.
trait Value<T: Type>: Clone {}

/// Stand-in for `ryft_core::BooleanLike`.
trait BooleanLike {}

/// Stand-in for `ryft_core::Parameterized`.
trait Parameterized<V> {
    type To<T>;
    type ParameterStructure;
}

impl<V> Parameterized<V> for Vec<V> {
    type To<T> = Vec<T>;
    type ParameterStructure = ();
}

/// Stand-in for `ryft_core::Slice`.
trait Slice {}

/// Stand-in for `ryft_core::UpdateSlice`.
trait UpdateSlice {}

/// Stand-in for `ryft_core::Reshape`.
trait Reshape {}

/// Stand-in for `ryft_core::Zero`.
trait Zero<T: Type, V: Value<T>> {}

impl<T: Type, V: Value<T>, O: Operation<T>> Zero<T, V> for TestContext<T, V, O> {}

/// Stand-in for `ryft_core::payloads`.
mod payloads {
    /// Stand-in for `ryft_core::payloads::Captured`.
    pub struct Captured;
}

/// Stand-in for `ryft_core::Constant`.
trait Constant<T: Type, V: Value<T>, Stored, Payload = payloads::Captured> {
    fn constant(&self, value: Stored) -> Result<V, ProgramError>;
}

impl<T: Type, V: Value<T>, O: Operation<T>, Stored: Clone, Payload> Constant<T, V, Stored, Payload>
    for TestContext<T, V, O>
where
    V: From<Stored>,
{
    fn constant(&self, value: Stored) -> Result<V, ProgramError> {
        Ok(V::from(value))
    }
}

/// Stand-in for `ryft_core::Effects`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Effects {
    Pure,
    Ordered,
}

/// Stand-in for `ryft_core::Operation`.
trait Operation<T: Type> {
    fn name(&self) -> &'static str;

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError>;

    fn effects(&self) -> Effects {
        Effects::Pure
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        let _ = indentation;
        formatter.write_str(self.name())
    }
}

/// Stand-in for `ryft_core::InterpretableOperation`.
trait InterpretableOperation<T: Type, V: Value<T>, C>: Operation<T> {
    fn interpret(&self, context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError>;
}

/// Stand-in for `ryft_core::TracingContext`.
struct TracingContext<T: Type, V: Value<T>, O: Operation<T>> {
    marker: PhantomData<(T, V, O)>,
}

/// Stand-in for `ryft_core::Tracer`. Mirrors the real `Tracer`'s `Value` membership so it can be the value type of a
/// `PartialValue` input in the generated transpose signature.
struct Tracer<C> {
    marker: PhantomData<C>,
}

impl<C> Clone for Tracer<C> {
    fn clone(&self) -> Self {
        Self { marker: PhantomData }
    }
}

impl<T: Type, V: Value<T>, O: Operation<T>> Value<T> for Tracer<TracingContext<T, V, O>> {}

/// Stand-in for `ryft_core::Cotangent`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Cotangent<T: Type, V: Value<T>, O: Operation<T>> {
    label: &'static str,
    marker: PhantomData<(T, V, O)>,
}

/// Stand-in for `ryft_core::TransposableOperation`.
trait TransposableOperation<T: Type, V: Value<T>, O: Operation<T>>: Operation<T> {
    fn transpose(
        &self,
        context: &mut TracingContext<T, V, O>,
        inputs: &[PartialValue<T, Tracer<TracingContext<T, V, O>>>],
        outputs: &[Cotangent<T, V, O>],
    ) -> Result<Vec<Cotangent<T, V, O>>, ProgramError>;
}

/// Stand-in for `ryft_core::MaybeZeroOperation`.
trait MaybeZeroOperation<T: Type>: Operation<T> {}

impl<T: Type, O: Operation<T>> MaybeZeroOperation<T> for O {}

/// Stand-in for `ryft_core::Program`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Program<T: Type, V: Value<T>, O: Operation<T>, Input, Output> {
    label: &'static str,
    constant: Option<V>,
    operation: Option<O>,
    marker: PhantomData<(T, V, O, Input, Output)>,
}

/// Stand-in for `ryft_core::Instruction`.
struct Instruction<'operation, O> {
    operation: &'operation O,
}

impl<O> Instruction<'_, O> {
    fn operation(&self) -> &O {
        self.operation
    }
}

impl<T, Constant, O, Input, Output> Program<T, Constant, O, Input, Output>
where
    T: Type,
    Constant: Value<T>,
    O: Operation<T>,
{
    fn interpret_with<V, LiftConstantFn, InterpretInstructionFn>(
        &self,
        mut input: Vec<V>,
        mut lift_constant: LiftConstantFn,
        mut interpret_instruction: InterpretInstructionFn,
    ) -> Result<Vec<V>, ProgramError>
    where
        V: Value<T>,
        LiftConstantFn: FnMut(usize, &Constant) -> Result<V, ProgramError>,
        InterpretInstructionFn: FnMut(&Instruction<'_, O>, &[V]) -> Result<Vec<V>, ProgramError>,
    {
        if let Some(constant) = &self.constant {
            input.push(lift_constant(0, constant)?);
        }
        if let Some(operation) = &self.operation {
            interpret_instruction(&Instruction { operation }, &input)
        } else {
            Ok(input)
        }
    }
}

impl<T, V, O, Input, Output> Program<T, V, O, Input, Output>
where
    T: DifferentiableType,
    V: Value<T>,
    O: TransposableOperation<T, V, O> + MaybeZeroOperation<T> + From<ZeroOperation<T>> + From<AddOperation>,
{
    fn transpose_partitioned(
        &self,
        input_linearity: &[bool],
    ) -> Result<Program<T, V, O, Vec<V>, Vec<V>>, ProgramError> {
        let _ = input_linearity;
        Ok(Program { label: "program_transpose_partitioned", constant: None, operation: None, marker: PhantomData })
    }
}

/// Stand-in for `ryft_core::InterpretableProgramOperation`.
trait InterpretableProgramOperation<T: Type, V: Value<T>, C, Constant: Value<T> = V>: Operation<T> + Sized {
    fn interpret_program(
        context: &C,
        program: &Program<T, Constant, Self, Vec<Constant>, Vec<Constant>>,
        input: Vec<V>,
    ) -> Result<Vec<V>, ProgramError>;
}

/// Stand-in for `ryft_core::TransposableProgramOperation`.
trait TransposableProgramOperation<T: DifferentiableType, V: Value<T>>: Operation<T> + Sized {
    fn transpose_program(
        program: &Program<T, V, Self, Vec<V>, Vec<V>>,
        input_linearity: &[bool],
    ) -> Result<Program<T, V, Self, Vec<V>, Vec<V>>, ProgramError>;
}

/// Stand-in for `ryft_core::StagingContext`.
trait StagingContext: Context {}

impl<T: Type, V: Value<T>, O: Operation<T>> StagingContext for TestContext<T, V, O> {}

/// Stand-in for `ryft_core::JvpTracer`. Mirrors only what the generated forward-mode dispatcher references: the
/// generated `jvp` signature names the type, so a label field suffices to observe payload dispatch.
struct JvpTracer<C: StagingContext> {
    label: &'static str,
    marker: PhantomData<C>,
}

impl<C: StagingContext> std::fmt::Debug for JvpTracer<C> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("JvpTracer").field("label", &self.label).finish()
    }
}

/// Stand-in for `ryft_core::DifferentiableOperation`.
trait DifferentiableOperation<C: StagingContext>: Operation<C::Type> {
    fn jvp(&self, context: &C, inputs: &[JvpTracer<C>]) -> Result<Vec<JvpTracer<C>>, ProgramError>;
}

/// Stand-in for `ryft_core::DifferentiableProgramOperation`. The generated forward-mode dispatcher only names this
/// witness in a `Self` bound, so a marker trait suffices.
trait DifferentiableProgramOperation<T: Type, V: Value<T>, O> {}

/// Stand-in for the `ryft_core::partial` module, mirroring the shapes the `Operation` derive emits against for the
/// generated `PartiallyEvaluatableOperation` implementation.
mod partial {
    use super::{Context, PhantomData, ProgramError, Type, Value};

    /// Stand-in for `ryft_core::partial::PartialValue`.
    pub(crate) enum PartialValue<T: Type, V: Value<T>> {
        Known(V),
        Unknown(T),
    }

    /// Stand-in for `ryft_core::partial::PartialEvaluationValue`.
    pub(crate) struct PartialEvaluationValue<T: Type, V: Value<T>> {
        marker: PhantomData<(T, V)>,
    }

    /// Stand-in for `ryft_core::partial::PartialEvaluator`.
    pub(crate) struct PartialEvaluator<C: Context> {
        context: C,
    }

    impl<C: Context> PartialEvaluator<C> {
        pub(crate) fn new(context: C) -> Self {
            Self { context }
        }

        fn context(&self) -> &C {
            &self.context
        }

        pub(crate) fn fold_or_residualize<P: Into<C::Operation>>(
            &mut self,
            operation: P,
            inputs: &[PartialEvaluationValue<C::Type, C::Value>],
        ) -> Result<Vec<PartialEvaluationValue<C::Type, C::Value>>, ProgramError> {
            let _ = (operation.into(), inputs);
            Ok(Vec::new())
        }
    }

    /// Stand-in for `ryft_core::partial::PartiallyEvaluatableOperation`.
    pub(crate) trait PartiallyEvaluatableOperation<C: Context>: Clone + Into<C::Operation> {
        fn partially_evaluate(
            &self,
            evaluator: &mut PartialEvaluator<C>,
            inputs: &[PartialEvaluationValue<C::Type, C::Value>],
        ) -> Result<Vec<PartialEvaluationValue<C::Type, C::Value>>, ProgramError>
        where
            Self: Clone + Into<C::Operation>,
        {
            let _ = (evaluator.context(), inputs);
            Ok(Vec::new())
        }
    }
}

fn transposed<T: Type, V: Value<T>, O: Operation<T>>(label: &'static str) -> Cotangent<T, V, O> {
    Cotangent { label, marker: PhantomData }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DataType;

impl Type for DataType {}
impl DifferentiableType for DataType {}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ArrayType;

impl Type for ArrayType {}
impl DifferentiableType for ArrayType {}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Factor(i64);

impl Value<DataType> for Factor {}

impl Value<ArrayType> for Factor {}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TranspositionFactor(i64);

impl Value<ArrayType> for TranspositionFactor {}

impl BooleanLike for Factor {}

impl Slice for Factor {}

impl UpdateSlice for Factor {}

impl Reshape for Factor {}

trait SpecialTransposableValue {}

impl SpecialTransposableValue for Factor {}

impl SpecialTransposableValue for TranspositionFactor {}

/// Extra value bound a recursive payload's partial-evaluation rule requires, used to exercise
/// `#[ryft(bounds(partial_evaluation(...)))]`.
trait SpecialPartiallyEvaluatableValue {}

impl SpecialPartiallyEvaluatableValue for Factor {}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ZeroOperation<T: Type> {
    r#type: T,
}

impl<T: Clone + Type> Operation<T> for ZeroOperation<T> {
    fn name(&self) -> &'static str {
        "zero"
    }

    fn infer_output_types(&self, _input_types: &[T]) -> Result<Vec<T>, TypeError> {
        Ok(vec![self.r#type.clone()])
    }
}

impl<T: Clone + Type, V: Value<T>, C> InterpretableOperation<T, V, C> for ZeroOperation<T> {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<T: Clone + Type, C: Context<Type = T>> partial::PartiallyEvaluatableOperation<C> for ZeroOperation<T> where C::Operation: From<ZeroOperation<T>> {}

impl<T: Clone + Type, V: Value<T>, O: Operation<T>> TransposableOperation<T, V, O> for ZeroOperation<T> {
    fn transpose(
        &self,
        _context: &mut TracingContext<T, V, O>,
        _inputs: &[PartialValue<T, Tracer<TracingContext<T, V, O>>>],
        _outputs: &[Cotangent<T, V, O>],
    ) -> Result<Vec<Cotangent<T, V, O>>, ProgramError> {
        Ok(vec![transposed("zero")])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AddOperation;

impl Operation<DataType> for AddOperation {
    fn name(&self) -> &'static str {
        "add"
    }

    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<V: Value<DataType>, C> InterpretableOperation<DataType, V, C> for AddOperation {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<C: Context<Type = DataType>> partial::PartiallyEvaluatableOperation<C> for AddOperation where C::Operation: From<AddOperation> {}

impl<V: Value<DataType>, O: Operation<DataType>> TransposableOperation<DataType, V, O> for AddOperation {
    fn transpose(
        &self,
        _context: &mut TracingContext<DataType, V, O>,
        _inputs: &[PartialValue<DataType, Tracer<TracingContext<DataType, V, O>>>],
        _outputs: &[Cotangent<DataType, V, O>],
    ) -> Result<Vec<Cotangent<DataType, V, O>>, ProgramError> {
        Ok(vec![transposed("add")])
    }
}

/// Stand-in for an operation with observable effects (e.g., `ryft_core`'s `PrintOperation`), overriding the
/// defaulted `Operation::effects` so the generated enum forwarding is observable.
#[derive(Clone, Debug, PartialEq, Eq)]
struct PrintOperation;

impl Operation<DataType> for PrintOperation {
    fn name(&self) -> &'static str {
        "print"
    }

    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        Ok(input_types.to_vec())
    }

    fn effects(&self) -> Effects {
        Effects::Ordered
    }
}

impl<V: Value<DataType>, C> InterpretableOperation<DataType, V, C> for PrintOperation {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<C: Context<Type = DataType>> partial::PartiallyEvaluatableOperation<C> for PrintOperation where C::Operation: From<PrintOperation> {}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FactorOperation<T: Type, V> {
    factor: V,
    marker: PhantomData<T>,
}

impl<T: Clone + Type, V> Operation<T> for FactorOperation<T, V> {
    fn name(&self) -> &'static str {
        "factor"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<T: Clone + Type, V: Value<T>, F, C> InterpretableOperation<T, V, C> for FactorOperation<T, F> {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<T: Clone + Type, F: Clone, C: Context<Type = T>> partial::PartiallyEvaluatableOperation<C>
    for FactorOperation<T, F>
where
    C::Operation: From<FactorOperation<T, F>>,
{
}

impl<T: Clone + Type, V: Value<T>, O: Operation<T>, F> TransposableOperation<T, V, O> for FactorOperation<T, F> {
    fn transpose(
        &self,
        _context: &mut TracingContext<T, V, O>,
        _inputs: &[PartialValue<T, Tracer<TracingContext<T, V, O>>>],
        _outputs: &[Cotangent<T, V, O>],
    ) -> Result<Vec<Cotangent<T, V, O>>, ProgramError> {
        Ok(vec![transposed("factor")])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ConstantOperation<T: Type, V> {
    value: V,
    marker: PhantomData<T>,
}

impl<T: Clone + Type, V> Operation<T> for ConstantOperation<T, V> {
    fn name(&self) -> &'static str {
        "constant"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<T: Clone + Type, V: Value<T>, Constant, C> InterpretableOperation<T, V, C> for ConstantOperation<T, Constant> {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<T: Clone + Type, Constant: Clone, C: Context<Type = T>> partial::PartiallyEvaluatableOperation<C>
    for ConstantOperation<T, Constant>
where
    C::Operation: From<ConstantOperation<T, Constant>>,
{
}

impl<T: Clone + Type, V: Value<T>, O: Operation<T>, F> TransposableOperation<T, V, O> for ConstantOperation<T, F> {
    fn transpose(
        &self,
        _context: &mut TracingContext<T, V, O>,
        _inputs: &[PartialValue<T, Tracer<TracingContext<T, V, O>>>],
        _outputs: &[Cotangent<T, V, O>],
    ) -> Result<Vec<Cotangent<T, V, O>>, ProgramError> {
        Ok(vec![transposed("constant")])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CustomJvpOperation<T: Type, V> {
    tag: &'static str,
    marker: PhantomData<(T, V)>,
}

impl<T: Clone + Type, V> Operation<T> for CustomJvpOperation<T, V> {
    fn name(&self) -> &'static str {
        "custom_jvp"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<T: Clone + Type, V: Value<T>, Constant, C> InterpretableOperation<T, V, C> for CustomJvpOperation<T, Constant> {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<T: Clone + Type, Constant: Clone, C: Context<Type = T>> partial::PartiallyEvaluatableOperation<C>
    for CustomJvpOperation<T, Constant>
where
    C::Operation: From<CustomJvpOperation<T, Constant>>,
{
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation)]
#[ryft(crate = "crate")]
enum ScalarOperation<V: Value<DataType>> {
    Zero(ZeroOperation<DataType>),
    Add(AddOperation),
    Print(PrintOperation),
    Factor(FactorOperation<DataType, V>),
    CustomJvp(Box<CustomJvpOperation<DataType, V>>),
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation, ryft::TransposableOperation)]
#[ryft(crate = "crate")]
enum LinearScalarOperation<V: Value<DataType>, C: Value<DataType> = V> {
    Zero(ZeroOperation<DataType>),
    Constant(ConstantOperation<DataType, V>),
    Add(AddOperation),
    Factor(FactorOperation<DataType, C>),
}

#[test]
fn test_scalar_operation() {
    let zero = ScalarOperation::<Factor>::from(ZeroOperation { r#type: DataType });
    let add = ScalarOperation::<Factor>::from(AddOperation);
    let factor = ScalarOperation::<Factor>::from(FactorOperation { factor: Factor(7), marker: PhantomData });
    let custom_jvp = ScalarOperation::<Factor>::from(CustomJvpOperation { tag: "tag", marker: PhantomData });

    assert_eq!(zero.name(), "zero");
    assert_eq!(add.name(), "add");
    assert_eq!(factor.name(), "factor");
    assert_eq!(custom_jvp.name(), "custom_jvp");

    assert_eq!(add.infer_output_types(&[DataType]), Ok(vec![DataType]));
    assert_eq!(zero.to_string(), "zero");
    assert_eq!(custom_jvp.to_string(), "custom_jvp");

    assert_eq!(<&ZeroOperation<DataType>>::try_from(&zero), Ok(&ZeroOperation { r#type: DataType }));
    assert_eq!(<&AddOperation>::try_from(&add), Ok(&AddOperation));
    assert_eq!(
        <&FactorOperation<DataType, Factor>>::try_from(&factor),
        Ok(&FactorOperation { factor: Factor(7), marker: PhantomData }),
    );
    assert_eq!(
        <&CustomJvpOperation<DataType, Factor>>::try_from(&custom_jvp),
        Ok(&CustomJvpOperation { tag: "tag", marker: PhantomData }),
    );
    assert_eq!(<&AddOperation>::try_from(&zero), Err(()));
}

#[test]
fn test_operation_generates_effects_forwarding() {
    let add = ScalarOperation::<Factor>::from(AddOperation);
    let print = ScalarOperation::<Factor>::from(PrintOperation);

    assert_eq!(add.effects(), Effects::Pure);
    assert_eq!(print.effects(), Effects::Ordered);
}

#[test]
fn test_operation_generates_interpretation_forwarding() {
    let context = TestContext::<DataType, Factor> { marker: PhantomData };
    let operation = ScalarOperation::<Factor>::from(AddOperation);

    assert_eq!(operation.interpret(&context, &[Factor(1), Factor(2)]), Ok(vec![Factor(1), Factor(2)]),);
}

#[test]
fn test_operation_generates_captured_program_interpretation_witness() {
    type Operation = ScalarOperation<Factor>;

    let context = TestContext::<DataType, Factor> { marker: PhantomData };
    let program = Program::<DataType, Factor, Operation, Vec<Factor>, Vec<Factor>> {
        label: "scalar",
        constant: Some(Factor(3)),
        operation: Some(Operation::from(AddOperation)),
        marker: PhantomData,
    };
    let outputs = <Operation as InterpretableProgramOperation<
        DataType,
        Factor,
        TestContext<DataType, Factor>,
        Factor,
    >>::interpret_program(&context, &program, vec![Factor(1)])
    .unwrap();

    assert_eq!(outputs, vec![Factor(1), Factor(3)]);
}

#[test]
fn test_operation_generates_direct_program_interpretation_witness() {
    type Operation = LinearScalarOperation<Factor>;

    let context = TestContext::<DataType, Factor> { marker: PhantomData };
    let operation = Operation::from(FactorOperation { factor: Factor(5), marker: PhantomData });

    assert_eq!(operation.interpret(&context, &[Factor(8)]), Ok(vec![Factor(8)]));

    let program = Program::<DataType, Factor, Operation, Vec<Factor>, Vec<Factor>> {
        label: "linear",
        constant: Some(Factor(13)),
        operation: Some(Operation::from(AddOperation)),
        marker: PhantomData,
    };
    let outputs = <Operation as InterpretableProgramOperation<DataType, Factor, TestContext<DataType, Factor>>>::interpret_program(
        &context,
        &program,
        vec![Factor(8)],
    )
    .unwrap();

    assert_eq!(outputs, vec![Factor(8), Factor(13)]);
}

#[test]
fn test_operation_generates_partial_evaluation_witness() {
    fn assert_partially_evaluatable<C: Context, O: partial::PartiallyEvaluatableOperation<C>>() {}

    // The derive now forwards partial evaluation for every variant, so each enum must satisfy the per-operation
    // partial-evaluation trait at any known-side context pinned to its program-constant value type and to itself as
    // the residual operation family. This covers leaf payloads, the generic `Backend` payload, and the boxed
    // nested-program payloads.
    assert_partially_evaluatable::<TestContext<DataType, Factor, ScalarOperation<Factor>>, ScalarOperation<Factor>>();
    assert_partially_evaluatable::<
        TestContext<DataType, Factor, LinearScalarOperation<Factor>>,
        LinearScalarOperation<Factor>,
    >();
    assert_partially_evaluatable::<TestContext<ArrayType, Factor, ArrayOperation<Factor>>, ArrayOperation<Factor>>();
    assert_partially_evaluatable::<
        TestContext<ArrayType, Factor, LinearArrayOperation<Factor, Factor>>,
        LinearArrayOperation<Factor, Factor>,
    >();
}

#[test]
fn test_operation_generates_partial_evaluation_value_bounds() {
    // The `Recursive` payload's partial-evaluation rule requires `SpecialPartiallyEvaluatableValue`, supplied to the
    // generated implementation by `#[ryft(bounds(partial_evaluation(...)))]`. Proving the witness for the enum
    // discharges that recursive arm's body obligation, which only resolves when the extra bound is injected.
    use partial::PartiallyEvaluatableOperation as _;

    fn assert_partially_evaluatable<C: Context, O: partial::PartiallyEvaluatableOperation<C>>() {}
    assert_partially_evaluatable::<
        TestContext<ArrayType, Factor, PartialEvaluationBoundOperation<Factor>>,
        PartialEvaluationBoundOperation<Factor>,
    >();

    let context =
        TestContext::<ArrayType, Factor, PartialEvaluationBoundOperation<Factor>> { marker: PhantomData };
    let mut evaluator = partial::PartialEvaluator::new(context);
    let operation = PartialEvaluationBoundOperation::<Factor>::from(ZeroOperation { r#type: ArrayType });
    let evaluation = operation.partially_evaluate(&mut evaluator, &[]).unwrap();
    assert!(evaluation.is_empty());
}

#[test]
fn test_transposable_operation_infers_value_type() {
    type Linear = LinearScalarOperation<Factor>;

    let mut context = TracingContext::<DataType, Factor, Linear> { marker: PhantomData };
    let add = Linear::from(AddOperation);

    assert_eq!(
        add.transpose(&mut context, &[PartialValue::Unknown(DataType)], &[]).unwrap(),
        vec![transposed::<DataType, Factor, Linear>("add")],
    );
}

#[derive(Clone, Debug, ryft::Operation)]
enum DefaultPathOperation<V: ryft::Value<ryft::DataType>> {
    Zero(ryft::ZeroOperation<ryft::DataType>),
    Constant(ryft::ConstantOperation<ryft::DataType, V>),
}

#[derive(Clone, Debug, ryft::Operation, ryft::TransposableOperation)]
enum DefaultPathLinearOperation<V: ryft::Value<ryft::DataType>> {
    Zero(ryft::ZeroOperation<ryft::DataType>),
    Constant(ryft::ConstantOperation<ryft::DataType, V>),
}

#[test]
fn test_operation_default_crate_path_is_ryft() {
    let operation = DefaultPathOperation::<ryft::Scalar>::from(ryft::ZeroOperation::new(ryft::DataType::F64));
    let linear_operation =
        DefaultPathLinearOperation::<ryft::Scalar>::from(ryft::ZeroOperation::new(ryft::DataType::F64));
    assert_eq!(ryft::Operation::name(&operation), "zero");
    assert_eq!(ryft::Operation::name(&linear_operation), "zero");
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DotOperation;

impl Operation<ArrayType> for DotOperation {
    fn name(&self) -> &'static str {
        "dot"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<V: Value<ArrayType>, C> InterpretableOperation<ArrayType, V, C> for DotOperation {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<C: Context<Type = ArrayType>> partial::PartiallyEvaluatableOperation<C> for DotOperation where C::Operation: From<DotOperation> {}

#[derive(Clone, Debug, PartialEq, Eq)]
enum BackendPayload {}

impl Operation<ArrayType> for BackendPayload {
    fn name(&self) -> &'static str {
        match *self {}
    }

    fn infer_output_types(&self, _input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        match *self {}
    }
}

impl<V: Value<ArrayType>, C> InterpretableOperation<ArrayType, V, C> for BackendPayload {
    fn interpret(&self, _context: &C, _inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        match *self {}
    }
}

impl<C: Context<Type = ArrayType>> partial::PartiallyEvaluatableOperation<C> for BackendPayload where C::Operation: From<BackendPayload> {}

impl<V: Value<ArrayType>, O: Operation<ArrayType>> TransposableOperation<ArrayType, V, O> for BackendPayload {
    fn transpose(
        &self,
        _context: &mut TracingContext<ArrayType, V, O>,
        _inputs: &[PartialValue<ArrayType, Tracer<TracingContext<ArrayType, V, O>>>],
        _outputs: &[Cotangent<ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<ArrayType, V, O>>, ProgramError> {
        match *self {}
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SpecialOperation;

impl Operation<ArrayType> for SpecialOperation {
    fn name(&self) -> &'static str {
        "special"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<V: Value<ArrayType>, C> InterpretableOperation<ArrayType, V, C> for SpecialOperation {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<C: Context<Type = ArrayType>> partial::PartiallyEvaluatableOperation<C> for SpecialOperation where C::Operation: From<SpecialOperation> {}

impl<V, O> TransposableOperation<ArrayType, V, O> for SpecialOperation
where
    V: Value<ArrayType> + SpecialTransposableValue,
    O: Operation<ArrayType>,
{
    fn transpose(
        &self,
        _context: &mut TracingContext<ArrayType, V, O>,
        _inputs: &[PartialValue<ArrayType, Tracer<TracingContext<ArrayType, V, O>>>],
        _outputs: &[Cotangent<ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<ArrayType, V, O>>, ProgramError> {
        Ok(vec![transposed("special")])
    }
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation, ryft::TransposableOperation)]
#[ryft(crate = "crate")]
enum SpecialLinearOperation<V: Value<ArrayType>> {
    Special(SpecialOperation),
    Constant(ConstantOperation<ArrayType, V>),
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation)]
#[ryft(crate = "crate")]
enum InferredArrayOperation<V: Value<ArrayType>, C: Value<ArrayType> = V> {
    Zero(ZeroOperation<ArrayType>),
    Constant(ConstantOperation<ArrayType, V>),
    Factor(FactorOperation<ArrayType, C>),
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation)]
#[ryft(crate = "crate")]
#[ryft(bounds(interpretation(Slice + UpdateSlice + Reshape)))]
enum InterpretationBoundOperation<C: Value<ArrayType> + BooleanLike> {
    Zero(ZeroOperation<ArrayType>),
    Constant(ConstantOperation<ArrayType, C>),
}

#[test]
fn test_array_operation_type_inference() {
    type Operation = InferredArrayOperation<Factor>;

    let zero = Operation::from(ZeroOperation { r#type: ArrayType });
    let constant = Operation::from(ConstantOperation { value: Factor(5), marker: PhantomData });
    let factor = Operation::from(FactorOperation { factor: Factor(17), marker: PhantomData });

    assert_eq!(zero.name(), "zero");
    assert_eq!(constant.name(), "constant");
    assert_eq!(factor.name(), "factor");
    assert_eq!(zero.infer_output_types(&[]), Ok(vec![ArrayType]));
    assert_eq!(
        <&FactorOperation<ArrayType, Factor>>::try_from(&factor),
        Ok(&FactorOperation { factor: Factor(17), marker: PhantomData }),
    );
}

#[test]
fn test_operation_generates_interpretation_value_bounds() {
    type Operation = InterpretationBoundOperation<Factor>;

    let context = TestContext::<ArrayType, Factor> { marker: PhantomData };
    let operation = Operation::from(ZeroOperation { r#type: ArrayType });

    assert_eq!(operation.interpret(&context, &[Factor(1)]), Ok(vec![Factor(1)]));

    let program = Program::<ArrayType, Factor, Operation, Vec<Factor>, Vec<Factor>> {
        label: "array",
        constant: Some(Factor(3)),
        operation: Some(Operation::from(ConstantOperation { value: Factor(5), marker: PhantomData })),
        marker: PhantomData,
    };
    let outputs = <Operation as InterpretableProgramOperation<
        ArrayType,
        Factor,
        TestContext<ArrayType, Factor>,
        Factor,
    >>::interpret_program(&context, &program, vec![Factor(1)])
    .unwrap();

    assert_eq!(outputs, vec![Factor(1), Factor(3)]);
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation)]
#[ryft(crate = "crate")]
enum ArrayOperation<V: Value<ArrayType>, Backend = BackendPayload> {
    Zero(ZeroOperation<ArrayType>),
    Dot(DotOperation),
    Factor(FactorOperation<ArrayType, V>),
    Backend(Backend),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct WhileOperation<T: Type, V, O> {
    marker: PhantomData<(T, V, O)>,
}

impl<T: Clone + Type, V, O> Operation<T> for WhileOperation<T, V, O> {
    fn name(&self) -> &'static str {
        "while"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<T: Clone + Type, V: Value<T>, W, O, C> InterpretableOperation<T, V, C> for WhileOperation<T, W, O> {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<T: Clone + Type, W: Clone, O: Clone + Operation<T>, C: Context<Type = T>>
    partial::PartiallyEvaluatableOperation<C> for WhileOperation<T, W, O>
where
    C::Operation: From<WhileOperation<T, W, O>>,
{
}

impl<T: Clone + Type, V: Value<T>, O: Operation<T>, W, P> TransposableOperation<T, V, O> for WhileOperation<T, W, P> {
    fn transpose(
        &self,
        _context: &mut TracingContext<T, V, O>,
        _inputs: &[PartialValue<T, Tracer<TracingContext<T, V, O>>>],
        _outputs: &[Cotangent<T, V, O>],
    ) -> Result<Vec<Cotangent<T, V, O>>, ProgramError> {
        Ok(vec![transposed("while")])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RecursiveOperation<V, O> {
    marker: PhantomData<(V, O)>,
}

impl<V, O> Operation<ArrayType> for RecursiveOperation<V, O> {
    fn name(&self) -> &'static str {
        "recursive"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<V: Value<ArrayType>, W, O, C> InterpretableOperation<ArrayType, V, C> for RecursiveOperation<W, O> {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<W: Clone, O: Clone + Operation<ArrayType>, C: Context<Type = ArrayType>>
    partial::PartiallyEvaluatableOperation<C> for RecursiveOperation<W, O>
where
    C::Operation: From<RecursiveOperation<W, O>>,
{
}

impl<StoredValue, TranspositionValue, O> TransposableOperation<ArrayType, TranspositionValue, O>
    for RecursiveOperation<StoredValue, O>
where
    TranspositionValue: Value<ArrayType>,
    O: RecursiveOperationTransposable<TranspositionValue>,
{
    fn transpose(
        &self,
        _context: &mut TracingContext<ArrayType, TranspositionValue, O>,
        _inputs: &[PartialValue<ArrayType, Tracer<TracingContext<ArrayType, TranspositionValue, O>>>],
        _outputs: &[Cotangent<ArrayType, TranspositionValue, O>],
    ) -> Result<Vec<Cotangent<ArrayType, TranspositionValue, O>>, ProgramError> {
        Ok(vec![transposed("recursive")])
    }
}

trait RecursiveOperationTransposable<V: Value<ArrayType>>: Operation<ArrayType> {}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation, ryft::TransposableOperation)]
#[ryft(crate = "crate")]
enum RecursiveLinearOperation<V: Value<ArrayType>> {
    Zero(ZeroOperation<ArrayType>),
    Recursive(RecursiveOperation<V, Self>),
}

impl<StoredValue: Value<ArrayType>, TranspositionValue: Value<ArrayType>>
    RecursiveOperationTransposable<TranspositionValue> for RecursiveLinearOperation<StoredValue>
{
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ProgramRecursiveOperation<V, O> {
    marker: PhantomData<(V, O)>,
}

impl<V, O> Operation<DataType> for ProgramRecursiveOperation<V, O> {
    fn name(&self) -> &'static str {
        "program_recursive"
    }

    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<V: Value<DataType>, W, O, C> InterpretableOperation<DataType, V, C> for ProgramRecursiveOperation<W, O> {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<W: Clone, O: Clone + Operation<DataType>, C: Context<Type = DataType>>
    partial::PartiallyEvaluatableOperation<C> for ProgramRecursiveOperation<W, O>
where
    C::Operation: From<ProgramRecursiveOperation<W, O>>,
{
}

impl<StoredValue, TranspositionValue, O> TransposableOperation<DataType, TranspositionValue, O>
    for ProgramRecursiveOperation<StoredValue, O>
where
    TranspositionValue: Value<DataType> + SpecialTransposableValue,
    O: TransposableProgramOperation<DataType, TranspositionValue>,
{
    fn transpose(
        &self,
        _context: &mut TracingContext<DataType, TranspositionValue, O>,
        _inputs: &[PartialValue<DataType, Tracer<TracingContext<DataType, TranspositionValue, O>>>],
        _outputs: &[Cotangent<DataType, TranspositionValue, O>],
    ) -> Result<Vec<Cotangent<DataType, TranspositionValue, O>>, ProgramError> {
        Ok(vec![transposed("program_recursive")])
    }
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation, ryft::TransposableOperation)]
#[ryft(crate = "crate")]
enum RecursiveProgramLinearOperation<V: Value<DataType> + SpecialTransposableValue> {
    Zero(ZeroOperation<DataType>),
    Add(AddOperation),
    Recursive(ProgramRecursiveOperation<V, Self>),
}

/// Recursive payload whose partial-evaluation rule requires an extra [`SpecialPartiallyEvaluatableValue`] bound on the
/// value type, mirroring how the array scan's carry-folding rule requires `PartialEq`. The owning enum supplies that
/// bound to the generated partial-evaluation implementation through `#[ryft(bounds(partial_evaluation(...)))]`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct PartialEvaluationRecursiveOperation<V, O> {
    marker: PhantomData<(V, O)>,
}

impl<V, O> Operation<ArrayType> for PartialEvaluationRecursiveOperation<V, O> {
    fn name(&self) -> &'static str {
        "partial_evaluation_recursive"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<V: Value<ArrayType>, W, O, C> InterpretableOperation<ArrayType, V, C>
    for PartialEvaluationRecursiveOperation<W, O>
{
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<W: Clone, O: Clone + Operation<ArrayType>, C: Context<Type = ArrayType>>
    partial::PartiallyEvaluatableOperation<C> for PartialEvaluationRecursiveOperation<W, O>
where
    C::Value: SpecialPartiallyEvaluatableValue,
    C::Operation: From<PartialEvaluationRecursiveOperation<W, O>>,
{
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation)]
#[ryft(crate = "crate")]
#[ryft(bounds(partial_evaluation(SpecialPartiallyEvaluatableValue)))]
enum PartialEvaluationBoundOperation<V: Value<ArrayType>> {
    Zero(ZeroOperation<ArrayType>),
    Recursive(PartialEvaluationRecursiveOperation<V, Self>),
}

/// Stand-in value capability required by one payload's forward-mode rule, verifying that the generated per-variant
/// `DifferentiableOperation` predicates transport payload capability requirements to the use site.
trait SpecialCombine {
    type Output;
}

impl SpecialCombine for Factor {
    type Output = Factor;
}

impl<C: StagingContext<Type = DataType>> DifferentiableOperation<C> for ZeroOperation<DataType> {
    fn jvp(&self, _context: &C, _inputs: &[JvpTracer<C>]) -> Result<Vec<JvpTracer<C>>, ProgramError> {
        Ok(vec![JvpTracer { label: "zero", marker: PhantomData }])
    }
}

impl<C: StagingContext<Type = DataType>> DifferentiableOperation<C> for AddOperation {
    fn jvp(&self, _context: &C, _inputs: &[JvpTracer<C>]) -> Result<Vec<JvpTracer<C>>, ProgramError> {
        Ok(vec![JvpTracer { label: "add", marker: PhantomData }])
    }
}

/// Forward-mode rule requiring a value capability that the generated per-variant predicate transports to the
/// owning enum's use sites without the enum spelling it.
impl<C, F> DifferentiableOperation<C> for FactorOperation<DataType, F>
where
    C: StagingContext<Type = DataType>,
    C::Value: SpecialCombine<Output = C::Value>,
{
    fn jvp(&self, _context: &C, _inputs: &[JvpTracer<C>]) -> Result<Vec<JvpTracer<C>>, ProgramError> {
        Ok(vec![JvpTracer { label: "factor", marker: PhantomData }])
    }
}

/// Stand-in payload without a capture-free forward-mode rule, mirroring payload-level erroring
/// `DifferentiableOperation` implementations such as the scalar `while` rule: the generated dispatcher still
/// delegates uniformly and the payload's own rule reports the error.
#[derive(Clone, Debug, PartialEq, Eq)]
struct NonDifferentiableOperation;

impl Operation<DataType> for NonDifferentiableOperation {
    fn name(&self) -> &'static str {
        "non_differentiable"
    }

    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<V: Value<DataType>, C> InterpretableOperation<DataType, V, C> for NonDifferentiableOperation {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<C: Context<Type = DataType>> partial::PartiallyEvaluatableOperation<C> for NonDifferentiableOperation where C::Operation: From<NonDifferentiableOperation> {}

impl<V: Value<DataType>, O: Operation<DataType>> TransposableOperation<DataType, V, O> for NonDifferentiableOperation {
    fn transpose(
        &self,
        _context: &mut TracingContext<DataType, V, O>,
        _inputs: &[PartialValue<DataType, Tracer<TracingContext<DataType, V, O>>>],
        _outputs: &[Cotangent<DataType, V, O>],
    ) -> Result<Vec<Cotangent<DataType, V, O>>, ProgramError> {
        Ok(vec![transposed("non_differentiable")])
    }
}

impl<C: StagingContext<Type = DataType>> DifferentiableOperation<C> for NonDifferentiableOperation {
    fn jvp(&self, _context: &C, _inputs: &[JvpTracer<C>]) -> Result<Vec<JvpTracer<C>>, ProgramError> {
        Err(ProgramError)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation, ryft::TransposableOperation, ryft::DifferentiableOperation)]
#[ryft(crate = "crate")]
enum DifferentiableScalarOperation<V: Value<DataType>> {
    Zero(ZeroOperation<DataType>),
    Add(AddOperation),
    Factor(FactorOperation<DataType, V>),
    NonDifferentiable(NonDifferentiableOperation),
}

impl<V: Value<DataType>> DifferentiableProgramOperation<DataType, V, DifferentiableScalarOperation<V>>
    for DifferentiableScalarOperation<V>
{
}

#[test]
fn test_differentiable_operation_dispatches_jvp_to_payloads() {
    type Operation = DifferentiableScalarOperation<Factor>;

    let context = TestContext::<DataType, Factor, Operation> { marker: PhantomData };

    let zero = Operation::from(ZeroOperation { r#type: DataType });
    let outputs = zero.jvp(&context, &[]).unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].label, "zero");

    let add = Operation::from(AddOperation);
    let outputs = add.jvp(&context, &[]).unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].label, "add");

    // The `Factor` rule requires `SpecialCombine<Output = C::Value>` on the flowing value, transported to this use
    // site by the generated per-variant `DifferentiableOperation` predicate.
    let factor = Operation::from(FactorOperation { factor: Factor(3), marker: PhantomData });
    let outputs = factor.jvp(&context, &[]).unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].label, "factor");
}

#[test]
fn test_differentiable_operation_delegates_unsupported_payloads() {
    type Operation = DifferentiableScalarOperation<Factor>;

    // The generated dispatcher delegates uniformly, so the unsupported payload's own erroring rule reports the
    // failure rather than an enum-level dispatch arm.
    let context = TestContext::<DataType, Factor, Operation> { marker: PhantomData };
    let operation = Operation::from(NonDifferentiableOperation);
    assert_eq!(operation.jvp(&context, &[]).unwrap_err(), ProgramError);
}

#[test]
fn test_differentiable_operation_generates_transposition_dispatchers() {
    type Operation = DifferentiableScalarOperation<Factor>;

    // The transposition dispatchers come from the separate `TransposableOperation` derive: `DifferentiableOperation`
    // only adds forward-mode support, and deriving both enables reverse mode too.
    let mut context = TracingContext::<DataType, Factor, Operation> { marker: PhantomData };
    let add = Operation::from(AddOperation);
    assert_eq!(
        add.transpose(&mut context, &[PartialValue::Unknown(DataType)], &[]).unwrap(),
        vec![transposed::<DataType, Factor, Operation>("add")],
    );

    let program = Program::<DataType, Factor, Operation, Vec<Factor>, Vec<Factor>> {
        label: "differentiable",
        constant: None,
        operation: None,
        marker: PhantomData,
    };
    let transposed_program =
        <Operation as TransposableProgramOperation<DataType, Factor>>::transpose_program(&program, &[true]).unwrap();
    assert_eq!(transposed_program.label, "program_transpose_partitioned");
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RecomputeOperation<O> {
    operation: O,
}

impl<O: Operation<ArrayType>> Operation<ArrayType> for RecomputeOperation<O> {
    fn name(&self) -> &'static str {
        self.operation.name()
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        self.operation.infer_output_types(input_types)
    }
}

impl<V: Value<ArrayType>, O: Operation<ArrayType>, C> InterpretableOperation<ArrayType, V, C>
    for RecomputeOperation<O>
{
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<RecomputedOperation: Clone + Operation<ArrayType>, C: Context<Type = ArrayType>>
    partial::PartiallyEvaluatableOperation<C> for RecomputeOperation<RecomputedOperation>
where
    C::Operation: From<RecomputeOperation<RecomputedOperation>>,
{
}

impl<V: Value<ArrayType>, O: Operation<ArrayType>, P: Operation<ArrayType>> TransposableOperation<ArrayType, V, O>
    for RecomputeOperation<P>
{
    fn transpose(
        &self,
        _context: &mut TracingContext<ArrayType, V, O>,
        _inputs: &[PartialValue<ArrayType, Tracer<TracingContext<ArrayType, V, O>>>],
        _outputs: &[Cotangent<ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<ArrayType, V, O>>, ProgramError> {
        Ok(vec![transposed("recompute")])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CustomVjpCallOperation<T: Type, C, O, F> {
    marker: PhantomData<(T, C, O, F)>,
}

impl<T: Clone + Type, C, O, F> Operation<T> for CustomVjpCallOperation<T, C, O, F> {
    fn name(&self) -> &'static str {
        "custom_vjp_call"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<T: Clone + Type, V: Value<T>, Constant, O, F, C> InterpretableOperation<T, V, C>
    for CustomVjpCallOperation<T, Constant, O, F>
{
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<T: Clone + Type, Constant: Clone, CallOperation: Clone, F: Clone, C: Context<Type = T>>
    partial::PartiallyEvaluatableOperation<C> for CustomVjpCallOperation<T, Constant, CallOperation, F>
where
    C::Operation: From<CustomVjpCallOperation<T, Constant, CallOperation, F>>,
{
}

impl<T: Clone + Type, V: Value<T>, O: Operation<T>, C, P, F> TransposableOperation<T, V, O>
    for CustomVjpCallOperation<T, C, P, F>
{
    fn transpose(
        &self,
        _context: &mut TracingContext<T, V, O>,
        _inputs: &[PartialValue<T, Tracer<TracingContext<T, V, O>>>],
        _outputs: &[Cotangent<T, V, O>],
    ) -> Result<Vec<Cotangent<T, V, O>>, ProgramError> {
        Ok(vec![transposed("custom_vjp_call")])
    }
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation, ryft::TransposableOperation)]
#[ryft(crate = "crate")]
enum LinearArrayOperation<
    V: Value<ArrayType>,
    C: Value<ArrayType>,
    Backend = BackendPayload,
    F: Value<ArrayType> = V,
    P: Operation<ArrayType> = ArrayOperation<C, Backend>,
> {
    Zero(ZeroOperation<ArrayType>),
    Factor(FactorOperation<ArrayType, F>),
    Recompute(RecomputeOperation<P>),
    While(Box<WhileOperation<ArrayType, V, Self>>),
    CustomVjpCall(Box<CustomVjpCallOperation<ArrayType, C, P, F>>),
    Backend(Backend),
}

#[test]
fn test_array_operation_generic_payload_conversion_skip() {
    let zero = ArrayOperation::<Factor>::from(ZeroOperation { r#type: ArrayType });
    let dot = ArrayOperation::<Factor>::from(DotOperation);
    let factor = ArrayOperation::<Factor>::from(FactorOperation { factor: Factor(11), marker: PhantomData });

    assert_eq!(zero.name(), "zero");
    assert_eq!(dot.name(), "dot");
    assert_eq!(factor.name(), "factor");
    assert_eq!(dot.infer_output_types(&[ArrayType]), Ok(vec![ArrayType]));
    assert_eq!(dot.to_string(), "dot");

    assert_eq!(<&DotOperation>::try_from(&dot), Ok(&DotOperation));
    assert_eq!(
        <&FactorOperation<ArrayType, Factor>>::try_from(&factor),
        Ok(&FactorOperation { factor: Factor(11), marker: PhantomData }),
    );

    // If the derive generated `From<Backend>` for `Backend(Backend)`, it would overlap with `From<DotOperation>`
    // because `Backend` is unconstrained and could be `DotOperation`. This test target compiling proves that the
    // bare generic payload was skipped automatically.
}

#[test]
fn test_linear_array_operation_shape() {
    type Linear = LinearArrayOperation<Factor, Factor>;

    let zero = Linear::from(ZeroOperation { r#type: ArrayType });
    let factor = Linear::from(FactorOperation { factor: Factor(13), marker: PhantomData });
    let recompute = Linear::from(RecomputeOperation { operation: ArrayOperation::<Factor>::from(DotOperation) });
    let while_operation = Linear::from(WhileOperation::<ArrayType, Factor, Linear> { marker: PhantomData });
    let custom_vjp_call =
        Linear::from(CustomVjpCallOperation::<ArrayType, Factor, ArrayOperation<Factor, BackendPayload>, Factor> {
            marker: PhantomData,
        });

    assert_eq!(zero.name(), "zero");
    assert_eq!(factor.name(), "factor");
    assert_eq!(recompute.name(), "dot");
    assert_eq!(while_operation.name(), "while");
    assert_eq!(custom_vjp_call.name(), "custom_vjp_call");
    assert_eq!(while_operation.infer_output_types(&[ArrayType]), Ok(vec![ArrayType]));
    assert_eq!(recompute.infer_output_types(&[ArrayType]), Ok(vec![ArrayType]));

    assert_eq!(recompute, Linear::Recompute(RecomputeOperation { operation: ArrayOperation::from(DotOperation) }));
    assert_eq!(
        <&RecomputeOperation<ArrayOperation<Factor, BackendPayload>>>::try_from(&recompute),
        Ok(&RecomputeOperation { operation: ArrayOperation::from(DotOperation) }),
    );
    assert_eq!(
        <&WhileOperation<ArrayType, Factor, Linear>>::try_from(&while_operation),
        Ok(&WhileOperation { marker: PhantomData }),
    );
    assert_eq!(
        <&CustomVjpCallOperation<ArrayType, Factor, ArrayOperation<Factor, BackendPayload>, Factor>>::try_from(
            &custom_vjp_call
        ),
        Ok(&CustomVjpCallOperation { marker: PhantomData }),
    );
    assert_eq!(<&ZeroOperation<ArrayType>>::try_from(&while_operation), Err(()));

    // `Backend(Backend)` is a bare generic payload, so its conversion is skipped automatically, while the
    // recompute wrapper and boxed payloads still expose conversions.
}

#[test]
fn test_transposable_operation_forwards_to_variant_payloads() {
    type Linear = LinearArrayOperation<Factor, Factor>;

    let mut context = TracingContext::<ArrayType, Factor, Linear> { marker: PhantomData };

    let zero = Linear::from(ZeroOperation { r#type: ArrayType });
    let factor = Linear::from(FactorOperation { factor: Factor(13), marker: PhantomData });
    let recompute = Linear::from(RecomputeOperation { operation: ArrayOperation::<Factor>::from(DotOperation) });
    let while_operation = Linear::from(WhileOperation::<ArrayType, Factor, Linear> { marker: PhantomData });
    let custom_vjp_call =
        Linear::from(CustomVjpCallOperation::<ArrayType, Factor, ArrayOperation<Factor, BackendPayload>, Factor> {
            marker: PhantomData,
        });

    assert_eq!(
        zero.transpose(&mut context, &[PartialValue::Unknown(ArrayType)], &[]).unwrap(),
        vec![transposed::<ArrayType, Factor, Linear>("zero")],
    );
    assert_eq!(
        factor.transpose(&mut context, &[PartialValue::Unknown(ArrayType)], &[]).unwrap(),
        vec![transposed::<ArrayType, Factor, Linear>("factor")],
    );
    assert_eq!(
        recompute.transpose(&mut context, &[PartialValue::Unknown(ArrayType)], &[]).unwrap(),
        vec![transposed::<ArrayType, Factor, Linear>("recompute")],
    );
    assert_eq!(
        while_operation.transpose(&mut context, &[PartialValue::Unknown(ArrayType)], &[]).unwrap(),
        vec![transposed::<ArrayType, Factor, Linear>("while")],
    );
    assert_eq!(
        custom_vjp_call.transpose(&mut context, &[PartialValue::Unknown(ArrayType)], &[]).unwrap(),
        vec![transposed::<ArrayType, Factor, Linear>("custom_vjp_call")],
    );
}

#[test]
fn test_transposable_operation_generates_program_transposition_witness() {
    type Linear = LinearScalarOperation<Factor>;

    let program = Program::<DataType, Factor, Linear, Vec<Factor>, Vec<Factor>> {
        label: "linear",
        constant: None,
        operation: None,
        marker: PhantomData,
    };
    let transposed =
        <Linear as TransposableProgramOperation<DataType, Factor>>::transpose_program(&program, &[true]).unwrap();

    assert_eq!(transposed.label, "program_transpose_partitioned");
}

#[test]
fn test_transposable_operation_generates_concrete_payload_bounds() {
    type Linear = SpecialLinearOperation<Factor>;

    let mut context = TracingContext::<ArrayType, TranspositionFactor, Linear> { marker: PhantomData };
    let operation = Linear::from(SpecialOperation);

    assert_eq!(
        operation.transpose(&mut context, &[PartialValue::Unknown(ArrayType)], &[]).unwrap(),
        vec![transposed::<ArrayType, TranspositionFactor, Linear>("special")],
    );
}

#[test]
fn test_transposable_operation_supports_recursive_payload_helpers() {
    type Linear = RecursiveLinearOperation<Factor>;

    let mut context = TracingContext::<ArrayType, Factor, Linear> { marker: PhantomData };
    let operation = Linear::from(RecursiveOperation::<Factor, Linear> { marker: PhantomData });

    assert_eq!(
        operation.transpose(&mut context, &[PartialValue::Unknown(ArrayType)], &[]).unwrap(),
        vec![transposed::<ArrayType, Factor, Linear>("recursive")],
    );
}

#[test]
fn test_transposable_operation_inherits_enum_bounds_for_recursive_program_witness() {
    type Linear = RecursiveProgramLinearOperation<Factor>;

    let mut context = TracingContext::<DataType, Factor, Linear> { marker: PhantomData };
    let operation = Linear::from(ProgramRecursiveOperation::<Factor, Linear> { marker: PhantomData });

    assert_eq!(
        operation.transpose(&mut context, &[PartialValue::Unknown(DataType)], &[]).unwrap(),
        vec![transposed::<DataType, Factor, Linear>("program_recursive")],
    );

    let program = Program::<DataType, Factor, Linear, Vec<Factor>, Vec<Factor>> {
        label: "linear",
        constant: None,
        operation: None,
        marker: PhantomData,
    };
    let transposed =
        <Linear as TransposableProgramOperation<DataType, Factor>>::transpose_program(&program, &[true]).unwrap();

    assert_eq!(transposed.label, "program_transpose_partitioned");
}

#[test]
fn test_errors() {
    let test_cases = trybuild::TestCases::new();
    test_cases.compile_fail("tests/operations/error_missing_type.rs");
    test_cases.compile_fail("tests/operations/error_ambiguous_type.rs");
    test_cases.compile_fail("tests/operations/error_bad_variant.rs");
    test_cases.compile_fail("tests/operations/error_bounds_attribute.rs");
    test_cases.compile_fail("tests/operations/error_unknown_bounds_attribute.rs");
    test_cases.compile_fail("tests/operations/error_unknown_transposition_bounds_attribute.rs");
    test_cases.compile_fail("tests/operations/error_type_attribute.rs");
}
