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

/// Stand-in for `ryft_core::BatchingError`, the error type the batching dispatchers return.
#[derive(Debug, PartialEq, Eq)]
struct BatchingError;

/// Stand-in for `ryft_core::Domain`.
trait Domain {
    type Type: Type;
    type Value: Value<Type = Self::Type>;
    type Constant: Value<Type = Self::Type>;
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
struct TestContext<V: Value, O: Operation<V::Type> = NoOperation> {
    marker: PhantomData<(V, O)>,
}

impl<V: Value, O: Operation<V::Type>> Domain for TestContext<V, O> {
    type Type = V::Type;
    type Value = V;
    type Constant = V;
    type Operation = O;
}

impl<V: Value, O: Operation<V::Type>> Context for TestContext<V, O> {
    fn lift(&self, constant: Self::Constant) -> Result<Self::Value, ProgramError> {
        Ok(constant)
    }
}

/// Stand-in for `ryft_core::Value`. Mirrors the real trait's associated `Type` descriptor, which the generated code
/// pins with `Value<Type = …>` equality bounds.
trait Value: Clone {
    type Type: Type;
}

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
trait Zero<V: Value> {}

impl<V: Value, O: Operation<V::Type>> Zero<V> for TestContext<V, O> {}

/// Stand-in for `ryft_core::payloads`.
mod payloads {
    /// Stand-in for `ryft_core::payloads::Captured`.
    pub struct Captured;
}

/// Stand-in for `ryft_core::Constant`.
trait Constant<V: Value, Stored, Payload = payloads::Captured> {
    fn constant(&self, value: Stored) -> Result<V, ProgramError>;
}

impl<V: Value, O: Operation<V::Type>, Stored: Clone, Payload> Constant<V, Stored, Payload> for TestContext<V, O>
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
trait InterpretableOperation<V: Value, C>: Operation<V::Type> {
    fn interpret(&self, context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError>;
}

/// Stand-in for `ryft_core::TracingContext`. Mirrors the real context's defaulted capture parameter and its
/// `StagingContext` membership at the capture-pinned form, which the generated program-batching witness names.
struct TracingContext<V: Value, O: Operation<V::Type>, Capture = V> {
    marker: PhantomData<(V, O, Capture)>,
}

impl<V: Value, O: Operation<V::Type>> Domain for TracingContext<V, O> {
    type Type = V::Type;
    type Value = Tracer<TracingContext<V, O>>;
    type Constant = V;
    type Operation = O;
}

impl<V: Value, O: Operation<V::Type>> Context for TracingContext<V, O> {
    fn lift(&self, _constant: Self::Constant) -> Result<Self::Value, ProgramError> {
        Err(ProgramError)
    }
}

impl<V: Value, O: Operation<V::Type>> StagingContext for TracingContext<V, O> {
    type Meta = ();
}

/// Stand-in for `ryft_core::Tracer`. Mirrors the real `Tracer`'s `Value` membership so it can be the value type of a
/// `PartialValue` input in the generated transpose signature, and its defaulted `Meta` parameter so the generated
/// batching dispatchers can name `Tracer<C, <C as StagingContext>::Meta>`.
struct Tracer<C, Meta = ()> {
    marker: PhantomData<(C, Meta)>,
}

impl<C, Meta> Clone for Tracer<C, Meta> {
    fn clone(&self) -> Self {
        Self { marker: PhantomData }
    }
}

impl<V: Value, O: Operation<V::Type>> Value for Tracer<TracingContext<V, O>> {
    type Type = V::Type;
}

/// Stand-in for `ryft_core::MaybeZero`. The manual trait implementations avoid bounding the value parameter, which
/// is instantiated at the `Debug`-less `Tracer` stand-in.
struct MaybeZero<V> {
    label: &'static str,
    marker: PhantomData<V>,
}

impl<V> Clone for MaybeZero<V> {
    fn clone(&self) -> Self {
        Self { label: self.label, marker: PhantomData }
    }
}

impl<V> std::fmt::Debug for MaybeZero<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.label)
    }
}

impl<V> PartialEq for MaybeZero<V> {
    fn eq(&self, other: &Self) -> bool {
        self.label == other.label
    }
}

impl<V> Eq for MaybeZero<V> {}

/// Stand-in for `ryft_core::TransposableOperation`.
trait TransposableOperation<V: Value, O: Operation<V::Type>>: Operation<V::Type> {
    fn transpose(
        &self,
        context: &mut TracingContext<V, O>,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError>;
}

/// Stand-in for `ryft_core::Program`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Program<V: Value, O: Operation<V::Type>, Input, Output> {
    label: &'static str,
    constant: Option<V>,
    operation: Option<O>,
    marker: PhantomData<(V, O, Input, Output)>,
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

impl<Constant, O, Input, Output> Program<Constant, O, Input, Output>
where
    Constant: Value,
    O: Operation<Constant::Type>,
{
    fn interpret_with<V, LiftConstantFn, InterpretInstructionFn>(
        &self,
        mut input: Vec<V>,
        mut lift_constant: LiftConstantFn,
        mut interpret_instruction: InterpretInstructionFn,
    ) -> Result<Vec<V>, ProgramError>
    where
        V: Value<Type = Constant::Type>,
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

impl<T, V, O, Input, Output> Program<V, O, Input, Output>
where
    T: DifferentiableType,
    V: Value<Type = T>,
    O: TransposableOperation<V, O> + From<ZeroOperation<T>> + From<AddOperation>,
{
    fn transpose_with_respect_to(
        &self,
        input_indices: &[usize],
    ) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError> {
        let _ = input_indices;
        Ok(Program { label: "program_transpose_with_respect_to", constant: None, operation: None, marker: PhantomData })
    }
}

/// Stand-in for `ryft_core::InterpretableProgramOperation`.
trait InterpretableProgramOperation<V: Value, C, Constant: Value<Type = V::Type> = V>: Operation<V::Type> + Sized {
    fn interpret_program(
        context: &C,
        program: &Program<Constant, Self, Vec<Constant>, Vec<Constant>>,
        input: Vec<V>,
    ) -> Result<Vec<V>, ProgramError>;
}

/// Stand-in for `ryft_core::TransposableProgramOperation`.
trait TransposableProgramOperation<V: Value>: Operation<V::Type> + Sized
where
    V::Type: DifferentiableType,
{
    fn transpose_program(
        program: &Program<V, Self, Vec<V>, Vec<V>>,
        input_linearity: &[bool],
    ) -> Result<Program<V, Self, Vec<V>, Vec<V>>, ProgramError>;
}

/// Stand-in for `ryft_core::StagingContext`. Mirrors the real trait's `Meta` associated type, which the generated
/// batching dispatchers project when naming the staged flowing tracer type.
trait StagingContext: Context {
    type Meta;
}

impl<V: Value, O: Operation<V::Type>> StagingContext for TestContext<V, O> {
    type Meta = ();
}

/// Stand-in for `ryft_core::JvpTracer`. Mirrors only what the generated forward-mode dispatcher references: the
/// generated `jvp` signature names the type, so a label field suffices to observe payload dispatch.
struct JvpTracer<C: Context> {
    label: &'static str,
    marker: PhantomData<C>,
}

impl<C: StagingContext> std::fmt::Debug for JvpTracer<C> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("JvpTracer").field("label", &self.label).finish()
    }
}

/// Stand-in for `ryft_core::DifferentiableOperation`.
trait DifferentiableOperation<C: Context>: Operation<C::Type> {
    fn jvp(&self, context: &C, inputs: &[JvpTracer<C>]) -> Result<Vec<JvpTracer<C>>, ProgramError>;
}

/// Stand-in for `ryft_core::Linearization`.
struct Linearization<V: Value, O> {
    label: &'static str,
    marker: PhantomData<(V, O)>,
}

/// Stand-in for `ryft_core::DifferentiableProgramOperation`, mirroring the two fixed-body methods the generated
/// witness implements.
trait DifferentiableProgramOperation<V: Value, O>: Operation<V::Type> + Sized {
    fn linearize_program(program: &Program<V, Self, Vec<V>, Vec<V>>) -> Result<Linearization<V, Self>, ProgramError>;

    fn jvp_program(
        program: &Program<V, Self, Vec<V>, Vec<V>>,
    ) -> Result<Program<V, Self, Vec<V>, Vec<V>>, ProgramError>;
}

impl<T, V, O, Input, Output> Program<V, O, Input, Output>
where
    T: Type,
    V: Value<Type = T> + SpecialTransposableValue,
    O: Clone + Operation<T> + From<ZeroOperation<T>>,
{
    /// Stand-in for `ryft_core::Program::linearize`. The `SpecialTransposableValue` bound on the value type stands
    /// in for the extra value leaves the real linearization needs, verifying that the generated witness transports
    /// the `#[ryft(bounds(differentiation(...)))]` list to this body check.
    fn linearize(&self) -> Result<Linearization<V, O>, ProgramError> {
        Ok(Linearization { label: "program_linearize", marker: PhantomData })
    }

    /// Stand-in for `ryft_core::Program::jvp_program`.
    fn jvp_program(&self) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError> {
        Ok(Program { label: "program_jvp_program", constant: None, operation: None, marker: PhantomData })
    }
}

/// Stand-in for the `ryft_core::partial` module, mirroring the shapes the `Operation` derive emits against for the
/// generated `PartiallyEvaluatableOperation` implementation.
mod partial {
    use super::{Context, PhantomData, ProgramError, Value};

    /// Stand-in for `ryft_core::partial::PartialValue`.
    pub(crate) enum PartialValue<V: Value> {
        Known(V),
        Unknown(V::Type),
    }

    /// Stand-in for `ryft_core::partial::PartialEvaluationValue`.
    pub(crate) struct PartialEvaluationValue<V: Value> {
        marker: PhantomData<V>,
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
            inputs: &[PartialEvaluationValue<C::Value>],
        ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
            let _ = (operation.into(), inputs);
            Ok(Vec::new())
        }
    }

    /// Stand-in for `ryft_core::partial::PartiallyEvaluatableOperation`.
    pub(crate) trait PartiallyEvaluatableOperation<C: Context>: Clone + Into<C::Operation> {
        fn partially_evaluate(
            &self,
            evaluator: &mut PartialEvaluator<C>,
            inputs: &[PartialEvaluationValue<C::Value>],
        ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError>
        where
            Self: Clone + Into<C::Operation>,
        {
            let _ = (evaluator.context(), inputs);
            Ok(Vec::new())
        }
    }
}

fn transposed<T: Type, V: Value<Type = T>, O: Operation<T>>(
    label: &'static str,
) -> MaybeZero<Tracer<TracingContext<V, O>>> {
    MaybeZero { label, marker: PhantomData }
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

impl Value for Factor {
    type Type = ArrayType;
}

/// Scalar-universe counterpart of [`Factor`]. A value type pins exactly one type descriptor through the associated
/// `Type`, so the scalar test enums flow this type instead of reusing [`Factor`] across universes.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ScalarFactor(i64);

impl Value for ScalarFactor {
    type Type = DataType;
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TranspositionFactor(i64);

impl Value for TranspositionFactor {
    type Type = ArrayType;
}

impl BooleanLike for Factor {}

impl BooleanLike for ScalarFactor {}

impl Slice for Factor {}

impl UpdateSlice for Factor {}

impl Reshape for Factor {}

trait SpecialTransposableValue {}

impl SpecialTransposableValue for Factor {}

impl SpecialTransposableValue for ScalarFactor {}

impl SpecialTransposableValue for TranspositionFactor {}

/// Extra value bound a recursive payload's partial-evaluation rule requires, used to exercise
/// `#[ryft(bounds(partial_evaluation(...)))]`.
trait SpecialPartiallyEvaluatableValue {}

impl SpecialPartiallyEvaluatableValue for Factor {}

impl SpecialPartiallyEvaluatableValue for ScalarFactor {}

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

impl<T: Clone + Type, V: Value<Type = T>, C> InterpretableOperation<V, C> for ZeroOperation<T> {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<T: Clone + Type, C: Context<Type = T>> partial::PartiallyEvaluatableOperation<C> for ZeroOperation<T> where
    C::Operation: From<ZeroOperation<T>>
{
}

impl<T: Clone + Type, V: Value<Type = T>, O: Operation<T>> TransposableOperation<V, O> for ZeroOperation<T> {
    fn transpose(
        &self,
        _context: &mut TracingContext<V, O>,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
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

impl<V: Value<Type = DataType>, C> InterpretableOperation<V, C> for AddOperation {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<C: Context<Type = DataType>> partial::PartiallyEvaluatableOperation<C> for AddOperation where
    C::Operation: From<AddOperation>
{
}

impl<V: Value<Type = DataType>, O: Operation<DataType>> TransposableOperation<V, O> for AddOperation {
    fn transpose(
        &self,
        _context: &mut TracingContext<V, O>,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
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

impl<V: Value<Type = DataType>, C> InterpretableOperation<V, C> for PrintOperation {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<C: Context<Type = DataType>> partial::PartiallyEvaluatableOperation<C> for PrintOperation where
    C::Operation: From<PrintOperation>
{
}

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

impl<T: Clone + Type, V: Value<Type = T>, F, C> InterpretableOperation<V, C> for FactorOperation<T, F> {
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

impl<T: Clone + Type, V: Value<Type = T>, O: Operation<T>, F> TransposableOperation<V, O> for FactorOperation<T, F> {
    fn transpose(
        &self,
        _context: &mut TracingContext<V, O>,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
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

impl<T: Clone + Type, V: Value<Type = T>, Constant, C> InterpretableOperation<V, C> for ConstantOperation<T, Constant> {
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

impl<T: Clone + Type, V: Value<Type = T>, O: Operation<T>, F> TransposableOperation<V, O> for ConstantOperation<T, F> {
    fn transpose(
        &self,
        _context: &mut TracingContext<V, O>,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
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

impl<T: Clone + Type, V: Value<Type = T>, Constant, C> InterpretableOperation<V, C>
    for CustomJvpOperation<T, Constant>
{
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
enum ScalarOperation<V: Value<Type = DataType>> {
    Zero(ZeroOperation<DataType>),
    Add(AddOperation),
    Print(PrintOperation),
    Factor(FactorOperation<DataType, V>),
    CustomJvp(Box<CustomJvpOperation<DataType, V>>),
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation, ryft::TransposableOperation)]
#[ryft(crate = "crate")]
enum LinearScalarOperation<V: Value<Type = DataType>, C: Value<Type = DataType> = V> {
    Zero(ZeroOperation<DataType>),
    Constant(ConstantOperation<DataType, V>),
    Add(AddOperation),
    Factor(FactorOperation<DataType, C>),
}

#[test]
fn test_scalar_operation() {
    let zero = ScalarOperation::<ScalarFactor>::from(ZeroOperation { r#type: DataType });
    let add = ScalarOperation::<ScalarFactor>::from(AddOperation);
    let factor =
        ScalarOperation::<ScalarFactor>::from(FactorOperation { factor: ScalarFactor(7), marker: PhantomData });
    let custom_jvp = ScalarOperation::<ScalarFactor>::from(CustomJvpOperation { tag: "tag", marker: PhantomData });

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
        <&FactorOperation<DataType, ScalarFactor>>::try_from(&factor),
        Ok(&FactorOperation { factor: ScalarFactor(7), marker: PhantomData }),
    );
    assert_eq!(
        <&CustomJvpOperation<DataType, ScalarFactor>>::try_from(&custom_jvp),
        Ok(&CustomJvpOperation { tag: "tag", marker: PhantomData }),
    );
    assert_eq!(<&AddOperation>::try_from(&zero), Err(()));
}

#[test]
fn test_operation_generates_effects_forwarding() {
    let add = ScalarOperation::<ScalarFactor>::from(AddOperation);
    let print = ScalarOperation::<ScalarFactor>::from(PrintOperation);

    assert_eq!(add.effects(), Effects::Pure);
    assert_eq!(print.effects(), Effects::Ordered);
}

#[test]
fn test_operation_generates_interpretation_forwarding() {
    let context = TestContext::<ScalarFactor> { marker: PhantomData };
    let operation = ScalarOperation::<ScalarFactor>::from(AddOperation);

    assert_eq!(
        operation.interpret(&context, &[ScalarFactor(1), ScalarFactor(2)]),
        Ok(vec![ScalarFactor(1), ScalarFactor(2)]),
    );
}

#[test]
fn test_operation_generates_captured_program_interpretation_witness() {
    type Operation = ScalarOperation<ScalarFactor>;

    let context = TestContext::<ScalarFactor> { marker: PhantomData };
    let program = Program::<ScalarFactor, Operation, Vec<ScalarFactor>, Vec<ScalarFactor>> {
        label: "scalar",
        constant: Some(ScalarFactor(3)),
        operation: Some(Operation::from(AddOperation)),
        marker: PhantomData,
    };
    let outputs = <Operation as InterpretableProgramOperation<
        ScalarFactor,
        TestContext<ScalarFactor>,
        ScalarFactor,
    >>::interpret_program(&context, &program, vec![ScalarFactor(1)])
    .unwrap();

    assert_eq!(outputs, vec![ScalarFactor(1), ScalarFactor(3)]);
}

#[test]
fn test_operation_generates_direct_program_interpretation_witness() {
    type Operation = LinearScalarOperation<ScalarFactor>;

    let context = TestContext::<ScalarFactor> { marker: PhantomData };
    let operation = Operation::from(FactorOperation { factor: ScalarFactor(5), marker: PhantomData });

    assert_eq!(operation.interpret(&context, &[ScalarFactor(8)]), Ok(vec![ScalarFactor(8)]));

    let program = Program::<ScalarFactor, Operation, Vec<ScalarFactor>, Vec<ScalarFactor>> {
        label: "linear",
        constant: Some(ScalarFactor(13)),
        operation: Some(Operation::from(AddOperation)),
        marker: PhantomData,
    };
    let outputs =
        <Operation as InterpretableProgramOperation<ScalarFactor, TestContext<ScalarFactor>>>::interpret_program(
            &context,
            &program,
            vec![ScalarFactor(8)],
        )
        .unwrap();

    assert_eq!(outputs, vec![ScalarFactor(8), ScalarFactor(13)]);
}

#[test]
fn test_operation_generates_partial_evaluation_witness() {
    fn assert_partially_evaluatable<C: Context, O: partial::PartiallyEvaluatableOperation<C>>() {}

    // The derive now forwards partial evaluation for every variant, so each enum must satisfy the per-operation
    // partial-evaluation trait at any known-side context pinned to its program-constant value type and to itself as
    // the residual operation family. This covers leaf payloads, the generic `Backend` payload, and the boxed
    // nested-program payloads.
    assert_partially_evaluatable::<
        TestContext<ScalarFactor, ScalarOperation<ScalarFactor>>,
        ScalarOperation<ScalarFactor>,
    >();
    assert_partially_evaluatable::<
        TestContext<ScalarFactor, LinearScalarOperation<ScalarFactor>>,
        LinearScalarOperation<ScalarFactor>,
    >();
    assert_partially_evaluatable::<TestContext<Factor, ArrayOperation<Factor>>, ArrayOperation<Factor>>();
    assert_partially_evaluatable::<
        TestContext<Factor, LinearArrayOperation<Factor, Factor>>,
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
        TestContext<Factor, PartialEvaluationBoundOperation<Factor>>,
        PartialEvaluationBoundOperation<Factor>,
    >();

    let context = TestContext::<Factor, PartialEvaluationBoundOperation<Factor>> { marker: PhantomData };
    let mut evaluator = partial::PartialEvaluator::new(context);
    let operation = PartialEvaluationBoundOperation::<Factor>::from(ZeroOperation { r#type: ArrayType });
    let evaluation = operation.partially_evaluate(&mut evaluator, &[]).unwrap();
    assert!(evaluation.is_empty());
}

#[test]
fn test_transposable_operation_infers_value_type() {
    type Linear = LinearScalarOperation<ScalarFactor>;

    let mut context = TracingContext::<ScalarFactor, Linear> { marker: PhantomData };
    let add = Linear::from(AddOperation);

    assert_eq!(
        add.transpose(&mut context, &[PartialValue::Unknown(DataType)], &[]).unwrap(),
        vec![transposed::<DataType, ScalarFactor, Linear>("add")],
    );
}

#[derive(Clone, Debug, ryft::Operation)]
enum DefaultPathOperation<V: ryft::Value<Type = ryft::DataType>> {
    Zero(ryft::ZeroOperation<ryft::DataType>),
    Constant(ryft::ConstantOperation<V>),
}

#[derive(Clone, Debug, ryft::Operation, ryft::TransposableOperation)]
enum DefaultPathLinearOperation<V: ryft::Value<Type = ryft::DataType>> {
    Zero(ryft::ZeroOperation<ryft::DataType>),
    Constant(ryft::ConstantOperation<V>),
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

impl<V: Value<Type = ArrayType>, C> InterpretableOperation<V, C> for DotOperation {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<C: Context<Type = ArrayType>> partial::PartiallyEvaluatableOperation<C> for DotOperation where
    C::Operation: From<DotOperation>
{
}

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

impl<V: Value<Type = ArrayType>, C> InterpretableOperation<V, C> for BackendPayload {
    fn interpret(&self, _context: &C, _inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        match *self {}
    }
}

impl<C: Context<Type = ArrayType>> partial::PartiallyEvaluatableOperation<C> for BackendPayload where
    C::Operation: From<BackendPayload>
{
}

impl<V: Value<Type = ArrayType>, O: Operation<ArrayType>> TransposableOperation<V, O> for BackendPayload {
    fn transpose(
        &self,
        _context: &mut TracingContext<V, O>,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
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

impl<V: Value<Type = ArrayType>, C> InterpretableOperation<V, C> for SpecialOperation {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<C: Context<Type = ArrayType>> partial::PartiallyEvaluatableOperation<C> for SpecialOperation where
    C::Operation: From<SpecialOperation>
{
}

impl<V, O> TransposableOperation<V, O> for SpecialOperation
where
    V: Value<Type = ArrayType> + SpecialTransposableValue,
    O: Operation<ArrayType>,
{
    fn transpose(
        &self,
        _context: &mut TracingContext<V, O>,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
        Ok(vec![transposed("special")])
    }
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation, ryft::TransposableOperation)]
#[ryft(crate = "crate")]
enum SpecialLinearOperation<V: Value<Type = ArrayType>> {
    Special(SpecialOperation),
    Constant(ConstantOperation<ArrayType, V>),
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation)]
#[ryft(crate = "crate")]
enum InferredArrayOperation<V: Value<Type = ArrayType>, C: Value<Type = ArrayType> = V> {
    Zero(ZeroOperation<ArrayType>),
    Constant(ConstantOperation<ArrayType, V>),
    Factor(FactorOperation<ArrayType, C>),
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation)]
#[ryft(crate = "crate")]
#[ryft(bounds(interpretation(Slice + UpdateSlice + Reshape)))]
enum InterpretationBoundOperation<C: Value<Type = ArrayType> + BooleanLike> {
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

    let context = TestContext::<Factor> { marker: PhantomData };
    let operation = Operation::from(ZeroOperation { r#type: ArrayType });

    assert_eq!(operation.interpret(&context, &[Factor(1)]), Ok(vec![Factor(1)]));

    let program = Program::<Factor, Operation, Vec<Factor>, Vec<Factor>> {
        label: "array",
        constant: Some(Factor(3)),
        operation: Some(Operation::from(ConstantOperation { value: Factor(5), marker: PhantomData })),
        marker: PhantomData,
    };
    let outputs = <Operation as InterpretableProgramOperation<Factor, TestContext<Factor>, Factor>>::interpret_program(
        &context,
        &program,
        vec![Factor(1)],
    )
    .unwrap();

    assert_eq!(outputs, vec![Factor(1), Factor(3)]);
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation)]
#[ryft(crate = "crate")]
enum ArrayOperation<V: Value<Type = ArrayType>, Backend = BackendPayload> {
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

impl<T: Clone + Type, V: Value<Type = T>, W, O, C> InterpretableOperation<V, C> for WhileOperation<T, W, O> {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<T: Clone + Type, W: Clone, O: Clone + Operation<T>, C: Context<Type = T>> partial::PartiallyEvaluatableOperation<C>
    for WhileOperation<T, W, O>
where
    C::Operation: From<WhileOperation<T, W, O>>,
{
}

impl<T: Clone + Type, V: Value<Type = T>, O: Operation<T>, W, P> TransposableOperation<V, O>
    for WhileOperation<T, W, P>
{
    fn transpose(
        &self,
        _context: &mut TracingContext<V, O>,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
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

impl<V: Value<Type = ArrayType>, W, O, C> InterpretableOperation<V, C> for RecursiveOperation<W, O> {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<W: Clone, O: Clone + Operation<ArrayType>, C: Context<Type = ArrayType>> partial::PartiallyEvaluatableOperation<C>
    for RecursiveOperation<W, O>
where
    C::Operation: From<RecursiveOperation<W, O>>,
{
}

impl<StoredValue, TranspositionValue, O> TransposableOperation<TranspositionValue, O>
    for RecursiveOperation<StoredValue, O>
where
    TranspositionValue: Value<Type = ArrayType>,
    O: RecursiveOperationTransposable<TranspositionValue>,
{
    fn transpose(
        &self,
        _context: &mut TracingContext<TranspositionValue, O>,
        _inputs: &[PartialValue<Tracer<TracingContext<TranspositionValue, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<TranspositionValue, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<TranspositionValue, O>>>>, ProgramError> {
        Ok(vec![transposed("recursive")])
    }
}

trait RecursiveOperationTransposable<V: Value<Type = ArrayType>>: Operation<ArrayType> {}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation, ryft::TransposableOperation)]
#[ryft(crate = "crate")]
enum RecursiveLinearOperation<V: Value<Type = ArrayType>> {
    Zero(ZeroOperation<ArrayType>),
    Recursive(RecursiveOperation<V, Self>),
}

impl<StoredValue: Value<Type = ArrayType>, TranspositionValue: Value<Type = ArrayType>>
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

impl<V: Value<Type = DataType>, W, O, C> InterpretableOperation<V, C> for ProgramRecursiveOperation<W, O> {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<W: Clone, O: Clone + Operation<DataType>, C: Context<Type = DataType>> partial::PartiallyEvaluatableOperation<C>
    for ProgramRecursiveOperation<W, O>
where
    C::Operation: From<ProgramRecursiveOperation<W, O>>,
{
}

impl<StoredValue, TranspositionValue, O> TransposableOperation<TranspositionValue, O>
    for ProgramRecursiveOperation<StoredValue, O>
where
    TranspositionValue: Value<Type = DataType> + SpecialTransposableValue,
    O: TransposableProgramOperation<TranspositionValue>,
{
    fn transpose(
        &self,
        _context: &mut TracingContext<TranspositionValue, O>,
        _inputs: &[PartialValue<Tracer<TracingContext<TranspositionValue, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<TranspositionValue, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<TranspositionValue, O>>>>, ProgramError> {
        Ok(vec![transposed("program_recursive")])
    }
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation, ryft::TransposableOperation)]
#[ryft(crate = "crate")]
enum RecursiveProgramLinearOperation<V: Value<Type = DataType> + SpecialTransposableValue> {
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

impl<V: Value<Type = ArrayType>, W, O, C> InterpretableOperation<V, C> for PartialEvaluationRecursiveOperation<W, O> {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<W: Clone, O: Clone + Operation<ArrayType>, C: Context<Type = ArrayType>> partial::PartiallyEvaluatableOperation<C>
    for PartialEvaluationRecursiveOperation<W, O>
where
    C::Value: SpecialPartiallyEvaluatableValue,
    C::Operation: From<PartialEvaluationRecursiveOperation<W, O>>,
{
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation)]
#[ryft(crate = "crate")]
#[ryft(bounds(partial_evaluation(SpecialPartiallyEvaluatableValue)))]
enum PartialEvaluationBoundOperation<V: Value<Type = ArrayType>> {
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

impl SpecialCombine for ScalarFactor {
    type Output = ScalarFactor;
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

impl<V: Value<Type = DataType>, C> InterpretableOperation<V, C> for NonDifferentiableOperation {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<C: Context<Type = DataType>> partial::PartiallyEvaluatableOperation<C> for NonDifferentiableOperation where
    C::Operation: From<NonDifferentiableOperation>
{
}

impl<V: Value<Type = DataType>, O: Operation<DataType>> TransposableOperation<V, O> for NonDifferentiableOperation {
    fn transpose(
        &self,
        _context: &mut TracingContext<V, O>,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
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
#[ryft(bounds(differentiation(SpecialTransposableValue)))]
enum DifferentiableScalarOperation<V: Value<Type = DataType>> {
    Zero(ZeroOperation<DataType>),
    Add(AddOperation),
    Factor(FactorOperation<DataType, V>),
    NonDifferentiable(NonDifferentiableOperation),
}

#[test]
fn test_differentiable_operation_dispatches_jvp_to_payloads() {
    type Operation = DifferentiableScalarOperation<ScalarFactor>;

    let context = TestContext::<ScalarFactor, Operation> { marker: PhantomData };

    let zero = Operation::from(ZeroOperation { r#type: DataType });
    let outputs = zero.jvp(&context, &[]).unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].label, "zero");

    let add = Operation::from(AddOperation);
    let outputs = add.jvp(&context, &[]).unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].label, "add");

    // The `ScalarFactor` rule requires `SpecialCombine<Output = C::Value>` on the flowing value, transported to this use
    // site by the generated per-variant `DifferentiableOperation` predicate.
    let factor = Operation::from(FactorOperation { factor: ScalarFactor(3), marker: PhantomData });
    let outputs = factor.jvp(&context, &[]).unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].label, "factor");
}

#[test]
fn test_differentiable_operation_delegates_unsupported_payloads() {
    type Operation = DifferentiableScalarOperation<ScalarFactor>;

    // The generated dispatcher delegates uniformly, so the unsupported payload's own erroring rule reports the
    // failure rather than an enum-level dispatch arm.
    let context = TestContext::<ScalarFactor, Operation> { marker: PhantomData };
    let operation = Operation::from(NonDifferentiableOperation);
    assert_eq!(operation.jvp(&context, &[]).unwrap_err(), ProgramError);
}

#[test]
fn test_differentiable_operation_generates_program_differentiation_witness() {
    type Operation = DifferentiableScalarOperation<ScalarFactor>;

    // The witness's fixed bodies forward to the program's own `linearize` / `jvp_program`; the stand-in `linearize`
    // requires `SpecialTransposableValue` on the value type, supplied to the generated impl by
    // `#[ryft(bounds(differentiation(...)))]`.
    let program = Program::<ScalarFactor, Operation, Vec<ScalarFactor>, Vec<ScalarFactor>> {
        label: "differentiable",
        constant: None,
        operation: None,
        marker: PhantomData,
    };
    let linearization =
        <Operation as DifferentiableProgramOperation<ScalarFactor, Operation>>::linearize_program(&program).unwrap();
    assert_eq!(linearization.label, "program_linearize");

    let jvp_program =
        <Operation as DifferentiableProgramOperation<ScalarFactor, Operation>>::jvp_program(&program).unwrap();
    assert_eq!(jvp_program.label, "program_jvp_program");
}

#[test]
fn test_differentiable_operation_generates_transposition_dispatchers() {
    type Operation = DifferentiableScalarOperation<ScalarFactor>;

    // The transposition dispatchers come from the separate `TransposableOperation` derive: `DifferentiableOperation`
    // only adds forward-mode support, and deriving both enables reverse mode too.
    let mut context = TracingContext::<ScalarFactor, Operation> { marker: PhantomData };
    let add = Operation::from(AddOperation);
    assert_eq!(
        add.transpose(&mut context, &[PartialValue::Unknown(DataType)], &[]).unwrap(),
        vec![transposed::<DataType, ScalarFactor, Operation>("add")],
    );

    let program = Program::<ScalarFactor, Operation, Vec<ScalarFactor>, Vec<ScalarFactor>> {
        label: "differentiable",
        constant: None,
        operation: None,
        marker: PhantomData,
    };
    let transposed_program =
        <Operation as TransposableProgramOperation<ScalarFactor>>::transpose_program(&program, &[true]).unwrap();
    assert_eq!(transposed_program.label, "program_transpose_with_respect_to");
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

impl<V: Value<Type = ArrayType>, O: Operation<ArrayType>, C> InterpretableOperation<V, C> for RecomputeOperation<O> {
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

impl<V: Value<Type = ArrayType>, O: Operation<ArrayType>, P: Operation<ArrayType>> TransposableOperation<V, O>
    for RecomputeOperation<P>
{
    fn transpose(
        &self,
        _context: &mut TracingContext<V, O>,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
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

impl<T: Clone + Type, V: Value<Type = T>, Constant, O, F, C> InterpretableOperation<V, C>
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

impl<T: Clone + Type, V: Value<Type = T>, O: Operation<T>, C, P, F> TransposableOperation<V, O>
    for CustomVjpCallOperation<T, C, P, F>
{
    fn transpose(
        &self,
        _context: &mut TracingContext<V, O>,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
        Ok(vec![transposed("custom_vjp_call")])
    }
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation, ryft::TransposableOperation)]
#[ryft(crate = "crate")]
enum LinearArrayOperation<
    V: Value<Type = ArrayType>,
    C: Value<Type = ArrayType>,
    Backend = BackendPayload,
    F: Value<Type = ArrayType> = V,
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

    let mut context = TracingContext::<Factor, Linear> { marker: PhantomData };

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
    type Linear = LinearScalarOperation<ScalarFactor>;

    let program = Program::<ScalarFactor, Linear, Vec<ScalarFactor>, Vec<ScalarFactor>> {
        label: "linear",
        constant: None,
        operation: None,
        marker: PhantomData,
    };
    let transposed =
        <Linear as TransposableProgramOperation<ScalarFactor>>::transpose_program(&program, &[true]).unwrap();

    assert_eq!(transposed.label, "program_transpose_with_respect_to");
}

#[test]
fn test_transposable_operation_generates_concrete_payload_bounds() {
    type Linear = SpecialLinearOperation<Factor>;

    let mut context = TracingContext::<TranspositionFactor, Linear> { marker: PhantomData };
    let operation = Linear::from(SpecialOperation);

    assert_eq!(
        operation.transpose(&mut context, &[PartialValue::Unknown(ArrayType)], &[]).unwrap(),
        vec![transposed::<ArrayType, TranspositionFactor, Linear>("special")],
    );
}

#[test]
fn test_transposable_operation_supports_recursive_payload_helpers() {
    type Linear = RecursiveLinearOperation<Factor>;

    let mut context = TracingContext::<Factor, Linear> { marker: PhantomData };
    let operation = Linear::from(RecursiveOperation::<Factor, Linear> { marker: PhantomData });

    assert_eq!(
        operation.transpose(&mut context, &[PartialValue::Unknown(ArrayType)], &[]).unwrap(),
        vec![transposed::<ArrayType, Factor, Linear>("recursive")],
    );
}

#[test]
fn test_transposable_operation_inherits_enum_bounds_for_recursive_program_witness() {
    type Linear = RecursiveProgramLinearOperation<ScalarFactor>;

    let mut context = TracingContext::<ScalarFactor, Linear> { marker: PhantomData };
    let operation = Linear::from(ProgramRecursiveOperation::<ScalarFactor, Linear> { marker: PhantomData });

    assert_eq!(
        operation.transpose(&mut context, &[PartialValue::Unknown(DataType)], &[]).unwrap(),
        vec![transposed::<DataType, ScalarFactor, Linear>("program_recursive")],
    );

    let program = Program::<ScalarFactor, Linear, Vec<ScalarFactor>, Vec<ScalarFactor>> {
        label: "linear",
        constant: None,
        operation: None,
        marker: PhantomData,
    };
    let transposed =
        <Linear as TransposableProgramOperation<ScalarFactor>>::transpose_program(&program, &[true]).unwrap();

    assert_eq!(transposed.label, "program_transpose_with_respect_to");
}

/// Stand-in for `ryft_core::ArrayBatch`. A label suffices to observe payload dispatch.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ArrayBatch<V> {
    label: &'static str,
    marker: PhantomData<V>,
}

impl<V> ArrayBatch<V> {
    fn labeled(label: &'static str) -> Self {
        Self { label, marker: PhantomData }
    }
}

/// Stand-in for `ryft_core::BatchableOperation`.
trait BatchableOperation<V, C>: Operation<ArrayType> {
    fn batch(&self, context: &C, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, BatchingError>;
}

/// Stand-in for `ryft_core::EagerContext`.
struct EagerContext<V: Value, O: Operation<V::Type>> {
    marker: PhantomData<(V, O)>,
}

impl<V: Value, O: Operation<V::Type>> Zero<V> for EagerContext<V, O> {}

/// Stand-in for `ryft_core::BatchingContext`.
struct BatchingContext<C> {
    parent: C,
}

impl<C> BatchingContext<C> {
    fn parent(&self) -> &C {
        &self.parent
    }
}

/// Stand-in for `ryft_core::BatchAxis`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct BatchAxis(Option<usize>);

/// Stand-in for `ryft_core::ProgramBatchingOutputAxesPolicy`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ProgramBatchingOutputAxesPolicy {
    Natural,
}

/// Stand-in for `ryft_core::BatchableProgramOperation`.
trait BatchableProgramOperation<V: Value<Type = ArrayType>>: Operation<ArrayType> + Sized {
    fn batch_program(
        program: &Program<V, Self, Vec<V>, Vec<V>>,
        axis_size: usize,
        input_batch_axes: &[BatchAxis],
        output_axes_policy: ProgramBatchingOutputAxesPolicy,
    ) -> Result<(Program<V, Self, Vec<V>, Vec<V>>, Vec<BatchAxis>), BatchingError>;
}

/// Stand-in for `ryft_core::batch_program`.
fn batch_program<V, O>(
    program: &Program<V, O, Vec<V>, Vec<V>>,
    _axis_size: usize,
    input_batch_axes: &[BatchAxis],
    _output_axes_policy: ProgramBatchingOutputAxesPolicy,
) -> Result<(Program<V, O, Vec<V>, Vec<V>>, Vec<BatchAxis>), BatchingError>
where
    V: Value<Type = ArrayType>,
    O: Clone + Operation<ArrayType>,
{
    Ok((program.clone(), input_batch_axes.to_vec()))
}

/// Stand-in value capability required by one payload's batching rule and by the recursive payload's leaf bounds,
/// verifying both per-variant predicate transport and the `#[ryft(bounds(batching(...)))]` leaf injection.
trait SpecialBatchValue {}

impl SpecialBatchValue for Factor {}

impl<C, Meta> SpecialBatchValue for Tracer<C, Meta> {}

impl<V, C> BatchableOperation<V, C> for ZeroOperation<ArrayType> {
    fn batch(&self, _context: &C, _inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, BatchingError> {
        Ok(vec![ArrayBatch::labeled("zero")])
    }
}

/// Batching rule requiring a value capability that the generated per-variant predicate transports to the owning
/// enum's use sites without the enum spelling it.
impl<V: SpecialBatchValue, C> BatchableOperation<V, C> for DotOperation {
    fn batch(&self, _context: &C, _inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, BatchingError> {
        Ok(vec![ArrayBatch::labeled("dot")])
    }
}

/// Stand-in for a named-axis collective: a non-recursive payload whose staged rule is keyed at the batching context
/// (for its axis metadata), exercised through the `#[ryft(batching(active))]` variant marker.
#[derive(Clone, Debug, PartialEq, Eq)]
struct CollectiveLikeOperation;

impl Operation<ArrayType> for CollectiveLikeOperation {
    fn name(&self) -> &'static str {
        "collective_like"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<V: Value<Type = ArrayType>, C> InterpretableOperation<V, C> for CollectiveLikeOperation {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<C: Context<Type = ArrayType>> partial::PartiallyEvaluatableOperation<C> for CollectiveLikeOperation where
    C::Operation: From<CollectiveLikeOperation>
{
}

impl<C: Context<Type = ArrayType>> BatchableOperation<<C as Domain>::Value, BatchingContext<C>>
    for CollectiveLikeOperation
{
    fn batch(
        &self,
        _context: &BatchingContext<C>,
        _inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError> {
        Ok(vec![ArrayBatch::labeled("collective_like_staged")])
    }
}

impl<V: Value<Type = ArrayType>, O: Operation<ArrayType>> BatchableOperation<V, EagerContext<V, O>>
    for CollectiveLikeOperation
{
    fn batch(
        &self,
        _context: &EagerContext<V, O>,
        _inputs: &[ArrayBatch<V>],
    ) -> Result<Vec<ArrayBatch<V>>, BatchingError> {
        Ok(vec![ArrayBatch::labeled("collective_like_eager")])
    }
}

/// Stand-in recursive higher-order payload whose eager and staged batching rules mirror the leaf obligations the
/// real control-flow rules carry: the staged rule requires an operation-shaped `From` conversion (discharged
/// structurally from the closed enum, like the real rules' staged structural operations) and the
/// `BatchableProgramOperation` fixed-point witness, while the eager rule requires the eager leaf capabilities and
/// the context's `Zero`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct BatchRecursiveOperation<V, O> {
    marker: PhantomData<(V, O)>,
}

impl<V, O> Operation<ArrayType> for BatchRecursiveOperation<V, O> {
    fn name(&self) -> &'static str {
        "batch_recursive"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<V: Value<Type = ArrayType>, W, O, C> InterpretableOperation<V, C> for BatchRecursiveOperation<W, O> {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<W: Clone, O: Clone + Operation<ArrayType>, C: Context<Type = ArrayType>> partial::PartiallyEvaluatableOperation<C>
    for BatchRecursiveOperation<W, O>
where
    C::Operation: From<BatchRecursiveOperation<W, O>>,
{
}

impl<C> BatchableOperation<<C as Domain>::Value, BatchingContext<C>>
    for BatchRecursiveOperation<C::Constant, C::Operation>
where
    C: Context<Type = ArrayType>,
    C::Operation: BatchableProgramOperation<C::Constant> + From<ZeroOperation<ArrayType>>,
{
    fn batch(
        &self,
        _context: &BatchingContext<C>,
        _inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError> {
        Ok(vec![ArrayBatch::labeled("batch_recursive_staged")])
    }
}

impl<V, O> BatchableOperation<V, EagerContext<V, O>> for BatchRecursiveOperation<V, O>
where
    V: Value<Type = ArrayType> + BooleanLike + SpecialBatchValue,
    O: Operation<ArrayType>,
    EagerContext<V, O>: Zero<V>,
{
    fn batch(
        &self,
        _context: &EagerContext<V, O>,
        _inputs: &[ArrayBatch<V>],
    ) -> Result<Vec<ArrayBatch<V>>, BatchingError> {
        Ok(vec![ArrayBatch::labeled("batch_recursive_eager")])
    }
}

#[derive(Clone, Debug, ryft::Operation, ryft::BatchableOperation)]
#[ryft(crate = "crate")]
#[ryft(bounds(batching(BooleanLike + SpecialBatchValue)))]
enum BatchableArrayOperation<V: Value<Type = ArrayType>> {
    Zero(ZeroOperation<ArrayType>),
    Dot(DotOperation),
    #[ryft(batching(active))]
    Collective(CollectiveLikeOperation),
    Recursive(Box<BatchRecursiveOperation<V, Self>>),
}

#[test]
fn test_batchable_operation_dispatches_eager_batching_to_payloads() {
    type Operation = BatchableArrayOperation<Factor>;

    let context = EagerContext::<Factor, Operation> { marker: PhantomData };

    let zero = Operation::from(ZeroOperation { r#type: ArrayType });
    assert_eq!(zero.batch(&context, &[]).unwrap(), vec![ArrayBatch::labeled("zero")]);

    // The `Dot` rule requires `SpecialBatchValue` on the flowing value, transported to this use site by the
    // generated per-variant `BatchableOperation` predicate.
    let dot = Operation::from(DotOperation);
    assert_eq!(dot.batch(&context, &[]).unwrap(), vec![ArrayBatch::labeled("dot")]);

    let collective = Operation::from(CollectiveLikeOperation);
    assert_eq!(collective.batch(&context, &[]).unwrap(), vec![ArrayBatch::labeled("collective_like_eager")]);

    let recursive = Operation::from(BatchRecursiveOperation::<Factor, Operation> { marker: PhantomData });
    assert_eq!(recursive.batch(&context, &[]).unwrap(), vec![ArrayBatch::labeled("batch_recursive_eager")]);
}

#[test]
fn test_batchable_operation_dispatches_staged_batching_to_payloads() {
    type Operation = BatchableArrayOperation<Factor>;
    type Staging = TestContext<Factor, Operation>;

    let context = BatchingContext::<Staging> { parent: TestContext { marker: PhantomData } };

    // Primitive arms dispatch at the parent context; the `active`-marked collective and the recursive payload dispatch
    // at the batching context itself. Every arm flows the parent context's own value (`<Staging as Domain>::Value`,
    // here `Factor`) rather than always a staged `Tracer`.
    let zero = Operation::from(ZeroOperation { r#type: ArrayType });
    let outputs: Vec<ArrayBatch<Factor>> = zero.batch(&context, &[]).unwrap();
    assert_eq!(outputs.iter().map(|output| output.label).collect::<Vec<_>>(), vec!["zero"]);

    let collective = Operation::from(CollectiveLikeOperation);
    let outputs: Vec<ArrayBatch<Factor>> = collective.batch(&context, &[]).unwrap();
    assert_eq!(outputs.iter().map(|output| output.label).collect::<Vec<_>>(), vec!["collective_like_staged"]);

    let recursive = Operation::from(BatchRecursiveOperation::<Factor, Operation> { marker: PhantomData });
    let outputs: Vec<ArrayBatch<Factor>> = recursive.batch(&context, &[]).unwrap();
    assert_eq!(outputs.iter().map(|output| output.label).collect::<Vec<_>>(), vec!["batch_recursive_staged"]);
}

#[test]
fn test_batchable_operation_generates_program_batching_witness() {
    type Operation = BatchableArrayOperation<Factor>;

    let program = Program::<Factor, Operation, Vec<Factor>, Vec<Factor>> {
        label: "batchable",
        constant: None,
        operation: None,
        marker: PhantomData,
    };
    let (batched, output_axes) = <Operation as BatchableProgramOperation<Factor>>::batch_program(
        &program,
        3,
        &[BatchAxis(Some(0))],
        ProgramBatchingOutputAxesPolicy::Natural,
    )
    .unwrap();

    assert_eq!(batched.label, "batchable");
    assert_eq!(output_axes, vec![BatchAxis(Some(0))]);
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
