use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::domains::Domain;
use crate::macros::check_builders;
use crate::operations::constants::{ConstantOperation, MaybeZeroOperation};
use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::{Parameterized, ParameterizedFamily};
use crate::programs::{AtomId, Program, ProgramBuilder, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};
use crate::types::{Type, Typed};

/// Active context that can *apply* an [`Operation`] to values, layered on top of the passive [`Domain`] substrate.
/// Where a [`Domain`] only describes the type, value, constant, and operation universe, a [`Context`] additionally
/// decides what *binding* a primitive means in this context and how to [`lift`](Context::lift) a staged constant into
/// a runtime value. There are two flavors:
///
/// - *Eager* contexts, whose flowing [`Domain::Value`] is a concrete value (equal to [`Domain::Constant`]).
///   [`Context::bind`] interprets the operation immediately and there is no [`ProgramBuilder`] involved anywhere.
///   Eager backends implement [`Context`] directly. An eager context is also where interpreters and program transforms
///   synthesize a type's additive or multiplicative identity from metadata alone, via `bind(ZeroOperation, &[])` or
///   `bind(OneOperation, &[])`).
/// - [**Staging**](StagingContext) contexts, whose flowing [`Domain::Value`] is a [`Tracer`] into an active
///   [`ProgramBuilder`]. [`Context::bind`] records the operation as a program instruction. Ordinary tracing appends
///   the operation to a program. Transform contexts such as batching or linearization intercept the same bind, update
///   transform-local state, and usually stage rewritten operations into a parent context.
pub trait Context: Domain + Clone {
    /// Lifts a staged [`Program`] constant into this [`Context`]'s runtime value representation. Most eager contexts
    /// use the same representation for [`Domain::Constant`] and [`Domain::Value`], and so this is just an identity
    /// function. Backends that use abstract, lifetime-free constants for compiled programs can either materialize
    /// a runtime value here when that is semantically valid, or return an error when an abstract constant cannot
    /// be interpreted as a concrete runtime value. [`StagingContext`]s lift constants by recording then as constant
    /// [`Tracer`]s.
    fn lift(&self, constant: Self::Constant) -> Result<Self::Value, ProgramError>;

    /// Binds the provided [`Operation`] to the provided input [`Value`]s in this [`Context`] and returns the resulting
    /// output values. Eager contexts bind by interpreting the operation over concrete values. [`StagingContext`]s bind
    /// by recording an [`Instruction`](crate::Instruction) in their underling [`ProgramBuilder`].
    fn bind<O: Into<Self::Operation>>(
        &self,
        operation: O,
        inputs: &[Self::Value],
    ) -> Result<Vec<Self::Value>, ProgramError>;

    /// Traces `function` into a [`Program`] and interprets that program on the provided `input`. This creates an
    /// ordinary symbolic trace over this context's `(Self::Type, Self::Constant, Self::Operation)` universe through a
    /// fresh [`TracingContext`], simplifies the resulting flat program, and interprets it with the provided concrete
    /// input values. Use this when a caller needs both the staged program and the corresponding concrete output for
    /// the same input. The runtime values flow through `self`, which supplies the concrete value type and the
    /// constant-lifting behavior.
    fn interpret_and_trace<
        F: FnOnce(
            Input::To<Tracer<TracingContext<Self::Type, Self::Constant, Self::Operation>>>,
        ) -> Result<Output, ProgramError>,
        Input: Parameterized<
                Self::Value,
                Family: ParameterizedFamily<Self::Constant>
                            + ParameterizedFamily<Tracer<TracingContext<Self::Type, Self::Constant, Self::Operation>>>,
            >,
        Output: Parameterized<
                Tracer<TracingContext<Self::Type, Self::Constant, Self::Operation>>,
                Family: ParameterizedFamily<Self::Value> + ParameterizedFamily<Self::Constant>,
            >,
    >(
        &self,
        function: F,
        input: Input,
    ) -> Result<
        (
            Output::To<Self::Value>,
            Program<Self::Type, Self::Constant, Self::Operation, Input::To<Self::Constant>, Output::To<Self::Constant>>,
        ),
        ProgramError,
    >
    where
        Self::Operation: InterpretableOperation<Self::Type, Self::Value>,
        <Self::Value as Value<Self::Type>>::InterpretationContext: Default,
    {
        let input_structure = input.parameter_structure();
        let input_values = input.into_parameters().collect::<Vec<_>>();
        let input_types = input_values.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
        let mut output_structure = None;
        let (_, flat_program) =
            Self::trace(
                |flat_input| {
                    let input = <Input::To<
                    Tracer<TracingContext<Self::Type, Self::Constant, Self::Operation>>,
                >>::from_parameters(input_structure.clone(), flat_input)?;
                    let output = function(input)?;
                    output_structure = Some(output.parameter_structure());
                    Ok(output.into_parameters().collect::<Vec<_>>())
                },
                input_types,
            )?;
        let output_structure = output_structure.unwrap();
        let flat_program = flat_program.into_simplified()?;
        let interpretation_context = <Self::Value as Value<Self::Type>>::InterpretationContext::default();
        let output_values = flat_program.interpret_with(
            input_values,
            |_, constant| self.lift(constant.clone()),
            |instruction, inputs| instruction.operation().interpret(&interpretation_context, inputs),
        )?;
        let output = Output::To::<Self::Value>::from_parameters(output_structure.clone(), output_values)?;
        let program = Program {
            atoms: flat_program.atoms,
            input_ids: flat_program.input_ids,
            output_ids: flat_program.output_ids,
            instructions: flat_program.instructions,
            input_structure,
            output_structure,
            marker: PhantomData,
        };
        Ok((output, program))
    }
}

/// Represents instances that can provide a [`Context`] value used by an owning or enclosing object. This trait
/// separates the type-level question "what context is needed?" from the value-level question "who can provide that
/// context?". For example, a [`StagingContext`] can provide itself as the interpretation context for its [`Tracer`]s,
/// while an **eager** [`Domain`] can provide a zero-sized [`EagerContext`] for concrete values.
pub trait ProvidesContext<C: Context> {
    /// Returns the [`Context`] provided by this instance.
    fn context(&self) -> C;
}

/// [`Context`] used for a concrete `(type, value, operation)` universe that carries no runtime state and for which,
/// binding an operation to some input values corresponds to directly interpreting/evaluating/executing that operation
/// for those input values using the value type's default interpretation context. [`EagerContext`] exists to make direct
/// interpretation contexts explicit in generic code that otherwise has no backend-owned eager context value to pass
/// around.
///
/// The default operation family is [`ConstantOperation<T, V>`](ConstantOperation), which is the minimal operation
/// family needed by ordinary eager value contexts that only materialize constants and expose context capabilities such
/// as zero, one, fill, and scale. Code that binds or batches a richer operation family should still specify `O`
/// explicitly, such as `EagerContext<ArrayType, V, ArrayOperation<V>>`.
pub struct EagerContext<T: Type, V: Value<T>, O: Operation<T> = ConstantOperation<T, V>> {
    /// [`PhantomData`] marker tying this zero-sized context to its associated types.
    marker: PhantomData<fn() -> (T, V, O)>,
}

impl<T: Type, V: Value<T>, O: Operation<T>> EagerContext<T, V, O> {
    /// Creates a new [`EagerContext`].
    #[inline]
    pub const fn new() -> Self {
        Self { marker: PhantomData }
    }
}

impl<T: Type, V: Value<T>, O: Operation<T>> Copy for EagerContext<T, V, O> {}

impl<T: Type, V: Value<T>, O: Operation<T>> Clone for EagerContext<T, V, O> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Type, V: Value<T>, O: Operation<T>> std::fmt::Debug for EagerContext<T, V, O> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("EagerContext")
    }
}

impl<T: Type, V: Value<T>, O: Operation<T>> Default for EagerContext<T, V, O> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Type, V: Value<T>, O: Operation<T>> Domain for EagerContext<T, V, O> {
    type Type = T;
    type Value = V;
    type Constant = V;
    type Operation = O;
}

impl<T: Type, V: Value<T, InterpretationContext: Default>, O: InterpretableOperation<T, V>> Context
    for EagerContext<T, V, O>
{
    #[inline]
    fn lift(&self, constant: V) -> Result<V, ProgramError> {
        Ok(constant)
    }

    #[inline]
    fn bind<P: Into<O>>(&self, operation: P, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        operation.into().interpret(&V::InterpretationContext::default(), inputs)
    }
}

/// [`Value`] that flows through a [`StagingContext`]. This is a handle to a staged [`Atom`](crate::Atom) in that
/// context's [`ProgramBuilder`], carrying the [`Context`] itself so that value can recover its [`ProgramBuilder`].
/// This generalizes the staged value representation away from the concrete [`Tracer`]. Ordinary tracing flows
/// [`Tracer`]s, while transform contexts may use richer value representations that additionally carry transform
/// metadata (e.g., a batch axis or a partial-evaluation known/unknown classification) on the value rather than in a
/// side table. The bound is on [`Context`] rather than a [`StagingContext`] so that the `Value: StagingValue<Self>`
/// constraint on [`StagingContext`] does not form a trait cycle. Staged values get interpreted in their own staging
/// context, and so [`InterpretationContext`](Value::InterpretationContext) is pinned to `C`. This is what lets generic
/// staging code recover `C` from the flowing value's [`Value`] projection.
pub trait StagingValue<C: Context>: Value<C::Type, InterpretationContext = C> {
    /// Creates a _live_ [`StagingValue`] referring to `atom` in `context`'s [`ProgramBuilder`], typed as `r#type`.
    fn live(context: C, atom: AtomId, r#type: C::Type) -> Self;

    /// Creates a _poisoned_ [`StagingValue`] (i.e., an error placeholder) of `r#type` in `context`.
    fn poison(context: C, r#type: C::Type) -> Self;

    /// Returns the staged [`AtomId`] this [`StagingValue`] refers to if it is _live_,
    /// or [`ProgramError::PoisonedValue`] if it is poisoned.
    fn atom_id(&self) -> Result<AtomId, ProgramError>;

    /// Returns the [`Context`] that this [`StagingValue`] carries.
    fn context(&self) -> &C;
}

/// Staging [`Context`] whose flowing [`Domain::Value`] is a [`StagingValue`] into an active [`ProgramBuilder`]
/// (typically a [`Tracer`], but transform contexts may use richer value representations carrying transform metadata).
/// Binding records [`Operation`] invocations as [`Program`] [`Instruction`](crate::Instruction)s rather than
/// interpreting them, and this trait owns the builder-dependent staging API: [`constant`](StagingContext::constant),
/// [`input`](StagingContext::input), [`tracer`](StagingContext::tracer), [`error`](StagingContext::error),
/// [`stage_operation`](StagingContext::stage_operation), and [`stage_program`](StagingContext::stage_program).
/// Ordinary backend tracing implements it through [`TracingContext`]. Stackable transform contexts such as batching
/// or linearization implement it by delegating to a parent context.
pub trait StagingContext: Context<Value: StagingValue<Self>> {
    /// Returns the shared [`ProgramBuilder`] owned by this [`StagingContext`].
    fn builder(&self) -> &Rc<RefCell<ProgramBuilder<Self::Type, Self::Constant, Self::Operation>>>;

    /// Creates a constant [`StagingValue`] in this context with the provided constant payload.
    #[inline]
    fn constant(&self, value: Self::Constant) -> Self::Value {
        let r#type = value.r#type().into_owned();
        let atom = self.builder().borrow_mut().add_constant(value);
        self.tracer(atom, Some(r#type))
    }

    /// Creates an input [`StagingValue`] in this context with the provided type.
    #[inline]
    fn input(&self, r#type: Self::Type) -> Self::Value {
        let atom = self.builder().borrow_mut().add_input(r#type.clone());
        self.tracer(atom, Some(r#type))
    }

    /// Constructs a _live_ [`StagingValue`] in this context referring to the provided [`AtomId`]. If the provided
    /// `r#type` is [`None`], the staged [`Atom`](crate::programs::Atom)'s type is read from the owned
    /// [`ProgramBuilder`].
    #[inline]
    fn tracer(&self, atom: AtomId, r#type: Option<Self::Type>) -> Self::Value {
        let r#type = r#type.unwrap_or_else(|| self.builder().borrow().atoms()[atom.index()].r#type().into_owned());
        Self::Value::live(self.clone(), atom, r#type)
    }

    /// Records the provided [`ProgramError`] in the underlying [`ProgramBuilder`] and returns it. If the underlying
    /// [`ProgramBuilder`] already has an error recorded, then it is left unchanged and this function acts simply as
    /// an identity function.
    #[inline]
    fn error(&self, error: ProgramError) -> ProgramError {
        let mut builder = self.builder().borrow_mut();
        if builder.error.is_none() {
            builder.error = Some(error.clone());
        }
        error
    }

    /// Returns `true` if the provided [`Tracer`] is produced by a nullary [`ZeroOperation`](crate::ZeroOperation)
    /// in this [`StagingContext`]. Structural zero recognition is intentionally narrow: inputs, constants, non-zero
    /// operations, and malformed non-nullary zero operations are not treated as **canonical** zeros. Callers that
    /// need broader algebraic simplification should perform it in [`Operation`]-owned rules rather than weakening
    /// this definition.
    fn is_zero(&self, value: &Self::Value) -> Result<bool, ProgramError>
    where
        Self::Operation: MaybeZeroOperation<Self::Type>,
    {
        check_builders!(self.builder(), value.context().builder()).map_err(|error| self.error(error))?;
        let atom = value.atom_id()?;
        let builder = self.builder().borrow();
        if builder.atoms().get(atom.index()).is_none() {
            return Err(ProgramError::UnboundAtomId { id: atom });
        }
        for instruction in builder.instructions() {
            if instruction.outputs().contains(&atom) {
                return Ok(instruction.inputs().is_empty() && instruction.operation().is_zero_operation());
            }
        }
        Ok(false)
    }

    /// Stages an application of the provided **nullary** [`Operation`] (i.e., an operation with no inputs) in this
    /// [`StagingContext`] and returns [`Tracer`]s for its outputs.
    #[inline]
    fn stage_nullary_operation<O: Into<Self::Operation>>(
        &self,
        operation: O,
    ) -> Result<Vec<Self::Value>, ProgramError> {
        self.stage_operation::<O, Self::Value>(operation, &[])
    }

    /// Stages an application of the provided [`Operation`] in this context and returns [`Tracer`]s for its outputs.
    fn stage_operation<O: Into<Self::Operation>, I: std::borrow::Borrow<Self::Value>>(
        &self,
        operation: O,
        inputs: &[I],
    ) -> Result<Vec<Self::Value>, ProgramError> {
        let operation = operation.into();
        check_builders!(self.builder(), [inputs.iter().map(|input| input.borrow().context().builder())])
            .map_err(|error| self.error(error))?;
        if self.builder().borrow().error.is_some() {
            let input_types = inputs.iter().map(|input| input.borrow().r#type().into_owned()).collect::<Vec<_>>();
            let output_types = operation.infer_output_types(input_types.as_slice())?;
            Ok(output_types
                .into_iter()
                .map(|r#type| Self::Value::poison(self.clone(), r#type))
                .collect())
        } else {
            let inputs = match inputs.iter().map(|input| input.borrow().atom_id()).collect::<Result<Vec<_>, _>>() {
                Ok(input_atom_ids) => input_atom_ids,
                Err(error) => return Err(self.error(error)),
            };
            let outputs = {
                let mut builder = self.builder().borrow_mut();
                match builder.add_instruction(operation, inputs) {
                    Ok(outputs) => outputs.to_vec(),
                    Err(error) => {
                        if builder.error.is_none() {
                            builder.error = Some(error.clone());
                        }
                        return Err(error);
                    }
                }
            };
            Ok(outputs.into_iter().map(|atom| self.tracer(atom, None)).collect::<Vec<_>>())
        }
    }

    /// Stages an entire [`Program`] as a sequence of [`Operation`]s in this context, using the supplied list of input
    /// [`Tracer`]s in the program's `input_ids` order, and returns the program's flat output tracers in the program's
    /// `output_ids` order. Constants embedded in the program are lifted into the outer context via
    /// [`Self::constant`]. This is the "inline a program into a fresh trace" primitive that transform-composition is
    /// built on. When the outer trace is a JVP, VJP, vectorization, etc., trace, the inlined operations route through
    /// the active transform's per-[`Operation`] rules automatically; there is no separate "transform a program" pass
    /// to write.
    #[inline]
    fn stage_program<Input: Parameterized<Self::Constant>, Output: Parameterized<Self::Constant>>(
        &self,
        program: &Program<Self::Type, Self::Constant, Self::Operation, Input, Output>,
        inputs: Vec<Self::Value>,
    ) -> Result<Vec<Self::Value>, ProgramError>
    where
        Self::Constant: Clone,
        Self::Operation: Clone,
    {
        program.interpret_with(
            inputs,
            |_, value| Ok::<_, ProgramError>(self.constant(value.clone())),
            |instruction, inputs| self.stage_operation(instruction.operation().clone(), inputs),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use pretty_assertions::assert_eq;

    use crate::operations::arithmetic::{AddOperation, NegOperation};
    use crate::operations::constants::{MaybeZeroOperation, OneOperation, ZeroOperation};
    use crate::operations::scalars::ScalarOperation;
    use crate::parameters::Placeholder;
    use crate::programs::{Atom, AtomId, Instruction, ProgramBuilder, ProgramError};
    use crate::scalars::{Scalar, ScalarDomain};
    use crate::tracing::{DomainTracingContext, TracerState};
    use crate::types::{DataType, Typed};

    use super::{Context, EagerContext, StagingContext};

    #[test]
    fn test_eager_context_binds_and_lifts_values() {
        let context = EagerContext::<DataType, Scalar, ScalarOperation<Scalar>>::new();
        let default_context = EagerContext::<DataType, Scalar, ScalarOperation<Scalar>>::default();
        let copied_context = context;
        let cloned_context = copied_context.clone();

        assert_eq!(format!("{context:?}"), "EagerContext");
        assert_eq!(format!("{default_context:?}"), "EagerContext");
        assert_eq!(format!("{cloned_context:?}"), "EagerContext");
        assert_eq!(context.lift(Scalar::from(2.5)), Ok(Scalar::from(2.5)));
        assert_eq!(context.bind(ZeroOperation::new(DataType::F64), &[]), Ok(vec![Scalar::from(0.0)]));
        assert_eq!(context.bind(OneOperation::new(DataType::F64), &[]), Ok(vec![Scalar::from(1.0)]));
        assert_eq!(context.bind(AddOperation, &[Scalar::from(2.0), Scalar::from(3.5)]), Ok(vec![Scalar::from(5.5)]));
    }

    #[test]
    fn test_staging_context_creates_inputs_constants_and_tracers() {
        let context = DomainTracingContext::<ScalarDomain>::new();
        let builder = context.builder().clone();

        let input = context.input(DataType::F64);
        let constant = context.constant(Scalar::from(2.5));
        let builder_typed = context.tracer(AtomId::new(0), None);
        let cached_typed = context.tracer(AtomId::new(0), Some(DataType::F64));

        assert_eq!(input.atom_id(), Ok(AtomId::new(0)));
        assert_eq!(constant.atom_id(), Ok(AtomId::new(1)));
        assert_eq!(input.r#type().into_owned(), DataType::F64);
        assert_eq!(constant.r#type().into_owned(), DataType::F64);
        assert!(matches!(builder_typed.r#type(), Cow::Borrowed(r#type) if *r#type == DataType::F64));
        assert!(matches!(cached_typed.r#type(), Cow::Borrowed(r#type) if *r#type == DataType::F64));

        let builder = builder.borrow();
        assert_eq!(builder.input_ids(), &[AtomId::new(0)]);
        assert!(builder.instructions().is_empty());
        assert!(matches!(&builder.atoms()[0], Atom::Variable(r#type) if *r#type == DataType::F64));
        assert!(matches!(&builder.atoms()[1], Atom::Constant(value) if *value == 2.5));
    }

    #[test]
    fn test_staging_context_stages_nullary_and_regular_operations() {
        let context = DomainTracingContext::<ScalarDomain>::new();
        let builder = context.builder().clone();

        let mut nullary_outputs = context.stage_nullary_operation(ZeroOperation::new(DataType::F64)).unwrap();
        assert_eq!(nullary_outputs.len(), 1);
        let zero = nullary_outputs.remove(0);
        assert_eq!(zero.atom_id(), Ok(AtomId::new(0)));
        assert_eq!(zero.r#type().into_owned(), DataType::F64);

        let lhs = context.input(DataType::F64);
        let rhs = context.input(DataType::F64);
        let mut add_outputs = context.stage_operation(AddOperation, &[&lhs, &rhs]).unwrap();
        assert_eq!(add_outputs.len(), 1);
        let sum = add_outputs.remove(0);
        assert_eq!(sum.atom_id(), Ok(AtomId::new(3)));
        assert_eq!(sum.r#type().into_owned(), DataType::F64);

        {
            let builder = builder.borrow();
            assert_eq!(builder.instructions().len(), 2);
            assert_eq!(builder.instructions()[0].inputs(), &[]);
            assert_eq!(builder.instructions()[0].outputs(), &[AtomId::new(0)]);
            assert!(builder.instructions()[0].operation().is_zero_operation());
            assert_eq!(builder.instructions()[1].inputs(), &[AtomId::new(1), AtomId::new(2)]);
            assert_eq!(builder.instructions()[1].outputs(), &[AtomId::new(3)]);
        }

        let program = builder
            .borrow()
            .clone()
            .build::<(Scalar, Scalar), Scalar>(vec![sum.atom_id().unwrap()], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        assert_eq!(program.interpret((Scalar::from(2.0), Scalar::from(3.5))), Ok(Scalar::from(5.5)));
    }

    #[test]
    fn test_staging_context_records_errors_and_returns_poisoned_outputs_after_failure() {
        let context = DomainTracingContext::<ScalarDomain>::new();
        let builder = context.builder().clone();
        let input = context.input(DataType::F64);

        let first_error = ProgramError::InvalidInputCount { expected: 1, actual: 0 };
        let second_error = ProgramError::InvalidOutputCount { expected: 1, actual: 0 };
        assert_eq!(context.error(first_error.clone()), first_error);
        assert_eq!(context.error(second_error.clone()), second_error);
        assert_eq!(builder.borrow().error().cloned(), Some(first_error.clone()));

        let mut outputs = context.stage_operation(NegOperation, &[&input]).unwrap();
        assert_eq!(outputs.len(), 1);
        let output = outputs.remove(0);
        assert_eq!(output.state(), &TracerState::Poison);
        assert_eq!(output.r#type().into_owned(), DataType::F64);
        assert_eq!(builder.borrow().error().cloned(), Some(first_error));

        let context = DomainTracingContext::<ScalarDomain>::new();
        let builder = context.builder().clone();
        let input = context.input(DataType::F64);
        let foreign_context = DomainTracingContext::<ScalarDomain>::new();
        let foreign_input = foreign_context.input(DataType::F64);

        assert!(matches!(
            context.stage_operation(AddOperation, &[&input, &foreign_input]),
            Err(ProgramError::MismatchedProgramBuilders),
        ));
        assert_eq!(builder.borrow().error().cloned(), Some(ProgramError::MismatchedProgramBuilders));
    }

    #[test]
    fn test_staging_context_stages_programs_by_lifting_constants_and_replaying_instructions() {
        let mut source_builder = ProgramBuilder::<DataType, Scalar, ScalarOperation<Scalar>>::new();
        let source_input = source_builder.add_input(DataType::F64);
        let source_constant = source_builder.add_constant(Scalar::from(4.0));
        let source_output =
            source_builder.add_instruction(AddOperation, vec![source_input, source_constant]).unwrap()[0];
        let source_program =
            source_builder.build::<Scalar, Scalar>(vec![source_output], Placeholder, Placeholder).unwrap();

        let context = DomainTracingContext::<ScalarDomain>::new();
        let builder = context.builder().clone();
        let input = context.input(DataType::F64);
        let mut outputs = context.stage_program(&source_program, vec![input]).unwrap();

        assert_eq!(outputs.len(), 1);
        let output = outputs.remove(0);
        assert_eq!(output.atom_id(), Ok(AtomId::new(2)));
        assert_eq!(output.r#type().into_owned(), DataType::F64);

        {
            let builder = builder.borrow();
            assert_eq!(builder.atoms().len(), 3);
            assert_eq!(builder.instructions().len(), 1);
            assert!(matches!(&builder.atoms()[1], Atom::Constant(value) if *value == 4.0));
            assert_eq!(builder.instructions()[0].inputs(), &[AtomId::new(0), AtomId::new(1)]);
            assert_eq!(builder.instructions()[0].outputs(), &[AtomId::new(2)]);
        }

        let program = builder
            .borrow()
            .clone()
            .build::<Scalar, Scalar>(vec![output.atom_id().unwrap()], Placeholder, Placeholder)
            .unwrap();
        assert_eq!(program.interpret(Scalar::from(3.0)), Ok(Scalar::from(7.0)));
    }

    #[test]
    fn test_staging_context_is_zero_recognizes_only_nullary_zero_operations_in_the_same_builder() {
        let context = DomainTracingContext::<ScalarDomain>::new();
        let builder = context.builder().clone();
        let input = context.input(DataType::F64);
        let constant = context.constant(Scalar::from(1.0));
        let mut zero_outputs = context.stage_nullary_operation(ZeroOperation::new(DataType::F64)).unwrap();
        let zero = zero_outputs.remove(0);
        let mut add_outputs = context.stage_operation(AddOperation, &[&input, &zero]).unwrap();
        let add_output = add_outputs.remove(0);
        let malformed_zero_output = {
            let mut builder = builder.borrow_mut();
            let output = builder.add_variable(DataType::F64);
            builder.add_instruction_unchecked(Instruction::new(
                ScalarOperation::from(ZeroOperation::new(DataType::F64)),
                vec![input.atom_id().unwrap()],
                vec![output],
            ));
            output
        };
        let malformed_zero = context.tracer(malformed_zero_output, None);

        assert!(!context.is_zero(&input).unwrap());
        assert!(!context.is_zero(&constant).unwrap());
        assert!(context.is_zero(&zero).unwrap());
        assert!(!context.is_zero(&add_output).unwrap());
        assert!(!context.is_zero(&malformed_zero).unwrap());
        assert_eq!(
            context.is_zero(&context.tracer(AtomId::new(999), Some(DataType::F64))),
            Err(ProgramError::UnboundAtomId { id: AtomId::new(999) }),
        );
        assert_eq!(builder.borrow().error(), None);

        let foreign_context = DomainTracingContext::<ScalarDomain>::new();
        let foreign_input = foreign_context.input(DataType::F64);
        assert_eq!(context.is_zero(&foreign_input), Err(ProgramError::MismatchedProgramBuilders));
        assert_eq!(builder.borrow().error().cloned(), Some(ProgramError::MismatchedProgramBuilders));
    }
}
