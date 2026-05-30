use std::cell::RefCell;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::Domain;
use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::{Parameter, Parameterized, ParameterizedFamily as ParameterFamily};
use crate::tracing::domains::{CapturingDomain, ProgramTracingDomain, Tracer, TracerState, TracingDomain};
use crate::tracing::{AtomId, Program, ProgramBuilder, Traceable, TracingError};
use crate::types::{Type, Typed};

/// Active staging frame that owns the effect of binding traced [`Operation`]s. A [`Context`] is the runtime object
/// carried by [`Tracer`]s. It owns the active [`ProgramBuilder`] used during tracing and decides what each primitive
/// bind means for an invocation. Ordinary tracing appends the operation to a program. Transform contexts such as
/// batching or linearization intercept the same bind, update transform-local state, and usually stage rewritten
/// operations into a parent context.
///
/// This is deliberately separate from [`TracingDomain`]. Domains describe passive backend capabilities such as value,
/// constant, and operation types. Contexts are active effect handlers for tracing or transform frames.
pub trait Context: Clone + Sized {
    /// Type metadata used by values handled by this [`Context`].
    type Type: Type + Parameter;

    /// Payload representation stored in constants owned by this [`Context`]'s [`ProgramBuilder`]. A [`Context`] does
    /// not have its own `Constant` associated type because transform contexts are not necessarily backend domains. For
    /// an ordinary [`TracingContext<'_, D>`], this is `D::Constant`. Transform contexts choose the payload
    /// representation required by their own builder.
    type Value: Traceable<Self::Type>;

    /// Operation representation accepted by this [`Context`].
    type Operation: Operation<Self::Type>;

    /// Returns the shared [`ProgramBuilder`] owned by this [`Context`].
    fn builder(&self) -> &Rc<RefCell<ProgramBuilder<Self::Type, Self::Value, Self::Operation>>>;

    /// Creates a constant [`Tracer`] in this [`Context`] for the provided concrete value.
    #[inline]
    fn constant(&self, value: Self::Value) -> Tracer<Self> {
        let r#type = value.r#type().into_owned();
        let atom = self.builder().borrow_mut().add_constant(value);
        self.tracer(atom, Some(r#type))
    }

    /// Creates an input [`Tracer`] in this [`Context`] for the provided type.
    #[inline]
    fn input(&self, r#type: Self::Type) -> Tracer<Self> {
        let atom = self.builder().borrow_mut().add_input(r#type.clone());
        self.tracer(atom, Some(r#type))
    }

    /// Constructs a [`TracerState::Live`] [`Tracer`] in this [`Context`] for the provided [`AtomId`]. If the
    /// provided `r#type` is [`None`], the staged [`Atom`](crate::tracing::Atom)'s type is read from the owned
    /// [`ProgramBuilder`].
    #[inline]
    fn tracer(&self, atom: AtomId, r#type: Option<Self::Type>) -> Tracer<Self> {
        let r#type = r#type.unwrap_or_else(|| self.builder().borrow().atoms()[atom.index()].r#type().into_owned());
        Tracer::new(TracerState::Live(atom), r#type, self.clone())
    }

    /// Records the provided [`TracingError`] in the underlying [`ProgramBuilder`] and returns it. If the underlying
    /// [`ProgramBuilder`] already has an error recorded, then it is left unchanged and this function acts simply as
    /// an identity function.
    #[inline]
    fn error(&self, error: TracingError) -> TracingError {
        let mut builder = self.builder().borrow_mut();
        if builder.error.is_none() {
            builder.error = Some(error.clone());
        }
        error
    }

    /// Stages an application of the provided [`Operation`] in this [`Context`] and returns [`Tracer`]s for its outputs.
    fn stage_operation<I: std::borrow::Borrow<Tracer<Self>>>(
        &self,
        operation: Self::Operation,
        inputs: &[I],
    ) -> Result<Vec<Tracer<Self>>, TracingError> {
        if inputs.iter().any(|input| !Rc::ptr_eq(self.builder(), input.borrow().context().builder())) {
            return Err(self.error(TracingError::MismatchedProgramBuilders));
        }
        if self.builder().borrow().error.is_some() {
            let input_types = inputs.iter().map(|input| input.borrow().r#type().into_owned()).collect::<Vec<_>>();
            let output_types = operation.infer_output_types(input_types.as_slice())?;
            Ok(output_types
                .into_iter()
                .map(|r#type| Tracer::new(TracerState::Poison, r#type, self.clone()))
                .collect())
        } else {
            let input_atom_ids =
                match inputs.iter().map(|input| input.borrow().atom_id()).collect::<Result<Vec<_>, _>>() {
                    Ok(input_atom_ids) => input_atom_ids,
                    Err(error) => return Err(self.error(error)),
                };
            let output_atom_ids = {
                let mut builder = self.builder().borrow_mut();
                match builder.add_instruction(operation, input_atom_ids) {
                    Ok(outputs) => outputs.to_vec(),
                    Err(error) => {
                        if builder.error.is_none() {
                            builder.error = Some(error.clone());
                        }
                        return Err(error);
                    }
                }
            };
            Ok(output_atom_ids.into_iter().map(|atom| self.tracer(atom, None)).collect::<Vec<_>>())
        }
    }

    /// Stages an entire [`Program`] as a sequence of [`Operation`]s in this [`Context`], using the supplied list of
    /// input [`Tracer`]s in the program's `input_ids` order, and returns the program's flat output tracers in the
    /// program's `output_ids` order. Constants embedded in the program are lifted into the outer context via
    /// [`Self::constant`]. This is the "inline a program into a fresh trace" primitive that transform-composition is
    /// built on. When the outer trace is a JVP, VJP, vectorization, etc., trace, the inlined operations route through
    /// the active transform's per-[`Operation`] rules automatically; there is no separate "transform a program" pass
    /// to write.
    #[inline]
    fn stage_program<Input: Parameterized<Self::Value>, Output: Parameterized<Self::Value>>(
        &self,
        program: &Program<Self::Type, Self::Value, Self::Operation, Input, Output>,
        inputs: Vec<Tracer<Self>>,
    ) -> Result<Vec<Tracer<Self>>, TracingError>
    where
        Self::Value: Clone,
        Self::Operation: Clone,
    {
        program.interpret_with(
            inputs,
            |_, value| Ok::<_, TracingError>(self.constant(value.clone())),
            |instruction, inputs| self.stage_operation(instruction.operation().clone(), inputs),
        )
    }
}

/// Active tracing [`Context`] that can register runtime values as captures of the program being built. The returned
/// value is the context's staged constant payload. For captured-program backends this is usually a lifetime-free
/// reference into a side table owned by the surrounding compiled function. Stackable transform contexts implement
/// this by delegating to their parent context so capture registration follows the same nesting path as ordinary
/// operation staging.
pub trait CaptureContext<C: Traceable<Self::Type>>: Context {
    /// Appends `capture` to the active capture table and returns the constant payload that refers to it.
    fn capture_value(&self, capture: C) -> Result<Self::Value, TracingError>;

    /// Appends all values in `captures` to the active capture table, preserving their order,
    /// and returns the constant payloads that refer to them.
    fn capture_values<I: IntoIterator<Item = C>>(&self, captures: I) -> Result<Vec<Self::Value>, TracingError> {
        captures.into_iter().map(|capture| self.capture_value(capture)).collect()
    }
}

/// Ordinary active tracing [`Context`] for a [`TracingDomain`]. [`TracingContext`] pairs a passive [`TracingDomain`]
/// with the [`ProgramBuilder`] used for one tracing invocation. Its default [`Context::stage_operation`] behavior
/// records each primitive bind as a program instruction. Transform contexts wrap or replace this context when they
/// need different bind behavior, but they still share the same [`Context`] protocol used by [`Tracer`] values.
///
/// Use [`TracingContext::trace`] to create a fresh ordinary tracing context, run a closure once over tracers, and
/// finalize the resulting [`Program`]. Use [`TracingContext::interpret_and_trace`] when concrete inputs should be
/// interpreted against the traced program as part of the same high-level operation.
pub struct TracingContext<'domain, D: TracingDomain, C = <D as Domain>::Value> {
    /// [`TracingDomain`] borrowed by this [`TracingContext`].
    domain: &'domain D,

    /// [`ProgramBuilder`] that owns the staged [`Program`] that is currently being traced.
    builder: Rc<RefCell<ProgramBuilder<D::Type, D::Constant, D::Operation>>>,

    /// Optional runtime capture table populated while tracing a captured [`Program`].
    captures: Option<Rc<RefCell<Vec<C>>>>,
}

impl<'domain, D: TracingDomain> TracingContext<'domain, D> {
    /// Creates a new [`TracingContext`] that borrows the provided [`TracingDomain`].
    #[inline]
    pub fn new(domain: &'domain D, builder: Rc<RefCell<ProgramBuilder<D::Type, D::Constant, D::Operation>>>) -> Self {
        Self { domain, builder, captures: None }
    }
}

impl<'domain, D: TracingDomain, C> TracingContext<'domain, D, C> {
    /// Creates a new [`TracingContext`] with a shared runtime capture table.
    #[inline]
    pub fn new_with_captures(
        domain: &'domain D,
        builder: Rc<RefCell<ProgramBuilder<D::Type, D::Constant, D::Operation>>>,
        captures: Rc<RefCell<Vec<C>>>,
    ) -> Self {
        Self { domain, builder, captures: Some(captures) }
    }

    /// Returns the [`TracingDomain`] borrowed by this context.
    #[inline]
    pub fn domain(&self) -> &'domain D {
        self.domain
    }

    /// Returns the shared [`ProgramBuilder`] owned by this context.
    #[inline]
    pub fn builder(&self) -> &Rc<RefCell<ProgramBuilder<D::Type, D::Constant, D::Operation>>> {
        &self.builder
    }

    /// Replaces this context's active [`ProgramBuilder`] and returns the previous builder. This is crate-visible for
    /// transforms that need to consume or temporarily swap a builder while preserving the rest of the context identity,
    /// such as nested [`Program`] transposition.
    #[inline]
    pub(crate) fn replace_builder(
        &mut self,
        builder: Rc<RefCell<ProgramBuilder<D::Type, D::Constant, D::Operation>>>,
    ) -> Rc<RefCell<ProgramBuilder<D::Type, D::Constant, D::Operation>>> {
        std::mem::replace(&mut self.builder, builder)
    }
}

impl<'domain, D: TracingDomain> TracingContext<'domain, D> {
    /// Traces `function` into a [`Program`] for the provided input types. This is the symbolic ordinary-tracing entry
    /// point. It creates a fresh [`TracingContext`] for `domain`, executes `function` once on [`Tracer`] inputs
    /// standing in for `input_types`, and returns the output types plus the finalized program. Primitive binds are
    /// handled by the context's [`Context::stage_operation`] implementation; the [`TracingDomain`] only supplies the
    /// constant and operation types used by that program.
    pub fn trace<F, I, O>(
        domain: &'domain D,
        function: F,
        input_types: I,
    ) -> Result<
        (O::To<D::Type>, Program<D::Type, D::Constant, D::Operation, I::To<D::Constant>, O::To<D::Constant>>),
        TracingError,
    >
    where
        F: FnOnce(I::To<Tracer<Self>>) -> Result<O, TracingError>,
        I: Parameterized<D::Type, Family: ParameterFamily<D::Constant> + ParameterFamily<Tracer<Self>>>,
        O: Parameterized<Tracer<Self>, Family: ParameterFamily<D::Type> + ParameterFamily<D::Constant>>,
    {
        let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        let input_structure = input_types.parameter_structure();
        let (output_types, outputs, output_structure) = {
            let context = Self::new(domain, builder.clone());
            let input = input_types.map_parameters(|t| context.input(t)).map_err(TracingError::from)?;
            let output = function(input).map_err(|e| builder.borrow_mut().error.take().unwrap_or_else(|| e))?;
            builder.borrow_mut().error.take().map_or(Ok(()), Err)?;
            let output_structure = output.parameter_structure();
            let outputs = output.parameters().map(|o| o.atom_id()).collect::<Result<Vec<_>, _>>()?;
            let output_types = output.map_parameters(|o| o.r#type().into_owned()).map_err(TracingError::from)?;
            (output_types, outputs, output_structure)
        };
        let builder = Rc::try_unwrap(builder).map_err(|_| TracingError::EscapedProgramBuilder)?.into_inner();
        let program = builder.build(outputs, input_structure, output_structure)?;
        Ok((output_types, program))
    }

    /// Traces `function` into a [`Program`] and interprets that program on `input`. This creates the same ordinary
    /// trace as [`TracingContext::trace`], simplifies the flat program, and interprets it with the provided concrete
    /// input values. Use this when a caller needs both the staged program and the corresponding concrete output for
    /// the same input.
    pub fn interpret_and_trace<F, I, O>(
        domain: &'domain D,
        function: F,
        input: I,
    ) -> Result<
        (O::To<D::Value>, Program<D::Type, D::Constant, D::Operation, I::To<D::Constant>, O::To<D::Constant>>),
        TracingError,
    >
    where
        F: FnOnce(I::To<Tracer<Self>>) -> Result<O, TracingError>,
        I: Parameterized<
                D::Value,
                Family: ParameterFamily<D::Constant> + ParameterFamily<Tracer<Self>>,
                ParameterStructure: Debug + PartialEq,
            >,
        O: Parameterized<Tracer<Self>, Family: ParameterFamily<D::Value> + ParameterFamily<D::Constant>>,
        D::Operation: Clone + InterpretableOperation<D::Type, D::Value>,
    {
        let input_structure = input.parameter_structure();
        let input_values = input.into_parameters().collect::<Vec<_>>();
        let input_types = input_values.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
        let mut output_structure = None;
        let (_, flat_program) = Self::trace(
            domain,
            |flat_input| {
                let input = I::To::<Tracer<Self>>::from_parameters(input_structure.clone(), flat_input)?;
                let output = function(input)?;
                output_structure = Some(output.parameter_structure());
                Ok(output.into_parameters().collect::<Vec<_>>())
            },
            input_types,
        )?;
        let output_structure = output_structure.expect("the function being traced should have been invoked");
        let flat_program = flat_program.into_simplified()?;
        let output_values = flat_program.interpret_with(
            input_values,
            |_, constant| domain.lift_constant(constant.clone()),
            |instruction, inputs| instruction.operation().interpret(inputs),
        )?;
        let output = O::To::<D::Value>::from_parameters(output_structure.clone(), output_values)?;
        let program: Program<D::Type, D::Constant, D::Operation, I::To<D::Constant>, O::To<D::Constant>> = Program {
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

impl<'domain, D: TracingDomain, C> Clone for TracingContext<'domain, D, C> {
    fn clone(&self) -> Self {
        Self { domain: self.domain, builder: self.builder.clone(), captures: self.captures.clone() }
    }
}

impl<'domain, D: TracingDomain, C> Debug for TracingContext<'domain, D, C> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("TracingContext").finish_non_exhaustive()
    }
}

impl<'domain, D: TracingDomain, C> Context for TracingContext<'domain, D, C> {
    type Type = D::Type;
    type Value = D::Constant;
    type Operation = D::Operation;

    #[inline]
    fn builder(&self) -> &Rc<RefCell<ProgramBuilder<Self::Type, Self::Value, Self::Operation>>> {
        &self.builder
    }
}

impl<'domain, D: CapturingDomain<C>, C: Traceable<D::Type>> CaptureContext<C> for TracingContext<'domain, D, C> {
    fn capture_value(&self, capture: C) -> Result<Self::Value, TracingError> {
        let captures = self.captures.as_ref().ok_or_else(|| {
            self.error(TracingError::MalformedProgram(
                "the active tracing context does not have a capture table".to_string(),
            ))
        })?;
        let mut captures = captures.borrow_mut();
        let constant = self.domain.capture_constant(captures.len(), &capture)?;
        captures.push(capture);
        Ok(constant)
    }
}

/// [`TracingContext`] used for tracing [`Program`]s.
pub type ProgramTracingContext<'domain, T, V, O> = TracingContext<'domain, ProgramTracingDomain<T, V, O>>;

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::cell::RefCell;
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::operations::constants::{OneLike, ZeroLike};
    use crate::operations::scalars::ScalarOperation;
    use crate::operations::trigonometric::Sin;
    use crate::parameters::Placeholder;
    use crate::tracing::domains::{RuntimeDomain, ScalarDomain};
    use crate::tracing::{AtomId, ProgramBuilder, TracerState, TracingError};
    use crate::types::{DataType, TypeError, Typed};

    use super::*;

    #[test]
    fn test_tracing_context() {
        let domain = ScalarDomain::<f64>::new();

        // Test construction, cloning, and debug formatting.
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let tracing_context = TracingContext::new(&domain, builder.clone());
        let cloned_context = tracing_context.clone();
        assert!(std::ptr::eq(tracing_context.domain(), &domain));
        assert!(Rc::ptr_eq(tracing_context.builder(), &builder));
        assert!(std::ptr::eq(cloned_context.domain(), &domain));
        assert!(Rc::ptr_eq(cloned_context.builder(), &builder));
        assert_eq!(format!("{tracing_context:?}"), "TracingContext { .. }");

        // Test creating a concrete constant in the staged program.
        let constant = tracing_context.constant(2.5f64);
        assert_eq!(constant.r#type().into_owned(), DataType::F64);
        let constant_atom = constant.atom_id().expect("constant tracer should remain live");
        assert_eq!(constant_atom.index(), 0);
        let program = builder
            .borrow()
            .clone()
            .build::<Vec<f64>, f64>(vec![constant_atom], Vec::<Placeholder>::new(), Placeholder)
            .unwrap();
        assert_eq!(program.interpret(Vec::new()), Ok(2.5));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:f64 = const
                in (%0)
            "}
            .trim_end(),
        );

        // Test constructing tracers from builder-owned and explicitly cached types.
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let atom = builder.borrow_mut().add_input(DataType::F64);
        let tracing_context = TracingContext::new(&domain, builder);
        let builder_typed = tracing_context.tracer(atom, None);
        let cached_typed = tracing_context.tracer(atom, Some(DataType::F64));
        assert!(matches!(builder_typed.r#type(), Cow::Borrowed(r#type) if *r#type == DataType::F64));
        assert!(matches!(cached_typed.r#type(), Cow::Borrowed(r#type) if *r#type == DataType::F64));

        // Test that only the first recorded builder error is retained.
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let tracing_context = TracingContext::new(&domain, builder.clone());
        let first_error = TracingError::InvalidInputCount { expected: 1, got: 0 };
        let second_error = TracingError::InvalidOutputCount { expected: 1, got: 0 };
        assert_eq!(tracing_context.error(first_error.clone()), first_error);
        assert_eq!(tracing_context.error(second_error), TracingError::InvalidOutputCount { expected: 1, got: 0 });
        assert_eq!(builder.borrow().error().cloned(), Some(first_error));

        // Test staging a valid operation through the context.
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let lhs_atom = builder.borrow_mut().add_input(DataType::F64);
        let rhs_atom = builder.borrow_mut().add_input(DataType::F64);
        let tracing_context = TracingContext::new(&domain, builder.clone());
        let lhs = tracing_context.tracer(lhs_atom, None);
        let rhs = tracing_context.tracer(rhs_atom, None);
        let outputs = tracing_context.stage_operation(ScalarOperation::Add, &[&lhs, &rhs]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].state(), &TracerState::Live(AtomId::new(2)));
        assert_eq!(outputs[0].r#type().into_owned(), DataType::F64);
        let output_atom = outputs[0].atom_id().expect("output tracer should remain live");
        let program = builder
            .borrow()
            .clone()
            .build::<(f64, f64), f64>(vec![output_atom], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        assert_eq!(program.interpret((2.0, 3.0)), Ok(5.0));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = add %0 %1
                in (%2)
            "}
            .trim_end(),
        );

        // Test rejecting inputs that belong to a different program builder.
        let builder_a = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let builder_b = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let atom_a = builder_a.borrow_mut().add_input(DataType::F64);
        let atom_b = builder_b.borrow_mut().add_input(DataType::F64);
        let tracer_a = TracingContext::new(&domain, builder_a.clone()).tracer(atom_a, None);
        let tracer_b = TracingContext::new(&domain, builder_b).tracer(atom_b, None);
        assert!(matches!(
            TracingContext::new(&domain, builder_a.clone())
                .stage_operation(ScalarOperation::Add, &[&tracer_a, &tracer_b]),
            Err(TracingError::MismatchedProgramBuilders),
        ));
        assert_eq!(builder_a.borrow().error().cloned(), Some(TracingError::MismatchedProgramBuilders));

        // Test tracing after a builder failure by returning poisoned tracers when output types can still be inferred.
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let atom = builder.borrow_mut().add_input(DataType::F64);
        let builder_error = TracingError::InvalidInputCount { expected: 1, got: 0 };
        builder.borrow_mut().error = Some(builder_error.clone());
        let tracing_context = TracingContext::new(&domain, builder.clone());
        let tracer = tracing_context.tracer(atom, None);
        let outputs = tracing_context.stage_operation(ScalarOperation::Neg, &[&tracer]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(matches!(outputs[0].state(), &TracerState::Poison));
        assert_eq!(outputs[0].r#type().into_owned(), DataType::F64);
        assert_eq!(builder.borrow().error().cloned(), Some(builder_error.clone()));
        assert!(matches!(
            tracing_context.stage_operation(ScalarOperation::Add, &[&tracer]),
            Err(TracingError::Type(TypeError { message })) if message == "expected 2 inputs but got 1",
        ));
        assert_eq!(builder.borrow().error().cloned(), Some(builder_error));

        // Test propagating abstract-evaluation errors and recording them on the builder.
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let lhs_atom = builder.borrow_mut().add_input(DataType::F8E3M4);
        let rhs_atom = builder.borrow_mut().add_input(DataType::F32);
        let tracing_context = TracingContext::new(&domain, builder.clone());
        let lhs = tracing_context.tracer(lhs_atom, None);
        let rhs = tracing_context.tracer(rhs_atom, None);
        let result = tracing_context.stage_operation(ScalarOperation::Add, &[&lhs, &rhs]);
        assert!(matches!(
            result,
            Err(TracingError::Type(TypeError { message }))
                if message == "add input types are not broadcast-compatible"
        ));
        assert!(matches!(
            builder.borrow().error().cloned(),
            Some(TracingError::Type(TypeError { message }))
                if message == "add input types are not broadcast-compatible"
        ));

        // Test staging concrete constants through the context without requiring the context itself to be a domain.
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let tracing_context = TracingContext::new(&domain, builder.clone());
        let zero = tracing_context.constant(domain.zero(&DataType::F64).unwrap());
        let one = tracing_context.constant(domain.one(&DataType::F64).unwrap());
        assert_eq!(zero.r#type().into_owned(), DataType::F64);
        assert_eq!(one.r#type().into_owned(), DataType::F64);
        let zero_atom = zero.atom_id().expect("zero tracer should remain live");
        let one_atom = one.atom_id().expect("one tracer should remain live");
        assert_eq!(zero_atom.index(), 0);
        assert_eq!(one_atom.index(), 1);
        let program = builder
            .borrow()
            .clone()
            .build::<Vec<f64>, Vec<f64>>(
                vec![zero_atom, one_atom],
                Vec::<Placeholder>::new(),
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        assert_eq!(program.interpret(Vec::new()), Ok(vec![0.0, 1.0]));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:f64 = const
                    %1:f64 = const
                in (%0, %1)
            "}
            .trim_end(),
        );

        // Test staging an existing program through the context, including lifting embedded constants.
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let input = builder.add_input(DataType::F64);
        let constant = builder.add_constant(4.0f64);
        let output = builder.add_instruction(ScalarOperation::Add, vec![input, constant]).unwrap()[0];
        let program = builder.build::<f64, f64>(vec![output], Placeholder, Placeholder).unwrap();
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let tracing_context = TracingContext::new(&domain, builder.clone());
        let input = tracing_context.input(DataType::F64);
        let outputs = tracing_context.stage_program(&program, vec![input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].r#type().into_owned(), DataType::F64);
        let output_atom = outputs[0].atom_id().expect("staged program output should remain live");
        let program = builder.borrow().clone().build::<f64, f64>(vec![output_atom], Placeholder, Placeholder).unwrap();
        assert_eq!(program.interpret(3.0), Ok(7.0));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = const
                    %2:f64 = add %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_tracing_context_trace() {
        let domain = ScalarDomain::<f64>::new();
        let (output_type, program) =
            TracingContext::trace(&domain, |x| Ok(x.clone() * x.clone() + x.one_like()), DataType::F64).unwrap();
        assert_eq!(output_type, DataType::F64);
        assert_eq!(program.interpret(3.0), Ok(10.0));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = mul %0 %0
                    %2:f64 = one_like %0
                    %3:f64 = add %1 %2
                in (%3)
            "}
            .trim_end(),
        );

        // Test using an escaped [`ProgramBuilder`].
        let escaped_builder = Rc::new(RefCell::new(None));
        assert!(matches!(
            TracingContext::trace(
                &domain,
                |x| {
                    *escaped_builder.borrow_mut() = Some(x.builder().clone());
                    Ok(x)
                },
                DataType::F64,
            ),
            Err(TracingError::EscapedProgramBuilder),
        ));

        // Test that [`TypeError`]s are returned in certain cases.
        assert!(matches!(
            TracingContext::trace(
                &domain,
                |inputs| Ok(inputs.0 + inputs.1),
                (DataType::F8E3M4, DataType::F32),
            ),
            Err(TracingError::Type(TypeError { message }))
                if message == "add input types are not broadcast-compatible",
        ));
    }

    #[test]
    fn test_tracing_context_interpret_and_trace() {
        let domain = ScalarDomain::<f64>::new();
        let (output, program) =
            TracingContext::interpret_and_trace(&domain, |x| Ok(x.clone() * x.clone() + x.sin()), 2.0f64).unwrap();
        assert_eq!(output, 2.0f64 * 2.0f64 + 2.0f64.sin());
        assert_eq!(program.interpret(0.5f64), Ok(0.5f64 * 0.5f64 + 0.5f64.sin()));
        assert_eq!(program.input_ids().len(), 1);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = mul %0 %0
                    %2:f64 = sin %0
                    %3:f64 = add %1 %2
                in (%3)
            "}
            .trim_end(),
        );

        // Test using a function with a tuple argument.
        let (_, compiled) =
            TracingContext::interpret_and_trace(&domain, |(x, y)| Ok(x.clone() * y + x.sin()), (2.0f64, 3.0f64))
                .unwrap();
        assert_eq!(
            compiled.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = mul %0 %1
                    %3:f64 = sin %0
                    %4:f64 = add %2 %3
                in (%4)
            "}
            .trim_end(),
        );

        // Test using a function that contains unused code.
        let (output, program) = TracingContext::interpret_and_trace(
            &domain,
            |x| {
                let _ = x.clone().sin();
                Ok(x.clone() * x)
            },
            2.0f64,
        )
        .unwrap();
        assert_eq!(output, 4.0);
        assert_eq!(program.interpret(0.5f64), Ok(0.25));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = mul %0 %0
                in (%1)
            "}
            .trim_end(),
        );

        // Test tracing value-level identity helpers as ordinary operations.
        let (output, program) =
            TracingContext::interpret_and_trace(&domain, |x| Ok((x.zero_like(), x.one_like())), 2.0f64).unwrap();
        assert_eq!(output, (0.0, 1.0));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = zero_like %0
                    %2:f64 = one_like %0
                in (%1, %2)
            "}
            .trim_end(),
        );
    }
}
