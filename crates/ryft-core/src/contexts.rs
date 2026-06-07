use std::cell::RefCell;
use std::rc::Rc;

use crate::domains::Domain;
use crate::operations::Operation;
use crate::parameters::Parameterized;
use crate::programs::{AtomId, Program, ProgramBuilder, ProgramError};
use crate::tracing::domains::{Tracer, TracerState};
use crate::types::Typed;

/// Active context that can *apply* an [`Operation`] to values, layered on top of the passive [`Domain`] substrate.
/// Where a [`Domain`] only describes the type, value, constant, and operation universe, a [`Context`] additionally
/// decides what *binding* a primitive means in this context and how to [`lift`](Context::lift) a staged constant into
/// a runtime value. There are two flavors:
///
/// - *Eager* contexts, whose flowing [`Domain::Value`] is a concrete value (equal to [`Domain::Constant`]).
///   [`Context::bind`] interprets the operation immediately and there is no [`ProgramBuilder`] involved anywhere.
///   Eager backends implement [`Context`] directly. An eager context is also where interpreters and program transforms
///   synthesize a type's additive or multiplicative identity from metadata alone, via `bind(ZeroOperation, &[])` or
///   `bind(OneOperation, &[])`.
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
    /// by recording an [`Instruction`] in their underling [`ProgramBuilder`].
    fn bind(&self, operation: Self::Operation, inputs: &[Self::Value]) -> Result<Vec<Self::Value>, ProgramError>;
}

/// Staging [`Context`] whose flowing [`Domain::Value`] is a [`Tracer`] into an active [`ProgramBuilder`]. Binding
/// records [`Operation`] invocations as [`Program`] [`Instruction`]s rather than interpreting them, and this trait
/// owns the builder-dependent staging API: [`constant`](StagingContext::constant), [`input`](StagingContext::input),
/// [`tracer`](StagingContext::tracer), [`error`](StagingContext::error),
/// [`stage_operation`](StagingContext::stage_operation), and [`stage_program`](StagingContext::stage_program).
/// Ordinary backend tracing implements it through [`TracingContext`](crate::TracingContext). Stackable transform
/// contexts such as batching or linearization implement it by delegating to a parent context.
pub trait StagingContext: Context + Domain<Value = Tracer<Self>> {
    /// Returns the shared [`ProgramBuilder`] owned by this [`StagingContext`].
    fn builder(&self) -> &Rc<RefCell<ProgramBuilder<Self::Type, Self::Constant, Self::Operation>>>;

    // TODO(eaplatanios): Review from here onwards.
    /// Creates a constant [`Tracer`] in this context for the provided constant payload.
    #[inline]
    fn constant(&self, value: Self::Constant) -> Tracer<Self> {
        let r#type = value.r#type().into_owned();
        let atom = self.builder().borrow_mut().add_constant(value);
        self.tracer(atom, Some(r#type))
    }

    /// Creates an input [`Tracer`] in this context for the provided type.
    #[inline]
    fn input(&self, r#type: Self::Type) -> Tracer<Self> {
        let atom = self.builder().borrow_mut().add_input(r#type.clone());
        self.tracer(atom, Some(r#type))
    }

    /// Constructs a [`TracerState::Live`] [`Tracer`] in this context for the provided [`AtomId`]. If the provided
    /// `r#type` is [`None`], the staged [`Atom`](crate::programs::Atom)'s type is read from the owned
    /// [`ProgramBuilder`].
    #[inline]
    fn tracer(&self, atom: AtomId, r#type: Option<Self::Type>) -> Tracer<Self> {
        let r#type = r#type.unwrap_or_else(|| self.builder().borrow().atoms()[atom.index()].r#type().into_owned());
        Tracer::new(TracerState::Live(atom), r#type, self.clone())
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

    /// Stages an application of the provided [`Operation`] in this context and returns [`Tracer`]s for its outputs.
    fn stage_operation<I: std::borrow::Borrow<Tracer<Self>>>(
        &self,
        operation: Self::Operation,
        inputs: &[I],
    ) -> Result<Vec<Tracer<Self>>, ProgramError> {
        if inputs.iter().any(|input| !Rc::ptr_eq(self.builder(), input.borrow().context().builder())) {
            return Err(self.error(ProgramError::MismatchedProgramBuilders));
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
        inputs: Vec<Tracer<Self>>,
    ) -> Result<Vec<Tracer<Self>>, ProgramError>
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
