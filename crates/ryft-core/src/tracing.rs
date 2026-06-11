use std::borrow::Cow;
use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::rc::Rc;

use ryft_macros::Parameter;

use crate::compilation::captures::CapturedConstant;
use crate::compilation::context::CapturingContext;
use crate::contexts::{Context, StagingContext};
use crate::domains::{AbstractDomain, Domain};
use crate::operations::InterpretableOperation;
use crate::parameters::{Parameter, Parameterized, ParameterizedFamily};
use crate::programs::{AtomId, Program, ProgramBuilder, ProgramError, Value};
use crate::types::{Type, Typed};

/// Traces the provided `function` into a [`Program`]. This is the module-level equivalent of [`TracingContext::trace`].
///
/// # Parameters
///
///   - `domain`: [`Domain`] that provides the traced operation, type, and constant representations.
///   - `function`: Function/closure to trace.
///   - `input_type`: Type of the input to the function being traced. This is used to determine the types of the
///     traced [`Program`] output.
#[inline]
pub fn trace<
    'domain,
    D: Domain,
    F: FnOnce(I::To<DomainTracer<'domain, D>>) -> Result<O, ProgramError>,
    I: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<DomainTracer<'domain, D>>>,
    O: Parameterized<DomainTracer<'domain, D>, Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant>>,
>(
    domain: &'domain D,
    function: F,
    input_type: I,
) -> Result<
    (O::To<D::Type>, Program<D::Type, D::Constant, D::Operation, I::To<D::Constant>, O::To<D::Constant>>),
    ProgramError,
> {
    TracingContext::trace(domain, function, input_type)
}

/// Traces the provided `function` into a [`Program`] and interprets that program on the provided `input`. This is the
/// module-level equivalent of [`TracingContext::interpret_and_trace`].
///
/// # Parameters
///
///   - `domain`: [`Domain`] that provides the traced operation, type, and constant representations.
///   - `function`: Function/closure to trace and interpret/execute.
///   - `input`: Input value to use for tracing and interpreting the provided function.
#[inline]
pub fn interpret_and_trace<
    'domain,
    D: Context<Operation: Clone + InterpretableOperation<D::Type, D::Value>>,
    F: FnOnce(I::To<DomainTracer<'domain, D>>) -> Result<O, ProgramError>,
    I: Parameterized<
            D::Value,
            Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<DomainTracer<'domain, D>>,
            ParameterStructure: Debug + PartialEq,
        >,
    O: Parameterized<DomainTracer<'domain, D>, Family: ParameterizedFamily<D::Value> + ParameterizedFamily<D::Constant>>,
>(
    domain: &'domain D,
    function: F,
    input: I,
) -> Result<
    (O::To<D::Value>, Program<D::Type, D::Constant, D::Operation, I::To<D::Constant>, O::To<D::Constant>>),
    ProgramError,
> {
    TracingContext::interpret_and_trace(domain, function, input)
}

/// Traces the provided `function` against `input_type` and returns its inferred output type. This is the module-level
/// equivalent of [`TracingContext::infer_output_type`].
///
/// # Parameters
///
///   - `domain`: [`Domain`] that provides the traced operation, type, and constant representations.
///   - `function`: Function/closure whose output type to infer.
///   - `input_type`: Type of the input to the function that will be used to infer the type of its output.
#[inline]
pub fn infer_output_type<
    'domain,
    D: Domain,
    F: FnOnce(I::To<DomainTracer<'domain, D>>) -> Result<O, ProgramError>,
    I: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<DomainTracer<'domain, D>>>,
    O: Parameterized<DomainTracer<'domain, D>, Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant>>,
>(
    domain: &'domain D,
    function: F,
    input_type: I,
) -> Result<O::To<D::Type>, ProgramError> {
    TracingContext::infer_output_type(domain, function, input_type)
}

/// State carried by a [`Tracer`] that indicates whether this tracer is _live_ and has a corresponding
/// [`Atom`](crate::Atom) or _poisoned_, meaning that it corresponds to an error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TracerState {
    /// The corresponding [`Tracer`] is _live_ and has a corresponding [`Atom`](crate::Atom).
    Live(AtomId),

    /// The corresponding [`Tracer`] has been _poisoned_, meaning that it corresponds to an error and will propagate
    /// that error wherever it is used (i.e., it will _poison_ those corresponding downstream [`Tracer`]s too).
    Poison,
}

/// Value used while tracing [`Program`]s through an active [`Context`], substituting actual runtime values and
/// recording the executed [`Operation`](crate::Operation)s in that [`Context`]. When tracing fails, later operations
/// return _poisoned_ tracers which are represented using [`TracerState::Poison`].
#[derive(Parameter)]
pub struct Tracer<C: Context> {
    /// [`TracerState`] of this [`Tracer`].
    state: TracerState,

    /// [`Type`] of the value that this [`Tracer`] represents.
    r#type: C::Type,

    /// [`Context`] associated with this [`Tracer`].
    context: C,
}

impl<C: Context> Tracer<C> {
    /// Creates a new [`Tracer`].
    #[inline]
    pub fn new(state: TracerState, r#type: C::Type, context: C) -> Self {
        Self { state, r#type, context }
    }

    /// Returns the [`TracerState`] of this [`Tracer`].
    #[inline]
    pub fn state(&self) -> &TracerState {
        &self.state
    }

    /// Returns the [`Context`] associated with this [`Tracer`].
    #[inline]
    pub fn context(&self) -> &C {
        &self.context
    }

    /// Returns the staged [`AtomId`] for this [`Tracer`] if it is _live_,
    /// and [`ProgramError::PoisonedValue`] otherwise.
    #[inline]
    pub fn atom_id(&self) -> Result<AtomId, ProgramError> {
        match &self.state {
            TracerState::Live(atom) => Ok(*atom),
            TracerState::Poison => Err(ProgramError::PoisonedValue),
        }
    }
}

impl<C: StagingContext> Tracer<C> {
    /// Returns the [`ProgramBuilder`] associated with this [`Tracer`].
    #[inline]
    pub fn builder(&self) -> &Rc<RefCell<ProgramBuilder<C::Type, C::Constant, C::Operation>>> {
        self.context.builder()
    }

    /// Applies the provided _unary_ [`Operation`](crate::Operation) to this [`Tracer`] returning the resulting
    /// [`Tracer`]. _Unary_ operations are operations that have a single input and a single output. If the provided
    /// operation is not a unary operation, then the resulting [`Tracer`] will contain a [`TracerState::Poison`].
    pub fn unary(self, operation: C::Operation) -> Self {
        match self.context.stage_operation(operation, &[&self]) {
            Ok(mut outputs) if outputs.len() == 1 => outputs.remove(0),
            Ok(outputs) => {
                self.context.error(ProgramError::InvalidOutputCount { expected: 1, actual: outputs.len() }.into());
                Self { state: TracerState::Poison, r#type: self.r#type.clone(), context: self.context.clone() }
            }
            Err(error) => {
                self.context.error(error);
                Self { state: TracerState::Poison, r#type: self.r#type.clone(), context: self.context.clone() }
            }
        }
    }

    /// Applies the provided _binary_ [`Operation`](crate::Operation) to this [`Tracer`] and the provided [`Tracer`]
    /// returning the resulting [`Tracer`]. _Binary_ operations are operations that have two inputs and a single
    /// output. If the provided operation is not a binary operation, then the resulting [`Tracer`] will contain a
    /// [`TracerState::Poison`].
    pub fn binary(self, rhs: Self, operation: C::Operation) -> Self {
        match self.context.stage_operation(operation, &[&self, &rhs]) {
            Ok(mut outputs) if outputs.len() == 1 => outputs.remove(0),
            Ok(outputs) => {
                self.context.error(ProgramError::InvalidOutputCount { expected: 1, actual: outputs.len() }.into());
                Self { state: TracerState::Poison, r#type: self.r#type.clone(), context: self.context.clone() }
            }
            Err(error) => {
                self.context.error(error);
                Self { state: TracerState::Poison, r#type: self.r#type.clone(), context: self.context.clone() }
            }
        }
    }
}

impl<C: Context> Clone for Tracer<C> {
    fn clone(&self) -> Self {
        Self { state: self.state.clone(), r#type: self.r#type.clone(), context: self.context.clone() }
    }
}

impl<C: Context> Debug for Tracer<C> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Tracer")
            .field("state", &self.state)
            .field("type", &self.r#type)
            .finish_non_exhaustive()
    }
}

impl<C: Context> Display for Tracer<C> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.state {
            TracerState::Live(atom_id) => write!(formatter, "{atom_id}"),
            TracerState::Poison => write!(formatter, "<poison:{}>", self.r#type),
        }
    }
}

impl<C: Context> Typed<C::Type> for Tracer<C> {
    #[inline]
    fn r#type(&self) -> Cow<'_, C::Type> {
        Cow::Borrowed(&self.r#type)
    }
}

impl<C: Context> Value<C::Type> for Tracer<C> {}

impl<'domain, D: Domain> Tracer<TracingContext<'domain, D>> {
    /// Returns the [`Domain`] associated with this [`Tracer`].
    #[inline]
    pub fn tracing_domain(&self) -> &'domain D {
        self.context().domain()
    }
}

/// Ordinary active tracing [`Context`] for a [`Domain`]. [`TracingContext`] pairs the type, constant, and operation
/// representations `(T, V, O)` of a [`Domain`] with the [`ProgramBuilder`] used for one tracing invocation. Its default
/// [`StagingContext::stage_operation`] behavior records each primitive bind as a program instruction. Transform
/// contexts wrap or replace this context when they need different bind behavior, but they still share the same
/// [`Context`] protocol used by [`Tracer`] values.
pub struct TracingContext<'domain, D: Domain, C = <D as Domain>::Value> {
    /// [`Domain`] borrowed by this [`TracingContext`].
    domain: &'domain D,

    /// [`ProgramBuilder`] that owns the staged [`Program`] that is currently being traced.
    builder: Rc<RefCell<ProgramBuilder<D::Type, D::Constant, D::Operation>>>,

    /// Optional runtime capture table populated while tracing a captured [`Program`].
    captures: Option<Rc<RefCell<Vec<C>>>>,
}

impl<'domain, D: Domain> TracingContext<'domain, D> {
    /// Creates a new [`TracingContext`] that borrows the provided [`Domain`].
    #[inline]
    pub fn new(domain: &'domain D, builder: Rc<RefCell<ProgramBuilder<D::Type, D::Constant, D::Operation>>>) -> Self {
        Self { domain, builder, captures: None }
    }
}

impl<'domain, D: Domain, C> TracingContext<'domain, D, C> {
    /// Creates a new [`TracingContext`] that borrows the provided [`Domain`] with the provided shared capture table.
    #[inline]
    pub fn new_with_captures(
        domain: &'domain D,
        builder: Rc<RefCell<ProgramBuilder<D::Type, D::Constant, D::Operation>>>,
        captures: Rc<RefCell<Vec<C>>>,
    ) -> Self {
        Self { domain, builder, captures: Some(captures) }
    }

    /// Returns the [`Domain`] borrowed by this [`TracingContext`].
    #[inline]
    pub fn domain(&self) -> &'domain D {
        self.domain
    }

    /// Returns the shared [`ProgramBuilder`] owned by this [`TracingContext`].
    #[inline]
    pub fn builder(&self) -> &Rc<RefCell<ProgramBuilder<D::Type, D::Constant, D::Operation>>> {
        &self.builder
    }

    /// Replaces this [`TracingContext`]'s active [`ProgramBuilder`] and returns the previous builder. This is intended
    /// for transforms that need to consume or temporarily swap a builder while preserving the rest of the context
    /// identity, such as nested [`Program`] transposition.
    #[inline]
    pub(crate) fn replace_builder(
        &mut self,
        builder: Rc<RefCell<ProgramBuilder<D::Type, D::Constant, D::Operation>>>,
    ) -> Rc<RefCell<ProgramBuilder<D::Type, D::Constant, D::Operation>>> {
        std::mem::replace(&mut self.builder, builder)
    }
}

impl<'domain, D: Domain> TracingContext<'domain, D> {
    /// Traces `function` into a [`Program`] for the provided input types. This is the symbolic ordinary-tracing entry
    /// point. It creates a fresh [`TracingContext`] for `domain`, executes `function` once on [`Tracer`] inputs
    /// standing in for `input_type`, and returns the output types plus the finalized program. Operation binds are
    /// handled by the context's [`StagingContext::stage_operation`] implementation; the [`Domain`] only supplies the
    /// constant and operation types used by that program.
    pub fn trace<
        F: FnOnce(I::To<Tracer<Self>>) -> Result<O, ProgramError>,
        I: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<Tracer<Self>>>,
        O: Parameterized<Tracer<Self>, Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant>>,
    >(
        domain: &'domain D,
        function: F,
        input_type: I,
    ) -> Result<
        (O::To<D::Type>, Program<D::Type, D::Constant, D::Operation, I::To<D::Constant>, O::To<D::Constant>>),
        ProgramError,
    > {
        let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        let input_structure = input_type.parameter_structure();
        let (output_types, outputs, output_structure) = {
            let context = Self { domain, builder: builder.clone(), captures: None };
            let input = input_type.map_parameters(|t| context.input(t)).map_err(ProgramError::from)?;
            let output = function(input).map_err(|e| builder.borrow_mut().error.take().unwrap_or_else(|| e))?;
            builder.borrow_mut().error.take().map_or(Ok(()), Err)?;
            let output_structure = output.parameter_structure();
            let outputs = output.parameters().map(|o| o.atom_id()).collect::<Result<Vec<_>, _>>()?;
            let output_types = output.map_parameters(|o| o.r#type().into_owned()).map_err(ProgramError::from)?;
            (output_types, outputs, output_structure)
        };
        let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let program = builder.build(outputs, input_structure, output_structure)?;
        Ok((output_types, program))
    }

    /// Traces `function` into a [`Program`] and interprets that program on `input`. This creates the same ordinary
    /// trace as [`TracingContext::trace`], simplifies the flat program, and interprets it with the provided concrete
    /// input values. Use this when a caller needs both the staged program and the corresponding concrete output for
    /// the same input.
    pub fn interpret_and_trace<
        F: FnOnce(I::To<Tracer<Self>>) -> Result<O, ProgramError>,
        I: Parameterized<
                D::Value,
                Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<Tracer<Self>>,
                ParameterStructure: Debug + PartialEq,
            >,
        O: Parameterized<Tracer<Self>, Family: ParameterizedFamily<D::Value> + ParameterizedFamily<D::Constant>>,
    >(
        domain: &'domain D,
        function: F,
        input: I,
    ) -> Result<
        (O::To<D::Value>, Program<D::Type, D::Constant, D::Operation, I::To<D::Constant>, O::To<D::Constant>>),
        ProgramError,
    >
    where
        D: Context<Operation: Clone + InterpretableOperation<D::Type, D::Value>>,
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
        let output_structure = output_structure.unwrap();
        let flat_program = flat_program.into_simplified()?;
        let output_values = flat_program.interpret_with(
            input_values,
            |_, constant| domain.lift(constant.clone()),
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
            marker: std::marker::PhantomData,
        };
        Ok((output, program))
    }

    /// Traces `function` against `input_type` and returns the output type, without retaining the traced [`Program`].
    /// Use this when callers only need the output types of an ordinary symbolic trace.
    #[inline]
    pub fn infer_output_type<
        F: FnOnce(I::To<Tracer<Self>>) -> Result<O, ProgramError>,
        I: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<Tracer<Self>>>,
        O: Parameterized<Tracer<Self>, Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant>>,
    >(
        domain: &'domain D,
        function: F,
        input_type: I,
    ) -> Result<O::To<D::Type>, ProgramError> {
        Ok(Self::trace(domain, function, input_type)?.0)
    }
}

impl<'domain, D: Domain, C> Clone for TracingContext<'domain, D, C> {
    fn clone(&self) -> Self {
        Self { domain: self.domain, builder: self.builder.clone(), captures: self.captures.clone() }
    }
}

impl<'domain, D: Domain, C> Debug for TracingContext<'domain, D, C> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("TracingContext").finish_non_exhaustive()
    }
}

impl<'domain, D: Domain, C> Domain for TracingContext<'domain, D, C> {
    type Type = D::Type;
    type Value = Tracer<Self>;
    type Constant = D::Constant;
    type Operation = D::Operation;
}

impl<'domain, D: Domain, C> Context for TracingContext<'domain, D, C> {
    #[inline]
    fn lift(&self, constant: D::Constant) -> Result<Tracer<Self>, ProgramError> {
        Ok(self.constant(constant))
    }

    #[inline]
    fn bind(&self, operation: D::Operation, inputs: &[Tracer<Self>]) -> Result<Vec<Tracer<Self>>, ProgramError> {
        self.stage_operation(operation, inputs)
    }
}

impl<'domain, D: Domain, C> StagingContext for TracingContext<'domain, D, C> {
    #[inline]
    fn builder(&self) -> &Rc<RefCell<ProgramBuilder<Self::Type, Self::Constant, Self::Operation>>> {
        &self.builder
    }
}

impl<'domain, T: Type, D: Domain<Type = T, Constant = CapturedConstant<T>>, C: Value<T>> CapturingContext<C>
    for TracingContext<'domain, D, C>
{
    fn capture(&self, value: C) -> Result<Self::Constant, ProgramError> {
        let captures = self.captures.as_ref().ok_or_else(|| {
            self.error(ProgramError::MalformedProgram("the tracing context does not have a capture table".to_string()))
        })?;
        let mut captures = captures.borrow_mut();
        let constant = CapturedConstant::new(captures.len(), value.r#type().into_owned());
        captures.push(value);
        Ok(constant)
    }
}

/// [`Tracer`] flowing through a [`TracingContext`] for a backend [`Domain`] `D`. This is the value that stands in for
/// a `D`-typed runtime value while a function is traced into a [`Program`]. Each [`Operation`](crate::Operation) bound
/// on these tracers records a program instruction and yields further [`DomainTracer`]s, and so ordinary backend
/// traces flow entirely in them. The `'domain` lifetime ties the tracer to the borrowed [`Domain`]. Refer to
/// [`AbstractTracer`] for the backend-less [`AbstractDomain`] specialization used during symbolic program tracing
/// and transposition.
pub type DomainTracer<'domain, D> = Tracer<TracingContext<'domain, D>>;

/// [`DomainTracer`] specialized to an [`AbstractDomain`] (i.e., a tracer staged against the backend-less `(T, V, O)`
/// type universe rather than a concrete backend [`Domain`]). These are the [`Tracer`]s produced while a [`Program`]
/// is being built or transposed purely symbolically (e.g., used for the cotangent values a transposition rule threads
/// through its [`AbstractTracingContext`]).
pub type AbstractTracer<'domain, T, V, O> = DomainTracer<'domain, AbstractDomain<T, V, O>>;

/// [`TracingContext`] over an [`AbstractDomain`] (i.e., an active tracing context bound to the backend-less `(T, V, O)`
/// type universe rather than a concrete backend [`Domain`]). Each [`AbstractTracingContext`] owns a [`ProgramBuilder`]
/// and stages [`Instruction`](crate::Instruction)s like any other [`TracingContext`], but borrows no backend and
/// therefore cannot interpret anything; it exists merely to build or transpose [`Program`]s symbolically when there
/// is no backend to borrow. Every transposition rule receives one of these as the context that it binds the transposed
/// operation into. The `T`, `V`, and `O` parameters name the [`Type`](crate::Type), [`Value`], and
/// [`Operation`](crate::Operation) representations for the program that is being traced.
pub type AbstractTracingContext<'domain, T, V, O> = TracingContext<'domain, AbstractDomain<T, V, O>>;

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::cell::RefCell;
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::operations::Operation;
    use crate::operations::constants::{OneLike, SupportsOne, SupportsZero, ZeroLike};
    use crate::operations::scalars::ScalarOperation;
    use crate::operations::trigonometric::Sin;
    use crate::parameters::Placeholder;
    use crate::programs::{AtomId, ProgramBuilder, ProgramError};
    use crate::scalars::ScalarDomain;
    use crate::types::{DataType, TypeError, Typed};

    use super::*;

    #[test]
    fn test_trace() {
        let domain = ScalarDomain::<f64>::new();
        let (output_type, program) = trace(&domain, |x| Ok(x.clone() * x), DataType::F64).unwrap();
        assert_eq!(output_type, DataType::F64);
        assert_eq!(program.interpret(3.0), Ok(9.0));
    }

    #[test]
    fn test_interpret_and_trace() {
        let domain = ScalarDomain::<f64>::new();
        let (output, program) = interpret_and_trace(&domain, |x| Ok(x.clone() * x.clone() + x.sin()), 2.0).unwrap();
        assert_eq!(output, 2.0 * 2.0 + 2.0f64.sin());
        assert_eq!(program.interpret(3.0), Ok(3.0 * 3.0 + 3.0f64.sin()));
    }

    #[test]
    fn test_infer_output_type() {
        let domain = ScalarDomain::<f64>::new();
        let output_type = infer_output_type(&domain, |x| Ok(x.sin()), DataType::F64).unwrap();
        assert_eq!(output_type, DataType::F64);
    }

    #[test]
    fn test_tracer_state_clone_debug_and_equality() {
        let live = TracerState::Live(AtomId::new(3));
        assert_eq!(live.clone(), TracerState::Live(AtomId::new(3)));
        assert_eq!(TracerState::Poison.clone(), TracerState::Poison);
        assert_ne!(live, TracerState::Poison);
        assert_eq!(format!("{live:?}"), "Live(AtomId { index: 3 })");
        assert_eq!(format!("{:?}", TracerState::Poison), "Poison");
    }

    #[test]
    fn test_tracer() {
        let domain = ScalarDomain::<f64>::new();

        // Test handles, atom lookup, cloning, typing, and rendering.
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let atom = builder.borrow_mut().add_input(DataType::F64);
        let tracing_context = TracingContext::new(&domain, builder.clone());
        let tracer = tracing_context.tracer(atom, None);
        let poisoned = Tracer::new(TracerState::Poison, DataType::F64, tracing_context.clone());
        let cloned_tracer = tracer.clone();
        assert!(std::ptr::eq(tracer.tracing_domain(), &domain));
        assert!(Rc::ptr_eq(tracer.builder(), &builder));
        assert_eq!(tracer.atom_id(), Ok(atom));
        assert_eq!(poisoned.atom_id(), Err(ProgramError::PoisonedValue));
        assert_eq!(cloned_tracer.state(), tracer.state());
        assert_eq!(cloned_tracer.r#type(), tracer.r#type());
        assert!(Rc::ptr_eq(cloned_tracer.builder(), &builder));
        assert!(matches!(tracer.r#type(), Cow::Borrowed(r#type) if *r#type == DataType::F64));
        assert_eq!(tracer.to_string(), "%0");
        assert_eq!(format!("{tracer:?}"), "Tracer { state: Live(AtomId { index: 0 }), type: F64, .. }");
        assert_eq!(poisoned.to_string(), "<poison:f64>");
        assert_eq!(format!("{poisoned:?}"), "Tracer { state: Poison, type: F64, .. }");

        // Test staging value-level identity helpers through the tracer convenience API.
        let zero = tracer.zero_like();
        let one = tracer.one_like();
        assert_eq!(zero.r#type().into_owned(), DataType::F64);
        assert_eq!(one.r#type().into_owned(), DataType::F64);
        let zero_atom = zero.atom_id().expect("zero_like output should remain live");
        let one_atom = one.atom_id().expect("one_like output should remain live");
        let program = builder
            .borrow()
            .clone()
            .build::<f64, Vec<f64>>(vec![zero_atom, one_atom], Placeholder, vec![Placeholder, Placeholder])
            .unwrap();
        assert_eq!(program.interpret(2.0), Ok(vec![0.0, 1.0]));
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

        // Test staging a unary operation through the tracer convenience API.
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let atom = builder.borrow_mut().add_input(DataType::F64);
        let tracer = TracingContext::new(&domain, builder.clone()).tracer(atom, None);
        let output = tracer.unary(ScalarOperation::Neg);
        assert_eq!(output.r#type().into_owned(), DataType::F64);
        let output_atom = output.atom_id().expect("unary output should remain live");
        let program = builder.borrow().clone().build::<f64, f64>(vec![output_atom], Placeholder, Placeholder).unwrap();
        assert_eq!(program.interpret(2.0), Ok(-2.0));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = neg %0
                in (%1)
            "}
            .trim_end(),
        );

        // Test staging a binary operation through the tracer convenience API.
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let lhs_atom = builder.borrow_mut().add_input(DataType::F64);
        let rhs_atom = builder.borrow_mut().add_input(DataType::F64);
        let tracing_context = TracingContext::new(&domain, builder.clone());
        let lhs = tracing_context.tracer(lhs_atom, None);
        let rhs = tracing_context.tracer(rhs_atom, None);
        let output = lhs.binary(rhs, ScalarOperation::Add);
        assert_eq!(output.r#type().into_owned(), DataType::F64);
        let output_atom = output.atom_id().expect("binary output should remain live");
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

        // Test that binary operations poison the result when inputs belong to different builders.
        let builder_a = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let builder_b = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let atom_a = builder_a.borrow_mut().add_input(DataType::F64);
        let atom_b = builder_b.borrow_mut().add_input(DataType::F64);
        let tracer_a = TracingContext::new(&domain, builder_a.clone()).tracer(atom_a, None);
        let tracer_b = TracingContext::new(&domain, builder_b).tracer(atom_b, None);
        let output = tracer_a.binary(tracer_b, ScalarOperation::Add);
        assert!(matches!(output.state(), TracerState::Poison));
        assert_eq!(output.r#type().into_owned(), DataType::F64);
        assert_eq!(builder_a.borrow().error().cloned(), Some(ProgramError::MismatchedProgramBuilders));
    }

    #[test]
    fn test_tracer_unary_records_invalid_output_count_and_returns_poisoned_tracer() {
        #[derive(Copy, Clone, Debug)]
        struct NoOutputOperation;

        impl Operation<DataType> for NoOutputOperation {
            #[inline]
            fn name(&self) -> &'static str {
                "no_output"
            }

            fn infer_output_types(&self, _input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
                Ok(Vec::new())
            }
        }

        impl InterpretableOperation<DataType, f64> for NoOutputOperation {
            #[inline]
            fn interpret(&self, _inputs: &[f64]) -> Result<Vec<f64>, ProgramError> {
                Ok(Vec::new())
            }
        }

        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, NoOutputOperation>::new()));
        let input_type = DataType::F64;
        let domain = AbstractDomain::<DataType, f64, NoOutputOperation>::new();
        let tracer = TracingContext::new(&domain, builder.clone()).input(input_type);
        let output = tracer.unary(NoOutputOperation);
        assert!(matches!(output.state(), TracerState::Poison));
        assert_eq!(output.r#type().into_owned(), DataType::F64);
        assert_eq!(
            builder.borrow().error().cloned(),
            Some(ProgramError::InvalidOutputCount { expected: 1, actual: 0 }),
        );
    }

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
        let first_error = ProgramError::InvalidInputCount { expected: 1, actual: 0 };
        let second_error = ProgramError::InvalidOutputCount { expected: 1, actual: 0 };
        assert_eq!(tracing_context.error(first_error.clone()), first_error);
        assert_eq!(tracing_context.error(second_error), ProgramError::InvalidOutputCount { expected: 1, actual: 0 });
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
            Err(ProgramError::MismatchedProgramBuilders),
        ));
        assert_eq!(builder_a.borrow().error().cloned(), Some(ProgramError::MismatchedProgramBuilders));

        // Test tracing after a builder failure by returning poisoned tracers when output types can still be inferred.
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let atom = builder.borrow_mut().add_input(DataType::F64);
        let builder_error = ProgramError::InvalidInputCount { expected: 1, actual: 0 };
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
            Err(ProgramError::Type(TypeError { message })) if message == "expected 2 inputs but got 1",
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
            Err(ProgramError::Type(TypeError { message }))
                if message == "add input types are not broadcast-compatible"
        ));
        assert!(matches!(
            builder.borrow().error().cloned(),
            Some(ProgramError::Type(TypeError { message }))
                if message == "add input types are not broadcast-compatible"
        ));

        // Test staging concrete constants through the context without requiring the context itself to be a domain.
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let tracing_context = TracingContext::new(&domain, builder.clone());
        let zero = tracing_context.constant(
            domain.bind(SupportsZero::zero_operation(DataType::F64), &[]).unwrap().into_iter().next().unwrap(),
        );
        let one = tracing_context
            .constant(domain.bind(SupportsOne::one_operation(DataType::F64), &[]).unwrap().into_iter().next().unwrap());
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
            Err(ProgramError::EscapedProgramBuilder),
        ));

        // Test that [`TypeError`]s are returned in certain cases.
        assert!(matches!(
            TracingContext::trace(
                &domain,
                |inputs| Ok(inputs.0 + inputs.1),
                (DataType::F8E3M4, DataType::F32),
            ),
            Err(ProgramError::Type(TypeError { message }))
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
