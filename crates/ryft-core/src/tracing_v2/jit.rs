use std::{
    borrow::Cow,
    cell::RefCell,
    fmt::Display,
    ops::{Add, Mul, Neg},
    rc::Rc,
};

use ryft_macros::Parameter;

use crate::types::Type;
use crate::{
    parameters::{Parameter, Parameterized, ParameterizedFamily},
    tracing::{AtomId, InterpretableOperation, Operation, Program, ProgramBuilder, Traceable, TracingError},
    tracing_v2::{
        engines::TracingEngine,
        operations::{
            SupportsAdd, SupportsMul, SupportsNeg,
            constants::{One, OneLike, ZeroLike},
        },
    },
    types::Typed,
};

/// Execution state carried by a [`Tracer`] leaf.
///
/// Live tracers point at a concrete staged atom in the shared program builder. Poisoned tracers
/// arise only after the tracing scope has already recorded an error and can no longer stage new
/// instructions safely. They still retain the inferred abstract output type so later type queries
/// and best-effort short-circuiting can continue without manufacturing a dummy atom.
#[derive(Clone, PartialEq, Eq)]
pub enum TracerState<T: Type> {
    /// Normal traced leaf backed by a concrete atom in the staged program.
    Live(AtomId, T),

    /// Poisoned traced leaf that carries only abstract output type information.
    Poison(T),
}

impl<T: Type> TracerState<T> {
    /// Returns the staged atom id for live tracers, if one exists.
    #[inline]
    pub fn live_atom(&self) -> Option<AtomId> {
        match self {
            Self::Live(atom, _) => Some(*atom),
            Self::Poison(_) => None,
        }
    }

    /// Returns the cached abstract type carried by this tracer state.
    #[inline]
    pub fn r#type(&self) -> &T {
        match self {
            Self::Live(_, r#type) | Self::Poison(r#type) => r#type,
        }
    }
}

impl<T: Type> std::fmt::Debug for TracerState<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Live(atom, _) => formatter.debug_tuple("Live").field(atom).finish(),
            Self::Poison(_) => formatter.write_str("Poison(..)"),
        }
    }
}

/// Shared scope used while staging one traced program.
///
/// [`TracingScope`] bundles the outer [`TracingEngine`] reference with the active
/// [`ProgramBuilder`](crate::tracing::ProgramBuilder). Individual [`Tracer`] leaves carry a clone
/// of this scope so ordinary Rust operator traits can keep staging without requiring an explicit
/// context argument at every call site.
pub struct TracingScope<'engine, E: TracingEngine + ?Sized> {
    /// Engine borrowed by this tracing scope for metadata-driven value synthesis and operation selection.
    engine: &'engine E,

    /// Shared builder that owns the staged program currently being traced.
    builder: Rc<RefCell<ProgramBuilder<E::Type, E::Value, E::Operation>>>,
}

impl<'engine, E: TracingEngine + ?Sized> Clone for TracingScope<'engine, E> {
    fn clone(&self) -> Self {
        Self { engine: self.engine, builder: self.builder.clone() }
    }
}

impl<'engine, E: TracingEngine + ?Sized> std::fmt::Debug for TracingScope<'engine, E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("TracingScope").finish_non_exhaustive()
    }
}

impl<'engine, E: TracingEngine + ?Sized> TracingScope<'engine, E> {
    /// Creates a tracing scope over `engine` and `builder`.
    #[inline]
    pub fn new(engine: &'engine E, builder: Rc<RefCell<ProgramBuilder<E::Type, E::Value, E::Operation>>>) -> Self {
        Self { engine, builder }
    }

    /// Returns the outer engine borrowed by this tracing scope.
    #[inline]
    pub fn outer_engine(&self) -> &'engine E {
        self.engine
    }

    /// Returns the shared program builder owned by this tracing scope.
    #[inline]
    pub fn builder(&self) -> &Rc<RefCell<ProgramBuilder<E::Type, E::Value, E::Operation>>> {
        &self.builder
    }

    /// Returns `true` if `other` points at the same engine and program builder.
    #[inline]
    pub fn same_scope(&self, other: &Self) -> bool {
        std::ptr::eq(self.engine, other.engine) && Rc::ptr_eq(&self.builder, &other.builder)
    }

    /// Constructs a live tracer in this scope from a staged atom and its cached abstract type.
    #[inline]
    pub fn tracer_from_staged_parts(&self, atom: AtomId, r#type: E::Type) -> Tracer<'engine, E> {
        Tracer { state: TracerState::Live(atom, r#type), engine: self.clone() }
    }

    /// Constructs a live tracer in this scope by reading the staged atom's abstract type.
    #[inline]
    pub fn tracer_from_atom(&self, atom: AtomId) -> Tracer<'engine, E> {
        let r#type = self.builder.borrow().atoms[atom.index].r#type().into_owned();
        self.tracer_from_staged_parts(atom, r#type)
    }

    /// Constructs a poisoned tracer in this scope.
    #[inline]
    pub(crate) fn poisoned_tracer(&self, r#type: E::Type) -> Tracer<'engine, E> {
        Tracer { state: TracerState::Poison(r#type), engine: self.clone() }
    }

    /// Stages one primitive application in this tracing scope and returns tracers for its outputs.
    pub fn apply_staged_op(
        &self,
        inputs: &[Tracer<'engine, E>],
        op: E::Operation,
    ) -> Result<Vec<Tracer<'engine, E>>, TracingError> {
        if inputs.iter().any(|input| !Rc::ptr_eq(&self.builder, input.engine.builder())) {
            return Err(TracingError::MismatchedProgramBuilders);
        }
        if inputs.iter().any(|input| !std::ptr::eq(input.engine.outer_engine(), self.engine)) {
            return Err(TracingError::MismatchedEngines);
        }

        let input_types = inputs.iter().map(|input| input.state.r#type().clone()).collect::<Vec<_>>();
        let output_types = match op.infer_output_types(input_types.as_slice()) {
            Ok(output_types) => output_types,
            Err(error) => {
                if self.builder.borrow().error.is_none() {
                    self.builder.borrow_mut().error = Some(TracingError::from(error.clone()));
                }
                let poison_type = input_types.first().cloned().ok_or(error)?;
                return Ok(vec![self.poisoned_tracer(poison_type)]);
            }
        };
        if self.builder.borrow().error.is_some() {
            return Ok(output_types.into_iter().map(|r#type| self.poisoned_tracer(r#type)).collect());
        }

        let input_atoms = inputs.iter().map(|input| input.atom_id()).collect::<Result<Vec<_>, _>>()?;
        let output_states = match self.builder.borrow_mut().add_instruction(op, input_atoms) {
            Ok(outputs) => outputs
                .iter()
                .copied()
                .zip(output_types)
                .map(|(atom, r#type)| TracerState::Live(atom, r#type))
                .collect::<Vec<_>>(),
            Err(error) => {
                if self.builder.borrow().error.is_none() {
                    self.builder.borrow_mut().error = Some(error.clone());
                }
                output_types.into_iter().map(TracerState::Poison).collect::<Vec<_>>()
            }
        };

        Ok(output_states.into_iter().map(|state| Tracer { state, engine: self.clone() }).collect())
    }
}

/// Symbolic leaf used while staging ordinary traced programs.
///
/// A [`Tracer`] is the value-level facade for one staged traced leaf. Primitive trait impls on
/// [`Tracer`] do not compute numerically; instead, they add instructions to a shared
/// [`ProgramBuilder`](crate::tracing::ProgramBuilder) and return new tracers for the staged
/// outputs. When tracing has already failed, later operations return poisoned tracers that retain
/// only abstract type metadata rather than manufacturing dummy atoms. This makes [`Tracer`] the
/// central "big picture" type for symbolic execution in `tracing_v2`: if a closure is being
/// traced rather than eagerly evaluated, its leaves are almost always instances of this type.
#[derive(Parameter)]
pub struct Tracer<'engine, E: TracingEngine + ?Sized> {
    /// Execution state for this traced leaf.
    state: TracerState<E::Type>,

    /// Tracing scope that owns the shared builder and outer engine reference.
    pub engine: TracingScope<'engine, E>,
}

/// Tracer used by AD transforms to restrict staging to differentiable operations.
pub type DifferentiableTracer<'engine, E> = Tracer<'engine, E>;

impl<'engine, E: TracingEngine + ?Sized> Clone for Tracer<'engine, E> {
    fn clone(&self) -> Self {
        Self { state: self.state.clone(), engine: self.engine.clone() }
    }
}

impl<'engine, E: TracingEngine + ?Sized> std::fmt::Debug for Tracer<'engine, E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Tracer").field("state", &self.state).finish_non_exhaustive()
    }
}

impl<'engine, E> Display for Tracer<'engine, E>
where
    E: TracingEngine + ?Sized,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.state {
            TracerState::Live(atom, _) => Display::fmt(atom, formatter),
            TracerState::Poison(r#type) => write!(formatter, "<poison:{type}>"),
        }
    }
}

impl<'engine, E: TracingEngine + ?Sized> Tracer<'engine, E> {
    /// Constructs a traced leaf from staged tracing parts.
    ///
    /// Callers that already know the staged atom's abstract type should prefer this constructor so
    /// the resulting tracer can answer future type queries without re-borrowing the shared builder.
    #[inline]
    pub fn from_staged_parts(
        atom: AtomId,
        r#type: E::Type,
        builder: Rc<RefCell<ProgramBuilder<E::Type, E::Value, E::Operation>>>,
        engine: &'engine E,
    ) -> Self {
        TracingScope::new(engine, builder).tracer_from_staged_parts(atom, r#type)
    }

    /// Constructs a traced leaf from an existing tracing scope.
    ///
    /// This compatibility helper recovers the staged atom's abstract type from the shared builder
    /// once up front and then delegates to [`Tracer::from_staged_parts`].
    #[inline]
    pub fn from_engine(
        atom: AtomId,
        builder: Rc<RefCell<ProgramBuilder<E::Type, E::Value, E::Operation>>>,
        engine: &'engine E,
    ) -> Self {
        TracingScope::new(engine, builder).tracer_from_atom(atom)
    }

    /// Returns this tracer's leaf state.
    #[inline]
    pub fn state(&self) -> &TracerState<E::Type> {
        &self.state
    }

    /// Returns the outer engine borrowed by this tracer's tracing scope.
    #[inline]
    pub fn outer_engine(&self) -> &'engine E {
        self.engine.outer_engine()
    }

    /// Returns the shared program builder for this tracer's tracing scope.
    #[inline]
    pub fn builder(&self) -> &Rc<RefCell<ProgramBuilder<E::Type, E::Value, E::Operation>>> {
        self.engine.builder()
    }

    /// Returns the staged atom id for this tracer when it is still live.
    pub fn atom_id(&self) -> Result<AtomId, TracingError> {
        self.state.live_atom().ok_or(TracingError::PoisonedTracer)
    }

    /// Stages one primitive application in the current trace and returns tracers for its outputs.
    ///
    /// This is the common helper behind both the arithmetic trait impls on [`Tracer`] and the
    /// higher-order transforms that need to inject backend-selected operations manually. The method
    /// validates that all inputs belong to the same tracing scope, runs abstract evaluation to
    /// determine the output arity, and records the instruction unless the scope has already
    /// failed. Once the shared builder has recorded an error, subsequent calls stop mutating the
    /// staged program and instead return poisoned tracers carrying only inferred output types.
    pub fn apply_staged_op(
        engine: &'engine E,
        builder: Rc<RefCell<ProgramBuilder<E::Type, E::Value, E::Operation>>>,
        inputs: &[Self],
        op: E::Operation,
    ) -> Result<Vec<Self>, TracingError> {
        TracingScope::new(engine, builder).apply_staged_op(inputs, op)
    }

    /// Stages a single-input primitive application and returns its unique output.
    pub fn unary(self, op: E::Operation) -> Self {
        let engine = self.engine.clone();
        engine
            .apply_staged_op(std::slice::from_ref(&self), op)
            .expect("unary traced staging should preserve non-empty inputs")
            .into_iter()
            .next()
            .expect("unary traced staging should produce one output")
    }

    /// Stages a two-input primitive application and returns its unique output.
    pub fn binary(self, rhs: Self, op: E::Operation) -> Self {
        debug_assert!(Rc::ptr_eq(self.builder(), rhs.builder()));
        let engine = self.engine.clone();
        engine
            .apply_staged_op(&[self, rhs], op)
            .expect("binary traced staging should preserve non-empty inputs")
            .into_iter()
            .next()
            .expect("binary traced staging should produce one output")
    }
}

impl<'engine, E: TracingEngine + ?Sized> Typed<E::Type> for Tracer<'engine, E> {
    #[inline]
    fn r#type(&self) -> Cow<'_, E::Type> {
        Cow::Borrowed(self.state.r#type())
    }
}

impl<'engine, E> Traceable<E::Type> for Tracer<'engine, E> where E: TracingEngine + ?Sized {}

impl<'engine, E: TracingEngine + ?Sized> ZeroLike for Tracer<'engine, E> {
    #[inline]
    fn zero_like(&self) -> Self {
        let r#type = self.r#type().into_owned();
        let value = match self.outer_engine().zero(&r#type) {
            Ok(value) => value,
            Err(error) => {
                if self.builder().borrow().error.is_none() {
                    self.builder().borrow_mut().error = Some(error);
                }
                return self.engine.poisoned_tracer(r#type);
            }
        };
        let atom = self.builder().borrow_mut().add_constant(value);
        self.engine.tracer_from_staged_parts(atom, r#type)
    }
}

impl<'engine, E: TracingEngine + ?Sized> crate::tracing_v2::operations::constants::Zero<E::Type>
    for Tracer<'engine, E>
{
    /// Tracer cannot synthesize a zero from an [`E::Type`] alone because constructing a tracer
    /// requires both an engine reference and a builder, neither of which is in scope here.
    /// Programs that contain `Zero` ops over tracer values must materialize them away (via the
    /// outer-trace builder) before being interpreted.
    #[inline]
    fn zero(_type: &E::Type) -> Result<Self, TracingError> {
        Err(crate::types::TypeError {
            message: "tracer values cannot synthesize zero from type metadata; materialize zero ops first".to_string(),
        }
        .into())
    }
}

impl<'engine, E: TracingEngine + ?Sized> One<E::Type> for Tracer<'engine, E> {
    /// Tracer cannot synthesize a one from an [`E::Type`] alone because constructing a tracer
    /// requires both an engine reference and a builder, neither of which is in scope here.
    /// Differentiable tracer seeds must be derived from an existing tracer exemplar instead.
    #[inline]
    fn one(_type: &E::Type) -> Result<Self, TracingError> {
        Err(crate::types::TypeError {
            message: "tracer values cannot synthesize one from type metadata; use an exemplar-backed seed".to_string(),
        }
        .into())
    }
}

impl<'engine, E: TracingEngine + ?Sized> OneLike for Tracer<'engine, E> {
    #[inline]
    fn one_like(&self) -> Self {
        let r#type = self.r#type().into_owned();
        let value = match self.outer_engine().one(&r#type) {
            Ok(value) => value,
            Err(error) => {
                if self.builder().borrow().error.is_none() {
                    self.builder().borrow_mut().error = Some(error);
                }
                return self.engine.poisoned_tracer(r#type);
            }
        };
        let atom = self.builder().borrow_mut().add_constant(value);
        self.engine.tracer_from_staged_parts(atom, r#type)
    }
}

impl<'engine, E: TracingEngine + ?Sized> Add for Tracer<'engine, E>
where
    E::Operation: SupportsAdd<E::Type, E::Value>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.binary(rhs, E::Operation::add_operation())
    }
}

impl<'engine, E: TracingEngine + ?Sized> Mul for Tracer<'engine, E>
where
    E::Operation: SupportsMul<E::Type, E::Value>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.binary(rhs, E::Operation::mul_operation())
    }
}

impl<'engine, E: TracingEngine + ?Sized> Neg for Tracer<'engine, E>
where
    E::Operation: SupportsNeg<E::Type, E::Value>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self.unary(E::Operation::neg_operation())
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
/// staged program itself.
fn trace_active<'engine, E, F, Input, Output>(
    engine: &'engine E,
    function: F,
    input_types: Input,
) -> Result<(Output, Program<E::Type, E::Value, E::Operation, Input::To<E::Value>, Output::To<E::Value>>), TracingError>
where
    E: TracingEngine<Type: Parameter> + ?Sized,
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
    let builder = Rc::new(RefCell::new(ProgramBuilder::<E::Type, E::Value, E::Operation>::new()));
    let traced_input = Input::To::<Tracer<'engine, E>>::from_parameters(
        input_types.parameter_structure(),
        input_types.into_parameters().map(|r#type| {
            let atom = builder.borrow_mut().add_input(r#type.clone());
            Tracer::from_staged_parts(atom, r#type, builder.clone(), engine)
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
        if let Some(tracing_error) = builder.borrow_mut().error.take() {
            return Err(tracing_error);
        }
        let outputs = traced_outputs.into_iter().map(|output| output.atom_id()).collect::<Result<Vec<_>, _>>()?;
        let output_structure = output_types.parameter_structure();
        (output_structure, output_types, outputs)
    };
    let builder = match Rc::try_unwrap(builder) {
        Ok(builder) => builder.into_inner(),
        Err(_) => return Err(TracingError::EscapedProgramBuilder),
    };
    let program =
        builder.build::<Input::To<E::Value>, Output::To<E::Value>>(outputs, input_structure, output_structure)?;
    Ok((output_types, program))
}

/// Stages `function` directly from type metadata using the ordinary staged op set selected by `engine`.
pub fn trace<'engine, E, F, Input, Output>(
    engine: &'engine E,
    function: F,
    input_types: Input,
) -> Result<(Output, Program<E::Type, E::Value, E::Operation, Input::To<E::Value>, Output::To<E::Value>>), TracingError>
where
    E: TracingEngine<Type: Parameter> + ?Sized,
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
    trace_active(engine, function, input_types)
}

/// Stages `function` with the ordinary op carrier, interprets it, and returns both results.
pub fn interpret_and_trace<'engine, E, F, Input, Output>(
    engine: &'engine E,
    function: F,
    input: Input,
) -> Result<(Output, Program<E::Type, E::Value, E::Operation, Input, Output>), TracingError>
where
    E: TracingEngine<Type: Parameter, Operation: InterpretableOperation<E::Type, E::Value>> + ?Sized,
    Input: Parameterized<
            E::Value,
            ParameterStructure: Clone + std::fmt::Debug + PartialEq,
            Family: ParameterizedFamily<Tracer<'engine, E>>,
        >,
    Output: Parameterized<E::Value, ParameterStructure: Clone, Family: ParameterizedFamily<Tracer<'engine, E>>>,
    F: FnOnce(Input::To<Tracer<'engine, E>>) -> Result<Output::To<Tracer<'engine, E>>, TracingError>,
{
    let input_structure = input.parameter_structure();
    let input_values = input.into_parameters().collect::<Vec<_>>();
    let input_types = input_values.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
    let mut output_structure = None;
    let (_, flat_program): (Vec<E::Type>, Program<E::Type, E::Value, E::Operation, Vec<E::Value>, Vec<E::Value>>) =
        trace(
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
    let output_structure = output_structure
        .expect("interpret_and_trace should record the staged output structure before returning successfully");
    let Program { atoms, input_ids, output_ids, instructions, .. } = flat_program;
    let mut builder = ProgramBuilder::<E::Type, E::Value, E::Operation>::new();
    builder.atoms = atoms;
    builder.input_ids = input_ids;
    builder.instructions = instructions;
    let program = builder.build::<Input, Output>(output_ids, input_structure, output_structure)?;
    let program = program.simplified()?;
    let concrete_input = Input::from_parameters(program.input_structure.clone(), input_values)?;
    Ok((program.interpret(concrete_input)?, program))
}

#[cfg(test)]
mod tests {
    use std::{borrow::Cow, cell::RefCell, rc::Rc};

    use indoc::indoc;

    use crate::{
        parameters::Placeholder,
        tracing::{ProgramBuilder, TracingError},
        tracing_v2::{Engine, PrimitiveOperation, Sin, TracingEngine, engines::ArrayScalarEngine, test_support},
        types::{ArrayType, TypeError},
    };

    use super::*;

    #[test]
    fn jit_tracer_zero_like_adds_constant_atoms() {
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new()));
        let atom = builder.borrow_mut().add_input(3.0f64.r#type().into_owned());
        let engine = ArrayScalarEngine::<f64>::new();
        let tracer: Tracer<ArrayScalarEngine<f64>> = Tracer::from_engine(atom, builder, &engine);
        let zero = tracer.zero_like();
        assert_eq!(zero.r#type().into_owned(), ArrayType::scalar(crate::types::DataType::F64));
        let zero_atom = zero.state().live_atom().expect("zero-like tracer should remain live");
        assert!(zero_atom > atom);

        let program = zero
            .builder()
            .borrow()
            .clone()
            .build::<f64, f64>(vec![zero_atom], Placeholder, Placeholder)
            .unwrap();
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
    fn traced_live_tracer_type_borrows_cached_type() {
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new()));
        let input_type = ArrayType::scalar(crate::types::DataType::F64);
        let atom = builder.borrow_mut().add_input(input_type.clone());
        let engine = ArrayScalarEngine::<f64>::new();
        let tracer: Tracer<ArrayScalarEngine<f64>> = Tracer::from_staged_parts(atom, input_type, builder, &engine);

        assert!(
            matches!(tracer.r#type(), Cow::Borrowed(r#type) if *r#type == ArrayType::scalar(crate::types::DataType::F64))
        );
    }

    #[test]
    fn traced_apply_staged_op_rejects_mismatched_program_builders() {
        let builder_a = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new()));
        let builder_b = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new()));
        let atom_a = builder_a.borrow_mut().add_input(1.0f64.r#type().into_owned());
        let atom_b = builder_b.borrow_mut().add_input(2.0f64.r#type().into_owned());
        let engine = TaggedEngine { id: 1 };
        let tracer_a = Tracer::from_engine(atom_a, builder_a.clone(), &engine);
        let tracer_b = Tracer::from_engine(atom_b, builder_b, &engine);

        assert!(matches!(
            Tracer::apply_staged_op(&engine, builder_a, &[tracer_a, tracer_b], PrimitiveOperation::Add),
            Err(TracingError::MismatchedProgramBuilders),
        ));
    }

    #[test]
    fn traced_apply_staged_op_rejects_mismatched_engines() {
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new()));
        let atom_a = builder.borrow_mut().add_input(1.0f64.r#type().into_owned());
        let atom_b = builder.borrow_mut().add_input(2.0f64.r#type().into_owned());
        let engine_a = TaggedEngine { id: 1 };
        let engine_b = TaggedEngine { id: 2 };
        let tracer_a = Tracer::from_engine(atom_a, builder.clone(), &engine_a);
        let tracer_b = Tracer::from_engine(atom_b, builder.clone(), &engine_b);

        assert!(matches!(
            Tracer::apply_staged_op(&engine_a, builder, &[tracer_a, tracer_b], PrimitiveOperation::Add),
            Err(TracingError::MismatchedEngines),
        ));
    }

    #[test]
    fn traced_apply_staged_op_returns_poisoned_tracers_after_builder_failure() {
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new()));
        let atom = builder.borrow_mut().add_input(1.0f64.r#type().into_owned());
        builder.borrow_mut().error = Some(TracingError::InvalidInputCount { expected: 1, got: 0 });
        let engine = TaggedEngine { id: 1 };
        let tracer = Tracer::from_engine(atom, builder.clone(), &engine);

        let outputs =
            Tracer::apply_staged_op(&engine, builder, std::slice::from_ref(&tracer), PrimitiveOperation::Neg).unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(matches!(
            outputs[0].state(),
            TracerState::Poison(output_type) if *output_type == ArrayType::scalar(crate::types::DataType::F64)
        ));
    }

    #[test]
    fn traced_apply_staged_op_caches_live_output_types() {
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new()));
        let input_type = ArrayType::scalar(crate::types::DataType::F64);
        let atom = builder.borrow_mut().add_input(input_type.clone());
        let engine = TaggedEngine { id: 1 };
        let tracer = Tracer::from_staged_parts(atom, input_type, builder.clone(), &engine);

        let outputs =
            Tracer::apply_staged_op(&engine, builder, std::slice::from_ref(&tracer), PrimitiveOperation::Neg).unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(
            matches!(outputs[0].state(), TracerState::Live(_, output_type) if *output_type == ArrayType::scalar(crate::types::DataType::F64))
        );
        assert!(
            matches!(outputs[0].r#type(), Cow::Borrowed(r#type) if *r#type == ArrayType::scalar(crate::types::DataType::F64))
        );
    }

    #[test]
    fn poisoned_tracer_atom_id_returns_poisoned_tracer_error() {
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new()));
        let engine = TaggedEngine { id: 1 };
        let tracer =
            TracingScope::new(&engine, builder).poisoned_tracer(ArrayType::scalar(crate::types::DataType::F64));

        assert_eq!(tracer.atom_id(), Err(TracingError::PoisonedTracer));
    }

    #[test]
    fn staged_program_replays_graphs() {
        let engine = ArrayScalarEngine::<f64>::new();
        let (output, program): (f64, Program<ArrayType, f64, PrimitiveOperation<f64>, f64, f64>) = interpret_and_trace(
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

    struct TaggedEngine {
        id: u8,
    }

    impl Engine for TaggedEngine {
        type Type = ArrayType;
        type Value = f64;

        fn zero(&self, _type: &ArrayType) -> Result<f64, TracingError> {
            let _ = self.id;
            Ok(0.0)
        }

        fn one(&self, _type: &ArrayType) -> Result<f64, TracingError> {
            let _ = self.id;
            Ok(1.0)
        }
    }

    impl TracingEngine for TaggedEngine {
        type Operation = PrimitiveOperation<f64>;
    }

    struct FailingOneEngine;

    impl Engine for FailingOneEngine {
        type Type = ArrayType;
        type Value = f64;

        fn zero(&self, _type: &ArrayType) -> Result<f64, TracingError> {
            Ok(0.0)
        }

        fn one(&self, _type: &ArrayType) -> Result<f64, TracingError> {
            Err(TypeError { message: "test engine cannot synthesize one".to_string() }.into())
        }
    }

    impl TracingEngine for FailingOneEngine {
        type Operation = PrimitiveOperation<f64>;
    }

    #[test]
    fn tracer_one_like_records_engine_identity_error() {
        let engine = FailingOneEngine;

        assert!(matches!(
            interpret_and_trace::<FailingOneEngine, _, f64, f64>(
                &engine,
                |x: Tracer<FailingOneEngine>| Ok(x.one_like()),
                1.0f64,
            ),
            Err(TracingError::Type(TypeError { message })) if message == "test engine cannot synthesize one"
        ));
    }

    #[test]
    fn test_interpret_and_trace_supports_non_array_types() {
        use std::fmt;

        use ryft_macros::Parameter;

        use Type;

        #[derive(Clone, Debug, PartialEq, Eq)]
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

        #[derive(Clone, Debug, PartialEq, Eq, Parameter)]
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

        impl fmt::Display for TestValue {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "{}", self.value)
            }
        }

        impl Traceable<TestType> for TestValue {}

        impl crate::tracing::Value<TestType> for TestValue {}

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

        impl SupportsAdd<TestType, TestValue> for TestAddOp {
            fn add_operation() -> Self {
                Self
            }
        }

        impl Operation<TestType> for TestAddOp {
            fn name(&self) -> &'static str {
                "test_add"
            }

            fn infer_output_types(&self, input_types: &[TestType]) -> Result<Vec<TestType>, TypeError> {
                if input_types.len() != 2 {
                    return Err(TypeError {
                        message: format!("test_add expected 2 input types but got {}", input_types.len()),
                    });
                }
                if !input_types[0].is_compatible_with(&input_types[1]) {
                    return Err(TypeError { message: "test_add input types are incompatible".to_string() });
                }
                Ok(vec![input_types[0].clone()])
            }
        }

        impl InterpretableOperation<TestType, TestValue> for TestAddOp {
            fn interpret(&self, inputs: &[TestValue]) -> Result<Vec<TestValue>, TracingError> {
                if inputs.len() != 2 {
                    return Err(TracingError::InvalidInputCount { expected: 2, got: inputs.len() });
                }
                if !inputs[0].r#type.is_compatible_with(&inputs[1].r#type) {
                    return Err(TracingError::Type(TypeError {
                        message: "test_add input types are incompatible".to_string(),
                    }));
                }
                Ok(vec![inputs[0].clone() + inputs[1].clone()])
            }
        }

        struct TestEngine;

        impl Engine for TestEngine {
            type Type = TestType;
            type Value = TestValue;

            fn zero(&self, r#type: &TestType) -> Result<TestValue, TracingError> {
                Ok(TestValue::new(r#type.clone(), 0))
            }

            fn one(&self, r#type: &TestType) -> Result<TestValue, TracingError> {
                Ok(TestValue::new(r#type.clone(), 1))
            }
        }

        impl TracingEngine for TestEngine {
            type Operation = TestAddOp;
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
            tracing::TracingError,
            tracing_v2::{
                Cos, MatrixOps, Sin,
                operations::{
                    ControlFlowError, ControlFlowValue,
                    constants::{OneLike, ZeroLike},
                    reshape::ReshapeOps,
                },
            },
            types::{ArrayType, DataType, Shape, Size, TypeError, Typed},
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

        impl std::fmt::Display for TestAbstractValue {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Display::fmt(&self.r#type, formatter)
            }
        }

        impl Traceable<ArrayType> for TestAbstractValue {}

        impl crate::tracing::Value<ArrayType> for TestAbstractValue {}

        impl ControlFlowValue for TestAbstractValue {
            fn control_flow_predicate(&self) -> Result<bool, TracingError> {
                Err(ControlFlowError::InvalidPredicateValue { type_: self.r#type().into_owned() }.into())
            }
        }

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

        impl crate::tracing_v2::engines::Engine for TestEngine {
            type Type = ArrayType;
            type Value = TestAbstractValue;

            fn zero(&self, r#type: &ArrayType) -> Result<TestAbstractValue, TracingError> {
                Ok(TestAbstractValue { r#type: r#type.clone() })
            }

            fn one(&self, r#type: &ArrayType) -> Result<TestAbstractValue, TracingError> {
                Ok(TestAbstractValue { r#type: r#type.clone() })
            }
        }

        impl crate::tracing_v2::engines::TracingEngine for TestEngine {
            type Operation = crate::tracing_v2::PrimitiveOperation<TestAbstractValue>;
        }

        let result: Result<
            (
                TestAbstractValue,
                Program<
                    ArrayType,
                    TestAbstractValue,
                    crate::tracing_v2::PrimitiveOperation<TestAbstractValue>,
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

        assert!(matches!(
            result,
            Err(TracingError::Type(TypeError { message }))
                if message == "add input types are not broadcast-compatible"
        ));
    }

    #[test]
    fn staged_program_display_renders_the_staged_program() {
        let engine = ArrayScalarEngine::<f64>::new();
        let (_, compiled): (f64, Program<ArrayType, f64, PrimitiveOperation<f64>, f64, f64>) = interpret_and_trace(
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
