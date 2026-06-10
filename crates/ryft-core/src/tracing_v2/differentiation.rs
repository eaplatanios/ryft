use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::rc::Rc;

use ryft_macros::Parameter;
use thiserror::Error;

use crate::contexts::{Context, StagingContext};
use crate::differentiation::{SupportsTransposition, Tangent};
use crate::domains::{AbstractDomain, Domain};
use crate::macros::check_count;
use crate::operations::constants::{SupportsOne, SupportsZero, Zero};
use crate::operations::scalars::LinearScalarOperation;
use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily};
use crate::programs::{Atom, AtomId, Program, ProgramBuilder, ProgramError, Value};
use crate::scalars::ScalarDomain;
use crate::tracing::{Tracer, TracerState, TracingContext};
use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation, align_batch_axis, broadcast_to_batched};
use crate::tracing_v2::operations::broadcast::BroadcastInDim;
use crate::tracing_v2::operations::transpose::Transpose;
use crate::types::{ArrayType, DataType, Size, Type, Typed};

/// Errors emitted by the differentiation helpers in [`crate::tracing_v2`].
#[derive(Clone, Debug, Error, PartialEq, Eq, Hash)]
pub enum DifferentiationError {
    /// Reverse-mode differentiation (`grad`/`value_and_grad`) was requested for a function whose output is not a
    /// single scalar. Reverse mode seeds the output cotangent with the multiplicative identity ("one") and pulls it
    /// back to the inputs, which yields a gradient only when the output is a rank-0 scalar. A non-scalar output
    /// describes a vector-valued function whose full derivative is a Jacobian; because program interpretation binds
    /// inputs positionally without checking their types, seeding such an output with a ones cotangent would not
    /// fail but would instead silently compute the gradient of the sum of the outputs, so the gradient entry points
    /// reject it up front. Use a Jacobian transform such as `jacrev`/`jacfwd` for non-scalar outputs.
    #[error("gradient output must be a rank-0 scalar but got {output_type}")]
    NonScalarGradientOutput {
        /// Rendered [`Type`](crate::types::Type) of the offending non-scalar output.
        output_type: String,
    },

    /// A program-level error surfaced while differentiating.
    #[error(transparent)]
    Program(#[from] ProgramError),
}

/// Factor payload used inside residualized linear programs.
#[derive(Clone, Debug, Parameter)]
pub enum ResidualFactor<T: Type, V: Value<T>> {
    /// Closed constant factor that is independent of primal inputs.
    Constant(V),

    /// Reference to a primal residual saved by the owning [`LinearizedProgram`].
    Reference {
        /// Zero-based residual index inside the owning [`LinearizedProgram`].
        index: usize,

        /// Type metadata for the residual value.
        r#type: T,
    },
}

impl<T: Type, V: Value<T>> ResidualFactor<T, V> {
    /// Instantiates this factor into a concrete value using `residuals`.
    pub(crate) fn instantiate(&self, residuals: &[V]) -> Result<V, ProgramError> {
        match self {
            Self::Constant(value) => Ok(value.clone()),
            Self::Reference { index, .. } => {
                residuals.get(*index).cloned().ok_or(ProgramError::UnboundAtomId { id: AtomId::new(*index) }.into())
            }
        }
    }

    /// Returns this factor's residual index, if it references one.
    fn residual_index(&self) -> Option<usize> {
        match self {
            Self::Constant(_) => None,
            Self::Reference { index, .. } => Some(*index),
        }
    }

    /// Remaps this factor through a compacted residual-index table.
    fn remap_residuals(&self, mapping: &[Option<usize>]) -> Result<Self, ProgramError> {
        match self {
            Self::Constant(value) => Ok(Self::Constant(value.clone())),
            Self::Reference { index: old_index, r#type } => {
                let Some(Some(index)) = mapping.get(*old_index) else {
                    return Err(ProgramError::UnboundAtomId { id: AtomId::new(*old_index) }.into());
                };
                Ok(Self::Reference { index: *index, r#type: r#type.clone() })
            }
        }
    }
}

impl<T: Type, V: Value<T>> Display for ResidualFactor<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Constant(value) => Display::fmt(value, formatter),
            Self::Reference { index, .. } => write!(formatter, "residual[{index}]"),
        }
    }
}

impl<T: Type, V: Value<T>> Typed<T> for ResidualFactor<T, V> {
    #[inline]
    fn r#type(&self) -> Cow<'_, T> {
        match self {
            Self::Constant(value) => value.r#type(),
            Self::Reference { r#type, .. } => Cow::Borrowed(r#type),
        }
    }
}

impl<T: Type, V: Value<T>> Value<T> for ResidualFactor<T, V> {}

/// Operation contract for mapping the factor payloads carried by a linear operation.
///
/// A linear operation acts on values of some carrier `V` chosen by the surrounding [`Program`]. Some linear operations
/// also carry coefficients from the primal computation, such as scale factors and product-rule factors. This trait is
/// parameterized by that factor carrier `F` and provides the single hook needed to rewrite those carried factors.
/// Residualized pushforwards use [`ResidualFactor`] as `F`; direct programs use the concrete primal value type as `F`.
pub trait FactorParameterizedOperation<T: Type, F: Value<T>>: Clone + Operation<T> {
    /// Operation type produced by replacing `F` payloads with `MappedFactor` payloads.
    type WithFactor<MappedFactor: Value<T>>: Clone + Operation<T>;

    /// Maps every factor payload carried by this operation through `map_factor`.
    ///
    /// Operations without factor payloads should return an equivalent operation and not call `map_factor`.
    fn try_map_factors<MappedFactor: Value<T>, MapFactorFn>(
        &self,
        map_factor: &mut MapFactorFn,
    ) -> Result<Self::WithFactor<MappedFactor>, ProgramError>
    where
        MapFactorFn: FnMut(&F) -> Result<MappedFactor, ProgramError>;
}

/// Contract for linear operations whose factor payloads may reference residual values.
///
/// This is the residual-aware specialization of [`FactorParameterizedOperation`]. It turns the low-level factor-mapping
/// hook into the operations needed by reusable pushforwards: finding referenced residuals, remapping compacted residual
/// indices, instantiating a direct operation, and rebinding residual references as closed constant factors.
pub trait ResidualizedOperation<D: DifferentiationContext>:
    FactorParameterizedOperation<
        D::Type,
        ResidualFactor<D::Type, D::Value>,
        WithFactor<D::Value> = DirectLinearOperationOf<D>,
        WithFactor<ResidualFactor<D::Type, D::Value>> = LinearOperationOf<D>,
    >
{
    /// Appends residual indices referenced by this operation to `indices`.
    fn append_residual_indices(&self, indices: &mut Vec<usize>) -> Result<(), ProgramError> {
        self.try_map_factors(&mut |factor| {
            if let Some(index) = factor.residual_index() {
                indices.push(index);
            }
            Ok(factor.clone())
        })?;
        Ok(())
    }

    /// Rewrites residual references using `mapping`.
    fn remap_residuals(&self, mapping: &[Option<usize>]) -> Result<LinearOperationOf<D>, ProgramError> {
        self.try_map_factors(&mut |factor| factor.remap_residuals(mapping))
    }

    /// Instantiates residual references using `residuals`, producing a direct linear operation.
    fn instantiate_residuals(&self, residuals: &[D::Value]) -> Result<DirectLinearOperationOf<D>, ProgramError> {
        self.try_map_factors(&mut |factor| factor.instantiate(residuals))
    }

    /// Replaces residual references with closed constant residual factors.
    fn bind_residuals_as_constants(&self, residuals: &[D::Value]) -> Result<LinearOperationOf<D>, ProgramError> {
        self.try_map_factors(&mut |factor| Ok(ResidualFactor::Constant(factor.instantiate(residuals)?)))
    }
}

impl<D, O> ResidualizedOperation<D> for O
where
    D: DifferentiationContext,
    O: FactorParameterizedOperation<
            D::Type,
            ResidualFactor<D::Type, D::Value>,
            WithFactor<D::Value> = DirectLinearOperationOf<D>,
            WithFactor<ResidualFactor<D::Type, D::Value>> = LinearOperationOf<D>,
        >,
{
}

/// Residualized linear operation type selected by a [`DifferentiationContext`] implementation.
pub type LinearOperationOf<E> = <E as DifferentiationContext>::LinearOperation<
    <E as DifferentiationContext>::Tangent,
    ResidualFactor<<E as Domain>::Type, <E as Domain>::Value>,
>;

/// Directly executable linear operation type selected by a [`DifferentiationContext`] implementation.
pub type DirectLinearOperationOf<E> =
    <E as DifferentiationContext>::LinearOperation<<E as DifferentiationContext>::Tangent, <E as Domain>::Value>;

/// Reusable pushforward produced by one linearization run.
///
/// A [`Pushforward`] owns the residual values saved from the primal execution together with the residualized linear
/// program that references them. The residualized representation keeps captured primal factors explicit instead of
/// embedding them directly in operation payloads. Use [`Self::apply`] for ordinary execution or
/// [`Self::instantiate_program`] when a direct, residual-free [`Program`] is needed for debugging, transposition, or
/// APIs that require ordinary linear operations.
#[derive(Clone)]
pub struct Pushforward<D, Input, Output>
where
    D: DifferentiationContext,
    Input: Parameterized<D::Tangent>,
    Output: Parameterized<D::Tangent>,
{
    /// Residual primal values captured by the linearization run.
    residuals: Vec<D::Value>,

    /// Residualized linear program.
    program: Program<D::Type, D::Tangent, LinearOperationOf<D>, Input, Output>,
}

impl<D, Input, Output> Pushforward<D, Input, Output>
where
    D: DifferentiationContext,
    Input: Parameterized<D::Tangent>,
    Output: Parameterized<D::Tangent>,
{
    /// Creates a new [`Pushforward`].
    #[inline]
    fn new(
        residuals: Vec<D::Value>,
        program: Program<D::Type, D::Tangent, LinearOperationOf<D>, Input, Output>,
    ) -> Self {
        Self { residuals, program }
    }

    /// Returns the residual values captured by this pushforward.
    #[inline]
    pub fn residuals(&self) -> &[D::Value] {
        self.residuals.as_slice()
    }

    /// Returns the residualized linear program.
    #[inline]
    pub fn program(&self) -> &Program<D::Type, D::Tangent, LinearOperationOf<D>, Input, Output> {
        &self.program
    }

    /// Drops residual values that are no longer referenced by this pushforward's program.
    fn compact_residuals(self) -> Result<Self, ProgramError>
    where
        LinearOperationOf<D>: ResidualizedOperation<D>,
    {
        let mut referenced_indices = Vec::new();
        for instruction in self.program.instructions() {
            instruction.operation().append_residual_indices(&mut referenced_indices)?;
        }
        let residual_count = self.residuals.len();
        let mut is_referenced = vec![false; residual_count];
        let mut referenced_count = 0;
        for index in referenced_indices {
            let Some(referenced) = is_referenced.get_mut(index) else {
                return Err(ProgramError::MalformedProgram(format!(
                    "residual reference index {index} is out of bounds for {residual_count} residuals",
                ))
                .into());
            };
            if !*referenced {
                *referenced = true;
                referenced_count += 1;
            }
        }
        if referenced_count == residual_count {
            return Ok(self);
        }

        let mut mapping = vec![None; residual_count];
        let mut residuals = Vec::with_capacity(referenced_count);
        for (index, residual) in self.residuals.into_iter().enumerate() {
            if is_referenced[index] {
                mapping[index] = Some(residuals.len());
                residuals.push(residual);
            }
        }
        let program = self.program.map_operations(|operation| operation.remap_residuals(mapping.as_slice()))?;
        Ok(Self { residuals, program })
    }

    /// Instantiates this pushforward into a direct linear program with concrete factor payloads.
    pub fn instantiate_program(
        &self,
    ) -> Result<Program<D::Type, D::Tangent, DirectLinearOperationOf<D>, Input, Output>, ProgramError>
    where
        LinearOperationOf<D>: ResidualizedOperation<D>,
    {
        self.program.map_operations(|operation| operation.instantiate_residuals(self.residuals.as_slice()))
    }

    /// Returns this pushforward's residualized program with residual references replaced by closed constant factors.
    pub(crate) fn program_with_residual_constants(
        &self,
    ) -> Result<Program<D::Type, D::Tangent, LinearOperationOf<D>, Input, Output>, ProgramError>
    where
        LinearOperationOf<D>: ResidualizedOperation<D>,
    {
        self.program
            .map_operations(|operation| operation.bind_residuals_as_constants(self.residuals.as_slice()))
    }

    /// Applies this pushforward to `tangents`.
    pub fn apply(&self, tangents: Input) -> Result<Output, ProgramError>
    where
        Input::ParameterStructure: Debug + PartialEq,
        DirectLinearOperationOf<D>: InterpretableOperation<D::Type, D::Tangent>,
        LinearOperationOf<D>: ResidualizedOperation<D>,
    {
        let tangent_structure = tangents.parameter_structure();
        if tangent_structure != *self.program.input_structure() {
            return Err(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{:?}", self.program.input_structure()),
                right_structure: format!("{tangent_structure:?}"),
            }
            .into());
        }

        let inputs = tangents.into_parameters().collect::<Vec<_>>();
        let outputs = self.program.interpret_with(
            inputs,
            |_, tangent| Ok::<_, ProgramError>(tangent.clone()),
            |instruction, inputs| {
                let operation = instruction.operation().instantiate_residuals(self.residuals.as_slice())?;
                operation.interpret(inputs)
            },
        )?;
        Ok(Output::from_parameters(self.program.output_structure().clone(), outputs)?)
    }
}

/// Primal output and reusable pushforward produced by one linearization run.
#[derive(Clone)]
pub struct LinearizedProgram<D, PrimalOutput, TangentInput, TangentOutput>
where
    D: DifferentiationContext,
    TangentInput: Parameterized<D::Tangent>,
    TangentOutput: Parameterized<D::Tangent>,
{
    /// Concrete primal output.
    output: PrimalOutput,

    /// Residualized pushforward at the same primal point.
    pushforward: Pushforward<D, TangentInput, TangentOutput>,
}

impl<D, PrimalOutput, TangentInput, TangentOutput> LinearizedProgram<D, PrimalOutput, TangentInput, TangentOutput>
where
    D: DifferentiationContext,
    TangentInput: Parameterized<D::Tangent>,
    TangentOutput: Parameterized<D::Tangent>,
{
    /// Creates a new [`LinearizedProgram`].
    #[inline]
    fn new(output: PrimalOutput, pushforward: Pushforward<D, TangentInput, TangentOutput>) -> Self {
        Self { output, pushforward }
    }

    /// Consumes this value and returns the primal output and residualized pushforward.
    #[inline]
    pub fn into_parts(self) -> (PrimalOutput, Pushforward<D, TangentInput, TangentOutput>) {
        (self.output, self.pushforward)
    }
}

/// Tracer leaf used while executing one active concrete-domain linearization pass.
pub type LinearizationTracer<'domain, D> = Tracer<LinearizationContext<'domain, D, D>>;

/// Per-run trace context used by [`DifferentiationContext::linearize`].
///
/// [`LinearizationContext`] is not a backend capability. It is a one-shot [`Context`] that
/// intercepts primitive staging while a function is being linearized, runs each primitive through
/// its JVP rule, stores primal values as they are computed, and records the tangent program in the
/// selected linear domain. No primal program is built for later interpretation.
///
/// `C` is the active operation context exposed to user-facing [`Tracer`] leaves. The `D` parameter is the
/// [`DifferentiationContext`] implementation used to execute primitive rules. Concrete linearization uses an ordinary
/// [`TracingContext`] as `C` and the underlying backend domain as `D`. Nested linearization uses the enclosing context
/// as `D`, so primal values are outer-context [`Tracer`] leaves and tangent operations are staged into a linear program
/// whose values are those outer tracers.
pub struct LinearizationContext<'domain, C, D>
where
    C: Context + 'domain,
    D: DifferentiationContext<Type = C::Type, Constant = C::Constant> + 'domain,
{
    /// [`DifferentiationContext`] implementation used to run primitive JVP rules.
    differentiable: DifferentiableStorage<'domain, D>,

    /// Builder used as the primal-side atom arena for user-facing linearization tracers.
    ///
    /// The active linearization path does not build or interpret a primal [`Program`]. This builder
    /// exists because traced values still need stable primal-side [`AtomId`] handles, input and
    /// variable type metadata, constants created through [`StagingContext::constant`], the shared
    /// construction error slot used by poisoned tracers, and builder-identity checks. The active
    /// [`StagingContext::stage_operation`] implementation never appends primal instructions here; it
    /// evaluates primal values immediately and stores them in [`Self::primal_values`].
    primal_builder: Rc<RefCell<ProgramBuilder<<C as Domain>::Type, <C as Domain>::Constant, C::Operation>>>,

    /// Builder that owns the staged pushforward program.
    linear_builder: Rc<RefCell<ProgramBuilder<C::Type, D::Tangent, LinearOperationOf<D>>>>,

    /// Primal values indexed by primal-side atom id in the differentiable implementation's value representation.
    primal_values: Rc<RefCell<Vec<Option<D::Value>>>>,

    /// Tangent atom identifiers indexed by primal-side atom id. Missing entries represent structural zeros.
    tangent_atoms: Rc<RefCell<Vec<Option<AtomId>>>>,

    /// Residual values captured by product-rule factors in the staged pushforward.
    residuals: Rc<RefCell<Vec<D::Value>>>,

    /// Residual indices keyed by primal-side atom id for deduplication.
    residual_atoms: Rc<RefCell<HashMap<AtomId, usize>>>,

    /// Tangent execution mode used by JVP rules in this linearization context.
    jvp_mode: TangentMode<'domain, D>,
}

/// Borrowed-or-owned [`DifferentiationContext`] storage used by [`LinearizationContext`].
enum DifferentiableStorage<'d, D: 'd + DifferentiationContext> {
    /// Borrowed trace used by concrete linearization.
    Borrowed(&'d D),

    /// Owned cloned trace used by traced linearization.
    Owned(Rc<D>),
}

impl<D: DifferentiationContext> Clone for DifferentiableStorage<'_, D> {
    fn clone(&self) -> Self {
        match self {
            Self::Borrowed(trace) => Self::Borrowed(trace),
            Self::Owned(trace) => Self::Owned(trace.clone()),
        }
    }
}

impl<D: DifferentiationContext> DifferentiableStorage<'_, D> {
    /// Returns the stored [`DifferentiationContext`] implementation.
    #[inline]
    fn as_ref(&self) -> &D {
        match self {
            Self::Borrowed(trace) => trace,
            Self::Owned(trace) => trace.as_ref(),
        }
    }
}

/// Runs one active linearization pass for either a concrete domain or an already-active context.
///
/// `C` is the context whose tracers are exposed to the user closure. `D` is the [`DifferentiationContext`]
/// implementation that owns primal semantics and the linear operation type. Concrete-domain linearization uses a
/// borrowed domain as `D` and an ordinary [`TracingContext`] as `C`; nested active-context linearization uses the same
/// context for both roles.
fn linearize_with_context<'context, C, D, F, Input, TracedOutput>(
    differentiable: DifferentiableStorage<'context, D>,
    primals: Input,
    function: F,
) -> Result<
    LinearizedProgram<D, TracedOutput::To<D::Value>, Input::To<D::Tangent>, TracedOutput::To<D::Tangent>>,
    ProgramError,
>
where
    C: Context + 'context,
    D: DifferentiationContext<Type = C::Type, Constant = C::Constant> + 'context,
    C::Operation: DifferentiableOperation<D>,
    F: FnOnce(Input::To<Tracer<LinearizationContext<'context, C, D>>>) -> Result<TracedOutput, ProgramError>,
    Input: Parameterized<
            D::Value,
            Family: ParameterizedFamily<Tracer<LinearizationContext<'context, C, D>>> + ParameterizedFamily<D::Tangent>,
            ParameterStructure: Debug + PartialEq,
        >,
    TracedOutput: Parameterized<
            Tracer<LinearizationContext<'context, C, D>>,
            Family: ParameterizedFamily<D::Value> + ParameterizedFamily<D::Tangent>,
        >,
    LinearOperationOf<D>: ResidualizedOperation<D>,
{
    let input_structure = primals.parameter_structure();
    let input_primals = primals.into_parameters().collect::<Vec<_>>();
    let primal_builder = Rc::new(RefCell::new(ProgramBuilder::<C::Type, C::Constant, C::Operation>::new()));
    let linear_builder = Rc::new(RefCell::new(ProgramBuilder::<C::Type, D::Tangent, LinearOperationOf<D>>::new()));
    let (output_structure, output_primals, output_tangent_atoms, residuals) = {
        let linearization_context = LinearizationContext::new_with_differentiable(
            differentiable,
            primal_builder.clone(),
            linear_builder.clone(),
        );
        let mut input_tracers = Vec::with_capacity(input_primals.len());
        for input_primal in input_primals {
            let input_type = input_primal.r#type().into_owned();
            let primal_atom = primal_builder.borrow_mut().add_input(input_type.clone());
            let tangent_atom = linear_builder.borrow_mut().add_input(input_type.clone());
            linearization_context.register_input(primal_atom, input_primal, tangent_atom);
            input_tracers.push(linearization_context.tracer(primal_atom, Some(input_type)));
        }

        let input = Input::To::<Tracer<LinearizationContext<'context, C, D>>>::from_parameters(
            input_structure.clone(),
            input_tracers,
        )?;
        let output = function(input).map_err(|error| primal_builder.borrow_mut().error.take().unwrap_or(error))?;
        primal_builder.borrow_mut().error.take().map_or(Ok(()), Err)?;

        let output_structure = output.parameter_structure();
        let output_atoms = output.parameters().map(Tracer::atom_id).collect::<Result<Vec<_>, _>>()?;
        drop(output);
        let (output_primals, output_tangent_atoms) = linearization_context.collect_outputs(output_atoms.as_slice())?;
        let residuals = linearization_context.residuals.borrow().clone();
        (output_structure, output_primals, output_tangent_atoms, residuals)
    };
    Rc::try_unwrap(primal_builder).map_err(|_| ProgramError::EscapedProgramBuilder)?;
    let linear_builder = Rc::try_unwrap(linear_builder).map_err(|_| ProgramError::EscapedProgramBuilder)?;
    let pushforward = linear_builder
        .into_inner()
        .build(output_tangent_atoms, input_structure, output_structure.clone())?
        .simplified()?;
    Ok(LinearizedProgram::new(
        TracedOutput::To::<D::Value>::from_parameters(output_structure, output_primals)?,
        Pushforward::new(residuals, pushforward).compact_residuals()?,
    ))
}

/// Runs one active JVP pass for either a concrete domain or an already-active context.
///
/// This is the immediate counterpart to [`linearize_with_context`]. It uses the same active
/// [`LinearizationContext`] machinery to execute the primal closure once, but the nested
/// [`TangentContext`] interprets tangent operations as they are staged. No reusable pushforward
/// program or residual environment is built.
fn jvp_with_context<'context, C, D, F, Input, TracedOutput>(
    differentiable: DifferentiableStorage<'context, D>,
    primals: Input,
    tangents: Input::To<D::Tangent>,
    jvp_mode: TangentMode<'context, D>,
    function: F,
) -> Result<(TracedOutput::To<D::Value>, TracedOutput::To<D::Tangent>), ProgramError>
where
    C: Context + 'context,
    D: DifferentiationContext<Type = C::Type, Constant = C::Constant> + 'context,
    C::Operation: DifferentiableOperation<D>,
    F: FnOnce(Input::To<Tracer<LinearizationContext<'context, C, D>>>) -> Result<TracedOutput, ProgramError>,
    Input: Parameterized<
            D::Value,
            Family: ParameterizedFamily<Tracer<LinearizationContext<'context, C, D>>> + ParameterizedFamily<D::Tangent>,
            ParameterStructure: Debug + PartialEq,
        >,
    TracedOutput: Parameterized<
            Tracer<LinearizationContext<'context, C, D>>,
            Family: ParameterizedFamily<D::Value> + ParameterizedFamily<D::Tangent>,
        >,
    LinearOperationOf<D>: ResidualizedOperation<D>,
{
    let input_structure = primals.parameter_structure();
    let tangent_structure = tangents.parameter_structure();
    if input_structure != tangent_structure {
        return Err(ParameterError::MismatchedParameterStructures {
            left_structure: format!("{input_structure:?}"),
            right_structure: format!("{tangent_structure:?}"),
        }
        .into());
    }

    let input_primals = primals.into_parameters().collect::<Vec<_>>();
    let input_tangents = tangents.into_parameters().collect::<Vec<_>>();
    let primal_builder = Rc::new(RefCell::new(ProgramBuilder::<C::Type, C::Constant, C::Operation>::new()));
    let linear_builder = Rc::new(RefCell::new(ProgramBuilder::<C::Type, D::Tangent, LinearOperationOf<D>>::new()));
    let (output_structure, output_primals, output_tangents) = {
        let linearization_context = LinearizationContext::new_with_jvp_mode(
            differentiable,
            primal_builder.clone(),
            linear_builder.clone(),
            jvp_mode,
        );
        let tangent_context = linearization_context.tangent_context();
        let mut input_tracers = Vec::with_capacity(input_primals.len());
        for (input_primal, input_tangent) in input_primals.into_iter().zip(input_tangents.into_iter()) {
            let input_type = input_primal.r#type().into_owned();
            let primal_atom = primal_builder.borrow_mut().add_input(input_type.clone());
            let tangent_atom = tangent_context.constant(input_tangent).atom_id()?;
            linearization_context.register_input(primal_atom, input_primal, tangent_atom);
            input_tracers.push(linearization_context.tracer(primal_atom, Some(input_type)));
        }

        let input = Input::To::<Tracer<LinearizationContext<'context, C, D>>>::from_parameters(
            input_structure.clone(),
            input_tracers,
        )?;
        let output = function(input).map_err(|error| primal_builder.borrow_mut().error.take().unwrap_or(error))?;
        primal_builder.borrow_mut().error.take().map_or(Ok(()), Err)?;

        let output_structure = output.parameter_structure();
        let output_atoms = output.parameters().map(Tracer::atom_id).collect::<Result<Vec<_>, _>>()?;
        drop(output);
        let (output_primals, output_tangents) =
            linearization_context.collect_direct_outputs(output_atoms.as_slice())?;
        (output_structure, output_primals, output_tangents)
    };
    Rc::try_unwrap(primal_builder).map_err(|_| ProgramError::EscapedProgramBuilder)?;
    Rc::try_unwrap(linear_builder).map_err(|_| ProgramError::EscapedProgramBuilder)?;
    Ok((
        TracedOutput::To::<D::Value>::from_parameters(output_structure.clone(), output_primals)?,
        TracedOutput::To::<D::Tangent>::from_parameters(output_structure, output_tangents)?,
    ))
}

/// Runs direct concrete-domain JVP with batched tangent execution for dense forward Jacobians.
#[allow(private_bounds)]
pub(crate) fn direct_batched_jvp<'domain, D, F, Input, TracedOutput>(
    domain: &'domain D,
    function: F,
    primals: Input,
    tangents: Input::To<D::Tangent>,
    lane_count: usize,
) -> Result<(TracedOutput::To<<D as Domain>::Value>, TracedOutput::To<D::Tangent>), ProgramError>
where
    D: DifferentiationContext + Domain<Type = ArrayType> + 'domain,
    <D as Domain>::Operation: DifferentiableOperation<D>,
    D::Tangent: Value<ArrayType> + Zero<ArrayType> + BroadcastInDim + Transpose,
    F: FnOnce(Input::To<LinearizationTracer<'domain, D>>) -> Result<TracedOutput, ProgramError>,
    Input: Parameterized<
            <D as Domain>::Value,
            Family: ParameterizedFamily<LinearizationTracer<'domain, D>> + ParameterizedFamily<D::Tangent>,
            ParameterStructure: Debug + PartialEq,
        >,
    TracedOutput: Parameterized<
            LinearizationTracer<'domain, D>,
            Family: ParameterizedFamily<<D as Domain>::Value> + ParameterizedFamily<D::Tangent>,
        >,
    DirectLinearOperationOf<D>: BatchableOperation<Tangent<ArrayType, D::Tangent>>,
    LinearOperationOf<D>: ResidualizedOperation<D>,
{
    jvp_with_context::<D, D, F, Input, TracedOutput>(
        DifferentiableStorage::Borrowed(domain),
        primals,
        tangents,
        TangentMode::direct_batched(lane_count),
        function,
    )
}

/// A [`Context`] that additionally supports automatic differentiation, and the single entry point for its
/// forward- and reverse-mode transforms.
///
/// Implementors supply the differentiation hooks: a tangent value type ([`Tangent`](Self::Tangent)), the linear
/// operation type used by pushforward and pullback programs ([`LinearOperation`](Self::LinearOperation)), how to
/// synthesize a canonical zero tangent ([`zero_tangent`](Self::zero_tangent)), and primal validation
/// ([`validate_primal`](Self::validate_primal)). Everything else an AD pass needs — the primal
/// value/constant/operation types, applying an operation (`bind`), and lifting a constant (`lift`) — comes from the
/// underlying [`Context`]/[`Domain`]. On top of those hooks the trait provides the user-facing transforms
/// [`linearize`](Self::linearize), [`jvp`](Self::jvp), [`vjp`](Self::vjp), and
/// [`value_and_gradient`](Self::value_and_gradient).
///
/// Both eager backends (e.g. an `ndarray` domain, whose value is concrete) and staging contexts ([`TracingContext`],
/// batching contexts) implement it; whether a transform runs eagerly or stages a program is decided by the context's
/// [`Domain::Value`] (concrete vs [`Tracer`]), not by a separate trait.
pub trait DifferentiationContext: Context {
    /// Tangent value type staged in the active linear program.
    type Tangent: Value<Self::Type>;

    /// Linear operation type specialized to the tangent and factor representations used by a transform context.
    type LinearOperation<V: Value<Self::Type>, F: Value<Self::Type>>: Clone + Operation<Self::Type>;

    /// Returns the canonical zero tangent for `type_`.
    fn zero_tangent(&self, type_: &Self::Type) -> Result<Self::Tangent, ProgramError>;

    /// Validates that `primal` may be used as a primal input to an automatic-differentiation entry point in this
    /// context. Eager contexts accept any concrete value and use the default no-op. Staging contexts override this
    /// to verify that the input [`Tracer`] belongs to this context's
    /// [`ProgramBuilder`](crate::programs::ProgramBuilder), rejecting tracers that escaped a different trace with
    /// [`ProgramError::MismatchedProgramBuilders`].
    #[inline]
    fn validate_primal(&self, primal: &Self::Value) -> Result<(), ProgramError> {
        let _ = primal;
        Ok(())
    }

    /// Executes `function` once through an active linearization context and returns the traced primal output plus a
    /// reusable pushforward program over tangent leaves from this same context.
    fn linearize<'context, F, Input, TracedOutput>(
        &self,
        function: F,
        primals: Input,
    ) -> Result<
        LinearizedProgram<
            Self,
            TracedOutput::To<<Self as Domain>::Value>,
            Input::To<Self::Tangent>,
            TracedOutput::To<Self::Tangent>,
        >,
        ProgramError,
    >
    where
        Self: 'context,
        <Self as Domain>::Operation: DifferentiableOperation<Self>,
        F: FnOnce(Input::To<Tracer<LinearizationContext<'context, Self, Self>>>) -> Result<TracedOutput, ProgramError>,
        Input: Parameterized<
                <Self as Domain>::Value,
                To<<Self as Domain>::Value> = Input,
                Family: ParameterizedFamily<Tracer<LinearizationContext<'context, Self, Self>>>
                            + ParameterizedFamily<Self::Tangent>,
                ParameterStructure: Debug + PartialEq,
            >,
        TracedOutput: Parameterized<
                Tracer<LinearizationContext<'context, Self, Self>>,
                Family: ParameterizedFamily<<Self as Domain>::Value> + ParameterizedFamily<Self::Tangent>,
            >,
        LinearOperationOf<Self>: ResidualizedOperation<Self>,
    {
        if primals.parameters().next().is_none() {
            return Err(ProgramError::InvalidInputCount { expected: 1, got: 0 }.into());
        }
        for primal in primals.parameters() {
            self.validate_primal(primal)?;
        }
        let context = self.clone();
        linearize_with_context::<Self, Self, F, Input, TracedOutput>(
            DifferentiableStorage::Owned(Rc::new(context)),
            primals,
            function,
        )
    }

    /// Evaluates `function` on already-traced primal values and propagates traced tangent values forward.
    fn jvp<'context, F, Input, TracedOutput>(
        &self,
        function: F,
        primals: Input,
        tangents: Input::To<Self::Tangent>,
    ) -> Result<(TracedOutput::To<<Self as Domain>::Value>, TracedOutput::To<Self::Tangent>), ProgramError>
    where
        Self: 'context,
        <Self as Domain>::Operation: DifferentiableOperation<Self>,
        DirectLinearOperationOf<Self>: InterpretableOperation<<Self as Domain>::Type, Self::Tangent>,
        LinearOperationOf<Self>: ResidualizedOperation<Self>,
        F: FnOnce(Input::To<Tracer<LinearizationContext<'context, Self, Self>>>) -> TracedOutput,
        Input: Parameterized<
                <Self as Domain>::Value,
                To<<Self as Domain>::Value> = Input,
                Family: ParameterizedFamily<Tracer<LinearizationContext<'context, Self, Self>>>
                            + ParameterizedFamily<Self::Tangent>,
                ParameterStructure: Debug + PartialEq,
            >,
        TracedOutput: Parameterized<
                Tracer<LinearizationContext<'context, Self, Self>>,
                Family: ParameterizedFamily<<Self as Domain>::Value> + ParameterizedFamily<Self::Tangent>,
            >,
    {
        if primals.parameters().next().is_none() {
            return Err(ProgramError::InvalidInputCount { expected: 1, got: 0 }.into());
        }
        for primal in primals.parameters() {
            self.validate_primal(primal)?;
        }

        let context = self.clone();
        jvp_with_context::<Self, Self, _, Input, TracedOutput>(
            DifferentiableStorage::Owned(Rc::new(context)),
            primals,
            tangents,
            TangentMode::direct(),
            |input| Ok(function(input)),
        )
    }

    /// Transposes a linear program whose values are tangent leaves in this active context.
    fn transpose_linear_program<Input, Output, O>(
        &self,
        program: &Program<<Self as Domain>::Type, Self::Tangent, O, Input, Output>,
    ) -> Result<Program<<Self as Domain>::Type, Self::Tangent, O, Output, Input>, ProgramError>
    where
        O: SupportsTransposition<<Self as Domain>::Type, Self::Tangent>,
        Input: Parameterized<Self::Tangent>,
        Output: Parameterized<Self::Tangent>,
    {
        let builder = Rc::new(RefCell::new(ProgramBuilder::<<Self as Domain>::Type, Self::Tangent, O>::new()));
        let domain = AbstractDomain::new();
        let mut context = TracingContext::new(&domain, builder);
        context.transpose_with_zero_fn(
            program,
            Some(
                |builder: &mut ProgramBuilder<<Self as Domain>::Type, Self::Tangent, O>,
                 r#type: &<Self as Domain>::Type| {
                    Ok(builder.add_constant(self.zero_tangent(r#type)?))
                },
            ),
        )
    }

    /// Returns the traced primal output and a traced pullback program by transposing the active pushforward.
    fn vjp<'context, F, Input, TracedOutput>(
        &self,
        function: F,
        primals: Input,
    ) -> Result<
        (
            TracedOutput::To<<Self as Domain>::Value>,
            Program<
                <Self as Domain>::Type,
                Self::Tangent,
                DirectLinearOperationOf<Self>,
                TracedOutput::To<Self::Tangent>,
                Input::To<Self::Tangent>,
            >,
        ),
        ProgramError,
    >
    where
        Self: 'context,
        <Self as Domain>::Operation: DifferentiableOperation<Self>,
        DirectLinearOperationOf<Self>: SupportsTransposition<<Self as Domain>::Type, Self::Tangent>,
        LinearOperationOf<Self>: ResidualizedOperation<Self>,
        F: FnOnce(Input::To<Tracer<LinearizationContext<'context, Self, Self>>>) -> Result<TracedOutput, ProgramError>,
        Input: Parameterized<
                <Self as Domain>::Value,
                To<<Self as Domain>::Value> = Input,
                Family: ParameterizedFamily<Tracer<LinearizationContext<'context, Self, Self>>>
                            + ParameterizedFamily<Self::Tangent>,
                ParameterStructure: Debug + PartialEq,
            >,
        TracedOutput: Parameterized<
                Tracer<LinearizationContext<'context, Self, Self>>,
                Family: ParameterizedFamily<<Self as Domain>::Value> + ParameterizedFamily<Self::Tangent>,
            >,
    {
        let linearized = self.linearize(function, primals)?;
        let (output, pushforward) = linearized.into_parts();
        let pullback = self.transpose_linear_program(&pushforward.instantiate_program()?)?;
        Ok((output, pullback))
    }

    /// Returns the traced scalar output and reverse-mode gradient for `function`.
    ///
    /// This is the active-context counterpart of [`crate::tracing_v2::value_and_grad`]. It uses
    /// [`DifferentiationContext::vjp`] directly, so nested reverse mode composes with any enclosing context that
    /// implements this trait instead of going through a separate tracer dispatch path.
    fn value_and_grad<'context, F, Input>(
        &self,
        function: F,
        primals: Input,
    ) -> Result<(<Self as Domain>::Value, Input::To<Self::Tangent>), DifferentiationError>
    where
        Self: 'context + DifferentiationContext<Tangent = <Self as Domain>::Value>,
        <Self as Domain>::Operation: DifferentiableOperation<Self>,
        <Self as Domain>::Operation: SupportsOne<<Self as Domain>::Type>,
        DirectLinearOperationOf<Self>: InterpretableOperation<<Self as Domain>::Type, Self::Tangent>
            + SupportsTransposition<<Self as Domain>::Type, Self::Tangent>,
        LinearOperationOf<Self>: ResidualizedOperation<Self>,
        F: FnOnce(
            Input::To<Tracer<LinearizationContext<'context, Self, Self>>>,
        ) -> Tracer<LinearizationContext<'context, Self, Self>>,
        Input: Parameterized<
                <Self as Domain>::Value,
                To<<Self as Domain>::Value> = Input,
                Family: ParameterizedFamily<Tracer<LinearizationContext<'context, Self, Self>>>
                            + ParameterizedFamily<Self::Tangent>,
                ParameterStructure: Debug + PartialEq,
            >,
    {
        let (output, pullback) = self.vjp(|input| Ok(function(input)), primals)?;
        // Reverse mode only defines a gradient for scalar-output functions; reject non-scalar outputs before
        // seeding (see `DifferentiationError::NonScalarGradientOutput`).
        if !output.r#type().is_scalar() {
            return Err(DifferentiationError::NonScalarGradientOutput { output_type: output.r#type().to_string() });
        }
        // Seed the cotangent with the multiplicative identity of the scalar output, staged through `bind`.
        let one_operation = <<Self as Domain>::Operation as SupportsOne<<Self as Domain>::Type>>::one_operation(
            output.r#type().into_owned(),
        );
        let mut seeds = self.bind(one_operation, &[])?;
        check_count!("output", seeds, 1, ProgramError);
        let seed = seeds.pop().expect("exactly one output checked above");
        Ok((output, pullback.interpret(seed)?))
    }

    /// Returns the reverse-mode gradient of a traced scalar-output function.
    #[inline]
    fn value_and_gradient<'context, F, Input>(
        &self,
        function: F,
        primals: Input,
    ) -> Result<Input::To<Self::Tangent>, DifferentiationError>
    where
        Self: 'context + DifferentiationContext<Tangent = <Self as Domain>::Value>,
        <Self as Domain>::Operation: DifferentiableOperation<Self>,
        <Self as Domain>::Operation: SupportsOne<<Self as Domain>::Type>,
        DirectLinearOperationOf<Self>: InterpretableOperation<<Self as Domain>::Type, Self::Tangent>
            + SupportsTransposition<<Self as Domain>::Type, Self::Tangent>,
        LinearOperationOf<Self>: ResidualizedOperation<Self>,
        F: FnOnce(
            Input::To<Tracer<LinearizationContext<'context, Self, Self>>>,
        ) -> Tracer<LinearizationContext<'context, Self, Self>>,
        Input: Parameterized<
                <Self as Domain>::Value,
                To<<Self as Domain>::Value> = Input,
                Family: ParameterizedFamily<Tracer<LinearizationContext<'context, Self, Self>>>
                            + ParameterizedFamily<Self::Tangent>,
                ParameterStructure: Debug + PartialEq,
            >,
    {
        self.value_and_grad(function, primals).map(|(_, gradient)| gradient)
    }

    /// Converts a staged primal [`Program`] into a staged pushforward linear map.
    ///
    /// This is the reusable IR-level form of forward-mode differentiation. It replays the primal program through JVP
    /// rules once, returning both the primal program output at `input_primals` and a staged [`Program`] over linear
    /// operations that can be replayed later on arbitrary tangent inputs at the same primal point.
    ///
    /// # Parameters
    ///
    ///   - `program`: Staged primal program to linearize.
    ///   - `input_primals`: Concrete primal values aligned with the program's input atoms.
    fn linearize_program<O, Input, Output>(
        &self,
        program: &Program<<Self as Domain>::Type, <Self as Domain>::Constant, O, Input, Output>,
        input_primals: Vec<<Self as Domain>::Value>,
    ) -> Result<
        (Output::To<<Self as Domain>::Value>, Pushforward<Self, Input::To<Self::Tangent>, Output::To<Self::Tangent>>),
        ProgramError,
    >
    where
        O: DifferentiableOperation<Self>,
        Input: Parameterized<<Self as Domain>::Constant, Family: ParameterizedFamily<Self::Tangent>>,
        Output: Parameterized<
                <Self as Domain>::Constant,
                Family: ParameterizedFamily<<Self as Domain>::Value> + ParameterizedFamily<Self::Tangent>,
            >,
        LinearOperationOf<Self>: ResidualizedOperation<Self>,
    {
        fn tangent_for_atom<'jvp, D>(
            primal_values: &[Option<<D as Domain>::Value>],
            tangents: &[Option<Tangent<<D as Domain>::Type, Tracer<TangentContext<'jvp, D>>>>],
            atom_id: AtomId,
        ) -> Result<Tangent<<D as Domain>::Type, Tracer<TangentContext<'jvp, D>>>, ProgramError>
        where
            D: DifferentiationContext,
        {
            if let Some(tangent) = &tangents[atom_id.index()] {
                return Ok(tangent.clone());
            }
            // Atoms that are not connected to an input tangent are structurally zero. Carry that as a symbolic
            // `Tangent::Zero` so downstream JVP rules can short-circuit; the linearize loop materializes a concrete
            // zero atom only at the program output boundary.
            let primal = primal_values[atom_id.index()].as_ref().ok_or(ProgramError::UnboundAtomId { id: atom_id })?;
            Ok(Tangent::Zero(primal.r#type().into_owned()))
        }

        check_count!("input", input_primals, program.input_ids().len(), ProgramError);
        let builder =
            Rc::new(RefCell::new(
                ProgramBuilder::<<Self as Domain>::Type, Self::Tangent, LinearOperationOf<Self>>::new(),
            ));
        let residuals = Rc::new(RefCell::new(Vec::new()));
        let residual_atoms = Rc::new(RefCell::new(HashMap::new()));
        // Keep every tracer and context that holds a clone of `builder` inside this scope. Only raw output atom IDs
        // escape, making `Rc::try_unwrap(builder)` below a real ownership check instead of depending on manual drops.
        let (output_primal_values, output_tangent_atoms) = {
            let mut primal_values: Vec<Option<<Self as Domain>::Value>> = vec![None; program.atoms().len()];
            let mut tangent_values: Vec<Option<Tangent<<Self as Domain>::Type, Tracer<TangentContext<'_, Self>>>>> =
                vec![None; program.atoms().len()];
            let mut context =
                TangentContext::new_with_residuals(self, builder.clone(), residuals.clone(), residual_atoms.clone());

            // Program inputs become linear-program inputs. Their concrete primal values are kept in parallel so JVP
            // rules can evaluate primal semantics while staging tangent operations.
            for (input_atom, input_primal) in program.input_ids().iter().copied().zip(input_primals.into_iter()) {
                let tangent = context.input(input_primal.r#type().into_owned());
                tangent_values[input_atom.index()] = Some(Tangent::Value(tangent));
                primal_values[input_atom.index()] = Some(input_primal);
            }
            // Constants already have primal values in the original program. Their tangents are derived lazily by
            // `tangent_for_atom` as `Tangent::Zero(type)`, propagating through JVP rules until they meet a non-zero
            // tangent that forces materialization.
            for (atom_index, atom) in program.atoms().iter().enumerate() {
                if let Atom::Constant(value) = atom {
                    primal_values[atom_index] = Some(self.lift(value.clone())?);
                }
            }

            // Replay each primal instruction in JVP form. The rule returns both the concrete primal result and a
            // (possibly symbolic) `Tangent`, which becomes the state for the instruction's output atoms.
            for instruction in program.instructions() {
                let input_duals = instruction
                    .inputs()
                    .iter()
                    .copied()
                    .map(|input_atom| {
                        let residual_atom = match program.atoms().get(input_atom.index()) {
                            Some(Atom::Variable(_)) => Some(input_atom),
                            Some(Atom::Constant(_)) => None,
                            None => return Err(ProgramError::UnboundAtomId { id: input_atom }.into()),
                        };
                        Ok(JvpTracer::new_with_residual_atom(
                            primal_values[input_atom.index()]
                                .clone()
                                .ok_or(ProgramError::UnboundAtomId { id: input_atom })?,
                            tangent_for_atom::<Self>(primal_values.as_slice(), tangent_values.as_slice(), input_atom)?,
                            residual_atom,
                        ))
                    })
                    .collect::<Result<Vec<_>, ProgramError>>()?;
                let output_duals = instruction.operation().jvp(&mut context, input_duals.as_slice())?;
                check_count!("output", output_duals, instruction.outputs().len(), ProgramError);
                for (output_atom, output_dual) in instruction.outputs().iter().copied().zip(output_duals.into_iter()) {
                    let (primal, tangent) = output_dual.into_parts();
                    primal_values[output_atom.index()] = Some(primal);
                    tangent_values[output_atom.index()] = Some(tangent);
                }
            }

            // Materialize tangents for the requested program outputs and retain the matching primal outputs. The
            // temporary tracers created here must not outlive this scope. A `Tangent::Zero` output is staged as a typed
            // zero constant on the linear builder so the resulting program has a concrete atom for every output.
            let mut output_remaining_uses = vec![0usize; program.atoms().len()];
            for output_atom in program.output_ids().iter().copied() {
                output_remaining_uses[output_atom.index()] += 1;
            }
            let mut output_primal_values = Vec::with_capacity(program.output_ids().len());
            let mut output_tangent_atoms = Vec::with_capacity(program.output_ids().len());
            for output_atom in program.output_ids().iter().copied() {
                let tangent =
                    tangent_for_atom::<Self>(primal_values.as_slice(), tangent_values.as_slice(), output_atom)?;
                let tangent_atom = context.materialize_tangent(tangent)?.atom_id()?;

                let remaining_uses = &mut output_remaining_uses[output_atom.index()];
                debug_assert!(*remaining_uses > 0);
                *remaining_uses -= 1;
                let primal = if *remaining_uses == 0 {
                    primal_values[output_atom.index()].take().ok_or(ProgramError::UnboundAtomId { id: output_atom })?
                } else {
                    primal_values[output_atom.index()]
                        .as_ref()
                        .ok_or(ProgramError::UnboundAtomId { id: output_atom })?
                        .clone()
                };
                output_primal_values.push(primal);
                output_tangent_atoms.push(tangent_atom);
            }
            (output_primal_values, output_tangent_atoms)
        };
        // At this point all tracing handles are out of scope, so the builder can be recovered and finalized.
        let builder = match Rc::try_unwrap(builder) {
            Ok(builder) => builder.into_inner(),
            Err(_) => {
                return Err(ProgramError::EscapedProgramBuilder);
            }
        };
        let pushforward = builder
            .build(output_tangent_atoms, program.input_structure().clone(), program.output_structure().clone())?
            .simplified()?;
        Ok((
            Output::To::<<Self as Domain>::Value>::from_parameters(
                program.output_structure().clone(),
                output_primal_values,
            )?,
            Pushforward::new(residuals.borrow().clone(), pushforward).compact_residuals()?,
        ))
    }
}

impl<'domain, D, Capture> DifferentiationContext for TracingContext<'domain, D, Capture>
where
    D: DifferentiationContext + Domain + 'domain,
    <D as Domain>::Operation: SupportsZero<<D as Domain>::Type> + SupportsOne<<D as Domain>::Type>,
{
    type Tangent = Tracer<TracingContext<'domain, D, Capture>>;
    type LinearOperation<V: Value<<D as Domain>::Type>, F: Value<<D as Domain>::Type>> =
        <D as DifferentiationContext>::LinearOperation<V, F>;

    #[inline]
    fn zero_tangent(&self, type_: &Self::Type) -> Result<Self::Tangent, ProgramError> {
        let outputs = self.stage_operation(
            <<D as Domain>::Operation as SupportsZero<<D as Domain>::Type>>::zero_operation(type_.clone()),
            &[] as &[Tracer<TracingContext<'domain, D, Capture>>],
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.into_iter().next().expect("checked above"))
    }

    #[inline]
    fn validate_primal(&self, primal: &Self::Value) -> Result<(), ProgramError> {
        if std::rc::Rc::ptr_eq(self.builder(), primal.context().builder()) {
            Ok(())
        } else {
            Err(self.error(ProgramError::MismatchedProgramBuilders))
        }
    }
}

/// Operation-level contract for forward-mode Jacobian-Vector Product (JVP) staging.
///
/// A [`DifferentiableOperation`] is keyed by the [`DifferentiationContext`] implementation that supplies the value,
/// type, and linear operation type used while differentiating. Implementors consume
/// [`JvpTracer`] inputs, each carrying a primal value and a tangent atom in the active linear
/// builder, and return traced primal/tangent outputs.
///
/// Primitive rules usually stage tangent operations through [`TangentContext::stage_operation`].
/// Higher-order rules use [`TangentContext::differentiable`] to recurse into nested programs with the same
/// [`DifferentiationContext`] implementation.
pub trait DifferentiableOperation<E: DifferentiationContext>: Operation<E::Type> {
    /// Applies this operation's forward-mode Jacobian-Vector Product (JVP) rule.
    ///
    /// The returned vector must be aligned with this operation's outputs and must carry both the
    /// primal output values and the staged tangent atoms for those outputs.
    ///
    /// # Parameters
    ///
    ///   - `context`: Active tangent context used to stage tangent operations and access the
    ///     [`DifferentiationContext`] implementation.
    ///   - `inputs`: Traced inputs aligned with this operation's inputs.
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, E>,
        inputs: &[JvpTracer<'jvp, E>],
    ) -> Result<Vec<JvpTracer<'jvp, E>>, ProgramError>
    where
        E: 'jvp;
}

/// Function used by direct JVP mode to interpret one tangent operation.
type DirectJvpStageFn<'domain, E> = dyn Fn(
        &Rc<RefCell<ProgramBuilder<<E as Domain>::Type, <E as DifferentiationContext>::Tangent, LinearOperationOf<E>>>>,
        &Rc<RefCell<Vec<Option<<E as DifferentiationContext>::Tangent>>>>,
        &[<E as Domain>::Value],
        LinearOperationOf<E>,
        &[AtomId],
    ) -> Result<Vec<(AtomId, <E as Domain>::Type)>, ProgramError>
    + 'domain;

/// Function used by direct JVP mode to materialize a structural zero tangent.
type DirectJvpZeroFn<'domain, E> =
    dyn Fn(&E, &<E as Domain>::Type) -> Result<<E as DifferentiationContext>::Tangent, ProgramError> + 'domain;

/// State carried by direct JVP mode.
struct DirectTangentState<'domain, E: DifferentiationContext> {
    /// Concrete tangent values keyed by tangent-side atom id.
    tangent_values: Rc<RefCell<Vec<Option<E::Tangent>>>>,

    /// Tangent operation interpreter specialized when direct mode is constructed.
    stage_operation: Rc<DirectJvpStageFn<'domain, E>>,

    /// Structural-zero materializer specialized for this direct mode.
    zero_tangent: Rc<DirectJvpZeroFn<'domain, E>>,
}

impl<'domain, E: DifferentiationContext> Clone for DirectTangentState<'domain, E> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            tangent_values: self.tangent_values.clone(),
            stage_operation: self.stage_operation.clone(),
            zero_tangent: self.zero_tangent.clone(),
        }
    }
}

/// Tangent execution mode used by [`TangentContext`].
enum TangentMode<'domain, E: DifferentiationContext> {
    /// Stage tangent operations into a reusable linear program.
    Staged,

    /// Interpret tangent operations immediately and store concrete tangent values by atom id.
    Direct(DirectTangentState<'domain, E>),
}

impl<'domain, E: DifferentiationContext> Clone for TangentMode<'domain, E> {
    #[inline]
    fn clone(&self) -> Self {
        match self {
            Self::Staged => Self::Staged,
            Self::Direct(state) => Self::Direct(state.clone()),
        }
    }
}

impl<'domain, E> TangentMode<'domain, E>
where
    E: DifferentiationContext + 'domain,
{
    /// Creates a direct JVP mode.
    fn direct() -> Self
    where
        DirectLinearOperationOf<E>: InterpretableOperation<E::Type, E::Tangent>,
        LinearOperationOf<E>: ResidualizedOperation<E>,
    {
        let stage_operation = Rc::new(
            |builder: &Rc<RefCell<ProgramBuilder<E::Type, E::Tangent, LinearOperationOf<E>>>>,
             tangent_values: &Rc<RefCell<Vec<Option<E::Tangent>>>>,
             residuals: &[E::Value],
             operation: LinearOperationOf<E>,
             input_atoms: &[AtomId]|
             -> Result<Vec<(AtomId, E::Type)>, ProgramError> {
                let input_values = {
                    let tangent_values = tangent_values.borrow();
                    input_atoms
                        .iter()
                        .copied()
                        .map(|atom| {
                            tangent_values
                                .get(atom.index())
                                .and_then(Option::as_ref)
                                .cloned()
                                .ok_or(ProgramError::UnboundAtomId { id: atom })
                        })
                        .collect::<Result<Vec<_>, _>>()?
                };
                let operation = operation.instantiate_residuals(residuals)?;
                let output_values = operation.interpret(input_values.as_slice())?;

                let mut builder = builder.borrow_mut();
                let mut tangent_values = tangent_values.borrow_mut();
                let mut output_atoms = Vec::with_capacity(output_values.len());
                for output_value in output_values {
                    let output_type = output_value.r#type().into_owned();
                    let output_atom = builder.add_variable(output_type.clone());
                    let capacity = output_atom.index() + 1;
                    if tangent_values.len() < capacity {
                        tangent_values.resize_with(capacity, || None);
                    }
                    tangent_values[output_atom.index()] = Some(output_value);
                    output_atoms.push((output_atom, output_type));
                }
                Ok(output_atoms)
            },
        );
        let zero_tangent = Rc::new(|differentiable: &E, r#type: &E::Type| differentiable.zero_tangent(r#type));
        Self::Direct(DirectTangentState {
            tangent_values: Rc::new(RefCell::new(Vec::new())),
            stage_operation,
            zero_tangent,
        })
    }

    /// Creates a direct JVP mode that executes tangent operations through batching rules.
    fn direct_batched(lane_count: usize) -> Self
    where
        E: DifferentiationContext<Type = ArrayType>,
        E::Tangent: Zero<ArrayType> + BroadcastInDim + Transpose,
        DirectLinearOperationOf<E>: BatchableOperation<Tangent<ArrayType, E::Tangent>>,
        LinearOperationOf<E>: ResidualizedOperation<E>,
    {
        let stage_operation = Rc::new(
            move |builder: &Rc<RefCell<ProgramBuilder<ArrayType, E::Tangent, LinearOperationOf<E>>>>,
                  tangent_values: &Rc<RefCell<Vec<Option<E::Tangent>>>>,
                  residuals: &[E::Value],
                  operation: LinearOperationOf<E>,
                  input_atoms: &[AtomId]|
                  -> Result<Vec<(AtomId, ArrayType)>, ProgramError> {
                let input_values = {
                    let tangent_values = tangent_values.borrow();
                    input_atoms
                        .iter()
                        .copied()
                        .map(|atom| {
                            tangent_values
                                .get(atom.index())
                                .and_then(Option::as_ref)
                                .cloned()
                                .ok_or(ProgramError::UnboundAtomId { id: atom })
                        })
                        .collect::<Result<Vec<_>, _>>()?
                };
                let input_batches = input_values
                    .into_iter()
                    .map(|value| ArrayBatch::mapped(Tangent::Value(value), 0))
                    .collect::<Result<Vec<_>, _>>()?;
                let operation = operation.instantiate_residuals(residuals)?;
                let output_batches = BatchableOperation::<Tangent<ArrayType, E::Tangent>>::batch(
                    &operation,
                    &(),
                    input_batches.as_slice(),
                )?;

                let mut builder = builder.borrow_mut();
                let mut tangent_values = tangent_values.borrow_mut();
                let mut output_atoms = Vec::with_capacity(output_batches.len());
                for output_batch in output_batches {
                    let output_batch = match output_batch.batch_axis() {
                        Some(0) => output_batch,
                        Some(_) => align_batch_axis(&output_batch, 0)?,
                        None => broadcast_to_batched(&output_batch, 0, lane_count)?,
                    };
                    let output_type = output_batch.r#type().into_owned();
                    let output_value = match output_batch.into_value() {
                        Tangent::Zero(r#type) => E::Tangent::zero(&r#type)?,
                        Tangent::Value(value) => value,
                    };
                    let output_atom = builder.add_variable(output_type.clone());
                    let capacity = output_atom.index() + 1;
                    if tangent_values.len() < capacity {
                        tangent_values.resize_with(capacity, || None);
                    }
                    tangent_values[output_atom.index()] = Some(output_value);
                    output_atoms.push((output_atom, output_type));
                }
                Ok(output_atoms)
            },
        );
        let zero_tangent = Rc::new(move |differentiable: &E, r#type: &ArrayType| {
            let physical_type = r#type.with_inserted_dimension(0, Size::Static(lane_count))?;
            differentiable.zero_tangent(&physical_type)
        });
        Self::Direct(DirectTangentState {
            tangent_values: Rc::new(RefCell::new(Vec::new())),
            stage_operation,
            zero_tangent,
        })
    }

    /// Returns whether this mode interprets tangent operations immediately.
    #[inline]
    fn is_direct(&self) -> bool {
        matches!(self, Self::Direct(_))
    }

    /// Records a concrete tangent value for `atom` when this is direct mode.
    fn record_tangent(&self, atom: AtomId, tangent: E::Tangent) {
        let Self::Direct(state) = self else {
            return;
        };
        let capacity = atom.index() + 1;
        let mut tangent_values = state.tangent_values.borrow_mut();
        if tangent_values.len() < capacity {
            tangent_values.resize_with(capacity, || None);
        }
        tangent_values[atom.index()] = Some(tangent);
    }

    /// Returns the concrete tangent value for `atom` in direct mode.
    fn tangent_value(&self, atom: AtomId) -> Result<E::Tangent, ProgramError> {
        let Self::Direct(state) = self else {
            return Err(ProgramError::UnboundAtomId { id: atom }.into());
        };
        state
            .tangent_values
            .borrow()
            .get(atom.index())
            .and_then(Option::as_ref)
            .cloned()
            .ok_or(ProgramError::UnboundAtomId { id: atom }.into())
    }

    /// Materializes a structural zero tangent for this mode.
    fn zero_tangent(&self, differentiable: &E, r#type: &E::Type) -> Result<E::Tangent, ProgramError> {
        match self {
            Self::Staged => differentiable.zero_tangent(r#type),
            Self::Direct(state) => (state.zero_tangent)(differentiable, r#type),
        }
    }

    /// Interprets a tangent operation in direct mode, returning `None` in staged mode.
    fn stage_operation(
        &self,
        builder: &Rc<RefCell<ProgramBuilder<E::Type, E::Tangent, LinearOperationOf<E>>>>,
        residuals: &[E::Value],
        operation: LinearOperationOf<E>,
        input_atoms: &[AtomId],
    ) -> Option<Result<Vec<(AtomId, E::Type)>, ProgramError>> {
        let Self::Direct(state) = self else {
            return None;
        };
        Some((state.stage_operation)(builder, &state.tangent_values, residuals, operation, input_atoms))
    }
}

/// Concrete state threaded through forward-mode JVP rules.
///
/// [`TangentContext`] owns the active linear-program builder where tangent ops are staged. It is itself a
/// [`Context`], so tangent tracers are ordinary [`Tracer`] leaves whose context is this tangent context. JVP rules
/// call [`stage_operation`](Self::stage_operation) to stage tangent ops and
/// [`differentiable`](Self::differentiable) to access primal constants or recursively linearize nested programs.
#[doc(hidden)]
pub struct TangentContext<'domain, E: DifferentiationContext> {
    /// [`DifferentiationContext`] implementation borrowed by this [`TangentContext`] for primal semantics.
    differentiable: &'domain E,

    /// [`ProgramBuilder`] that owns the staged linear [`Program`](crate::programs::Program) that is currently being
    /// traced.
    builder: Rc<RefCell<ProgramBuilder<E::Type, E::Tangent, LinearOperationOf<E>>>>,

    /// Residual values captured by product-rule factors in this tangent context.
    residuals: Rc<RefCell<Vec<E::Value>>>,

    /// Residual indices keyed by primal atom id.
    residual_atoms: Rc<RefCell<HashMap<AtomId, usize>>>,

    /// Tangent execution mode.
    mode: TangentMode<'domain, E>,
}

impl<'domain, E: DifferentiationContext> TangentContext<'domain, E> {
    /// Creates a tangent context that stages into `builder`.
    #[doc(hidden)]
    pub fn new(
        differentiable: &'domain E,
        builder: Rc<RefCell<ProgramBuilder<E::Type, E::Tangent, LinearOperationOf<E>>>>,
    ) -> Self {
        Self::new_with_residuals(
            differentiable,
            builder,
            Rc::new(RefCell::new(Vec::new())),
            Rc::new(RefCell::new(HashMap::new())),
        )
    }

    /// Creates a tangent context that shares residual storage with an enclosing linearization context.
    #[doc(hidden)]
    pub(crate) fn new_with_residuals(
        differentiable: &'domain E,
        builder: Rc<RefCell<ProgramBuilder<E::Type, E::Tangent, LinearOperationOf<E>>>>,
        residuals: Rc<RefCell<Vec<E::Value>>>,
        residual_atoms: Rc<RefCell<HashMap<AtomId, usize>>>,
    ) -> Self {
        Self { differentiable, builder, residuals, residual_atoms, mode: TangentMode::Staged }
    }

    /// Creates a tangent context with an explicit tangent execution mode.
    fn new_with_mode(
        differentiable: &'domain E,
        builder: Rc<RefCell<ProgramBuilder<E::Type, E::Tangent, LinearOperationOf<E>>>>,
        residuals: Rc<RefCell<Vec<E::Value>>>,
        residual_atoms: Rc<RefCell<HashMap<AtomId, usize>>>,
        mode: TangentMode<'domain, E>,
    ) -> Self {
        Self { differentiable, builder, residuals, residual_atoms, mode }
    }

    /// Returns the [`DifferentiationContext`] implementation borrowed by this tangent context.
    #[inline]
    pub fn differentiable(&self) -> &'domain E {
        self.differentiable
    }

    /// Produces the primal output(s) of `operation` applied to the primal `inputs` in this context.
    ///
    /// Convenience delegation to [`Context::bind`] on the borrowed implementation, used by nullary-operation JVP
    /// rules to stage their primal without reaching through [`differentiable`](Self::differentiable) explicitly.
    #[inline]
    pub fn bind_primal(&self, operation: E::Operation, inputs: &[E::Value]) -> Result<Vec<E::Value>, ProgramError> {
        self.differentiable.bind(operation, inputs)
    }

    /// Materializes a [`Tangent`] into a tracer owned by this tangent context.
    ///
    /// Structural zeros carry only type metadata. When a nested linear program needs an actual
    /// input atom, this method stages the differentiable implementation's canonical zero tangent in the active linear
    /// builder. Non-zero tangents are returned unchanged.
    pub fn materialize_tangent(
        &self,
        tangent: Tangent<E::Type, Tracer<TangentContext<'domain, E>>>,
    ) -> Result<Tracer<TangentContext<'domain, E>>, ProgramError> {
        match tangent {
            Tangent::Zero(r#type) => Ok(self.constant(self.mode.zero_tangent(self.differentiable, &r#type)?)),
            Tangent::Value(tracer) => Ok(tracer),
        }
    }

    /// Captures `value` as an anonymous residual factor.
    pub fn factor(&mut self, value: E::Value) -> ResidualFactor<E::Type, E::Value> {
        if self.mode.is_direct() {
            return ResidualFactor::Constant(value);
        }
        let r#type = value.r#type().into_owned();
        let mut residuals = self.residuals.borrow_mut();
        let index = residuals.len();
        residuals.push(value);
        ResidualFactor::Reference { index, r#type }
    }

    /// Captures `value` as a residual factor, deduplicating by `atom` when one is available.
    fn factor_for_atom(&mut self, atom: AtomId, value: E::Value) -> ResidualFactor<E::Type, E::Value> {
        if self.mode.is_direct() {
            return ResidualFactor::Constant(value);
        }
        let r#type = value.r#type().into_owned();
        if let Some(index) = self.residual_atoms.borrow().get(&atom).copied() {
            return ResidualFactor::Reference { index, r#type };
        }
        let mut residuals = self.residuals.borrow_mut();
        let index = residuals.len();
        residuals.push(value);
        self.residual_atoms.borrow_mut().insert(atom, index);
        ResidualFactor::Reference { index, r#type }
    }
}

impl<'domain, E: DifferentiationContext> Clone for TangentContext<'domain, E> {
    fn clone(&self) -> Self {
        Self {
            differentiable: self.differentiable,
            builder: self.builder.clone(),
            residuals: self.residuals.clone(),
            residual_atoms: self.residual_atoms.clone(),
            mode: self.mode.clone(),
        }
    }
}

impl<'domain, E: DifferentiationContext> Debug for TangentContext<'domain, E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("TangentContext").finish_non_exhaustive()
    }
}

impl<'domain, E: DifferentiationContext> Domain for TangentContext<'domain, E> {
    type Type = E::Type;
    type Value = Tracer<Self>;
    type Constant = E::Tangent;
    type Operation = LinearOperationOf<E>;
}

impl<'domain, E: DifferentiationContext> Context for TangentContext<'domain, E> {
    /// Lifts a tangent payload into this tangent context by recording it as a constant tangent [`Tracer`].
    #[inline]
    fn lift(&self, constant: E::Tangent) -> Result<Tracer<Self>, ProgramError> {
        Ok(self.constant(constant))
    }

    /// Binding in a tangent context stages the linear (tangent) operation into the active linear program.
    #[inline]
    fn bind(&self, operation: Self::Operation, inputs: &[Self::Value]) -> Result<Vec<Self::Value>, ProgramError> {
        self.stage_operation(operation, inputs)
    }
}

impl<'domain, E: DifferentiationContext> StagingContext for TangentContext<'domain, E> {
    #[inline]
    fn builder(&self) -> &Rc<RefCell<ProgramBuilder<Self::Type, Self::Constant, Self::Operation>>> {
        &self.builder
    }

    #[inline]
    fn constant(&self, value: Self::Constant) -> Tracer<Self> {
        let r#type = value.r#type().into_owned();
        let atom = self.builder().borrow_mut().add_constant(value.clone());
        self.mode.record_tangent(atom, value);
        self.tracer(atom, Some(r#type))
    }

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
            return Ok(output_types
                .into_iter()
                .map(|r#type| Tracer::new(TracerState::Poison, r#type, self.clone()))
                .collect());
        }

        let input_atom_ids = match inputs.iter().map(|input| input.borrow().atom_id()).collect::<Result<Vec<_>, _>>() {
            Ok(input_atom_ids) => input_atom_ids,
            Err(error) => return Err(self.error(error)),
        };
        if let Some(output_atom_ids) = self.mode.stage_operation(
            self.builder(),
            self.residuals.borrow().as_slice(),
            operation.clone(),
            input_atom_ids.as_slice(),
        ) {
            return Ok(output_atom_ids?.into_iter().map(|(atom, r#type)| self.tracer(atom, Some(r#type))).collect());
        }

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

/// Forward-mode JVP tracer carrying both a primal and a [`Tangent`].
///
/// [`JvpTracer`] is the value wrapper primitive operations see while a function is evaluated in JVP mode. The `primal`
/// field carries the usual runtime value, while the `tangent` field carries the directional derivative information
/// flowing alongside it as a [`Tangent`]: either a structural [`Tangent::Zero`] with no atom staged on the linear
/// program, or a concrete [`Tangent::Value`] wrapping a tangent atom. Encoding the [`Tangent`] in the type makes the
/// symbolic-zero state part of the JVP rule contract. Rules pattern-match on the tangent variant, and the [`Tangent`]
/// arithmetic impls in [`crate::differentiation::tangent`] propagate `Zero` short-circuits through `+`, `-`, unary
/// negation, and `.scale(_)` without per-rule bookkeeping.
pub struct JvpTracer<'domain, E: DifferentiationContext> {
    /// The primal value.
    primal: E::Value,

    /// The tangent associated with the primal, possibly structurally zero.
    tangent: Tangent<E::Type, Tracer<TangentContext<'domain, E>>>,

    /// Primal atom that can be used to deduplicate residual factors for this value.
    residual_atom: Option<AtomId>,
}

impl<'domain, E> Clone for JvpTracer<'domain, E>
where
    E: DifferentiationContext,
{
    #[inline]
    fn clone(&self) -> Self {
        Self { primal: self.primal.clone(), tangent: self.tangent.clone(), residual_atom: self.residual_atom }
    }
}

impl<'domain, E> Debug for JvpTracer<'domain, E>
where
    E: DifferentiationContext,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("JvpTracer")
            .field("primal", &self.primal)
            .field("tangent", &self.tangent)
            .field("residual_atom", &self.residual_atom)
            .finish()
    }
}

impl<'domain, E: DifferentiationContext> Parameter for JvpTracer<'domain, E> {}

impl<'domain, E> JvpTracer<'domain, E>
where
    E: DifferentiationContext + 'domain,
{
    /// Constructs a [`JvpTracer`] from an explicit primal value and [`Tangent`].
    #[inline]
    pub fn new(primal: E::Value, tangent: Tangent<E::Type, Tracer<TangentContext<'domain, E>>>) -> Self {
        Self { primal, tangent, residual_atom: None }
    }

    /// Constructs a [`JvpTracer`] with an optional primal atom used for residual deduplication.
    #[inline]
    fn new_with_residual_atom(
        primal: E::Value,
        tangent: Tangent<E::Type, Tracer<TangentContext<'domain, E>>>,
        residual_atom: Option<AtomId>,
    ) -> Self {
        Self { primal, tangent, residual_atom }
    }

    /// Constructs a [`JvpTracer`] with a concrete [`Tangent::Value`] tangent.
    #[inline]
    pub fn from_value(primal: E::Value, tangent_value: Tracer<TangentContext<'domain, E>>) -> Self {
        Self { primal, tangent: Tangent::Value(tangent_value), residual_atom: None }
    }

    /// Constructs a [`JvpTracer`] with a structurally-zero [`Tangent::Zero`] tangent carrying the
    /// provided tangent type.
    #[inline]
    pub fn from_zero_tangent(primal: E::Value, tangent_type: E::Type) -> Self {
        Self { primal, tangent: Tangent::Zero(tangent_type), residual_atom: None }
    }

    /// Returns the primal value carried by this JVP tracer.
    #[inline]
    pub fn primal(&self) -> &E::Value {
        &self.primal
    }

    /// Returns the tangent carried by this JVP tracer.
    #[inline]
    pub fn tangent(&self) -> &Tangent<E::Type, Tracer<TangentContext<'domain, E>>> {
        &self.tangent
    }

    /// Returns this tracer's primal value as a residual factor.
    #[inline]
    pub fn factor(&self, context: &mut TangentContext<'domain, E>) -> ResidualFactor<E::Type, E::Value> {
        match self.residual_atom {
            Some(atom) => context.factor_for_atom(atom, self.primal.clone()),
            None => ResidualFactor::Constant(self.primal.clone()),
        }
    }

    /// Consumes this JVP tracer and returns its primal and tangent components.
    #[inline]
    pub fn into_parts(self) -> (E::Value, Tangent<E::Type, Tracer<TangentContext<'domain, E>>>) {
        (self.primal, self.tangent)
    }
}

impl<'domain, E: DifferentiationContext> Typed<E::Type> for JvpTracer<'domain, E> {
    #[inline]
    fn r#type(&self) -> Cow<'_, E::Type> {
        self.primal.r#type()
    }
}

impl<'domain, E: DifferentiationContext> Display for JvpTracer<'domain, E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.primal, formatter)
    }
}

impl<'domain, E: DifferentiationContext> Value<E::Type> for JvpTracer<'domain, E> {}

impl<'parent, C, D> LinearizationContext<'parent, C, D>
where
    C: Context + 'parent,
    D: DifferentiationContext<Type = C::Type, Constant = C::Constant> + 'parent,
{
    /// Creates a new active linearization context from prepared differentiable storage.
    #[inline]
    fn new_with_differentiable(
        differentiable: DifferentiableStorage<'parent, D>,
        primal_builder: Rc<RefCell<ProgramBuilder<C::Type, C::Constant, C::Operation>>>,
        linear_builder: Rc<RefCell<ProgramBuilder<C::Type, D::Tangent, LinearOperationOf<D>>>>,
    ) -> Self {
        Self {
            differentiable,
            primal_builder,
            linear_builder,
            primal_values: Rc::new(RefCell::new(Vec::new())),
            tangent_atoms: Rc::new(RefCell::new(Vec::new())),
            residuals: Rc::new(RefCell::new(Vec::new())),
            residual_atoms: Rc::new(RefCell::new(HashMap::new())),
            jvp_mode: TangentMode::Staged,
        }
    }

    /// Creates a new active linearization context with an explicit tangent execution mode.
    #[inline]
    fn new_with_jvp_mode(
        differentiable: DifferentiableStorage<'parent, D>,
        primal_builder: Rc<RefCell<ProgramBuilder<C::Type, C::Constant, C::Operation>>>,
        linear_builder: Rc<RefCell<ProgramBuilder<C::Type, D::Tangent, LinearOperationOf<D>>>>,
        jvp_mode: TangentMode<'parent, D>,
    ) -> Self {
        Self {
            differentiable,
            primal_builder,
            linear_builder,
            primal_values: Rc::new(RefCell::new(Vec::new())),
            tangent_atoms: Rc::new(RefCell::new(Vec::new())),
            residuals: Rc::new(RefCell::new(Vec::new())),
            residual_atoms: Rc::new(RefCell::new(HashMap::new())),
            jvp_mode,
        }
    }

    /// Registers an input atom together with its concrete primal and matching tangent input atom.
    fn register_input(&self, atom: AtomId, primal: D::Value, tangent: AtomId) {
        self.ensure_atom_capacity(atom);
        self.primal_values.borrow_mut()[atom.index()] = Some(primal);
        self.tangent_atoms.borrow_mut()[atom.index()] = Some(tangent);
    }

    /// Ensures that all per-atom state tables can address `atom`.
    fn ensure_atom_capacity(&self, atom: AtomId) {
        let capacity = atom.index() + 1;
        {
            let mut primals = self.primal_values.borrow_mut();
            if primals.len() < capacity {
                primals.resize_with(capacity, || None);
            }
        }
        let mut tangents = self.tangent_atoms.borrow_mut();
        if tangents.len() < capacity {
            tangents.resize_with(capacity, || None);
        }
    }

    /// Returns the stored primal for `atom`, lazily registering primal constants when needed.
    fn primal_for_atom(&self, atom: AtomId) -> Result<D::Value, ProgramError> {
        self.ensure_atom_capacity(atom);
        if let Some(primal) = &self.primal_values.borrow()[atom.index()] {
            return Ok(primal.clone());
        }
        let constant = {
            let builder = self.primal_builder.borrow();
            match builder.atoms().get(atom.index()) {
                Some(Atom::Constant(value)) => Some(value.clone()),
                Some(Atom::Variable(_)) => None,
                None => return Err(ProgramError::UnboundAtomId { id: atom }.into()),
            }
        };
        let Some(constant) = constant else {
            return Err(ProgramError::UnboundAtomId { id: atom }.into());
        };
        let primal = self.differentiable.as_ref().lift(constant)?;
        self.primal_values.borrow_mut()[atom.index()] = Some(primal.clone());
        Ok(primal)
    }

    /// Returns the stored tangent for `atom`, materialized as a tracer in `context`.
    fn tangent_for_atom<'jvp>(
        &self,
        context: &TangentContext<'jvp, D>,
        atom: AtomId,
    ) -> Result<Tangent<C::Type, Tracer<TangentContext<'jvp, D>>>, ProgramError>
    where
        D: 'jvp,
    {
        self.ensure_atom_capacity(atom);
        if let Some(tangent_atom) = self.tangent_atoms.borrow()[atom.index()] {
            return Ok(Tangent::Value(context.tracer(tangent_atom, None)));
        }
        Ok(Tangent::Zero(self.primal_for_atom(atom)?.r#type().into_owned()))
    }

    /// Returns the residual atom for `atom`, if this atom should be residualized by reference.
    fn residual_atom_for_atom(&self, atom: AtomId) -> Result<Option<AtomId>, ProgramError> {
        let builder = self.primal_builder.borrow();
        match builder.atoms().get(atom.index()) {
            Some(Atom::Variable(_)) => Ok(Some(atom)),
            Some(Atom::Constant(_)) => Ok(None),
            None => Err(ProgramError::UnboundAtomId { id: atom }.into()),
        }
    }

    /// Creates a [`TangentContext`] that shares this linearization context's builder and residual tables.
    fn tangent_context<'jvp>(&'jvp self) -> TangentContext<'jvp, D> {
        TangentContext::new_with_mode(
            self.differentiable.as_ref(),
            self.linear_builder.clone(),
            self.residuals.clone(),
            self.residual_atoms.clone(),
            self.jvp_mode.clone(),
        )
    }

    /// Returns the [`DifferentiationContext`] implementation this linearization context drives. Crate-visible for the
    /// [`CapturingContext`](crate::compilation::context::CapturingContext) implementation, which delegates capture
    /// registration to it.
    #[inline]
    pub(crate) fn differentiable(&self) -> &D {
        self.differentiable.as_ref()
    }

    /// Collects concrete primal outputs and linear-program output atom ids.
    fn collect_outputs(&self, output_atoms: &[AtomId]) -> Result<(Vec<D::Value>, Vec<AtomId>), ProgramError> {
        let context = self.tangent_context();
        let mut output_primals = Vec::with_capacity(output_atoms.len());
        let mut output_tangents = Vec::with_capacity(output_atoms.len());
        for output_atom in output_atoms.iter().copied() {
            let primal = self.primal_for_atom(output_atom)?;
            let tangent = self.tangent_for_atom(&context, output_atom)?;
            let tangent_atom = context.materialize_tangent(tangent)?.atom_id()?;
            output_primals.push(primal);
            output_tangents.push(tangent_atom);
        }
        Ok((output_primals, output_tangents))
    }

    /// Collects concrete primal and tangent outputs from a direct JVP pass.
    fn collect_direct_outputs(
        &self,
        output_atoms: &[AtomId],
    ) -> Result<(Vec<D::Value>, Vec<D::Tangent>), ProgramError> {
        let context = self.tangent_context();
        let mut output_primals = Vec::with_capacity(output_atoms.len());
        let mut output_tangents = Vec::with_capacity(output_atoms.len());
        for output_atom in output_atoms.iter().copied() {
            let primal = self.primal_for_atom(output_atom)?;
            let tangent = self.tangent_for_atom(&context, output_atom)?;
            let tangent_atom = context.materialize_tangent(tangent)?.atom_id()?;
            output_primals.push(primal);
            output_tangents.push(context.mode.tangent_value(tangent_atom)?);
        }
        Ok((output_primals, output_tangents))
    }
}

impl<'domain, C, D> Clone for LinearizationContext<'domain, C, D>
where
    C: Context + 'domain,
    D: DifferentiationContext<Type = C::Type, Constant = C::Constant> + 'domain,
{
    fn clone(&self) -> Self {
        Self {
            differentiable: self.differentiable.clone(),
            primal_builder: self.primal_builder.clone(),
            linear_builder: self.linear_builder.clone(),
            primal_values: self.primal_values.clone(),
            tangent_atoms: self.tangent_atoms.clone(),
            residuals: self.residuals.clone(),
            residual_atoms: self.residual_atoms.clone(),
            jvp_mode: self.jvp_mode.clone(),
        }
    }
}

impl<'parent, C, D> Domain for LinearizationContext<'parent, C, D>
where
    C: Context + 'parent,
    D: DifferentiationContext<Type = C::Type, Constant = C::Constant> + 'parent,
    C::Operation: DifferentiableOperation<D>,
{
    type Type = C::Type;
    type Value = Tracer<Self>;
    type Constant = C::Constant;
    type Operation = C::Operation;
}

impl<'parent, C, D> Context for LinearizationContext<'parent, C, D>
where
    C: Context + 'parent,
    D: DifferentiationContext<Type = C::Type, Constant = C::Constant> + 'parent,
    C::Operation: DifferentiableOperation<D>,
{
    /// Lifts a constant payload into this linearization context by recording it as a constant primal [`Tracer`].
    #[inline]
    fn lift(&self, constant: C::Constant) -> Result<Tracer<Self>, ProgramError> {
        Ok(self.constant(constant))
    }

    /// Binding in a linearization context stages the primal operation while threading its forward-mode rule into the
    /// active pushforward program.
    #[inline]
    fn bind(&self, operation: Self::Operation, inputs: &[Self::Value]) -> Result<Vec<Self::Value>, ProgramError> {
        self.stage_operation(operation, inputs)
    }
}

impl<'parent, C, D> StagingContext for LinearizationContext<'parent, C, D>
where
    C: Context + 'parent,
    D: DifferentiationContext<Type = C::Type, Constant = C::Constant> + 'parent,
    C::Operation: DifferentiableOperation<D>,
{
    #[inline]
    fn builder(&self) -> &Rc<RefCell<ProgramBuilder<Self::Type, Self::Constant, Self::Operation>>> {
        &self.primal_builder
    }

    fn stage_operation<I: std::borrow::Borrow<Tracer<Self>>>(
        &self,
        operation: Self::Operation,
        inputs: &[I],
    ) -> Result<Vec<Tracer<Self>>, ProgramError> {
        if inputs.iter().any(|input| !Rc::ptr_eq(self.builder(), input.borrow().context().builder())) {
            return Err(self.error(ProgramError::MismatchedProgramBuilders));
        }
        if self.builder().borrow().error().is_some() {
            let input_types = inputs.iter().map(|input| input.borrow().r#type().into_owned()).collect::<Vec<_>>();
            let output_types = operation.infer_output_types(input_types.as_slice())?;
            return Ok(output_types
                .into_iter()
                .map(|r#type| Tracer::new(TracerState::Poison, r#type, self.clone()))
                .collect());
        }

        let input_atoms = inputs
            .iter()
            .map(|input| input.borrow().atom_id())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| self.error(error))?;
        let input_types = inputs.iter().map(|input| input.borrow().r#type().into_owned()).collect::<Vec<_>>();
        let output_types = operation.infer_output_types(input_types.as_slice())?;
        let mut tangent_context = self.tangent_context();
        let input_duals = input_atoms
            .iter()
            .copied()
            .map(|atom| {
                Ok(JvpTracer::new_with_residual_atom(
                    self.primal_for_atom(atom)?,
                    self.tangent_for_atom(&tangent_context, atom)?,
                    self.residual_atom_for_atom(atom)?,
                ))
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;
        let output_duals = operation.jvp(&mut tangent_context, input_duals.as_slice())?;
        check_count!("output", output_duals, output_types.len(), ProgramError);

        let mut output_tracers = Vec::with_capacity(output_duals.len());
        let mut primal_builder = self.builder().borrow_mut();
        for (output_dual, output_type) in output_duals.into_iter().zip(output_types.into_iter()) {
            let (primal, tangent) = output_dual.into_parts();
            let atom = primal_builder.add_variable(output_type.clone());
            self.ensure_atom_capacity(atom);
            self.primal_values.borrow_mut()[atom.index()] = Some(primal);
            self.tangent_atoms.borrow_mut()[atom.index()] = match tangent {
                Tangent::Zero(_) => None,
                Tangent::Value(tracer) => Some(tracer.atom_id()?),
            };
            output_tracers.push(self.tracer(atom, Some(output_type)));
        }
        Ok(output_tracers)
    }
}

impl<S: Value<DataType>> DifferentiationContext for ScalarDomain<S>
where
    ScalarDomain<S>: Context
        + Domain<
            Type = DataType,
            Value = S,
            Constant = S,
            Operation: Clone + InterpretableOperation<DataType, S> + SupportsZero<DataType>,
        >,
{
    type Tangent = S;
    type LinearOperation<V: Value<DataType>, F: Value<DataType>> = LinearScalarOperation<S, F>;

    #[inline]
    fn zero_tangent(&self, type_: &DataType) -> Result<Self::Tangent, ProgramError> {
        let mut outputs = self.bind(SupportsZero::zero_operation(type_.clone()), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.pop().expect("zero operation produces exactly one output"))
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use half::{bf16, f16};
    use pretty_assertions::assert_eq;

    use crate::differentiation::Tangent;
    use crate::operations::constants::{One, Zero, ZeroLike};
    use crate::scalars::ScalarDomain;
    use crate::types::{ArrayType, DataType, Shape, Size, Typed};

    use super::DifferentiationContext;

    #[test]
    fn test_tangent_value_carries_symbolic_zero_or_value_tangent() {
        let zero = Tangent::<DataType, f64>::zero(DataType::F64);
        let value = Tangent::<DataType, f64>::value(2.5);

        assert!(zero.is_zero());
        assert_eq!(zero.as_value(), None);
        assert_eq!(zero.r#type().into_owned(), DataType::F64);
        assert_eq!(zero.to_string(), "Zero[f64]");
        assert_eq!(<Tangent<DataType, f64> as Zero<DataType>>::zero(&DataType::F64), Ok(zero.clone()));
        assert_eq!(value.as_value(), Some(&2.5));
        assert_eq!(value.r#type().into_owned(), DataType::F64);
        assert_eq!(value.to_string(), "2.5");
        assert_eq!(<Tangent<DataType, f64> as One<DataType>>::one(&DataType::F64), Ok(Tangent::value(1.0)));
        assert_eq!(value.zero_like(), zero);

        let zero_only = Tangent::<DataType, Infallible>::zero(DataType::I32);
        assert_eq!(zero_only.r#type().into_owned(), DataType::I32);
        assert_eq!(zero_only.to_string(), "Zero[i32]");
        assert_eq!(<Tangent<DataType, Infallible> as Zero<DataType>>::zero(&DataType::I32), Ok(zero_only.clone()));
        assert_eq!(zero_only.zero_like(), zero_only);

        let array_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(2)]), None, None).unwrap();
        let array_tangent = Tangent::<ArrayType, Infallible>::zero(array_type.clone());
        assert_eq!(array_tangent.r#type().into_owned(), array_type);
    }

    #[test]
    fn test_scalar_domain_half_and_float_domains_are_differentiable() {
        let _: Option<<ScalarDomain<bf16> as DifferentiationContext>::LinearOperation<bf16, bf16>> = None;
        let _: Option<<ScalarDomain<f16> as DifferentiationContext>::LinearOperation<f16, f16>> = None;
        let _: Option<<ScalarDomain<f32> as DifferentiationContext>::LinearOperation<f32, f32>> = None;
        let _: Option<<ScalarDomain<f64> as DifferentiationContext>::LinearOperation<f64, f64>> = None;
    }

    #[test]
    fn test_scalar_domain_half_domains_run_jvp() {
        let bf16_domain = ScalarDomain::<bf16>::new();
        assert_eq!(
            bf16_domain.jvp(|x| x.clone() + x, bf16::from_f32(3.0), bf16::ONE),
            Ok((bf16::from_f32(6.0), bf16::from_f32(2.0)))
        );

        let f16_domain = ScalarDomain::<f16>::new();
        assert_eq!(
            f16_domain.jvp(|x| x.clone() + x, f16::from_f32(3.0), f16::ONE),
            Ok((f16::from_f32(6.0), f16::from_f32(2.0)))
        );
    }
}
