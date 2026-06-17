use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::rc::Rc;

use ryft_macros::Parameter;
use thiserror::Error;

use crate::contexts::{Context, StagingContext};
use crate::differentiation::{SupportsTransposition, Tangent};
use crate::domains::{AbstractDomain, Domain};
use crate::macros::check_count;
use crate::operations::constants::{OneOperation, ZeroOperation};
use crate::operations::control_flow::SelectCondition;
use crate::operations::scalars::LinearScalarOperation;
use crate::operations::{BooleanLike, InterpretableOperation, Operation};
use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily, Placeholder};
use crate::programs::{Atom, AtomId, Program, ProgramBuilder, ProgramError, Value};
use crate::scalars::ScalarDomain;
use crate::tracing::{Tracer, TracerState, TracingContext};
use crate::types::{DataType, Type, Typed};

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

    /// Reference to a primal residual saved by the owning [`Pushforward`].
    Reference {
        /// Zero-based residual index inside the owning [`Pushforward`].
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

// TODO(eaplatanios): Why do we need this? Also, we should move it to `select.rs`.
/// Scalar captured-condition factors carry the [`SelectOperation`](crate::operations::control_flow::SelectOperation)
/// condition as an in-band Boolean over a [`DataType`] value, so the linear select interprets them by decoding that
/// Boolean. References are residuals of the primal computation and must be instantiated before interpretation, so the
/// reference form errors here, matching [`CustomVjpResidual::residual_value`](crate::tracing_v2::operations::CustomVjpResidual).
impl<V: Value<DataType> + BooleanLike> SelectCondition for ResidualFactor<DataType, V> {
    type Condition = bool;

    fn select_condition(&self) -> Result<bool, ProgramError> {
        match self {
            Self::Constant(value) => value.boolean(),
            Self::Reference { .. } => Err(ProgramError::Concretization {
                message: "captured select condition requires instantiated residuals".to_string(),
            }),
        }
    }
}

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
///
/// The [`From<ZeroOperation>`](ZeroOperation) supertrait records that every residualized linear operation can build a
/// nullary zero. This is what lets [`materialize_tangent`](TangentContext::materialize_tangent) stage a structural zero
/// tangent as an operation while requiring only that conversion (rather than the whole of [`ResidualizedOperation`]) at
/// each per-operation JVP rule, and keeps the public forward-mode entry points (which already require this trait) free
/// of an extra bound. It is declared on this residual-aware specialization — implemented only by whole linear-operation
/// algebras — rather than on [`FactorParameterizedOperation`], which component backend extensions that have no
/// standalone zero also implement.
pub trait ResidualizedOperation<D: DifferentiationContext>:
    FactorParameterizedOperation<
        D::Type,
        ResidualFactor<D::Type, D::Value>,
        WithFactor<D::Value> = DirectLinearOperationOf<D>,
        WithFactor<ResidualFactor<D::Type, D::Value>> = LinearOperationOf<D>,
    > + From<ZeroOperation<D::Type>>
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
}

impl<D, O> ResidualizedOperation<D> for O
where
    D: DifferentiationContext,
    O: FactorParameterizedOperation<
            D::Type,
            ResidualFactor<D::Type, D::Value>,
            WithFactor<D::Value> = DirectLinearOperationOf<D>,
            WithFactor<ResidualFactor<D::Type, D::Value>> = LinearOperationOf<D>,
        > + From<ZeroOperation<D::Type>>,
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

/// Tracer leaf passed to user closures by the program-level forward-mode entry points
/// ([`DifferentiationContext::linearize`], [`DifferentiationContext::jvp`], [`DifferentiationContext::vjp`], and the
/// forward/reverse Jacobian helpers built on them). All of them trace the closure into a primal [`Program`] through a
/// [`PrimalTracingContext`] before differentiating it symbolically, so every closure sees the same primal-staging
/// tracer. The `'domain` lifetime parameter is unused (the tracer owns a clone of the enclosing context rather than
/// borrowing it) and retained only so that annotated closure signatures across the crate keep compiling.
pub type LinearizationTracer<'domain, D> = Tracer<PrimalTracingContext<D>>;

/// Plain primal-staging [`Context`] used by [`DifferentiationContext::linearize`] to trace a user closure into a
/// primal [`Program`] over the enclosing context's types before linearizing that program symbolically.
///
/// This context stages every bound operation as an ordinary program instruction, exactly like
/// [`TracingContext`], but it is the *nesting* trace context for differentiation — the direct analog of
/// [`BatchingContext`](crate::tracing_v2::batching::BatchingContext) on the batching side. Where [`TracingContext`]
/// is a *root* trace context (it borrows a backend [`Domain`](crate::domains::Domain) and owns its own capture
/// table), this context is keyed by the *enclosing* [`Context`] `E`: it owns a clone of `E` (cheap, since contexts
/// are [`Rc`]-based) and delegates runtime-capture registration through
/// [`CapturingContext`](crate::compilation::context::CapturingContext) to that enclosing context rather than owning
/// a capture table of its own.
///
/// It cannot simply be a [`TracingContext`] for three reasons: (1) [`TracingContext`]'s [`CapturingContext`] impl is
/// table-owning and pinned to a [`CapturedConstant`](crate::compilation::captures::CapturedConstant) domain, so it
/// cannot propagate captures into an arbitrary enclosing [`CapturingContext`]; (2) a parent-delegating
/// [`CapturingContext`] impl for [`TracingContext`] would collide with that table-owning impl under coherence
/// (there is no negative reasoning to disjoin the two domains); and (3) [`TracingContext`] borrows its domain, which
/// would thread a borrow lifetime through every closure signature, whereas owning the clone keeps the tracer leaf
/// lifetime-free. No differentiation work happens while the closure runs; the traced program is differentiated
/// afterwards — symbolically by [`linearize_program`] for [`linearize`](DifferentiationContext::linearize) and the
/// reverse-mode entry points, or through the value-level JVP replay for [`jvp`](DifferentiationContext::jvp).
pub struct PrimalTracingContext<E: Context> {
    /// Enclosing [`Context`] on whose behalf the closure is being traced.
    parent: E,

    /// [`ProgramBuilder`] that owns the staged primal [`Program`].
    builder: Rc<RefCell<ProgramBuilder<E::Type, E::Constant, E::Operation>>>,
}

impl<E: Context> PrimalTracingContext<E> {
    /// Creates a new [`PrimalTracingContext`] that owns a fresh [`ProgramBuilder`] and traces on behalf of `parent`.
    fn new(parent: E) -> Self {
        Self { parent, builder: Rc::new(RefCell::new(ProgramBuilder::new())) }
    }

    /// Returns the enclosing [`Context`] this closure trace runs on behalf of.
    #[inline]
    pub(crate) fn parent(&self) -> &E {
        &self.parent
    }
}

impl<E: Context> Clone for PrimalTracingContext<E> {
    fn clone(&self) -> Self {
        Self { parent: self.parent.clone(), builder: self.builder.clone() }
    }
}

impl<E: Context> Debug for PrimalTracingContext<E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("PrimalTracingContext").finish_non_exhaustive()
    }
}

impl<E: Context> Domain for PrimalTracingContext<E> {
    type Type = E::Type;
    type Value = Tracer<Self>;
    type Constant = E::Constant;
    type Operation = E::Operation;
}

impl<E: Context> Context for PrimalTracingContext<E> {
    /// Lifts a constant payload into this context by recording it as a constant primal [`Tracer`].
    #[inline]
    fn lift(&self, constant: E::Constant) -> Result<Tracer<Self>, ProgramError> {
        Ok(self.constant(constant))
    }

    /// Binding stages the primal operation as an ordinary program instruction.
    #[inline]
    fn bind(&self, operation: Self::Operation, inputs: &[Self::Value]) -> Result<Vec<Self::Value>, ProgramError> {
        self.stage_operation(operation, inputs)
    }
}

impl<E: Context> StagingContext for PrimalTracingContext<E> {
    #[inline]
    fn builder(&self) -> &Rc<RefCell<ProgramBuilder<Self::Type, Self::Constant, Self::Operation>>> {
        &self.builder
    }
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

    // TODO(eaplatanios): Do we really need this function?
    /// Returns `true` if this context's primal values are concrete, so concretizing extractions such as
    /// [`BooleanLike::boolean`](crate::operations::BooleanLike::boolean) on primal values can succeed and the trip
    /// count of a data-dependent loop is decidable at rule time.
    ///
    /// Eager domains use the default. Staging contexts (whose primal values are [`Tracer`]s, and abstract domains
    /// whose primal values carry only type metadata) override this to return `false` so that higher-order rules —
    /// in particular the [`WhileOperation`](crate::operations::control_flow::WhileOperation) JVP rule — choose
    /// staged, non-concretizing strategies (the masked-scan or fused linear-loop paths) instead of eagerly unrolling
    /// a loop whose primal state cannot be evaluated.
    #[inline]
    fn supports_primal_concretization(&self) -> bool {
        true
    }

    // TODO(eaplatanios): Do we really need this function?
    /// Returns `true` if a structural zero tangent reaching a linear-program boundary should be materialized as a
    /// staged nullary zero **operation** rather than as a constant zero **value**.
    ///
    /// This is the JAX-analogous distinction between leaving a symbolic `zero` in the residual jaxpr versus
    /// [`instantiate_zeros`](https://docs.jax.dev/en/latest/jax.interpreters.ad.html) producing a concrete zero array:
    ///
    ///   - The default (`false`) synthesizes the zero **as a value** through [`zero_tangent`](Self::zero_tangent) — a
    ///     throwaway zero for one primal point that [`Pushforward::apply`] can clone even when the tangent leaves are
    ///     [`Tracer`]s. Eager domains and `jvp`-under-tracing use this: they replay the pushforward immediately, and a
    ///     bare nullary zero operation has no operand from which to recover an active builder.
    ///   - Returning `true` stages the zero **as an operation** into the pushforward program itself, keeping that
    ///     program self-contained and reusable at every primal point. The nested symbolic-linearization context
    ///     ([`LinearizationContext`]) overrides this to `true`: it has no concrete tangent values to synthesize and
    ///     returns the pushforward as reusable IR embedded into linear `condition`/`scan`/`while` bodies.
    ///
    /// The staged operation is built via `From<ZeroOperation>`, available because a pushforward's linear operation type
    /// is always a [`ResidualizedOperation`], which is bounded by [`From<ZeroOperation>`](ZeroOperation).
    #[inline]
    fn materializes_zero_tangents_as_operations(&self) -> bool {
        false
    }

    /// Traces `function` into a flat primal [`Program`] over this context's types.
    ///
    /// This is the shared tracing prologue of the program-level forward-mode entry points
    /// ([`linearize`](Self::linearize) and [`jvp`](Self::jvp)). The closure runs inside a [`PrimalTracingContext`]
    /// over this context, so runtime captures registered while tracing delegate to this context, and every operation
    /// is staged without running any differentiation rule. The traced program is then simplified so closure dead code
    /// is dropped before linearization. Returns the simplified program, the closure's output structure, and the
    /// primal input values aligned with the program's input atoms.
    ///
    /// # Parameters
    ///
    ///   - `function`: Closure traced into a primal program.
    ///   - `primals`: Structured primal input values; their count must be non-zero and each must belong to this
    ///     context (validated through [`validate_primal`](Self::validate_primal)).
    fn trace_into_primal_program<F, Input, TracedOutput>(
        &self,
        function: F,
        primals: Input,
    ) -> Result<
        (
            Program<
                <Self as Domain>::Type,
                <Self as Domain>::Constant,
                <Self as Domain>::Operation,
                Vec<<Self as Domain>::Constant>,
                Vec<<Self as Domain>::Constant>,
            >,
            Input::ParameterStructure,
            TracedOutput::ParameterStructure,
            Vec<<Self as Domain>::Value>,
        ),
        ProgramError,
    >
    where
        <Self as Domain>::Operation: Clone,
        F: FnOnce(Input::To<Tracer<PrimalTracingContext<Self>>>) -> Result<TracedOutput, ProgramError>,
        Input: Parameterized<
                <Self as Domain>::Value,
                Family: ParameterizedFamily<Tracer<PrimalTracingContext<Self>>>,
                ParameterStructure: Debug + PartialEq,
            >,
        TracedOutput: Parameterized<Tracer<PrimalTracingContext<Self>>>,
    {
        if primals.parameters().next().is_none() {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        }
        for primal in primals.parameters() {
            self.validate_primal(primal)?;
        }
        let input_structure = primals.parameter_structure();
        let input_values = primals.into_parameters().collect::<Vec<_>>();

        // Trace the closure into a flat primal program over this context's types. Tracing stages every operation
        // without running any differentiation rule; simplification then drops staged dead code so the JVP replay
        // below does not pay for it.
        let context = PrimalTracingContext::new(self.clone());
        let (output_structure, output_atoms) = {
            let input_tracers =
                input_values.iter().map(|value| context.input(value.r#type().into_owned())).collect::<Vec<_>>();
            let input = Input::To::<Tracer<PrimalTracingContext<Self>>>::from_parameters(
                input_structure.clone(),
                input_tracers,
            )?;
            let output =
                function(input).map_err(|error| context.builder().borrow_mut().error.take().unwrap_or(error))?;
            context.builder().borrow_mut().error.take().map_or(Ok(()), Err)?;
            let output_structure = output.parameter_structure();
            let output_atoms = output.parameters().map(Tracer::atom_id).collect::<Result<Vec<_>, _>>()?;
            (output_structure, output_atoms)
        };
        let PrimalTracingContext { parent: _, builder } = context;
        let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let output_count = output_atoms.len();
        let program = builder
            .build::<Vec<<Self as Domain>::Constant>, Vec<<Self as Domain>::Constant>>(
                output_atoms,
                vec![Placeholder; input_values.len()],
                vec![Placeholder; output_count],
            )?
            .into_simplified()?;
        Ok((program, input_structure, output_structure, input_values))
    }

    /// Executes `function` once and returns the primal output plus a reusable pushforward program over tangent
    /// leaves from this same context.
    ///
    /// The closure is first traced into a primal [`Program`] through a [`PrimalTracingContext`] over this context
    /// (so runtime captures registered while tracing delegate to this context), and that program is then linearized
    /// at `primals` via [`linearize_program`](Self::linearize_program). In an eager context the primal side
    /// therefore evaluates concretely here; in a staging context (whose values are [`Tracer`]s) it splices into the
    /// active trace.
    fn linearize<F, Input, TracedOutput>(
        &self,
        function: F,
        primals: Input,
    ) -> Result<
        (
            TracedOutput::To<<Self as Domain>::Value>,
            Pushforward<Self, Input::To<Self::Tangent>, TracedOutput::To<Self::Tangent>>,
        ),
        ProgramError,
    >
    where
        <Self as Domain>::Operation: Clone + DifferentiableOperation<Self> + ProgramLinearizableOperation<Self>,
        F: FnOnce(Input::To<Tracer<PrimalTracingContext<Self>>>) -> Result<TracedOutput, ProgramError>,
        Input: Parameterized<
                <Self as Domain>::Value,
                To<<Self as Domain>::Value> = Input,
                Family: ParameterizedFamily<Tracer<PrimalTracingContext<Self>>> + ParameterizedFamily<Self::Tangent>,
                ParameterStructure: Debug + PartialEq,
            >,
        TracedOutput: Parameterized<
                Tracer<PrimalTracingContext<Self>>,
                Family: ParameterizedFamily<<Self as Domain>::Value> + ParameterizedFamily<Self::Tangent>,
            >,
        LinearOperationOf<Self>: ResidualizedOperation<Self>,
    {
        let (program, input_structure, output_structure, input_values) =
            self.trace_into_primal_program::<F, Input, TracedOutput>(function, primals)?;

        // Linearize the traced program by replaying it through the JVP rules against this context: primal results
        // are bound through `self` (evaluated eagerly in concrete contexts, spliced into the active trace in staging
        // contexts) while tangent operations are staged into the reusable pushforward program.
        let (output_values, pushforward) = self.linearize_program(&program, input_values)?;
        let output =
            TracedOutput::To::<<Self as Domain>::Value>::from_parameters(output_structure.clone(), output_values)?;

        // Re-key the flat pushforward program onto the closure's input/output parameterizations; the structures are
        // family-invariant, so only the program's type-level packaging changes.
        let Pushforward { residuals, program: pushforward_program } = pushforward;
        let pushforward_program = Program {
            atoms: pushforward_program.atoms,
            input_ids: pushforward_program.input_ids,
            output_ids: pushforward_program.output_ids,
            instructions: pushforward_program.instructions,
            input_structure,
            output_structure,
            marker: PhantomData,
        };
        Ok((output, Pushforward::new(residuals, pushforward_program)))
    }

    /// Evaluates `function` on the primal `primals` and propagates the tangent `tangents` forward.
    ///
    /// This is the value-level forward-mode transform: the closure is traced into a primal [`Program`] through a
    /// [`PrimalTracingContext`] (the same prologue [`linearize`](Self::linearize) uses), that program is replayed
    /// through the JVP rules against this context at `primals` to produce the primal outputs and a reusable
    /// [`Pushforward`], and the tangent outputs are computed by applying the pushforward to `tangents`. In an eager
    /// context the primal side evaluates concretely; in a staging context (whose values are [`Tracer`]s) it splices
    /// into the active trace, and the tangent leaves likewise stage into that trace (jvp-under-tracing).
    fn jvp<F, Input, TracedOutput>(
        &self,
        function: F,
        primals: Input,
        tangents: Input::To<Self::Tangent>,
    ) -> Result<(TracedOutput::To<<Self as Domain>::Value>, TracedOutput::To<Self::Tangent>), ProgramError>
    where
        <Self as Domain>::Operation: Clone + DifferentiableOperation<Self>,
        DirectLinearOperationOf<Self>: InterpretableOperation<<Self as Domain>::Type, Self::Tangent>,
        LinearOperationOf<Self>: ResidualizedOperation<Self>,
        F: FnOnce(Input::To<Tracer<PrimalTracingContext<Self>>>) -> TracedOutput,
        Input: Parameterized<
                <Self as Domain>::Value,
                To<<Self as Domain>::Value> = Input,
                Family: ParameterizedFamily<Tracer<PrimalTracingContext<Self>>> + ParameterizedFamily<Self::Tangent>,
                ParameterStructure: Debug + PartialEq,
            >,
        Input::To<Self::Tangent>: Parameterized<Self::Tangent>,
        TracedOutput: Parameterized<
                Tracer<PrimalTracingContext<Self>>,
                Family: ParameterizedFamily<<Self as Domain>::Value> + ParameterizedFamily<Self::Tangent>,
            >,
    {
        let tangent_structure = tangents.parameter_structure();
        let primal_structure = primals.parameter_structure();
        let (program, _input_structure, output_structure, input_values) =
            self.trace_into_primal_program::<_, Input, TracedOutput>(|input| Ok(function(input)), primals)?;

        // Linearize the traced flat program at the primal point through the value-level replay. Because this context
        // concretizes primals and is not the nested symbolic-linearization context, structural zero tangents are
        // materialized as this context's canonical zero tangents — concrete values in an eager context, staged zero
        // tracers in a staging context — rather than as nullary linear zero *operations* (see
        // [`materializes_zero_tangents_as_operations`](Self::materializes_zero_tangents_as_operations)). This keeps
        // the resulting pushforward interpretable by [`Pushforward::apply`] below even when the tangent leaves are
        // themselves tracers (jvp-under-tracing), where a bare nullary zero operation has no active context to stage
        // into. The symbolic [`linearize_program`](Self::linearize_program) core is reserved for nested linearization.
        let (output_values, pushforward) = linearize_program_by_replay::<
            Self,
            <Self as Domain>::Operation,
            Vec<<Self as Domain>::Constant>,
            Vec<<Self as Domain>::Constant>,
        >(self, &program, input_values)?;
        let output =
            TracedOutput::To::<<Self as Domain>::Value>::from_parameters(output_structure.clone(), output_values)?;

        // Apply the pushforward to the flat tangent leaves. The pushforward's input structure mirrors the primal
        // input structure flattened to a `Vec`, so a structure check here surfaces a mismatched tangent up front
        // with the same error `Pushforward::apply` would otherwise raise against the flattened structure.
        if tangent_structure != primal_structure {
            return Err(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{primal_structure:?}"),
                right_structure: format!("{tangent_structure:?}"),
            }
            .into());
        }
        let tangent_values = tangents.into_parameters().collect::<Vec<_>>();
        let tangent_outputs = pushforward.apply(tangent_values)?;
        let tangent_outputs = TracedOutput::To::<Self::Tangent>::from_parameters(output_structure, tangent_outputs)?;
        Ok((output, tangent_outputs))
    }

    /// Transposes a linear program whose values are tangent leaves in this active context.
    fn transpose_linear_program<Input, Output, O>(
        &self,
        program: &Program<<Self as Domain>::Type, Self::Tangent, O, Input, Output>,
    ) -> Result<Program<<Self as Domain>::Type, Self::Tangent, O, Output, Input>, ProgramError>
    where
        O: SupportsTransposition<<Self as Domain>::Type, Self::Tangent>,
        for<'a> &'a ZeroOperation<<Self as Domain>::Type>: TryFrom<&'a O>,
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
    fn vjp<F, Input, TracedOutput>(
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
        <Self as Domain>::Operation: Clone + DifferentiableOperation<Self> + ProgramLinearizableOperation<Self>,
        DirectLinearOperationOf<Self>: SupportsTransposition<<Self as Domain>::Type, Self::Tangent>,
        for<'a> &'a ZeroOperation<<Self as Domain>::Type>: TryFrom<&'a DirectLinearOperationOf<Self>>,
        LinearOperationOf<Self>: ResidualizedOperation<Self>,
        F: FnOnce(Input::To<Tracer<PrimalTracingContext<Self>>>) -> Result<TracedOutput, ProgramError>,
        Input: Parameterized<
                <Self as Domain>::Value,
                To<<Self as Domain>::Value> = Input,
                Family: ParameterizedFamily<Tracer<PrimalTracingContext<Self>>> + ParameterizedFamily<Self::Tangent>,
                ParameterStructure: Debug + PartialEq,
            >,
        TracedOutput: Parameterized<
                Tracer<PrimalTracingContext<Self>>,
                Family: ParameterizedFamily<<Self as Domain>::Value> + ParameterizedFamily<Self::Tangent>,
            >,
    {
        let (output, pushforward) = self.linearize(function, primals)?;
        let pullback = self.transpose_linear_program(&pushforward.instantiate_program()?)?;
        Ok((output, pullback))
    }

    /// Returns the traced scalar output and reverse-mode gradient for `function`.
    ///
    /// This is the active-context counterpart of [`crate::tracing_v2::value_and_grad`]. It uses
    /// [`DifferentiationContext::vjp`] directly, so nested reverse mode composes with any enclosing context that
    /// implements this trait instead of going through a separate tracer dispatch path.
    fn value_and_grad<F, Input>(
        &self,
        function: F,
        primals: Input,
    ) -> Result<(<Self as Domain>::Value, Input::To<Self::Tangent>), DifferentiationError>
    where
        Self: DifferentiationContext<Tangent = <Self as Domain>::Value>,
        <Self as Domain>::Operation: Clone + DifferentiableOperation<Self> + ProgramLinearizableOperation<Self>,
        <Self as Domain>::Operation: From<OneOperation<<Self as Domain>::Type>>,
        DirectLinearOperationOf<Self>: InterpretableOperation<<Self as Domain>::Type, Self::Tangent>
            + SupportsTransposition<<Self as Domain>::Type, Self::Tangent>,
        for<'a> &'a ZeroOperation<<Self as Domain>::Type>: TryFrom<&'a DirectLinearOperationOf<Self>>,
        LinearOperationOf<Self>: ResidualizedOperation<Self>,
        F: FnOnce(Input::To<Tracer<PrimalTracingContext<Self>>>) -> Tracer<PrimalTracingContext<Self>>,
        Input: Parameterized<
                <Self as Domain>::Value,
                To<<Self as Domain>::Value> = Input,
                Family: ParameterizedFamily<Tracer<PrimalTracingContext<Self>>> + ParameterizedFamily<Self::Tangent>,
                ParameterStructure: Debug + PartialEq,
            >,
    {
        let (output, pullback) = self.vjp(|input| Ok(function(input)), primals)?;
        // Reverse mode only defines a gradient for scalar-output functions; reject non-scalar outputs before
        // seeding (see `DifferentiationError::NonScalarGradientOutput`).
        if !output.r#type().is_scalar() {
            return Err(DifferentiationError::NonScalarGradientOutput { output_type: output.r#type().to_string() });
        }
        // Seed the cotangent with the multiplicative identity of the scalar output, typed with the output's cotangent
        // type (e.g., swapping unreduced and reduced sharding axes for arrays) and staged through `bind`.
        let one_operation =
            <Self as Domain>::Operation::from(OneOperation::new(output.r#type().cotangent_type()));
        let mut seeds = self.bind(one_operation, &[])?;
        check_count!("output", seeds, 1, ProgramError);
        let seed = seeds.pop().unwrap();
        Ok((output, pullback.interpret(seed)?))
    }

    /// Returns the reverse-mode gradient of a traced scalar-output function.
    #[inline]
    fn value_and_gradient<F, Input>(
        &self,
        function: F,
        primals: Input,
    ) -> Result<Input::To<Self::Tangent>, DifferentiationError>
    where
        Self: DifferentiationContext<Tangent = <Self as Domain>::Value>,
        <Self as Domain>::Operation: Clone + DifferentiableOperation<Self> + ProgramLinearizableOperation<Self>,
        <Self as Domain>::Operation: From<OneOperation<<Self as Domain>::Type>>,
        DirectLinearOperationOf<Self>: InterpretableOperation<<Self as Domain>::Type, Self::Tangent>
            + SupportsTransposition<<Self as Domain>::Type, Self::Tangent>,
        for<'a> &'a ZeroOperation<<Self as Domain>::Type>: TryFrom<&'a DirectLinearOperationOf<Self>>,
        LinearOperationOf<Self>: ResidualizedOperation<Self>,
        F: FnOnce(Input::To<Tracer<PrimalTracingContext<Self>>>) -> Tracer<PrimalTracingContext<Self>>,
        Input: Parameterized<
                <Self as Domain>::Value,
                To<<Self as Domain>::Value> = Input,
                Family: ParameterizedFamily<Tracer<PrimalTracingContext<Self>>> + ParameterizedFamily<Self::Tangent>,
                ParameterStructure: Debug + PartialEq,
            >,
    {
        self.value_and_grad(function, primals).map(|(_, gradient)| gradient)
    }

    /// Converts a staged primal [`Program`] into a staged pushforward linear map.
    ///
    /// This is the reusable IR-level form of forward-mode differentiation. It picks one of two cores depending on
    /// whether this context concretizes primal values, decided by
    /// [`supports_primal_concretization`](Self::supports_primal_concretization):
    ///
    ///   - **Concretizing contexts** (eager domains, gate `true`) replay the program through the value-level JVP loop
    ///     ([`linearize_program_by_replay`]) with this context supplying concrete primal
    ///     values. Because the JVP rules see a concretizing context, data-dependent higher-order rules — in particular
    ///     the [`WhileOperation`](crate::operations::control_flow::WhileOperation) rule — take their eager strategy
    ///     (unrolling the loop into a straight-line, transposable pushforward), so eager reverse mode works.
    ///   - **Staging contexts** ([`TracingContext`], abstract domains, gate `false`) cannot concretize primals, so the
    ///     program is linearized *symbolically* once through [`linearize_program`] (via the
    ///     [`ProgramLinearizableOperation`] witness), producing a residual-extended primal program and a residualized
    ///     pushforward program, and the resulting [`Linearization`] is evaluated [`at`](Linearization::at)
    ///     `input_primals` against this context (splicing the primal program into the active trace).
    ///
    /// Both cores return the same residualized [`Pushforward`] at the same primal point, so callers replay or transpose
    /// it identically regardless of which core produced it.
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
        Self: Domain<Operation = O>,
        O: Clone + DifferentiableOperation<Self> + ProgramLinearizableOperation<Self>,
        Input: Parameterized<<Self as Domain>::Constant, Family: ParameterizedFamily<Self::Tangent>>,
        Output: Parameterized<
                <Self as Domain>::Constant,
                Family: ParameterizedFamily<<Self as Domain>::Value> + ParameterizedFamily<Self::Tangent>,
            >,
        LinearOperationOf<Self>: ResidualizedOperation<Self>,
    {
        // Concretizing contexts replay through the value-level JVP loop so data-dependent higher-order rules (the
        // `while` rule in particular) take their eager, transposable strategy; staging contexts fall back to the
        // symbolic core, which never concretizes primals.
        if self.supports_primal_concretization() {
            return linearize_program_by_replay::<Self, O, Input, Output>(self, program, input_primals);
        }
        let linearization = O::linearize_program(self, &program.to_flat_program())?;
        let (output_values, pushforward) = linearization.at(self, input_primals)?;
        let outputs =
            Output::To::<<Self as Domain>::Value>::from_parameters(program.output_structure().clone(), output_values)?;
        // Re-key the flat pushforward program onto the source program's input/output parameterizations; the
        // structures are family-invariant, so only the program's type-level packaging changes.
        let Pushforward { residuals, program: pushforward_program } = pushforward;
        let pushforward_program = Program {
            atoms: pushforward_program.atoms,
            input_ids: pushforward_program.input_ids,
            output_ids: pushforward_program.output_ids,
            instructions: pushforward_program.instructions,
            input_structure: program.input_structure().clone(),
            output_structure: program.output_structure().clone(),
            marker: PhantomData,
        };
        Ok((outputs, Pushforward::new(residuals, pushforward_program)))
    }
}

/// Converts a staged primal [`Program`] into a staged pushforward linear map by replaying it value by value.
///
/// This is the value-level forward-mode replay loop. [`DifferentiationContext::jvp`] and the eager branch of
/// [`linearize_program`](DifferentiationContext::linearize_program) run it against a concretizing context; the
/// symbolic core (the free [`linearize_program`]) runs it against a nested [`LinearizationContext`]. The contexts
/// differ only in how a structural zero tangent reaching a program boundary is realized by
/// [`TangentContext::materialize_tangent`] — a throwaway concrete zero value versus a staged nullary zero operation,
/// selected by
/// [`materializes_zero_tangents_as_operations`](DifferentiationContext::materializes_zero_tangents_as_operations).
fn linearize_program_by_replay<'context, E, O, Input, Output>(
    context: &'context E,
    program: &Program<<E as Domain>::Type, <E as Domain>::Constant, O, Input, Output>,
    input_primals: Vec<<E as Domain>::Value>,
) -> Result<
    (Output::To<<E as Domain>::Value>, Pushforward<E, Input::To<E::Tangent>, Output::To<E::Tangent>>),
    ProgramError,
>
where
    E: DifferentiationContext + Domain<Operation = O>,
    O: Clone + DifferentiableOperation<E>,
    Input: Parameterized<<E as Domain>::Constant, Family: ParameterizedFamily<E::Tangent>>,
    Output: Parameterized<
            <E as Domain>::Constant,
            Family: ParameterizedFamily<<E as Domain>::Value> + ParameterizedFamily<E::Tangent>,
        >,
    LinearOperationOf<E>: ResidualizedOperation<E>,
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
    let builder = Rc::new(RefCell::new(ProgramBuilder::<<E as Domain>::Type, E::Tangent, LinearOperationOf<E>>::new()));
    let residuals = Rc::new(RefCell::new(Vec::new()));
    let residual_atoms = Rc::new(RefCell::new(HashMap::new()));
    // Keep every tracer and context that holds a clone of `builder` inside this scope. Only raw output atom IDs
    // escape, making `Rc::try_unwrap(builder)` below a real ownership check instead of depending on manual drops.
    let (output_primal_values, output_tangent_atoms) = {
        let mut primal_values: Vec<Option<<E as Domain>::Value>> = vec![None; program.atoms().len()];
        let mut tangent_values: Vec<Option<Tangent<<E as Domain>::Type, Tracer<TangentContext<'_, E>>>>> =
            vec![None; program.atoms().len()];
        let mut tangent_context =
            TangentContext::new_with_residuals(context, builder.clone(), residuals.clone(), residual_atoms.clone());

        // Program inputs become linear-program inputs. Their concrete primal values are kept in parallel so JVP
        // rules can evaluate primal semantics while staging tangent operations.
        for (input_atom, input_primal) in program.input_ids().iter().copied().zip(input_primals.into_iter()) {
            let tangent = tangent_context.input(input_primal.r#type().into_owned());
            tangent_values[input_atom.index()] = Some(Tangent::Value(tangent));
            primal_values[input_atom.index()] = Some(input_primal);
        }
        // Constants already have primal values in the original program. Their tangents are derived lazily by
        // `tangent_for_atom` as `Tangent::Zero(type)`, propagating through JVP rules until they meet a non-zero
        // tangent that forces materialization.
        for (atom_index, atom) in program.atoms().iter().enumerate() {
            if let Atom::Constant(value) = atom {
                primal_values[atom_index] = Some(context.lift(value.clone())?);
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
                        tangent_for_atom::<E>(primal_values.as_slice(), tangent_values.as_slice(), input_atom)?,
                        residual_atom,
                    ))
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            // Symbolic-zero fast path: when an operation consumes at least one input and every input tangent is a
            // symbolic `Tangent::Zero`, its JVP rule is skipped entirely. The primal outputs are produced by binding
            // the primal operation on the differentiation context (interpreting it eagerly or staging it into the
            // nested primal program) and each output tangent stays a symbolic `Tangent::Zero`. This is sound by the
            // chain rule — the Jacobian applied to all-zero tangents is zero regardless of the operation — and it
            // also makes operations without JVP rules linearizable whenever no derivatives flow into them.
            // Zero-input operations are excluded so their dedicated rules keep handling primal synthesis and tangent
            // typing.
            let output_duals = if !input_duals.is_empty() && input_duals.iter().all(|dual| dual.tangent().is_zero()) {
                let primal_inputs = input_duals.iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>();
                context
                    .bind(instruction.operation().clone(), primal_inputs.as_slice())?
                    .into_iter()
                    .map(|primal| {
                        let tangent_type = primal.r#type().into_owned();
                        JvpTracer::from_zero_tangent(primal, tangent_type)
                    })
                    .collect()
            } else {
                instruction.operation().jvp(&mut tangent_context, input_duals.as_slice())?
            };
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
            let tangent = tangent_for_atom::<E>(primal_values.as_slice(), tangent_values.as_slice(), output_atom)?;
            let tangent_atom = tangent_context.materialize_tangent(tangent)?.atom_id()?;

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
        Output::To::<<E as Domain>::Value>::from_parameters(program.output_structure().clone(), output_primal_values)?,
        Pushforward::new(residuals.borrow().clone(), pushforward).compact_residuals()?,
    ))
}

impl<'domain, D, Capture> DifferentiationContext for TracingContext<'domain, D, Capture>
where
    D: DifferentiationContext + Domain + 'domain,
    <D as Domain>::Operation: From<ZeroOperation<<D as Domain>::Type>> + From<OneOperation<<D as Domain>::Type>>,
{
    type Tangent = Tracer<TracingContext<'domain, D, Capture>>;
    type LinearOperation<V: Value<<D as Domain>::Type>, F: Value<<D as Domain>::Type>> =
        <D as DifferentiationContext>::LinearOperation<V, F>;

    #[inline]
    fn zero_tangent(&self, type_: &Self::Type) -> Result<Self::Tangent, ProgramError> {
        let outputs = self.stage_operation(
            <D as Domain>::Operation::from(ZeroOperation::new(type_.clone())),
            &[] as &[Tracer<TracingContext<'domain, D, Capture>>],
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.into_iter().next().unwrap())
    }

    #[inline]
    fn validate_primal(&self, primal: &Self::Value) -> Result<(), ProgramError> {
        if std::rc::Rc::ptr_eq(self.builder(), primal.context().builder()) {
            Ok(())
        } else {
            Err(self.error(ProgramError::MismatchedProgramBuilders))
        }
    }

    /// Tracing contexts stage primal values as tracers, so concretizing extractions on them cannot succeed.
    #[inline]
    fn supports_primal_concretization(&self) -> bool {
        false
    }
}

/// Staging [`DifferentiationContext`] used to linearize a nested [`Program`] symbolically on behalf of an enclosing
/// differentiation context.
///
/// Higher-order JVP rules such as the one for
/// [`ConditionOperation`](crate::operations::control_flow::ConditionOperation) must linearize nested branch programs
/// *without* primal values: running a branch's primals eagerly would evaluate computations the predicate may never
/// select, and tracer-valued enclosing contexts have no concrete values to offer in the first place. This context
/// makes [`linearize_program_by_replay`] fully symbolic by splitting the two roles a differentiation context plays
/// during linearization:
///
///   - **Primal side**: [`Domain::Value`] is this context's own [`Tracer`], so JVP rules that compute primal results
///     through [`Context::bind`] stage ordinary primal instructions into the owned [`ProgramBuilder`] instead of
///     evaluating them. The staged program becomes the nested primal program (extended with residual outputs by
///     [`linearize_program`]).
///   - **Linear side**: [`DifferentiationContext::Tangent`] and
///     [`DifferentiationContext::LinearOperation`] are inherited from the *enclosing* context (through the
///     `TangentValue` and `CanonicalLinearOperation` parameters), so the staged pushforward program is directly
///     expressed in the enclosing context's linear representation and can be embedded into one of its linear
///     operations, such as a linear condition branch.
///
/// This differs from the [`TracingContext`] composition used by jvp-under-tracing, which pins
/// `Tangent = Tracer<Self>` and therefore fuses tangent values into the primal trace; here the pushforward must
/// remain a standalone program over the enclosing context's tangent carrier.
///
/// The parameters are deliberately the *components* of the enclosing context rather than the context itself, and
/// every component is a fixed point under nesting (see [`LinearizationContextOf`]): the nested context derived
/// from a [`LinearizationContext`] is the same type again. This keeps the trait-solver obligations finite when
/// condition rules require `O: DifferentiableOperation<LinearizationContextOf<D, O>>` for branches that
/// themselves contain conditions.
///
/// `CanonicalLinearOperation` is the enclosing context's linear operation family pinned at the factor type
/// `C` (the constant type): `E::LinearOperation<E::Tangent, E::Constant>`. Because
/// [`FactorParameterizedOperation::try_map_factors`] maps only the factor parameter,
/// `CanonicalLinearOperation::WithFactor<F>` recovers `E::LinearOperation<E::Tangent, F>` for every factor type `F`,
/// which is how this context defines its own [`LinearOperation`](DifferentiationContext::LinearOperation) family
/// without referring to `E`.
///
/// This context cannot synthesize concrete tangent values, so
/// [`zero_tangent`](DifferentiationContext::zero_tangent) is unsupported; it overrides
/// [`materializes_zero_tangents_as_operations`](DifferentiationContext::materializes_zero_tangents_as_operations) to
/// stage structural zeros as nullary linear zero operations instead.
#[doc(hidden)]
pub struct LinearizationContext<T, C, O, TangentValue, CanonicalLinearOperation>
where
    T: Type,
    C: Value<T>,
    O: Operation<T>,
    TangentValue: Value<T>,
    CanonicalLinearOperation: FactorParameterizedOperation<T, C>,
{
    /// [`ProgramBuilder`] that owns the nested primal [`Program`] staged by this context.
    builder: Rc<RefCell<ProgramBuilder<T, C, O>>>,

    /// [`PhantomData`] marker tying this context to the enclosing context's tangent value and canonical linear
    /// operation types.
    marker: PhantomData<(TangentValue, CanonicalLinearOperation)>,
}

impl<T, C, O, TangentValue, CanonicalLinearOperation>
    LinearizationContext<T, C, O, TangentValue, CanonicalLinearOperation>
where
    T: Type,
    C: Value<T>,
    O: Operation<T>,
    TangentValue: Value<T>,
    CanonicalLinearOperation: FactorParameterizedOperation<T, C>,
{
    /// Creates a new [`LinearizationContext`] that owns a fresh [`ProgramBuilder`].
    fn new() -> Self {
        Self { builder: Rc::new(RefCell::new(ProgramBuilder::new())), marker: PhantomData }
    }
}

impl<T, C, O, TangentValue, CanonicalLinearOperation> Clone
    for LinearizationContext<T, C, O, TangentValue, CanonicalLinearOperation>
where
    T: Type,
    C: Value<T>,
    O: Operation<T>,
    TangentValue: Value<T>,
    CanonicalLinearOperation: FactorParameterizedOperation<T, C>,
{
    fn clone(&self) -> Self {
        Self { builder: self.builder.clone(), marker: PhantomData }
    }
}

impl<T, C, O, TangentValue, CanonicalLinearOperation> Debug
    for LinearizationContext<T, C, O, TangentValue, CanonicalLinearOperation>
where
    T: Type,
    C: Value<T>,
    O: Operation<T>,
    TangentValue: Value<T>,
    CanonicalLinearOperation: FactorParameterizedOperation<T, C>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("LinearizationContext").finish_non_exhaustive()
    }
}

impl<T, C, O, TangentValue, CanonicalLinearOperation> Domain
    for LinearizationContext<T, C, O, TangentValue, CanonicalLinearOperation>
where
    T: Type,
    C: Value<T>,
    O: Operation<T>,
    TangentValue: Value<T>,
    CanonicalLinearOperation: FactorParameterizedOperation<T, C>,
{
    type Type = T;
    type Value = Tracer<Self>;
    type Constant = C;
    type Operation = O;
}

impl<T, C, O, TangentValue, CanonicalLinearOperation> Context
    for LinearizationContext<T, C, O, TangentValue, CanonicalLinearOperation>
where
    T: Type,
    C: Value<T>,
    O: Operation<T>,
    TangentValue: Value<T>,
    CanonicalLinearOperation: FactorParameterizedOperation<T, C>,
{
    /// Lifts a constant payload into this context by recording it as a constant primal [`Tracer`].
    #[inline]
    fn lift(&self, constant: C) -> Result<Tracer<Self>, ProgramError> {
        Ok(self.constant(constant))
    }

    /// Binding in a nested linearization context stages the primal operation into the nested primal program.
    #[inline]
    fn bind(&self, operation: O, inputs: &[Tracer<Self>]) -> Result<Vec<Tracer<Self>>, ProgramError> {
        self.stage_operation(operation, inputs)
    }
}

impl<T, C, O, TangentValue, CanonicalLinearOperation> StagingContext
    for LinearizationContext<T, C, O, TangentValue, CanonicalLinearOperation>
where
    T: Type,
    C: Value<T>,
    O: Operation<T>,
    TangentValue: Value<T>,
    CanonicalLinearOperation: FactorParameterizedOperation<T, C>,
{
    #[inline]
    fn builder(&self) -> &Rc<RefCell<ProgramBuilder<T, C, O>>> {
        &self.builder
    }
}

impl<T, C, O, TangentValue, CanonicalLinearOperation> DifferentiationContext
    for LinearizationContext<T, C, O, TangentValue, CanonicalLinearOperation>
where
    T: Type,
    C: Value<T>,
    O: Operation<T>,
    TangentValue: Value<T>,
    CanonicalLinearOperation: FactorParameterizedOperation<T, C>,
{
    type Tangent = TangentValue;
    type LinearOperation<V: Value<T>, F: Value<T>> =
        <CanonicalLinearOperation as FactorParameterizedOperation<T, C>>::WithFactor<F>;

    /// Nested symbolic linearization has no concrete tangent values to synthesize; structural zero tangents are
    /// staged as nullary linear zero operations (this context overrides
    /// [`materializes_zero_tangents_as_operations`](Self::materializes_zero_tangents_as_operations) to `true`)
    /// instead, so this method is never consulted by [`linearize_program`] and reports an error if reached through
    /// another path.
    fn zero_tangent(&self, _type: &T) -> Result<TangentValue, ProgramError> {
        Err(ProgramError::UnsupportedOperation {
            message: "nested symbolic linearization materializes structural zero tangents as staged zero operations \
                      and cannot synthesize constant tangent values"
                .to_string(),
        })
    }

    #[inline]
    fn validate_primal(&self, primal: &Self::Value) -> Result<(), ProgramError> {
        if Rc::ptr_eq(self.builder(), primal.context().builder()) {
            Ok(())
        } else {
            Err(self.error(ProgramError::MismatchedProgramBuilders))
        }
    }

    /// Nested symbolic linearization stages primal values as tracers, so concretizing extractions on them cannot
    /// succeed.
    #[inline]
    fn supports_primal_concretization(&self) -> bool {
        false
    }

    /// Nested symbolic linearization has no concrete tangent values to synthesize, so structural zero tangents are
    /// materialized as staged nullary zero operations rather than constant values.
    #[inline]
    fn materializes_zero_tangents_as_operations(&self) -> bool {
        true
    }
}

/// [`LinearizationContext`] derived from the enclosing differentiation context `E` for nested programs over
/// operations `O`.
///
/// Every component of this instantiation is a fixed point under nesting: the nested context's type, constant, and
/// tangent associated types equal `E`'s, its operation type is `O` again, and its canonical linear operation
/// `E::LinearOperation<E::Tangent, E::Constant>` maps to itself under `WithFactor<E::Constant>`. Consequently
/// `LinearizationContextOf<LinearizationContextOf<E, O>, O>` normalizes to
/// `LinearizationContextOf<E, O>`, which keeps `DifferentiableOperation` bounds for nested control flow
/// finite for the trait solver.
#[doc(hidden)]
pub type LinearizationContextOf<E, O> = LinearizationContext<
    <E as Domain>::Type,
    <E as Domain>::Constant,
    O,
    <E as DifferentiationContext>::Tangent,
    <E as DifferentiationContext>::LinearOperation<<E as DifferentiationContext>::Tangent, <E as Domain>::Constant>,
>;

/// Result of one symbolic linearization run: a pair of programs that together represent a function and its
/// derivative at *every* primal point, produced without any concrete primal values (see [`linearize_program`]).
///
/// A [`Linearization`] is the fully program-level form of forward-mode differentiation. Where a [`Pushforward`]
/// pairs a residualized linear program with the concrete residual values saved by one primal execution, a
/// [`Linearization`] keeps the primal side symbolic too, so the same artifact can be interpreted eagerly in a
/// concrete domain, spliced into an enclosing trace when `E`'s values are [`Tracer`]s, or embedded as program data
/// inside higher-order operations such as linear `condition` branches and `scan`/`while` bodies.
///
/// The two programs obey the following shape contract:
///
///   - [`primal_program`](Self::primal_program) computes the original function, *extended* so that the residual
///     values captured by product-rule factors become extra outputs appended after the original outputs: output
///     `i` of the original program remains output `i`, and residual index `j` becomes appended output
///     `original_output_count + j`.
///   - [`pushforward_program`](Self::pushforward_program) is the residualized linear map of the function at the
///     primal point. Its [`ResidualFactor::Reference`] factors index into [`residual_types`](Self::residual_types)
///     positionally, which by the contract above means reference `j` is satisfied by appended primal output
///     `original_output_count + j`. References are *not* baked into constants, so callers can rebind them against
///     any residual environment.
///   - [`residual_types`](Self::residual_types) lists the residual value types, aligned with both the appended
///     primal outputs and the pushforward's residual references.
///
/// The `Input` and `Output` type parameters are the linearized program's input and output [`Parameterized`]
/// families over `E`'s constant type, exactly as they appear on the source [`Program`]; the pushforward program
/// reuses them reparameterized to `E`'s tangent type. [`NestedLinearization`] fixes both to flat [`Vec`]s, which is
/// the shape used by higher-order JVP rules that linearize captured branch and body programs.
///
/// Use [`interpret_primal`](Self::interpret_primal) to evaluate the primal side at concrete primal values and
/// recover the residual environment, or [`at`](Self::at) to do that and bind the residuals into a reusable
/// value-level [`Pushforward`].
pub struct Linearization<E, O, Input, Output>
where
    E: DifferentiationContext,
    O: Operation<E::Type>,
    Input: Parameterized<E::Constant, Family: ParameterizedFamily<E::Tangent>>,
    Output: Parameterized<E::Constant, Family: ParameterizedFamily<E::Tangent>>,
{
    /// Staged primal program, extended so that the residual values captured by the pushforward become extra
    /// outputs appended after the original outputs. Residual index `i` of
    /// [`pushforward_program`](Self::pushforward_program) corresponds to appended output `i`.
    pub primal_program: Program<E::Type, E::Constant, O, Input, Vec<E::Constant>>,

    /// Residualized pushforward program, expressed directly in the enclosing context's linear representation. Its
    /// [`ResidualFactor::Reference`] factors index the residual outputs appended to
    /// [`primal_program`](Self::primal_program).
    // The `tangent_program` name is reserved for this field once the remaining `pushforward_program` consumers are
    // migrated; renaming now would break the `NestedLinearization` alias's field accesses.
    pub pushforward_program:
        Program<E::Type, E::Tangent, LinearOperationOf<E>, Input::To<E::Tangent>, Output::To<E::Tangent>>,

    /// Types of the residual values, aligned with the outputs appended to [`primal_program`](Self::primal_program).
    pub residual_types: Vec<E::Type>,
}

impl<E, O, Input, Output> Linearization<E, O, Input, Output>
where
    E: DifferentiationContext,
    O: Operation<E::Type>,
    Input: Parameterized<E::Constant, Family: ParameterizedFamily<E::Tangent>>,
    Output: Parameterized<E::Constant, Family: ParameterizedFamily<E::Tangent>>,
{
    /// Interprets [`primal_program`](Self::primal_program) at `primals` through `context` and returns the original
    /// primal outputs together with the captured residual values.
    ///
    /// Every instruction is executed through [`Context::bind`] and every program constant is lifted through
    /// [`Context::lift`], so the primal side takes on `context`'s value semantics: a concrete domain evaluates it
    /// eagerly, while a staging context (whose values are [`Tracer`]s) splices the program's instructions into its
    /// active trace. The flat interpreted outputs are split using [`residual_types`](Self::residual_types): the
    /// leading outputs are reassembled into the original structured output and the trailing
    /// `residual_types.len()` values are returned as the residual environment, aligned with the pushforward
    /// program's [`ResidualFactor::Reference`] indices.
    ///
    /// # Parameters
    ///
    ///   - `context`: Differentiation context that supplies the primal value semantics.
    ///   - `primals`: Structured primal input values matching the program's input structure.
    pub fn interpret_primal(
        &self,
        context: &E,
        primals: Input::To<<E as Domain>::Value>,
    ) -> Result<(Output::To<<E as Domain>::Value>, Vec<<E as Domain>::Value>), ProgramError>
    where
        E: Domain<Operation = O>,
        O: Clone,
        Input::Family: ParameterizedFamily<<E as Domain>::Value>,
        Output::Family: ParameterizedFamily<<E as Domain>::Value>,
        Input::ParameterStructure: Debug + PartialEq,
    {
        let primal_structure = primals.parameter_structure();
        if primal_structure != *self.primal_program.input_structure() {
            return Err(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{:?}", self.primal_program.input_structure()),
                right_structure: format!("{primal_structure:?}"),
            }
            .into());
        }

        let inputs = primals.into_parameters().collect::<Vec<_>>();
        let mut outputs = self.primal_program.interpret_with(
            inputs,
            |_, constant| context.lift(constant.clone()),
            |instruction, instruction_inputs| context.bind(instruction.operation().clone(), instruction_inputs),
        )?;
        let residual_count = self.residual_types.len();
        let Some(output_count) = outputs.len().checked_sub(residual_count) else {
            return Err(ProgramError::MalformedProgram(format!(
                "residual-extended primal program produced {} outputs which is fewer than its {} residual types",
                outputs.len(),
                residual_count,
            )));
        };
        let residual_values = outputs.split_off(output_count);
        let outputs = Output::To::<<E as Domain>::Value>::from_parameters(
            self.pushforward_program.output_structure().clone(),
            outputs,
        )?;
        Ok((outputs, residual_values))
    }

    /// Evaluates this linearization at `primals` and returns the primal outputs together with the [`Pushforward`]
    /// of the function at that primal point.
    ///
    /// This is the bridge from the symbolic artifact back to value-level forward mode: it runs
    /// [`interpret_primal`](Self::interpret_primal) once and binds the resulting residual values into a
    /// [`Pushforward`] over [`pushforward_program`](Self::pushforward_program), which can then be applied to
    /// arbitrary tangent inputs at the same primal point, instantiated into a direct program, or transposed.
    ///
    /// # Parameters
    ///
    ///   - `context`: Differentiation context that supplies the primal value semantics.
    ///   - `primals`: Structured primal input values matching the program's input structure.
    pub fn at(
        &self,
        context: &E,
        primals: Input::To<<E as Domain>::Value>,
    ) -> Result<
        (Output::To<<E as Domain>::Value>, Pushforward<E, Input::To<E::Tangent>, Output::To<E::Tangent>>),
        ProgramError,
    >
    where
        E: Domain<Operation = O>,
        O: Clone,
        Input::Family: ParameterizedFamily<<E as Domain>::Value>,
        Output::Family: ParameterizedFamily<<E as Domain>::Value>,
        Input::ParameterStructure: Debug + PartialEq,
    {
        let (outputs, residual_values) = self.interpret_primal(context, primals)?;
        Ok((outputs, Pushforward::new(residual_values, self.pushforward_program.clone())))
    }
}

/// [`Linearization`] of a nested flat program whose inputs and outputs are plain [`Vec`]s, as produced by one
/// nested symbolic linearization run through [`Program::linearize`] (see [`ProgramLinearizableOperation`]).
pub type NestedLinearization<E, O> = Linearization<E, O, Vec<<E as Domain>::Constant>, Vec<<E as Domain>::Constant>>;

/// Operation types whose captured flat programs can be linearized symbolically on behalf of an enclosing
/// differentiation context.
///
/// This is the forward-mode counterpart of
/// [`ProgramBatchableOperation`](crate::tracing_v2::batching::ProgramBatchableOperation), implemented by closed
/// operation enums (via [`linearize_program`]) so that higher-order JVP rules can differentiate the programs they
/// capture without concrete primal values. Routing nested linearization through a dedicated witness trait keeps the
/// trait solver's recursion finite: the closed enum impl discharges every derived
/// [`LinearizationContext`] obligation once, as a definition-time body check against the single
/// [`LinearizationContextOf`] type derived from `E`, instead of each higher-order rule carrying a
/// `Self: DifferentiableOperation<LinearizationContextOf<E, Self>>` where-clause whose re-instantiation at the
/// derived context overflows.
pub trait ProgramLinearizableOperation<C: DifferentiationContext>: Operation<<C as Domain>::Type> + Sized {
    /// Linearizes `program` symbolically on behalf of `differentiable`; refer to the documentation of
    /// [`linearize_program`] for the returned packaging.
    fn linearize_program(
        differentiable: &C,
        program: &Program<
            <C as Domain>::Type,
            <C as Domain>::Constant,
            Self,
            Vec<<C as Domain>::Constant>,
            Vec<<C as Domain>::Constant>,
        >,
    ) -> Result<NestedLinearization<C, Self>, ProgramError>;
}

impl<T: Type, V: Value<T>, O: Operation<T>> Program<T, V, O, Vec<V>, Vec<V>> {
    /// Linearizes this [`Program`] symbolically on behalf of `differentiable`.
    ///
    /// Refer to [`linearize_program`] for the returned packaging and replay semantics.
    pub fn linearize<C: DifferentiationContext<Type = T, Constant = V>>(
        &self,
        context: &C,
    ) -> Result<NestedLinearization<C, O>, ProgramError>
    where
        O: ProgramLinearizableOperation<C>,
    {
        O::linearize_program(context, self)
    }
}

/// Linearizes a staged [`Program`] symbolically on behalf of the enclosing differentiation context `differentiable`.
///
/// This is the typed core behind every program-level forward-mode transform, and in particular the building block
/// for higher-order JVP rules that must differentiate nested programs *without* concrete primal values, exactly
/// like JAX's [`jvp_jaxpr`](https://docs.jax.dev/en/latest/jax.interpreters.ad.html) + partial evaluation split for
/// control-flow primitives. The program is replayed through JVP rules against a fresh
/// [`LinearizationContext`] whose values are tracers, so the primal side is *staged* into a fresh program
/// instead of being evaluated, while the tangent side is staged into a pushforward expressed directly over the
/// enclosing context's [`Tangent`](DifferentiationContext::Tangent) and
/// [`LinearOperation`](DifferentiationContext::LinearOperation) types.
///
/// The returned [`Linearization`] preserves the program's `Input`/`Output` parameterizations and packages the
/// residual-extended primal program, the residualized pushforward program, and the residual types; refer to the
/// [`Linearization`] documentation for the exact program-shape contract. Factor payloads captured from program
/// *constants* (which have no residual atom) are converted into closed [`ResidualFactor::Constant`] factors by
/// lifting the constant through `differentiable`.
///
/// # Parameters
///
///   - `differentiable`: Enclosing [`DifferentiationContext`] implementation. It is only used to lift program
///     constants that JVP rules captured as closed factors; no primal computation runs on it.
///   - `program`: Staged primal program to linearize symbolically.
pub fn linearize_program<E, O, Input, Output>(
    differentiable: &E,
    program: &Program<<E as Domain>::Type, <E as Domain>::Constant, O, Input, Output>,
) -> Result<Linearization<E, O, Input, Output>, ProgramError>
where
    E: DifferentiationContext,
    O: Clone + Operation<E::Type> + DifferentiableOperation<LinearizationContextOf<E, O>>,
    Input: Parameterized<<E as Domain>::Constant, Family: ParameterizedFamily<E::Tangent>>,
    Output: Parameterized<
            <E as Domain>::Constant,
            Family: ParameterizedFamily<Tracer<LinearizationContextOf<E, O>>> + ParameterizedFamily<E::Tangent>,
        >,
    E::LinearOperation<E::Tangent, <E as Domain>::Constant>:
        FactorParameterizedOperation<<E as Domain>::Type, <E as Domain>::Constant>,
    LinearOperationOf<LinearizationContextOf<E, O>>: ResidualizedOperation<LinearizationContextOf<E, O>>
        + FactorParameterizedOperation<
            <E as Domain>::Type,
            ResidualFactor<<E as Domain>::Type, Tracer<LinearizationContextOf<E, O>>>,
            WithFactor<ResidualFactor<<E as Domain>::Type, <E as Domain>::Value>> = LinearOperationOf<E>,
        >,
{
    let nested_context = LinearizationContextOf::<E, O>::new();
    let input_tracers = program
        .input_types()
        .into_iter()
        .map(|input_type| nested_context.input(input_type))
        .collect::<Vec<_>>();
    let (output_primals, pushforward) = linearize_program_by_replay(&nested_context, program, input_tracers)?;
    // Surface errors that nested staging recorded on the builder while a rule continued with poisoned tracers, so
    // callers see the original failure instead of an opaque `PoisonedValue` from the atom collection below.
    if let Some(error) = nested_context.builder().borrow_mut().error.take() {
        return Err(error);
    }

    // Collect the atoms and types that outlive the nested run before dropping any tracer.
    let output_atoms = output_primals.parameters().map(Tracer::atom_id).collect::<Result<Vec<_>, _>>()?;
    let residual_atoms = pushforward.residuals().iter().map(Tracer::atom_id).collect::<Result<Vec<_>, _>>()?;
    let residual_types =
        pushforward.residuals().iter().map(|residual| residual.r#type().into_owned()).collect::<Vec<_>>();

    // Rebase the pushforward's factor payloads from nested tracers onto the enclosing context's value type. Residual
    // references are positional and carry over unchanged, while closed factors can only have been captured from
    // nested program constants and are lifted through the enclosing context.
    let pushforward_program = pushforward.program().map_operations(|operation| {
        operation.try_map_factors(&mut |factor| match factor {
            ResidualFactor::Reference { index, r#type } => {
                Ok(ResidualFactor::Reference { index: *index, r#type: r#type.clone() })
            }
            ResidualFactor::Constant(tracer) => {
                let atom = tracer.atom_id()?;
                let builder = nested_context.builder().borrow();
                match builder.atoms().get(atom.index()) {
                    Some(Atom::Constant(constant)) => {
                        Ok(ResidualFactor::Constant(differentiable.lift(constant.clone())?))
                    }
                    Some(Atom::Variable(_)) => Err(ProgramError::MalformedProgram(format!(
                        "nested symbolic linearization captured non-constant primal atom {atom} as a closed factor",
                    ))),
                    None => Err(ProgramError::UnboundAtomId { id: atom }),
                }
            }
        })
    })?;

    // Drop every tracer so the nested primal builder can be recovered, then build the residual-extended program.
    drop(output_primals);
    drop(pushforward);
    let LinearizationContext { builder, marker: _ } = nested_context;
    let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
    let mut output_ids = output_atoms;
    output_ids.extend(residual_atoms);
    let extended_output_count = output_ids.len();
    let primal_program = builder
        .build(output_ids, program.input_structure().clone(), vec![Placeholder; extended_output_count])?
        .simplified()?;
    Ok(Linearization { primal_program, pushforward_program, residual_types })
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
        E: 'jvp,
        LinearOperationOf<E>: From<ZeroOperation<E::Type>>;
}

/// Forward-mode rule for operations whose outputs have no tangent space.
///
/// [`ZeroTangentOperation`] is the analogue of JAX's
/// [`defjvp_zero`](https://docs.jax.dev/en/latest/jax.interpreters.ad.html#jax.interpreters.ad.defjvp_zero): it marks
/// operations whose outputs are discrete (for example, Boolean comparison and logical operations), so every output
/// tangent is identically zero. Such operations are piecewise constant as maps into their discrete codomain: at every
/// primal point where they are defined, an infinitesimal input perturbation leaves the outputs unchanged, so the
/// pushforward is the zero map. Equivalently, discrete output spaces have no tangent space to carry derivative
/// information in the first place, which is why declaring the tangents to be symbolically zero is mathematically
/// sound rather than an approximation.
///
/// The provided [`zero_tangent_jvp`](Self::zero_tangent_jvp) method implements the complete rule: it computes the
/// primal outputs by interpreting the operation on the input primals (mirroring how
/// [`StopGradientOperation`](crate::operations::differentiation::StopGradientOperation)'s rule passes its primal
/// through), and pairs each primal output with a symbolic [`Tangent::Zero`] of that output's type. No linear
/// operation is staged, so these operations never appear in pushforward programs and need no transpose rules; the
/// symbolic zeros short-circuit through downstream JVP rules instead.
///
/// Implementors only declare the marker impl; [`DifferentiableOperation::jvp`] impls then delegate to
/// [`zero_tangent_jvp`](Self::zero_tangent_jvp).
pub trait ZeroTangentOperation<E: DifferentiationContext>: InterpretableOperation<E::Type, E::Value> {
    /// Applies the zero-tangent forward-mode rule: interprets the operation on the input primals and pairs each
    /// output with a symbolic [`Tangent::Zero`] of the output's type.
    fn zero_tangent_jvp<'jvp>(
        &self,
        _context: &mut TangentContext<'jvp, E>,
        inputs: &[JvpTracer<'jvp, E>],
    ) -> Result<Vec<JvpTracer<'jvp, E>>, ProgramError>
    where
        E: 'jvp,
    {
        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal_outputs = self.interpret(primal_inputs.as_slice())?;
        Ok(primal_outputs
            .into_iter()
            .map(|primal| {
                let tangent_type = primal.r#type().into_owned();
                JvpTracer::from_zero_tangent(primal, tangent_type)
            })
            .collect())
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
        Self { differentiable, builder, residuals, residual_atoms }
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
    /// Structural zeros carry only type metadata. When a nested linear program needs an actual input atom, this
    /// method realizes the zero in the active linear builder — either as the differentiation context's canonical zero
    /// tangent **value** (the default) or, when the context's
    /// [`materializes_zero_tangents_as_operations`](DifferentiationContext::materializes_zero_tangents_as_operations)
    /// is `true`, as a staged nullary zero **operation**. Non-zero tangents are returned unchanged.
    pub fn materialize_tangent(
        &self,
        tangent: Tangent<E::Type, Tracer<TangentContext<'domain, E>>>,
    ) -> Result<Tracer<TangentContext<'domain, E>>, ProgramError>
    where
        LinearOperationOf<E>: From<ZeroOperation<E::Type>>,
    {
        match tangent {
            Tangent::Zero(r#type) => {
                if self.differentiable.materializes_zero_tangents_as_operations() {
                    let mut outputs = self.stage_operation(
                        LinearOperationOf::<E>::from(ZeroOperation::new(r#type)),
                        &[] as &[Tracer<TangentContext<'domain, E>>],
                    )?;
                    check_count!("output", outputs, 1, ProgramError);
                    return Ok(outputs.remove(0));
                }
                Ok(self.constant(self.differentiable.zero_tangent(&r#type)?))
            }
            Tangent::Value(tracer) => Ok(tracer),
        }
    }

    /// Captures `value` as an anonymous residual factor.
    ///
    /// Higher-order JVP rules also use this to register primal values they computed themselves — for example, the
    /// residual outputs of a staged primal condition — in the active linearization residual environment, obtaining
    /// [`ResidualFactor::Reference`] factors that reusable pushforwards instantiate later.
    pub fn factor(&mut self, value: E::Value) -> ResidualFactor<E::Type, E::Value> {
        let r#type = value.r#type().into_owned();
        let mut residuals = self.residuals.borrow_mut();
        let index = residuals.len();
        residuals.push(value);
        ResidualFactor::Reference { index, r#type }
    }

    /// Captures `value` as a residual factor, deduplicating by `atom` when one is available.
    fn factor_for_atom(&mut self, atom: AtomId, value: E::Value) -> ResidualFactor<E::Type, E::Value> {
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

impl<S: Value<DataType>> DifferentiationContext for ScalarDomain<S>
where
    ScalarDomain<S>: Context
        + Domain<
            Type = DataType,
            Value = S,
            Constant = S,
            Operation: Clone + InterpretableOperation<DataType, S> + From<ZeroOperation<DataType>>,
        >,
{
    type Tangent = S;
    type LinearOperation<V: Value<DataType>, F: Value<DataType>> = LinearScalarOperation<S, F>;

    #[inline]
    fn zero_tangent(&self, type_: &DataType) -> Result<Self::Tangent, ProgramError> {
        let mut outputs = self.bind(ZeroOperation::new(type_.clone()).into(), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.pop().unwrap())
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

        let array_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(2)]));
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

    #[test]
    fn test_jvp_takes_the_symbolic_zero_fast_path_for_rule_less_operations() {
        use crate::contexts::StagingContext;
        use crate::operations::differentiation::StopGradient;
        use crate::programs::ProgramError;
        use crate::tests::{TestArray, TestArrayDomain};
        use crate::tracing_v2::operations::collective::CollectiveKind;
        use crate::tracing_v2::{ArrayOperation, LinearizationTracer};

        // `ArrayOperation::Collective` has no JVP rule (its dispatch arm errors), but severing the incoming tangent
        // with `stop_gradient` makes every collective input tangent a symbolic zero, so the symbolic linearization
        // replay's all-zero fast path computes the primal by binding the operation and emits a zero output tangent
        // without consulting the (missing) rule. The function is `f(x) = x + psum(stop_gradient(x))`, which
        // differentiates like `x + c`, so the tangent equals the input tangent. `jvp` traces the closure into a
        // primal program first, so the fast path fires during the replay in `linearize_program` rather than during
        // value-level interpretation.
        let (primal, tangent) = TestArrayDomain
            .jvp(
                |x: LinearizationTracer<'_, TestArrayDomain>| {
                    let severed = x.stop_gradient();
                    let mut outputs = severed
                        .context()
                        .stage_operation(
                            ArrayOperation::Collective { axis_name: "lanes".to_string(), kind: CollectiveKind::PSum },
                            &[&severed],
                        )
                        .unwrap();
                    x + outputs.remove(0)
                },
                TestArray::scalar(2.0),
                TestArray::scalar(1.0),
            )
            .unwrap();
        assert_eq!(primal.values, vec![4.0]);
        assert_eq!(tangent.values, vec![1.0]);

        // Without the severed tangent, the missing collective rule is still reported.
        let result = TestArrayDomain
            .linearize(
                |x: LinearizationTracer<'_, TestArrayDomain>| {
                    let mut outputs = x.context().stage_operation(
                        ArrayOperation::Collective { axis_name: "lanes".to_string(), kind: CollectiveKind::PSum },
                        &[&x],
                    )?;
                    Ok(outputs.remove(0))
                },
                TestArray::scalar(2.0),
            )
            .map(|_| ());
        assert!(matches!(
            result,
            Err(ProgramError::Type(crate::types::TypeError { message }))
                if message == "psum does not support generic array jvp dispatch",
        ));
    }

    #[test]
    fn test_linearization_at_matches_eager_linearize_program() {
        use crate::operations::trigonometric::Sin;
        use crate::tests::{TestArray, TestArrayDomain};
        use crate::tracing::TracingContext;

        // `f(x, y) = x * y + sin(x)` exercises the product-rule residual path: the pushforward captures both primal
        // inputs as residuals, so `at` must recover them by interpreting the residual-extended primal program.
        let domain = TestArrayDomain;
        let primals = vec![TestArray::scalar(2.0), TestArray::scalar(3.0)];
        let (_, program) = TracingContext::interpret_and_trace(
            &domain,
            |inputs: Vec<_>| {
                let x = inputs[0].clone();
                let y = inputs[1].clone();
                Ok(vec![x.clone() * y + x.sin()])
            },
            primals.clone(),
        )
        .unwrap();

        let linearization = program.linearize(&domain).unwrap();
        let (symbolic_outputs, symbolic_pushforward) = linearization.at(&domain, primals.clone()).unwrap();
        let (eager_outputs, eager_pushforward) = domain.linearize_program(&program, primals).unwrap();

        assert_eq!(symbolic_outputs.len(), 1);
        assert_eq!(symbolic_outputs[0].values, vec![2.0 * 3.0 + 2.0f64.sin()]);
        assert_eq!(symbolic_outputs[0].values, eager_outputs[0].values);
        assert_eq!(symbolic_pushforward.residuals().len(), eager_pushforward.residuals().len());
        for (symbolic_residual, eager_residual) in
            symbolic_pushforward.residuals().iter().zip(eager_pushforward.residuals())
        {
            assert_eq!(symbolic_residual.values, eager_residual.values);
        }
        for tangents in [
            vec![TestArray::scalar(1.0), TestArray::scalar(0.0)],
            vec![TestArray::scalar(0.5), TestArray::scalar(-2.0)],
        ] {
            let symbolic_tangents = symbolic_pushforward.apply(tangents.clone()).unwrap();
            let eager_tangents = eager_pushforward.apply(tangents.clone()).unwrap();
            assert_eq!(symbolic_tangents.len(), 1);
            assert_eq!(
                symbolic_tangents[0].values,
                vec![(3.0 + 2.0f64.cos()) * tangents[0].values[0] + 2.0 * tangents[1].values[0]]
            );
            assert_eq!(symbolic_tangents[0].values, eager_tangents[0].values);
        }
    }

    #[test]
    fn test_linearization_at_keeps_structural_zero_tangents_symbolic() {
        use crate::operations::differentiation::StopGradient;
        use crate::tests::{TestArray, TestArrayDomain};
        use crate::tracing::TracingContext;

        // `f(x, y) = (x * stop_gradient(x), stop_gradient(y))`: the severed tangents stay symbolically zero through
        // the replay, so the first output's product rule keeps only its `dx` branch and the second output's tangent
        // reaches the output boundary as a structural zero, which the symbolic core stages as a nullary linear zero
        // operation (whereas the eager path materializes a constant zero tangent). Both must agree numerically.
        let domain = TestArrayDomain;
        let primals = vec![TestArray::scalar(2.0), TestArray::scalar(5.0)];
        let (_, program) = TracingContext::interpret_and_trace(
            &domain,
            |inputs: Vec<_>| {
                let x = inputs[0].clone();
                let y = inputs[1].clone();
                Ok(vec![x.clone() * x.stop_gradient(), y.stop_gradient()])
            },
            primals.clone(),
        )
        .unwrap();

        let linearization = program.linearize(&domain).unwrap();
        let (symbolic_outputs, symbolic_pushforward) = linearization.at(&domain, primals.clone()).unwrap();
        let (eager_outputs, eager_pushforward) = domain.linearize_program(&program, primals).unwrap();

        assert_eq!(symbolic_outputs.len(), 2);
        assert_eq!(symbolic_outputs[0].values, vec![4.0]);
        assert_eq!(symbolic_outputs[1].values, vec![5.0]);
        assert_eq!(symbolic_outputs[0].values, eager_outputs[0].values);
        assert_eq!(symbolic_outputs[1].values, eager_outputs[1].values);
        for tangents in [
            vec![TestArray::scalar(1.0), TestArray::scalar(7.0)],
            vec![TestArray::scalar(-0.5), TestArray::scalar(1.0)],
        ] {
            let symbolic_tangents = symbolic_pushforward.apply(tangents.clone()).unwrap();
            let eager_tangents = eager_pushforward.apply(tangents.clone()).unwrap();
            assert_eq!(symbolic_tangents.len(), 2);
            assert_eq!(symbolic_tangents[0].values, vec![2.0 * tangents[0].values[0]]);
            assert_eq!(symbolic_tangents[1].values, vec![0.0]);
            assert_eq!(symbolic_tangents[0].values, eager_tangents[0].values);
            assert_eq!(symbolic_tangents[1].values, eager_tangents[1].values);
        }
    }

    #[test]
    fn test_linearization_interpret_primal_splices_into_active_trace() {
        use std::cell::RefCell;
        use std::rc::Rc;

        use crate::contexts::StagingContext;
        use crate::operations::Operation;
        use crate::programs::ProgramBuilder;
        use crate::tests::{TestArray, TestArrayDomain};
        use crate::tracing::TracingContext;
        use crate::tracing_v2::ArrayOperation;

        // Interpreting the residual-extended primal program with tracer inputs must splice its instructions into
        // the tracers' active trace: `bind` on a `TracingContext` stages instead of evaluating. This is the
        // composition that lets program-level forward mode run under an enclosing trace.
        let domain = TestArrayDomain;
        let (_, program) = TracingContext::interpret_and_trace(
            &domain,
            |inputs: Vec<_>| Ok(vec![inputs[0].clone() * inputs[0].clone()]),
            vec![TestArray::scalar(3.0)],
        )
        .unwrap();

        let builder =
            Rc::new(RefCell::new(ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray, ArrayType>>::new()));
        let context = TracingContext::new(&domain, builder.clone());
        let x = context.input(ArrayType::scalar(DataType::F64));
        let linearization = program.linearize(&context).unwrap();
        let (outputs, residuals) = linearization.interpret_primal(&context, vec![x.clone()]).unwrap();

        // One original output plus one deduplicated residual (the primal input), which interpretation resolves to
        // the outer input tracer itself.
        assert_eq!(outputs.len(), 1);
        assert_eq!(residuals.len(), 1);
        assert_eq!(residuals[0].atom_id().unwrap(), x.atom_id().unwrap());

        // The outer builder gained exactly the program's `mul` instruction, producing the returned output tracer.
        let builder = builder.borrow();
        assert_eq!(builder.instructions().len(), 1);
        assert_eq!(builder.instructions()[0].operation().name(), "mul");
        assert_eq!(builder.instructions()[0].outputs(), &[outputs[0].atom_id().unwrap()]);
    }
}
