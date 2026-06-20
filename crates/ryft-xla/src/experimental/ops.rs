use std::cell::RefCell;
use std::fmt::Display;
use std::rc::Rc;

use ryft_core::batching::BatchingError;
use ryft_core::compilation::CapturedConstant;
use ryft_core::contexts::StagingContext;
use ryft_core::differentiation::{Cotangent, TransposableOperation};
use ryft_core::domains::Domain;
use ryft_core::macros::check_count;
use ryft_core::operations::constants::ZeroOperation;
use ryft_core::operations::{InterpretableOperation, Operation, OperationFormatter};
use ryft_core::parameters::Placeholder;
use ryft_core::programs::{Program, ProgramBuilder, ProgramError, Value};
use ryft_core::tracing::{AbstractTracingContext, Tracer, TracingContext};
use ryft_core::tracing_v2::batching::{ArrayBatch, BatchableOperation, BatchingContext, batch_input_metadata};
use ryft_core::tracing_v2::{
    ArrayOperation, CapturedFactor, DifferentiableOperation, DifferentiationContext, FactorParameterizedOperation,
    JvpTracer, LinearArrayOperation, LinearOperationOf, TangentContext,
};
use ryft_core::types::{ArrayType, Size, TypeError, Typed};

use crate::experimental::domains::{XlaDomain, XlaTracer};
use crate::experimental::operations::{LinearShardMapOperation, ShardMapOperation};

/// Backend-owned ordinary operations that extend the reusable core array operation set: a call to a jitted XLA
/// sub-program ([`JitCall`](Self::JitCall)), an XLA `shard_map` ([`ShardMap`](Self::ShardMap)), and an XLA
/// `linear_shard_map` staged in ordinary traced programs ([`LinearShardMap`](Self::LinearShardMap)).
///
/// The [`Operation`]/[`Display`] and per-variant [`From`]/[`TryFrom`] impls below are uniform per-variant delegations
/// to the backing operation structs. The backend-specific interpretation, batching, and differentiation rules are
/// bespoke (per-variant exemplar extraction, an erroring `shard_map` batching rule, and distinct `jvp` entry points),
/// so they are hand-written individually further below.
#[derive(Clone, Debug)]
pub enum XlaOperationExtension<V: Value<ArrayType>> {
    /// Call to a jitted XLA sub-program.
    JitCall(Box<JitCallOperation>),

    /// XLA-specific `shard_map`.
    ShardMap(Box<ShardMapOperation<V>>),

    /// XLA-specific `linear_shard_map` staged in ordinary traced programs.
    LinearShardMap(Box<LinearShardMapOperation<V>>),
}

impl<V: Value<ArrayType>> Operation<ArrayType> for XlaOperationExtension<V> {
    fn name(&self) -> &'static str {
        match self {
            Self::JitCall(operation) => operation.name(),
            Self::ShardMap(operation) => operation.name(),
            Self::LinearShardMap(operation) => operation.name(),
        }
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        match self {
            Self::JitCall(operation) => operation.infer_output_types(input_types),
            Self::ShardMap(operation) => operation.infer_output_types(input_types),
            Self::LinearShardMap(operation) => operation.infer_output_types(input_types),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::JitCall(operation) => operation.render(formatter, indentation),
            Self::ShardMap(operation) => operation.render(formatter, indentation),
            Self::LinearShardMap(operation) => operation.render(formatter, indentation),
        }
    }
}

impl<V: Value<ArrayType>> Display for XlaOperationExtension<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<V: Value<ArrayType>> From<JitCallOperation> for XlaOperationExtension<V> {
    fn from(operation: JitCallOperation) -> Self {
        Self::JitCall(Box::new(operation))
    }
}

impl<'a, V: Value<ArrayType>> TryFrom<&'a XlaOperationExtension<V>> for &'a JitCallOperation {
    type Error = ();

    fn try_from(value: &'a XlaOperationExtension<V>) -> Result<Self, ()> {
        match value {
            XlaOperationExtension::JitCall(operation) => Ok(&**operation),
            _ => Err(()),
        }
    }
}

impl<V: Value<ArrayType>> From<ShardMapOperation<V>> for XlaOperationExtension<V> {
    fn from(operation: ShardMapOperation<V>) -> Self {
        Self::ShardMap(Box::new(operation))
    }
}

impl<'a, V: Value<ArrayType>> TryFrom<&'a XlaOperationExtension<V>> for &'a ShardMapOperation<V> {
    type Error = ();

    fn try_from(value: &'a XlaOperationExtension<V>) -> Result<Self, ()> {
        match value {
            XlaOperationExtension::ShardMap(operation) => Ok(&**operation),
            _ => Err(()),
        }
    }
}

impl<V: Value<ArrayType>> From<LinearShardMapOperation<V>> for XlaOperationExtension<V> {
    fn from(operation: LinearShardMapOperation<V>) -> Self {
        Self::LinearShardMap(Box::new(operation))
    }
}

impl<'a, V: Value<ArrayType>> TryFrom<&'a XlaOperationExtension<V>> for &'a LinearShardMapOperation<V> {
    type Error = ();

    fn try_from(value: &'a XlaOperationExtension<V>) -> Result<Self, ()> {
        match value {
            XlaOperationExtension::LinearShardMap(operation) => Ok(&**operation),
            _ => Err(()),
        }
    }
}

/// Lifetime-free reference to a concrete XLA value captured by a compiled program.
pub type XlaConstant = CapturedConstant<ArrayType>;

/// Ordinary staged-op universe owned by the XLA backend.
impl<V: Value<ArrayType>> ryft_core::tracing_v2::operations::MaybeDot for XlaOperationExtension<V> {
    #[inline]
    fn dot_dimensions(&self) -> Option<&ryft_core::tracing_v2::DotDimensionNumbers> {
        // Higher-order XLA calls may contain dots in their bodies but are not themselves dot primitives, mirroring
        // how JAX's `dots_saveable` rematerialization policy matches only dot primitives.
        None
    }
}

impl<V: Value<ArrayType>> ryft_core::tracing_v2::rematerialization::MaybeRematerializationName
    for XlaOperationExtension<V>
{
    #[inline]
    fn rematerialization_name(&self) -> Option<&str> {
        None
    }
}

pub type XlaOperation = ArrayOperation<ArrayType, XlaConstant, XlaOperationExtension<XlaConstant>>;

/// Staged XLA program specialized to the backend-owned XLA op universe.
pub type XlaProgram<Input, Output> = Program<ArrayType, XlaConstant, XlaOperation, Input, Output>;

/// Program builder specialized to the backend-owned XLA op universe.
pub type XlaProgramBuilder = ProgramBuilder<ArrayType, XlaConstant, XlaOperation>;

/// Flat XLA program payload used by staged call operations.
pub type FlatXlaProgram = XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>;

/// Backend-owned linear operations that extend the reusable core linear array operation set: a linearized call to
/// a jitted XLA sub-program ([`LinearJitCall`](Self::LinearJitCall)) and an XLA linear `shard_map`
/// ([`LinearShardMap`](Self::LinearShardMap)).
///
/// The enum is parameterized by the tangent carrier `V` *and* the factor carrier `F` of the linear program it is
/// staged in, mirroring [`LinearArrayOperation`]'s factor parameter: captured primal payloads (the prefix inputs of
/// a linear jitted call and the captured global primals of a linear shard-map) are stored as `F` factors, so they
/// participate in the residual-factor machinery ([`FactorParameterizedOperation`] and everything built on it, such
/// as residual compaction, rebasing onto an enclosing linearization context, and instantiation into a directly
/// executable program). Residualized pushforwards use [`CapturedFactor`] as `F`; direct (instantiated) programs use
/// the concrete value type, in which case `F` defaults to `V`.
///
/// The [`Operation`]/[`Display`], per-variant [`From`]/[`TryFrom`], and the per-variant interpretation delegation below
/// are uniform per-variant delegations to the backing operation structs. Transposition (which targets the factor-split
/// linear universe) and factor mapping are bespoke and hand-written further below, since neither is a uniform
/// per-variant delegation.
#[derive(Clone, Debug)]
pub enum LinearXlaOperationExtension<V: Value<ArrayType>, F: Value<ArrayType> = V> {
    /// Linearized call to a jitted XLA sub-program.
    LinearJitCall(Box<LinearJitCallOperation<F>>),

    /// XLA-specific linear `shard_map`.
    LinearShardMap(Box<LinearShardMapOperation<V, F>>),
}

impl<V: Value<ArrayType>, F: Value<ArrayType>> Operation<ArrayType> for LinearXlaOperationExtension<V, F> {
    fn name(&self) -> &'static str {
        match self {
            Self::LinearJitCall(operation) => operation.name(),
            Self::LinearShardMap(operation) => operation.name(),
        }
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        match self {
            Self::LinearJitCall(operation) => operation.infer_output_types(input_types),
            Self::LinearShardMap(operation) => operation.infer_output_types(input_types),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::LinearJitCall(operation) => operation.render(formatter, indentation),
            Self::LinearShardMap(operation) => operation.render(formatter, indentation),
        }
    }
}

impl<V: Value<ArrayType>, F: Value<ArrayType>> Display for LinearXlaOperationExtension<V, F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<V: Value<ArrayType>, F: Value<ArrayType>> From<LinearJitCallOperation<F>> for LinearXlaOperationExtension<V, F> {
    fn from(operation: LinearJitCallOperation<F>) -> Self {
        Self::LinearJitCall(Box::new(operation))
    }
}

impl<'a, V: Value<ArrayType>, F: Value<ArrayType>> TryFrom<&'a LinearXlaOperationExtension<V, F>>
    for &'a LinearJitCallOperation<F>
{
    type Error = ();

    fn try_from(value: &'a LinearXlaOperationExtension<V, F>) -> Result<Self, ()> {
        match value {
            LinearXlaOperationExtension::LinearJitCall(operation) => Ok(&**operation),
            _ => Err(()),
        }
    }
}

impl<V: Value<ArrayType>, F: Value<ArrayType>> From<LinearShardMapOperation<V, F>>
    for LinearXlaOperationExtension<V, F>
{
    fn from(operation: LinearShardMapOperation<V, F>) -> Self {
        Self::LinearShardMap(Box::new(operation))
    }
}

impl<'a, V: Value<ArrayType>, F: Value<ArrayType>> TryFrom<&'a LinearXlaOperationExtension<V, F>>
    for &'a LinearShardMapOperation<V, F>
{
    type Error = ();

    fn try_from(value: &'a LinearXlaOperationExtension<V, F>) -> Result<Self, ()> {
        match value {
            LinearXlaOperationExtension::LinearShardMap(operation) => Ok(&**operation),
            _ => Err(()),
        }
    }
}

/// Per-variant interpretation delegation. Each backing operation interprets over the tangent carrier `V` only when the
/// factor carrier `F` equals `V` (concrete values rather than residual factors); the per-variant bounds capture that.
impl<V: Value<ArrayType>, F: Value<ArrayType>> InterpretableOperation<ArrayType, V>
    for LinearXlaOperationExtension<V, F>
where
    LinearJitCallOperation<F>: InterpretableOperation<ArrayType, V>,
    LinearShardMapOperation<V, F>: InterpretableOperation<ArrayType, V>,
{
    fn interpret(
        &self,
        context: &<V as Value<ArrayType>>::InterpretationContext,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        match self {
            Self::LinearJitCall(operation) => operation.interpret(context, inputs),
            Self::LinearShardMap(operation) => operation.interpret(context, inputs),
        }
    }
}

/// Linear staged-op universe owned by the XLA backend.
pub type LinearXlaOperation<V, C = V, Factor = V> =
    LinearArrayOperation<ArrayType, V, C, LinearXlaOperationExtension<V, Factor>, Factor, XlaOperation>;

/// [`LinearXlaOperation`] with the extension's factor carrier decoupled from the universe's factor carrier.
///
/// Scan bodies pin their factor payloads to the scan-local `CapturedFactor` namespace while extension captures stay
/// in the enclosing factor space, so transposing an extension operation inside a scan body targets a universe whose
/// factor (`UniverseFactor`) differs from the extension's own (`Factor`). The XLA transposition rules are
/// implemented against this split form; [`LinearXlaOperation`] is the aligned special case
/// `UniverseFactor = Factor`.
pub(crate) type FactorSplitLinearXlaOperation<V, Factor, UniverseFactor> = LinearArrayOperation<
    ArrayType,
    V,
    XlaConstant,
    LinearXlaOperationExtension<V, Factor>,
    UniverseFactor,
    XlaOperation,
>;

/// Staged call to a flat jitted XLA program.
#[derive(Clone, Debug)]
pub struct JitCallOperation {
    /// Flat callee program called by this operation. Shared via [`Rc`] so repeated calls staged from the same
    /// function handle carry one program and remain identity-comparable for call-site deduplication at lowering.
    program: Rc<FlatXlaProgram>,
}

impl JitCallOperation {
    /// Creates a staged jitted-call operation for `program`.
    #[inline]
    pub(crate) fn new(program: Rc<FlatXlaProgram>) -> Self {
        Self { program }
    }

    /// Returns the flat callee program.
    #[inline]
    pub(crate) fn program(&self) -> &FlatXlaProgram {
        self.program.as_ref()
    }

    /// Returns the shared handle to the flat callee program, used for call-site deduplication at lowering.
    #[inline]
    pub(crate) fn program_rc(&self) -> &Rc<FlatXlaProgram> {
        &self.program
    }
}

/// Linearized jitted call used inside tangent and cotangent programs.
///
/// The captured primal prefix inputs are stored as factors of the linear program's factor carrier `F`
/// ([`CapturedFactor`] references in residualized pushforwards, concrete values in instantiated direct programs),
/// so they flow through residual compaction, rebasing, and instantiation like every other captured primal factor.
#[derive(Clone, Debug)]
pub struct LinearJitCallOperation<F: Value<ArrayType>> {
    /// Program applied by this linear call. Its inputs are `captured_inputs` followed by the operation inputs.
    /// Shared via [`Rc`] so transposed clones carry one program and remain identity-comparable for call-site
    /// deduplication at lowering.
    program: Rc<FlatXlaProgram>,

    /// Program for the transposed linear call with the same captured prefix inputs.
    transpose_program: Rc<FlatXlaProgram>,

    /// Captured primal prefix inputs supplied to `program` before the linear operation inputs, stored as factors.
    captured_inputs: Vec<F>,

    /// Flat linear input types expected by this operation.
    input_types: Vec<ArrayType>,

    /// Flat output types produced by this operation.
    output_types: Vec<ArrayType>,
}

impl<F: Value<ArrayType>> LinearJitCallOperation<F> {
    /// Creates a linear jitted-call operation.
    fn new(
        program: Rc<FlatXlaProgram>,
        transpose_program: Rc<FlatXlaProgram>,
        captured_inputs: Vec<F>,
        input_types: Vec<ArrayType>,
        output_types: Vec<ArrayType>,
    ) -> Self {
        Self { program, transpose_program, captured_inputs, input_types, output_types }
    }

    /// Returns the flat transformed callee program.
    #[inline]
    pub(crate) fn program(&self) -> &FlatXlaProgram {
        self.program.as_ref()
    }

    /// Returns captured prefix inputs supplied before the operation's explicit inputs.
    #[inline]
    pub(crate) fn captured_inputs(&self) -> &[F] {
        self.captured_inputs.as_slice()
    }

    /// Maps this call's captured prefix factors through `map_factor`, preserving the carried programs and types.
    fn map_captured_inputs<MappedFactor: Value<ArrayType>, MapFactorFn>(
        &self,
        map_factor: &mut MapFactorFn,
    ) -> Result<LinearJitCallOperation<MappedFactor>, ProgramError>
    where
        MapFactorFn: FnMut(&F) -> Result<MappedFactor, ProgramError>,
    {
        Ok(LinearJitCallOperation::new(
            self.program.clone(),
            self.transpose_program.clone(),
            self.captured_inputs.iter().map(&mut *map_factor).collect::<Result<Vec<_>, _>>()?,
            self.input_types.clone(),
            self.output_types.clone(),
        ))
    }
}

fn missing_traced_input() -> ProgramError {
    ProgramError::InvalidInputCount { expected: 1, actual: 0 }
}

macro_rules! delegate_extension {
    ($self:expr, [$($variant:ident),+ $(,)?], |$operation:ident| $body:expr) => {
        match $self {
            $(Self::$variant($operation) => $body,)+
        }
    };
}

fn ensure_call_input_types(
    operation_name: &'static str,
    expected_types: &[ArrayType],
    input_types: &[ArrayType],
) -> Result<(), TypeError> {
    if expected_types.len() != input_types.len() {
        return Err(TypeError {
            message: format!(
                "{operation_name} expected {} input(s) but got {}",
                expected_types.len(),
                input_types.len(),
            ),
        });
    }
    for (index, (expected, actual)) in expected_types.iter().zip(input_types).enumerate() {
        if expected != actual {
            return Err(TypeError {
                message: format!("{operation_name} input #{index} expected {expected} but got {actual}"),
            });
        }
    }
    Ok(())
}

fn build_jvp_call_program(program: &FlatXlaProgram) -> Result<FlatXlaProgram, ProgramError> {
    let input_types = program.input_types();
    let signature = input_types.iter().cloned().chain(input_types.iter().cloned()).collect::<Vec<_>>();
    let token = XlaDomain::token();
    let (_, traced): (Vec<ArrayType>, FlatXlaProgram) = TracingContext::trace(
        token,
        |inputs: Vec<XlaTracer<'static, 'static>>| -> Result<Vec<XlaTracer<'static, 'static>>, ProgramError> {
            let input_count = inputs.len() / 2;
            let primals = inputs[..input_count].to_vec();
            let tangents = inputs[input_count..].to_vec();
            let context = inputs.first().ok_or_else(missing_traced_input)?.context().clone();
            let (_, pushforward) = context.linearize(
                |linearized_inputs| {
                    let linearization_context =
                        linearized_inputs.first().ok_or_else(missing_traced_input)?.context().clone();
                    linearization_context.stage_program(program, linearized_inputs)
                },
                primals,
            )?;
            pushforward.apply(&context, tangents)
        },
        signature,
    )?;
    traced.into_simplified()
}

fn build_pullback_call_program(program: &FlatXlaProgram) -> Result<FlatXlaProgram, ProgramError> {
    let input_types = program.input_types();
    let output_types = program.output_types();
    let signature = input_types.iter().cloned().chain(output_types.iter().cloned()).collect::<Vec<_>>();
    let token = XlaDomain::token();
    let (_, traced): (Vec<ArrayType>, FlatXlaProgram) = TracingContext::trace(
        token,
        |inputs: Vec<XlaTracer<'static, 'static>>| -> Result<Vec<XlaTracer<'static, 'static>>, ProgramError> {
            let input_count = input_types.len();
            let primals = inputs[..input_count].to_vec();
            let cotangents = inputs[input_count..].to_vec();
            let context = inputs.first().ok_or_else(missing_traced_input)?.context().clone();
            let (_, pullback) = context.vjp(
                |linearized_inputs| {
                    let linearization_context =
                        linearized_inputs.first().ok_or_else(missing_traced_input)?.context().clone();
                    linearization_context.stage_program(program, linearized_inputs)
                },
                primals,
            )?;
            let interpretation_context = cotangents
                .iter()
                .find_map(|cotangent| cotangent.interpretation_context())
                .ok_or_else(missing_traced_input)?;
            pullback.interpret_in_context(&interpretation_context, cotangents)
        },
        signature,
    )?;
    traced.into_simplified()
}

fn build_batched_call_program(
    program: &FlatXlaProgram,
    input_axes: &[Option<usize>],
    axis_size: usize,
) -> Result<(FlatXlaProgram, Vec<Option<usize>>), ProgramError> {
    let logical_input_types = program.input_types();
    check_count!("input", input_axes, logical_input_types.len(), ProgramError);
    let physical_input_types = logical_input_types
        .iter()
        .zip(input_axes)
        .map(|(logical_type, axis)| match axis {
            Some(axis) => logical_type.with_inserted_dimension(*axis, Size::Static(axis_size)),
            None => Ok(logical_type.clone()),
        })
        .collect::<Result<Vec<_>, _>>()?;

    let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
    let parent_context = TracingContext::new(XlaDomain::token(), builder.clone());
    let batching_context = BatchingContext::new(parent_context, axis_size);
    let mut input_tracers = Vec::with_capacity(physical_input_types.len());
    for ((physical_type, logical_type), axis) in physical_input_types.iter().zip(&logical_input_types).zip(input_axes) {
        let atom = builder.borrow_mut().add_input(physical_type.clone());
        batching_context.register_axis(atom, *axis);
        input_tracers.push(batching_context.tracer(atom, Some(logical_type.clone())));
    }
    let output_tracers = batching_context.stage_program(program, input_tracers)?;
    let output_atom_ids = output_tracers.iter().map(Tracer::atom_id).collect::<Result<Vec<_>, _>>()?;
    let output_axes = output_atom_ids.iter().map(|atom| batching_context.axis_for(*atom)).collect::<Vec<_>>();
    drop(output_tracers);
    drop(batching_context);

    let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
    let batched_program = builder
        .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
            output_atom_ids,
            vec![Placeholder; physical_input_types.len()],
            vec![Placeholder; output_axes.len()],
        )?
        .into_simplified()?;
    Ok((batched_program, output_axes))
}

impl Operation<ArrayType> for JitCallOperation {
    #[inline]
    fn name(&self) -> &'static str {
        "jit_call"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        ensure_call_input_types(self.name(), self.program.input_types().as_slice(), input_types)?;
        Ok(self.program.output_types())
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("inputs", self.program.input_ids().len())?;
            operation.field("outputs", self.program.output_ids().len())
        })
    }
}

impl JitCallOperation {
    /// Stages this call operation into the active XLA tracing context carried by `inputs`.
    fn interpret_traced_with_context(
        &self,
        inputs: &[XlaTracer<'static, 'static>],
    ) -> Result<Vec<XlaTracer<'static, 'static>>, ProgramError> {
        let context = inputs.first().ok_or_else(missing_traced_input)?.context().clone();
        context.stage_operation(XlaOperation::Extension(XlaOperationExtension::JitCall(Box::new(self.clone()))), inputs)
    }

    /// Creates the linear call operation corresponding to this ordinary call, capturing the primal inputs as
    /// `captured_inputs` factors.
    fn linear_call_operation<F: Value<ArrayType>>(
        &self,
        captured_inputs: Vec<F>,
    ) -> Result<LinearJitCallOperation<F>, ProgramError> {
        Ok(LinearJitCallOperation::new(
            Rc::new(build_jvp_call_program(self.program())?),
            Rc::new(build_pullback_call_program(self.program())?),
            captured_inputs,
            self.program.input_types(),
            self.program.output_types(),
        ))
    }

    /// Returns the call operation and output-axis metadata for batching this call.
    fn batched_call_operation<V: Typed<ArrayType>>(
        &self,
        inputs: &[ArrayBatch<V>],
    ) -> Result<(Self, Vec<Option<usize>>), ProgramError> {
        let (_, input_axes, axis_size) = batch_input_metadata(inputs)?;
        let axis_size = input_axes.iter().any(Option::is_some).then_some(axis_size);
        match axis_size {
            Some(axis_size) => {
                let (batched_program, output_axes) =
                    build_batched_call_program(&self.program, input_axes.as_slice(), axis_size)?;
                Ok((JitCallOperation::new(Rc::new(batched_program)), output_axes))
            }
            None => Ok((self.clone(), vec![None; self.program.output_types().len()])),
        }
    }

    /// Completes the JVP rule after the caller has produced primal outputs in its host representation.
    ///
    /// The primal inputs are captured as residual factors through [`JvpTracer::factor`] — environment references
    /// under reusable (staged) linearization, closed constants under direct execution — so the staged linear call
    /// participates in residual compaction, rebasing, and instantiation. The primal and tangent carriers are kept
    /// separate so the rule also serves nested symbolic linearization contexts, whose primal values are nested
    /// tracers while tangents stay in the enclosing context's representation.
    fn jvp_from_primal_outputs<'jvp, E, PrimalValue, TangentValue>(
        &self,
        context: &mut TangentContext<'jvp, E>,
        inputs: &[JvpTracer<'jvp, E>],
        primal_outputs: Vec<PrimalValue>,
    ) -> Result<Vec<JvpTracer<'jvp, E>>, ProgramError>
    where
        PrimalValue: Value<ArrayType>,
        TangentValue: Value<ArrayType>,
        E: DifferentiationContext<
                Tangent = TangentValue,
                LinearOperation<TangentValue, CapturedFactor<ArrayType, PrimalValue>> = LinearXlaOperation<
                    TangentValue,
                    XlaConstant,
                    CapturedFactor<ArrayType, PrimalValue>,
                >,
            > + Domain<Type = ArrayType, Value = PrimalValue>
            + 'jvp,
        LinearOperationOf<E>: From<ZeroOperation<ArrayType>>,
    {
        let captured_inputs = inputs.iter().map(|input| input.factor(context)).collect::<Vec<_>>();
        let tangent_inputs = inputs
            .iter()
            .map(|input| context.materialize_tangent(input.tangent().clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let linear_operation = self.linear_call_operation(captured_inputs)?;
        let operation: LinearXlaOperation<TangentValue, XlaConstant, CapturedFactor<ArrayType, PrimalValue>> =
            LinearXlaOperation::Extension(LinearXlaOperationExtension::LinearJitCall(Box::new(linear_operation)));
        let tangent_outputs = context.stage_operation(operation, tangent_inputs.as_slice())?;
        check_count!("output", tangent_outputs, primal_outputs.len(), ProgramError);
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| JvpTracer::from_value(primal, tangent))
            .collect())
    }
}

impl<C> BatchableOperation<ArrayType, C> for JitCallOperation {
    fn batch(
        &self,
        _context: &C,
        inputs: &[ArrayBatch<ArrayType>],
    ) -> Result<Vec<ArrayBatch<ArrayType>>, ProgramError> {
        let physical_inputs = inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
        let (operation, output_axes) = self.batched_call_operation(inputs)?;
        let outputs = operation.infer_output_types(physical_inputs.as_slice())?;
        outputs
            .into_iter()
            .zip(output_axes)
            .map(|(output, axis)| ArrayBatch::new(output.r#type().into_owned(), output, axis))
            .collect()
    }
}

impl<S, C> BatchableOperation<Tracer<S>, C> for JitCallOperation
where
    S: StagingContext<Type = ArrayType, Operation = XlaOperation>,
{
    fn batch(
        &self,
        _context: &C,
        inputs: &[ArrayBatch<Tracer<S>>],
    ) -> Result<Vec<ArrayBatch<Tracer<S>>>, ProgramError> {
        let context = inputs.first().ok_or_else(missing_traced_input)?.value().context().clone();
        let physical_inputs = inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
        let (operation, output_axes) = self.batched_call_operation(inputs)?;
        let outputs = context.stage_operation(
            XlaOperation::Extension(XlaOperationExtension::JitCall(Box::new(operation))),
            physical_inputs.as_slice(),
        )?;
        outputs
            .into_iter()
            .zip(output_axes)
            .map(|(output, axis)| ArrayBatch::new(output.r#type().into_owned(), output, axis))
            .collect()
    }
}

/// Forward-mode rule for staged jitted calls against any staging differentiation context: the primal `jit_call` is
/// staged into the context's primal program through [`TangentContext::bind_primal`] and the linear call captures
/// the primal inputs as residual factors. This serves both ordinary XLA tracing contexts and nested symbolic
/// linearization contexts.
impl<E> DifferentiableOperation<E> for JitCallOperation
where
    E: StagingContext<Type = ArrayType, Constant = XlaConstant, Operation = XlaOperation>
        + DifferentiationContext<
            LinearOperation<
                <E as DifferentiationContext>::Tangent,
                CapturedFactor<ArrayType, Tracer<E>>,
            > = LinearXlaOperation<
                <E as DifferentiationContext>::Tangent,
                XlaConstant,
                CapturedFactor<ArrayType, Tracer<E>>,
            >,
        >,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, E>,
        inputs: &[JvpTracer<'jvp, E>],
    ) -> Result<Vec<JvpTracer<'jvp, E>>, ProgramError>
    where
        E: 'jvp,
        LinearOperationOf<E>: From<ZeroOperation<ArrayType>>,
    {
        check_count!("input", inputs, self.program.input_types().len(), ProgramError);
        let primals = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal_outputs = context.bind_primal(
            XlaOperation::Extension(XlaOperationExtension::JitCall(Box::new(self.clone()))),
            primals.as_slice(),
        )?;
        self.jvp_from_primal_outputs(context, inputs, primal_outputs)
    }
}

impl<F: Value<ArrayType>> Operation<ArrayType> for LinearJitCallOperation<F> {
    #[inline]
    fn name(&self) -> &'static str {
        "linear_jit_call"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        ensure_call_input_types(self.name(), self.input_types.as_slice(), input_types)?;
        Ok(self.output_types.clone())
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("captured", self.captured_inputs.len())?;
            operation.field("inputs", self.input_types.len())?;
            operation.field("outputs", self.output_types.len())
        })
    }
}

impl<C> InterpretableOperation<ArrayType, Tracer<C>> for LinearJitCallOperation<Tracer<C>>
where
    C: StagingContext<Type = ArrayType, Operation = XlaOperation>,
{
    fn interpret(&self, _context: &C, inputs: &[Tracer<C>]) -> Result<Vec<Tracer<C>>, ProgramError> {
        let context = self
            .captured_inputs
            .first()
            .or_else(|| inputs.first())
            .ok_or_else(missing_traced_input)?
            .context()
            .clone();
        let full_inputs = self.captured_inputs.iter().cloned().chain(inputs.iter().cloned()).collect::<Vec<_>>();
        context.stage_operation(
            XlaOperation::Extension(XlaOperationExtension::JitCall(Box::new(JitCallOperation::new(
                self.program.clone(),
            )))),
            full_inputs.as_slice(),
        )
    }
}

impl<V: Value<ArrayType>, Factor: Value<ArrayType>, UniverseFactor: Value<ArrayType>>
    TransposableOperation<ArrayType, V, FactorSplitLinearXlaOperation<V, Factor, UniverseFactor>>
    for LinearJitCallOperation<Factor>
where
    FactorSplitLinearXlaOperation<V, Factor, UniverseFactor>: From<ZeroOperation<ArrayType>>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<
            'transpose,
            ArrayType,
            V,
            FactorSplitLinearXlaOperation<V, Factor, UniverseFactor>,
        >,
        _input_types: &[&ArrayType],
        output_cotangents: &[Cotangent<
            'transpose,
            ArrayType,
            V,
            FactorSplitLinearXlaOperation<V, Factor, UniverseFactor>,
        >],
    ) -> Result<
        Vec<Cotangent<'transpose, ArrayType, V, FactorSplitLinearXlaOperation<V, Factor, UniverseFactor>>>,
        ProgramError,
    > {
        check_count!("output", output_cotangents, self.output_types.len(), ProgramError);
        let mut cotangent_inputs = Vec::with_capacity(output_cotangents.len());
        for (cotangent, output_type) in output_cotangents.iter().zip(self.output_types.iter()) {
            match cotangent {
                Cotangent::Staged(cotangent) => cotangent_inputs.push(cotangent.clone()),
                Cotangent::Zero => {
                    let zero_outputs = context.stage_operation(
                        FactorSplitLinearXlaOperation::<V, Factor, UniverseFactor>::from(ZeroOperation::new(
                            output_type.clone(),
                        )),
                        &[] as &[ryft_core::tracing::AbstractTracer<
                            'transpose,
                            ArrayType,
                            V,
                            FactorSplitLinearXlaOperation<V, Factor, UniverseFactor>,
                        >],
                    )?;
                    check_count!("output", zero_outputs, 1, ProgramError);
                    cotangent_inputs.push(zero_outputs[0].clone());
                }
            }
        }
        let transposed = LinearJitCallOperation::new(
            self.transpose_program.clone(),
            self.program.clone(),
            self.captured_inputs.clone(),
            self.output_types.clone(),
            self.input_types.clone(),
        );
        let input_cotangents = context.stage_operation(
            LinearArrayOperation::Extension(LinearXlaOperationExtension::LinearJitCall(Box::new(transposed))),
            cotangent_inputs.as_slice(),
        )?;
        Ok(input_cotangents.into_iter().map(Cotangent::Staged).collect())
    }
}

impl InterpretableOperation<ArrayType, XlaTracer<'static, 'static>> for XlaOperationExtension<XlaConstant> {
    fn interpret(
        &self,
        _context: &<XlaTracer<'static, 'static> as Value<ArrayType>>::InterpretationContext,
        inputs: &[XlaTracer<'static, 'static>],
    ) -> Result<Vec<XlaTracer<'static, 'static>>, ProgramError> {
        match self {
            Self::JitCall(op) => op.interpret_traced_with_context(inputs),
            Self::ShardMap(op) => {
                let exemplar = inputs.first().ok_or_else(missing_traced_input)?;
                op.interpret_traced_with_context(exemplar.builder().clone(), inputs)
            }
            Self::LinearShardMap(op) => {
                let exemplar = inputs.first().ok_or_else(missing_traced_input)?;
                op.interpret_traced_with_context(exemplar.builder().clone(), inputs)
            }
        }
    }
}

/// Batching rules for the XLA-specific extension variants.
///
/// `ShardMap` and `LinearShardMap` carry inner programs whose proper batching requires lifting the captured body
/// through context-aware batching rules per instruction. These variants return
/// [`BatchingError::UnsupportedOperation`] so that programs which use shard_map can't be silently mis-batched;
/// programs that don't touch these ops batch correctly through this trait impl.
impl<V, C> BatchableOperation<V, C> for XlaOperationExtension<XlaConstant>
where
    V: Value<ArrayType>,
    JitCallOperation: BatchableOperation<V, C>,
{
    fn batch(&self, context: &C, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        match self {
            Self::JitCall(op) => op.batch(context, inputs),
            _ => Err(BatchingError::UnsupportedOperation {
                message: format!("missing batching rule for operation '{}'", self.name()),
            }
            .into()),
        }
    }
}

/// Forward-mode rule for the XLA extension variants against any staging differentiation context whose linear
/// operations are the XLA backend's, covering both ordinary XLA tracing contexts and the
/// [`LinearizationContext`]s derived from them by nested symbolic linearization (the contexts that
/// higher-order JVP rules and [`linearize_program`](ryft_core::tracing_v2::linearize_program) use to linearize
/// captured programs without primal values).
///
/// Every arm stages its primal operation through [`TangentContext::bind_primal`] — splicing it into the active
/// trace in an ordinary tracing context and into the nested primal program under nested symbolic linearization —
/// and captures the primal payloads its linear operation needs as residual factors through [`JvpTracer::factor`]
/// (environment references under reusable linearization, closed constants under direct execution).
impl<E> DifferentiableOperation<E> for XlaOperationExtension<XlaConstant>
where
    E: StagingContext<Type = ArrayType, Constant = XlaConstant, Operation = XlaOperation>
        + DifferentiationContext<
            LinearOperation<
                <E as DifferentiationContext>::Tangent,
                CapturedFactor<ArrayType, Tracer<E>>,
            > = LinearXlaOperation<
                <E as DifferentiationContext>::Tangent,
                XlaConstant,
                CapturedFactor<ArrayType, Tracer<E>>,
            >,
        >,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, E>,
        inputs: &[JvpTracer<'jvp, E>],
    ) -> Result<Vec<JvpTracer<'jvp, E>>, ProgramError>
    where
        E: 'jvp,
        LinearOperationOf<E>: From<ZeroOperation<ArrayType>>,
    {
        match self {
            Self::JitCall(op) => op.jvp(context, inputs),
            Self::ShardMap(op) => op.jvp_with_staging_context(context, inputs),
            Self::LinearShardMap(op) => op.jvp_with_staging_context(context, inputs),
        }
    }
}

impl<V: Value<ArrayType>, Factor: Value<ArrayType>, UniverseFactor: Value<ArrayType>>
    TransposableOperation<ArrayType, V, FactorSplitLinearXlaOperation<V, Factor, UniverseFactor>>
    for LinearXlaOperationExtension<V, Factor>
where
    LinearShardMapOperation<V, Factor>:
        TransposableOperation<ArrayType, V, FactorSplitLinearXlaOperation<V, Factor, UniverseFactor>>,
    LinearJitCallOperation<Factor>:
        TransposableOperation<ArrayType, V, FactorSplitLinearXlaOperation<V, Factor, UniverseFactor>>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<
            'transpose,
            ArrayType,
            V,
            FactorSplitLinearXlaOperation<V, Factor, UniverseFactor>,
        >,
        input_types: &[&ArrayType],
        output_cotangents: &[Cotangent<
            'transpose,
            ArrayType,
            V,
            FactorSplitLinearXlaOperation<V, Factor, UniverseFactor>,
        >],
    ) -> Result<
        Vec<Cotangent<'transpose, ArrayType, V, FactorSplitLinearXlaOperation<V, Factor, UniverseFactor>>>,
        ProgramError,
    > {
        delegate_extension!(self, [LinearJitCall, LinearShardMap], |op| {
            op.transpose(context, input_types, output_cotangents)
        })
    }
}

/// Maps the captured primal factor payloads carried by the XLA linear extension operations, so they participate in
/// residual compaction, rebasing onto an enclosing linearization context, and instantiation into directly
/// executable linear programs exactly like the core linear array operations' factor payloads.
impl<V: Value<ArrayType>, F: Value<ArrayType>> FactorParameterizedOperation<ArrayType, F>
    for LinearXlaOperationExtension<V, F>
{
    type WithFactor<MappedFactor: Value<ArrayType>> = LinearXlaOperationExtension<V, MappedFactor>;

    fn try_map_factors<MappedFactor: Value<ArrayType>, MapFactorFn>(
        &self,
        map_factor: &mut MapFactorFn,
    ) -> Result<Self::WithFactor<MappedFactor>, ProgramError>
    where
        MapFactorFn: FnMut(&F) -> Result<MappedFactor, ProgramError>,
    {
        match self {
            Self::LinearJitCall(operation) => {
                Ok(LinearXlaOperationExtension::LinearJitCall(Box::new(operation.map_captured_inputs(map_factor)?)))
            }
            Self::LinearShardMap(operation) => Ok(LinearXlaOperationExtension::LinearShardMap(Box::new(
                operation.map_captured_global_primals(map_factor)?,
            ))),
        }
    }
}
