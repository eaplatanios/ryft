use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::rc::Rc;

use ryft_core::contexts::{Context, StagingContext};
use ryft_core::differentiation::{Cotangent, TransposableOperation};
use ryft_core::domains::Domain;
use ryft_core::macros::check_count;
use ryft_core::operations::constants::ZeroOperation;
use ryft_core::operations::{BooleanLike, InterpretableOperation, Operation};
use ryft_core::parameters::{Parameterized, ParameterizedFamily};
use ryft_core::programs::{AtomId, ProgramError, Value};
use ryft_core::sharding::{LogicalMesh, MeshAxisType, Sharding};
use ryft_core::tracing::{AbstractTracer, AbstractTracingContext, DomainTracer, Tracer, TracingContext};
use ryft_core::tracing_v2::differentiation::JvpTracer;
use ryft_core::tracing_v2::{
    DifferentiableOperation, DifferentiationContext, LinearOperationOf, ResidualizedOperation, TangentContext,
    ValueOrCapture,
};
use ryft_core::types::{ArrayType, TypeError, Typed};

use crate::experimental::domains::XlaDomain;
use crate::experimental::ops::{FlatXlaProgram, LinearXlaOperation, XlaConstant, XlaOperation, XlaProgramBuilder};
use crate::experimental::shard_map::{
    FlatTracedShardMap, ShardMap, ShardMapInvocationLeaf, ShardMapLocalTraceInput, ShardMapLocalTraceOutput,
    ShardMapTraceError, ShardMapTracer, TracedShardMap,
};

/// Source from which one factorized transpose residual apply input is obtained.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum FactorizedTransposeResidualSource {
    /// Residual value is forwarded from the captured primal input at `index`.
    CapturedInput { index: usize },

    /// Residual value is produced by the residual body output at `index`.
    ResidualOutput { index: usize },
}

/// Source from which one factorized transpose output (an input cotangent) is obtained.
///
/// The apply body computes every input cotangent, including the `zero` operations the transpose materializes for
/// structural zeros, so outputs are normally [`ApplyOutput`](Self::ApplyOutput). [`Constant`](Self::Constant) covers the
/// case where the transpose surfaces a closed constant atom directly rather than through the apply body.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum FactorizedTransposeOutputSource {
    /// Output value is the closed constant `value` surfaced directly by the transpose.
    Constant { value: XlaConstant },

    /// Output value is produced by the compact apply body output at `index`.
    ApplyOutput { index: usize },
}

/// Two-stage transpose factorization for one linear shard-map body.
#[derive(Clone, Debug)]
pub(crate) struct FactorizedTransposeShardMapBodies {
    /// Primals-only residual computation staged as its own shard-map body.
    residual_body: FlatTracedShardMap,

    /// Original primal input indices consumed by [`Self::residual_body`].
    residual_input_indices: Vec<usize>,

    /// Source for each residual apply input after the cotangent inputs.
    residual_sources: Vec<FactorizedTransposeResidualSource>,

    /// Cotangent application staged separately from the residual computation.
    apply_body: FlatTracedShardMap,

    /// Original cotangent input indices consumed by [`Self::apply_body`] before residual values.
    apply_input_indices: Vec<usize>,

    /// Source for each original transpose output.
    output_sources: Vec<FactorizedTransposeOutputSource>,
}

impl FactorizedTransposeShardMapBodies {
    /// Creates a new [`FactorizedTransposeShardMapBodies`].
    #[inline]
    fn new(
        residual_body: FlatTracedShardMap,
        residual_input_indices: Vec<usize>,
        residual_sources: Vec<FactorizedTransposeResidualSource>,
        apply_body: FlatTracedShardMap,
        apply_input_indices: Vec<usize>,
        output_sources: Vec<FactorizedTransposeOutputSource>,
    ) -> Self {
        Self {
            residual_body,
            residual_input_indices,
            residual_sources,
            apply_body,
            apply_input_indices,
            output_sources,
        }
    }

    /// Returns the primals-only residual shard-map body.
    #[inline]
    pub(crate) fn residual_body(&self) -> &FlatTracedShardMap {
        &self.residual_body
    }

    /// Returns the original primal input indices consumed by [`Self::residual_body`].
    #[inline]
    pub(crate) fn residual_input_indices(&self) -> &[usize] {
        self.residual_input_indices.as_slice()
    }

    /// Returns the source for each residual apply input after the cotangent inputs.
    #[inline]
    pub(crate) fn residual_sources(&self) -> &[FactorizedTransposeResidualSource] {
        self.residual_sources.as_slice()
    }

    /// Returns the cotangent application shard-map body.
    #[inline]
    pub(crate) fn apply_body(&self) -> &FlatTracedShardMap {
        &self.apply_body
    }

    /// Returns the original cotangent input indices consumed by [`Self::apply_body`] before residual values.
    #[inline]
    pub(crate) fn apply_input_indices(&self) -> &[usize] {
        self.apply_input_indices.as_slice()
    }

    /// Returns the source for each original transpose output.
    #[inline]
    pub(crate) fn output_sources(&self) -> &[FactorizedTransposeOutputSource] {
        self.output_sources.as_slice()
    }
}

/// Evaluation mode used by linear shard-map higher-order ops.
#[derive(Clone, Debug)]
pub(crate) enum LinearShardMapEvalMode {
    /// Evaluate the linear shard map by running one fused body.
    Body(FlatTracedShardMap),

    /// Evaluate the transposed linear shard map through residual and apply bodies.
    FactorizedTranspose(FactorizedTransposeShardMapBodies),
}

/// Linear execution state carried by one dedicated linear shard-map op.
///
/// `captured_global_primals` holds the global primal inputs captured when the shard-map body was linearized, in a
/// representation chosen by the `Capture` parameter:
///
///   - Linear shard-map ops staged in tangent/cotangent programs ([`LinearXlaOperation`]) capture the primals as
///     factors of the linear program's factor carrier (residual references in reusable residualized pushforwards,
///     concrete values in instantiated direct programs), so the captures flow through residual compaction, rebasing,
///     and instantiation like every other captured primal factor.
///   - The ordinary staged form ([`XlaOperation`]) captures [`AtomId`]s of the staging program the op is re-staged
///     into; those ids are minted from live tracers at re-staging time and resolved against the program being lowered.
///
/// The vector is empty for tensor-leaf shard-map ops, where captures are never read.
#[derive(Clone, Debug)]
pub(crate) struct LinearShardMapState<Capture = AtomId> {
    /// Global primals captured when the shard-map body was linearized.
    captured_global_primals: Vec<Capture>,

    /// Evaluation strategy used when replaying the forward linear body.
    eval_mode: LinearShardMapEvalMode,

    /// Evaluation strategy used when replaying the transpose body.
    transpose_mode: LinearShardMapEvalMode,
}

impl<Capture> LinearShardMapState<Capture> {
    /// Creates a new [`LinearShardMapState`].
    #[inline]
    fn new(
        captured_global_primals: Vec<Capture>,
        eval_mode: LinearShardMapEvalMode,
        transpose_mode: LinearShardMapEvalMode,
    ) -> Self {
        Self { captured_global_primals, eval_mode, transpose_mode }
    }

    /// Returns the global primals captured when the shard-map body was linearized.
    #[inline]
    pub(crate) fn captured_global_primals(&self) -> &[Capture] {
        &self.captured_global_primals
    }

    /// Returns the evaluation strategy used for forward linear replay.
    #[inline]
    pub(crate) fn eval_mode(&self) -> &LinearShardMapEvalMode {
        &self.eval_mode
    }

    /// Returns the evaluation strategy used for transposed linear replay.
    #[inline]
    #[cfg(feature = "benchmarking")]
    pub(crate) fn transpose_mode(&self) -> &LinearShardMapEvalMode {
        &self.transpose_mode
    }
}

fn missing_linear_shard_map_staging_context() -> ProgramError {
    ProgramError::Type(TypeError {
        message: "linear shard_map with non-empty outputs requires at least one traced input leaf".into(),
    })
}

/// Canonical higher-order shard-map op used for staged tracing, differentiation, and lowering.
#[derive(Clone, Debug)]
pub struct ShardMapOperation<V> {
    /// Canonical erased shard-map body carried by this higher-order op.
    body: FlatTracedShardMap,

    /// Global input types expected by the carried body.
    input_types: Vec<ArrayType>,

    /// Global output types produced by the carried body.
    output_types: Vec<ArrayType>,

    /// Phantom marker tying the op to the traced leaf type it will replay with.
    marker: PhantomData<fn() -> V>,
}

impl<V> ShardMapOperation<V> {
    /// Creates one ordinary staged shard-map op from its erased body payload.
    #[inline]
    pub(crate) fn new(body: FlatTracedShardMap) -> Self {
        Self {
            input_types: body.global_input_types().to_vec(),
            output_types: body.global_output_types().to_vec(),
            body,
            marker: PhantomData,
        }
    }

    /// Returns the erased shard-map body carried by this operation.
    #[inline]
    pub(crate) fn body(&self) -> &FlatTracedShardMap {
        &self.body
    }
}

/// Canonical linear shard-map op used in tangent/cotangent programs and traced linear replay.
///
/// `V` is the traced leaf type the op replays with and `Capture` is the captured-primal representation carried by
/// the op's [`LinearShardMapState`] (factors in linear programs, [`AtomId`]s in the ordinary staged form; refer to
/// the state type's documentation).
#[derive(Clone, Debug)]
pub struct LinearShardMapOperation<V, Capture = AtomId> {
    /// Canonical erased primal shard-map body carried by this linear higher-order op.
    body: FlatTracedShardMap,

    /// Global input types expected by the carried body.
    input_types: Vec<ArrayType>,

    /// Global output types produced by the carried body.
    output_types: Vec<ArrayType>,

    /// Linear execution state for replaying this linear shard-map.
    linear_state: LinearShardMapState<Capture>,

    /// Phantom marker tying the op to the traced leaf type it will replay with.
    marker: PhantomData<fn() -> V>,
}

impl<V, Capture> LinearShardMapOperation<V, Capture> {
    /// Creates one linear shard-map op with captured primals and explicit transpose state.
    #[inline]
    fn new(
        body: FlatTracedShardMap,
        captured_global_primals: Vec<Capture>,
        input_types: Vec<ArrayType>,
        output_types: Vec<ArrayType>,
        eval_mode: LinearShardMapEvalMode,
        transpose_mode: LinearShardMapEvalMode,
    ) -> Self {
        Self {
            body,
            input_types,
            output_types,
            linear_state: LinearShardMapState::new(captured_global_primals, eval_mode, transpose_mode),
            marker: PhantomData,
        }
    }

    /// Creates one linear shard-map op like this one but with the provided captured primals, preserving the
    /// carried body, types, and evaluation strategies.
    #[inline]
    pub(crate) fn with_captured_global_primals<V2, Capture2>(
        &self,
        captured_global_primals: Vec<Capture2>,
    ) -> LinearShardMapOperation<V2, Capture2> {
        LinearShardMapOperation::new(
            self.body.clone(),
            captured_global_primals,
            self.input_types.clone(),
            self.output_types.clone(),
            self.linear_state.eval_mode.clone(),
            self.linear_state.transpose_mode.clone(),
        )
    }

    /// Returns the transposed linear shard-map op, carrying the captured primals verbatim.
    fn transpose_op(&self) -> Self
    where
        Capture: Clone,
    {
        Self::new(
            self.body.clone(),
            self.linear_state.captured_global_primals.clone(),
            self.output_types.clone(),
            self.input_types.clone(),
            self.linear_state.transpose_mode.clone(),
            self.linear_state.eval_mode.clone(),
        )
    }

    /// Maps this op's captured primal payloads through `map_capture`, preserving everything else.
    pub(crate) fn map_captured_global_primals<MappedCapture, MapCaptureFn>(
        &self,
        map_capture: &mut MapCaptureFn,
    ) -> Result<LinearShardMapOperation<V, MappedCapture>, ProgramError>
    where
        MapCaptureFn: FnMut(&Capture) -> Result<MappedCapture, ProgramError>,
    {
        let captured_global_primals = self
            .linear_state
            .captured_global_primals
            .iter()
            .map(&mut *map_capture)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self.with_captured_global_primals(captured_global_primals))
    }

    /// Returns the erased primal shard-map body carried by this operation.
    #[inline]
    #[cfg(feature = "benchmarking")]
    pub(crate) fn body(&self) -> &FlatTracedShardMap {
        &self.body
    }

    /// Returns the linear execution state for this shard-map operation.
    #[inline]
    pub(crate) fn linear_state(&self) -> &LinearShardMapState<Capture> {
        &self.linear_state
    }
}

impl<V, C: Context<Type = ArrayType>> LinearShardMapOperation<V, Tracer<C>> {
    /// Rebuilds this value-captured (instantiated) linear shard-map op as the tensor-leaf XLA operation variant,
    /// minting capture atom ids from the captured tracers at re-staging time so the ids always belong to the
    /// staging program the op is re-staged into.
    pub(crate) fn to_tensor_xla_op(&self) -> Result<LinearShardMapOperation<XlaConstant>, ProgramError> {
        let captured_atoms = self
            .linear_state
            .captured_global_primals
            .iter()
            .map(Tracer::atom_id)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self.with_captured_global_primals(captured_atoms))
    }
}

/// Completes a shard-map JVP once the caller has produced primal outputs and the matching linear shard-map
/// operation, whose captured global primals are factors over the context's primal value type. The primal and
/// tangent carriers are kept separate so the helper also serves nested symbolic linearization contexts, whose
/// primal values are nested tracers while tangents stay in the enclosing context's representation.
fn complete_shard_map_jvp<'jvp, E, PrimalValue, TangentValue>(
    context: &mut TangentContext<'jvp, E>,
    inputs: &[JvpTracer<'jvp, E>],
    primal_outputs: Vec<PrimalValue>,
    output_count: usize,
    linear_operation: LinearShardMapOperation<TangentValue, ValueOrCapture<ArrayType, PrimalValue>>,
) -> Result<Vec<JvpTracer<'jvp, E>>, ProgramError>
where
    PrimalValue: Value<ArrayType>,
    TangentValue: Value<ArrayType> + BooleanLike,
    E: DifferentiationContext<
            Tangent = TangentValue,
            LinearOperation<TangentValue, ValueOrCapture<ArrayType, PrimalValue>> = LinearXlaOperation<
                TangentValue,
                XlaConstant,
                ValueOrCapture<ArrayType, PrimalValue>,
            >,
        > + Domain<Type = ArrayType, Value = PrimalValue>
        + 'jvp,
    LinearOperationOf<E>: From<ZeroOperation<ArrayType>>,
{
    check_count!("output", primal_outputs, output_count, ProgramError);
    let tangent_inputs = inputs.iter().map(|input| input.tangent().clone()).collect::<Vec<_>>();
    let operation: LinearXlaOperation<TangentValue, XlaConstant, ValueOrCapture<ArrayType, PrimalValue>> =
        LinearXlaOperation::LinearShardMap(Box::new(linear_operation));
    let tangent_outputs = context.stage_operation(operation, tangent_inputs.as_slice())?;
    check_count!("output", tangent_outputs, output_count, ProgramError);
    Ok(primal_outputs
        .into_iter()
        .zip(tangent_outputs)
        .map(|(primal, tangent)| JvpTracer::from_value(primal, tangent))
        .collect())
}

impl<LeafV> ShardMapOperation<LeafV> {
    /// Applies this shard-map JVP against any staging differentiation context: the primal shard-map op is staged
    /// into `context`'s primal program through [`TangentContext::bind_primal`], the global primal inputs are
    /// captured as residual factors through [`JvpTracer::factor`], and the matching linear shard-map op is staged
    /// into the tangent program. This serves both ordinary XLA tracing contexts and nested symbolic linearization
    /// contexts (whose primal values are nested tracers while tangents stay in the enclosing representation).
    pub(crate) fn jvp_with_staging_context<'jvp, E>(
        &self,
        context: &mut TangentContext<'jvp, E>,
        inputs: &[JvpTracer<'jvp, E>],
    ) -> Result<Vec<JvpTracer<'jvp, E>>, ProgramError>
    where
        E: StagingContext<Type = ArrayType, Constant = XlaConstant, Operation = XlaOperation>
            + DifferentiationContext<
                LinearOperation<
                    <E as DifferentiationContext>::Tangent,
                    ValueOrCapture<ArrayType, Tracer<E>>,
                > = LinearXlaOperation<
                    <E as DifferentiationContext>::Tangent,
                    XlaConstant,
                    ValueOrCapture<ArrayType, Tracer<E>>,
                >,
            > + 'jvp,
        LinearOperationOf<E>: From<ZeroOperation<ArrayType>>,
    {
        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal_outputs = context.bind_primal(
            XlaOperation::ShardMap(Box::new(ShardMapOperation::new(self.body.clone()))),
            primal_inputs.as_slice(),
        )?;
        let captured_global_primals = inputs.iter().map(|input| input.factor(context)).collect::<Vec<_>>();
        let linear_operation =
            make_linear_shard_map(&self.body, captured_global_primals).map_err(trace_error_from_shard_map)?;
        complete_shard_map_jvp(context, inputs, primal_outputs, self.output_types.len(), linear_operation)
    }
}

impl ShardMapOperation<ShardMapTracer> {
    /// Replays this traced-leaf shard-map op into one explicit outer tracing builder.
    fn interpret_with_tracing_builder(
        &self,
        tracing_builder: Rc<RefCell<XlaProgramBuilder>>,
        inputs: &[ShardMapTracer],
    ) -> Result<Vec<ShardMapTracer>, ProgramError> {
        let abstract_inputs = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.infer_output_types(abstract_inputs.as_slice())?;
        apply_flat_traced_shard_map(tracing_builder, self.body.clone(), inputs.to_vec())
            .map_err(trace_error_from_shard_map)
    }

    /// Applies this traced-leaf shard-map JVP using an explicit outer tracing builder and a
    /// [`TangentContext`] for the linear builder.
    pub(crate) fn jvp_with_builders<'jvp, D>(
        &self,
        tracing_builder: Rc<RefCell<XlaProgramBuilder>>,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: Domain<Type = ArrayType, Value = ShardMapTracer>
            + Domain<Type = ArrayType>
            + DifferentiationContext<
                Tangent = ShardMapTracer,
                LinearOperation<ShardMapTracer, ValueOrCapture<ArrayType, ShardMapTracer>> = LinearXlaOperation<
                    ShardMapTracer,
                    XlaConstant,
                    ValueOrCapture<ArrayType, ShardMapTracer>,
                >,
            > + 'jvp,
        LinearOperationOf<D>: From<ZeroOperation<ArrayType>>,
    {
        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal_outputs = self.interpret_with_tracing_builder(tracing_builder, primal_inputs.as_slice())?;
        let captured_global_primals = inputs.iter().map(|input| input.factor(context)).collect::<Vec<_>>();
        let linear_operation =
            make_linear_shard_map(&self.body, captured_global_primals).map_err(trace_error_from_shard_map)?;
        complete_shard_map_jvp(context, inputs, primal_outputs, self.output_types.len(), linear_operation)
    }
}

impl LinearShardMapOperation<XlaConstant> {
    /// Applies this tensor-leaf linear shard-map JVP against any staging differentiation context: the primal
    /// (ordinary) linear shard-map op is staged into `context`'s primal program through
    /// [`TangentContext::bind_primal`] and the tangent op rebinds the current primal inputs as captured residual
    /// factors through [`JvpTracer::factor`].
    pub(crate) fn jvp_with_staging_context<'jvp, E>(
        &self,
        context: &mut TangentContext<'jvp, E>,
        inputs: &[JvpTracer<'jvp, E>],
    ) -> Result<Vec<JvpTracer<'jvp, E>>, ProgramError>
    where
        E: StagingContext<Type = ArrayType, Constant = XlaConstant, Operation = XlaOperation>
            + DifferentiationContext<
                LinearOperation<
                    <E as DifferentiationContext>::Tangent,
                    ValueOrCapture<ArrayType, Tracer<E>>,
                > = LinearXlaOperation<
                    <E as DifferentiationContext>::Tangent,
                    XlaConstant,
                    ValueOrCapture<ArrayType, Tracer<E>>,
                >,
            > + 'jvp,
        LinearOperationOf<E>: From<ZeroOperation<ArrayType>>,
    {
        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal_outputs =
            context.bind_primal(XlaOperation::LinearShardMap(Box::new(self.clone())), primal_inputs.as_slice())?;
        let captured_global_primals = inputs.iter().map(|input| input.factor(context)).collect::<Vec<_>>();
        let linear_operation = self.with_captured_global_primals(captured_global_primals);
        complete_shard_map_jvp(context, inputs, primal_outputs, self.output_types.len(), linear_operation)
    }
}

impl InterpretableOperation<ArrayType, ShardMapTracer> for ShardMapOperation<ShardMapTracer> {
    fn interpret(
        &self,
        _context: &<ShardMapTracer as Value<ArrayType>>::InterpretationContext,
        inputs: &[ShardMapTracer],
    ) -> Result<Vec<ShardMapTracer>, ProgramError> {
        match inputs.first() {
            Some(input) => self.interpret_with_tracing_builder(input.builder().clone(), inputs),
            None => {
                self.infer_output_types(&[])?;
                Ok(Vec::new())
            }
        }
    }
}

impl<'domain, D> InterpretableOperation<ArrayType, DomainTracer<'domain, D>>
    for LinearShardMapOperation<DomainTracer<'domain, D>, DomainTracer<'domain, D>>
where
    D: Domain<Type = ArrayType, Constant = XlaConstant, Operation = XlaOperation>,
{
    fn interpret(
        &self,
        _context: &TracingContext<'domain, D>,
        inputs: &[DomainTracer<'domain, D>],
    ) -> Result<Vec<DomainTracer<'domain, D>>, ProgramError> {
        let context = self
            .linear_state
            .captured_global_primals
            .first()
            .or_else(|| inputs.first())
            .ok_or_else(missing_traced_shard_map_staging_context)?
            .context()
            .clone();
        context.stage_operation(XlaOperation::LinearShardMap(Box::new(self.to_tensor_xla_op()?)), inputs)
    }
}

impl<V> Display for ShardMapOperation<V>
where
    Self: Operation<ArrayType>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl<V, Capture> Display for LinearShardMapOperation<V, Capture> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

/// Returns `true` when two shard-map boundary types agree apart from carried sharding metadata.
fn shard_map_boundary_types_match(actual: &ArrayType, expected: &ArrayType) -> bool {
    fn varying_manual_axes_match(actual: &Sharding, expected: &Sharding) -> bool {
        actual
            .varying_manual_axes()
            .iter()
            .filter(|axis_name| expected.mesh().axis_type(axis_name.as_str()) == Some(MeshAxisType::Manual))
            .eq(expected.varying_manual_axes().iter())
    }

    actual.data_type() == expected.data_type()
        && actual.shape() == expected.shape()
        && actual.layout() == expected.layout()
        && match (actual.sharding(), expected.sharding()) {
            (_, None) => true,
            (Some(actual), Some(expected)) => {
                actual.unreduced_axes() == expected.unreduced_axes()
                    && actual.reduced_axes() == expected.reduced_axes()
                    && varying_manual_axes_match(actual, expected)
            }
            (None, Some(expected)) => {
                expected.unreduced_axes().is_empty()
                    && expected.reduced_axes().is_empty()
                    && expected.varying_manual_axes().is_empty()
            }
        }
}

/// Re-embeds one traced shard-map output type into the caller's ambient sharding envelope.
///
/// Traced nested `shard_map` invocations stage as ordinary higher-order ops inside an already-local
/// caller context. When the caller's ambient sharding envelope differs from the captured shard-map
/// boundary, the staged result must use the ambient envelope again so downstream traced primitives
/// see a value in the surrounding local context rather than in the nested shard-map boundary space.
///
/// This mirrors the JAX intuition that a nested `shard_map` body returns to the enclosing
/// per-instance context after the inner manual region finishes; the inner boundary should not leak
/// out as the ambient type seen by surrounding primitives in the outer body.
fn adapt_traced_shard_map_output_type(
    actual_input_types: &[ArrayType],
    captured_input_types: &[ArrayType],
    captured_output_type: &ArrayType,
) -> ArrayType {
    if actual_input_types.len() == 1
        && captured_input_types.len() == 1
        && actual_input_types[0].sharding() != captured_input_types[0].sharding()
        && actual_input_types[0].shape().rank() == captured_output_type.shape().rank()
    {
        ArrayType::new(captured_output_type.data_type(), captured_output_type.shape().clone())
            .with_layout(captured_output_type.layout().cloned())
            .with_sharding(actual_input_types[0].sharding().cloned())
            .expect("adapted shard_map output type should preserve rank-compatible sharding")
    } else {
        captured_output_type.clone()
    }
}

fn infer_shard_map_output_types(
    operation_name: &'static str,
    captured_input_types: &[ArrayType],
    captured_output_types: &[ArrayType],
    input_types: &[ArrayType],
) -> Result<Vec<ArrayType>, TypeError> {
    check_count!("input", input_types, captured_input_types.len(), TypeError);
    if !input_types
        .iter()
        .zip(captured_input_types.iter())
        .all(|(actual, expected)| shard_map_boundary_types_match(actual, expected))
    {
        return Err(TypeError {
            message: format!("{} input types do not match the captured shard-map boundary", operation_name),
        });
    }
    Ok(captured_output_types
        .iter()
        .map(|output_type| adapt_traced_shard_map_output_type(input_types, captured_input_types, output_type))
        .collect::<Vec<_>>())
}

impl<V: Value<ArrayType>> Operation<ArrayType> for ShardMapOperation<V> {
    #[inline]
    fn name(&self) -> &'static str {
        "shard_map"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        infer_shard_map_output_types(
            self.name(),
            self.input_types.as_slice(),
            self.output_types.as_slice(),
            input_types,
        )
    }
}

impl<V, Capture> Operation<ArrayType> for LinearShardMapOperation<V, Capture> {
    #[inline]
    fn name(&self) -> &'static str {
        "linear_shard_map"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        infer_shard_map_output_types(
            self.name(),
            self.input_types.as_slice(),
            self.output_types.as_slice(),
            input_types,
        )
    }
}

impl<V, Factor, Target> TransposableOperation<ArrayType, V, Target> for LinearShardMapOperation<V, Factor>
where
    V: Value<ArrayType>,
    Factor: Value<ArrayType>,
    Target: Operation<ArrayType> + From<ZeroOperation<ArrayType>> + From<LinearShardMapOperation<V, Factor>>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<'transpose, ArrayType, V, Target>,
        _input_types: &[&ArrayType],
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, Target>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, Target>>, ProgramError> {
        check_count!("output", output_cotangents, self.output_types.len(), ProgramError);
        if output_cotangents.is_empty() {
            return Ok(vec![Cotangent::Zero; self.input_types.len()]);
        }
        if output_cotangents.iter().all(Cotangent::is_zero) {
            return Ok(vec![Cotangent::Zero; self.input_types.len()]);
        }
        let materialized = output_cotangents
            .iter()
            .zip(self.output_types.iter())
            .map(|(cotangent, output_type)| materialize_cotangent(context, cotangent, output_type))
            .collect::<Vec<_>>();
        let contributions = context.stage_operation(Target::from(self.transpose_op()), materialized.as_slice())?;
        check_count!("output", contributions, self.input_types.len(), ProgramError);
        Ok(contributions.into_iter().map(Cotangent::Staged).collect::<Vec<_>>())
    }
}

/// Returns a concrete atom for `cotangent`, staging a typed `Zero` op when the cotangent is
/// structurally zero. Higher-order linear rules use this when they must consume all output
/// cotangents jointly.
fn materialize_cotangent<'transpose, V: Value<ArrayType>, O>(
    context: &AbstractTracingContext<'transpose, ArrayType, V, O>,
    cotangent: &Cotangent<'transpose, ArrayType, V, O>,
    output_type: &ArrayType,
) -> AbstractTracer<'transpose, ArrayType, V, O>
where
    O: Operation<ArrayType> + From<ZeroOperation<ArrayType>>,
{
    match cotangent {
        Cotangent::Staged(cotangent) => return cotangent.clone(),
        Cotangent::Zero => {}
    }
    let builder = &context.builder();
    let mut builder_borrow = builder.borrow_mut();
    let output = builder_borrow.add_variable(output_type.clone());
    builder_borrow.add_instruction_unchecked(ryft_core::programs::Instruction::new(
        O::from(ZeroOperation::new(output_type.clone())),
        vec![],
        vec![output],
    ));
    drop(builder_borrow);
    context.tracer(output, None)
}

impl<D> DifferentiableOperation<D> for ShardMapOperation<ShardMapTracer>
where
    D: Domain<Type = ArrayType, Value = ShardMapTracer>
        + Domain<Type = ArrayType>
        + DifferentiationContext<
            Tangent = ShardMapTracer,
            LinearOperation<ShardMapTracer, ValueOrCapture<ArrayType, ShardMapTracer>> = LinearXlaOperation<
                ShardMapTracer,
                XlaConstant,
                ValueOrCapture<ArrayType, ShardMapTracer>,
            >,
        >,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        let Some(first_input) = inputs.first() else {
            return if self.output_types.is_empty() {
                Ok(Vec::new())
            } else {
                Err(missing_linear_shard_map_staging_context())
            };
        };
        self.jvp_with_builders(first_input.primal().builder().clone(), context, inputs)
    }
}

fn trace_error_from_shard_map(error: ShardMapTraceError) -> ProgramError {
    ProgramError::Type(TypeError { message: error.to_string() })
}

fn missing_traced_shard_map_staging_context() -> ProgramError {
    ProgramError::Type(TypeError {
        message: "traced shard_map with non-empty outputs requires at least one traced input leaf".to_string(),
    })
}

/// Builds an empty shard-map body that consumes no inputs and produces no outputs.
///
/// This is the residual body used when the factorized transpose needs no computed residuals (every residual is either a
/// forwarded primal input or there are no residuals at all). It owns an empty program over the body's mesh so that the
/// lowering consumer can keep treating the residual body uniformly.
fn empty_residual_shard_map_body(body: &FlatTracedShardMap) -> Result<FlatTracedShardMap, ShardMapTraceError> {
    let empty_shard_map = ShardMap::from_shardings(
        body.shard_map().mesh().clone(),
        Vec::new(),
        Vec::new(),
        body.shard_map().manual_axes().to_vec(),
        body.shard_map().check_vma(),
    );
    let empty_program = XlaProgramBuilder::new().build::<Vec<XlaConstant>, Vec<XlaConstant>>(
        Vec::new(),
        Vec::<ryft_core::parameters::Placeholder>::new(),
        Vec::<ryft_core::parameters::Placeholder>::new(),
    )?;
    Ok(FlatTracedShardMap::from_parts(empty_shard_map, Vec::new(), Vec::new(), Vec::new(), Vec::new(), empty_program))
}

/// Factorizes one primal shard-map body's transpose into a primals-only residual stage and a cotangent-application
/// stage, building both directly from the body's [`Pushforward`](ryft_core::tracing_v2::differentiation::Pushforward).
///
/// The primal body is linearized once over fresh local-primal inputs to obtain its residualized pushforward. Each saved
/// residual is classified by its primal-side atom: residuals that are forwarded primal inputs become
/// [`FactorizedTransposeResidualSource::CapturedInput`], while computed residuals become
/// [`FactorizedTransposeResidualSource::ResidualOutput`] and are projected into a dedicated residual body via
/// [`Program::into_filtered`], which also prunes primal inputs that no residual depends on. The pushforward's linear
/// program is then re-pointed at fresh residual inputs through
/// [`ResidualizedOperation::instantiate_residuals`](ryft_core::tracing_v2::differentiation::ResidualizedOperation::instantiate_residuals)
/// and transposed against the body's local-output cotangents. Each resulting input cotangent is recorded as a
/// [`FactorizedTransposeOutputSource::ApplyOutput`] when the apply body computes it (including the `zero` operations the
/// transpose materializes for input cotangents that are structural zeros), and as a
/// [`FactorizedTransposeOutputSource::Constant`] only in the rare case the transpose surfaces a closed constant atom
/// directly. Returns [`None`] when the body has no local inputs or outputs, when it saves no residuals, or when a
/// computed residual lacks a concrete sharding.
fn factorize_transpose_shard_map_body(
    body: &FlatTracedShardMap,
) -> Result<Option<FactorizedTransposeShardMapBodies>, ShardMapTraceError> {
    let simplified_body = body.simplified()?;
    let local_input_count = simplified_body.local_input_types().len();
    let local_output_count = simplified_body.local_output_types().len();
    if local_input_count == 0 || local_output_count == 0 {
        return Ok(None);
    }

    // Linearize the primal body once over fresh local-primal inputs. The residuals saved by this run are tracers in
    // `residual_builder` and the residualized linear program references them by index.
    let residual_builder = Rc::new(RefCell::new(XlaProgramBuilder::new()));
    let residual_context = TracingContext::new(XlaDomain::token(), residual_builder.clone());
    let local_primals = simplified_body
        .local_input_types()
        .iter()
        .cloned()
        .map(|input_type| residual_context.input(input_type))
        .collect::<Vec<_>>();
    let (_, pushforward) = DifferentiationContext::linearize(
        &residual_context,
        |linearized_inputs| {
            let linearization_context = linearized_inputs
                .first()
                .ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?
                .context()
                .clone();
            linearization_context.stage_program(simplified_body.program(), linearized_inputs)
        },
        local_primals.clone(),
    )?;
    if pushforward.residuals().is_empty() {
        return Ok(None);
    }

    // Classify each residual in residual order. Residuals whose atom is one of the primal inputs are forwarded captures;
    // every other residual is a computed value routed through the residual body.
    let primal_atoms = local_primals.iter().map(|primal| primal.atom_id()).collect::<Result<Vec<_>, _>>()?;
    let mut residual_sources = Vec::with_capacity(pushforward.residuals().len());
    let mut residual_value_types = Vec::with_capacity(pushforward.residuals().len());
    let mut residual_output_tracers = Vec::new();
    for residual in pushforward.residuals() {
        residual_value_types.push(residual.r#type().into_owned());
        let residual_atom = residual.atom_id()?;
        match primal_atoms.iter().position(|primal_atom| *primal_atom == residual_atom) {
            Some(index) => residual_sources.push(FactorizedTransposeResidualSource::CapturedInput { index }),
            None => {
                residual_sources
                    .push(FactorizedTransposeResidualSource::ResidualOutput { index: residual_output_tracers.len() });
                residual_output_tracers.push(residual.clone());
            }
        }
    }

    // Computed residuals become outputs of the residual body, so each one needs a concrete sharding. Fall back to the
    // fused transpose body when any computed residual sharding is missing.
    let residual_output_shardings = residual_output_tracers
        .iter()
        .map(|residual| residual.r#type().sharding().cloned())
        .collect::<Option<Vec<_>>>();
    let Some(residual_output_shardings) = residual_output_shardings else {
        return Ok(None);
    };
    let residual_output_local_types =
        residual_output_tracers.iter().map(|residual| residual.r#type().into_owned()).collect::<Vec<_>>();

    // Keep the residualized linear program, then release every other holder of the residual builder so the residual body
    // can be built by unwrapping it. `residual_output_tracers` are intentionally kept; they are consumed by
    // `build_traced_xla_program` before it unwraps the builder.
    let linear_program = pushforward.program().clone();
    drop(residual_context);
    drop(pushforward);
    drop(local_primals);

    // Build the residual body. With no computed residuals the body is empty; otherwise it is the subprogram producing the
    // computed residuals, with primal inputs that no residual depends on pruned away.
    let (residual_body, residual_input_indices) = if residual_output_tracers.is_empty() {
        (empty_residual_shard_map_body(&simplified_body)?, Vec::new())
    } else {
        let residual_output_count = residual_output_tracers.len();
        let residual_full = build_traced_xla_program(
            residual_builder,
            residual_output_tracers,
            local_input_count,
            residual_output_count,
        )?;
        let residual_full_input_atoms = residual_full.input_ids().to_vec();
        let residual_full_output_atoms = residual_full.output_ids().to_vec();
        let (residual_program, residual_input_indices) =
            residual_full.into_filtered(&residual_full_input_atoms, &residual_full_output_atoms)?;
        let residual_in_shardings = residual_input_indices
            .iter()
            .copied()
            .map(|input_index| simplified_body.shard_map().in_shardings()[input_index].clone())
            .collect::<Vec<_>>();
        let residual_global_input_types = residual_input_indices
            .iter()
            .copied()
            .map(|input_index| simplified_body.global_input_types()[input_index].clone())
            .collect::<Vec<_>>();
        let residual_local_input_types = residual_input_indices
            .iter()
            .copied()
            .map(|input_index| simplified_body.local_input_types()[input_index].clone())
            .collect::<Vec<_>>();
        let residual_shard_map = ShardMap::from_shardings(
            simplified_body.shard_map().mesh().clone(),
            residual_in_shardings,
            residual_output_shardings.clone(),
            simplified_body.shard_map().manual_axes().to_vec(),
            simplified_body.shard_map().check_vma(),
        );
        let residual_global_output_types = crate::experimental::shard_map::derive_global_output_types(
            &residual_shard_map,
            &Vec::<ArrayType>::from_parameters(
                vec![ryft_core::parameters::Placeholder; residual_output_local_types.len()],
                residual_output_local_types.clone(),
            )
            .expect("residual output types should preserve placeholder structure"),
        )?;
        let residual_body = FlatTracedShardMap::from_parts(
            residual_shard_map,
            residual_global_input_types,
            residual_local_input_types,
            residual_global_output_types,
            residual_output_local_types.clone(),
            residual_program,
        );
        (residual_body, residual_input_indices)
    };
    let residual_body_global_output_types = residual_body.global_output_types().to_vec();

    // Stage the cotangent-application body. Residual references are first instantiated to the fresh residual inputs
    // in this apply trace, then the direct program is transposed and interpreted through the standard tracer
    // interpretation path.
    let apply_builder = Rc::new(RefCell::new(XlaProgramBuilder::new()));
    let apply_context = TracingContext::new(XlaDomain::token(), apply_builder.clone());
    let output_cotangents = simplified_body
        .local_output_types()
        .iter()
        .cloned()
        .map(|output_type| apply_context.input(output_type))
        .collect::<Vec<_>>();
    let residual_inputs = residual_value_types
        .iter()
        .cloned()
        .map(|residual_type| apply_context.input(residual_type))
        .collect::<Vec<_>>();
    let direct = linear_program.map_operations(|operation| {
        ResidualizedOperation::<TracingContext<'static, XlaDomain<'static>>>::instantiate_residuals(
            operation,
            residual_inputs.as_slice(),
        )
    })?;
    let transposed = apply_context.transpose(&direct)?;
    let input_cotangents = transposed.interpret_in_context(&apply_context, output_cotangents.clone())?;
    check_count!("output", input_cotangents, local_input_count, ProgramError);

    // Record where each input cotangent comes from. The transpose materializes structural-zero cotangents as `zero`
    // operations, so they flow through the apply body as ordinary apply outputs; only a closed constant atom surfaced
    // directly by the transpose is recorded as a captured constant.
    let mut output_sources = Vec::with_capacity(local_input_count);
    let mut apply_output_tracers = Vec::new();
    let mut apply_output_input_indices = Vec::new();
    for (input_index, input_cotangent) in input_cotangents.iter().enumerate() {
        let atom = input_cotangent.atom_id()?;
        match apply_builder.borrow().atoms()[atom.index()].as_constant().cloned() {
            Some(value) => output_sources.push(FactorizedTransposeOutputSource::Constant { value }),
            None => {
                output_sources.push(FactorizedTransposeOutputSource::ApplyOutput { index: apply_output_tracers.len() });
                apply_output_tracers.push(input_cotangent.clone());
                apply_output_input_indices.push(input_index);
            }
        }
    }

    // Release every other holder of the apply builder before unwrapping it to build the apply body program.
    drop(apply_context);
    drop(transposed);
    drop(direct);
    drop(output_cotangents);
    drop(residual_inputs);
    drop(input_cotangents);
    let apply_body_program = build_traced_xla_program(
        apply_builder,
        apply_output_tracers,
        local_output_count + residual_value_types.len(),
        apply_output_input_indices.len(),
    )?;

    // The apply body keeps every output-cotangent input and appends the residual inputs in residual order. Each residual
    // input's sharding/types come from its source (a forwarded primal input or a computed residual output).
    let apply_input_indices = (0..local_output_count).collect::<Vec<_>>();
    let mut apply_in_shardings = Vec::with_capacity(local_output_count + residual_sources.len());
    let mut apply_global_input_types = Vec::with_capacity(local_output_count + residual_sources.len());
    let mut apply_local_input_types = Vec::with_capacity(local_output_count + residual_sources.len());
    for output_index in 0..local_output_count {
        apply_in_shardings.push(simplified_body.shard_map().out_shardings()[output_index].clone());
        apply_global_input_types.push(simplified_body.global_output_types()[output_index].clone());
        apply_local_input_types.push(simplified_body.local_output_types()[output_index].clone());
    }
    for residual_source in residual_sources.iter().copied() {
        match residual_source {
            FactorizedTransposeResidualSource::CapturedInput { index } => {
                apply_in_shardings.push(simplified_body.shard_map().in_shardings()[index].clone());
                apply_global_input_types.push(simplified_body.global_input_types()[index].clone());
                apply_local_input_types.push(simplified_body.local_input_types()[index].clone());
            }
            FactorizedTransposeResidualSource::ResidualOutput { index } => {
                apply_in_shardings.push(residual_output_shardings[index].clone());
                apply_global_input_types.push(residual_body_global_output_types[index].clone());
                apply_local_input_types.push(residual_output_local_types[index].clone());
            }
        }
    }
    let apply_shard_map = ShardMap::from_shardings(
        simplified_body.shard_map().mesh().clone(),
        apply_in_shardings,
        apply_output_input_indices
            .iter()
            .copied()
            .map(|input_index| simplified_body.shard_map().in_shardings()[input_index].clone())
            .collect::<Vec<_>>(),
        simplified_body.shard_map().manual_axes().to_vec(),
        simplified_body.shard_map().check_vma(),
    );
    let apply_body = FlatTracedShardMap::from_parts(
        apply_shard_map,
        apply_global_input_types,
        apply_local_input_types,
        apply_output_input_indices
            .iter()
            .copied()
            .map(|input_index| simplified_body.global_input_types()[input_index].clone())
            .collect::<Vec<_>>(),
        apply_output_input_indices
            .iter()
            .copied()
            .map(|input_index| simplified_body.local_input_types()[input_index].clone())
            .collect::<Vec<_>>(),
        apply_body_program,
    );
    Ok(Some(FactorizedTransposeShardMapBodies::new(
        residual_body,
        residual_input_indices,
        residual_sources,
        apply_body,
        apply_input_indices,
        output_sources,
    )))
}

/// Builds one linear shard-map op over abstract tensor leaves.
///
/// Tensor-leaf linear shard-map ops do not read `captured_global_primals` during interpretation or MLIR lowering; the
/// bodies themselves already encode everything the downstream consumers need, so the capture vector is left
/// empty here.
#[cfg(test)]
fn make_linear_tensor_shard_map(
    body: &FlatTracedShardMap,
) -> Result<LinearShardMapOperation<XlaConstant>, ShardMapTraceError> {
    Ok(LinearShardMapOperation::new(
        body.clone(),
        Vec::new(),
        body.global_input_types().to_vec(),
        body.global_output_types().to_vec(),
        LinearShardMapEvalMode::Body(trace_pushforward_body(body)?),
        LinearShardMapEvalMode::Body(trace_pullback_body(body)?),
    ))
}

fn apply_flat_traced_shard_map(
    tracing_builder: Rc<RefCell<XlaProgramBuilder>>,
    body: FlatTracedShardMap,
    traced_inputs: Vec<ShardMapTracer>,
) -> Result<Vec<ShardMapTracer>, ShardMapTraceError> {
    TracingContext::new(XlaDomain::token(), tracing_builder)
        .stage_operation(
            XlaOperation::ShardMap(Box::new(ShardMapOperation::new(body.clone()))),
            traced_inputs.as_slice(),
        )
        .map_err(ShardMapTraceError::from)
}

fn build_traced_xla_program(
    tracing_builder: Rc<RefCell<XlaProgramBuilder>>,
    traced_outputs: Vec<ShardMapTracer>,
    input_count: usize,
    output_count: usize,
) -> Result<FlatXlaProgram, ProgramError> {
    if let Some(tracing_error) = tracing_builder.borrow().error().cloned() {
        return Err(tracing_error);
    }
    let output_atoms = traced_outputs.into_iter().map(|output| output.atom_id()).collect::<Result<Vec<_>, _>>()?;
    let tracing_builder = match Rc::try_unwrap(tracing_builder) {
        Ok(tracing_builder) => tracing_builder.into_inner(),
        Err(_) => {
            return Err(ProgramError::EscapedProgramBuilder);
        }
    };
    let program = tracing_builder.build(
        output_atoms,
        vec![ryft_core::parameters::Placeholder; input_count],
        vec![ryft_core::parameters::Placeholder; output_count],
    )?;
    program.simplified()
}

fn make_linear_shard_map<V, Capture>(
    body: &FlatTracedShardMap,
    captured_global_primals: Vec<Capture>,
) -> Result<LinearShardMapOperation<V, Capture>, ShardMapTraceError> {
    let transpose_mode = match factorize_transpose_shard_map_body(body)? {
        Some(factorized) => LinearShardMapEvalMode::FactorizedTranspose(factorized),
        None => LinearShardMapEvalMode::Body(trace_pullback_body(body)?),
    };
    Ok(LinearShardMapOperation::new(
        body.clone(),
        captured_global_primals,
        body.global_input_types().to_vec(),
        body.global_output_types().to_vec(),
        LinearShardMapEvalMode::Body(trace_pushforward_body(body)?),
        transpose_mode,
    ))
}

/// Traces the forward linear shard-map body used for tangent evaluation.
///
/// The returned body takes the local primals followed by the local tangents and produces the local output tangents. It
/// is obtained by linearizing the primal body once and applying the resulting pushforward to fresh tangent inputs.
fn trace_pushforward_body(body: &FlatTracedShardMap) -> Result<FlatTracedShardMap, ShardMapTraceError> {
    let local_input_count = body.local_input_types().len();
    let local_output_count = body.local_output_types().len();

    let pushforward_local_input_types = body
        .local_input_types()
        .iter()
        .cloned()
        .chain(body.local_input_types().iter().cloned())
        .collect::<Vec<_>>();
    let pushforward_global_input_types = body
        .global_input_types()
        .iter()
        .cloned()
        .chain(body.global_input_types().iter().cloned())
        .collect::<Vec<_>>();
    let pushforward_shard_map = ShardMap::from_shardings(
        body.shard_map().mesh().clone(),
        body.shard_map()
            .in_shardings()
            .iter()
            .cloned()
            .chain(body.shard_map().in_shardings().iter().cloned())
            .collect::<Vec<_>>(),
        body.shard_map().out_shardings().to_vec(),
        body.shard_map().manual_axes().to_vec(),
        body.shard_map().check_vma(),
    );

    let pushforward_compiled_builder = Rc::new(RefCell::new(XlaProgramBuilder::new()));
    let pushforward_compiled_context = TracingContext::new(XlaDomain::token(), pushforward_compiled_builder.clone());
    let pushforward_compiled_outputs = {
        let combined_inputs = pushforward_local_input_types
            .iter()
            .cloned()
            .map(|input_type| pushforward_compiled_context.input(input_type))
            .collect::<Vec<_>>();
        let local_primals = combined_inputs[..local_input_count].to_vec();
        let local_tangents = combined_inputs[local_input_count..].to_vec();
        let (_, pushforward) = DifferentiationContext::linearize(
            &pushforward_compiled_context,
            |linearized_inputs| {
                let linearization_context = linearized_inputs
                    .first()
                    .ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?
                    .context()
                    .clone();
                linearization_context.stage_program(body.program(), linearized_inputs)
            },
            local_primals,
        )?;
        pushforward.apply(&pushforward_compiled_context, local_tangents)?
    };
    drop(pushforward_compiled_context);
    let pushforward_compiled = build_traced_xla_program(
        pushforward_compiled_builder,
        pushforward_compiled_outputs,
        local_input_count * 2,
        local_output_count,
    )?;

    Ok(FlatTracedShardMap::from_parts(
        pushforward_shard_map,
        pushforward_global_input_types,
        pushforward_local_input_types,
        body.global_output_types().to_vec(),
        body.local_output_types().to_vec(),
        pushforward_compiled,
    ))
}

/// Traces the fused transpose shard-map body used as the fallback for cotangent evaluation.
///
/// The returned body takes the local primals followed by the local output cotangents and produces the local input
/// cotangents, fusing residual computation and cotangent application into a single body. It is used only when
/// [`factorize_transpose_shard_map_body`] cannot factorize the transpose into separate residual and apply stages.
fn trace_pullback_body(body: &FlatTracedShardMap) -> Result<FlatTracedShardMap, ShardMapTraceError> {
    let local_input_count = body.local_input_types().len();
    let local_output_count = body.local_output_types().len();

    let pullback_local_input_types = body
        .local_input_types()
        .iter()
        .cloned()
        .chain(body.local_output_types().iter().cloned())
        .collect::<Vec<_>>();
    let pullback_global_input_types = body
        .global_input_types()
        .iter()
        .cloned()
        .chain(body.global_output_types().iter().cloned())
        .collect::<Vec<_>>();
    let pullback_shard_map = ShardMap::from_shardings(
        body.shard_map().mesh().clone(),
        body.shard_map()
            .in_shardings()
            .iter()
            .cloned()
            .chain(body.shard_map().out_shardings().iter().cloned())
            .collect::<Vec<_>>(),
        body.shard_map().in_shardings().to_vec(),
        body.shard_map().manual_axes().to_vec(),
        body.shard_map().check_vma(),
    );

    let pullback_compiled_builder = Rc::new(RefCell::new(XlaProgramBuilder::new()));
    let pullback_compiled_context = TracingContext::new(XlaDomain::token(), pullback_compiled_builder.clone());
    let pullback_compiled_outputs = {
        let combined_inputs = pullback_local_input_types
            .iter()
            .cloned()
            .map(|input_type| pullback_compiled_context.input(input_type))
            .collect::<Vec<_>>();
        let local_primals = combined_inputs[..local_input_count].to_vec();
        let local_output_cotangents = combined_inputs[local_input_count..].to_vec();
        let (_, pushforward) = DifferentiationContext::linearize(
            &pullback_compiled_context,
            |linearized_inputs| {
                let linearization_context = linearized_inputs
                    .first()
                    .ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?
                    .context()
                    .clone();
                linearization_context.stage_program(body.program(), linearized_inputs)
            },
            local_primals,
        )?;
        let pushforward_program = pushforward.instantiate_program()?;
        let pullback_program = pullback_compiled_context.transpose(&pushforward_program)?;
        let interpretation_context = local_output_cotangents
            .iter()
            .find_map(|cotangent| cotangent.interpretation_context())
            .ok_or_else(missing_traced_shard_map_staging_context)?;
        pullback_program.interpret_in_context(&interpretation_context, local_output_cotangents)?
    };
    drop(pullback_compiled_context);
    let pullback_compiled = build_traced_xla_program(
        pullback_compiled_builder,
        pullback_compiled_outputs,
        local_input_count + local_output_count,
        local_input_count,
    )?;

    Ok(FlatTracedShardMap::from_parts(
        pullback_shard_map,
        pullback_global_input_types,
        pullback_local_input_types,
        body.global_input_types().to_vec(),
        body.local_input_types().to_vec(),
        pullback_compiled,
    ))
}

fn trace_flat_shard_map<
    F: FnOnce(ShardMapLocalTraceInput<Input>) -> ShardMapLocalTraceOutput<Output>,
    Input: Parameterized<ArrayType>,
    Output: Parameterized<ArrayType>,
>(
    function: F,
    global_input_types: Input,
    mesh: LogicalMesh,
    in_specs: Input::To<Sharding>,
    out_specs: Output::To<Sharding>,
    manual_axes: Vec<String>,
    check_vma: bool,
) -> Result<FlatTracedShardMap, ShardMapTraceError>
where
    Input::Family: ParameterizedFamily<Sharding>
        + ParameterizedFamily<ArrayType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<ShardMapTracer>,
    Output::Family: ParameterizedFamily<Sharding>
        + ParameterizedFamily<ArrayType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<ShardMapTracer>,
    Output::To<ShardMapTracer>: Parameterized<ShardMapTracer, To<ArrayType> = Output>,
{
    let shard_map = ShardMap::new(
        mesh,
        in_specs.into_parameters().collect::<Vec<_>>(),
        out_specs.into_parameters().collect::<Vec<_>>(),
        manual_axes,
        check_vma,
    )?;
    Ok(FlatTracedShardMap::from_traced(&shard_map.trace::<F, Input, Output>(function, global_input_types)?))
}

fn apply_traced_shard_map<C, Output>(
    context: C,
    traced: FlatTracedShardMap,
    traced_inputs: Vec<Tracer<C>>,
    output_structure: Output::ParameterStructure,
) -> Result<Output, ShardMapTraceError>
where
    C: StagingContext<Type = ArrayType, Operation = XlaOperation>,
    Output: Parameterized<Tracer<C>>,
{
    let staged_outputs = context.stage_operation(
        XlaOperation::ShardMap(Box::new(ShardMapOperation::new(traced.clone()))),
        traced_inputs.as_slice(),
    )?;
    Ok(Output::from_parameters(output_structure, staged_outputs)?)
}

fn global_input_types_from_traced_inputs<C, Input>(
    traced_inputs: &Input,
) -> Result<Input::To<ArrayType>, ShardMapTraceError>
where
    C: Context<Type = ArrayType>,
    Input: Parameterized<Tracer<C>>,
    Input::Family: ParameterizedFamily<ArrayType>,
{
    Ok(Input::To::<ArrayType>::from_parameters(
        traced_inputs.parameter_structure(),
        traced_inputs.parameters().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
    )?)
}

fn reparameterize_shardings<Source: Parameterized<Sharding>, Target: Parameterized<Sharding>>(
    specs: Source,
    target_structure: Target::ParameterStructure,
) -> Result<Target, ShardMapTraceError> {
    Ok(Target::from_parameters(target_structure, specs.into_parameters().collect::<Vec<_>>())?)
}

impl ShardMapInvocationLeaf for ArrayType {
    type Return<Input: Parameterized<Self>, Output: Parameterized<ArrayType>>
        = TracedShardMap<Input, Output>
    where
        Input::Family: ParameterizedFamily<ArrayType>
            + ParameterizedFamily<Sharding>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ArrayType>
            + ParameterizedFamily<ShardMapTracer>,
        Output::Family: ParameterizedFamily<Sharding>
            + ParameterizedFamily<ArrayType>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ShardMapTracer>
            + ParameterizedFamily<ArrayType>,
        Output::To<ShardMapTracer>: Parameterized<ShardMapTracer, To<ArrayType> = Output>,
        Output::To<ArrayType>: Parameterized<ArrayType>;

    fn invoke<F, Input, Output>(
        function: F,
        inputs: Input,
        mesh: LogicalMesh,
        in_specs: Input::To<Sharding>,
        out_specs: Output::To<Sharding>,
        manual_axes: Vec<String>,
        check_vma: bool,
    ) -> Result<Self::Return<Input, Output>, ShardMapTraceError>
    where
        Input: Parameterized<Self>,
        Input::Family: ParameterizedFamily<ArrayType>
            + ParameterizedFamily<Sharding>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ArrayType>
            + ParameterizedFamily<ShardMapTracer>,
        Output: Parameterized<ArrayType>,
        Output::Family: ParameterizedFamily<Sharding>
            + ParameterizedFamily<ArrayType>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ShardMapTracer>
            + ParameterizedFamily<ArrayType>,
        Output::To<ShardMapTracer>: Parameterized<ShardMapTracer, To<ArrayType> = Output>,
        Output::To<ArrayType>: Parameterized<ArrayType>,
        F: FnOnce(ShardMapLocalTraceInput<Input::To<ArrayType>>) -> ShardMapLocalTraceOutput<Output>,
    {
        let shard_map = ShardMap::new(
            mesh,
            in_specs.into_parameters().collect::<Vec<_>>(),
            out_specs.into_parameters().collect::<Vec<_>>(),
            manual_axes,
            check_vma,
        )?;
        shard_map.trace(
            |local_inputs: ShardMapLocalTraceInput<Input>| {
                let adapted_inputs = ShardMapLocalTraceInput::<Input::To<ArrayType>>::from_parameters(
                    local_inputs.parameter_structure(),
                    local_inputs.into_parameters().collect::<Vec<_>>(),
                )
                .expect("array-typed shard_map inputs should preserve their canonical tracer structure");
                function(adapted_inputs)
            },
            inputs,
        )
    }
}

impl<C> ShardMapInvocationLeaf for Tracer<C>
where
    C: StagingContext<Type = ArrayType, Operation = XlaOperation>,
{
    type Return<Input: Parameterized<Self>, Output: Parameterized<ArrayType>>
        = Output::To<Tracer<C>>
    where
        Input::Family: ParameterizedFamily<ArrayType>
            + ParameterizedFamily<Sharding>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ArrayType>
            + ParameterizedFamily<ShardMapTracer>,
        Output::Family: ParameterizedFamily<Sharding>
            + ParameterizedFamily<ArrayType>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ShardMapTracer>
            + ParameterizedFamily<Tracer<C>>,
        Output::To<ShardMapTracer>: Parameterized<ShardMapTracer, To<ArrayType> = Output>,
        Output::To<Tracer<C>>: Parameterized<Tracer<C>>;

    fn invoke<F, Input, Output>(
        function: F,
        inputs: Input,
        mesh: LogicalMesh,
        in_specs: Input::To<Sharding>,
        out_specs: Output::To<Sharding>,
        manual_axes: Vec<String>,
        check_vma: bool,
    ) -> Result<Self::Return<Input, Output>, ShardMapTraceError>
    where
        Input: Parameterized<Self>,
        Input::Family: ParameterizedFamily<ArrayType>
            + ParameterizedFamily<Sharding>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ArrayType>
            + ParameterizedFamily<ShardMapTracer>,
        Output: Parameterized<ArrayType>,
        Output::Family: ParameterizedFamily<Sharding>
            + ParameterizedFamily<ArrayType>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ShardMapTracer>
            + ParameterizedFamily<Tracer<C>>,
        Output::To<ShardMapTracer>: Parameterized<ShardMapTracer, To<ArrayType> = Output>,
        Output::To<Tracer<C>>: Parameterized<Tracer<C>>,
        F: FnOnce(ShardMapLocalTraceInput<Input::To<ArrayType>>) -> ShardMapLocalTraceOutput<Output>,
    {
        let output_structure = out_specs.parameter_structure();
        let global_input_types = global_input_types_from_traced_inputs::<C, _>(&inputs)?;
        let global_in_specs = reparameterize_shardings::<
            Input::To<Sharding>,
            <Input::To<ArrayType> as Parameterized<ArrayType>>::To<Sharding>,
        >(in_specs, global_input_types.parameter_structure())?;
        let traced_inputs = inputs.into_parameters().collect::<Vec<_>>();
        let context = match traced_inputs.first() {
            Some(input) => input.context().clone(),
            None if output_structure.parameter_count() == 0 => {
                return Ok(Output::To::<Tracer<C>>::from_parameters(output_structure, Vec::new())?);
            }
            None => return Err(ShardMapTraceError::MissingTracedInvocationDomain),
        };
        let traced = trace_flat_shard_map::<F, Input::To<ArrayType>, Output>(
            function,
            global_input_types,
            mesh,
            global_in_specs,
            out_specs,
            manual_axes,
            check_vma,
        )?;
        apply_traced_shard_map(context, traced, traced_inputs, output_structure)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use ryft_core::contexts::StagingContext;
    use ryft_core::domains::AbstractDomain;
    use ryft_core::operations::BooleanLike;
    use ryft_core::operations::arithmetic::MulOperation;
    use ryft_core::operations::trigonometric::SinOperation;
    use ryft_core::parameters::Placeholder;
    use ryft_core::programs::{AtomId, ProgramBuilder, Value};
    use ryft_core::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding};
    use ryft_core::tracing::{AbstractTracingContext, TracingContext};
    use ryft_core::tracing_v2::differentiation::JvpTracer;
    use ryft_core::tracing_v2::{DifferentiableOperation, TangentContext, ValueOrCapture};
    use ryft_core::types::{ArrayType, DataType, Typed};

    use crate::experimental::domains::XlaTracer;
    use crate::experimental::ops::{LinearXlaOperation, XlaConstant, XlaOperation, XlaProgramBuilder};
    use crate::experimental::shard_map::{FlatTracedShardMap, ShardMap, ShardMapTracer};

    use super::{
        FactorizedTransposeOutputSource, FactorizedTransposeResidualSource, LinearShardMapEvalMode,
        LinearShardMapOperation, ShardMapOperation, factorize_transpose_shard_map_body, make_linear_tensor_shard_map,
    };

    fn test_array_type() -> ArrayType {
        ArrayType::scalar(DataType::F32)
    }

    fn replicated_test_array_type() -> ArrayType {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        ArrayType::new(DataType::F32, ryft_core::types::Shape::scalar())
            .with_sharding(Sharding::replicated(mesh, 0))
            .unwrap()
    }

    fn test_transposition_context<'transpose, V: Value<ArrayType> + BooleanLike>(
        domain: &'transpose AbstractDomain<ArrayType, V, LinearXlaOperation<V, XlaConstant>>,
        builder: Rc<RefCell<ProgramBuilder<ArrayType, V, LinearXlaOperation<V, XlaConstant>>>>,
    ) -> AbstractTracingContext<'transpose, ArrayType, V, LinearXlaOperation<V, XlaConstant>> {
        AbstractTracingContext::new(domain, builder)
    }

    fn test_shard_map() -> ShardMap {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        ShardMap::from_shardings(
            mesh.clone(),
            vec![Sharding::replicated(mesh.clone(), 0)],
            vec![Sharding::replicated(mesh, 0)],
            vec!["x".to_string()],
            true,
        )
    }

    fn two_input_test_shard_map() -> ShardMap {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        ShardMap::from_shardings(
            mesh.clone(),
            vec![Sharding::replicated(mesh.clone(), 0), Sharding::replicated(mesh.clone(), 0)],
            vec![Sharding::replicated(mesh, 0)],
            vec!["x".to_string()],
            true,
        )
    }

    fn zero_input_test_shard_map() -> ShardMap {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        ShardMap::from_shardings(
            mesh.clone(),
            Vec::new(),
            vec![Sharding::replicated(mesh, 0)],
            vec!["x".to_string()],
            true,
        )
    }

    fn zero_output_test_shard_map() -> ShardMap {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        ShardMap::from_shardings(
            mesh.clone(),
            vec![Sharding::replicated(mesh, 0)],
            Vec::new(),
            vec!["x".to_string()],
            true,
        )
    }

    fn simple_traced_shard_map_body() -> FlatTracedShardMap {
        let array_type = test_array_type();
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(array_type.clone());
        let output = builder
            .add_instruction(SinOperation, vec![input])
            .expect("simple shard_map body should stage one sine op")
            .into_iter()
            .copied()
            .next()
            .expect("sine should produce one output");
        FlatTracedShardMap::from_parts(
            test_shard_map(),
            vec![array_type.clone()],
            vec![array_type.clone()],
            vec![array_type.clone()],
            vec![array_type],
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap(),
        )
    }

    fn product_primal_traced_shard_map_body() -> FlatTracedShardMap {
        let array_type = test_array_type();
        let mut builder = XlaProgramBuilder::new();
        let left = builder.add_input(array_type.clone());
        let right = builder.add_input(array_type.clone());
        let output = builder.add_instruction(MulOperation, vec![left, right]).unwrap()[0];
        FlatTracedShardMap::from_parts(
            two_input_test_shard_map(),
            vec![array_type.clone(), array_type.clone()],
            vec![array_type.clone(), array_type.clone()],
            vec![array_type.clone()],
            vec![array_type],
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![output],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap(),
        )
    }

    fn computed_residual_primal_traced_shard_map_body() -> FlatTracedShardMap {
        let array_type = replicated_test_array_type();
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(array_type.clone());
        let output = builder.add_instruction(SinOperation, vec![input]).unwrap()[0];
        FlatTracedShardMap::from_parts(
            test_shard_map(),
            vec![array_type.clone()],
            vec![array_type.clone()],
            vec![array_type.clone()],
            vec![array_type],
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap(),
        )
    }

    fn partial_dependency_primal_traced_shard_map_body() -> FlatTracedShardMap {
        let array_type = replicated_test_array_type();
        let mut builder = XlaProgramBuilder::new();
        let used = builder.add_input(array_type.clone());
        builder.add_input(array_type.clone());
        let output = builder.add_instruction(SinOperation, vec![used]).unwrap()[0];
        FlatTracedShardMap::from_parts(
            two_input_test_shard_map(),
            vec![array_type.clone(), array_type.clone()],
            vec![array_type.clone(), array_type.clone()],
            vec![array_type.clone()],
            vec![array_type],
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![output],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap(),
        )
    }

    fn zero_input_traced_shard_map_body() -> FlatTracedShardMap {
        let array_type = test_array_type();
        let mut builder = XlaProgramBuilder::new();
        let output = builder.add_instruction(ryft_core::ZeroOperation::new(array_type.clone()), vec![]).unwrap()[0];
        FlatTracedShardMap::from_parts(
            zero_input_test_shard_map(),
            Vec::new(),
            Vec::new(),
            vec![array_type.clone()],
            vec![array_type],
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], Vec::<Placeholder>::new(), vec![Placeholder])
                .unwrap(),
        )
    }

    fn zero_output_traced_shard_map_body() -> FlatTracedShardMap {
        let array_type = test_array_type();
        let mut builder = XlaProgramBuilder::new();
        builder.add_input(array_type.clone());
        FlatTracedShardMap::from_parts(
            zero_output_test_shard_map(),
            vec![array_type.clone()],
            vec![array_type],
            Vec::new(),
            Vec::new(),
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(Vec::new(), vec![Placeholder], Vec::<Placeholder>::new())
                .unwrap(),
        )
    }

    fn zero_output_linear_shard_map_operation<V>() -> LinearShardMapOperation<V, V> {
        let body = zero_output_traced_shard_map_body();
        LinearShardMapOperation::new(
            body.clone(),
            Vec::new(),
            vec![test_array_type()],
            Vec::new(),
            LinearShardMapEvalMode::Body(body.clone()),
            LinearShardMapEvalMode::Body(body),
        )
    }

    #[test]
    fn test_factorized_transpose_forwards_primal_input_residuals() {
        // The product body `f(a, b) = a * b` has a JVP whose residuals are the opposite operands, both of which are
        // forwarded primal inputs. The transpose therefore needs no residual body and recovers both input cotangents
        // through the apply body.
        let body = product_primal_traced_shard_map_body();
        let factorized = factorize_transpose_shard_map_body(&body)
            .expect("product primal body should factorize")
            .expect("product primal body should need forwarded residuals");

        assert!(factorized.residual_input_indices().is_empty());
        assert_eq!(
            factorized.residual_sources(),
            &[
                FactorizedTransposeResidualSource::CapturedInput { index: 1 },
                FactorizedTransposeResidualSource::CapturedInput { index: 0 },
            ],
        );
        assert_eq!(
            factorized.output_sources(),
            &[
                FactorizedTransposeOutputSource::ApplyOutput { index: 0 },
                FactorizedTransposeOutputSource::ApplyOutput { index: 1 },
            ],
        );

        // The residual body is empty because every residual is a forwarded primal input.
        assert!(factorized.residual_body().global_input_types().is_empty());
        assert!(factorized.residual_body().local_input_types().is_empty());
        assert!(factorized.residual_body().global_output_types().is_empty());
        assert!(factorized.residual_body().local_output_types().is_empty());
        assert!(factorized.residual_body().program().input_ids().is_empty());
        assert!(factorized.residual_body().program().output_ids().is_empty());

        // The apply body consumes the single output cotangent followed by the two forwarded residuals, and multiplies
        // the cotangent by each opposite operand to recover the two input cotangents.
        assert_eq!(factorized.apply_input_indices(), &[0]);
        assert_eq!(factorized.apply_body().global_input_types().len(), 3);
        assert_eq!(factorized.apply_body().global_output_types().len(), 2);
        assert_eq!(
            factorized.apply_body().program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[], %2:f32[] .
                let %3:f32[] = mul %1 %0
                    %4:f32[] = mul %2 %0
                in (%3, %4)"},
        );
    }

    #[test]
    fn test_factorized_transpose_computes_residual_outputs() {
        // The body `f(x) = sin(x)` has a JVP that scales the tangent by the computed residual `cos(x)`. Factorization
        // routes `cos(x)` through a dedicated residual body and the transpose multiplies the output cotangent by it.
        let body = computed_residual_primal_traced_shard_map_body();
        let factorized = factorize_transpose_shard_map_body(&body)
            .expect("sine primal body should factorize")
            .expect("sine primal body should need one computed residual");

        assert_eq!(factorized.residual_input_indices(), &[0]);
        assert_eq!(factorized.residual_sources(), &[FactorizedTransposeResidualSource::ResidualOutput { index: 0 }]);
        assert_eq!(factorized.output_sources(), &[FactorizedTransposeOutputSource::ApplyOutput { index: 0 }]);

        // The residual body recomputes `cos(x)` from the single primal input.
        assert_eq!(factorized.residual_body().global_input_types().len(), 1);
        assert_eq!(factorized.residual_body().global_output_types().len(), 1);
        assert_eq!(
            factorized.residual_body().program().to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=2]>, []}] .
                let %1:f32[][sharding={mesh<['x'=2]>, []}] = cos %0
                in (%1)"},
        );

        // The apply body consumes the output cotangent followed by the `cos(x)` residual and multiplies them.
        assert_eq!(factorized.apply_input_indices(), &[0]);
        assert_eq!(factorized.apply_body().global_input_types().len(), 2);
        assert_eq!(factorized.apply_body().global_output_types().len(), 1);
        assert_eq!(
            factorized.apply_body().program().to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=2]>, []}], %1:f32[][sharding={mesh<['x'=2]>, []}] .
                let %2:f32[][sharding={mesh<['x'=2]>, []}] = mul %1 %0
                in (%2)"},
        );
    }

    #[test]
    fn test_factorized_transpose_materializes_structural_zero_cotangents() {
        // The body `f(a, b) = sin(a)` ignores `b`, so `b`'s input cotangent is a structural zero. The transpose
        // materializes that zero inside the apply body while still computing `a`'s cotangent from the `cos(a)` residual.
        let body = partial_dependency_primal_traced_shard_map_body();
        let factorized = factorize_transpose_shard_map_body(&body)
            .expect("partial-dependency primal body should factorize")
            .expect("partial-dependency primal body should need one computed residual");

        assert_eq!(factorized.residual_input_indices(), &[0]);
        assert_eq!(factorized.residual_sources(), &[FactorizedTransposeResidualSource::ResidualOutput { index: 0 }]);
        assert_eq!(
            factorized.output_sources(),
            &[
                FactorizedTransposeOutputSource::ApplyOutput { index: 0 },
                FactorizedTransposeOutputSource::ApplyOutput { index: 1 },
            ],
        );

        // The residual body recomputes `cos(a)` from the used primal input only; the unused input is pruned away.
        assert_eq!(factorized.residual_input_indices(), &[0]);
        assert_eq!(factorized.residual_body().global_input_types().len(), 1);
        assert_eq!(factorized.residual_body().global_output_types().len(), 1);
        assert_eq!(
            factorized.residual_body().program().to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=2]>, []}] .
                let %1:f32[][sharding={mesh<['x'=2]>, []}] = cos %0
                in (%1)"},
        );

        // The apply body computes `a`'s cotangent as `cos(a) * cotangent` and emits a structural zero for `b`'s
        // cotangent.
        assert_eq!(factorized.apply_input_indices(), &[0]);
        assert_eq!(factorized.apply_body().global_input_types().len(), 2);
        assert_eq!(factorized.apply_body().global_output_types().len(), 2);
        assert_eq!(
            factorized.apply_body().program().to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=2]>, []}], %1:f32[][sharding={mesh<['x'=2]>, []}] .
                let %2:f32[][sharding={mesh<['x'=2]>, []}] = mul %1 %0
                    %3:f32[][sharding={mesh<['x'=2]>, []}] = zero [type=f32[][sharding={mesh<['x'=2]>, []}]]
                in (%2, %3)"},
        );
    }

    #[test]
    fn test_linear_tensor_shard_map_jvp_stages_linear_tangent() {
        let body = simple_traced_shard_map_body();
        let operation = make_linear_tensor_shard_map(&body).expect("linear tensor shard_map should be buildable");
        let domain = crate::experimental::domains::XlaDomain::token();
        let primal_builder = Rc::new(RefCell::new(XlaProgramBuilder::new()));
        let tracing_context = TracingContext::new(domain, primal_builder);
        let tangent_builder = Rc::new(RefCell::new(ProgramBuilder::<
            ArrayType,
            XlaTracer<'_, '_>,
            LinearXlaOperation<XlaTracer<'_, '_>, XlaConstant, ValueOrCapture<ArrayType, XlaTracer<'_, '_>>>,
        >::new()));
        let mut context = TangentContext::new(&tracing_context, tangent_builder.clone());
        let primal_input = tracing_context.input(test_array_type());
        let tangent_input = context.input(test_array_type());

        let outputs = XlaOperation::LinearShardMap(Box::new(operation))
            .jvp(&mut context, &[JvpTracer::from_value(primal_input, tangent_input)])
            .expect("linear tensor shard_map jvp should succeed");

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal().r#type().into_owned(), test_array_type());

        let output_atoms = outputs.into_iter().map(|output| output.tangent().atom_id().unwrap()).collect::<Vec<_>>();
        drop(context);
        let tangent_builder = Rc::try_unwrap(tangent_builder)
            .expect("traced shard_map jvp should not leak linear terms")
            .into_inner();
        let tangent_program = tangent_builder
            .build::<Vec<XlaTracer<'_, '_>>, Vec<XlaTracer<'_, '_>>>(output_atoms, vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert!(
            tangent_program.to_string().contains("linear_shard_map"),
            "expected linear tensor shard_map jvp to stage a linear shard-map op: {}",
            tangent_program
        );
    }

    #[test]
    fn test_linear_tensor_shard_map_transpose_supports_zero_outputs() {
        let operation = zero_output_linear_shard_map_operation::<ArrayType>();
        let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        let domain = AbstractDomain::new();
        let mut context = test_transposition_context(&domain, builder);

        let contributions = ryft_core::differentiation::TransposableOperation::transpose(
            &operation,
            &mut context,
            &[&test_array_type()],
            &[],
        )
        .expect("zero-output linear shard_map transpose should succeed");

        assert_eq!(contributions.len(), 1);
        assert!(contributions[0].is_zero());
    }

    #[test]
    fn test_linear_traced_shard_map_transpose_supports_zero_outputs() {
        let operation = zero_output_linear_shard_map_operation::<ShardMapTracer>();
        let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        let domain = AbstractDomain::new();
        let mut context = test_transposition_context(&domain, builder);

        let contributions = ryft_core::differentiation::TransposableOperation::transpose(
            &operation,
            &mut context,
            &[&test_array_type()],
            &[],
        )
        .expect("zero-output traced linear shard_map transpose should succeed");

        assert_eq!(contributions.len(), 1);
        assert!(contributions[0].is_zero());
    }

    #[test]
    fn test_traced_shard_map_interpret_with_explicit_builder_supports_zero_inputs() {
        let body = zero_input_traced_shard_map_body();
        let operation = ShardMapOperation::<ShardMapTracer>::new(body);
        let tracing_builder = Rc::new(RefCell::new(XlaProgramBuilder::new()));

        let outputs = operation
            .interpret_with_tracing_builder(tracing_builder.clone(), &[])
            .expect("explicit traced shard_map staging should support zero-input bodies");

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].r#type().into_owned(), test_array_type());

        let output_atoms = outputs
            .into_iter()
            .map(|output| output.atom_id().expect("staged output should remain live"))
            .collect::<Vec<AtomId>>();
        let tracing_builder = Rc::try_unwrap(tracing_builder)
            .expect("explicit shard_map replay should not leak the tracing builder")
            .into_inner();
        let staged_program = tracing_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(output_atoms, Vec::<Placeholder>::new(), vec![Placeholder])
            .unwrap();

        assert_eq!(staged_program.instructions().len(), 1);
        assert!(matches!(staged_program.instructions()[0].operation(), XlaOperation::ShardMap(_)));
        assert!(staged_program.instructions()[0].inputs().is_empty());
    }
}
