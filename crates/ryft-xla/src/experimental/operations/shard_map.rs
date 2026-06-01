use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::rc::Rc;

use ryft_core::differentiation::{Cotangent, LinearOperation};
use ryft_core::macros::check_count;
use ryft_core::operations::constants::SupportsZero;
use ryft_core::operations::{InterpretableOperation, Operation};
use ryft_core::parameters::{Parameterized, ParameterizedFamily};
use ryft_core::sharding::{LogicalMesh, MeshAxisType, Sharding};
use ryft_core::tracing::contexts::{Context, TracingContext};
use ryft_core::tracing::domains::{DomainTracer, ProgramTracer, Tracer, TracingDomain};
use ryft_core::tracing::{Atom, AtomId, Instruction, ProgramTracingContext, Traceable, TracingError};
use ryft_core::tracing_v2::differentiation::JvpTracer;
use ryft_core::tracing_v2::{
    Differentiable, DifferentiableContext, DifferentiableDomain, DifferentiableOperation, JvpContext,
};
use ryft_core::types::{ArrayType, TypeError, Typed};

use crate::experimental::domains::XlaDomain;
use crate::experimental::ops::{
    FlatXlaProgram, LinearXlaOperation, LinearXlaOperationExtension, XlaConstant, XlaOperation, XlaOperationExtension,
    XlaProgramBuilder,
};
use crate::experimental::shard_map::{
    FlatTracedShardMap, ShardMap, ShardMapInvocationLeaf, ShardMapLocalTraceInput, ShardMapLocalTraceOutput,
    ShardMapTraceError, ShardMapTracer, TracedShardMap,
};

#[derive(Clone)]
struct LinearShardMapBodies {
    /// Forward linear shard-map body used for tangent evaluation.
    pushforward: FlatTracedShardMap,

    /// Transposed linear shard-map body used for cotangent evaluation.
    pullback: FlatTracedShardMap,
}

/// Two-stage transpose factorization for one linear shard-map body.
#[derive(Clone, Debug)]
pub(crate) struct FactorizedTransposeShardMapBodies {
    /// Primals-only residual computation staged as its own shard-map body.
    residual_body: FlatTracedShardMap,

    /// Cotangent application staged separately from the residual computation.
    apply_body: FlatTracedShardMap,
}

impl FactorizedTransposeShardMapBodies {
    /// Creates a new [`FactorizedTransposeShardMapBodies`].
    #[inline]
    fn new(residual_body: FlatTracedShardMap, apply_body: FlatTracedShardMap) -> Self {
        Self { residual_body, apply_body }
    }

    /// Returns the primals-only residual shard-map body.
    #[inline]
    pub(crate) fn residual_body(&self) -> &FlatTracedShardMap {
        &self.residual_body
    }

    /// Returns the cotangent application shard-map body.
    #[inline]
    pub(crate) fn apply_body(&self) -> &FlatTracedShardMap {
        &self.apply_body
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
/// `captured_global_primals` holds the staging-program atom ids of the primals captured at linearization time. The
/// vector is empty for tensor-leaf shard-map ops (where captures are never read) and populated with atom ids for
/// tracer-leaf ops, where [`LinearShardMapOperation`] reifies each atom back into a `Tracer`.
#[derive(Clone, Debug)]
pub(crate) struct LinearShardMapState {
    /// Staged primal atom ids captured when the shard-map body was linearized.
    captured_global_primals: Vec<AtomId>,

    /// Evaluation strategy used when replaying the forward linear body.
    eval_mode: LinearShardMapEvalMode,

    /// Evaluation strategy used when replaying the transpose body.
    transpose_mode: LinearShardMapEvalMode,
}

impl LinearShardMapState {
    /// Creates a new [`LinearShardMapState`].
    #[inline]
    fn new(
        captured_global_primals: Vec<AtomId>,
        eval_mode: LinearShardMapEvalMode,
        transpose_mode: LinearShardMapEvalMode,
    ) -> Self {
        Self { captured_global_primals, eval_mode, transpose_mode }
    }

    /// Returns the staged primal atom IDs captured when the shard-map body was linearized.
    #[inline]
    pub(crate) fn captured_global_primals(&self) -> &[AtomId] {
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

fn missing_traced_shard_map_staging_context() -> TracingError {
    TracingError::Type(TypeError {
        message: "traced shard_map with non-empty outputs requires at least one traced input leaf".into(),
    })
}

fn missing_linear_shard_map_staging_context() -> TracingError {
    TracingError::Type(TypeError {
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

impl ShardMapOperation<XlaConstant> {
    /// Replays this tensor-leaf shard-map op with already-traced global inputs.
    pub(crate) fn interpret_traced_with_context(
        &self,
        tracing_builder: Rc<RefCell<XlaProgramBuilder>>,
        inputs: &[ShardMapTracer],
    ) -> Result<Vec<ShardMapTracer>, TracingError> {
        apply_flat_traced_shard_map(tracing_builder, self.body.clone(), inputs.to_vec())
            .map_err(trace_error_from_shard_map)
    }
}

/// Canonical linear shard-map op used in tangent/cotangent programs and traced linear replay.
#[derive(Clone, Debug)]
pub struct LinearShardMapOperation<V> {
    /// Canonical erased primal shard-map body carried by this linear higher-order op.
    body: FlatTracedShardMap,

    /// Global input types expected by the carried body.
    input_types: Vec<ArrayType>,

    /// Global output types produced by the carried body.
    output_types: Vec<ArrayType>,

    /// Linear execution state for replaying this linear shard-map.
    linear_state: LinearShardMapState,

    /// Phantom marker tying the op to the traced leaf type it will replay with.
    marker: PhantomData<fn() -> V>,
}

impl<V> LinearShardMapOperation<V> {
    /// Creates one linear shard-map op with captured primals and explicit transpose state.
    #[inline]
    fn new(
        body: FlatTracedShardMap,
        captured_global_primals: Vec<AtomId>,
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

    /// Returns the transposed linear shard-map op.
    fn transpose_op(&self) -> Self {
        Self::new(
            self.body.clone(),
            self.linear_state.captured_global_primals.clone(),
            self.output_types.clone(),
            self.input_types.clone(),
            self.linear_state.transpose_mode.clone(),
            self.linear_state.eval_mode.clone(),
        )
    }

    /// Rebuilds this linear shard-map op as the tensor-leaf XLA operation variant.
    fn to_tensor_xla_op<'o>(&self) -> LinearShardMapOperation<XlaConstant> {
        LinearShardMapOperation::new(
            self.body.clone(),
            self.linear_state.captured_global_primals.clone(),
            self.input_types.clone(),
            self.output_types.clone(),
            self.linear_state.eval_mode.clone(),
            self.linear_state.transpose_mode.clone(),
        )
    }

    /// Returns the erased primal shard-map body carried by this operation.
    #[inline]
    #[cfg(feature = "benchmarking")]
    pub(crate) fn body(&self) -> &FlatTracedShardMap {
        &self.body
    }

    /// Returns the linear execution state for this shard-map operation.
    #[inline]
    pub(crate) fn linear_state(&self) -> &LinearShardMapState {
        &self.linear_state
    }
}

impl LinearShardMapOperation<XlaConstant> {
    /// Replays this tensor-leaf linear shard-map op with already-traced global inputs.
    pub(crate) fn interpret_traced_with_context(
        &self,
        tracing_builder: Rc<RefCell<XlaProgramBuilder>>,
        inputs: &[ShardMapTracer],
    ) -> Result<Vec<ShardMapTracer>, TracingError> {
        let traced_op = self.to_tracer_linear_op(inputs)?;
        traced_op
            .interpret_with_tracing_builder(tracing_builder, inputs)
            .map_err(ShardMapTraceError::TracingError)
            .map_err(trace_error_from_shard_map)
    }

    /// Rebuilds this tensor-leaf shard-map op for traced linear-program staging.
    fn to_tracer_linear_op<C>(&self, primals: &[Tracer<C>]) -> Result<LinearShardMapOperation<Tracer<C>>, TracingError>
    where
        C: Context<Type = ArrayType>,
    {
        let captured_atoms = primals.iter().map(|primal| primal.atom_id()).collect::<Result<Vec<_>, _>>()?;
        Ok(LinearShardMapOperation::new(
            self.body.clone(),
            captured_atoms,
            self.input_types.clone(),
            self.output_types.clone(),
            self.linear_state.eval_mode.clone(),
            self.linear_state.transpose_mode.clone(),
        ))
    }
}

/// Completes a shard-map JVP once the caller has produced primal outputs and the matching linear shard-map operation.
fn complete_shard_map_jvp<'jvp, E, V>(
    context: &mut JvpContext<'jvp, E>,
    inputs: &[JvpTracer<'jvp, E>],
    primal_outputs: Vec<V>,
    output_count: usize,
    linear_operation: LinearShardMapOperation<V>,
) -> Result<Vec<JvpTracer<'jvp, E>>, TracingError>
where
    E: Differentiable<Type = ArrayType, Value = V, Tangent = V, LinearOperation<V> = LinearXlaOperation<V>> + 'jvp,
    V: Traceable<ArrayType>,
{
    check_count!("output", primal_outputs, output_count, TracingError);
    let tangent_inputs = inputs
        .iter()
        .map(|input| context.materialize_tangent(input.tangent().clone()))
        .collect::<Result<Vec<_>, _>>()?;
    let tangent_outputs = context.stage_operation(
        LinearXlaOperation::Extension(LinearXlaOperationExtension::LinearShardMap(Box::new(linear_operation))),
        tangent_inputs.as_slice(),
    )?;
    check_count!("output", tangent_outputs, output_count, TracingError);
    Ok(primal_outputs
        .into_iter()
        .zip(tangent_outputs)
        .map(|(primal, tangent)| JvpTracer::from_value(primal, tangent))
        .collect())
}

impl<'domain, 'context, Capture> ShardMapOperation<Tracer<TracingContext<'domain, XlaDomain<'context>, Capture>>>
where
    XlaDomain<'context>: 'domain,
    'context: 'domain,
    Capture: Traceable<ArrayType>,
{
    /// Applies this traced-leaf shard-map JVP using the active outer tracing context for primals.
    pub(crate) fn jvp_with_context<'jvp>(
        &self,
        primal_context: &TracingContext<'domain, XlaDomain<'context>, Capture>,
        context: &mut JvpContext<'jvp, TracingContext<'domain, XlaDomain<'context>, Capture>>,
        inputs: &[JvpTracer<'jvp, TracingContext<'domain, XlaDomain<'context>, Capture>>],
    ) -> Result<Vec<JvpTracer<'jvp, TracingContext<'domain, XlaDomain<'context>, Capture>>>, TracingError>
    where
        TracingContext<'domain, XlaDomain<'context>, Capture>: 'jvp,
    {
        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal_outputs = primal_context.stage_operation(
            XlaOperation::Extension(XlaOperationExtension::ShardMap(Box::new(ShardMapOperation::new(
                self.body.clone(),
            )))),
            primal_inputs.as_slice(),
        )?;
        let linear_operation = make_linear_shard_map(&self.body, primal_inputs).map_err(trace_error_from_shard_map)?;
        complete_shard_map_jvp(context, inputs, primal_outputs, self.output_types.len(), linear_operation)
    }
}

impl ShardMapOperation<ShardMapTracer> {
    /// Replays this traced-leaf shard-map op into one explicit outer tracing builder.
    fn interpret_with_tracing_builder(
        &self,
        tracing_builder: Rc<RefCell<XlaProgramBuilder>>,
        inputs: &[ShardMapTracer],
    ) -> Result<Vec<ShardMapTracer>, TracingError> {
        let abstract_inputs = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.infer_output_types(abstract_inputs.as_slice())?;
        apply_flat_traced_shard_map(tracing_builder, self.body.clone(), inputs.to_vec())
            .map_err(trace_error_from_shard_map)
    }

    /// Applies this traced-leaf shard-map JVP using an explicit outer tracing builder and a
    /// [`JvpContext`] for the linear builder.
    pub(crate) fn jvp_with_builders<'jvp, D>(
        &self,
        tracing_builder: Rc<RefCell<XlaProgramBuilder>>,
        context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, TracingError>
    where
        D: DifferentiableDomain<
                Type = ArrayType,
                Value = ShardMapTracer,
                Tangent = ShardMapTracer,
                LinearOperation<ShardMapTracer> = LinearXlaOperation<ShardMapTracer>,
            > + 'jvp,
    {
        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal_outputs = self.interpret_with_tracing_builder(tracing_builder, primal_inputs.as_slice())?;
        let linear_operation = make_linear_shard_map(&self.body, primal_inputs).map_err(trace_error_from_shard_map)?;
        complete_shard_map_jvp(context, inputs, primal_outputs, self.output_types.len(), linear_operation)
    }
}

impl LinearShardMapOperation<ShardMapTracer> {
    /// Replays this traced-leaf shard-map op into one explicit outer tracing builder.
    fn interpret_with_tracing_builder(
        &self,
        tracing_builder: Rc<RefCell<XlaProgramBuilder>>,
        inputs: &[ShardMapTracer],
    ) -> Result<Vec<ShardMapTracer>, TracingError> {
        let abstract_inputs = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.infer_output_types(abstract_inputs.as_slice())?;
        TracingContext::new(XlaDomain::token(), tracing_builder).stage_operation(
            XlaOperation::Extension(XlaOperationExtension::LinearShardMap(Box::new(self.to_tensor_xla_op()))),
            inputs,
        )
    }
}

impl LinearShardMapOperation<XlaConstant> {
    /// Applies this tensor-leaf linear shard-map JVP using the active outer tracing context for primals.
    pub(crate) fn jvp_traced_with_context<'domain, 'context, 'jvp, Capture>(
        &self,
        primal_context: &TracingContext<'domain, XlaDomain<'context>, Capture>,
        context: &mut JvpContext<'jvp, TracingContext<'domain, XlaDomain<'context>, Capture>>,
        inputs: &[JvpTracer<'jvp, TracingContext<'domain, XlaDomain<'context>, Capture>>],
    ) -> Result<Vec<JvpTracer<'jvp, TracingContext<'domain, XlaDomain<'context>, Capture>>>, TracingError>
    where
        XlaDomain<'context>: 'domain,
        'context: 'domain,
        Capture: Traceable<ArrayType>,
        TracingContext<'domain, XlaDomain<'context>, Capture>: 'jvp,
    {
        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal_outputs = primal_context.stage_operation(
            XlaOperation::Extension(XlaOperationExtension::LinearShardMap(Box::new(self.clone()))),
            primal_inputs.as_slice(),
        )?;
        let traced_op = self.to_tracer_linear_op(primal_inputs.as_slice())?;
        complete_shard_map_jvp(context, inputs, primal_outputs, self.output_types.len(), traced_op)
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

impl<V> Display for LinearShardMapOperation<V>
where
    V: Traceable<ArrayType>,
{
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
                    && actual.reduced_manual_axes() == expected.reduced_manual_axes()
                    && varying_manual_axes_match(actual, expected)
            }
            (None, Some(expected)) => {
                expected.unreduced_axes().is_empty()
                    && expected.reduced_manual_axes().is_empty()
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
        ArrayType::new(
            captured_output_type.data_type(),
            captured_output_type.shape().clone(),
            captured_output_type.layout().cloned(),
            actual_input_types[0].sharding().cloned(),
        )
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

impl<V: Traceable<ArrayType>> Operation<ArrayType> for ShardMapOperation<V> {
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

impl<V: Traceable<ArrayType>> Operation<ArrayType> for LinearShardMapOperation<V> {
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

impl<V> LinearOperation<ArrayType, V, LinearXlaOperation<V>> for LinearShardMapOperation<V>
where
    V: Traceable<ArrayType>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut ProgramTracingContext<'transpose, ArrayType, V, LinearXlaOperation<V>>,
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, LinearXlaOperation<V>>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, LinearXlaOperation<V>>>, TracingError> {
        check_count!("output", output_cotangents, self.output_types.len(), TracingError);
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
        let contributions = context.stage_operation(
            LinearXlaOperation::Extension(LinearXlaOperationExtension::LinearShardMap(Box::new(self.transpose_op()))),
            materialized.as_slice(),
        )?;
        check_count!("output", contributions, self.input_types.len(), TracingError);
        Ok(contributions.into_iter().map(Cotangent::Staged).collect::<Vec<_>>())
    }
}

/// Returns a concrete atom for `cotangent`, staging a typed `Zero` op when the cotangent is
/// structurally zero. Higher-order linear rules use this when they must consume all output
/// cotangents jointly.
fn materialize_cotangent<'transpose, V, O>(
    context: &ProgramTracingContext<'transpose, ArrayType, V, O>,
    cotangent: &Cotangent<'transpose, ArrayType, V, O>,
    output_type: &ArrayType,
) -> ProgramTracer<'transpose, ArrayType, V, O>
where
    V: Traceable<ArrayType>,
    O: Operation<ArrayType> + SupportsZero<ArrayType, V>,
{
    match cotangent {
        Cotangent::Staged(cotangent) => return cotangent.clone(),
        Cotangent::Zero => {}
    }
    let builder = &context.builder();
    let mut builder_borrow = builder.borrow_mut();
    let output = builder_borrow.add_variable(output_type.clone());
    builder_borrow.add_instruction_unchecked(ryft_core::tracing::Instruction::new(
        O::zero_operation(output_type.clone()),
        vec![],
        vec![output],
    ));
    drop(builder_borrow);
    context.tracer(output, None)
}

impl InterpretableOperation<ArrayType, ShardMapTracer> for ShardMapOperation<ShardMapTracer> {
    fn interpret(&self, inputs: &[ShardMapTracer]) -> Result<Vec<ShardMapTracer>, TracingError> {
        let tracing_builder = match inputs.first() {
            Some(input) => input.builder().clone(),
            None if self.output_types.is_empty() => return Ok(Vec::new()),
            None => return Err(missing_traced_shard_map_staging_context()),
        };
        self.interpret_with_tracing_builder(tracing_builder, inputs)
    }
}

impl<'domain, 'o, D> InterpretableOperation<ArrayType, DomainTracer<'domain, D>>
    for LinearShardMapOperation<DomainTracer<'domain, D>>
where
    D: TracingDomain<Type = ArrayType, Constant = XlaConstant, Operation = XlaOperation>,
{
    fn interpret(&self, inputs: &[DomainTracer<'domain, D>]) -> Result<Vec<DomainTracer<'domain, D>>, TracingError> {
        let exemplar = match inputs.first() {
            Some(input) => input,
            None if self.output_types.is_empty() => return Ok(Vec::new()),
            None => return Err(missing_traced_shard_map_staging_context()),
        };
        exemplar.context().stage_operation(
            XlaOperation::Extension(XlaOperationExtension::LinearShardMap(Box::new(self.to_tensor_xla_op()))),
            inputs,
        )
    }
}

impl<D> DifferentiableOperation<D> for ShardMapOperation<ShardMapTracer>
where
    D: DifferentiableDomain<
            Type = ArrayType,
            Value = ShardMapTracer,
            Tangent = ShardMapTracer,
            LinearOperation<ShardMapTracer> = LinearXlaOperation<ShardMapTracer>,
        >,
{
    fn jvp<'jvp>(
        &self,
        context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, TracingError>
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

fn trace_error_from_shard_map(error: ShardMapTraceError) -> TracingError {
    TracingError::Type(TypeError { message: error.to_string() })
}

/// Returns the number of primal inputs consumed by one transpose shard-map body.
fn transpose_body_primal_input_count(body: &FlatTracedShardMap) -> usize {
    body.global_output_types().len()
}

/// Returns the number of cotangent inputs consumed by one transpose shard-map body.
fn transpose_body_cotangent_input_count(body: &FlatTracedShardMap) -> usize {
    body.global_input_types().len() - transpose_body_primal_input_count(body)
}

/// Computes dense owning-instruction indices for one flat shard-map program.
fn instruction_by_output(program: &FlatXlaProgram) -> Vec<Option<usize>> {
    let mut instruction_by_output = vec![None; program.atoms().len()];
    for (instruction_index, instruction) in program.instructions().iter().enumerate() {
        for output in instruction.outputs().iter().copied() {
            instruction_by_output[output.index()] = Some(instruction_index);
        }
    }
    instruction_by_output
}

/// Marks one atom and all of its dependencies as live.
fn mark_live_flat_program(
    program: &FlatXlaProgram,
    atom_id: AtomId,
    live_atoms: &mut [bool],
    live_instructions: &mut [bool],
    instruction_by_output: &[Option<usize>],
) {
    if live_atoms[atom_id.index()] {
        return;
    }

    live_atoms[atom_id.index()] = true;
    if let Some(instruction_index) = instruction_by_output[atom_id.index()] {
        if live_instructions[instruction_index] {
            return;
        }

        live_instructions[instruction_index] = true;
        let instruction = &program.instructions()[instruction_index];
        for input in instruction.inputs().iter().copied() {
            mark_live_flat_program(program, input, live_atoms, live_instructions, instruction_by_output);
        }
    }
}

/// Returns live atom/instruction masks for one flat shard-map program.
fn live_sets_for_flat_program(program: &FlatXlaProgram) -> (Vec<bool>, Vec<bool>) {
    let instruction_by_output = instruction_by_output(program);
    let mut live_atoms = vec![false; program.atoms().len()];
    let mut live_instructions = vec![false; program.instructions().len()];
    for output in program.output_ids().iter().copied() {
        mark_live_flat_program(
            program,
            output,
            live_atoms.as_mut_slice(),
            live_instructions.as_mut_slice(),
            instruction_by_output.as_slice(),
        );
    }
    (live_atoms, live_instructions)
}

/// Tracks whether each atom in one transpose body depends on a cotangent input.
fn cotangent_dependencies_for_transpose_body(body: &FlatTracedShardMap) -> Vec<bool> {
    let program = body.program();
    let primal_input_count = transpose_body_primal_input_count(body);
    let mut depends_on_cotangent = vec![false; program.atoms().len()];
    for (input_index, atom_id) in program.input_ids().iter().copied().enumerate() {
        depends_on_cotangent[atom_id.index()] = input_index >= primal_input_count;
    }

    for instruction in program.instructions() {
        let instruction_depends_on_cotangent =
            instruction.inputs().iter().copied().any(|input| depends_on_cotangent[input.index()]);
        for output in instruction.outputs().iter().copied() {
            depends_on_cotangent[output.index()] = instruction_depends_on_cotangent;
        }
    }
    depends_on_cotangent
}

/// Rebuilds one projected flat shard-map program over a subset of the original inputs and outputs.
fn project_flat_shard_map_program(
    program: &FlatXlaProgram,
    kept_input_atoms: &[AtomId],
    output_atoms: &[AtomId],
) -> Result<FlatXlaProgram, ShardMapTraceError> {
    fn remap_atom(
        atom_id: AtomId,
        program: &FlatXlaProgram,
        builder: &mut XlaProgramBuilder,
        atom_mapping: &mut std::collections::HashMap<AtomId, AtomId>,
        kept_input_atoms: &std::collections::HashMap<AtomId, AtomId>,
        instruction_by_output: &[Option<usize>],
    ) -> Result<AtomId, ShardMapTraceError> {
        if let Some(mapped_atom) = atom_mapping.get(&atom_id) {
            return Ok(*mapped_atom);
        }
        if let Some(mapped_input) = kept_input_atoms.get(&atom_id) {
            atom_mapping.insert(atom_id, *mapped_input);
            return Ok(*mapped_input);
        }

        let atom = program.atoms().get(atom_id.index()).ok_or(TracingError::UnboundAtomId { id: atom_id })?;
        let mapped_atom = match atom {
            Atom::Constant(value) => builder.add_constant(value.clone()),
            Atom::Variable(_) => {
                let instruction_index = instruction_by_output[atom_id.index()]
                    .ok_or(ShardMapTraceError::ProjectedProgramMissingSourceAtom { atom_id })?;
                let instruction = &program.instructions()[instruction_index];
                let remapped_inputs = instruction
                    .inputs()
                    .iter()
                    .copied()
                    .map(|input| {
                        remap_atom(input, program, builder, atom_mapping, kept_input_atoms, instruction_by_output)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let output_abstracts = instruction
                    .outputs()
                    .iter()
                    .map(|output| program.atoms()[output.index()].r#type().into_owned())
                    .collect::<Vec<_>>();
                let remapped_outputs =
                    output_abstracts.into_iter().map(|r#type| builder.add_variable(r#type)).collect::<Vec<_>>();
                builder.add_instruction_unchecked(Instruction::new(
                    instruction.operation().clone(),
                    remapped_inputs,
                    remapped_outputs.clone(),
                ));
                for (old_output, new_output) in
                    instruction.outputs().iter().copied().zip(remapped_outputs.iter().copied())
                {
                    atom_mapping.insert(old_output, new_output);
                }
                *atom_mapping
                    .get(&atom_id)
                    .expect("projected shard_map instruction outputs should populate the atom mapping")
            }
        };

        atom_mapping.insert(atom_id, mapped_atom);
        Ok(mapped_atom)
    }

    let instruction_by_output = instruction_by_output(program);
    let mut builder = XlaProgramBuilder::new();
    let mut input_mapping = std::collections::HashMap::new();
    for atom_id in kept_input_atoms.iter().copied() {
        let mapped_atom = builder.add_input(program.atoms()[atom_id.index()].r#type().into_owned());
        input_mapping.insert(atom_id, mapped_atom);
    }

    let mut atom_mapping = input_mapping.clone();
    let projected_outputs = output_atoms
        .iter()
        .copied()
        .map(|output| {
            remap_atom(
                output,
                program,
                &mut builder,
                &mut atom_mapping,
                &input_mapping,
                instruction_by_output.as_slice(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    builder
        .build(
            projected_outputs,
            vec![ryft_core::parameters::Placeholder; kept_input_atoms.len()],
            vec![ryft_core::parameters::Placeholder; output_atoms.len()],
        )
        .map_err(ShardMapTraceError::from)
}

/// Rebuilds one apply-stage program whose primal-only dependencies have been replaced by residual inputs.
fn build_factorized_apply_program(
    body: &FlatTracedShardMap,
    residual_atoms: &[AtomId],
    depends_on_cotangent: &[bool],
) -> Result<FlatXlaProgram, ShardMapTraceError> {
    fn remap_atom(
        atom_id: AtomId,
        program: &FlatXlaProgram,
        builder: &mut XlaProgramBuilder,
        atom_mapping: &mut std::collections::HashMap<AtomId, AtomId>,
        replacement_inputs: &std::collections::HashMap<AtomId, AtomId>,
        depends_on_cotangent: &[bool],
        instruction_by_output: &[Option<usize>],
    ) -> Result<AtomId, ShardMapTraceError> {
        if let Some(mapped_atom) = atom_mapping.get(&atom_id) {
            return Ok(*mapped_atom);
        }
        if let Some(mapped_input) = replacement_inputs.get(&atom_id) {
            atom_mapping.insert(atom_id, *mapped_input);
            return Ok(*mapped_input);
        }

        let atom = program.atoms().get(atom_id.index()).ok_or(TracingError::UnboundAtomId { id: atom_id })?;
        let mapped_atom = match atom {
            Atom::Constant(value) => builder.add_constant(value.clone()),
            Atom::Variable(_) => {
                if !depends_on_cotangent[atom_id.index()] {
                    return Err(ShardMapTraceError::FactorizedApplyMissingResidualForCotangentIndependentAtom {
                        atom_id,
                    });
                }
                let instruction_index = instruction_by_output[atom_id.index()]
                    .ok_or(ShardMapTraceError::FactorizedApplyMissingResidualForPrimalInput { atom_id })?;
                let instruction = &program.instructions()[instruction_index];
                let remapped_inputs = instruction
                    .inputs()
                    .iter()
                    .copied()
                    .map(|input| {
                        remap_atom(
                            input,
                            program,
                            builder,
                            atom_mapping,
                            replacement_inputs,
                            depends_on_cotangent,
                            instruction_by_output,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let output_abstracts = instruction
                    .outputs()
                    .iter()
                    .map(|output| program.atoms()[output.index()].r#type().into_owned())
                    .collect::<Vec<_>>();
                let remapped_outputs =
                    output_abstracts.into_iter().map(|r#type| builder.add_variable(r#type)).collect::<Vec<_>>();
                builder.add_instruction_unchecked(Instruction::new(
                    instruction.operation().clone(),
                    remapped_inputs,
                    remapped_outputs.clone(),
                ));
                for (old_output, new_output) in
                    instruction.outputs().iter().copied().zip(remapped_outputs.iter().copied())
                {
                    atom_mapping.insert(old_output, new_output);
                }
                *atom_mapping
                    .get(&atom_id)
                    .expect("factorized shard_map apply instruction outputs should populate the atom mapping")
            }
        };

        atom_mapping.insert(atom_id, mapped_atom);
        Ok(mapped_atom)
    }

    let program = body.program();
    let primal_input_count = transpose_body_primal_input_count(body);
    let cotangent_input_atoms = program.input_ids()[primal_input_count..].to_vec();
    let instruction_by_output = instruction_by_output(program);
    let mut builder = XlaProgramBuilder::new();
    let mut replacement_inputs = std::collections::HashMap::new();

    for atom_id in cotangent_input_atoms.iter().copied() {
        let mapped_atom = builder.add_input(program.atoms()[atom_id.index()].r#type().into_owned());
        replacement_inputs.insert(atom_id, mapped_atom);
    }
    for atom_id in residual_atoms.iter().copied() {
        let mapped_atom = builder.add_input(program.atoms()[atom_id.index()].r#type().into_owned());
        replacement_inputs.insert(atom_id, mapped_atom);
    }

    let mut atom_mapping = replacement_inputs.clone();
    let outputs = program
        .output_ids()
        .iter()
        .copied()
        .map(|output| {
            remap_atom(
                output,
                program,
                &mut builder,
                &mut atom_mapping,
                &replacement_inputs,
                depends_on_cotangent,
                instruction_by_output.as_slice(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    builder
        .build(
            outputs,
            vec![ryft_core::parameters::Placeholder; cotangent_input_atoms.len() + residual_atoms.len()],
            vec![ryft_core::parameters::Placeholder; program.output_ids().len()],
        )
        .map_err(ShardMapTraceError::from)
}

/// Splits one fused transpose shard-map body into a residual stage and a cotangent-application stage.
fn factorize_transpose_shard_map_body(
    body: &FlatTracedShardMap,
) -> Result<Option<FactorizedTransposeShardMapBodies>, ShardMapTraceError> {
    let simplified_body = body.simplified()?;
    let program = simplified_body.program();
    let primal_input_count = transpose_body_primal_input_count(&simplified_body);
    let cotangent_input_count = transpose_body_cotangent_input_count(&simplified_body);
    if primal_input_count == 0 || cotangent_input_count == 0 {
        return Ok(None);
    }

    let (live_atoms, live_instructions) = live_sets_for_flat_program(program);
    let depends_on_cotangent = cotangent_dependencies_for_transpose_body(&simplified_body);
    let mut needed_as_residual = vec![false; program.atoms().len()];
    for (instruction_index, instruction) in program.instructions().iter().enumerate() {
        if !live_instructions[instruction_index] {
            continue;
        }
        let instruction_depends_on_cotangent =
            instruction.outputs().iter().copied().any(|output| depends_on_cotangent[output.index()]);
        if !instruction_depends_on_cotangent {
            continue;
        }
        for input in instruction.inputs().iter().copied() {
            if live_atoms[input.index()] && !depends_on_cotangent[input.index()] {
                needed_as_residual[input.index()] = true;
            }
        }
    }

    let residual_atoms = (0..program.atoms().len())
        .map(AtomId::new)
        .filter(|atom_id| {
            live_atoms[atom_id.index()] && !depends_on_cotangent[atom_id.index()] && needed_as_residual[atom_id.index()]
        })
        .collect::<Vec<_>>();
    if residual_atoms.is_empty() {
        return Ok(None);
    }

    let residual_out_shardings = residual_atoms
        .iter()
        .map(|atom_id| program.atoms()[atom_id.index()].r#type().sharding().cloned())
        .collect::<Option<Vec<_>>>();
    let Some(residual_out_shardings) = residual_out_shardings else {
        return Ok(None);
    };

    let primal_input_atoms = program.input_ids()[..primal_input_count].to_vec();
    let residual_program =
        project_flat_shard_map_program(program, primal_input_atoms.as_slice(), residual_atoms.as_slice())?
            .simplified()?;
    let residual_local_output_types = residual_atoms
        .iter()
        .map(|atom_id| program.atoms()[atom_id.index()].r#type().into_owned())
        .collect::<Vec<_>>();
    let residual_shard_map = crate::experimental::shard_map::ShardMap::from_shardings(
        simplified_body.shard_map().mesh().clone(),
        simplified_body.shard_map().in_shardings()[..primal_input_count].to_vec(),
        residual_out_shardings.clone(),
        simplified_body.shard_map().manual_axes().to_vec(),
        simplified_body.shard_map().check_vma(),
    );
    let residual_body = FlatTracedShardMap::from_parts(
        residual_shard_map.clone(),
        simplified_body.global_input_types()[..primal_input_count].to_vec(),
        simplified_body.local_input_types()[..primal_input_count].to_vec(),
        crate::experimental::shard_map::derive_global_output_types(
            &residual_shard_map,
            &Vec::<ArrayType>::from_parameters(
                vec![ryft_core::parameters::Placeholder; residual_local_output_types.len()],
                residual_local_output_types.clone(),
            )
            .expect("residual output types should preserve placeholder structure"),
        )?,
        residual_local_output_types,
        residual_program,
    );

    let apply_program =
        build_factorized_apply_program(&simplified_body, residual_atoms.as_slice(), depends_on_cotangent.as_slice())?
            .simplified()?;
    let residual_global_output_types = residual_body.global_output_types().to_vec();
    let residual_local_output_types = residual_body.local_output_types().to_vec();
    let apply_shard_map = crate::experimental::shard_map::ShardMap::from_shardings(
        simplified_body.shard_map().mesh().clone(),
        simplified_body.shard_map().in_shardings()[primal_input_count..]
            .iter()
            .cloned()
            .chain(residual_out_shardings)
            .collect::<Vec<_>>(),
        simplified_body.shard_map().out_shardings().to_vec(),
        simplified_body.shard_map().manual_axes().to_vec(),
        simplified_body.shard_map().check_vma(),
    );
    let apply_body = FlatTracedShardMap::from_parts(
        apply_shard_map,
        simplified_body.global_input_types()[primal_input_count..]
            .iter()
            .cloned()
            .chain(residual_global_output_types)
            .collect::<Vec<_>>(),
        simplified_body.local_input_types()[primal_input_count..]
            .iter()
            .cloned()
            .chain(residual_local_output_types)
            .collect::<Vec<_>>(),
        simplified_body.global_output_types().to_vec(),
        simplified_body.local_output_types().to_vec(),
        apply_program,
    );
    Ok(Some(FactorizedTransposeShardMapBodies::new(residual_body, apply_body)))
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
    let linear_bodies = trace_linear_shard_map_bodies(body)?;
    Ok(LinearShardMapOperation::new(
        body.clone(),
        Vec::new(),
        body.global_input_types().to_vec(),
        body.global_output_types().to_vec(),
        LinearShardMapEvalMode::Body(linear_bodies.pushforward),
        LinearShardMapEvalMode::Body(linear_bodies.pullback),
    ))
}

fn apply_flat_traced_shard_map(
    tracing_builder: Rc<RefCell<XlaProgramBuilder>>,
    body: FlatTracedShardMap,
    traced_inputs: Vec<ShardMapTracer>,
) -> Result<Vec<ShardMapTracer>, ShardMapTraceError> {
    TracingContext::new(XlaDomain::token(), tracing_builder)
        .stage_operation(
            XlaOperation::Extension(XlaOperationExtension::ShardMap(Box::new(ShardMapOperation::new(body.clone())))),
            traced_inputs.as_slice(),
        )
        .map_err(ShardMapTraceError::from)
}

fn build_traced_xla_program(
    tracing_builder: Rc<RefCell<XlaProgramBuilder>>,
    traced_outputs: Vec<ShardMapTracer>,
    input_count: usize,
    output_count: usize,
) -> Result<FlatXlaProgram, TracingError> {
    if let Some(tracing_error) = tracing_builder.borrow().error().cloned() {
        return Err(tracing_error);
    }
    let output_atoms = traced_outputs.into_iter().map(|output| output.atom_id()).collect::<Result<Vec<_>, _>>()?;
    let tracing_builder = match Rc::try_unwrap(tracing_builder) {
        Ok(tracing_builder) => tracing_builder.into_inner(),
        Err(_) => {
            return Err(TracingError::EscapedProgramBuilder);
        }
    };
    let program = tracing_builder.build(
        output_atoms,
        vec![ryft_core::parameters::Placeholder; input_count],
        vec![ryft_core::parameters::Placeholder; output_count],
    )?;
    program.simplified()
}

fn make_linear_shard_map<C>(
    body: &FlatTracedShardMap,
    captured_global_primals: Vec<Tracer<C>>,
) -> Result<LinearShardMapOperation<Tracer<C>>, ShardMapTraceError>
where
    C: Context<Type = ArrayType>,
{
    let linear_bodies = trace_linear_shard_map_bodies(body)?;
    let transpose_mode = match factorize_transpose_shard_map_body(&linear_bodies.pullback)? {
        Some(factorized) => LinearShardMapEvalMode::FactorizedTranspose(factorized),
        None => LinearShardMapEvalMode::Body(linear_bodies.pullback.clone()),
    };
    Ok(LinearShardMapOperation::new(
        body.clone(),
        captured_global_primals.into_iter().map(|primal| primal.atom_id()).collect::<Result<Vec<_>, _>>()?,
        body.global_input_types().to_vec(),
        body.global_output_types().to_vec(),
        LinearShardMapEvalMode::Body(linear_bodies.pushforward),
        transpose_mode,
    ))
}

fn trace_linear_shard_map_bodies(body: &FlatTracedShardMap) -> Result<LinearShardMapBodies, ShardMapTraceError> {
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
    let pushforward_shard_map = crate::experimental::shard_map::ShardMap::from_shardings(
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
    let pullback_shard_map = crate::experimental::shard_map::ShardMap::from_shardings(
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
        let (_, pushforward_program) = pushforward_compiled_context.linearize(
            |linearized_inputs| {
                let linearization_context = linearized_inputs
                    .first()
                    .ok_or(TracingError::InvalidInputCount { expected: 1, got: 0 })?
                    .context()
                    .clone();
                linearization_context.stage_program(body.program(), linearized_inputs)
            },
            local_primals,
        )?;
        pushforward_program.interpret(local_tangents)?
    };
    drop(pushforward_compiled_context);
    let pushforward_compiled = build_traced_xla_program(
        pushforward_compiled_builder,
        pushforward_compiled_outputs,
        local_input_count * 2,
        local_output_count,
    )?;

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
        let (_, pushforward_program) = pullback_compiled_context.linearize(
            |linearized_inputs| {
                let linearization_context = linearized_inputs
                    .first()
                    .ok_or(TracingError::InvalidInputCount { expected: 1, got: 0 })?
                    .context()
                    .clone();
                linearization_context.stage_program(body.program(), linearized_inputs)
            },
            local_primals,
        )?;
        let pullback_program = pullback_compiled_context.transpose(&pushforward_program)?;
        pullback_program.interpret(local_output_cotangents)?
    };
    drop(pullback_compiled_context);
    let pullback_compiled = build_traced_xla_program(
        pullback_compiled_builder,
        pullback_compiled_outputs,
        local_input_count + local_output_count,
        local_input_count,
    )?;

    Ok(LinearShardMapBodies {
        pushforward: FlatTracedShardMap::from_parts(
            pushforward_shard_map,
            pushforward_global_input_types,
            pushforward_local_input_types,
            body.global_output_types().to_vec(),
            body.local_output_types().to_vec(),
            pushforward_compiled,
        ),
        pullback: FlatTracedShardMap::from_parts(
            pullback_shard_map,
            pullback_global_input_types,
            pullback_local_input_types,
            body.global_input_types().to_vec(),
            body.local_input_types().to_vec(),
            pullback_compiled,
        ),
    })
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
    C: Context<Type = ArrayType, Operation = XlaOperation>,
    Output: Parameterized<Tracer<C>>,
{
    let staged_outputs = context.stage_operation(
        XlaOperation::Extension(XlaOperationExtension::ShardMap(Box::new(ShardMapOperation::new(traced.clone())))),
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
    C: Context<Type = ArrayType, Operation = XlaOperation>,
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

    use ryft_core::parameters::Placeholder;
    use ryft_core::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding};
    use ryft_core::tracing::contexts::TracingContext;
    use ryft_core::tracing::domains::ProgramTracingDomain;
    use ryft_core::tracing::{AtomId, Context, ProgramBuilder, ProgramTracingContext, Traceable};
    use ryft_core::tracing_v2::differentiation::JvpTracer;
    use ryft_core::tracing_v2::{DifferentiableOperation, JvpContext};
    use ryft_core::types::{ArrayType, DataType, Typed};

    use crate::experimental::domains::XlaTracer;
    use crate::experimental::ops::{
        LinearXlaOperation, XlaConstant, XlaOperation, XlaOperationExtension, XlaProgramBuilder,
    };
    use crate::experimental::shard_map::{FlatTracedShardMap, ShardMap, ShardMapTraceError, ShardMapTracer};

    use super::{
        LinearShardMapEvalMode, LinearShardMapOperation, ShardMapOperation, build_factorized_apply_program,
        make_linear_tensor_shard_map, project_flat_shard_map_program,
    };

    fn test_array_type() -> ArrayType {
        ArrayType::scalar(DataType::F32)
    }

    fn test_transposition_context<'transpose, V: Traceable<ArrayType>>(
        domain: &'transpose ProgramTracingDomain<ArrayType, V, LinearXlaOperation<V>>,
        builder: Rc<RefCell<ProgramBuilder<ArrayType, V, LinearXlaOperation<V>>>>,
    ) -> ProgramTracingContext<'transpose, ArrayType, V, LinearXlaOperation<V>> {
        ProgramTracingContext::new(domain, builder)
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
            .add_instruction(XlaOperation::Sin, vec![input])
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

    fn zero_input_traced_shard_map_body() -> FlatTracedShardMap {
        let array_type = test_array_type();
        let mut builder = XlaProgramBuilder::new();
        let output = builder
            .add_instruction(XlaOperation::Zero(ryft_core::ZeroOperation::new(array_type.clone())), vec![])
            .unwrap()[0];
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

    fn zero_output_linear_shard_map_operation<V>() -> LinearShardMapOperation<V> {
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
    fn test_project_flat_shard_map_program_rejects_unmapped_variable_atom() {
        let array_type = test_array_type();
        let mut builder = XlaProgramBuilder::new();
        let atom_id = builder.add_input(array_type);
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![atom_id], vec![Placeholder], vec![Placeholder])
            .unwrap();

        assert!(matches!(
            project_flat_shard_map_program(&program, &[], &[atom_id]),
            Err(ShardMapTraceError::ProjectedProgramMissingSourceAtom { atom_id: actual_atom_id })
                if actual_atom_id == atom_id
        ));
    }

    #[test]
    fn test_build_factorized_apply_program_rejects_missing_residual_for_cotangent_independent_atom() {
        let array_type = test_array_type();
        let body = FlatTracedShardMap::from_parts(
            test_shard_map(),
            vec![array_type.clone()],
            vec![array_type.clone()],
            vec![array_type.clone()],
            vec![array_type.clone()],
            {
                let mut builder = XlaProgramBuilder::new();
                let input = builder.add_input(array_type);
                let output = builder
                    .add_instruction(XlaOperation::Sin, vec![input])
                    .expect("test body should stage one sine instruction")
                    .into_iter()
                    .copied()
                    .next()
                    .expect("sine should produce one output");
                builder
                    .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                    .unwrap()
            },
        );

        assert!(matches!(
            build_factorized_apply_program(&body, &[], &[false, false]),
            Err(ShardMapTraceError::FactorizedApplyMissingResidualForCotangentIndependentAtom {
                atom_id,
            }) if atom_id == AtomId::new(1)
        ));
    }

    #[test]
    fn test_build_factorized_apply_program_rejects_missing_residual_for_primal_input() {
        let array_type = test_array_type();
        let atom_id = AtomId::new(0);
        let body = FlatTracedShardMap::from_parts(
            test_shard_map(),
            vec![array_type.clone()],
            vec![array_type.clone()],
            vec![array_type.clone()],
            vec![array_type.clone()],
            {
                let mut builder = XlaProgramBuilder::new();
                let input = builder.add_input(array_type);
                builder
                    .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![input], vec![Placeholder], vec![Placeholder])
                    .unwrap()
            },
        );

        assert!(matches!(
            build_factorized_apply_program(&body, &[], &[true]),
            Err(ShardMapTraceError::FactorizedApplyMissingResidualForPrimalInput { atom_id: actual_atom_id })
                if actual_atom_id == atom_id
        ));
    }

    #[test]
    fn test_linear_tensor_shard_map_jvp_stages_linear_tangent() {
        let body = simple_traced_shard_map_body();
        let operation = make_linear_tensor_shard_map(&body).expect("linear tensor shard_map should be buildable");
        let domain = crate::experimental::domains::XlaDomain::token();
        let primal_builder = Rc::new(RefCell::new(XlaProgramBuilder::new()));
        let tracing_context = TracingContext::new(domain, primal_builder);
        let tangent_builder =
            Rc::new(RefCell::new(
                ProgramBuilder::<ArrayType, XlaTracer<'_, '_>, LinearXlaOperation<XlaTracer<'_, '_>>>::new(),
            ));
        let mut context = JvpContext::new(&tracing_context, tangent_builder.clone());
        let primal_input = tracing_context.input(test_array_type());
        let tangent_input = context.input(test_array_type());

        let outputs = XlaOperationExtension::LinearShardMap(Box::new(operation))
            .jvp(&mut context, &[JvpTracer::from_value(primal_input, tangent_input)])
            .expect("linear tensor shard_map jvp should succeed");

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal().r#type().into_owned(), test_array_type());

        let output_atoms = outputs
            .into_iter()
            .map(|output| match output.tangent() {
                ryft_core::differentiation::Tangent::Value(tracer) => tracer.atom_id().unwrap(),
                ryft_core::differentiation::Tangent::Zero(_) => panic!("expected materialized tangent"),
            })
            .collect::<Vec<_>>();
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
        let domain = ProgramTracingDomain::new();
        let mut context = test_transposition_context(&domain, builder);

        let contributions = ryft_core::differentiation::LinearOperation::transpose(&operation, &mut context, &[])
            .expect("zero-output linear shard_map transpose should succeed");

        assert_eq!(contributions.len(), 1);
        assert!(contributions[0].is_zero());
    }

    #[test]
    fn test_linear_traced_shard_map_transpose_supports_zero_outputs() {
        let operation = zero_output_linear_shard_map_operation::<ShardMapTracer>();
        let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        let domain = ProgramTracingDomain::new();
        let mut context = test_transposition_context(&domain, builder);

        let contributions = ryft_core::differentiation::LinearOperation::transpose(&operation, &mut context, &[])
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
        assert!(matches!(
            staged_program.instructions()[0].operation(),
            XlaOperation::Extension(XlaOperationExtension::ShardMap(_))
        ));
        assert!(staged_program.instructions()[0].inputs().is_empty());
    }
}
