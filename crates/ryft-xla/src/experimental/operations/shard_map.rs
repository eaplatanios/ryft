use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

use ryft_core::operations::{InterpretableOperation, Operation};
use ryft_core::parameters::{Parameterized, ParameterizedFamily};
use ryft_core::sharding::{LogicalMesh, MeshAxisType, Sharding};
use ryft_core::tracing::engines::TracingContext;
use ryft_core::tracing::transposition::LinearOperation;
use ryft_core::tracing::{Atom, AtomId, Instruction, Program, ProgramBuilder, Traceable, TracingError};
use ryft_core::tracing_v2::differentiation::{Differentiable, JvpTracer};
use ryft_core::tracing_v2::linear::{linearize_traced_program, transpose_traced_linear_program};
use ryft_core::tracing_v2::{
    CustomPrimitive, DifferentiableEngine, DifferentiableOperation, JvpContext, LinearArrayOperation,
    LinearCustomPrimitive,
};
use ryft_core::types::{ArrayType, TypeError, Typed};

use crate::experimental::engines::XlaEngine;
use crate::experimental::lowering::{
    LoweringError, ShardMapMlirLowerer, StableHloCustomLowering, StableHloCustomLoweringExtension,
};
use crate::experimental::ops::XlaOperation;
use crate::experimental::shard_map::{
    FlatTracedShardMap, ShardMap, ShardMapInvocationLeaf, ShardMapLocalTraceInput, ShardMapLocalTraceOutput,
    ShardMapTensor, ShardMapTraceError, ShardMapTracer, TracedShardMap, fold_xla_program_constants,
};

/// Shared program type used by erased shard-map bodies.
type FlatShardMapProgram = Program<ArrayType, ShardMapTensor, XlaOperation, Vec<ShardMapTensor>, Vec<ShardMapTensor>>;

#[derive(Clone)]
pub(crate) struct ShardMapReplayContext {
    tracing_builder: Rc<RefCell<ProgramBuilder<ArrayType, ShardMapTensor, XlaOperation>>>,
}

impl ShardMapReplayContext {
    /// Creates a replay context that stages into `tracing_builder`.
    pub(crate) fn new(tracing_builder: Rc<RefCell<ProgramBuilder<ArrayType, ShardMapTensor, XlaOperation>>>) -> Self {
        Self { tracing_builder }
    }
}

#[derive(Clone)]
struct LinearShardMapBodies {
    /// Forward linear shard-map body used for tangent evaluation.
    pushforward: FlatTracedShardMap,

    /// Transposed linear shard-map body used for cotangent evaluation.
    pullback: FlatTracedShardMap,
}

/// Two-stage transpose factorization for one linear shard-map body.
#[derive(Clone)]
pub struct FactorizedTransposeShardMapBodies {
    /// Primals-only residual computation staged as its own shard-map body.
    pub residual_body: FlatTracedShardMap,

    /// Cotangent application staged separately from the residual computation.
    pub apply_body: FlatTracedShardMap,
}

/// Evaluation mode used by linear shard-map higher-order ops.
#[derive(Clone)]
pub enum LinearShardMapEvalMode {
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
#[derive(Clone)]
struct LinearShardMapState {
    /// Staged primal atom ids captured when the shard-map body was linearized.
    captured_global_primals: Vec<AtomId>,

    /// Evaluation strategy used when replaying the forward linear body.
    eval_mode: LinearShardMapEvalMode,

    /// Evaluation strategy used when replaying the transpose body.
    transpose_mode: LinearShardMapEvalMode,
}

fn missing_traced_shard_map_staging_context() -> TracingError {
    TracingError::Type(TypeError {
        message: "traced shard_map with non-empty outputs requires at least one traced input leaf".to_string(),
    })
}

fn missing_linear_shard_map_staging_context() -> TracingError {
    TracingError::Type(TypeError {
        message: "linear shard_map with non-empty outputs requires at least one traced input leaf".to_string(),
    })
}

/// Canonical higher-order shard-map op used for staged tracing, differentiation, and lowering.
#[derive(Clone)]
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
    pub fn new(body: FlatTracedShardMap) -> Self {
        Self {
            input_types: body.global_input_types.clone(),
            output_types: body.global_output_types.clone(),
            body,
            marker: PhantomData,
        }
    }

    /// Returns the canonical primal shard-map body carried by this higher-order op.
    #[inline]
    pub fn body(&self) -> &FlatTracedShardMap {
        &self.body
    }
}

impl ShardMapOperation<ShardMapTensor> {
    /// Replays this tensor-leaf shard-map op with already-traced global inputs.
    pub(crate) fn interpret_traced_with_context(
        &self,
        tracing_builder: Rc<RefCell<ProgramBuilder<ArrayType, ShardMapTensor, XlaOperation>>>,
        inputs: &[ShardMapTracer],
    ) -> Result<Vec<ShardMapTracer>, TracingError> {
        apply_flat_traced_shard_map(tracing_builder, self.body.clone(), inputs.to_vec())
            .map_err(trace_error_from_shard_map)
    }
}

/// Canonical linear shard-map op used in tangent/cotangent programs and traced linear replay.
#[derive(Clone)]
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
            linear_state: LinearShardMapState { captured_global_primals, eval_mode, transpose_mode },
            marker: PhantomData,
        }
    }

    /// Returns the canonical primal shard-map body carried by this linear higher-order op.
    #[inline]
    pub fn body(&self) -> &FlatTracedShardMap {
        &self.body
    }

    /// Returns the active linear evaluation mode for this linear shard-map op.
    #[inline]
    pub fn eval_mode(&self) -> &LinearShardMapEvalMode {
        &self.linear_state.eval_mode
    }

    /// Returns the outer-program atom ids of the primals captured when this linear shard-map was staged.
    #[inline]
    pub(crate) fn captured_global_primals(&self) -> &[AtomId] {
        self.linear_state.captured_global_primals.as_slice()
    }

    /// Returns the transpose evaluation mode for this linear shard-map op.
    #[inline]
    #[cfg(feature = "benchmarking")]
    pub fn transpose_mode(&self) -> &LinearShardMapEvalMode {
        &self.linear_state.transpose_mode
    }

    /// Returns the shared custom-primitive registration used by this linear shard-map variant.
    fn base_custom_primitive(&self) -> CustomPrimitive<ArrayType, V>
    where
        V: Traceable<ArrayType>,
        Self: Clone
            + InterpretableOperation<ArrayType, V>
            + LinearOperation<ArrayType, V, LinearArrayOperation<V>>
            + 'static,
    {
        CustomPrimitive::new(self.clone()).with_transpose_rule(self.clone())
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
}

impl LinearShardMapOperation<ShardMapTensor> {
    /// Replays this tensor-leaf linear shard-map op with already-traced global inputs.
    pub(crate) fn interpret_traced_with_context(
        &self,
        tracing_builder: Rc<RefCell<ProgramBuilder<ArrayType, ShardMapTensor, XlaOperation>>>,
        inputs: &[ShardMapTracer],
    ) -> Result<Vec<ShardMapTracer>, TracingError> {
        let traced_op = self.to_tracer_linear_op(inputs)?;
        traced_op
            .interpret_with_tracing_builder(tracing_builder, inputs)
            .map_err(ShardMapTraceError::TracingError)
            .map_err(trace_error_from_shard_map)
    }

    /// Returns the tensor-leaf linear custom primitive registration for this shard-map op.
    pub(crate) fn to_tensor_linear_custom_primitive(&self) -> LinearCustomPrimitive<ArrayType, ShardMapTensor> {
        self.base_custom_primitive()
            .with_extension(self.clone())
            .with_extension(ShardMapCustomReplayExtension::new({
                let op = self.clone();
                move |replay_context, inputs| {
                    let traced_op = op.to_tracer_linear_op(inputs.as_slice())?;
                    traced_op
                        .interpret_with_tracing_builder(replay_context.tracing_builder.clone(), inputs.as_slice())
                        .map_err(ShardMapTraceError::TracingError)
                }
            }))
            .with_extension(StableHloCustomLoweringExtension::new(Arc::new(self.clone())))
            .into_linear()
            .expect("linear tensor shard_map primitive should carry a transpose rule")
    }

    /// Rebuilds this tensor-leaf shard-map op for traced linear-program staging.
    fn to_tracer_linear_op(
        &self,
        primals: &[ShardMapTracer],
    ) -> Result<LinearShardMapOperation<ShardMapTracer>, TracingError> {
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

impl ShardMapOperation<ShardMapTracer> {
    /// Replays this traced-leaf shard-map op into one explicit outer tracing builder.
    fn interpret_with_tracing_builder(
        &self,
        tracing_builder: Rc<RefCell<ProgramBuilder<ArrayType, ShardMapTensor, XlaOperation>>>,
        inputs: &[ShardMapTracer],
    ) -> Result<Vec<ShardMapTracer>, TracingError> {
        let abstract_inputs = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let _ = self.infer_output_types(abstract_inputs.as_slice())?;
        apply_flat_traced_shard_map(tracing_builder, self.body.clone(), inputs.to_vec())
            .map_err(trace_error_from_shard_map)
    }

    /// Applies this traced-leaf shard-map JVP using an explicit outer tracing builder and a
    /// [`JvpContext`] for the linear builder.
    pub(crate) fn jvp_with_builders<E>(
        &self,
        tracing_builder: Rc<RefCell<ProgramBuilder<ArrayType, ShardMapTensor, XlaOperation>>>,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<ShardMapTracer, AtomId>],
    ) -> Result<Vec<JvpTracer<ShardMapTracer, AtomId>>, TracingError>
    where
        E: DifferentiableEngine<
                Type = ArrayType,
                Value = ShardMapTracer,
                LinearOperation = LinearArrayOperation<ShardMapTracer>,
            > + ?Sized,
    {
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let primal_outputs = self.interpret_with_tracing_builder(tracing_builder, primal_inputs.as_slice())?;
        let tangent_inputs = inputs.iter().map(|input| input.tangent).collect::<Vec<_>>();
        let tangent_outputs = context.apply_operation(
            tangent_inputs.as_slice(),
            LinearArrayOperation::Custom(Arc::new(
                make_linear_shard_map(&self.body, primal_inputs)
                    .map_err(trace_error_from_shard_map)?
                    .to_tracer_linear_custom_primitive(),
            )),
            self.output_types.len(),
        )?;
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| JvpTracer { primal, tangent })
            .collect::<Vec<_>>())
    }
}

impl LinearShardMapOperation<ShardMapTracer> {
    /// Returns the traced-leaf linear custom primitive registration for this shard-map op.
    pub(crate) fn to_tracer_linear_custom_primitive(&self) -> LinearCustomPrimitive<ArrayType, ShardMapTracer> {
        self.base_custom_primitive()
            .into_linear()
            .expect("linear traced shard_map primitive should carry a transpose rule")
    }

    /// Rebuilds this traced linear shard-map op as the tensor-leaf XLA carrier variant.
    fn to_tensor_xla_op(&self) -> LinearShardMapOperation<ShardMapTensor> {
        LinearShardMapOperation::new(
            self.body.clone(),
            self.linear_state.captured_global_primals.clone(),
            self.input_types.clone(),
            self.output_types.clone(),
            self.linear_state.eval_mode.clone(),
            self.linear_state.transpose_mode.clone(),
        )
    }

    /// Replays this traced-leaf shard-map op into one explicit outer tracing builder.
    fn interpret_with_tracing_builder(
        &self,
        tracing_builder: Rc<RefCell<ProgramBuilder<ArrayType, ShardMapTensor, XlaOperation>>>,
        inputs: &[ShardMapTracer],
    ) -> Result<Vec<ShardMapTracer>, TracingError> {
        let abstract_inputs = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let _ = self.infer_output_types(abstract_inputs.as_slice())?;
        let input_refs = inputs.iter().collect::<Vec<_>>();
        TracingContext::new(XlaEngine::token(), tracing_builder)
            .trace(XlaOperation::LinearShardMap(Box::new(self.to_tensor_xla_op())), input_refs.as_slice())
    }
}

impl LinearShardMapOperation<ShardMapTensor> {
    /// Applies this tensor-leaf linear shard-map JVP with traced primals.
    pub(crate) fn jvp_traced_with_builders<E>(
        &self,
        tracing_builder: Rc<RefCell<ProgramBuilder<ArrayType, ShardMapTensor, XlaOperation>>>,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<ShardMapTracer, AtomId>],
    ) -> Result<Vec<JvpTracer<ShardMapTracer, AtomId>>, TracingError>
    where
        E: DifferentiableEngine<
                Type = ArrayType,
                Value = ShardMapTracer,
                LinearOperation = LinearArrayOperation<ShardMapTracer>,
            > + ?Sized,
    {
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let primal_input_refs = primal_inputs.iter().collect::<Vec<_>>();
        let primal_outputs = TracingContext::new(XlaEngine::token(), tracing_builder)
            .trace(XlaOperation::LinearShardMap(Box::new(self.clone())), primal_input_refs.as_slice())?;
        let traced_op = self.to_tracer_linear_op(primal_inputs.as_slice())?;
        let tangent_inputs = inputs.iter().map(|input| input.tangent).collect::<Vec<_>>();
        let tangent_outputs = context.apply_operation(
            tangent_inputs.as_slice(),
            LinearArrayOperation::Custom(Arc::new(traced_op.to_tracer_linear_custom_primitive())),
            self.output_types.len(),
        )?;
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| JvpTracer { primal, tangent })
            .collect())
    }
}

impl<V> Debug for ShardMapOperation<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "ShardMap")
    }
}

impl<V> Display for ShardMapOperation<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "shard_map")
    }
}

impl<V> Debug for LinearShardMapOperation<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "LinearShardMap")
    }
}

impl<V> Display for LinearShardMapOperation<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "linear_shard_map")
    }
}

/// Returns `true` when two shard-map boundary types agree apart from carried sharding metadata.
fn shard_map_boundary_types_match(actual: &ArrayType, expected: &ArrayType) -> bool {
    fn varying_manual_axes_match(actual: &Sharding, expected: &Sharding) -> bool {
        actual
            .varying_manual_axes
            .iter()
            .filter(|axis_name| expected.mesh.axis_type(axis_name.as_str()) == Some(MeshAxisType::Manual))
            .eq(expected.varying_manual_axes.iter())
    }

    actual.data_type == expected.data_type
        && actual.shape == expected.shape
        && actual.layout == expected.layout
        && match (&actual.sharding, &expected.sharding) {
            (_, None) => true,
            (Some(actual), Some(expected)) => {
                actual.unreduced_axes == expected.unreduced_axes
                    && actual.reduced_manual_axes == expected.reduced_manual_axes
                    && varying_manual_axes_match(actual, expected)
            }
            (None, Some(expected)) => {
                expected.unreduced_axes.is_empty()
                    && expected.reduced_manual_axes.is_empty()
                    && expected.varying_manual_axes.is_empty()
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
        && actual_input_types[0].sharding != captured_input_types[0].sharding
        && actual_input_types[0].shape.rank() == captured_output_type.shape.rank()
    {
        let mut adapted_output_type = captured_output_type.clone();
        adapted_output_type.sharding = actual_input_types[0].sharding.clone();
        adapted_output_type
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
    if input_types.len() != captured_input_types.len() {
        return Err(TypeError {
            message: format!(
                "{} expected {} input types but got {}",
                operation_name,
                captured_input_types.len(),
                input_types.len()
            ),
        });
    }
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

impl Operation<ArrayType> for ShardMapOperation<ShardMapTensor> {
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

impl InterpretableOperation<ArrayType, ShardMapTensor> for ShardMapOperation<ShardMapTensor> {
    fn interpret(&self, inputs: &[ShardMapTensor]) -> Result<Vec<ShardMapTensor>, TracingError> {
        let abstract_inputs = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let _ = self.infer_output_types(abstract_inputs.as_slice())?;
        Ok(self.output_types.iter().cloned().map(ShardMapTensor::new).collect::<Vec<_>>())
    }
}

impl Operation<ArrayType> for LinearShardMapOperation<ShardMapTensor> {
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

impl InterpretableOperation<ArrayType, ShardMapTensor> for LinearShardMapOperation<ShardMapTensor> {
    fn interpret(&self, inputs: &[ShardMapTensor]) -> Result<Vec<ShardMapTensor>, TracingError> {
        let abstract_inputs = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let _ = self.infer_output_types(abstract_inputs.as_slice())?;
        Ok(self.output_types.iter().cloned().map(ShardMapTensor::new).collect::<Vec<_>>())
    }
}

impl LinearOperation<ArrayType, ShardMapTensor, LinearArrayOperation<ShardMapTensor>>
    for LinearShardMapOperation<ShardMapTensor>
{
    fn transpose(
        &self,
        context: &mut ryft_core::tracing::transposition::TranspositionContext<
            ArrayType,
            ShardMapTensor,
            LinearArrayOperation<ShardMapTensor>,
        >,
        output_cotangents: &[Option<AtomId>],
    ) -> Result<Vec<Option<AtomId>>, TracingError> {
        if output_cotangents.len() != self.output_types.len() {
            return Err(TracingError::InvalidInputCount {
                expected: self.output_types.len(),
                got: output_cotangents.len(),
            });
        }
        if output_cotangents.is_empty() {
            return Ok((0..self.input_types.len()).map(|_| None).collect::<Vec<_>>());
        }
        if output_cotangents.iter().all(Option::is_none) {
            return Ok((0..self.input_types.len()).map(|_| None).collect::<Vec<_>>());
        }
        let materialized = output_cotangents
            .iter()
            .zip(self.output_types.iter())
            .map(|(cotangent, output_type)| materialize_optional_cotangent(context, cotangent, output_type))
            .collect::<Vec<_>>();
        let contributions = context.stage(
            LinearArrayOperation::Custom(Arc::new(self.transpose_op().to_tensor_linear_custom_primitive())),
            materialized.as_slice(),
        )?;
        Ok(contributions.into_iter().map(Some).collect::<Vec<_>>())
    }
}

/// Returns a concrete atom for `cotangent`, staging a typed `Zero` op when the cotangent is
/// structurally zero. Higher-order linear rules use this when they must consume all output
/// cotangents jointly.
fn materialize_optional_cotangent<V>(
    context: &ryft_core::tracing::transposition::TranspositionContext<ArrayType, V, LinearArrayOperation<V>>,
    cotangent: &Option<AtomId>,
    output_type: &ArrayType,
) -> AtomId
where
    V: ryft_core::tracing::Traceable<ArrayType>,
{
    if let Some(atom) = cotangent {
        return *atom;
    }
    let builder = &context.builder;
    let mut builder_borrow = builder.borrow_mut();
    let output = builder_borrow.add_variable(output_type.clone());
    builder_borrow.instructions.push(ryft_core::tracing::Instruction {
        operation: LinearArrayOperation::Zero(ryft_core::tracing_v2::operations::constants::ZeroOperation::new(
            output_type.clone(),
        )),
        inputs: vec![],
        outputs: vec![output],
    });
    output
}

impl<E> DifferentiableOperation<E> for ShardMapOperation<ShardMapTensor>
where
    E: DifferentiableEngine<
            Type = ArrayType,
            Value = ShardMapTensor,
            LinearOperation = LinearArrayOperation<ShardMapTensor>,
        > + ?Sized,
    ShardMapTensor: Differentiable<ArrayType, Tangent = ShardMapTensor>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<ShardMapTensor, AtomId>],
    ) -> Result<Vec<JvpTracer<ShardMapTensor, AtomId>>, TracingError> {
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let primal_outputs = InterpretableOperation::interpret(self, primal_inputs.as_slice())?;
        let tangent_inputs = inputs.iter().map(|input| input.tangent).collect::<Vec<_>>();
        if tangent_inputs.is_empty() && !self.output_types.is_empty() {
            return Err(missing_linear_shard_map_staging_context());
        }
        let tangent_outputs = context.apply_operation(
            tangent_inputs.as_slice(),
            LinearArrayOperation::Custom(Arc::new(
                make_linear_tensor_shard_map(self.body())
                    .map_err(trace_error_from_shard_map)?
                    .to_tensor_linear_custom_primitive(),
            )),
            self.output_types.len(),
        )?;
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| JvpTracer { primal, tangent })
            .collect::<Vec<_>>())
    }
}

impl<E> DifferentiableOperation<E> for LinearShardMapOperation<ShardMapTensor>
where
    E: DifferentiableEngine<
            Type = ArrayType,
            Value = ShardMapTensor,
            LinearOperation = LinearArrayOperation<ShardMapTensor>,
        > + ?Sized,
    ShardMapTensor: Differentiable<ArrayType, Tangent = ShardMapTensor>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<ShardMapTensor, AtomId>],
    ) -> Result<Vec<JvpTracer<ShardMapTensor, AtomId>>, TracingError> {
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let primal_outputs = InterpretableOperation::interpret(self, primal_inputs.as_slice())?;
        let tangent_inputs = inputs.iter().map(|input| input.tangent).collect::<Vec<_>>();
        if tangent_inputs.is_empty() && !self.output_types.is_empty() {
            return Err(missing_linear_shard_map_staging_context());
        }
        let tangent_outputs = context.apply_operation(
            tangent_inputs.as_slice(),
            LinearArrayOperation::Custom(Arc::new(self.to_tensor_linear_custom_primitive())),
            self.output_types.len(),
        )?;
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| JvpTracer { primal, tangent })
            .collect::<Vec<_>>())
    }
}

impl Operation<ArrayType> for ShardMapOperation<ShardMapTracer> {
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

impl Operation<ArrayType> for LinearShardMapOperation<ShardMapTracer> {
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

impl InterpretableOperation<ArrayType, ShardMapTracer> for LinearShardMapOperation<ShardMapTracer> {
    fn interpret(&self, inputs: &[ShardMapTracer]) -> Result<Vec<ShardMapTracer>, TracingError> {
        let tracing_builder = match inputs.first() {
            Some(input) => input.builder().clone(),
            None if self.output_types.is_empty() => return Ok(Vec::new()),
            None => return Err(missing_traced_shard_map_staging_context()),
        };
        self.interpret_with_tracing_builder(tracing_builder, inputs)
    }
}

impl LinearOperation<ArrayType, ShardMapTracer, LinearArrayOperation<ShardMapTracer>>
    for LinearShardMapOperation<ShardMapTracer>
{
    fn transpose(
        &self,
        context: &mut ryft_core::tracing::transposition::TranspositionContext<
            ArrayType,
            ShardMapTracer,
            LinearArrayOperation<ShardMapTracer>,
        >,
        output_cotangents: &[Option<AtomId>],
    ) -> Result<Vec<Option<AtomId>>, TracingError> {
        if output_cotangents.len() != self.output_types.len() {
            return Err(TracingError::InvalidInputCount {
                expected: self.output_types.len(),
                got: output_cotangents.len(),
            });
        }
        if output_cotangents.is_empty() {
            return Ok((0..self.input_types.len()).map(|_| None).collect::<Vec<_>>());
        }
        if output_cotangents.iter().all(Option::is_none) {
            return Ok((0..self.input_types.len()).map(|_| None).collect::<Vec<_>>());
        }
        let materialized = output_cotangents
            .iter()
            .zip(self.output_types.iter())
            .map(|(cotangent, output_type)| materialize_optional_cotangent(context, cotangent, output_type))
            .collect::<Vec<_>>();
        let contributions = context.stage(
            LinearArrayOperation::Custom(Arc::new(self.transpose_op().to_tracer_linear_custom_primitive())),
            materialized.as_slice(),
        )?;
        Ok(contributions.into_iter().map(Some).collect::<Vec<_>>())
    }
}

impl<E> DifferentiableOperation<E> for ShardMapOperation<ShardMapTracer>
where
    E: DifferentiableEngine<
            Type = ArrayType,
            Value = ShardMapTracer,
            LinearOperation = LinearArrayOperation<ShardMapTracer>,
        > + ?Sized,
    ShardMapTracer: Differentiable<ArrayType, Tangent = ShardMapTracer>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<ShardMapTracer, AtomId>],
    ) -> Result<Vec<JvpTracer<ShardMapTracer, AtomId>>, TracingError> {
        let Some(first_input) = inputs.first() else {
            return if self.output_types.is_empty() {
                Ok(Vec::new())
            } else {
                Err(missing_linear_shard_map_staging_context())
            };
        };
        self.jvp_with_builders(first_input.primal.builder().clone(), context, inputs)
    }
}

impl StableHloCustomLowering<ShardMapTensor> for LinearShardMapOperation<ShardMapTensor> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        _op: &CustomPrimitive<ArrayType, ShardMapTensor>,
        input_values: &[ryft_mlir::ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        lowerer: &mut ShardMapMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ryft_mlir::ValueRef<'b, 'c, 't>>, LoweringError> {
        lowerer.lower_linear_shard_map_eval_mode(self.eval_mode(), &[], input_values)
    }
}

#[derive(Clone)]
pub(crate) struct ShardMapCustomReplayExtension {
    replay: Arc<dyn Fn(&ShardMapReplayContext, Vec<ShardMapTracer>) -> Result<Vec<ShardMapTracer>, ShardMapTraceError>>,
}

impl ShardMapCustomReplayExtension {
    pub(crate) fn new<F>(replay: F) -> Self
    where
        F: Fn(&ShardMapReplayContext, Vec<ShardMapTracer>) -> Result<Vec<ShardMapTracer>, ShardMapTraceError> + 'static,
    {
        Self { replay: Arc::new(replay) }
    }

    pub(crate) fn replay(
        &self,
        replay_context: &ShardMapReplayContext,
        inputs: Vec<ShardMapTracer>,
    ) -> Result<Vec<ShardMapTracer>, ShardMapTraceError> {
        (self.replay)(replay_context, inputs)
    }
}

fn trace_error_from_shard_map(error: ShardMapTraceError) -> TracingError {
    TracingError::Type(TypeError { message: error.to_string() })
}

/// Returns the number of primal inputs consumed by one transpose shard-map body.
fn transpose_body_primal_input_count(body: &FlatTracedShardMap) -> usize {
    body.global_output_types.len()
}

/// Returns the number of cotangent inputs consumed by one transpose shard-map body.
fn transpose_body_cotangent_input_count(body: &FlatTracedShardMap) -> usize {
    body.global_input_types.len() - transpose_body_primal_input_count(body)
}

/// Computes dense owning-instruction indices for one flat shard-map program.
fn instruction_by_output(program: &FlatShardMapProgram) -> Vec<Option<usize>> {
    let mut instruction_by_output = vec![None; program.atoms.len()];
    for (instruction_index, instruction) in program.instructions.iter().enumerate() {
        for output in instruction.outputs.iter().copied() {
            instruction_by_output[output.index] = Some(instruction_index);
        }
    }
    instruction_by_output
}

/// Marks one atom and all of its dependencies as live.
fn mark_live_flat_program(
    program: &FlatShardMapProgram,
    atom_id: AtomId,
    live_atoms: &mut [bool],
    live_instructions: &mut [bool],
    instruction_by_output: &[Option<usize>],
) {
    if live_atoms[atom_id.index] {
        return;
    }

    live_atoms[atom_id.index] = true;
    if let Some(instruction_index) = instruction_by_output[atom_id.index] {
        if live_instructions[instruction_index] {
            return;
        }

        live_instructions[instruction_index] = true;
        let instruction = &program.instructions[instruction_index];
        for input in instruction.inputs.iter().copied() {
            mark_live_flat_program(program, input, live_atoms, live_instructions, instruction_by_output);
        }
    }
}

/// Returns live atom/instruction masks for one flat shard-map program.
fn live_sets_for_flat_program(program: &FlatShardMapProgram) -> (Vec<bool>, Vec<bool>) {
    let instruction_by_output = instruction_by_output(program);
    let mut live_atoms = vec![false; program.atoms.len()];
    let mut live_instructions = vec![false; program.instructions.len()];
    for output in program.output_ids.iter().copied() {
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
    let program = &body.program;
    let primal_input_count = transpose_body_primal_input_count(body);
    let mut depends_on_cotangent = vec![false; program.atoms.len()];
    for (input_index, atom_id) in program.input_ids.iter().copied().enumerate() {
        depends_on_cotangent[atom_id.index] = input_index >= primal_input_count;
    }

    for instruction in &program.instructions {
        let instruction_depends_on_cotangent =
            instruction.inputs.iter().copied().any(|input| depends_on_cotangent[input.index]);
        for output in instruction.outputs.iter().copied() {
            depends_on_cotangent[output.index] = instruction_depends_on_cotangent;
        }
    }
    depends_on_cotangent
}

/// Rebuilds one projected flat shard-map program over a subset of the original inputs and outputs.
fn project_flat_shard_map_program(
    program: &FlatShardMapProgram,
    kept_input_atoms: &[AtomId],
    output_atoms: &[AtomId],
) -> Result<FlatShardMapProgram, ShardMapTraceError> {
    fn remap_atom(
        atom_id: AtomId,
        program: &FlatShardMapProgram,
        builder: &mut ProgramBuilder<ArrayType, ShardMapTensor, XlaOperation>,
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

        let atom = program.atoms.get(atom_id.index).ok_or(TracingError::UnboundAtomId { id: atom_id })?;
        let mapped_atom = match atom {
            Atom::Constant(value) => builder.add_constant(value.clone()),
            Atom::Variable(_) => {
                let instruction_index = instruction_by_output[atom_id.index]
                    .ok_or(ShardMapTraceError::ProjectedProgramMissingSourceAtom { atom_id })?;
                let instruction = &program.instructions[instruction_index];
                let remapped_inputs = instruction
                    .inputs
                    .iter()
                    .copied()
                    .map(|input| {
                        remap_atom(input, program, builder, atom_mapping, kept_input_atoms, instruction_by_output)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let output_abstracts = instruction
                    .outputs
                    .iter()
                    .map(|output| program.atoms[output.index].r#type().into_owned())
                    .collect::<Vec<_>>();
                let remapped_outputs =
                    output_abstracts.into_iter().map(|r#type| builder.add_variable(r#type)).collect::<Vec<_>>();
                builder.instructions.push(Instruction {
                    operation: instruction.operation.clone(),
                    inputs: remapped_inputs,
                    outputs: remapped_outputs.clone(),
                });
                for (old_output, new_output) in
                    instruction.outputs.iter().copied().zip(remapped_outputs.iter().copied())
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
    let mut builder = ProgramBuilder::<ArrayType, ShardMapTensor, XlaOperation>::new();
    let mut input_mapping = std::collections::HashMap::new();
    for atom_id in kept_input_atoms.iter().copied() {
        let mapped_atom = builder.add_input(program.atoms[atom_id.index].r#type().into_owned());
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
) -> Result<FlatShardMapProgram, ShardMapTraceError> {
    fn remap_atom(
        atom_id: AtomId,
        program: &FlatShardMapProgram,
        builder: &mut ProgramBuilder<ArrayType, ShardMapTensor, XlaOperation>,
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

        let atom = program.atoms.get(atom_id.index).ok_or(TracingError::UnboundAtomId { id: atom_id })?;
        let mapped_atom = match atom {
            Atom::Constant(value) => builder.add_constant(value.clone()),
            Atom::Variable(_) => {
                if !depends_on_cotangent[atom_id.index] {
                    return Err(ShardMapTraceError::FactorizedApplyMissingResidualForCotangentIndependentAtom {
                        atom_id,
                    });
                }
                let instruction_index = instruction_by_output[atom_id.index]
                    .ok_or(ShardMapTraceError::FactorizedApplyMissingResidualForPrimalInput { atom_id })?;
                let instruction = &program.instructions[instruction_index];
                let remapped_inputs = instruction
                    .inputs
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
                    .outputs
                    .iter()
                    .map(|output| program.atoms[output.index].r#type().into_owned())
                    .collect::<Vec<_>>();
                let remapped_outputs =
                    output_abstracts.into_iter().map(|r#type| builder.add_variable(r#type)).collect::<Vec<_>>();
                builder.instructions.push(Instruction {
                    operation: instruction.operation.clone(),
                    inputs: remapped_inputs,
                    outputs: remapped_outputs.clone(),
                });
                for (old_output, new_output) in
                    instruction.outputs.iter().copied().zip(remapped_outputs.iter().copied())
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

    let program = &body.program;
    let primal_input_count = transpose_body_primal_input_count(body);
    let cotangent_input_atoms = program.input_ids[primal_input_count..].to_vec();
    let instruction_by_output = instruction_by_output(program);
    let mut builder = ProgramBuilder::<ArrayType, ShardMapTensor, XlaOperation>::new();
    let mut replacement_inputs = std::collections::HashMap::new();

    for atom_id in cotangent_input_atoms.iter().copied() {
        let mapped_atom = builder.add_input(program.atoms[atom_id.index].r#type().into_owned());
        replacement_inputs.insert(atom_id, mapped_atom);
    }
    for atom_id in residual_atoms.iter().copied() {
        let mapped_atom = builder.add_input(program.atoms[atom_id.index].r#type().into_owned());
        replacement_inputs.insert(atom_id, mapped_atom);
    }

    let mut atom_mapping = replacement_inputs.clone();
    let outputs = program
        .output_ids
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
            vec![ryft_core::parameters::Placeholder; program.output_ids.len()],
        )
        .map_err(ShardMapTraceError::from)
}

/// Splits one fused transpose shard-map body into a residual stage and a cotangent-application stage.
fn factorize_transpose_shard_map_body(
    body: &FlatTracedShardMap,
) -> Result<Option<FactorizedTransposeShardMapBodies>, ShardMapTraceError> {
    let simplified_body = body.simplified()?;
    let program = &simplified_body.program;
    let primal_input_count = transpose_body_primal_input_count(&simplified_body);
    let cotangent_input_count = transpose_body_cotangent_input_count(&simplified_body);
    if primal_input_count == 0 || cotangent_input_count == 0 {
        return Ok(None);
    }

    let (live_atoms, live_instructions) = live_sets_for_flat_program(program);
    let depends_on_cotangent = cotangent_dependencies_for_transpose_body(&simplified_body);
    let mut needed_as_residual = vec![false; program.atoms.len()];
    for (instruction_index, instruction) in program.instructions.iter().enumerate() {
        if !live_instructions[instruction_index] {
            continue;
        }
        let instruction_depends_on_cotangent =
            instruction.outputs.iter().copied().any(|output| depends_on_cotangent[output.index]);
        if !instruction_depends_on_cotangent {
            continue;
        }
        for input in instruction.inputs.iter().copied() {
            if live_atoms[input.index] && !depends_on_cotangent[input.index] {
                needed_as_residual[input.index] = true;
            }
        }
    }

    let residual_atoms = (0..program.atoms.len())
        .map(|index| AtomId { index })
        .filter(|atom_id| {
            live_atoms[atom_id.index] && !depends_on_cotangent[atom_id.index] && needed_as_residual[atom_id.index]
        })
        .collect::<Vec<_>>();
    if residual_atoms.is_empty() {
        return Ok(None);
    }

    let residual_out_shardings = residual_atoms
        .iter()
        .map(|atom_id| program.atoms[atom_id.index].r#type().sharding.clone())
        .collect::<Option<Vec<_>>>();
    let Some(residual_out_shardings) = residual_out_shardings else {
        return Ok(None);
    };

    let primal_input_atoms = program.input_ids[..primal_input_count].to_vec();
    let residual_program = fold_xla_program_constants(&project_flat_shard_map_program(
        program,
        primal_input_atoms.as_slice(),
        residual_atoms.as_slice(),
    )?)?
    .simplified()?;
    let residual_local_output_types = residual_atoms
        .iter()
        .map(|atom_id| program.atoms[atom_id.index].r#type().into_owned())
        .collect::<Vec<_>>();
    let residual_shard_map = crate::experimental::shard_map::ShardMap::from_shardings(
        simplified_body.shard_map.mesh().clone(),
        simplified_body.shard_map.in_shardings()[..primal_input_count].to_vec(),
        residual_out_shardings.clone(),
        simplified_body.shard_map.manual_axes().to_vec(),
        simplified_body.shard_map.check_vma(),
    );
    let residual_body = FlatTracedShardMap::from_parts(
        residual_shard_map.clone(),
        simplified_body.global_input_types[..primal_input_count].to_vec(),
        simplified_body.local_input_types[..primal_input_count].to_vec(),
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

    let apply_program = fold_xla_program_constants(&build_factorized_apply_program(
        &simplified_body,
        residual_atoms.as_slice(),
        depends_on_cotangent.as_slice(),
    )?)?
    .simplified()?;
    let residual_global_output_types = residual_body.global_output_types.clone();
    let residual_local_output_types = residual_body.local_output_types.clone();
    let apply_shard_map = crate::experimental::shard_map::ShardMap::from_shardings(
        simplified_body.shard_map.mesh().clone(),
        simplified_body.shard_map.in_shardings()[primal_input_count..]
            .iter()
            .cloned()
            .chain(residual_out_shardings)
            .collect::<Vec<_>>(),
        simplified_body.shard_map.out_shardings().to_vec(),
        simplified_body.shard_map.manual_axes().to_vec(),
        simplified_body.shard_map.check_vma(),
    );
    let apply_body = FlatTracedShardMap::from_parts(
        apply_shard_map,
        simplified_body.global_input_types[primal_input_count..]
            .iter()
            .cloned()
            .chain(residual_global_output_types)
            .collect::<Vec<_>>(),
        simplified_body.local_input_types[primal_input_count..]
            .iter()
            .cloned()
            .chain(residual_local_output_types)
            .collect::<Vec<_>>(),
        simplified_body.global_output_types.clone(),
        simplified_body.local_output_types.clone(),
        apply_program,
    );
    Ok(Some(FactorizedTransposeShardMapBodies { residual_body, apply_body }))
}

/// Builds one linear shard-map op over abstract tensor leaves.
///
/// Tensor-leaf linear shard-map ops do not read `captured_global_primals` during interpretation or MLIR lowering â€”
/// the bodies themselves already encode everything the downstream consumers need â€” so the capture vector is left
/// empty here.
fn make_linear_tensor_shard_map(
    body: &FlatTracedShardMap,
) -> Result<LinearShardMapOperation<ShardMapTensor>, ShardMapTraceError> {
    let linear_bodies = trace_linear_shard_map_bodies(body)?;
    Ok(LinearShardMapOperation::new(
        body.clone(),
        Vec::new(),
        body.global_input_types.clone(),
        body.global_output_types.clone(),
        LinearShardMapEvalMode::Body(linear_bodies.pushforward),
        LinearShardMapEvalMode::Body(linear_bodies.pullback),
    ))
}

fn apply_flat_traced_shard_map(
    tracing_builder: Rc<RefCell<ProgramBuilder<ArrayType, ShardMapTensor, XlaOperation>>>,
    body: FlatTracedShardMap,
    traced_inputs: Vec<ShardMapTracer>,
) -> Result<Vec<ShardMapTracer>, ShardMapTraceError> {
    let traced_input_refs = traced_inputs.iter().collect::<Vec<_>>();
    TracingContext::new(XlaEngine::token(), tracing_builder)
        .trace(XlaOperation::ShardMap(Box::new(ShardMapOperation::new(body.clone()))), traced_input_refs.as_slice())
        .map_err(ShardMapTraceError::from)
}

fn build_traced_xla_program(
    tracing_builder: Rc<RefCell<ProgramBuilder<ArrayType, ShardMapTensor, XlaOperation>>>,
    traced_outputs: Vec<ShardMapTracer>,
    input_count: usize,
    output_count: usize,
) -> Result<FlatShardMapProgram, TracingError> {
    if let Some(tracing_error) = tracing_builder.borrow_mut().error.take() {
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
    fold_xla_program_constants(&program)?.simplified()
}

fn make_linear_shard_map(
    body: &FlatTracedShardMap,
    captured_global_primals: Vec<ShardMapTracer>,
) -> Result<LinearShardMapOperation<ShardMapTracer>, ShardMapTraceError> {
    let linear_bodies = trace_linear_shard_map_bodies(body)?;
    let transpose_mode = match factorize_transpose_shard_map_body(&linear_bodies.pullback)? {
        Some(factorized) => LinearShardMapEvalMode::FactorizedTranspose(factorized),
        None => LinearShardMapEvalMode::Body(linear_bodies.pullback.clone()),
    };
    Ok(LinearShardMapOperation::new(
        body.clone(),
        captured_global_primals.into_iter().map(|primal| primal.atom_id()).collect::<Result<Vec<_>, _>>()?,
        body.global_input_types.clone(),
        body.global_output_types.clone(),
        LinearShardMapEvalMode::Body(linear_bodies.pushforward),
        transpose_mode,
    ))
}

fn trace_linear_shard_map_bodies(body: &FlatTracedShardMap) -> Result<LinearShardMapBodies, ShardMapTraceError> {
    let local_input_count = body.local_input_types.len();
    let local_output_count = body.local_output_types.len();

    let pushforward_local_input_types = body
        .local_input_types
        .iter()
        .cloned()
        .chain(body.local_input_types.iter().cloned())
        .collect::<Vec<_>>();
    let pushforward_global_input_types = body
        .global_input_types
        .iter()
        .cloned()
        .chain(body.global_input_types.iter().cloned())
        .collect::<Vec<_>>();
    let pushforward_shard_map = crate::experimental::shard_map::ShardMap::from_shardings(
        body.shard_map.mesh().clone(),
        body.shard_map
            .in_shardings()
            .iter()
            .cloned()
            .chain(body.shard_map.in_shardings().iter().cloned())
            .collect::<Vec<_>>(),
        body.shard_map.out_shardings().to_vec(),
        body.shard_map.manual_axes().to_vec(),
        body.shard_map.check_vma(),
    );

    let pullback_local_input_types = body
        .local_input_types
        .iter()
        .cloned()
        .chain(body.local_output_types.iter().cloned())
        .collect::<Vec<_>>();
    let pullback_global_input_types = body
        .global_input_types
        .iter()
        .cloned()
        .chain(body.global_output_types.iter().cloned())
        .collect::<Vec<_>>();
    let pullback_shard_map = crate::experimental::shard_map::ShardMap::from_shardings(
        body.shard_map.mesh().clone(),
        body.shard_map
            .in_shardings()
            .iter()
            .cloned()
            .chain(body.shard_map.out_shardings().iter().cloned())
            .collect::<Vec<_>>(),
        body.shard_map.in_shardings().to_vec(),
        body.shard_map.manual_axes().to_vec(),
        body.shard_map.check_vma(),
    );

    let pushforward_compiled_builder =
        Rc::new(RefCell::new(ProgramBuilder::<ArrayType, ShardMapTensor, XlaOperation>::new()));
    let pushforward_compiled_context = TracingContext::new(XlaEngine::token(), pushforward_compiled_builder.clone());
    let pushforward_compiled_outputs = {
        let combined_inputs = pushforward_local_input_types
            .iter()
            .cloned()
            .map(|input_type| pushforward_compiled_context.input(input_type))
            .collect::<Vec<_>>();
        let local_primals = combined_inputs[..local_input_count].to_vec();
        let local_tangents = combined_inputs[local_input_count..].to_vec();
        let (_, pushforward_program) =
            linearize_traced_program(pushforward_compiled_context.clone(), &body.program, local_primals)?;
        pushforward_program.interpret(local_tangents)?
    };
    drop(pushforward_compiled_context);
    let pushforward_compiled = build_traced_xla_program(
        pushforward_compiled_builder,
        pushforward_compiled_outputs,
        local_input_count * 2,
        local_output_count,
    )?;

    let pullback_compiled_builder =
        Rc::new(RefCell::new(ProgramBuilder::<ArrayType, ShardMapTensor, XlaOperation>::new()));
    let pullback_compiled_context = TracingContext::new(XlaEngine::token(), pullback_compiled_builder.clone());
    let pullback_compiled_outputs = {
        let combined_inputs = pullback_local_input_types
            .iter()
            .cloned()
            .map(|input_type| pullback_compiled_context.input(input_type))
            .collect::<Vec<_>>();
        let local_primals = combined_inputs[..local_input_count].to_vec();
        let local_output_cotangents = combined_inputs[local_input_count..].to_vec();
        let (_, pushforward_program) =
            linearize_traced_program(pullback_compiled_context.clone(), &body.program, local_primals)?;
        let pullback_program =
            transpose_traced_linear_program(pullback_compiled_context.clone(), &pushforward_program)?;
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
            body.global_output_types.clone(),
            body.local_output_types.clone(),
            pushforward_compiled,
        ),
        pullback: FlatTracedShardMap::from_parts(
            pullback_shard_map,
            pullback_global_input_types,
            pullback_local_input_types,
            body.global_input_types.clone(),
            body.local_input_types.clone(),
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
    Input::Family:
        ParameterizedFamily<Sharding> + ParameterizedFamily<ShardMapTensor> + ParameterizedFamily<ShardMapTracer>,
    Output::Family:
        ParameterizedFamily<Sharding> + ParameterizedFamily<ShardMapTensor> + ParameterizedFamily<ShardMapTracer>,
    Output::To<ShardMapTracer>:
        Parameterized<ShardMapTracer, To<ArrayType> = Output, To<ShardMapTensor> = Output::To<ShardMapTensor>>,
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

fn apply_traced_shard_map<Output: Parameterized<ShardMapTracer>>(
    tracing_builder: Rc<RefCell<ProgramBuilder<ArrayType, ShardMapTensor, XlaOperation>>>,
    traced: FlatTracedShardMap,
    traced_inputs: Vec<ShardMapTracer>,
    output_structure: Output::ParameterStructure,
) -> Result<Output, ShardMapTraceError> {
    let traced_input_refs = traced_inputs.iter().collect::<Vec<_>>();
    let staged_outputs = TracingContext::new(XlaEngine::token(), tracing_builder).trace(
        XlaOperation::ShardMap(Box::new(ShardMapOperation::new(traced.clone()))),
        traced_input_refs.as_slice(),
    )?;
    Ok(Output::from_parameters(output_structure, staged_outputs)?)
}

fn global_input_types_from_traced_inputs<Input: Parameterized<ShardMapTracer>>(
    traced_inputs: &Input,
) -> Result<Input::To<ArrayType>, ShardMapTraceError>
where
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
            + ParameterizedFamily<ShardMapTensor>
            + ParameterizedFamily<ShardMapTracer>,
        Output::Family:
            ParameterizedFamily<Sharding> + ParameterizedFamily<ShardMapTensor> + ParameterizedFamily<ShardMapTracer>,
        Output::To<ShardMapTracer>:
            Parameterized<ShardMapTracer, To<ArrayType> = Output, To<ShardMapTensor> = Output::To<ShardMapTensor>>;

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
            + ParameterizedFamily<ShardMapTensor>
            + ParameterizedFamily<ShardMapTracer>,
        Output: Parameterized<ArrayType>,
        Output::Family:
            ParameterizedFamily<Sharding> + ParameterizedFamily<ShardMapTensor> + ParameterizedFamily<ShardMapTracer>,
        Output::To<ShardMapTracer>:
            Parameterized<ShardMapTracer, To<ArrayType> = Output, To<ShardMapTensor> = Output::To<ShardMapTensor>>,
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

impl ShardMapInvocationLeaf for ShardMapTracer {
    type Return<Input: Parameterized<Self>, Output: Parameterized<ArrayType>>
        = Output::To<ShardMapTracer>
    where
        Input::Family: ParameterizedFamily<ArrayType>
            + ParameterizedFamily<Sharding>
            + ParameterizedFamily<ShardMapTensor>
            + ParameterizedFamily<ShardMapTracer>,
        Output::Family:
            ParameterizedFamily<Sharding> + ParameterizedFamily<ShardMapTensor> + ParameterizedFamily<ShardMapTracer>,
        Output::To<ShardMapTracer>:
            Parameterized<ShardMapTracer, To<ArrayType> = Output, To<ShardMapTensor> = Output::To<ShardMapTensor>>;

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
            + ParameterizedFamily<ShardMapTensor>
            + ParameterizedFamily<ShardMapTracer>,
        Output: Parameterized<ArrayType>,
        Output::Family:
            ParameterizedFamily<Sharding> + ParameterizedFamily<ShardMapTensor> + ParameterizedFamily<ShardMapTracer>,
        Output::To<ShardMapTracer>:
            Parameterized<ShardMapTracer, To<ArrayType> = Output, To<ShardMapTensor> = Output::To<ShardMapTensor>>,
        F: FnOnce(ShardMapLocalTraceInput<Input::To<ArrayType>>) -> ShardMapLocalTraceOutput<Output>,
    {
        let output_structure = out_specs.parameter_structure();
        let global_input_types = global_input_types_from_traced_inputs(&inputs)?;
        let global_in_specs = reparameterize_shardings::<
            Input::To<Sharding>,
            <Input::To<ArrayType> as Parameterized<ArrayType>>::To<Sharding>,
        >(in_specs, global_input_types.parameter_structure())?;
        let traced_inputs = inputs.into_parameters().collect::<Vec<_>>();
        let tracing_builder = match traced_inputs.first() {
            Some(input) => input.builder().clone(),
            None if output_structure.parameter_count() == 0 => {
                return Ok(Output::To::<ShardMapTracer>::from_parameters(output_structure, Vec::new())?);
            }
            None => return Err(ShardMapTraceError::MissingTracedInvocationEngine),
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
        apply_traced_shard_map(tracing_builder, traced, traced_inputs, output_structure)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use ryft_core::parameters::Placeholder;
    use ryft_core::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding};
    use ryft_core::tracing::transposition::TranspositionContext;
    use ryft_core::tracing::{Atom, AtomId, ProgramBuilder, Traceable};
    use ryft_core::tracing_v2::differentiation::JvpTracer;
    use ryft_core::tracing_v2::{DifferentiableOperation, JvpContext, LinearArrayOperation};
    use ryft_core::types::{ArrayType, DataType, Typed};

    use crate::experimental::ops::XlaOperation;
    use crate::experimental::shard_map::{
        FlatTracedShardMap, ShardMap, ShardMapTensor, ShardMapTraceError, ShardMapTracer,
    };

    use super::{
        LinearShardMapEvalMode, LinearShardMapOperation, ShardMapOperation, build_factorized_apply_program,
        make_linear_tensor_shard_map, project_flat_shard_map_program,
    };

    fn test_array_type() -> ArrayType {
        ArrayType::scalar(DataType::F32)
    }

    fn test_transposition_context<V: Traceable<ArrayType>>(
        builder: Rc<RefCell<ProgramBuilder<ArrayType, V, LinearArrayOperation<V>>>>,
    ) -> TranspositionContext<ArrayType, V, LinearArrayOperation<V>> {
        TranspositionContext::new(builder)
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
        let mut builder = ProgramBuilder::<ArrayType, ShardMapTensor, XlaOperation>::new();
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
                .build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap(),
        )
    }

    fn zero_input_traced_shard_map_body() -> FlatTracedShardMap {
        let array_type = test_array_type();
        let mut builder = ProgramBuilder::<ArrayType, ShardMapTensor, XlaOperation>::new();
        let output = builder.add_constant(ShardMapTensor::new(array_type.clone()));
        FlatTracedShardMap::from_parts(
            zero_input_test_shard_map(),
            Vec::new(),
            Vec::new(),
            vec![array_type.clone()],
            vec![array_type],
            builder
                .build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(
                    vec![output],
                    Vec::<Placeholder>::new(),
                    vec![Placeholder],
                )
                .unwrap(),
        )
    }

    fn zero_output_traced_shard_map_body() -> FlatTracedShardMap {
        let array_type = test_array_type();
        let mut builder = ProgramBuilder::<ArrayType, ShardMapTensor, XlaOperation>::new();
        builder.add_input(array_type.clone());
        FlatTracedShardMap::from_parts(
            zero_output_test_shard_map(),
            vec![array_type.clone()],
            vec![array_type],
            Vec::new(),
            Vec::new(),
            builder
                .build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(
                    Vec::new(),
                    vec![Placeholder],
                    Vec::<Placeholder>::new(),
                )
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
        let mut builder = ProgramBuilder::<ArrayType, ShardMapTensor, XlaOperation>::new();
        let atom_id = builder.add_input(array_type);
        let program = builder
            .build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(vec![atom_id], vec![Placeholder], vec![Placeholder])
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
                let mut builder = ProgramBuilder::<ArrayType, ShardMapTensor, XlaOperation>::new();
                let input = builder.add_input(array_type);
                let output = builder
                    .add_instruction(XlaOperation::Sin, vec![input])
                    .expect("test body should stage one sine instruction")
                    .into_iter()
                    .copied()
                    .next()
                    .expect("sine should produce one output");
                builder
                    .build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(
                        vec![output],
                        vec![Placeholder],
                        vec![Placeholder],
                    )
                    .unwrap()
            },
        );

        assert!(matches!(
            build_factorized_apply_program(&body, &[], &[false, false]),
            Err(ShardMapTraceError::FactorizedApplyMissingResidualForCotangentIndependentAtom {
                atom_id: AtomId { index: 1 },
            })
        ));
    }

    #[test]
    fn test_build_factorized_apply_program_rejects_missing_residual_for_primal_input() {
        let array_type = test_array_type();
        let atom_id = AtomId { index: 0 };
        let body = FlatTracedShardMap::from_parts(
            test_shard_map(),
            vec![array_type.clone()],
            vec![array_type.clone()],
            vec![array_type.clone()],
            vec![array_type.clone()],
            {
                let mut builder = ProgramBuilder::<ArrayType, ShardMapTensor, XlaOperation>::new();
                builder.atoms = vec![Atom::Variable(array_type)];
                builder.input_ids = vec![atom_id];
                builder
                    .build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(
                        vec![atom_id],
                        vec![Placeholder],
                        vec![Placeholder],
                    )
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
    fn test_linear_tensor_shard_map_jvp_stages_linear_shard_map() {
        let body = simple_traced_shard_map_body();
        let operation = make_linear_tensor_shard_map(&body).expect("linear tensor shard_map should be buildable");
        let tangent_builder =
            Rc::new(RefCell::new(
                ProgramBuilder::<ArrayType, ShardMapTensor, LinearArrayOperation<ShardMapTensor>>::new(),
            ));
        let tangent_atom = tangent_builder.borrow_mut().add_input(test_array_type());
        let engine = crate::experimental::engines::XlaEngine::token();
        let mut context = JvpContext::new(engine, tangent_builder.clone());

        let outputs = operation
            .jvp(&mut context, &[JvpTracer { primal: ShardMapTensor::new(test_array_type()), tangent: tangent_atom }])
            .expect("linear tensor shard_map jvp should succeed");

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal.r#type().into_owned(), test_array_type());

        let output_atoms = outputs.into_iter().map(|output| output.tangent).collect::<Vec<_>>();
        drop(context);
        let tangent_builder = Rc::try_unwrap(tangent_builder)
            .expect("traced shard_map jvp should not leak linear terms")
            .into_inner();
        let tangent_program = tangent_builder
            .build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(output_atoms, vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert!(
            tangent_program.to_string().contains("linear_shard_map"),
            "expected linear tensor shard_map jvp to stage a linear_shard_map op: {}",
            tangent_program
        );
    }

    #[test]
    fn test_linear_tensor_shard_map_transpose_supports_zero_outputs() {
        let operation = zero_output_linear_shard_map_operation::<ShardMapTensor>();
        let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        let mut context = test_transposition_context(builder);

        let contributions =
            ryft_core::tracing::transposition::LinearOperation::transpose(&operation, &mut context, &[])
                .expect("zero-output linear shard_map transpose should succeed");

        assert_eq!(contributions.len(), 1);
        assert!(contributions[0].is_none());
    }

    #[test]
    fn test_linear_traced_shard_map_transpose_supports_zero_outputs() {
        let operation = zero_output_linear_shard_map_operation::<ShardMapTracer>();
        let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        let mut context = test_transposition_context(builder);

        let contributions =
            ryft_core::tracing::transposition::LinearOperation::transpose(&operation, &mut context, &[])
                .expect("zero-output traced linear shard_map transpose should succeed");

        assert_eq!(contributions.len(), 1);
        assert!(contributions[0].is_none());
    }

    #[test]
    fn test_traced_shard_map_interpret_with_explicit_builder_supports_zero_inputs() {
        let body = zero_input_traced_shard_map_body();
        let operation = ShardMapOperation::<ShardMapTracer>::new(body);
        let tracing_builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, ShardMapTensor, XlaOperation>::new()));

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
            .build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(
                output_atoms,
                Vec::<Placeholder>::new(),
                vec![Placeholder],
            )
            .unwrap();

        assert_eq!(staged_program.instructions.len(), 1);
        assert!(matches!(staged_program.instructions[0].operation, XlaOperation::ShardMap(_)));
        assert!(staged_program.instructions[0].inputs.is_empty());
    }
}
