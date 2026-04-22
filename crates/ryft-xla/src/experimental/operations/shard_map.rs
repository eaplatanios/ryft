//! Higher-order `shard_map` operations for traced XLA programs.

use std::{
    cell::RefCell,
    fmt::{Debug, Display},
    marker::PhantomData,
    ops::{Add, Mul, Neg},
    rc::Rc,
    sync::Arc,
};

use ryft_core::{
    parameters::{Parameterized, ParameterizedFamily},
    sharding::{LogicalMesh, MeshAxisType, Sharding},
    tracing::{
        Atom, AtomId, Instruction, InterpretableOperation, OneLike, Operation, Program, ProgramBuilder, Traceable,
        TracingError, ZeroLike,
    },
    tracing_v2::{
        Cos, CustomOperationError, CustomPrimitive, DifferentiableOperation, LinearCustomPrimitive, LinearOperation,
        LinearPrimitiveOperation, LinearTerm, Linearized, MatrixOps, PrimitiveOperation, Sin, Tracer, engine::Engine,
        forward::JvpTracer, operations::reshape::ReshapeOps,
    },
    types::{ArrayType, TypeError, Typed},
};

use crate::experimental::lowering::{
    LoweringError, ShardMapMlirLowerer, StableHloCustomLowering, StableHloCustomLoweringExtension,
};
use crate::experimental::shard_map::{
    FlatTracedShardMap, ShardMap, ShardMapInvocationLeaf, ShardMapLocalTraceInput, ShardMapLocalTraceOutput,
    ShardMapTensor, ShardMapTraceError, ShardMapTracer, TracedShardMap, fold_xla_program_constants,
};
use crate::experimental::{engine::XlaEngine, ops::XlaPrimitiveOperation};

/// Shared program type used by erased shard-map bodies.
type FlatShardMapProgram =
    Program<ArrayType, ShardMapTensor, XlaPrimitiveOperation, Vec<ShardMapTensor>, Vec<ShardMapTensor>>;

#[derive(Clone)]
pub(crate) struct ShardMapReplayContext {
    tracing_builder: Rc<RefCell<ProgramBuilder<ArrayType, ShardMapTensor, XlaPrimitiveOperation>>>,
}

#[derive(Clone)]
pub(crate) struct LinearizedShardMapReplayContext {
    tracing_builder: Rc<RefCell<ProgramBuilder<ArrayType, ShardMapTensor, XlaPrimitiveOperation>>>,
    linear_builder:
        Rc<RefCell<ProgramBuilder<ArrayType, ShardMapTracer, LinearPrimitiveOperation<ArrayType, ShardMapTracer>>>>,
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

fn missing_linearized_shard_map_staging_context() -> TracingError {
    TracingError::Type(TypeError {
        message: "linearized shard_map with non-empty outputs requires at least one traced input leaf".to_string(),
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
    /// Replays this tensor-leaf shard-map op inside explicit traced and linear staging contexts.
    fn interpret_linearized_with_context(
        &self,
        replay_context: &LinearizedShardMapReplayContext,
        inputs: &[Linearized<ShardMapTracer>],
    ) -> Result<Vec<Linearized<ShardMapTracer>>, TracingError> {
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let primal_values = primal_inputs
            .iter()
            .map(|input| ShardMapTensor::new(input.r#type().into_owned()))
            .collect::<Vec<_>>();
        let _primal_output_values = InterpretableOperation::interpret(self, primal_values.as_slice())?;
        let primal_outputs = Tracer::apply_staged_op(
            XlaEngine::token(),
            replay_context.tracing_builder.clone(),
            primal_inputs.as_slice(),
            XlaPrimitiveOperation::ShardMap(Box::new(self.clone())),
        )?;

        let tangent_inputs = inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
        let tangent_outputs = LinearTerm::apply_staged_op(
            replay_context.linear_builder.clone(),
            tangent_inputs.as_slice(),
            LinearPrimitiveOperation::Custom(Arc::new(
                make_linear_shard_map(&self.body, primal_inputs)
                    .map_err(trace_error_from_shard_map)?
                    .to_tracer_linear_custom_primitive(),
            )),
            self.output_types.len(),
        )?;

        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| Linearized { primal, tangent })
            .collect::<Vec<_>>())
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
        Self: Clone + InterpretableOperation<ArrayType, V> + LinearOperation<ArrayType, V> + 'static,
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
    /// Returns the tensor-leaf linear custom primitive registration for this shard-map op.
    pub(crate) fn to_tensor_linear_custom_primitive(&self) -> LinearCustomPrimitive<ArrayType, ShardMapTensor> {
        self.base_custom_primitive()
            .with_extension(self.clone())
            .with_extension(ShardMapCustomReplayExtension::<ShardMapTracer>::new({
                let op = self.clone();
                move |replay_context, inputs| {
                    let traced_op = op.to_linearized_jit_tracer_op(inputs.as_slice())?;
                    traced_op
                        .interpret_with_tracing_builder(replay_context.tracing_builder.clone(), inputs.as_slice())
                        .map_err(ShardMapTraceError::TracingError)
                }
            }))
            .with_extension(ShardMapCustomReplayExtension::<Linearized<ShardMapTracer>>::new({
                let op = self.clone();
                move |replay_context, inputs| {
                    let traced_primals = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
                    let traced_tangents = inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
                    let traced_op = op.to_linearized_jit_tracer_op(traced_primals.as_slice())?;
                    let primal_outputs = traced_op
                        .interpret_with_tracing_builder(
                            replay_context.tracing_builder.clone(),
                            traced_primals.as_slice(),
                        )
                        .map_err(ShardMapTraceError::TracingError)?;
                    let tangent_outputs = LinearTerm::apply_staged_op(
                        replay_context.linear_builder.clone(),
                        traced_tangents.as_slice(),
                        LinearPrimitiveOperation::Custom(Arc::new(traced_op.to_tracer_linear_custom_primitive())),
                        op.output_types.len(),
                    )
                    .map_err(ShardMapTraceError::TracingError)?;
                    Ok(primal_outputs
                        .into_iter()
                        .zip(tangent_outputs)
                        .map(|(primal, tangent)| Linearized { primal, tangent })
                        .collect::<Vec<_>>())
                }
            }))
            .with_extension(StableHloCustomLoweringExtension::new(Arc::new(self.clone())))
            .into_linear()
            .expect("linear tensor shard_map primitive should carry a transpose rule")
    }

    /// Rebuilds this tensor-leaf shard-map op for traced linearized-JIT replay.
    fn to_linearized_jit_tracer_op(
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

    /// Replays this tensor-leaf shard-map op inside explicit traced and linear staging contexts.
    fn interpret_linearized_with_context(
        &self,
        replay_context: &LinearizedShardMapReplayContext,
        inputs: &[Linearized<ShardMapTracer>],
    ) -> Result<Vec<Linearized<ShardMapTracer>>, TracingError> {
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let primal_values = primal_inputs
            .iter()
            .map(|input| ShardMapTensor::new(input.r#type().into_owned()))
            .collect::<Vec<_>>();
        let _primal_output_values = InterpretableOperation::interpret(self, primal_values.as_slice())?;
        let primal_outputs = Tracer::apply_staged_op(
            XlaEngine::token(),
            replay_context.tracing_builder.clone(),
            primal_inputs.as_slice(),
            XlaPrimitiveOperation::LinearShardMap(Box::new(self.clone())),
        )?;

        let tangent_inputs = inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
        let tangent_outputs = LinearTerm::apply_staged_op(
            replay_context.linear_builder.clone(),
            tangent_inputs.as_slice(),
            LinearPrimitiveOperation::Custom(Arc::new(
                self.to_linearized_jit_tracer_op(primal_inputs.as_slice())?.to_tracer_linear_custom_primitive(),
            )),
            self.output_types.len(),
        )?;

        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| Linearized { primal, tangent })
            .collect::<Vec<_>>())
    }
}

impl ShardMapOperation<ShardMapTracer> {
    /// Replays this traced-leaf shard-map op into one explicit outer tracing builder.
    fn interpret_with_tracing_builder(
        &self,
        tracing_builder: Rc<RefCell<ProgramBuilder<ArrayType, ShardMapTensor, XlaPrimitiveOperation>>>,
        inputs: &[ShardMapTracer],
    ) -> Result<Vec<ShardMapTracer>, TracingError> {
        let abstract_inputs = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let _ = self.infer_output_types(abstract_inputs.as_slice())?;
        apply_flat_traced_shard_map(tracing_builder, self.body.clone(), inputs.to_vec())
            .map_err(trace_error_from_shard_map)
    }

    /// Applies this traced-leaf shard-map JVP using one explicit outer linear builder.
    fn jvp_with_builders(
        &self,
        tracing_builder: Rc<RefCell<ProgramBuilder<ArrayType, ShardMapTensor, XlaPrimitiveOperation>>>,
        linear_builder: Rc<
            RefCell<ProgramBuilder<ArrayType, ShardMapTracer, LinearPrimitiveOperation<ArrayType, ShardMapTracer>>>,
        >,
        inputs: &[JvpTracer<
            ShardMapTracer,
            LinearTerm<ArrayType, ShardMapTracer, LinearPrimitiveOperation<ArrayType, ShardMapTracer>>,
        >],
    ) -> Result<
        Vec<
            JvpTracer<
                ShardMapTracer,
                LinearTerm<ArrayType, ShardMapTracer, LinearPrimitiveOperation<ArrayType, ShardMapTracer>>,
            >,
        >,
        TracingError,
    > {
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let primal_outputs = self.interpret_with_tracing_builder(tracing_builder, primal_inputs.as_slice())?;
        let tangent_inputs = inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
        let tangent_outputs = LinearTerm::apply_staged_op(
            linear_builder,
            tangent_inputs.as_slice(),
            LinearPrimitiveOperation::Custom(Arc::new(
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
        tracing_builder: Rc<RefCell<ProgramBuilder<ArrayType, ShardMapTensor, XlaPrimitiveOperation>>>,
        inputs: &[ShardMapTracer],
    ) -> Result<Vec<ShardMapTracer>, TracingError> {
        let abstract_inputs = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let _ = self.infer_output_types(abstract_inputs.as_slice())?;
        Tracer::apply_staged_op(
            XlaEngine::token(),
            tracing_builder,
            inputs,
            XlaPrimitiveOperation::LinearShardMap(Box::new(self.to_tensor_xla_op())),
        )
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
/// caller scope. When the caller's ambient sharding envelope differs from the captured shard-map
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

impl Operation for ShardMapOperation<ShardMapTensor> {
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

impl InterpretableOperation<ArrayType, Linearized<ShardMapTracer>> for ShardMapOperation<ShardMapTensor> {
    fn interpret(
        &self,
        inputs: &[Linearized<ShardMapTracer>],
    ) -> Result<Vec<Linearized<ShardMapTracer>>, TracingError> {
        let Some(first_input) = inputs.first() else {
            return if self.output_types.is_empty() {
                Ok(Vec::new())
            } else {
                Err(missing_linearized_shard_map_staging_context())
            };
        };
        let replay_context = LinearizedShardMapReplayContext {
            tracing_builder: first_input.primal.builder.clone(),
            linear_builder: first_input.tangent.builder.clone(),
        };
        self.interpret_linearized_with_context(&replay_context, inputs)
    }
}

impl Operation for LinearShardMapOperation<ShardMapTensor> {
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

impl InterpretableOperation<ArrayType, Linearized<ShardMapTracer>> for LinearShardMapOperation<ShardMapTensor> {
    fn interpret(
        &self,
        inputs: &[Linearized<ShardMapTracer>],
    ) -> Result<Vec<Linearized<ShardMapTracer>>, TracingError> {
        let Some(first_input) = inputs.first() else {
            return if self.output_types.is_empty() {
                Ok(Vec::new())
            } else {
                Err(missing_linearized_shard_map_staging_context())
            };
        };
        let replay_context = LinearizedShardMapReplayContext {
            tracing_builder: first_input.primal.builder.clone(),
            linear_builder: first_input.tangent.builder.clone(),
        };
        self.interpret_linearized_with_context(&replay_context, inputs)
    }
}

impl InterpretableOperation<ArrayType, ShardMapTensor> for LinearShardMapOperation<ShardMapTensor> {
    fn interpret(&self, inputs: &[ShardMapTensor]) -> Result<Vec<ShardMapTensor>, TracingError> {
        let abstract_inputs = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let _ = self.infer_output_types(abstract_inputs.as_slice())?;
        Ok(self.output_types.iter().cloned().map(ShardMapTensor::new).collect::<Vec<_>>())
    }
}

impl LinearOperation<ArrayType, ShardMapTensor> for LinearShardMapOperation<ShardMapTensor> {
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, ShardMapTensor>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, ShardMapTensor>>>, TracingError> {
        if output_cotangents.len() != self.output_types.len() {
            return Err(TracingError::InvalidInputCount {
                expected: self.output_types.len(),
                got: output_cotangents.len(),
            });
        }
        let contributions = LinearTerm::apply_staged_op(
            output_cotangents[0].builder.clone(),
            output_cotangents,
            LinearPrimitiveOperation::Custom(Arc::new(self.transpose_op().to_tensor_linear_custom_primitive())),
            self.input_types.len(),
        )?;
        Ok(contributions.into_iter().map(Some).collect::<Vec<_>>())
    }
}

impl
    DifferentiableOperation<
        ArrayType,
        ShardMapTensor,
        LinearTerm<ArrayType, ShardMapTensor, LinearPrimitiveOperation<ArrayType, ShardMapTensor>>,
        XlaPrimitiveOperation,
        LinearPrimitiveOperation<ArrayType, ShardMapTensor>,
    > for ShardMapOperation<ShardMapTensor>
{
    fn jvp(
        &self,
        _engine: &dyn Engine<
            Type = ArrayType,
            Value = ShardMapTensor,
            TracingOperation = XlaPrimitiveOperation,
            LinearOperation = LinearPrimitiveOperation<ArrayType, ShardMapTensor>,
        >,
        inputs: &[JvpTracer<
            ShardMapTensor,
            LinearTerm<ArrayType, ShardMapTensor, LinearPrimitiveOperation<ArrayType, ShardMapTensor>>,
        >],
    ) -> Result<
        Vec<
            JvpTracer<
                ShardMapTensor,
                LinearTerm<ArrayType, ShardMapTensor, LinearPrimitiveOperation<ArrayType, ShardMapTensor>>,
            >,
        >,
        TracingError,
    > {
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let primal_outputs = InterpretableOperation::interpret(self, primal_inputs.as_slice())?;
        let tangent_inputs = inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
        let Some(first_tangent) = tangent_inputs.first() else {
            return if self.output_types.is_empty() {
                Ok(Vec::new())
            } else {
                Err(missing_linearized_shard_map_staging_context())
            };
        };
        let tangent_outputs = LinearTerm::apply_staged_op(
            first_tangent.builder.clone(),
            tangent_inputs.as_slice(),
            LinearPrimitiveOperation::Custom(Arc::new(
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

impl
    DifferentiableOperation<
        ArrayType,
        ShardMapTensor,
        LinearTerm<ArrayType, ShardMapTensor, LinearPrimitiveOperation<ArrayType, ShardMapTensor>>,
        XlaPrimitiveOperation,
        LinearPrimitiveOperation<ArrayType, ShardMapTensor>,
    > for LinearShardMapOperation<ShardMapTensor>
{
    fn jvp(
        &self,
        _engine: &dyn Engine<
            Type = ArrayType,
            Value = ShardMapTensor,
            TracingOperation = XlaPrimitiveOperation,
            LinearOperation = LinearPrimitiveOperation<ArrayType, ShardMapTensor>,
        >,
        inputs: &[JvpTracer<
            ShardMapTensor,
            LinearTerm<ArrayType, ShardMapTensor, LinearPrimitiveOperation<ArrayType, ShardMapTensor>>,
        >],
    ) -> Result<
        Vec<
            JvpTracer<
                ShardMapTensor,
                LinearTerm<ArrayType, ShardMapTensor, LinearPrimitiveOperation<ArrayType, ShardMapTensor>>,
            >,
        >,
        TracingError,
    > {
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let primal_outputs = InterpretableOperation::interpret(self, primal_inputs.as_slice())?;
        let tangent_inputs = inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
        let Some(first_tangent) = tangent_inputs.first() else {
            return if self.output_types.is_empty() {
                Ok(Vec::new())
            } else {
                Err(missing_linearized_shard_map_staging_context())
            };
        };
        let tangent_outputs = LinearTerm::apply_staged_op(
            first_tangent.builder.clone(),
            tangent_inputs.as_slice(),
            LinearPrimitiveOperation::Custom(Arc::new(self.to_tensor_linear_custom_primitive())),
            self.output_types.len(),
        )?;
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| JvpTracer { primal, tangent })
            .collect::<Vec<_>>())
    }
}

impl Operation for ShardMapOperation<ShardMapTracer> {
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
            Some(input) => input.builder.clone(),
            None if self.output_types.is_empty() => return Ok(Vec::new()),
            None => return Err(missing_traced_shard_map_staging_context()),
        };
        self.interpret_with_tracing_builder(tracing_builder, inputs)
    }
}

impl Operation for LinearShardMapOperation<ShardMapTracer> {
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
            Some(input) => input.builder.clone(),
            None if self.output_types.is_empty() => return Ok(Vec::new()),
            None => return Err(missing_traced_shard_map_staging_context()),
        };
        self.interpret_with_tracing_builder(tracing_builder, inputs)
    }
}

impl LinearOperation<ArrayType, ShardMapTracer> for LinearShardMapOperation<ShardMapTracer> {
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, ShardMapTracer>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, ShardMapTracer>>>, TracingError> {
        if output_cotangents.len() != self.output_types.len() {
            return Err(TracingError::InvalidInputCount {
                expected: self.output_types.len(),
                got: output_cotangents.len(),
            });
        }
        let contributions = LinearTerm::apply_staged_op(
            output_cotangents[0].builder.clone(),
            output_cotangents,
            LinearPrimitiveOperation::Custom(Arc::new(self.transpose_op().to_tracer_linear_custom_primitive())),
            self.input_types.len(),
        )?;
        Ok(contributions.into_iter().map(Some).collect::<Vec<_>>())
    }
}

impl
    DifferentiableOperation<
        ArrayType,
        ShardMapTracer,
        LinearTerm<ArrayType, ShardMapTracer, LinearPrimitiveOperation<ArrayType, ShardMapTracer>>,
        PrimitiveOperation<ArrayType, ShardMapTracer>,
        LinearPrimitiveOperation<ArrayType, ShardMapTracer>,
    > for ShardMapOperation<ShardMapTracer>
{
    fn jvp(
        &self,
        _engine: &dyn Engine<
            Type = ArrayType,
            Value = ShardMapTracer,
            TracingOperation = PrimitiveOperation<ArrayType, ShardMapTracer>,
            LinearOperation = LinearPrimitiveOperation<ArrayType, ShardMapTracer>,
        >,
        _inputs: &[JvpTracer<
            ShardMapTracer,
            LinearTerm<ArrayType, ShardMapTracer, LinearPrimitiveOperation<ArrayType, ShardMapTracer>>,
        >],
    ) -> Result<
        Vec<
            JvpTracer<
                ShardMapTracer,
                LinearTerm<ArrayType, ShardMapTracer, LinearPrimitiveOperation<ArrayType, ShardMapTracer>>,
            >,
        >,
        TracingError,
    > {
        let Some(first_input) = _inputs.first() else {
            return if self.output_types.is_empty() {
                Ok(Vec::new())
            } else {
                Err(missing_linearized_shard_map_staging_context())
            };
        };
        self.jvp_with_builders(first_input.primal.builder.clone(), first_input.tangent.builder.clone(), _inputs)
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

pub(crate) trait ReplayShardMapValue:
    Clone
    + Traceable<ArrayType>
    + Add<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + Sin
    + Cos
    + MatrixOps
    + ReshapeOps
    + ZeroLike
    + OneLike
{
    type ReplayContext;

    fn lift_constant(constant: &ShardMapTensor, replay_context: &Self::ReplayContext) -> Result<Self, TracingError>;

    fn apply_shard_map_operation(
        replay_context: &Self::ReplayContext,
        shard_map_op: &ShardMapOperation<ShardMapTensor>,
        inputs: Vec<Self>,
    ) -> Result<Vec<Self>, ShardMapTraceError>;

    fn apply_linear_shard_map_operation(
        replay_context: &Self::ReplayContext,
        shard_map_op: &LinearShardMapOperation<ShardMapTensor>,
        inputs: Vec<Self>,
    ) -> Result<Vec<Self>, ShardMapTraceError>;

    fn apply_custom_operation(
        replay_context: &Self::ReplayContext,
        custom_op: &CustomPrimitive<ArrayType, ShardMapTensor>,
        inputs: Vec<Self>,
    ) -> Result<Vec<Self>, ShardMapTraceError>;
}

#[derive(Clone)]
pub(crate) struct ShardMapCustomReplayExtension<V: ReplayShardMapValue + 'static> {
    replay: Arc<dyn Fn(&V::ReplayContext, Vec<V>) -> Result<Vec<V>, ShardMapTraceError>>,
}

impl<V: ReplayShardMapValue + 'static> ShardMapCustomReplayExtension<V> {
    pub(crate) fn new<F>(replay: F) -> Self
    where
        F: Fn(&V::ReplayContext, Vec<V>) -> Result<Vec<V>, ShardMapTraceError> + 'static,
    {
        Self { replay: Arc::new(replay) }
    }

    fn replay(&self, replay_context: &V::ReplayContext, inputs: Vec<V>) -> Result<Vec<V>, ShardMapTraceError> {
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
        builder: &mut ProgramBuilder<ArrayType, ShardMapTensor, XlaPrimitiveOperation>,
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
    let mut builder = ProgramBuilder::<ArrayType, ShardMapTensor, XlaPrimitiveOperation>::new();
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
    Ok(builder.build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(
        projected_outputs,
        vec![ryft_core::parameters::Placeholder; kept_input_atoms.len()],
        vec![ryft_core::parameters::Placeholder; output_atoms.len()],
    ))
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
        builder: &mut ProgramBuilder<ArrayType, ShardMapTensor, XlaPrimitiveOperation>,
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
    let mut builder = ProgramBuilder::<ArrayType, ShardMapTensor, XlaPrimitiveOperation>::new();
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
    Ok(builder.build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(
        outputs,
        vec![ryft_core::parameters::Placeholder; cotangent_input_atoms.len() + residual_atoms.len()],
        vec![ryft_core::parameters::Placeholder; program.output_ids.len()],
    ))
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

fn try_linearize_traced_shard_map_body<
    F: FnOnce(
        Rc<RefCell<ProgramBuilder<ArrayType, ShardMapTracer, LinearPrimitiveOperation<ArrayType, ShardMapTracer>>>>,
        Vec<Linearized<ShardMapTracer>>,
    ) -> Result<Vec<Linearized<ShardMapTracer>>, TracingError>,
>(
    function: F,
    primals: Vec<ShardMapTracer>,
) -> Result<
    (
        Vec<ShardMapTracer>,
        Program<
            ArrayType,
            ShardMapTracer,
            LinearPrimitiveOperation<ArrayType, ShardMapTracer>,
            Vec<ShardMapTracer>,
            Vec<ShardMapTracer>,
        >,
    ),
    TracingError,
> {
    let input_structure = vec![ryft_core::parameters::Placeholder; primals.len()];
    let builder = std::rc::Rc::new(std::cell::RefCell::new(ProgramBuilder::<
        ArrayType,
        ShardMapTracer,
        LinearPrimitiveOperation<ArrayType, ShardMapTracer>,
    >::new()));
    let traced_input = primals
        .into_iter()
        .map(|primal| {
            let atom = builder.borrow_mut().add_input(primal.r#type().into_owned());
            Linearized { primal, tangent: LinearTerm::from_staged_parts(atom, builder.clone()) }
        })
        .collect::<Vec<_>>();
    let traced_output = function(builder.clone(), traced_input)?;
    let output_structure = vec![ryft_core::parameters::Placeholder; traced_output.len()];
    let primal_outputs = traced_output.iter().map(|output| output.primal.clone()).collect::<Vec<_>>();
    let tangent_outputs = traced_output.iter().map(|output| output.tangent.atom).collect::<Vec<_>>();
    drop(traced_output);
    let builder = match std::rc::Rc::try_unwrap(builder) {
        Ok(builder) => builder.into_inner(),
        Err(_) => {
            return Err(TracingError::EscapedProgramBuilder);
        }
    };
    let program =
        builder.build::<Vec<ShardMapTracer>, Vec<ShardMapTracer>>(tangent_outputs, input_structure, output_structure);
    let program = program.with_folded_constants()?.simplified()?;
    Ok((primal_outputs.clone(), program))
}

fn try_transpose_traced_shard_map_body<
    F: FnOnce(
        Rc<RefCell<ProgramBuilder<ArrayType, ShardMapTracer, LinearPrimitiveOperation<ArrayType, ShardMapTracer>>>>,
        Vec<Linearized<ShardMapTracer>>,
    ) -> Result<Vec<Linearized<ShardMapTracer>>, TracingError>,
>(
    tracing_builder: Rc<RefCell<ProgramBuilder<ArrayType, ShardMapTensor, XlaPrimitiveOperation>>>,
    function: F,
    primals: Vec<ShardMapTracer>,
) -> Result<
    (
        Vec<ShardMapTracer>,
        Program<
            ArrayType,
            ShardMapTracer,
            LinearPrimitiveOperation<ArrayType, ShardMapTracer>,
            Vec<ShardMapTracer>,
            Vec<ShardMapTracer>,
        >,
    ),
    TracingError,
> {
    let (outputs, pushforward) = try_linearize_traced_shard_map_body(function, primals)?;
    let pullback = ryft_core::tracing_v2::linear::transpose_traced_linear_program(
        XlaEngine::token(),
        tracing_builder,
        &pushforward,
    )?;
    Ok((outputs, pullback))
}

fn apply_flat_traced_shard_map(
    tracing_builder: Rc<RefCell<ProgramBuilder<ArrayType, ShardMapTensor, XlaPrimitiveOperation>>>,
    body: FlatTracedShardMap,
    traced_inputs: Vec<ShardMapTracer>,
) -> Result<Vec<ShardMapTracer>, ShardMapTraceError> {
    Tracer::apply_staged_op(
        XlaEngine::token(),
        tracing_builder,
        traced_inputs.as_slice(),
        XlaPrimitiveOperation::ShardMap(Box::new(ShardMapOperation::new(body.clone()))),
    )
    .map_err(ShardMapTraceError::from)
}

fn replay_traced_xla_program<
    ProgramInput: ryft_core::parameters::Parameterized<ShardMapTensor>,
    ProgramOutput: ryft_core::parameters::Parameterized<ShardMapTensor>,
    V: ReplayShardMapValue,
>(
    replay_context: &V::ReplayContext,
    program: &Program<ArrayType, ShardMapTensor, XlaPrimitiveOperation, ProgramInput, ProgramOutput>,
    inputs: Vec<V>,
) -> Result<Vec<V>, ShardMapTraceError> {
    let mut values = vec![None; program.atoms.len()];
    for (atom_id, value) in program.input_ids.iter().copied().zip(inputs.iter().cloned()) {
        values[atom_id.index] = Some(value);
    }

    let mut instruction_by_first_output = vec![None; program.atoms.len()];
    for (instruction_index, instruction) in program.instructions.iter().enumerate() {
        if let Some(first_output) = instruction.outputs.first() {
            instruction_by_first_output[first_output.index] = Some(instruction_index);
        }
    }
    let mut input_atom_flags = vec![false; program.atoms.len()];
    for input_atom in program.input_ids.iter().copied() {
        input_atom_flags[input_atom.index] = true;
    }

    for atom_index in 0..program.atoms.len() {
        let atom = &program.atoms[atom_index];
        match atom {
            Atom::Constant(value) => {
                values[atom_index] = Some(V::lift_constant(value, replay_context)?);
            }
            Atom::Variable(_) if input_atom_flags[atom_index] => {}
            Atom::Variable(_) => {
                let Some(instruction_index) = instruction_by_first_output[atom_index] else {
                    continue;
                };
                let instruction = &program.instructions[instruction_index];
                let input_values = instruction
                    .inputs
                    .iter()
                    .map(|input| values[input.index].clone().ok_or(TracingError::UnboundAtomId { id: *input }))
                    .collect::<Result<Vec<_>, _>>()?;
                let outputs = match &instruction.operation {
                    XlaPrimitiveOperation::ShardMap(shard_map_op) => {
                        V::apply_shard_map_operation(replay_context, shard_map_op.as_ref(), input_values)?
                    }
                    XlaPrimitiveOperation::LinearShardMap(shard_map_op) => {
                        V::apply_linear_shard_map_operation(replay_context, shard_map_op.as_ref(), input_values)?
                    }
                    XlaPrimitiveOperation::WithShardingConstraint(_) => vec![input_values[0].clone()],
                    XlaPrimitiveOperation::Custom(custom_op) => {
                        V::apply_custom_operation(replay_context, custom_op.as_ref(), input_values)?
                    }
                    XlaPrimitiveOperation::Add => vec![input_values[0].clone() + input_values[1].clone()],
                    XlaPrimitiveOperation::Mul => vec![input_values[0].clone() * input_values[1].clone()],
                    XlaPrimitiveOperation::Neg => vec![-input_values[0].clone()],
                    XlaPrimitiveOperation::Sin => vec![input_values[0].clone().sin()],
                    XlaPrimitiveOperation::Cos => vec![input_values[0].clone().cos()],
                    XlaPrimitiveOperation::MatMul => vec![input_values[0].clone().matmul(input_values[1].clone())],
                    XlaPrimitiveOperation::MatrixTranspose => vec![input_values[0].clone().transpose_matrix()],
                    XlaPrimitiveOperation::Scale { factor } => {
                        vec![V::lift_constant(factor, replay_context)? * input_values[0].clone()]
                    }
                    XlaPrimitiveOperation::LeftMatMul { factor } => {
                        vec![V::lift_constant(factor, replay_context)?.matmul(input_values[0].clone())]
                    }
                    XlaPrimitiveOperation::RightMatMul { factor } => {
                        vec![input_values[0].clone().matmul(V::lift_constant(factor, replay_context)?)]
                    }
                    XlaPrimitiveOperation::Reshape { output_type, .. } => {
                        vec![input_values[0].clone().reshape(output_type.shape.clone())?]
                    }
                    XlaPrimitiveOperation::VMap(vmap) => {
                        if input_values.len() != vmap.body().total_input_count() {
                            return Err(ShardMapTraceError::TracingError(TracingError::InvalidInputCount {
                                expected: vmap.body().total_input_count(),
                                got: input_values.len(),
                            }));
                        }
                        let lane_input_count = vmap.body().input_types().len();
                        let mut outputs = Vec::with_capacity(vmap.body().total_output_count());
                        for lane_inputs in input_values.chunks(lane_input_count) {
                            outputs.extend(replay_traced_xla_program(
                                replay_context,
                                &vmap.body().program(),
                                lane_inputs.to_vec(),
                            )?);
                        }
                        outputs
                    }
                    XlaPrimitiveOperation::Rematerialize(remat) => {
                        replay_traced_xla_program(replay_context, &remat.body().program(), input_values)?
                    }
                };
                for (output_atom, output_value) in instruction.outputs.iter().copied().zip(outputs) {
                    values[output_atom.index] = Some(output_value);
                }
            }
        }
    }

    program
        .output_ids
        .iter()
        .map(|output| {
            values[output.index]
                .clone()
                .ok_or(ShardMapTraceError::TracingError(TracingError::UnboundAtomId { id: *output }))
        })
        .collect()
}

fn replay_flat_program<V: ReplayShardMapValue>(
    replay_context: &V::ReplayContext,
    body: &FlatTracedShardMap,
    inputs: Vec<V>,
) -> Result<Vec<V>, ShardMapTraceError> {
    replay_traced_xla_program(replay_context, &body.program, inputs)
}

fn build_traced_xla_program(
    tracing_builder: Rc<RefCell<ProgramBuilder<ArrayType, ShardMapTensor, XlaPrimitiveOperation>>>,
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
    let program = tracing_builder.build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(
        output_atoms,
        vec![ryft_core::parameters::Placeholder; input_count],
        vec![ryft_core::parameters::Placeholder; output_count],
    );
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
        Rc::new(RefCell::new(ProgramBuilder::<ArrayType, ShardMapTensor, XlaPrimitiveOperation>::new()));
    let pushforward_compiled_outputs = {
        let combined_inputs = pushforward_local_input_types
            .iter()
            .cloned()
            .map(|input_type| {
                let atom = pushforward_compiled_builder.borrow_mut().add_input(input_type);
                Tracer::from_engine(atom, pushforward_compiled_builder.clone(), XlaEngine::token())
            })
            .collect::<Vec<_>>();
        let local_primals = combined_inputs[..local_input_count].to_vec();
        let local_tangents = combined_inputs[local_input_count..].to_vec();
        let (_, pushforward_program) = try_linearize_traced_shard_map_body(
            {
                let body = body.clone();
                let pushforward_compiled_builder = pushforward_compiled_builder.clone();
                move |linear_builder, replay_inputs: Vec<Linearized<ShardMapTracer>>| {
                    let replay_context = LinearizedShardMapReplayContext {
                        tracing_builder: pushforward_compiled_builder.clone(),
                        linear_builder,
                    };
                    replay_flat_program(&replay_context, &body, replay_inputs).map_err(trace_error_from_shard_map)
                }
            },
            local_primals,
        )?;
        pushforward_program.interpret(local_tangents)?
    };
    let pushforward_compiled = build_traced_xla_program(
        pushforward_compiled_builder,
        pushforward_compiled_outputs,
        local_input_count * 2,
        local_output_count,
    )?;

    let pullback_compiled_builder =
        Rc::new(RefCell::new(ProgramBuilder::<ArrayType, ShardMapTensor, XlaPrimitiveOperation>::new()));
    let pullback_compiled_outputs = {
        let combined_inputs = pullback_local_input_types
            .iter()
            .cloned()
            .map(|input_type| {
                let atom = pullback_compiled_builder.borrow_mut().add_input(input_type);
                Tracer::from_engine(atom, pullback_compiled_builder.clone(), XlaEngine::token())
            })
            .collect::<Vec<_>>();
        let local_primals = combined_inputs[..local_input_count].to_vec();
        let local_output_cotangents = combined_inputs[local_input_count..].to_vec();
        let (_, pullback_program) = try_transpose_traced_shard_map_body(
            pullback_compiled_builder.clone(),
            {
                let body = body.clone();
                let pullback_compiled_builder = pullback_compiled_builder.clone();
                move |linear_builder, replay_inputs: Vec<Linearized<ShardMapTracer>>| {
                    let replay_context = LinearizedShardMapReplayContext {
                        tracing_builder: pullback_compiled_builder.clone(),
                        linear_builder,
                    };
                    replay_flat_program(&replay_context, &body, replay_inputs).map_err(trace_error_from_shard_map)
                }
            },
            local_primals,
        )?;
        pullback_program.interpret(local_output_cotangents)?
    };
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
            Program {
                atoms: pushforward_compiled.atoms.clone(),
                input_ids: pushforward_compiled.input_ids.clone(),
                output_ids: pushforward_compiled.output_ids.clone(),
                instructions: pushforward_compiled.instructions.clone(),
                input_structure: vec![ryft_core::parameters::Placeholder; local_input_count * 2],
                output_structure: vec![ryft_core::parameters::Placeholder; local_output_count],
                marker: std::marker::PhantomData,
            },
        ),
        pullback: FlatTracedShardMap::from_parts(
            pullback_shard_map,
            pullback_global_input_types,
            pullback_local_input_types,
            body.global_input_types.clone(),
            body.local_input_types.clone(),
            Program {
                atoms: pullback_compiled.atoms.clone(),
                input_ids: pullback_compiled.input_ids.clone(),
                output_ids: pullback_compiled.output_ids.clone(),
                instructions: pullback_compiled.instructions.clone(),
                input_structure: vec![ryft_core::parameters::Placeholder; local_input_count + local_output_count],
                output_structure: vec![ryft_core::parameters::Placeholder; local_input_count],
                marker: std::marker::PhantomData,
            },
        ),
    })
}

/// Applies one linearized shard-map body to already-traced values.
pub(crate) fn apply_linearized_flat_shard_map(
    tracing_builder: Rc<RefCell<ProgramBuilder<ArrayType, ShardMapTensor, XlaPrimitiveOperation>>>,
    linear_builder: Rc<
        RefCell<ProgramBuilder<ArrayType, ShardMapTracer, LinearPrimitiveOperation<ArrayType, ShardMapTracer>>>,
    >,
    body: FlatTracedShardMap,
    traced_inputs: Vec<Linearized<ShardMapTracer>>,
) -> Result<Vec<Linearized<ShardMapTracer>>, ShardMapTraceError> {
    let traced_primals = traced_inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
    let traced_tangents = traced_inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
    let primal_outputs = apply_flat_traced_shard_map(tracing_builder, body.clone(), traced_primals.clone())?;
    let tangent_outputs = LinearTerm::apply_staged_op(
        linear_builder,
        traced_tangents.as_slice(),
        LinearPrimitiveOperation::Custom(Arc::new(
            make_linear_shard_map(&body, traced_primals)?.to_tracer_linear_custom_primitive(),
        )),
        body.global_output_types.len(),
    )?;
    Ok(primal_outputs
        .into_iter()
        .zip(tangent_outputs)
        .map(|(primal, tangent)| Linearized { primal, tangent })
        .collect::<Vec<_>>())
}

impl ReplayShardMapValue for ShardMapTracer {
    type ReplayContext = ShardMapReplayContext;

    fn lift_constant(constant: &ShardMapTensor, replay_context: &Self::ReplayContext) -> Result<Self, TracingError> {
        let atom = replay_context.tracing_builder.borrow_mut().add_constant(constant.clone());
        Ok(Tracer::from_engine(atom, replay_context.tracing_builder.clone(), XlaEngine::token()))
    }

    fn apply_shard_map_operation(
        replay_context: &Self::ReplayContext,
        shard_map_op: &ShardMapOperation<ShardMapTensor>,
        inputs: Vec<Self>,
    ) -> Result<Vec<Self>, ShardMapTraceError> {
        apply_flat_traced_shard_map(replay_context.tracing_builder.clone(), shard_map_op.body().clone(), inputs)
    }

    fn apply_linear_shard_map_operation(
        replay_context: &Self::ReplayContext,
        shard_map_op: &LinearShardMapOperation<ShardMapTensor>,
        inputs: Vec<Self>,
    ) -> Result<Vec<Self>, ShardMapTraceError> {
        let traced_op = shard_map_op.to_linearized_jit_tracer_op(inputs.as_slice())?;
        traced_op
            .interpret_with_tracing_builder(replay_context.tracing_builder.clone(), inputs.as_slice())
            .map_err(ShardMapTraceError::TracingError)
    }

    fn apply_custom_operation(
        replay_context: &Self::ReplayContext,
        custom_op: &CustomPrimitive<ArrayType, ShardMapTensor>,
        inputs: Vec<Self>,
    ) -> Result<Vec<Self>, ShardMapTraceError> {
        let replay_extension =
            custom_op.extensions().get::<ShardMapCustomReplayExtension<Self>>().ok_or_else(|| {
                ShardMapTraceError::TracingError(TracingError::CustomOperation(CustomOperationError::MissingRule {
                    op: custom_op.name(),
                    transform: "shard_map replay",
                }))
            })?;
        replay_extension.replay(replay_context, inputs)
    }
}

impl ReplayShardMapValue for Linearized<ShardMapTracer> {
    type ReplayContext = LinearizedShardMapReplayContext;

    fn lift_constant(constant: &ShardMapTensor, replay_context: &Self::ReplayContext) -> Result<Self, TracingError> {
        let primal = <ShardMapTracer as ReplayShardMapValue>::lift_constant(
            constant,
            &ShardMapReplayContext { tracing_builder: replay_context.tracing_builder.clone() },
        )?;
        let zero = primal.zero_like();
        let tangent_atom = replay_context.linear_builder.borrow_mut().add_constant(zero);
        let tangent = LinearTerm::from_staged_parts(tangent_atom, replay_context.linear_builder.clone());
        Ok(Linearized { primal, tangent })
    }

    fn apply_shard_map_operation(
        replay_context: &Self::ReplayContext,
        shard_map_op: &ShardMapOperation<ShardMapTensor>,
        inputs: Vec<Self>,
    ) -> Result<Vec<Self>, ShardMapTraceError> {
        apply_linearized_flat_shard_map(
            replay_context.tracing_builder.clone(),
            replay_context.linear_builder.clone(),
            shard_map_op.body().clone(),
            inputs,
        )
    }

    fn apply_linear_shard_map_operation(
        replay_context: &Self::ReplayContext,
        shard_map_op: &LinearShardMapOperation<ShardMapTensor>,
        inputs: Vec<Self>,
    ) -> Result<Vec<Self>, ShardMapTraceError> {
        let traced_primals = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let traced_tangents = inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
        let traced_op = shard_map_op.to_linearized_jit_tracer_op(traced_primals.as_slice())?;
        let primal_outputs = traced_op
            .interpret_with_tracing_builder(replay_context.tracing_builder.clone(), traced_primals.as_slice())
            .map_err(ShardMapTraceError::TracingError)?;
        let tangent_outputs = LinearTerm::apply_staged_op(
            replay_context.linear_builder.clone(),
            traced_tangents.as_slice(),
            LinearPrimitiveOperation::Custom(Arc::new(traced_op.to_tracer_linear_custom_primitive())),
            shard_map_op.output_types.len(),
        )
        .map_err(ShardMapTraceError::TracingError)?;
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| Linearized { primal, tangent })
            .collect::<Vec<_>>())
    }

    fn apply_custom_operation(
        replay_context: &Self::ReplayContext,
        custom_op: &CustomPrimitive<ArrayType, ShardMapTensor>,
        inputs: Vec<Self>,
    ) -> Result<Vec<Self>, ShardMapTraceError> {
        let replay_extension =
            custom_op.extensions().get::<ShardMapCustomReplayExtension<Self>>().ok_or_else(|| {
                ShardMapTraceError::TracingError(TracingError::CustomOperation(CustomOperationError::MissingRule {
                    op: custom_op.name(),
                    transform: "shard_map replay",
                }))
            })?;
        replay_extension.replay(replay_context, inputs)
    }
}

fn trace_flat_shard_map<
    F: FnOnce(ShardMapLocalTraceInput<Input>) -> ShardMapLocalTraceOutput<Output>,
    Input: Parameterized<ArrayType, ParameterStructure: Clone>,
    Output: Parameterized<ArrayType, ParameterStructure: Clone>,
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
    tracing_builder: Rc<RefCell<ProgramBuilder<ArrayType, ShardMapTensor, XlaPrimitiveOperation>>>,
    traced: FlatTracedShardMap,
    traced_inputs: Vec<ShardMapTracer>,
    output_structure: Output::ParameterStructure,
) -> Result<Output, ShardMapTraceError> {
    let staged_outputs = Tracer::apply_staged_op(
        XlaEngine::token(),
        tracing_builder,
        traced_inputs.as_slice(),
        XlaPrimitiveOperation::ShardMap(Box::new(ShardMapOperation::new(traced.clone()))),
    )?;
    Ok(Output::from_parameters(output_structure, staged_outputs)?)
}

fn global_input_types_from_traced_inputs<Input: Parameterized<ShardMapTracer, ParameterStructure: Clone>>(
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
    type Return<
        Input: Parameterized<Self, ParameterStructure: Clone>,
        Output: Parameterized<ArrayType, ParameterStructure: Clone>,
    >
        = TracedShardMap<Input, Output>
    where
        Input::Family: ParameterizedFamily<ArrayType>
            + ParameterizedFamily<Sharding>
            + ParameterizedFamily<ShardMapTensor>
            + ParameterizedFamily<ShardMapTracer>,
        Output::Family: ParameterizedFamily<Sharding>
            + ParameterizedFamily<ShardMapTensor>
            + ParameterizedFamily<ShardMapTracer>
            + ParameterizedFamily<Linearized<ShardMapTracer>>;

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
        Input: Parameterized<Self, ParameterStructure: Clone>,
        Input::Family: ParameterizedFamily<ArrayType>
            + ParameterizedFamily<Sharding>
            + ParameterizedFamily<ShardMapTensor>
            + ParameterizedFamily<ShardMapTracer>,
        Output: Parameterized<ArrayType, ParameterStructure: Clone>,
        Output::Family: ParameterizedFamily<Sharding>
            + ParameterizedFamily<ShardMapTensor>
            + ParameterizedFamily<ShardMapTracer>
            + ParameterizedFamily<Linearized<ShardMapTracer>>,
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
    type Return<
        Input: Parameterized<Self, ParameterStructure: Clone>,
        Output: Parameterized<ArrayType, ParameterStructure: Clone>,
    >
        = Output::To<ShardMapTracer>
    where
        Input::Family: ParameterizedFamily<ArrayType>
            + ParameterizedFamily<Sharding>
            + ParameterizedFamily<ShardMapTensor>
            + ParameterizedFamily<ShardMapTracer>,
        Output::Family: ParameterizedFamily<Sharding>
            + ParameterizedFamily<ShardMapTensor>
            + ParameterizedFamily<ShardMapTracer>
            + ParameterizedFamily<Linearized<ShardMapTracer>>;

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
        Input: Parameterized<Self, ParameterStructure: Clone>,
        Input::Family: ParameterizedFamily<ArrayType>
            + ParameterizedFamily<Sharding>
            + ParameterizedFamily<ShardMapTensor>
            + ParameterizedFamily<ShardMapTracer>,
        Output: Parameterized<ArrayType, ParameterStructure: Clone>,
        Output::Family: ParameterizedFamily<Sharding>
            + ParameterizedFamily<ShardMapTensor>
            + ParameterizedFamily<ShardMapTracer>
            + ParameterizedFamily<Linearized<ShardMapTracer>>,
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
            Some(input) => input.builder.clone(),
            None if output_structure.parameter_count() == 0 => {
                return Ok(Output::To::<ShardMapTracer>::from_parameters(output_structure, Vec::new())?);
            }
            None => return Err(ShardMapTraceError::MissingTracedInvocationContext),
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

impl ShardMapInvocationLeaf for Linearized<ShardMapTracer> {
    type Return<
        Input: Parameterized<Self, ParameterStructure: Clone>,
        Output: Parameterized<ArrayType, ParameterStructure: Clone>,
    >
        = Output::To<Linearized<ShardMapTracer>>
    where
        Input::Family: ParameterizedFamily<ArrayType>
            + ParameterizedFamily<Sharding>
            + ParameterizedFamily<ShardMapTensor>
            + ParameterizedFamily<ShardMapTracer>,
        Output::Family: ParameterizedFamily<Sharding>
            + ParameterizedFamily<ShardMapTensor>
            + ParameterizedFamily<ShardMapTracer>
            + ParameterizedFamily<Linearized<ShardMapTracer>>;

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
        Input: Parameterized<Self, ParameterStructure: Clone>,
        Input::Family: ParameterizedFamily<ArrayType>
            + ParameterizedFamily<Sharding>
            + ParameterizedFamily<ShardMapTensor>
            + ParameterizedFamily<ShardMapTracer>,
        Output: Parameterized<ArrayType, ParameterStructure: Clone>,
        Output::Family: ParameterizedFamily<Sharding>
            + ParameterizedFamily<ShardMapTensor>
            + ParameterizedFamily<ShardMapTracer>
            + ParameterizedFamily<Linearized<ShardMapTracer>>,
        F: FnOnce(ShardMapLocalTraceInput<Input::To<ArrayType>>) -> ShardMapLocalTraceOutput<Output>,
    {
        let input_structure = inputs.parameter_structure();
        let output_structure = out_specs.parameter_structure();
        let traced_inputs = inputs.into_parameters().collect::<Vec<_>>();
        let tracing_builder = match traced_inputs.first() {
            Some(input) => input.primal.builder.clone(),
            None if output_structure.parameter_count() == 0 => {
                return Ok(Output::To::<Linearized<ShardMapTracer>>::from_parameters(output_structure, Vec::new())?);
            }
            None => return Err(ShardMapTraceError::MissingLinearizedInvocationContext),
        };
        let linear_builder = match traced_inputs.first() {
            Some(input) => input.tangent.builder.clone(),
            None if output_structure.parameter_count() == 0 => {
                return Ok(Output::To::<Linearized<ShardMapTracer>>::from_parameters(output_structure, Vec::new())?);
            }
            None => return Err(ShardMapTraceError::MissingLinearizedInvocationContext),
        };
        let global_input_primals = Input::To::<ShardMapTracer>::from_parameters(
            input_structure.clone(),
            traced_inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>(),
        )?;
        let global_input_types = Input::To::<ArrayType>::from_parameters(
            input_structure,
            global_input_primals.parameters().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
        )?;
        let global_in_specs = reparameterize_shardings::<
            Input::To<Sharding>,
            <Input::To<ArrayType> as Parameterized<ArrayType>>::To<Sharding>,
        >(in_specs, global_input_types.parameter_structure())?;
        let traced = trace_flat_shard_map::<F, Input::To<ArrayType>, Output>(
            function,
            global_input_types,
            mesh,
            global_in_specs,
            out_specs,
            manual_axes,
            check_vma,
        )?;
        let staged_outputs = apply_linearized_flat_shard_map(tracing_builder, linear_builder, traced, traced_inputs)?;
        Ok(Output::To::<Linearized<ShardMapTracer>>::from_parameters(output_structure, staged_outputs)?)
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, fmt::Display, marker::PhantomData, rc::Rc, sync::Arc};

    use ryft_core::{
        parameters::Placeholder,
        sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding},
        tracing::{Atom, AtomId, InterpretableOperation, Operation, Program, ProgramBuilder, TracingError},
        tracing_v2::{
            CustomOperationError, CustomPrimitive, DifferentiableOperation, LinearPrimitiveOperation, LinearTerm,
            Tracer, forward::JvpTracer,
        },
        types::{ArrayType, DataType, Shape, Size, TypeError, Typed},
    };

    use crate::experimental::XlaEngine;
    use crate::experimental::ops::XlaPrimitiveOperation;
    use crate::experimental::shard_map::{
        FlatTracedShardMap, ShardMap, ShardMapTensor, ShardMapTraceError, ShardMapTracer,
    };

    use super::{
        ShardMapOperation, ShardMapReplayContext, build_factorized_apply_program, make_linear_tensor_shard_map,
        project_flat_shard_map_program, replay_traced_xla_program,
    };

    fn test_array_type() -> ArrayType {
        ArrayType::scalar(DataType::F32)
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

    fn vector_type(length: usize) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(length)]), None, None).unwrap()
    }

    fn simple_traced_shard_map_body() -> FlatTracedShardMap {
        let array_type = test_array_type();
        let mut builder = ProgramBuilder::<ArrayType, ShardMapTensor, XlaPrimitiveOperation>::new();
        let input = builder.add_input(array_type.clone());
        let output = builder
            .add_instruction(XlaPrimitiveOperation::Sin, vec![input])
            .expect("simple shard_map body should stage one sine op")
            .into_iter()
            .next()
            .expect("sine should produce one output");
        FlatTracedShardMap::from_parts(
            test_shard_map(),
            vec![array_type.clone()],
            vec![array_type.clone()],
            vec![array_type.clone()],
            vec![array_type],
            builder.build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            ),
        )
    }

    fn zero_input_traced_shard_map_body() -> FlatTracedShardMap {
        let array_type = test_array_type();
        let mut builder = ProgramBuilder::<ArrayType, ShardMapTensor, XlaPrimitiveOperation>::new();
        let output = builder.add_constant(ShardMapTensor::new(array_type.clone()));
        FlatTracedShardMap::from_parts(
            zero_input_test_shard_map(),
            Vec::new(),
            Vec::new(),
            vec![array_type.clone()],
            vec![array_type],
            builder.build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(
                vec![output],
                Vec::<Placeholder>::new(),
                vec![Placeholder],
            ),
        )
    }

    #[derive(Clone, Debug)]
    struct TestCustomReplayOp;

    impl Display for TestCustomReplayOp {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "test_custom_replay")
        }
    }

    impl Operation for TestCustomReplayOp {
        fn name(&self) -> &'static str {
            "test_custom_replay"
        }

        fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
            Ok(input_types.to_vec())
        }
    }

    impl InterpretableOperation<ArrayType, ShardMapTensor> for TestCustomReplayOp {
        fn interpret(&self, inputs: &[ShardMapTensor]) -> Result<Vec<ShardMapTensor>, TracingError> {
            Ok(inputs.to_vec())
        }
    }

    #[test]
    fn test_project_flat_shard_map_program_rejects_unmapped_variable_atom() {
        let array_type = test_array_type();
        let atom_id = AtomId { index: 0 };
        let program: Program<
            ArrayType,
            ShardMapTensor,
            XlaPrimitiveOperation,
            Vec<ShardMapTensor>,
            Vec<ShardMapTensor>,
        > = Program {
            atoms: vec![Atom::Variable(array_type)],
            input_ids: Vec::new(),
            output_ids: vec![atom_id],
            instructions: Vec::new(),
            input_structure: Vec::<Placeholder>::new(),
            output_structure: vec![Placeholder],
            marker: PhantomData,
        };

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
            Program {
                atoms: vec![Atom::Variable(array_type.clone()), Atom::Variable(array_type)],
                input_ids: vec![AtomId { index: 0 }],
                output_ids: vec![AtomId { index: 1 }],
                instructions: Vec::new(),
                input_structure: vec![Placeholder],
                output_structure: vec![Placeholder],
                marker: PhantomData,
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
            Program {
                atoms: vec![Atom::Variable(array_type)],
                input_ids: vec![atom_id],
                output_ids: vec![atom_id],
                instructions: Vec::new(),
                input_structure: vec![Placeholder],
                output_structure: vec![Placeholder],
                marker: PhantomData,
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
        let tangent_builder = Rc::new(RefCell::new(ProgramBuilder::<
            ArrayType,
            ShardMapTensor,
            LinearPrimitiveOperation<ArrayType, ShardMapTensor>,
        >::new()));
        let tangent_atom = tangent_builder.borrow_mut().add_input(test_array_type());
        let tangent = LinearTerm::from_staged_parts(tangent_atom, tangent_builder.clone());

        let outputs = operation
            .jvp(
                crate::experimental::engine::XlaEngine::token(),
                &[JvpTracer { primal: ShardMapTensor::new(test_array_type()), tangent }],
            )
            .expect("linear tensor shard_map jvp should succeed");

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal.r#type().into_owned(), test_array_type());

        let output_atoms = outputs.into_iter().map(|output| output.tangent.atom).collect::<Vec<_>>();
        let tangent_builder = Rc::try_unwrap(tangent_builder)
            .expect("traced shard_map jvp should not leak linear terms")
            .into_inner();
        let tangent_program = tangent_builder.build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(
            output_atoms,
            vec![Placeholder],
            vec![Placeholder],
        );
        assert!(
            tangent_program.to_string().contains("linear_shard_map"),
            "expected linear tensor shard_map jvp to stage a linear_shard_map op: {}",
            tangent_program
        );
    }

    #[test]
    fn test_replay_traced_xla_program_supports_reshape() {
        let input_type = vector_type(2);
        let output_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(1), Size::Static(2)]), None, None).unwrap();
        let program = {
            let mut builder = ProgramBuilder::<ArrayType, ShardMapTensor, XlaPrimitiveOperation>::new();
            let input = builder.add_input(input_type.clone());
            let output = builder
                .add_instruction(
                    XlaPrimitiveOperation::Reshape { input_type: input_type.clone(), output_type: output_type.clone() },
                    vec![input],
                )
                .expect("reshape should stage")
                .into_iter()
                .next()
                .expect("reshape should produce one output");
            builder.build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
        };
        let tracing_builder =
            Rc::new(RefCell::new(ProgramBuilder::<ArrayType, ShardMapTensor, XlaPrimitiveOperation>::new()));
        let input_atom = tracing_builder.borrow_mut().add_input(input_type);
        let input = Tracer::from_engine(input_atom, tracing_builder.clone(), XlaEngine::token());

        let outputs = replay_traced_xla_program(&ShardMapReplayContext { tracing_builder }, &program, vec![input])
            .expect("replay should support reshape inside shard_map bodies");

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].r#type().into_owned(), output_type);
    }

    #[test]
    fn test_traced_shard_map_interpret_with_explicit_builder_supports_zero_inputs() {
        let body = zero_input_traced_shard_map_body();
        let operation = ShardMapOperation::<ShardMapTracer>::new(body);
        let tracing_builder =
            Rc::new(RefCell::new(ProgramBuilder::<ArrayType, ShardMapTensor, XlaPrimitiveOperation>::new()));

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
        let staged_program = tracing_builder.build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(
            output_atoms,
            Vec::<Placeholder>::new(),
            vec![Placeholder],
        );

        assert_eq!(staged_program.instructions.len(), 1);
        assert!(matches!(staged_program.instructions[0].operation, XlaPrimitiveOperation::ShardMap(_)));
        assert!(staged_program.instructions[0].inputs.is_empty());
    }

    #[test]
    fn test_replay_traced_xla_program_missing_custom_replay_rule_reports_missing_rule() {
        let array_type = test_array_type();
        let program = {
            let mut builder = ProgramBuilder::<ArrayType, ShardMapTensor, XlaPrimitiveOperation>::new();
            let input = builder.add_input(array_type.clone());
            let output = builder
                .add_instruction(
                    XlaPrimitiveOperation::Custom(Arc::new(CustomPrimitive::new(TestCustomReplayOp))),
                    vec![input],
                )
                .expect("custom op should stage")
                .into_iter()
                .next()
                .expect("custom op should produce one output");
            builder.build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
        };
        let tracing_builder =
            Rc::new(RefCell::new(ProgramBuilder::<ArrayType, ShardMapTensor, XlaPrimitiveOperation>::new()));
        let input_atom = tracing_builder.borrow_mut().add_input(array_type);
        let input = Tracer::from_engine(input_atom, tracing_builder.clone(), XlaEngine::token());

        assert!(matches!(
            replay_traced_xla_program(&ShardMapReplayContext { tracing_builder }, &program, vec![input]),
            Err(ShardMapTraceError::TracingError(TracingError::CustomOperation(CustomOperationError::MissingRule {
                op: "test_custom_replay",
                transform: "shard_map replay",
            })))
        ));
    }
}
