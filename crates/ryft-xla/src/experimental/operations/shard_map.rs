use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use ryft_core::batching::{ArrayBatch, BatchableOperation, BatchingError};
use ryft_core::contexts::{Context, StagingContext};
use ryft_core::differentiation::TransposableOperation;
use ryft_core::effects::Effects;
use ryft_core::interpretation::InterpretableOperation;
use ryft_core::macros::check_count;
use ryft_core::materialize;
use ryft_core::operations::Operation;
use ryft_core::operations::constants::ZeroOperation;
use ryft_core::parameters::{Parameterized, ParameterizedFamily};
use ryft_core::partial::{
    PartialEvaluationInput, PartialEvaluationValue, PartialEvaluator, PartialValue, PartiallyEvaluatableOperation,
};
use ryft_core::programs::{MaybeZero, ProgramError, Value};
use ryft_core::sharding::{LogicalMesh, MeshAxisType, Sharding, ShardingDimension};
use ryft_core::tracing::{Tracer, TracingContext};

use ryft_core::tracing_v2::differentiation::{DifferentiableOperation, JvpTracer, Linearization};
use ryft_core::types::{ArrayType, TypeError, Typed};

use crate::experimental::ops::{XlaConstant, XlaOperation};
use crate::experimental::shard_map::{
    FlatTracedShardMap, ShardMap, ShardMapInvocationLeaf, ShardMapLocalTraceInput, ShardMapLocalTraceOutput,
    ShardMapTraceError, ShardMapTracer, TracedShardMap,
};

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

impl<C> InterpretableOperation<ShardMapTracer, C> for ShardMapOperation<ShardMapTracer> {
    /// Replays this traced-leaf shard-map op by staging it onto the trace its inputs already belong to: an input's
    /// [`context`](Tracer::context) shares that trace's program builder, so
    /// [`stage_operation`](StagingContext::stage_operation) on it appends to the same program. An empty-input body has
    /// no such input, so its outputs are inferred directly and no operation is staged.
    fn interpret(&self, _context: &C, inputs: &[ShardMapTracer]) -> Result<Vec<ShardMapTracer>, ProgramError> {
        match inputs.first() {
            Some(first) => {
                let abstract_inputs = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
                self.infer_output_types(abstract_inputs.as_slice())?;
                apply_flat_traced_shard_map(first, self.body.clone(), inputs).map_err(trace_error_from_shard_map)
            }
            None => {
                self.infer_output_types(&[])?;
                Ok(Vec::new())
            }
        }
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

impl<V: Value<Type = ArrayType>> Operation<ArrayType> for ShardMapOperation<V> {
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

    #[inline]
    fn effects(&self) -> Effects {
        self.body.program().effects()
    }
}

/// Batching rule for [`ShardMapOperation`]: batching through a staged `shard_map` boundary has no rule yet — the
/// mapped batch axis would need to compose with the boundary's global-to-local sharding on both sides — so batching
/// is rejected for every value and context.
impl<Constant, V, C> BatchableOperation<V, C> for ShardMapOperation<Constant>
where
    Constant: Value<Type = ArrayType>,
    V: Value<Type = ArrayType>,
{
    fn batch(&self, _context: &C, _inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, BatchingError> {
        Err(BatchingError::UnsupportedOperation {
            message: format!("missing batching rule for operation '{}'", self.name()),
        }
        .into())
    }
}

/// Online partial-evaluation rule for a staged `shard_map` — the map-boundary sibling of the
/// [`JitCallOperation`](crate::experimental::ops::JitCallOperation) call rule: it splits the local body against the
/// caller's known-ness while preserving the `shard_map` boundary, its mesh, and its shardings on both sides.
///
/// The split fires only when some known input does *not* [`resolve`](Context::resolve) to a concrete constant in
/// the known-side context — a genuine tracer into a live outer trace. All-known, all-unknown, and concrete-known
/// calls defer to the default fold-or-residualize behavior, which preserves the original boundary exactly.
///
/// When the split fires, the *local* body program is split through the shared
/// [`PartitionedProgram`](ryft_core::partial::PartitionedProgram) machinery. The known side is rewrapped as a
/// `shard_map` whose global outputs
/// are the fully known boundary outputs followed by the known→unknown residual edges (each edge's global type and
/// sharding derived through `residual_boundary`, exactly as in the structural split), bound into the enclosing
/// known-side context over the original known boundary inputs. The residual side is rewrapped as a `shard_map` over
/// the surviving unknown boundary inputs plus those residual edges and emitted into the residual program.
impl<V, C> PartiallyEvaluatableOperation<C> for ShardMapOperation<V>
where
    V: Value<Type = ArrayType>,
    C: Context<Type = ArrayType, Operation = XlaOperation<V>>,
    ShardMapOperation<V>: Operation<ArrayType>,
{
    fn partially_evaluate(
        &self,
        evaluator: &mut PartialEvaluator<C>,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        // Split only a mixed boundary with at least one known-but-symbolic input; everything else keeps the default
        // fold-or-residualize behavior and therefore the original boundary.
        if !evaluator.any_known_is_symbolic(inputs) || inputs.iter().all(PartialEvaluationValue::is_known) {
            return evaluator.fold_or_residualize(XlaOperation::ShardMap(Box::new(self.clone())), inputs);
        }

        // Split the local body through the shared online boundary machinery. The body's inputs are index-aligned
        // with the boundary inputs and carry the *local* types, so both split sides stay local programs.
        let input_known = inputs.iter().map(PartialEvaluationValue::is_known).collect::<Vec<bool>>();
        let partition = self.body.program().partition(input_known.as_slice())?;
        // A trivial partition — one whose known program contains no instructions — hoists no work (its known side
        // can only forward known inputs as residual edges), so keep the original boundary and let the default
        // materialize those knowns directly as residual feeders.
        if partition.known_program().instructions().is_empty() {
            return evaluator.fold_or_residualize(XlaOperation::ShardMap(Box::new(self.clone())), inputs);
        }

        // Derive each residual edge's global boundary type and sharding from its local type.
        let mesh = self.body.shard_map().mesh();
        let residual_edge_boundaries = partition
            .residual_inputs()
            .iter()
            .zip(partition.residual_program().input_types())
            .filter_map(|(source, edge_type)| source.is_known().then_some(edge_type))
            .map(|edge_type| residual_boundary(&edge_type, mesh).map_err(trace_error_from_shard_map))
            .collect::<Result<Vec<_>, _>>()?;

        // Gather the known-side boundary metadata: shardings and global types per original index, with the residual
        // edges appended.
        let known_global_input_types = partition
            .known_input_indices()
            .iter()
            .map(|&index| self.body.global_input_types()[index].clone())
            .collect::<Vec<_>>();
        let known_in_shardings = partition
            .known_input_indices()
            .iter()
            .map(|&index| self.body.shard_map().in_shardings()[index].clone())
            .collect::<Vec<_>>();
        let known_output_indices = partition
            .outputs()
            .iter()
            .enumerate()
            .filter_map(|(index, output)| output.is_known().then_some(index))
            .collect::<Vec<_>>();
        let mut known_global_output_types = known_output_indices
            .iter()
            .map(|&index| self.body.global_output_types()[index].clone())
            .collect::<Vec<_>>();
        let mut known_out_shardings = known_output_indices
            .iter()
            .map(|&index| self.body.shard_map().out_shardings()[index].clone())
            .collect::<Vec<_>>();
        for (global_type, sharding) in residual_edge_boundaries.iter() {
            known_global_output_types.push(global_type.clone());
            known_out_shardings.push(sharding.clone());
        }

        // Gather the staged-side boundary metadata: the surviving unknown boundary inputs plus the residual edges,
        // and the residual-owned outputs.
        let mut staged_global_input_types = Vec::with_capacity(partition.residual_inputs().len());
        let mut staged_in_shardings = Vec::with_capacity(partition.residual_inputs().len());
        for source in partition.residual_inputs().iter() {
            match source {
                PartialEvaluationInput::Unknown(index) => {
                    staged_global_input_types.push(self.body.global_input_types()[*index].clone());
                    staged_in_shardings.push(self.body.shard_map().in_shardings()[*index].clone());
                }
                PartialEvaluationInput::Known(edge) => {
                    let (global_type, sharding) = &residual_edge_boundaries[*edge];
                    staged_global_input_types.push(global_type.clone());
                    staged_in_shardings.push(sharding.clone());
                }
            }
        }
        let mut staged_global_output_types = Vec::new();
        let mut staged_out_shardings = Vec::new();
        for (index, output) in partition.outputs().iter().enumerate() {
            if output.is_unknown() {
                staged_global_output_types.push(self.body.global_output_types()[index].clone());
                staged_out_shardings.push(self.body.shard_map().out_shardings()[index].clone());
            }
        }

        // Bind the known-side `shard_map` into the enclosing known-side evaluator, emit the residual `shard_map`
        // over the surviving unknown boundary inputs plus the residual edges, and reassemble the original outputs.
        evaluator.inline_partitioned_program(
            partition,
            inputs,
            |known_program| {
                let known_local_input_types = known_program.input_types();
                let known_local_output_types = known_program.output_types();
                let known_shard_map = ShardMap::from_shardings(
                    mesh.clone(),
                    known_in_shardings,
                    known_out_shardings,
                    self.body.shard_map().manual_axes().to_vec(),
                    self.body.shard_map().check_vma(),
                );
                let known_body = FlatTracedShardMap::from_parts(
                    known_shard_map,
                    known_global_input_types,
                    known_local_input_types,
                    known_global_output_types,
                    known_local_output_types,
                    known_program,
                );
                XlaOperation::ShardMap(Box::new(ShardMapOperation::new(known_body)))
            },
            |residual_program| {
                let staged_local_input_types = residual_program.input_types();
                let staged_local_output_types = residual_program.output_types();
                let staged_shard_map = ShardMap::from_shardings(
                    mesh.clone(),
                    staged_in_shardings,
                    staged_out_shardings,
                    self.body.shard_map().manual_axes().to_vec(),
                    self.body.shard_map().check_vma(),
                );
                let staged_body = FlatTracedShardMap::from_parts(
                    staged_shard_map,
                    staged_global_input_types,
                    staged_local_input_types,
                    staged_global_output_types,
                    staged_local_output_types,
                    residual_program,
                );
                XlaOperation::ShardMap(Box::new(ShardMapOperation::new(staged_body)))
            },
        )
    }
}

/// Returns a concrete atom for `cotangent`, staging a typed `Zero` op when the cotangent is
/// structurally zero. Higher-order linear rules use this when they must consume all output
/// cotangents jointly.
fn materialize_cotangent<V: Value<Type = ArrayType>, O>(
    context: &TracingContext<V, O>,
    cotangent: &MaybeZero<Tracer<TracingContext<V, O>>>,
    output_type: &ArrayType,
) -> Tracer<TracingContext<V, O>>
where
    O: Operation<ArrayType> + From<ZeroOperation<ArrayType>>,
{
    match cotangent {
        MaybeZero::Value(cotangent) => return cotangent.clone(),
        MaybeZero::Zero(_) => {}
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

/// Builds the residual boundary sharding and matching global type for one residual edge.
///
/// Residuals are arbitrary shard-local body intermediates threaded across the shard-map boundary as plain operand
/// edges from the primal body to the tangent body (see [`shard_map_bodies`]). Each is carried fully replicated
/// over the manual axes, so its global shape equals its local shape and
/// [`local_input_shape`](ShardMap::local_input_shape) round-trips the value unchanged on both the primal-output side and
/// the tangent-input side.
///
/// The returned [`Sharding`] is the fully-replicated boundary sharding rendered into the `sdy.manual_computation`
/// attributes, while the returned global [`ArrayType`] deliberately carries *no* sharding: re-embedding the primal
/// body's outputs into the caller's ambient envelope (see [`adapt_traced_shard_map_output_type`]) erases the sharding
/// from the threaded value, so the tangent body's declared residual input must match the unsharded value the primal
/// `shard_map` actually produces rather than the explicit replicated annotation.
///
/// # Parameters
///
///   - `local_type`: Shard-local residual type produced by the primal sub-program.
///   - `mesh`: Logical mesh the shard-map body is defined over.
fn residual_boundary(local_type: &ArrayType, mesh: &LogicalMesh) -> Result<(ArrayType, Sharding), ShardMapTraceError> {
    let dimensions = vec![ShardingDimension::Replicated; local_type.shape().rank()];
    let sharding = Sharding::new(mesh.clone(), dimensions)?;
    let global_type =
        ArrayType::new(local_type.data_type(), local_type.shape().clone()).with_layout(local_type.layout().cloned());
    Ok((global_type, sharding))
}

/// Fuse-linearizes a shard-map body capture-free into a primal body and a tangent body that thread residuals as plain
/// operand edges across the shard-map boundary.
///
/// The body's flat program is fuse-linearized once through [`Program::linearize`](ryft_core::Program::linearize),
/// yielding a primal sub-program
/// `local_inputs -> [local_outputs..., local_residuals...]` and a tangent sub-program
/// `[local_input_tangents..., local_residuals...] -> [local_output_tangents...]` together with the residual count. Each
/// sub-program is re-wrapped as its own [`FlatTracedShardMap`]: the primal body's boundary gains the residual edges as
/// trailing outputs, the tangent body's boundary gains them as trailing inputs, and both use the replicated residual
/// boundary from [`residual_boundary`]. This is the shard-map counterpart of the jitted-call rule, realizing
/// `jvp(shard_map(f)) = shard_map(jvp f)` with no symbolic capture ever introduced.
///
/// # Parameters
///
///   - `body`: Erased primal shard-map body to linearize.
fn shard_map_bodies(
    body: &FlatTracedShardMap,
) -> Result<(FlatTracedShardMap, FlatTracedShardMap, usize), ShardMapTraceError> {
    let output_count = body.global_output_types().len();
    let Linearization { primal_program, tangent_program, residual_count, .. } = body.program().linearize()?;

    let mesh = body.shard_map().mesh();
    let manual_axes = body.shard_map().manual_axes();

    // The primal sub-program's trailing outputs beyond the original outputs are the residual edges; their local
    // types are authoritative and back the residual boundary on both bodies.
    let primal_local_output_types = primal_program.output_types();
    let residual_local_types = &primal_local_output_types[output_count..];
    let mut residual_global_types = Vec::with_capacity(residual_count);
    let mut residual_shardings = Vec::with_capacity(residual_count);
    for residual_local_type in residual_local_types {
        let (residual_global_type, residual_sharding) = residual_boundary(residual_local_type, mesh)?;
        residual_global_types.push(residual_global_type);
        residual_shardings.push(residual_sharding);
    }

    let primal_body = {
        let shard_map = ShardMap::from_shardings(
            mesh.clone(),
            body.shard_map().in_shardings().to_vec(),
            body.shard_map().out_shardings().iter().cloned().chain(residual_shardings.iter().cloned()).collect(),
            manual_axes.to_vec(),
            body.shard_map().check_vma(),
        );
        let global_output_types =
            body.global_output_types().iter().cloned().chain(residual_global_types.iter().cloned()).collect();
        FlatTracedShardMap::from_parts(
            shard_map,
            body.global_input_types().to_vec(),
            primal_program.input_types(),
            global_output_types,
            primal_local_output_types,
            primal_program,
        )
    };

    let tangent_body = {
        let shard_map = ShardMap::from_shardings(
            mesh.clone(),
            body.shard_map().in_shardings().iter().cloned().chain(residual_shardings).collect(),
            body.shard_map().out_shardings().to_vec(),
            manual_axes.to_vec(),
            body.shard_map().check_vma(),
        );
        let global_input_types = body.global_input_types().iter().cloned().chain(residual_global_types).collect();
        let tangent_local_input_types = tangent_program.input_types();
        let tangent_local_output_types = tangent_program.output_types();
        FlatTracedShardMap::from_parts(
            shard_map,
            global_input_types,
            tangent_local_input_types,
            body.global_output_types().to_vec(),
            tangent_local_output_types,
            tangent_program,
        )
    };

    Ok((primal_body, tangent_body, residual_count))
}

/// Capture-free forward-mode (JVP) rule for [`ShardMapOperation`], staging a primal `shard_map` and a tangent
/// `shard_map` as ordinary [`XlaOperation`]s over the shared builder.
///
/// This realizes the identity `jvp(shard_map(f)) = shard_map(jvp f)`: rather than capturing the global primals as
/// residual factors and staging a linear `shard_map`, the rule keeps the
/// manual region intact and threads every residual as a plain primal operand edge between two `shard_map`s, so no
/// symbolic capture is ever introduced. The enclosing partial-evaluation split then discovers the residual operand
/// edges structurally, exactly as for the jitted-call rule.
///
/// The rule linearizes the body capture-free through `shard_map_bodies`, giving a primal body
/// `inputs -> [outputs..., residuals...]` and a tangent body `[input_tangents..., residuals...] -> [output_tangents...]`
/// together with the residual count. It then stages the primal `shard_map` over the operand primals (recovering the
/// primal outputs followed by the residual values), stages the tangent `shard_map` over the operand tangents followed
/// by those residual values (recovering one output tangent per primal output), and pairs each primal output with its
/// tangent output. The body program is keyed on [`XlaConstant`] regardless of the enclosing value type `V`, so the
/// sub-programs are valid for every `V`.
impl<C, V> DifferentiableOperation<C> for ShardMapOperation<V>
where
    C: StagingContext<Type = ArrayType, Constant = V, Operation = XlaOperation<V>>,
    V: Value<Type = ArrayType>,
{
    fn jvp(&self, context: &C, inputs: &[JvpTracer<C>]) -> Result<Vec<JvpTracer<C>>, ProgramError> {
        let output_count = self.output_types.len();
        check_count!("input", inputs, self.input_types.len(), ProgramError);

        let (primal_body, tangent_body, _residual_count) =
            shard_map_bodies(&self.body).map_err(trace_error_from_shard_map)?;

        // Stage the primal `shard_map`, recovering the primal outputs followed by the residual values.
        let primal_operands = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal_operation = XlaOperation::ShardMap(Box::new(ShardMapOperation::new(primal_body)));
        let mut primal_outputs = context.stage_operation(primal_operation, primal_operands.as_slice())?;
        if primal_outputs.len() < output_count {
            return Err(ProgramError::MalformedProgram(format!(
                "shard_map primal body produced {} outputs which is fewer than its {output_count} primal \
                 output(s)",
                primal_outputs.len(),
            )));
        }
        let residuals = primal_outputs.split_off(output_count);

        // Forward-mode requires each output tangent to carry the same type as its primal output, including the ambient
        // sharding envelope the primal `shard_map` re-embeds its outputs into (see `adapt_traced_shard_map_output_type`).
        // The primal `shard_map` adapts because it has a single leading operand, while the tangent `shard_map` carries
        // trailing residual operands and so skips that adaptation; re-embed the tangent body's global output boundary
        // into the staged primal output types directly to keep the two output signatures aligned.
        let primal_output_types = primal_outputs.iter().map(|output| output.r#type().into_owned()).collect::<Vec<_>>();
        let tangent_body =
            tangent_body.with_global_output_types(primal_output_types).map_err(trace_error_from_shard_map)?;

        // Stage the tangent `shard_map` over the operand tangents followed by the residual values, recovering one
        // output tangent per primal output.
        // The tangent `shard_map` takes every operand tangent as a real program input, so materialize structural
        // zeros at this sub-program boundary.
        let mut tangent_operands = inputs
            .iter()
            .map(|input| materialize(context, input.tangent().clone()))
            .collect::<Result<Vec<_>, _>>()?;
        tangent_operands.extend(residuals);
        let tangent_operation = XlaOperation::ShardMap(Box::new(ShardMapOperation::new(tangent_body)));
        let tangent_outputs = context.stage_operation(tangent_operation, tangent_operands.as_slice())?;
        check_count!("output", tangent_outputs, output_count, ProgramError);

        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| JvpTracer::new(primal, tangent))
            .collect())
    }
}

/// Partition-aware transpose rule for a *primal* tangent [`ShardMapOperation`], the shard-map counterpart of
/// [`transpose_primal_jit_call`](crate::experimental::ops::transpose_primal_jit_call).
///
/// The forward ([`ShardMapOperation::jvp`]) stages the tangent `shard_map` over the operand tangents
/// followed by the primal body's residual values, wrapping a body whose global inputs match that operand signature
/// one-to-one and whose global outputs are the output tangents. Each operand is therefore independently linear (an input
/// tangent the reverse must accumulate) or known (a residual value); the linear operands need not lead. This rule:
///
///   1. Reads the runtime value of every known operand from `operand_values`, in body-input order, to feed the
///      transposed body's known inputs.
///   2. Transposes the tangent body's flat program with
///      [`transpose_with_respect_to`](ryft_core::Program::transpose_with_respect_to) with respect to the same
///      linear operands,
///      so the transposed body maps `[outputs..., known_input_values...]` to `[linear_input_cotangents...]`,
///      in body-input order on each side. It re-wraps the transposed program in a fresh [`FlatTracedShardMap`] whose
///      boundary shardings are permuted to match — its inputs carry the original output shardings (for the cotangents)
///      followed by the known operands' input shardings, and its outputs carry the linear operands' input shardings.
///   3. Re-wraps the transposed body in a fresh [`ShardMapOperation`] and stages it over
///      `[outputs..., known_input_values...]`, keeping the manual region intact so both forward and reverse
///      mode over a `shard_map` stay manual rather than inlined.
///
/// The returned cotangents place the transposed `shard_map`'s outputs at the linear-operand positions and a structural
/// [`MaybeZero::Zero`] at the known positions. The body transposition happens through
/// [`transpose_with_respect_to`](ryft_core::Program::transpose_with_respect_to) in this same operation family, so it is
/// value-level and introduces no recursive transposition obligation on [`XlaOperation`].
///
/// # Parameters
///
///   - `operation`: Primal tangent `shard_map` staged into the tangent program.
///   - `context`: Active transpose tracing context the pullback is staged into.
///   - `inputs`: Per-operand [`PartialValue`] knowledge, mirroring the body's global inputs one-to-one. The
///     [`Unknown`](PartialValue::Unknown) entries are the input tangents; the [`Known`](PartialValue::Known) entries
///     carry the residual tracers the pullback reads.
///   - `outputs`: Symbolic cotangents for the tangent `shard_map`'s outputs.
pub fn transpose_primal_shard_map<V: Value<Type = ArrayType>>(
    operation: &ShardMapOperation<V>,
    context: &mut TracingContext<V, XlaOperation<V>>,
    inputs: &[PartialValue<Tracer<TracingContext<V, XlaOperation<V>>>>],
    outputs: &[MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>],
) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>>, ProgramError> {
    let operand_linear = inputs.iter().map(PartialValue::is_unknown).collect::<Vec<_>>();
    let body = operation.body();
    check_count!("input", operand_linear, body.global_input_types().len(), ProgramError);

    // A shard_map with no live output cotangents is a zero linear map, so every operand cotangent is zero.
    if outputs.iter().all(MaybeZero::is_zero) {
        return Ok(inputs.iter().map(|input| MaybeZero::Zero(input.r#type().into_owned())).collect());
    }

    // Each operand maps to one body global input, independently linear (an input tangent) or known (a residual value).
    // The dispatch guarantees a `Known` operand carries its pullback value, so each known tracer is read directly in
    // body-input order.
    let known_values = inputs
        .iter()
        .filter(|input| input.is_known())
        .map(|input| input.as_known().expect("dispatch guarantees a known operand carries its pullback value").clone())
        .collect::<Vec<_>>();

    // Transpose the tangent body's flat program under the same per-operand linearity mask. The transposed program maps
    // `[outputs..., known_input_values...]` to `[linear_input_cotangents...]`, in body-input order on each
    // side; re-wrap it as a transposed shard-map body whose boundary shardings are permuted to match.
    let transposed_body = transpose_shard_map_body(body, operand_linear.as_slice())?;

    // Stage the output cotangents, materializing a typed zero for each structurally zero cotangent, then stage a fresh
    // `shard_map` over the transposed body on `[outputs..., known_input_values...]`. Its outputs are the
    // linear-input cotangents.
    let output_types = &operation.output_types;
    check_count!("output", outputs, output_types.len(), ProgramError);
    let mut operands = Vec::with_capacity(output_types.len() + known_values.len());
    for (cotangent, output_type) in outputs.iter().zip(output_types.iter()) {
        operands.push(materialize_cotangent(context, cotangent, output_type));
    }
    operands.extend(known_values);
    let transposed_operation = XlaOperation::ShardMap(Box::new(ShardMapOperation::new(transposed_body)));
    let input_cotangents = context.stage_operation(transposed_operation, operands.as_slice())?;
    let linear_count = operand_linear.iter().filter(|&&linear| linear).count();
    check_count!("output", input_cotangents, linear_count, ProgramError);

    // Reassemble one cotangent per operand: the known operands carry structural zeros, while the linear input tangents
    // receive the transposed `shard_map`'s outputs in body-input order.
    let mut input_cotangents = input_cotangents.into_iter().map(MaybeZero::Value);
    let cotangents = operand_linear
        .iter()
        .zip(inputs)
        .map(
            |(&linear, input)| {
                if linear { input_cotangents.next().unwrap() } else { MaybeZero::Zero(input.r#type().into_owned()) }
            },
        )
        .collect();
    Ok(cotangents)
}

/// Transpose rule for a primal tangent [`ShardMapOperation`], forwarding to [`transpose_primal_shard_map`]. The body
/// transposition happens on the body's flat [`XlaConstant`]-keyed program, so the recursion is resolved once at
/// definition time and instantiating this implementation introduces no recursive
/// [`TransposableOperation`] obligation on [`XlaOperation`].
impl<V: Value<Type = ArrayType>> TransposableOperation<V, XlaOperation<V>> for ShardMapOperation<V> {
    fn transpose(
        &self,
        context: &mut TracingContext<V, XlaOperation<V>>,
        inputs: &[PartialValue<Tracer<TracingContext<V, XlaOperation<V>>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>>, ProgramError> {
        transpose_primal_shard_map(self, context, inputs, outputs)
    }
}

/// Transposes one tangent shard-map body into the reverse body consumed by [`transpose_primal_shard_map`].
///
/// The tangent body's flat program is transposed with [`transpose_with_respect_to`](ryft_core::Program::transpose_with_respect_to)
/// under `input_linearity`, producing a program mapping `[outputs..., known_input_values...]` to
/// `[linear_input_cotangents...]`. The transposed [`FlatTracedShardMap`] permutes the original boundary to match: its
/// global inputs are the original global outputs (for the output cotangents) followed by the known operands' original
/// global inputs, and its global outputs are the linear operands' original global inputs. Local types come straight
/// from the transposed program, which is authoritative for the manual region.
///
/// # Parameters
///
///   - `body`: Tangent shard-map body produced by [`shard_map_bodies`], whose global inputs are
///     `[input_tangents..., residuals...]` and whose global outputs are `[output_tangents...]`.
///   - `input_linearity`: Per-input linearity flags over the tangent body's global inputs.
fn transpose_shard_map_body(
    body: &FlatTracedShardMap,
    input_linearity: &[bool],
) -> Result<FlatTracedShardMap, ProgramError> {
    let with_respect_to = input_linearity
        .iter()
        .enumerate()
        .filter_map(|(index, &linear)| linear.then_some(index))
        .collect::<Vec<_>>();
    let transposed_program = body.program().transpose_with_respect_to(with_respect_to.as_slice())?;

    let in_shardings = body
        .shard_map()
        .out_shardings()
        .iter()
        .cloned()
        .chain(
            input_linearity
                .iter()
                .zip(body.shard_map().in_shardings().iter())
                .filter(|&(&linear, _)| !linear)
                .map(|(_, sharding)| sharding.clone()),
        )
        .collect::<Vec<_>>();
    let out_shardings = input_linearity
        .iter()
        .zip(body.shard_map().in_shardings().iter())
        .filter(|&(&linear, _)| linear)
        .map(|(_, sharding)| sharding.clone())
        .collect::<Vec<_>>();
    let shard_map = ShardMap::from_shardings(
        body.shard_map().mesh().clone(),
        in_shardings,
        out_shardings,
        body.shard_map().manual_axes().to_vec(),
        body.shard_map().check_vma(),
    );

    let global_input_types = body
        .global_output_types()
        .iter()
        .cloned()
        .chain(
            input_linearity
                .iter()
                .zip(body.global_input_types().iter())
                .filter(|&(&linear, _)| !linear)
                .map(|(_, global_type)| global_type.clone()),
        )
        .collect::<Vec<_>>();
    let global_output_types = input_linearity
        .iter()
        .zip(body.global_input_types().iter())
        .filter(|&(&linear, _)| linear)
        .map(|(_, global_type)| global_type.clone())
        .collect::<Vec<_>>();

    Ok(FlatTracedShardMap::from_parts(
        shard_map,
        global_input_types,
        transposed_program.input_types(),
        global_output_types,
        transposed_program.output_types(),
        transposed_program,
    ))
}

fn trace_error_from_shard_map(error: ShardMapTraceError) -> ProgramError {
    ProgramError::Type(TypeError { message: error.to_string() })
}

fn apply_flat_traced_shard_map(
    first: &ShardMapTracer,
    body: FlatTracedShardMap,
    traced_inputs: &[ShardMapTracer],
) -> Result<Vec<ShardMapTracer>, ShardMapTraceError> {
    first
        .context()
        .stage_operation(XlaOperation::ShardMap(Box::new(ShardMapOperation::new(body.clone()))), traced_inputs)
        .map_err(ShardMapTraceError::from)
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
    C: StagingContext<Type = ArrayType, Operation = XlaOperation>,
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
    use ryft_core::contexts::StagingContext;
    use ryft_core::interpretation::InterpretableOperation;
    use ryft_core::operations::arithmetic::{AddOperation, MulOperation};
    use ryft_core::parameters::Placeholder;
    use ryft_core::partial::PartialValue;
    use ryft_core::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding};
    use ryft_core::tracing::DomainTracingContext;
    use ryft_core::types::{ArrayType, DataType, Typed};

    use crate::experimental::domains::XlaDomain;
    use crate::experimental::ops::{XlaConstant, XlaOperation, XlaProgramBuilder};
    use crate::experimental::shard_map::{FlatTracedShardMap, ShardMap, ShardMapTracer};

    use super::ShardMapOperation;

    fn test_array_type() -> ArrayType {
        ArrayType::scalar(DataType::F32)
    }

    fn single_input_test_shard_map() -> ShardMap {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        ShardMap::from_shardings(
            mesh.clone(),
            vec![Sharding::replicated(mesh.clone(), 0)],
            vec![Sharding::replicated(mesh, 0)],
            vec!["x".to_string()],
            true,
        )
    }

    /// Builds a one-input, one-output identity shard-map body whose global boundary carries no sharding, so its
    /// boundary types match any same-shape input fed to it through [`ShardMapOperation::interpret`].
    fn single_input_traced_shard_map_body() -> FlatTracedShardMap {
        let array_type = test_array_type();
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(array_type.clone());
        FlatTracedShardMap::from_parts(
            single_input_test_shard_map(),
            vec![array_type.clone()],
            vec![array_type.clone()],
            vec![array_type.clone()],
            vec![array_type.clone()],
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![input], vec![Placeholder], vec![Placeholder])
                .unwrap(),
        )
    }

    fn mixed_known_unknown_traced_shard_map_body() -> FlatTracedShardMap {
        let array_type = test_array_type();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let shard_map = ShardMap::from_shardings(
            mesh,
            vec![replicated.clone(), replicated.clone()],
            vec![replicated.clone(), replicated.clone(), replicated],
            vec!["x".to_string()],
            true,
        );

        let mut builder = XlaProgramBuilder::new();
        let known_input = builder.add_input(array_type.clone());
        let runtime_input = builder.add_input(array_type.clone());
        let doubled = builder.add_instruction(AddOperation, vec![known_input, known_input]).unwrap()[0];
        let product = builder.add_instruction(MulOperation, vec![known_input, runtime_input]).unwrap()[0];
        let sum = builder.add_instruction(AddOperation, vec![runtime_input, known_input]).unwrap()[0];
        FlatTracedShardMap::from_parts(
            shard_map,
            vec![array_type.clone(), array_type.clone()],
            vec![array_type.clone(), array_type.clone()],
            vec![array_type.clone(), array_type.clone(), array_type.clone()],
            vec![array_type.clone(), array_type.clone(), array_type],
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![doubled, product, sum],
                    vec![Placeholder; 2],
                    vec![Placeholder; 3],
                )
                .unwrap(),
        )
    }

    #[test]
    fn test_traced_shard_map_interpret_composes_staging_onto_input_trace() {
        let body = single_input_traced_shard_map_body();
        let operation = ShardMapOperation::<ShardMapTracer>::new(body);

        // The input tracer already belongs to a trace; `interpret` must compose the shard_map staging onto that same
        // trace via the input's own context rather than originating a fresh builder.
        let context = DomainTracingContext::<XlaDomain<'static>>::new();
        let input = context.input(test_array_type());

        let outputs = operation
            .interpret(&context, std::slice::from_ref(&input))
            .expect("traced shard_map staging should compose onto the input tracer's trace");

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].r#type().into_owned(), test_array_type());

        let builder = context.builder().borrow();
        assert_eq!(builder.instructions().len(), 1);
        assert!(matches!(builder.instructions()[0].operation(), XlaOperation::ShardMap(_)));
        assert_eq!(builder.instructions()[0].inputs(), &[input.atom_id().unwrap()]);
    }

    /// Online partial evaluation of a mixed `shard_map` against a live outer trace: the known half of the local body
    /// is rewrapped as a known-side `shard_map` staged into the outer program over the symbolic known input, the
    /// unknown half stays behind a residual `shard_map`, the known→unknown residual edges flow between them, and the
    /// mesh and shardings are threaded onto both boundaries.
    #[test]
    fn test_shard_map_online_partial_evaluation_splits_body_against_a_live_outer_trace() {
        use ryft_core::partial::{PartialEvaluationInput, PartialEvaluationOutput};
        use ryft_core::programs::ProgramBuilder;
        use ryft_core::tracing::TracingContext;

        let array_type = test_array_type();
        let operation = ShardMapOperation::<XlaConstant>::new(mixed_known_unknown_traced_shard_map_body());

        // Enclosing program staging one `shard_map` over `[a, x]`.
        let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let known_input = builder.add_input(array_type.clone());
        let runtime_input = builder.add_input(array_type.clone());
        let outputs = builder
            .add_instruction(XlaOperation::ShardMap(Box::new(operation)), vec![known_input, runtime_input])
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 3])
            .unwrap();

        let outer = TracingContext::<XlaConstant, XlaOperation>::new();
        let known = outer.input(array_type.clone());
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(known), PartialValue::Unknown(array_type.clone())],
            )
            .unwrap();

        // The known half landed in the outer program as one known-side `shard_map` over the symbolic known input,
        // producing the fully known boundary output (`a + a`) plus the residual edge (`a` itself), with the residual
        // edge's boundary derived from its local type.
        {
            let outer_builder = outer.builder().borrow();
            assert_eq!(outer_builder.instructions().len(), 1);
            let XlaOperation::ShardMap(known_side) = outer_builder.instructions()[0].operation() else {
                panic!("expected the outer program to contain the known-side shard_map");
            };
            let known_body = known_side.body();
            assert_eq!(known_body.program().input_ids().len(), 1);
            assert_eq!(known_body.program().output_ids().len(), 2);
            assert_eq!(known_body.program().instructions().len(), 1);
            assert_eq!(known_body.global_output_types().len(), 2);
            assert_eq!(known_body.shard_map().in_shardings().len(), 1);
            assert_eq!(known_body.shard_map().out_shardings().len(), 2);
        }

        // The unknown half stayed behind one residual `shard_map` over the unknown boundary input plus the residual
        // edge, with the shardings threaded per input.
        assert_eq!(evaluation.program().instructions().len(), 1);
        let XlaOperation::ShardMap(residual_side) = evaluation.program().instructions()[0].operation() else {
            panic!("expected the residual program to contain the residual shard_map");
        };
        let residual_body = residual_side.body();
        assert_eq!(residual_body.program().input_ids().len(), 2);
        assert_eq!(residual_body.program().instructions().len(), 2);
        assert_eq!(residual_body.global_input_types().len(), 2);
        assert_eq!(residual_body.shard_map().in_shardings().len(), 2);
        assert_eq!(residual_body.shard_map().out_shardings().len(), 2);
        assert_eq!(residual_body.global_output_types().len(), 2);

        // The boundary descriptors: the unknown enclosing input feeds the residual side, the residual edge is a
        // known feeder naming the known-side call's staged output, and the outputs reassemble in original order.
        assert_eq!(evaluation.inputs().len(), 2);
        assert!(matches!(&evaluation.inputs()[0], PartialEvaluationInput::Unknown(1)));
        assert!(matches!(&evaluation.inputs()[1], PartialEvaluationInput::Known(value) if value.atom_id().is_ok()));
        assert_eq!(evaluation.outputs().len(), 3);
        assert!(matches!(&evaluation.outputs()[0], PartialEvaluationOutput::Known(value) if value.atom_id().is_ok()));
        assert!(matches!(&evaluation.outputs()[1], PartialEvaluationOutput::Unknown(0)));
        assert!(matches!(&evaluation.outputs()[2], PartialEvaluationOutput::Unknown(1)));
    }
}
