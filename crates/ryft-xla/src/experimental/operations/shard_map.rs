use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use ryft_core::contexts::{Context, StagingContext};
use ryft_core::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationError, TransposableOperation,
    TranspositionDriver,
};
use ryft_core::macros::check_count;
use ryft_core::operations::constants::Zero;
use ryft_core::parameters::{Parameterized, ParameterizedFamily};
use ryft_core::partial::{
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationInput, PartialEvaluationValue, PartialValue,
    PartiallyEvaluatableOperation,
};
use ryft_core::programs::operations::Operation;
use ryft_core::programs::regions::{RegionInterface, RegionRef, RegionSlot};
use ryft_core::programs::{Concretizable, MaybeZero, Program, ProgramError, ProjectedValue, Value, ValueProjection};
use ryft_core::sharding::{LogicalMesh, MeshAxisType, Sharding, ShardingDimension};
use ryft_core::tracing::{Tracer, TracingContext};

use ryft_core::differentiation::DifferentiationDual;
use ryft_core::programs::types::{Type, TypeError, Typed};
use ryft_core::types::{ArrayProgramType, ArrayType};

use crate::experimental::ops::{XlaConstant, XlaOperation, XlaProgram, materialize_transpose_cotangent};
use crate::experimental::shard_map::{
    FlatTracedShardMap, ShardMap, ShardMapInvocationLeaf, ShardMapLocalTraceInput, ShardMapLocalTraceOutput,
    ShardMapTraceError, ShardMapTracer, TracedShardMap,
};

/// Canonical higher-order shard-map op used for staged tracing, differentiation, and lowering. The local body
/// program is not part of this payload: it is the operation's one attached `body` region, so this payload carries
/// only the manual SPMD boundary metadata that the region program cannot represent.
#[derive(Clone, Debug)]
pub struct ShardMapOperation<V> {
    /// Manual SPMD metadata (mesh, boundary shardings, and manual axes) governing the attached body region.
    shard_map: ShardMap,

    /// Global input types declared at the shard-map boundary.
    input_types: Vec<ArrayType>,

    /// Global output types declared at the shard-map boundary.
    output_types: Vec<ArrayType>,

    /// Phantom marker tying the operation to the traced leaf type it will replay with.
    marker: PhantomData<fn() -> V>,
}

impl<V> ShardMapOperation<V> {
    /// Splits the provided erased shard-map body into a metadata-only operation and the local body program that the
    /// caller attaches as the operation's `body` region (or interns as a shared callee).
    #[inline]
    pub(crate) fn from_body(body: FlatTracedShardMap) -> (Self, XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>) {
        let (shard_map, input_types, output_types, program) = body.into_operation_parts();
        (Self { shard_map, input_types, output_types, marker: PhantomData }, program)
    }

    /// Returns the manual SPMD metadata governing the attached body region.
    #[inline]
    pub(crate) fn shard_map(&self) -> &ShardMap {
        &self.shard_map
    }

    /// Returns the global input types declared at the shard-map boundary.
    #[inline]
    pub(crate) fn global_input_types(&self) -> &[ArrayType] {
        &self.input_types
    }

    /// Returns the global output types declared at the shard-map boundary.
    #[inline]
    pub(crate) fn global_output_types(&self) -> &[ArrayType] {
        &self.output_types
    }

    /// Creates a metadata-only shard-map operation directly from its boundary parts. The local body program that
    /// realizes this boundary is authored separately and attached as the operation's `body` region.
    #[inline]
    pub(crate) fn from_boundary(
        shard_map: ShardMap,
        input_types: Vec<ArrayType>,
        output_types: Vec<ArrayType>,
    ) -> Self {
        Self { shard_map, input_types, output_types, marker: PhantomData }
    }

    /// Returns a copy of this operation whose global output types are replaced by `global_output_types`, keeping the
    /// manual SPMD metadata and global input types unchanged. Forward-mode differentiation uses this to align a
    /// tangent boundary's global output types with the tangent descriptors derived from the staged primal
    /// `shard_map`'s adapted output types (see `adapt_traced_shard_map_output_type`).
    pub(crate) fn with_global_output_types(
        mut self,
        global_output_types: Vec<ArrayType>,
    ) -> Result<Self, ShardMapTraceError> {
        if global_output_types.len() != self.output_types.len() {
            return Err(ShardMapTraceError::OutputTypeCountMismatch {
                expected: self.output_types.len(),
                actual: global_output_types.len(),
            });
        }
        self.output_types = global_output_types;
        Ok(self)
    }
}

impl<V> Display for ShardMapOperation<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("shard_map")
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
        return Err(TypeError::invalid(format!(
            "{} input types do not match the captured shard-map boundary",
            operation_name
        )));
    }
    Ok(captured_output_types
        .iter()
        .map(|output_type| adapt_traced_shard_map_output_type(input_types, captured_input_types, output_type))
        .collect::<Vec<_>>())
}

/// Validates the single attached shard-map body boundary and returns its interface.
fn shard_map_body_interface<T: Type>(
    region_interfaces: &[RegionInterface<T>],
    input_count: usize,
    output_count: usize,
) -> Result<&RegionInterface<T>, TypeError> {
    check_count!("region", region_interfaces, 1, TypeError);
    let interface = &region_interfaces[0];
    check_count!("body input", interface.input_types(), input_count, TypeError);
    check_count!("body output", interface.output_types(), output_count, TypeError);
    Ok(interface)
}

impl<V: Clone> Operation for ShardMapOperation<V> {
    type Type = ArrayProgramType;

    #[inline]
    fn name(&self) -> &'static str {
        "shard_map"
    }

    #[inline]
    fn region_slots(&self) -> &'static [RegionSlot] {
        const { &[RegionSlot::computation("body")] }
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayProgramType],
        region_interfaces: &[RegionInterface<ArrayProgramType>],
    ) -> Result<Vec<ArrayProgramType>, TypeError> {
        let body_interface =
            shard_map_body_interface(region_interfaces, self.input_types.len(), self.output_types.len())?;
        let input_types = input_types
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
            .collect::<Result<Vec<_>, TypeError>>()?;
        for r#type in body_interface.input_types().iter().chain(body_interface.output_types()) {
            <&ArrayType>::try_from(r#type)?;
        }
        Ok(infer_shard_map_output_types(
            self.name(),
            self.input_types.as_slice(),
            self.output_types.as_slice(),
            input_types.as_slice(),
        )?
        .into_iter()
        .map(Into::into)
        .collect())
    }
}

/// Online partial-evaluation rule for a staged `shard_map` — the map-boundary sibling of the
/// [`JitCallOperation`](crate::experimental::ops::JitCallOperation) call rule: it splits the local body against the
/// caller's known-ness while preserving the `shard_map` boundary, its mesh, and its shardings on both sides.
///
/// The split fires only when some known input does *not* [`resolve`](Context::resolve) to a program constant in
/// the known-side context — a genuine tracer into a live outer trace. All-known, all-unknown, and constant-resolved
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
    V: PartialEq
        + Value<Type = ArrayProgramType>
        + ryft_core::ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + Concretizable<bool>,
    C: Context<Type = ArrayProgramType, Constant = V, Operation = XlaOperation<V>>,
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        // Split only a mixed boundary with at least one known-but-symbolic input; everything else keeps the default
        // fold-or-residualize behavior and therefore the original boundary.
        if !context.any_known_is_symbolic(inputs) || inputs.iter().all(PartialEvaluationValue::is_known) {
            return context.fold_or_residualize(
                XlaOperation::ShardMap(Box::new(self.clone())),
                driver.regions().map(|region| region.to_program()).collect(),
                inputs,
            );
        }

        // Split the local body through the shared online boundary machinery. The body's inputs are index-aligned
        // with the boundary inputs and carry the *local* types, so both split sides stay local programs.
        let body_program = driver.region(0)?;
        let input_known = inputs.iter().map(PartialEvaluationValue::is_known).collect::<Vec<bool>>();
        let partition = body_program.partition(input_known.as_slice())?;
        // A trivial partition — one whose known program contains no instructions — hoists no work (its known side
        // can only forward known inputs as residual edges), so keep the original boundary and let the default
        // materialize those knowns directly as residual feeders.
        if partition.known_program().instructions().is_empty() {
            return context.fold_or_residualize(
                XlaOperation::ShardMap(Box::new(self.clone())),
                vec![body_program.to_program()],
                inputs,
            );
        }

        // Derive each residual edge's global boundary type and sharding from its local type.
        let mesh = self.shard_map.mesh();
        let residual_edge_boundaries = partition
            .residual_inputs()
            .iter()
            .zip(partition.residual_program().input_types())
            .filter_map(|(source, edge_type)| source.is_known().then_some(edge_type))
            .map(|edge_type| {
                let edge_type = <&ArrayType>::try_from(&edge_type).map_err(ProgramError::from)?;
                residual_boundary(edge_type, mesh).map_err(trace_error_from_shard_map)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Gather the known-side boundary metadata: shardings and global types per original index, with the residual
        // edges appended.
        let known_global_input_types = partition
            .known_input_indices()
            .iter()
            .map(|&index| self.input_types[index].clone())
            .collect::<Vec<_>>();
        let known_in_shardings = partition
            .known_input_indices()
            .iter()
            .map(|&index| self.shard_map.in_shardings()[index].clone())
            .collect::<Vec<_>>();
        let known_output_indices = partition
            .outputs()
            .iter()
            .enumerate()
            .filter_map(|(index, output)| output.is_known().then_some(index))
            .collect::<Vec<_>>();
        let mut known_global_output_types =
            known_output_indices.iter().map(|&index| self.output_types[index].clone()).collect::<Vec<_>>();
        let mut known_out_shardings = known_output_indices
            .iter()
            .map(|&index| self.shard_map.out_shardings()[index].clone())
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
                    staged_global_input_types.push(self.input_types[*index].clone());
                    staged_in_shardings.push(self.shard_map.in_shardings()[*index].clone());
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
                staged_global_output_types.push(self.output_types[index].clone());
                staged_out_shardings.push(self.shard_map.out_shardings()[index].clone());
            }
        }

        // Bind the known-side `shard_map` into the enclosing known-side context, emit the residual `shard_map`
        // over the surviving unknown boundary inputs plus the residual edges, and reassemble the original outputs.
        context.inline_partitioned_program(
            partition,
            inputs,
            |known_program| {
                let known_shard_map = ShardMap::from_shardings(
                    mesh.clone(),
                    known_in_shardings,
                    known_out_shardings,
                    self.shard_map.manual_axes().to_vec(),
                    self.shard_map.check_vma(),
                );
                let known_operation = ShardMapOperation::from_boundary(
                    known_shard_map,
                    known_global_input_types,
                    known_global_output_types,
                );
                (XlaOperation::ShardMap(Box::new(known_operation)), vec![known_program])
            },
            |residual_program| {
                let staged_shard_map = ShardMap::from_shardings(
                    mesh.clone(),
                    staged_in_shardings,
                    staged_out_shardings,
                    self.shard_map.manual_axes().to_vec(),
                    self.shard_map.check_vma(),
                );
                let staged_operation = ShardMapOperation::from_boundary(
                    staged_shard_map,
                    staged_global_input_types,
                    staged_global_output_types,
                );
                (XlaOperation::ShardMap(Box::new(staged_operation)), vec![residual_program])
            },
        )
    }
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

/// Returns the descriptor used for one tangent shard-map boundary value. Differentiable values use their declared
/// tangent representation, while non-differentiable positional values use their first-class zero-space descriptor.
fn tangent_boundary_type(r#type: &ArrayType) -> ArrayType {
    r#type.tangent()
}

/// Fuse-linearizes a shard-map body capture-free into a primal body and a tangent body that thread residuals as plain
/// operand edges across the shard-map boundary.
///
/// The borrowed `body` region is fuse-linearized once, yielding a primal sub-program
/// `local_inputs -> [local_outputs..., local_residuals...]` and a tangent sub-program
/// `[local_input_tangents..., local_residuals...] -> [local_output_tangents...]` together with the residual count.
/// Each sub-program pairs with a fresh boundary [`ShardMapOperation`]: the primal boundary gains the residual edges
/// as trailing outputs, the tangent boundary gains them as trailing inputs, and both use the replicated residual
/// boundary from [`residual_boundary`]. This is the shard-map counterpart of the jitted-call rule, realizing
/// `jvp(shard_map(f)) = shard_map(jvp f)` with no symbolic capture ever introduced.
///
/// # Parameters
///
///   - `operation`: Boundary metadata of the primal shard-map being linearized.
///   - `program`: Borrowed `body` region of that shard-map.
#[allow(clippy::type_complexity)]
fn shard_map_bodies<
    V: PartialEq
        + Value<Type = ArrayProgramType>
        + ryft_core::ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + Concretizable<bool>,
>(
    operation: &ShardMapOperation<V>,
    program: RegionRef<'_, V, XlaOperation<V>>,
) -> Result<
    (
        (ShardMapOperation<V>, Program<V, XlaOperation<V>, Vec<V>, Vec<V>>),
        (ShardMapOperation<V>, Program<V, XlaOperation<V>, Vec<V>, Vec<V>>),
        usize,
    ),
    ShardMapTraceError,
> {
    let output_count = operation.global_output_types().len();
    let (primal_program, tangent_program, residual_count) =
        program.linearize().map_err(ProgramError::from)?.into_parts();

    let mesh = operation.shard_map().mesh();
    let manual_axes = operation.shard_map().manual_axes();

    // The primal sub-program's trailing outputs beyond the original outputs are the residual edges; their local
    // types are authoritative and back the residual boundary on both bodies.
    let primal_local_output_types = primal_program.output_types();
    let residual_local_types = &primal_local_output_types[output_count..];
    let mut residual_global_types = Vec::with_capacity(residual_count);
    let mut residual_shardings = Vec::with_capacity(residual_count);
    for residual_local_type in residual_local_types {
        let residual_local_type = <&ArrayType>::try_from(residual_local_type).map_err(ProgramError::from)?;
        let (residual_global_type, residual_sharding) = residual_boundary(residual_local_type, mesh)?;
        residual_global_types.push(residual_global_type);
        residual_shardings.push(residual_sharding);
    }

    let primal_operation = {
        let shard_map = ShardMap::from_shardings(
            mesh.clone(),
            operation.shard_map().in_shardings().to_vec(),
            operation
                .shard_map()
                .out_shardings()
                .iter()
                .cloned()
                .chain(residual_shardings.iter().cloned())
                .collect(),
            manual_axes.to_vec(),
            operation.shard_map().check_vma(),
        );
        let global_output_types = operation
            .global_output_types()
            .iter()
            .cloned()
            .chain(residual_global_types.iter().cloned())
            .collect();
        ShardMapOperation::from_boundary(shard_map, operation.global_input_types().to_vec(), global_output_types)
    };

    let tangent_operation = {
        let shard_map = ShardMap::from_shardings(
            mesh.clone(),
            operation.shard_map().in_shardings().iter().cloned().chain(residual_shardings).collect(),
            operation.shard_map().out_shardings().to_vec(),
            manual_axes.to_vec(),
            operation.shard_map().check_vma(),
        );
        let global_input_types = operation
            .global_input_types()
            .iter()
            .map(tangent_boundary_type)
            .chain(residual_global_types)
            .collect();
        let global_output_types = operation.global_output_types().iter().map(tangent_boundary_type).collect();
        ShardMapOperation::from_boundary(shard_map, global_input_types, global_output_types)
    };

    Ok(((primal_operation, primal_program), (tangent_operation, tangent_program), residual_count))
}

/// Capture-free forward-mode (JVP) rule for [`ShardMapOperation`], binding a primal `shard_map` and a tangent
/// `shard_map` as ordinary [`XlaOperation`]s through the active context: a staging context stages both operations
/// over its shared builder, while an eager context compiles and executes them immediately.
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
///
/// # Parameters
///
///   - `context`: Active evaluation or staging context used to bind the differentiated shard-map operations.
///   - `driver`: Call-scoped access to the attached shard-map body region.
///   - `inputs`: Primal and tangent values for the shard-map operands.
impl<C, V> DifferentiableOperation<C> for ShardMapOperation<V>
where
    C: Context<Type = ArrayProgramType, Constant = V, Operation = XlaOperation<V>> + Zero<C::Value>,
    V: PartialEq
        + Value<Type = ArrayProgramType>
        + ryft_core::ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + Concretizable<bool>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let output_count = self.output_types.len();
        check_count!("input", inputs, self.input_types.len(), ProgramError);

        let body_program = driver.region(0)?;
        let ((primal_operation, primal_body_program), (tangent_operation, tangent_body_program), _residual_count) =
            shard_map_bodies(self, body_program).map_err(trace_error_from_shard_map)?;

        // Bind the primal `shard_map`, recovering the primal outputs followed by the residual values.
        let primal_operands = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal_operation = XlaOperation::ShardMap(Box::new(primal_operation));
        let mut primal_outputs = context.bind(primal_operation, vec![primal_body_program], &primal_operands)?;
        if primal_outputs.len() < output_count {
            return Err(ProgramError::MalformedProgram(format!(
                "shard_map primal body produced {} outputs which is fewer than its {output_count} primal \
                 output(s)",
                primal_outputs.len(),
            ))
            .into());
        }
        let residuals = primal_outputs.split_off(output_count);

        // The primal `shard_map` may re-embed its outputs into the caller's ambient sharding envelope. The tangent
        // `shard_map` carries residual operands and therefore cannot infer that envelope through the single-input
        // adaptation path, so derive its output descriptors from the adapted primal outputs while retaining the
        // tangent element representation.
        let tangent_output_types = primal_outputs
            .iter()
            .map(|output| {
                <&ArrayType>::try_from(output.r#type().as_ref())
                    .map(tangent_boundary_type)
                    .map_err(ProgramError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let tangent_operation = tangent_operation
            .with_global_output_types(tangent_output_types)
            .map_err(trace_error_from_shard_map)?;

        // Bind the tangent `shard_map` over the operand tangents followed by the residual values, recovering one
        // output tangent per primal output.
        // The tangent `shard_map` takes every operand tangent as a real program input, so materialize structural
        // zeros at this sub-program boundary.
        let mut tangent_operands = inputs
            .iter()
            .map(|input| input.tangent().clone().materialize(context))
            .collect::<Result<Vec<_>, _>>()?;
        tangent_operands.extend(residuals);
        let tangent_operation = XlaOperation::ShardMap(Box::new(tangent_operation));
        let tangent_outputs = context.bind(tangent_operation, vec![tangent_body_program], &tangent_operands)?;
        check_count!("output", tangent_outputs, output_count, ProgramError);

        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| DifferentiationDual::new(primal, tangent))
            .collect::<Result<Vec<_>, _>>()?)
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
///   2. Transposes the tangent body's flat program with [`TranspositionDriver::transpose_program`] with respect
///      to the same linear operands,
///      so the transposed body maps `[outputs..., known_input_values...]` to `[linear_input_cotangents...]`,
///      in body-input order on each side. It re-wraps the transposed program in a fresh [`FlatTracedShardMap`] whose
///      boundary shardings are permuted and dualized to match — its inputs carry the cotangent duals of the original
///      output shardings followed by the known operands' input shardings, and its outputs carry the cotangent duals of
///      the linear operands' input shardings.
///   3. Re-wraps the transposed body in a fresh [`ShardMapOperation`] and stages it over
///      `[outputs..., known_input_values...]`, keeping the manual region intact so both forward and reverse
///      mode over a `shard_map` stay manual rather than inlined.
///
/// The returned cotangents place the transposed `shard_map`'s outputs at the linear-operand positions and a structural
/// [`MaybeZero::Zero`] at the known positions. The body transposition happens through
/// [`TranspositionDriver::transpose_program`] in this same operation family, so it is value-level and introduces
/// no recursive transposition obligation on [`XlaOperation`].
///
/// # Parameters
///
///   - `operation`: Primal tangent `shard_map` staged into the tangent program.
///   - `context`: Active transpose tracing context the pullback is staged into.
///   - `driver`: Instruction-scoped access to the attached body region and its recursive transposition machinery.
///   - `inputs`: Per-operand [`PartialValue`] knowledge, mirroring the body's global inputs one-to-one. The
///     [`Unknown`](PartialValue::Unknown) entries are the input tangents; the [`Known`](PartialValue::Known) entries
///     carry the residual tracers the pullback reads.
///   - `outputs`: Symbolic cotangents for the tangent `shard_map`'s outputs.
pub fn transpose_primal_shard_map<
    V: Value<Type = ArrayProgramType> + ryft_core::ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    D: TranspositionDriver<V, XlaOperation<V>>,
>(
    operation: &ShardMapOperation<V>,
    context: &mut TracingContext<V, XlaOperation<V>>,
    driver: &D,
    inputs: &[PartialValue<Tracer<TracingContext<V, XlaOperation<V>>>>],
    outputs: &[MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>],
) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>>, ProgramError> {
    let operand_linear = inputs.iter().map(PartialValue::is_unknown).collect::<Vec<_>>();
    check_count!("input", operand_linear, operation.global_input_types().len(), ProgramError);

    // A shard_map with no live output cotangents is a zero linear map, so every operand cotangent is zero.
    if outputs.iter().all(MaybeZero::is_zero) {
        return Ok(inputs.iter().map(|input| MaybeZero::Zero(input.r#type().cotangent())).collect());
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
    // side; re-wrap it as a transposed shard-map boundary whose shardings are permuted to match.
    let (transposed_operation, transposed_body_program) =
        transpose_shard_map_body(operation, driver, operand_linear.as_slice())?;

    // Stage the output cotangents, materializing a typed zero for each structurally zero cotangent, then stage a fresh
    // `shard_map` over the transposed body on `[outputs..., known_input_values...]`. Its outputs are the
    // linear-input cotangents.
    let output_types = &operation.output_types;
    check_count!("output", outputs, output_types.len(), ProgramError);
    let mut operands = Vec::with_capacity(output_types.len() + known_values.len());
    for (cotangent, output_type) in outputs.iter().zip(output_types.iter()) {
        let output_type = ArrayProgramType::Array(output_type.cotangent());
        operands.push(materialize_transpose_cotangent(context, cotangent, &output_type, inputs)?);
    }
    operands.extend(known_values);
    let transposed_operation = XlaOperation::ShardMap(Box::new(transposed_operation));
    let input_cotangents =
        context.stage_operation(transposed_operation, vec![transposed_body_program], operands.as_slice())?;
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
                if linear { input_cotangents.next().unwrap() } else { MaybeZero::Zero(input.r#type().cotangent()) }
            },
        )
        .collect();
    Ok(cotangents)
}

/// Transpose rule for a primal tangent [`ShardMapOperation`], forwarding to [`transpose_primal_shard_map`]. The body
/// transposition happens on the body's flat [`XlaConstant`]-keyed program, so the recursion is resolved once at
/// definition time and instantiating this implementation introduces no recursive
/// [`TransposableOperation`] obligation on [`XlaOperation`].
impl<V> TransposableOperation<V, XlaOperation<V>> for ShardMapOperation<V>
where
    V: Value<Type = ArrayProgramType> + ryft_core::ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    fn transpose<D: TranspositionDriver<V, XlaOperation<V>>>(
        &self,
        context: &mut TracingContext<V, XlaOperation<V>>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, XlaOperation<V>>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>>, DifferentiationError> {
        transpose_primal_shard_map(self, context, driver, inputs, outputs).map_err(DifferentiationError::from)
    }
}

/// Transposes one tangent shard-map body into the reverse boundary and body consumed by
/// [`transpose_primal_shard_map`].
///
/// The tangent body's flat program is transposed with [`TranspositionDriver::transpose_program`] under
/// `input_linearity`, producing a program mapping `[outputs..., known_input_values...]` to
/// `[linear_input_cotangents...]`. The transposed boundary permutes and dualizes the original one to match: its global
/// inputs are the cotangent descriptors of the original global outputs followed by the known operands' original
/// global inputs, and its global outputs are the cotangent descriptors of the linear operands' original global inputs.
///
/// # Parameters
///
///   - `operation`: Boundary metadata of the tangent `shard_map` produced by [`shard_map_bodies`], whose global
///     inputs are `[input_tangents..., residuals...]` and whose global outputs are `[output_tangents...]`.
///   - `driver`: Instruction-scoped access to the attached body region and its recursive transposition machinery.
///   - `input_linearity`: Per-input linearity flags over the tangent boundary's global inputs.
#[allow(clippy::type_complexity)]
fn transpose_shard_map_body<
    V: Value<Type = ArrayProgramType> + ryft_core::ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    D: TranspositionDriver<V, XlaOperation<V>>,
>(
    operation: &ShardMapOperation<V>,
    driver: &D,
    input_linearity: &[bool],
) -> Result<(ShardMapOperation<V>, Program<V, XlaOperation<V>, Vec<V>, Vec<V>>), ProgramError> {
    let transposed_program = driver.transpose_program(driver.region(0)?, input_linearity)?;

    let in_shardings = operation
        .shard_map()
        .out_shardings()
        .iter()
        .map(Sharding::cotangent)
        .chain(
            input_linearity
                .iter()
                .zip(operation.shard_map().in_shardings().iter())
                .filter(|&(&linear, _)| !linear)
                .map(|(_, sharding)| sharding.clone()),
        )
        .collect::<Vec<_>>();
    let out_shardings = input_linearity
        .iter()
        .zip(operation.shard_map().in_shardings().iter())
        .filter(|&(&linear, _)| linear)
        .map(|(_, sharding)| sharding.cotangent())
        .collect::<Vec<_>>();
    let shard_map = ShardMap::from_shardings(
        operation.shard_map().mesh().clone(),
        in_shardings,
        out_shardings,
        operation.shard_map().manual_axes().to_vec(),
        operation.shard_map().check_vma(),
    );

    let global_input_types = operation
        .global_output_types()
        .iter()
        .map(DifferentiableType::cotangent)
        .chain(
            input_linearity
                .iter()
                .zip(operation.global_input_types().iter())
                .filter(|&(&linear, _)| !linear)
                .map(|(_, global_type)| global_type.clone()),
        )
        .collect::<Vec<_>>();
    let global_output_types = input_linearity
        .iter()
        .zip(operation.global_input_types().iter())
        .filter(|&(&linear, _)| linear)
        .map(|(_, global_type)| global_type.cotangent())
        .collect::<Vec<_>>();

    Ok((ShardMapOperation::from_boundary(shard_map, global_input_types, global_output_types), transposed_program))
}

fn trace_error_from_shard_map(error: ShardMapTraceError) -> ProgramError {
    ProgramError::Type(TypeError::invalid(error.to_string()))
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
        + ParameterizedFamily<ArrayProgramType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<ShardMapTracer>,
    Output::Family: ParameterizedFamily<Sharding>
        + ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayProgramType>
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

fn apply_traced_shard_map<C>(
    context: C,
    traced: FlatTracedShardMap,
    traced_inputs: Vec<C::Value>,
) -> Result<Vec<C::Value>, ShardMapTraceError>
where
    C: Context<Type = ArrayProgramType, Constant = XlaConstant, Operation = XlaOperation>,
{
    let (operation, body_program) = ShardMapOperation::from_body(traced);
    Ok(context.bind(XlaOperation::ShardMap(Box::new(operation)), vec![body_program], traced_inputs.as_slice())?)
}

fn global_input_types_from_traced_inputs<V, Input>(
    traced_inputs: &Input,
) -> Result<Input::To<ArrayType>, ShardMapTraceError>
where
    V: Value<Type = ArrayType>,
    Input: Parameterized<V>,
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
            + ParameterizedFamily<ArrayProgramType>
            + ParameterizedFamily<Sharding>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ShardMapTracer>,
        Output::Family: ParameterizedFamily<Sharding>
            + ParameterizedFamily<ArrayType>
            + ParameterizedFamily<ArrayProgramType>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ShardMapTracer>,
        Output::To<ShardMapTracer>: Parameterized<ShardMapTracer, To<ArrayType> = Output>;

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
            + ParameterizedFamily<ArrayProgramType>
            + ParameterizedFamily<Sharding>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ShardMapTracer>,
        Output: Parameterized<ArrayType>,
        Output::Family: ParameterizedFamily<Sharding>
            + ParameterizedFamily<ArrayType>
            + ParameterizedFamily<ArrayProgramType>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ShardMapTracer>,
        Output::To<ShardMapTracer>: Parameterized<ShardMapTracer, To<ArrayType> = Output>,
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

/// Invokes a traced shard map through the composite value behind one public array projection.
impl<V> ShardMapInvocationLeaf for ProjectedValue<ArrayType, V>
where
    V: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected = ProjectedValue<ArrayType, V>>,
    ProjectedValue<ArrayType, V>: Value<Type = ArrayType>,
    V::DispatchDomain: Context<Type = ArrayProgramType, Constant = XlaConstant, Operation = XlaOperation>,
{
    type Return<Input: Parameterized<Self>, Output: Parameterized<ArrayType>>
        = Output::To<Self>
    where
        Input::Family: ParameterizedFamily<ArrayType>
            + ParameterizedFamily<ArrayProgramType>
            + ParameterizedFamily<Sharding>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ShardMapTracer>,
        Output::Family: ParameterizedFamily<Sharding>
            + ParameterizedFamily<ArrayType>
            + ParameterizedFamily<ArrayProgramType>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ShardMapTracer>
            + ParameterizedFamily<Self>,
        Output::To<ShardMapTracer>: Parameterized<ShardMapTracer, To<ArrayType> = Output>;

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
            + ParameterizedFamily<ArrayProgramType>
            + ParameterizedFamily<Sharding>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ShardMapTracer>,
        Output: Parameterized<ArrayType>,
        Output::Family: ParameterizedFamily<Sharding>
            + ParameterizedFamily<ArrayType>
            + ParameterizedFamily<ArrayProgramType>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ShardMapTracer>
            + ParameterizedFamily<Self>,
        Output::To<ShardMapTracer>: Parameterized<ShardMapTracer, To<ArrayType> = Output>,
        F: FnOnce(ShardMapLocalTraceInput<Input::To<ArrayType>>) -> ShardMapLocalTraceOutput<Output>,
    {
        let output_structure = out_specs.parameter_structure();
        let global_input_types = global_input_types_from_traced_inputs::<Self, _>(&inputs)?;
        let global_in_specs = reparameterize_shardings::<
            Input::To<Sharding>,
            <Input::To<ArrayType> as Parameterized<ArrayType>>::To<Sharding>,
        >(in_specs, global_input_types.parameter_structure())?;
        let traced_inputs = inputs.into_parameters().map(ProjectedValue::into_value).collect::<Vec<_>>();
        let context = match traced_inputs.first() {
            Some(input) => input.dispatch_domain(),
            None if output_structure.parameter_count() == 0 => {
                return Ok(Output::To::<Self>::from_parameters(output_structure, Vec::new())?);
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
        let outputs = apply_traced_shard_map(context, traced, traced_inputs)?
            .into_iter()
            .map(|value| ValueProjection::<ArrayType>::into_projected(value).map_err(ProgramError::from))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Output::To::<Self>::from_parameters(output_structure, outputs)?)
    }
}

#[cfg(test)]
mod tests {
    use std::ops::{Deref, DerefMut};

    use ryft_core::contexts::{Context, StagingContext};
    use ryft_core::differentiation::{DifferentiableType, DifferentiationError, TranspositionDriver};
    use ryft_core::operations::math::{AddOperation, MulOperation};
    use ryft_core::parameters::Placeholder;
    use ryft_core::partial::PartialValue;
    use ryft_core::programs::MaybeZero;
    use ryft_core::programs::Program;
    use ryft_core::programs::effects::Effects;
    use ryft_core::programs::operations::Operation;
    use ryft_core::programs::regions::{EmptyRegionDriver, RegionDriver, RegionInterface, RegionRef};
    use ryft_core::programs::types::{TypeError, Typed};
    use ryft_core::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use ryft_core::tracing::{DomainTracingContext, TracingContext};
    use ryft_core::types::{
        ArrayProgramType, ArrayType, DataType, Dimension, DimensionBounds, DimensionType, DimensionVariable, Shape,
    };

    use crate::experimental::domains::XlaDomain;
    use crate::experimental::ops::{XlaArrayConstant, XlaConstant, XlaOperation, XlaProgram};
    use crate::experimental::shard_map::{FlatTracedShardMap, ShardMap};

    use super::{ShardMapOperation, transpose_primal_shard_map, transpose_shard_map_body};

    /// Array-oriented facade over the production composite XLA program builder.
    struct XlaProgramBuilder(crate::experimental::ops::XlaProgramBuilder);

    impl XlaProgramBuilder {
        /// Creates an empty composite builder.
        fn new() -> Self {
            Self(crate::experimental::ops::XlaProgramBuilder::new())
        }

        /// Adds an array input to the composite builder.
        fn add_input(&mut self, r#type: ArrayType) -> ryft_core::AtomId {
            self.0.add_input(ArrayProgramType::Array(r#type))
        }

        /// Finalizes the composite program.
        fn build<Input: ryft_core::Parameterized<XlaConstant>, Output: ryft_core::Parameterized<XlaConstant>>(
            self,
            output_ids: Vec<ryft_core::AtomId>,
            input_structure: Input::ParameterStructure,
            output_structure: Output::ParameterStructure,
        ) -> Result<XlaProgram<Input, Output>, ryft_core::ProgramError> {
            self.0.build(output_ids, input_structure, output_structure)
        }
    }

    impl Deref for XlaProgramBuilder {
        type Target = crate::experimental::ops::XlaProgramBuilder;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl DerefMut for XlaProgramBuilder {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }

    /// Test-only driver that returns a predetermined transpose for its one attached source region.
    struct TestTranspositionDriver {
        /// Source region exposed to the operation rule.
        source: XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>,

        /// Predetermined transposed program returned by the recursive request.
        transposed: XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>,
    }

    impl RegionDriver<XlaConstant, XlaOperation> for TestTranspositionDriver {
        fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, XlaConstant, XlaOperation>>
        where
            XlaConstant: 'r,
            XlaOperation: 'r,
        {
            std::iter::once(self.source.entry_region_ref())
        }
    }

    impl TranspositionDriver<XlaConstant, XlaOperation> for TestTranspositionDriver {
        fn transpose_program(
            &self,
            _region: RegionRef<'_, XlaConstant, XlaOperation>,
            _input_linearity: &[bool],
        ) -> Result<Program<XlaConstant, XlaOperation, Vec<XlaConstant>, Vec<XlaConstant>>, DifferentiationError>
        {
            Ok(self.transposed.clone())
        }
    }

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

    #[test]
    fn test_shard_map_composite_boundary_is_array_only() {
        let array_type = ArrayType::scalar(DataType::F32);
        let composite_array_type = ArrayProgramType::Array(array_type.clone());
        let operation = ShardMapOperation::<XlaArrayConstant>::from_boundary(
            single_input_test_shard_map(),
            vec![array_type.clone()],
            vec![array_type],
        );
        let array_body =
            RegionInterface::new(vec![composite_array_type.clone()], vec![composite_array_type.clone()], Effects::PURE);

        assert_eq!(
            operation
                .infer_output_types(std::slice::from_ref(&composite_array_type), std::slice::from_ref(&array_body)),
            Ok(vec![composite_array_type.clone()]),
        );
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&composite_array_type), &[]),
            Err(TypeError::invalid("expected 1 region but got 0")),
        );

        let dimension_type = ArrayProgramType::Dimension(DimensionType::new(DimensionVariable::new(
            "size",
            DimensionBounds::positive(Some(8)).unwrap(),
        )));
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&dimension_type), std::slice::from_ref(&array_body)),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );

        let dimension_input_body =
            RegionInterface::new(vec![dimension_type.clone()], vec![composite_array_type.clone()], Effects::PURE);
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&composite_array_type),
                std::slice::from_ref(&dimension_input_body),
            ),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );

        let dimension_output_body =
            RegionInterface::new(vec![composite_array_type.clone()], vec![dimension_type], Effects::PURE);
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&composite_array_type),
                std::slice::from_ref(&dimension_output_body),
            ),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
    }

    #[test]
    fn test_shard_map_jvp_uses_tangent_boundary_descriptors() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let boundary_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(4)]));
        let ambient_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let ambient_input_type = boundary_type.clone().with_sharding(ambient_sharding).unwrap();
        let ambient_tangent_type = ambient_input_type.tangent();
        let boundary_tangent_type = boundary_type.tangent();

        let body = {
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(boundary_type.clone());
            let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, input]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let shard_map = ShardMap::from_shardings(
            mesh.clone(),
            vec![Sharding::replicated(mesh.clone(), 1)],
            vec![Sharding::replicated(mesh, 1)],
            vec!["x".to_string()],
            true,
        );
        let operation =
            ShardMapOperation::from_boundary(shard_map, vec![boundary_type.clone()], vec![boundary_type.clone()]);
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(ambient_input_type);
        let body_region = builder.import_program(body);
        let output = builder
            .add_instruction(XlaOperation::ShardMap(Box::new(operation)), vec![body_region], vec![input])
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let fused = program.jvp().unwrap();
        let shard_maps = fused
            .instructions()
            .iter()
            .filter_map(|instruction| match instruction.operation() {
                XlaOperation::ShardMap(operation) => Some((instruction, operation.as_ref())),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(shard_maps.len(), 2);
        let (primal_instruction, primal_operation) = shard_maps[0];
        let (tangent_instruction, tangent_operation) = shard_maps[1];

        assert!(primal_operation.global_output_types().len() > 1);
        assert_eq!(tangent_operation.global_input_types()[0], boundary_tangent_type);
        assert_eq!(&tangent_operation.global_input_types()[1..], &primal_operation.global_output_types()[1..],);
        assert_eq!(tangent_operation.global_output_types(), std::slice::from_ref(&ambient_tangent_type));

        let primal_body = fused.region_ref(primal_instruction.regions()[0]).unwrap().to_program();
        let tangent_body = fused.region_ref(tangent_instruction.regions()[0]).unwrap().to_program();
        assert_eq!(primal_body.input_types(), vec![ArrayProgramType::Array(boundary_type)]);
        assert_eq!(tangent_body.input_types()[0], ArrayProgramType::Array(boundary_tangent_type));
        assert_eq!(&tangent_body.input_types()[1..], &primal_body.output_types()[1..]);
        assert_eq!(
            tangent_body.output_types(),
            vec![ArrayProgramType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]),))]
        );
    }

    #[test]
    fn test_shard_map_transpose_dualizes_boundary_descriptors() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let tangent_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
            .unwrap()
            .with_unreduced_axes(["x"])
            .unwrap();
        let tangent_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(tangent_sharding.clone())
            .unwrap();
        let cotangent_type = tangent_type.cotangent();

        let source = {
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(tangent_type.clone());
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![input], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let transposed = {
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(cotangent_type.clone());
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![input], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let driver = TestTranspositionDriver { source, transposed };
        let shard_map = ShardMap::from_shardings(
            mesh,
            vec![tangent_sharding.clone()],
            vec![tangent_sharding.clone()],
            Vec::new(),
            true,
        );
        let operation = ShardMapOperation::from_boundary(shard_map, vec![tangent_type.clone()], vec![tangent_type]);

        let (transposed, _) = transpose_shard_map_body(&operation, &driver, &[true]).unwrap();
        assert_eq!(transposed.global_input_types(), std::slice::from_ref(&cotangent_type));
        assert_eq!(transposed.global_output_types(), std::slice::from_ref(&cotangent_type));
        assert_eq!(transposed.shard_map().in_shardings(), &[tangent_sharding.cotangent()]);
        assert_eq!(transposed.shard_map().out_shardings(), &[tangent_sharding.cotangent()]);
    }

    #[test]
    fn test_shard_map_zero_transpose_uses_cotangent_descriptors() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let tangent_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
            .unwrap()
            .with_unreduced_axes(["x"])
            .unwrap();
        let tangent_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(tangent_sharding.clone())
            .unwrap();
        let cotangent_type = tangent_type.cotangent();
        let shard_map =
            ShardMap::from_shardings(mesh, vec![tangent_sharding.clone()], vec![tangent_sharding], Vec::new(), true);
        let operation =
            ShardMapOperation::from_boundary(shard_map, vec![tangent_type.clone()], vec![tangent_type.clone()]);
        let mut context = TracingContext::<XlaConstant, XlaOperation>::new();
        let known = context.input(ArrayProgramType::Array(tangent_type.clone()));
        let cotangents = transpose_primal_shard_map(
            &operation,
            &mut context,
            &EmptyRegionDriver,
            &[PartialValue::Known(known)],
            &[MaybeZero::Zero(ArrayProgramType::Array(tangent_type))],
        )
        .unwrap();
        assert!(
            matches!(&cotangents[..], [MaybeZero::Zero(actual)] if actual == &ArrayProgramType::Array(cotangent_type))
        );
    }

    #[test]
    fn test_shard_map_mixed_output_transpose_materializes_zero_space_values() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 1);
        let value_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(sharding.clone())
            .unwrap();
        let predicate_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(sharding.clone())
            .unwrap();
        let source = {
            let mut builder = XlaProgramBuilder::new();
            let value = builder.add_input(value_type.clone());
            let predicate = builder.add_constant(XlaConstant::new(0, ArrayProgramType::Array(predicate_type.clone())));
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![value, predicate],
                    vec![Placeholder],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let transposed = {
            let mut builder = XlaProgramBuilder::new();
            let value_cotangent = builder.add_input(value_type.clone());
            let _predicate_cotangent = builder.add_input(predicate_type.cotangent());
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![value_cotangent],
                    vec![Placeholder; 2],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let driver = TestTranspositionDriver { source, transposed };
        let shard_map =
            ShardMap::from_shardings(mesh, vec![sharding.clone()], vec![sharding.clone(), sharding], Vec::new(), true);
        let operation = ShardMapOperation::from_boundary(
            shard_map,
            vec![value_type.clone()],
            vec![value_type.clone(), predicate_type.clone()],
        );
        let mut context = TracingContext::<XlaConstant, XlaOperation>::new();
        let value_cotangent = context.input(ArrayProgramType::Array(value_type.clone()));

        let contributions = transpose_primal_shard_map(
            &operation,
            &mut context,
            &driver,
            &[PartialValue::Unknown(ArrayProgramType::Array(value_type.clone()))],
            &[MaybeZero::Value(value_cotangent), MaybeZero::Zero(ArrayProgramType::Array(predicate_type.cotangent()))],
        )
        .unwrap();

        assert!(matches!(&contributions[..], [MaybeZero::Value(value)]
                if value.r#type().as_ref() == &ArrayProgramType::Array(value_type)));
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
        let doubled = builder
            .add_instruction(AddOperation::<ArrayProgramType>::new(), Vec::new(), vec![known_input, known_input])
            .unwrap()[0];
        let product =
            builder.add_instruction(MulOperation::new(), Vec::new(), vec![known_input, runtime_input]).unwrap()[0];
        let sum = builder
            .add_instruction(AddOperation::<ArrayProgramType>::new(), Vec::new(), vec![runtime_input, known_input])
            .unwrap()[0];
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
    fn test_traced_shard_map_binds_staging_onto_input_trace() {
        let body = single_input_traced_shard_map_body();

        // The input tracer already belongs to a trace; binding the shard_map through that trace's context composes
        // the staging onto the same builder, attaching the local body as the instruction's `body` region.
        let context = DomainTracingContext::<XlaDomain<'static>>::new();
        let input = context.input(ArrayProgramType::Array(test_array_type()));

        let (operation, body_program) = ShardMapOperation::from_body(body);
        let outputs = context
            .bind(XlaOperation::ShardMap(Box::new(operation)), vec![body_program], std::slice::from_ref(&input))
            .expect("traced shard_map staging should compose onto the input tracer's trace");

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].r#type().into_owned(), ArrayProgramType::Array(test_array_type()));

        let builder = context.builder().borrow();
        assert_eq!(builder.instructions().len(), 1);
        assert!(matches!(builder.instructions()[0].operation(), XlaOperation::ShardMap(_)));
        assert_eq!(builder.instructions()[0].inputs(), &[input.atom_id().unwrap()]);
        assert_eq!(builder.instructions()[0].regions().len(), 1);
    }

    /// Online partial evaluation of a mixed `shard_map` against a live outer trace: the known half of the local body
    /// is rewrapped as a known-side `shard_map` staged into the outer program over the symbolic known input, the
    /// unknown half stays behind a residual `shard_map`, the known→unknown residual edges flow between them, and the
    /// mesh and shardings are threaded onto both boundaries.
    #[test]
    fn test_shard_map_online_partial_evaluation_splits_body_against_a_live_outer_trace() {
        use ryft_core::partial::{PartialEvaluationInput, PartialEvaluationOutput};
        use ryft_core::tracing::TracingContext;

        let array_type = test_array_type();
        let (operation, body_program) =
            ShardMapOperation::<XlaConstant>::from_body(mixed_known_unknown_traced_shard_map_body());

        // Enclosing program staging one `shard_map` over `[a, x]`, with the local body attached as its region.
        let mut builder = XlaProgramBuilder::new();
        let known_input = builder.add_input(array_type.clone());
        let runtime_input = builder.add_input(array_type.clone());
        let body_region = builder.import_program(body_program);
        let outputs = builder
            .add_instruction(
                XlaOperation::ShardMap(Box::new(operation)),
                vec![body_region],
                vec![known_input, runtime_input],
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 3])
            .unwrap();

        let outer = TracingContext::<XlaConstant, XlaOperation>::new();
        let known = outer.input(ArrayProgramType::Array(array_type.clone()));
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(known), PartialValue::Unknown(ArrayProgramType::Array(array_type.clone()))],
            )
            .unwrap();

        // The known half landed in the outer program as one known-side `shard_map` over the symbolic known input,
        // producing the fully known boundary output (`a + a`) plus the residual edge (`a` itself), with the residual
        // edge's boundary derived from its local type.
        {
            let outer_builder = outer.builder().borrow();
            assert_eq!(outer_builder.instructions().len(), 1);
            let known_instruction = &outer_builder.instructions()[0];
            let XlaOperation::ShardMap(known_side) = known_instruction.operation() else {
                panic!("expected the outer program to contain the known-side shard_map");
            };
            let known_body = outer_builder.region_ref(known_instruction.regions()[0]).unwrap().to_program();
            assert_eq!(known_body.input_ids().len(), 1);
            assert_eq!(known_body.output_ids().len(), 2);
            assert_eq!(known_body.instructions().len(), 1);
            assert_eq!(known_side.global_output_types().len(), 2);
            assert_eq!(known_side.shard_map().in_shardings().len(), 1);
            assert_eq!(known_side.shard_map().out_shardings().len(), 2);
        }

        // The unknown half stayed behind one residual `shard_map` over the unknown boundary input plus the residual
        // edge, with the shardings threaded per input.
        assert_eq!(evaluation.program().instructions().len(), 1);
        let residual_instruction = &evaluation.program().instructions()[0];
        let XlaOperation::ShardMap(residual_side) = residual_instruction.operation() else {
            panic!("expected the residual program to contain the residual shard_map");
        };
        let residual_body = evaluation.program().region_ref(residual_instruction.regions()[0]).unwrap().to_program();
        assert_eq!(residual_body.input_ids().len(), 2);
        assert_eq!(residual_body.instructions().len(), 2);
        assert_eq!(residual_side.global_input_types().len(), 2);
        assert_eq!(residual_side.shard_map().in_shardings().len(), 2);
        assert_eq!(residual_side.shard_map().out_shardings().len(), 2);
        assert_eq!(residual_side.global_output_types().len(), 2);

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
