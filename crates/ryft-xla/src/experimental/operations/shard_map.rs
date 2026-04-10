//! Higher-order `shard_map` operations for traced XLA programs.

use std::{
    fmt::{Debug, Display},
    ops::{Add, Mul, Neg},
};

use ryft_core::{
    parameters::{Parameterized, ParameterizedFamily},
    sharding::{LogicalMesh, MeshAxisType, Sharding},
    tracing_v2::{
        AtomId, DifferentiableOp, FloatExt, JitTracer, JvpTracer, LinearTerm, Linearized, MatrixOps, OneLike, Op,
        PrimitiveOp, ProgramBuilder, TraceError, TraceValue, ZeroLike,
    },
    types::{ArrayType, Typed},
};

use crate::experimental::shard_map::{
    FlatTracedShardMap, ShardMap, ShardMapInvocationLeaf, ShardMapLocalTraceInput, ShardMapLocalTraceOutput,
    ShardMapTensor, ShardMapTraceError, ShardMapTracer, TracedShardMap,
};

/// Shared graph type used by erased shard-map bodies.
type FlatShardMapGraph = ryft_core::tracing_v2::Graph<
    ryft_core::tracing_v2::PrimitiveOp<ShardMapTensor>,
    ShardMapTensor,
    Vec<ShardMapTensor>,
    Vec<ShardMapTensor>,
>;

#[derive(Clone)]
struct LinearShardMapBodies {
    pushforward: FlatTracedShardMap,
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

/// Linear execution state carried by one canonical traced shard-map op.
#[derive(Clone)]
struct LinearShardMapState<V: TraceValue> {
    captured_global_primals: Vec<V>,
    eval_mode: LinearShardMapEvalMode,
    transpose_mode: LinearShardMapEvalMode,
}

/// Canonical higher-order shard-map op used for staged tracing, differentiation, and lowering.
#[derive(Clone)]
pub struct ShardMapOp<V: TraceValue> {
    body: FlatTracedShardMap,
    input_types: Vec<ArrayType>,
    output_types: Vec<ArrayType>,
    linear_state: Option<LinearShardMapState<V>>,
}

impl<V: TraceValue> ShardMapOp<V> {
    /// Creates one ordinary staged shard-map op from its erased body payload.
    #[inline]
    pub fn new(body: FlatTracedShardMap) -> Self {
        Self {
            input_types: body.global_input_types.clone(),
            output_types: body.global_output_types.clone(),
            body,
            linear_state: None,
        }
    }

    /// Creates one linear shard-map op with captured primals and explicit transpose state.
    #[inline]
    fn new_linear(
        body: FlatTracedShardMap,
        captured_global_primals: Vec<V>,
        input_types: Vec<ArrayType>,
        output_types: Vec<ArrayType>,
        eval_mode: LinearShardMapEvalMode,
        transpose_mode: LinearShardMapEvalMode,
    ) -> Self {
        Self {
            body,
            input_types,
            output_types,
            linear_state: Some(LinearShardMapState { captured_global_primals, eval_mode, transpose_mode }),
        }
    }

    /// Returns the canonical primal shard-map body carried by this higher-order op.
    #[inline]
    pub fn body(&self) -> &FlatTracedShardMap {
        &self.body
    }

    /// Returns the active linear evaluation mode, if this is a linear shard-map op.
    #[inline]
    pub fn eval_mode(&self) -> Option<&LinearShardMapEvalMode> {
        self.linear_state.as_ref().map(|state| &state.eval_mode)
    }

    /// Returns the transpose evaluation mode, if this is a linear shard-map op.
    #[inline]
    #[cfg(feature = "benchmarking")]
    pub fn transpose_mode(&self) -> Option<&LinearShardMapEvalMode> {
        self.linear_state.as_ref().map(|state| &state.transpose_mode)
    }

    /// Returns `true` when this shard-map op represents one linearized body.
    #[inline]
    pub fn has_linear_state(&self) -> bool {
        self.linear_state.is_some()
    }

    fn transpose_op(&self) -> Result<Self, TraceError> {
        let linear_state = self.linear_state.clone().ok_or(TraceError::HigherOrderOpFailure {
            op: "shard_map",
            message: "transpose requested for a non-linear shard_map op".to_string(),
        })?;
        Ok(Self::new_linear(
            self.body.clone(),
            linear_state.captured_global_primals,
            self.output_types.clone(),
            self.input_types.clone(),
            linear_state.transpose_mode,
            linear_state.eval_mode,
        ))
    }
}

impl<V: TraceValue> Debug for ShardMapOp<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.has_linear_state() { write!(formatter, "LinearShardMap") } else { write!(formatter, "ShardMap") }
    }
}

impl<V: TraceValue> Display for ShardMapOp<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.has_linear_state() { write!(formatter, "linear_shard_map") } else { write!(formatter, "shard_map") }
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

impl Op<ShardMapTensor> for ShardMapOp<ShardMapTensor> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &'static str {
        if self.has_linear_state() { "linear_shard_map" } else { "shard_map" }
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        if inputs.len() != self.input_types.len() {
            return Err(TraceError::InvalidInputCount { expected: self.input_types.len(), got: inputs.len() });
        }
        if !inputs
            .iter()
            .zip(self.input_types.iter())
            .all(|(actual, expected)| shard_map_boundary_types_match(actual, expected))
        {
            return Err(TraceError::IncompatibleAbstractValues { op: self.name() });
        }
        Ok(self.output_types.clone())
    }

    fn eval(&self, inputs: &[ShardMapTensor]) -> Result<Vec<ShardMapTensor>, TraceError> {
        let abstract_inputs = inputs.iter().map(Typed::tpe).collect::<Vec<_>>();
        let _ = self.abstract_eval(abstract_inputs.as_slice())?;
        Ok(self.output_types.iter().cloned().map(ShardMapTensor::new).collect::<Vec<_>>())
    }
}

impl DifferentiableOp<ShardMapTensor> for ShardMapOp<ShardMapTensor> {
    fn replay_linearized_jit(
        &self,
        inputs: Vec<Linearized<ShardMapTracer>>,
    ) -> Result<Vec<Linearized<ShardMapTracer>>, TraceError>
    where
        ShardMapTensor: FloatExt + ZeroLike + OneLike + MatrixOps,
    {
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let primal_values = primal_inputs.iter().map(|input| input.value.clone()).collect::<Vec<_>>();
        let primal_output_values = self.eval(primal_values.as_slice())?;
        let primal_outputs = JitTracer::apply_staged_op(
            primal_inputs.as_slice(),
            PrimitiveOp::Custom(std::sync::Arc::new(self.clone())),
            primal_output_values,
        )?;

        let tangent_inputs = inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
        let tangent_outputs = LinearTerm::apply_staged_op(
            tangent_inputs.as_slice(),
            PrimitiveOp::Custom(std::sync::Arc::new(
                make_replayed_linear_shard_map(self, primal_inputs).map_err(trace_error_from_shard_map)?,
            )),
            self.output_types.len(),
        )?;

        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| Linearized { primal, tangent })
            .collect::<Vec<_>>())
    }

    fn apply_program_jvp_rule(
        &self,
        inputs: &[JvpTracer<ShardMapTensor, LinearTerm<ShardMapTensor>>],
    ) -> Result<Vec<JvpTracer<ShardMapTensor, LinearTerm<ShardMapTensor>>>, TraceError>
    where
        ShardMapTensor: FloatExt + ZeroLike + MatrixOps,
    {
        if self.has_linear_state() {
            return Err(TraceError::HigherOrderOpFailure {
                op: "linearize_program",
                message: "JVP rule for staged op 'linear_shard_map' is not implemented".to_string(),
            });
        }
        apply_staged_shard_map_jvp_rule(self, inputs)
    }

    fn transpose_program_op(
        &self,
        builder: &mut ProgramBuilder<ShardMapTensor>,
        inputs: &[AtomId],
        outputs: &[AtomId],
        output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError>
    where
        ShardMapTensor: FloatExt + ZeroLike + MatrixOps,
    {
        if !self.has_linear_state() {
            return Err(TraceError::HigherOrderOpFailure {
                op: "transpose_linear_program",
                message: "transpose rule for staged op 'shard_map' is not implemented".to_string(),
            });
        }
        if inputs.len() != self.input_types.len() {
            return Err(TraceError::InvalidInputCount { expected: self.input_types.len(), got: inputs.len() });
        }
        if outputs.len() != self.output_types.len() {
            return Err(TraceError::InvalidOutputCount { expected: self.output_types.len(), got: outputs.len() });
        }
        if output_cotangents.len() != self.output_types.len() {
            return Err(TraceError::InvalidInputCount {
                expected: self.output_types.len(),
                got: output_cotangents.len(),
            });
        }
        let contributions = builder.add_equation(
            PrimitiveOp::Custom(std::sync::Arc::new(self.transpose_op()?)),
            output_cotangents.to_vec(),
        )?;
        Ok(contributions.into_iter().map(Some).collect::<Vec<_>>())
    }
}

impl Op<ShardMapTracer> for ShardMapOp<ShardMapTracer> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &'static str {
        if self.has_linear_state() { "linear_shard_map" } else { "shard_map" }
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        if inputs.len() != self.input_types.len() {
            return Err(TraceError::InvalidInputCount { expected: self.input_types.len(), got: inputs.len() });
        }
        if !inputs
            .iter()
            .zip(self.input_types.iter())
            .all(|(actual, expected)| shard_map_boundary_types_match(actual, expected))
        {
            return Err(TraceError::IncompatibleAbstractValues { op: self.name() });
        }
        Ok(self.output_types.clone())
    }

    fn eval(&self, inputs: &[ShardMapTracer]) -> Result<Vec<ShardMapTracer>, TraceError> {
        let abstract_inputs = inputs.iter().map(Typed::tpe).collect::<Vec<_>>();
        let _ = self.abstract_eval(abstract_inputs.as_slice())?;
        match &self.linear_state {
            None => apply_flat_traced_shard_map(self.body.clone(), inputs.to_vec()).map_err(trace_error_from_shard_map),
            Some(linear_state) => match &linear_state.eval_mode {
                LinearShardMapEvalMode::Body(body) => {
                    let combined_inputs = linear_state
                        .captured_global_primals
                        .iter()
                        .cloned()
                        .chain(inputs.iter().cloned())
                        .collect::<Vec<_>>();
                    apply_flat_traced_shard_map(body.clone(), combined_inputs).map_err(trace_error_from_shard_map)
                }
                LinearShardMapEvalMode::FactorizedTranspose(factorized) => {
                    let residuals = apply_flat_traced_shard_map(
                        factorized.residual_body.clone(),
                        linear_state.captured_global_primals.clone(),
                    )
                    .map_err(trace_error_from_shard_map)?;
                    let apply_inputs = inputs.iter().cloned().chain(residuals).collect::<Vec<_>>();
                    apply_flat_traced_shard_map(factorized.apply_body.clone(), apply_inputs)
                        .map_err(trace_error_from_shard_map)
                }
            },
        }
    }
}

impl DifferentiableOp<ShardMapTracer> for ShardMapOp<ShardMapTracer> {
    fn transpose_program_op(
        &self,
        builder: &mut ProgramBuilder<ShardMapTracer>,
        inputs: &[AtomId],
        outputs: &[AtomId],
        output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError>
    where
        ShardMapTracer: FloatExt + ZeroLike + MatrixOps,
    {
        if !self.has_linear_state() {
            return Err(TraceError::HigherOrderOpFailure {
                op: "transpose_linear_program",
                message: "transpose rule for staged op 'shard_map' is not implemented".to_string(),
            });
        }
        if inputs.len() != self.input_types.len() {
            return Err(TraceError::InvalidInputCount { expected: self.input_types.len(), got: inputs.len() });
        }
        if outputs.len() != self.output_types.len() {
            return Err(TraceError::InvalidOutputCount { expected: self.output_types.len(), got: outputs.len() });
        }
        if output_cotangents.len() != self.output_types.len() {
            return Err(TraceError::InvalidInputCount {
                expected: self.output_types.len(),
                got: output_cotangents.len(),
            });
        }
        let contributions = builder.add_equation(
            PrimitiveOp::Custom(std::sync::Arc::new(self.transpose_op()?)),
            output_cotangents.to_vec(),
        )?;
        Ok(contributions.into_iter().map(Some).collect::<Vec<_>>())
    }
}

trait ReplayShardMapValue:
    Clone
    + TraceValue
    + Add<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + FloatExt
    + MatrixOps
    + ZeroLike
    + OneLike
{
    fn lift_constant(constant: &ShardMapTensor, inputs: &[Self]) -> Result<Self, TraceError>;

    fn apply_flat_body(body: FlatTracedShardMap, inputs: Vec<Self>) -> Result<Vec<Self>, ShardMapTraceError>;
}

fn trace_error_from_shard_map(error: ShardMapTraceError) -> TraceError {
    TraceError::HigherOrderOpFailure { op: "shard_map", message: error.to_string() }
}

/// Returns the number of primal inputs consumed by one transpose shard-map body.
fn transpose_body_primal_input_count(body: &FlatTracedShardMap) -> usize {
    body.global_output_types.len()
}

/// Returns the number of cotangent inputs consumed by one transpose shard-map body.
fn transpose_body_cotangent_input_count(body: &FlatTracedShardMap) -> usize {
    body.global_input_types.len() - transpose_body_primal_input_count(body)
}

/// Computes dense owning-equation indices for one flat shard-map graph.
fn equation_by_output(graph: &FlatShardMapGraph) -> Vec<Option<usize>> {
    let mut equation_by_output = vec![None; graph.atom_count()];
    for (equation_index, equation) in graph.equations().iter().enumerate() {
        for output in equation.outputs.iter().copied() {
            equation_by_output[output] = Some(equation_index);
        }
    }
    equation_by_output
}

/// Marks one atom and all of its dependencies as live.
fn mark_live_flat_graph(
    graph: &FlatShardMapGraph,
    atom_id: usize,
    live_atoms: &mut [bool],
    live_equations: &mut [bool],
    equation_by_output: &[Option<usize>],
) {
    if live_atoms[atom_id] {
        return;
    }

    live_atoms[atom_id] = true;
    if let Some(equation_index) = equation_by_output[atom_id] {
        if live_equations[equation_index] {
            return;
        }

        live_equations[equation_index] = true;
        let equation = &graph.equations()[equation_index];
        for input in equation.inputs.iter().copied() {
            mark_live_flat_graph(graph, input, live_atoms, live_equations, equation_by_output);
        }
    }
}

/// Returns live atom/equation masks for one flat shard-map graph.
fn live_sets_for_flat_graph(graph: &FlatShardMapGraph) -> (Vec<bool>, Vec<bool>) {
    let equation_by_output = equation_by_output(graph);
    let mut live_atoms = vec![false; graph.atom_count()];
    let mut live_equations = vec![false; graph.equations().len()];
    for output in graph.outputs().iter().copied() {
        mark_live_flat_graph(
            graph,
            output,
            live_atoms.as_mut_slice(),
            live_equations.as_mut_slice(),
            equation_by_output.as_slice(),
        );
    }
    (live_atoms, live_equations)
}

/// Tracks whether each atom in one transpose body depends on a cotangent input.
fn cotangent_dependencies_for_transpose_body(body: &FlatTracedShardMap) -> Vec<bool> {
    let graph = body.compiled.graph();
    let primal_input_count = transpose_body_primal_input_count(body);
    let mut depends_on_cotangent = vec![false; graph.atom_count()];
    for (input_index, atom_id) in graph.input_atoms().iter().copied().enumerate() {
        depends_on_cotangent[atom_id] = input_index >= primal_input_count;
    }

    for equation in graph.equations() {
        let equation_depends_on_cotangent = equation.inputs.iter().copied().any(|input| depends_on_cotangent[input]);
        for output in equation.outputs.iter().copied() {
            depends_on_cotangent[output] = equation_depends_on_cotangent;
        }
    }
    depends_on_cotangent
}

/// Rebuilds one projected flat shard-map graph over a subset of the original inputs and outputs.
fn project_flat_shard_map_graph(
    graph: &FlatShardMapGraph,
    kept_input_atoms: &[usize],
    output_atoms: &[usize],
) -> Result<FlatShardMapGraph, TraceError> {
    fn remap_atom(
        atom_id: usize,
        graph: &FlatShardMapGraph,
        builder: &mut ProgramBuilder<ShardMapTensor>,
        atom_mapping: &mut std::collections::HashMap<usize, usize>,
        kept_input_atoms: &std::collections::HashMap<usize, usize>,
        equation_by_output: &[Option<usize>],
    ) -> Result<usize, TraceError> {
        if let Some(mapped_atom) = atom_mapping.get(&atom_id) {
            return Ok(*mapped_atom);
        }

        let atom = graph.atom(atom_id).ok_or(TraceError::UnboundAtomId { id: atom_id })?;
        let mapped_atom = match atom.source {
            ryft_core::tracing_v2::AtomSource::Input => *kept_input_atoms.get(&atom_id).ok_or(
                TraceError::InternalInvariantViolation("projected flat shard-map graph referenced a removed input"),
            )?,
            ryft_core::tracing_v2::AtomSource::Constant => builder.add_constant(atom.example_value.clone()),
            ryft_core::tracing_v2::AtomSource::Derived => {
                let equation_index = equation_by_output[atom_id]
                    .ok_or(TraceError::InternalInvariantViolation("derived atom had no owning equation"))?;
                let equation = &graph.equations()[equation_index];
                let remapped_inputs = equation
                    .inputs
                    .iter()
                    .copied()
                    .map(|input| remap_atom(input, graph, builder, atom_mapping, kept_input_atoms, equation_by_output))
                    .collect::<Result<Vec<_>, _>>()?;
                let remapped_outputs = builder.add_equation(equation.op.clone(), remapped_inputs)?;
                for (old_output, new_output) in equation.outputs.iter().copied().zip(remapped_outputs.iter().copied()) {
                    atom_mapping.insert(old_output, new_output);
                }
                *atom_mapping.get(&atom_id).ok_or(TraceError::InternalInvariantViolation(
                    "failed to record projected flat shard-map graph outputs",
                ))?
            }
        };

        atom_mapping.insert(atom_id, mapped_atom);
        Ok(mapped_atom)
    }

    let equation_by_output = equation_by_output(graph);
    let mut builder = ProgramBuilder::<ShardMapTensor>::new();
    let mut input_mapping = std::collections::HashMap::new();
    for atom_id in kept_input_atoms.iter().copied() {
        let atom = graph.atom(atom_id).ok_or(TraceError::UnboundAtomId { id: atom_id })?;
        let mapped_atom = builder.add_input_abstract(atom.abstract_value.clone(), atom.example_value.clone());
        input_mapping.insert(atom_id, mapped_atom);
    }

    let mut atom_mapping = input_mapping.clone();
    let projected_outputs = output_atoms
        .iter()
        .copied()
        .map(|output| {
            remap_atom(output, graph, &mut builder, &mut atom_mapping, &input_mapping, equation_by_output.as_slice())
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(builder.build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(
        projected_outputs,
        vec![ryft_core::parameters::Placeholder; kept_input_atoms.len()],
        vec![ryft_core::parameters::Placeholder; output_atoms.len()],
    ))
}

/// Rebuilds one apply-stage graph whose primal-only dependencies have been replaced by residual inputs.
fn build_factorized_apply_graph(
    body: &FlatTracedShardMap,
    residual_atoms: &[usize],
    depends_on_cotangent: &[bool],
) -> Result<FlatShardMapGraph, TraceError> {
    fn remap_atom(
        atom_id: usize,
        graph: &FlatShardMapGraph,
        builder: &mut ProgramBuilder<ShardMapTensor>,
        atom_mapping: &mut std::collections::HashMap<usize, usize>,
        replacement_inputs: &std::collections::HashMap<usize, usize>,
        depends_on_cotangent: &[bool],
        equation_by_output: &[Option<usize>],
    ) -> Result<usize, TraceError> {
        if let Some(mapped_atom) = atom_mapping.get(&atom_id) {
            return Ok(*mapped_atom);
        }
        if let Some(mapped_input) = replacement_inputs.get(&atom_id) {
            atom_mapping.insert(atom_id, *mapped_input);
            return Ok(*mapped_input);
        }

        let atom = graph.atom(atom_id).ok_or(TraceError::UnboundAtomId { id: atom_id })?;
        let mapped_atom = match atom.source {
            ryft_core::tracing_v2::AtomSource::Input => {
                return Err(TraceError::InternalInvariantViolation(
                    "factorized apply graph referenced a primal input that was not materialized as a residual",
                ));
            }
            ryft_core::tracing_v2::AtomSource::Constant => builder.add_constant(atom.example_value.clone()),
            ryft_core::tracing_v2::AtomSource::Derived => {
                if !depends_on_cotangent[atom_id] {
                    return Err(TraceError::InternalInvariantViolation(
                        "factorized apply graph referenced a cotangent-independent atom that was not materialized as a residual",
                    ));
                }
                let equation_index = equation_by_output[atom_id]
                    .ok_or(TraceError::InternalInvariantViolation("derived atom had no owning equation"))?;
                let equation = &graph.equations()[equation_index];
                let remapped_inputs = equation
                    .inputs
                    .iter()
                    .copied()
                    .map(|input| {
                        remap_atom(
                            input,
                            graph,
                            builder,
                            atom_mapping,
                            replacement_inputs,
                            depends_on_cotangent,
                            equation_by_output,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let remapped_outputs = builder.add_equation(equation.op.clone(), remapped_inputs)?;
                for (old_output, new_output) in equation.outputs.iter().copied().zip(remapped_outputs.iter().copied()) {
                    atom_mapping.insert(old_output, new_output);
                }
                *atom_mapping
                    .get(&atom_id)
                    .ok_or(TraceError::InternalInvariantViolation("failed to record factorized apply graph outputs"))?
            }
        };

        atom_mapping.insert(atom_id, mapped_atom);
        Ok(mapped_atom)
    }

    let graph = body.compiled.graph();
    let primal_input_count = transpose_body_primal_input_count(body);
    let cotangent_input_atoms = graph.input_atoms()[primal_input_count..].to_vec();
    let equation_by_output = equation_by_output(graph);
    let mut builder = ProgramBuilder::<ShardMapTensor>::new();
    let mut replacement_inputs = std::collections::HashMap::new();

    for atom_id in cotangent_input_atoms.iter().copied() {
        let atom = graph.atom(atom_id).ok_or(TraceError::UnboundAtomId { id: atom_id })?;
        let mapped_atom = builder.add_input_abstract(atom.abstract_value.clone(), atom.example_value.clone());
        replacement_inputs.insert(atom_id, mapped_atom);
    }
    for atom_id in residual_atoms.iter().copied() {
        let atom = graph.atom(atom_id).ok_or(TraceError::UnboundAtomId { id: atom_id })?;
        let mapped_atom = builder.add_input_abstract(atom.abstract_value.clone(), atom.example_value.clone());
        replacement_inputs.insert(atom_id, mapped_atom);
    }

    let mut atom_mapping = replacement_inputs.clone();
    let outputs = graph
        .outputs()
        .iter()
        .copied()
        .map(|output| {
            remap_atom(
                output,
                graph,
                &mut builder,
                &mut atom_mapping,
                &replacement_inputs,
                depends_on_cotangent,
                equation_by_output.as_slice(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(builder.build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(
        outputs,
        vec![ryft_core::parameters::Placeholder; cotangent_input_atoms.len() + residual_atoms.len()],
        vec![ryft_core::parameters::Placeholder; graph.outputs().len()],
    ))
}

/// Splits one fused transpose shard-map body into a residual stage and a cotangent-application stage.
fn factorize_transpose_shard_map_body(
    body: &FlatTracedShardMap,
) -> Result<Option<FactorizedTransposeShardMapBodies>, ShardMapTraceError> {
    let simplified_body = body.simplified()?;
    let graph = simplified_body.compiled.graph();
    let primal_input_count = transpose_body_primal_input_count(&simplified_body);
    let cotangent_input_count = transpose_body_cotangent_input_count(&simplified_body);
    if primal_input_count == 0 || cotangent_input_count == 0 {
        return Ok(None);
    }

    let (live_atoms, live_equations) = live_sets_for_flat_graph(graph);
    let depends_on_cotangent = cotangent_dependencies_for_transpose_body(&simplified_body);
    let mut needed_as_residual = vec![false; graph.atom_count()];
    for (equation_index, equation) in graph.equations().iter().enumerate() {
        if !live_equations[equation_index] {
            continue;
        }
        let equation_depends_on_cotangent = equation.outputs.iter().copied().any(|output| depends_on_cotangent[output]);
        if !equation_depends_on_cotangent {
            continue;
        }
        for input in equation.inputs.iter().copied() {
            if live_atoms[input] && !depends_on_cotangent[input] {
                needed_as_residual[input] = true;
            }
        }
    }

    let residual_atoms = (0..graph.atom_count())
        .filter(|atom_id| live_atoms[*atom_id] && !depends_on_cotangent[*atom_id] && needed_as_residual[*atom_id])
        .collect::<Vec<_>>();
    if residual_atoms.is_empty() {
        return Ok(None);
    }

    let residual_out_shardings = residual_atoms
        .iter()
        .map(|atom_id| graph.atom(*atom_id).expect("residual atoms should exist").abstract_value.sharding.clone())
        .collect::<Option<Vec<_>>>();
    let Some(residual_out_shardings) = residual_out_shardings else {
        return Ok(None);
    };

    let primal_input_atoms = graph.input_atoms()[..primal_input_count].to_vec();
    let residual_graph = ryft_core::tracing_v2::Program::from_graph(project_flat_shard_map_graph(
        graph,
        primal_input_atoms.as_slice(),
        residual_atoms.as_slice(),
    )?)
    .simplify()?;
    let residual_local_output_types = residual_atoms
        .iter()
        .map(|atom_id| graph.atom(*atom_id).expect("residual atoms should exist").abstract_value.clone())
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
        ryft_core::tracing_v2::CompiledFunction::from_program(residual_graph),
    );

    let apply_graph = ryft_core::tracing_v2::Program::from_graph(build_factorized_apply_graph(
        &simplified_body,
        residual_atoms.as_slice(),
        depends_on_cotangent.as_slice(),
    )?)
    .simplify()?;
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
        ryft_core::tracing_v2::CompiledFunction::from_program(apply_graph),
    );
    Ok(Some(FactorizedTransposeShardMapBodies { residual_body, apply_body }))
}

/// Builds one linear shard-map op over abstract tensor leaves.
fn make_linear_tensor_shard_map(
    body: &FlatTracedShardMap,
    captured_global_primals: Vec<ShardMapTensor>,
) -> Result<ShardMapOp<ShardMapTensor>, ShardMapTraceError> {
    let linear_bodies = trace_linear_shard_map_bodies(body)?;
    Ok(ShardMapOp::new_linear(
        body.clone(),
        captured_global_primals,
        body.global_input_types.clone(),
        body.global_output_types.clone(),
        LinearShardMapEvalMode::Body(linear_bodies.pushforward),
        LinearShardMapEvalMode::Body(linear_bodies.pullback),
    ))
}

fn apply_staged_shard_map_jvp_rule(
    op: &ShardMapOp<ShardMapTensor>,
    inputs: &[JvpTracer<ShardMapTensor, LinearTerm<ShardMapTensor>>],
) -> Result<Vec<JvpTracer<ShardMapTensor, LinearTerm<ShardMapTensor>>>, TraceError> {
    let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
    let primal_outputs = op.eval(primal_inputs.as_slice())?;
    let tangent_inputs = inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
    let tangent_outputs = LinearTerm::apply_staged_op(
        tangent_inputs.as_slice(),
        PrimitiveOp::Custom(std::sync::Arc::new(
            make_linear_tensor_shard_map(op.body(), primal_inputs).map_err(trace_error_from_shard_map)?,
        )),
        primal_outputs.len(),
    )?;
    Ok(primal_outputs
        .into_iter()
        .zip(tangent_outputs)
        .map(|(primal, tangent)| JvpTracer { primal, tangent })
        .collect::<Vec<_>>())
}

fn make_replayed_linear_shard_map(
    op: &ShardMapOp<ShardMapTensor>,
    captured_global_primals: Vec<ShardMapTracer>,
) -> Result<ShardMapOp<ShardMapTracer>, ShardMapTraceError> {
    match &op.linear_state {
        Some(linear_state) => Ok(ShardMapOp::new_linear(
            op.body.clone(),
            captured_global_primals,
            op.input_types.clone(),
            op.output_types.clone(),
            linear_state.eval_mode.clone(),
            linear_state.transpose_mode.clone(),
        )),
        None => make_linear_shard_map(op.body(), captured_global_primals),
    }
}

fn try_linearize_traced_shard_map_body<
    F: FnOnce(Vec<Linearized<ShardMapTracer>>) -> Result<Vec<Linearized<ShardMapTracer>>, TraceError>,
>(
    function: F,
    primals: Vec<ShardMapTracer>,
) -> Result<
    (Vec<ShardMapTracer>, ryft_core::tracing_v2::LinearProgram<ShardMapTracer, Vec<ShardMapTracer>, Vec<ShardMapTracer>>),
    TraceError,
> {
    let zero = primals.first().map(ZeroLike::zero_like).ok_or(TraceError::EmptyParameterizedValue)?;
    let input_structure = vec![ryft_core::parameters::Placeholder; primals.len()];
    let builder = std::rc::Rc::new(std::cell::RefCell::new(ProgramBuilder::new()));
    let traced_input = primals
        .into_iter()
        .map(|primal| {
            let atom = builder.borrow_mut().add_input(&primal);
            Linearized { primal, tangent: LinearTerm::from_staged_parts(atom, builder.clone()) }
        })
        .collect::<Vec<_>>();
    let traced_output = function(traced_input)?;
    let output_structure = vec![ryft_core::parameters::Placeholder; traced_output.len()];
    let primal_outputs = traced_output.iter().map(|output| output.primal.clone()).collect::<Vec<_>>();
    let tangent_outputs = traced_output.iter().map(|output| output.tangent.atom()).collect::<Vec<_>>();
    drop(traced_output);
    let builder = match std::rc::Rc::try_unwrap(builder) {
        Ok(builder) => builder.into_inner(),
        Err(_) => {
            return Err(TraceError::InternalInvariantViolation("linearization builder escaped the tracing scope"));
        }
    };
    let program = ryft_core::tracing_v2::Program::from_graph(builder.build::<Vec<ShardMapTracer>, Vec<ShardMapTracer>>(
        tangent_outputs,
        input_structure,
        output_structure,
    ))
    .simplify()?;
    Ok((primal_outputs, ryft_core::tracing_v2::LinearProgram::from_program(program, zero)))
}

fn try_transpose_traced_shard_map_body<
    F: FnOnce(Vec<Linearized<ShardMapTracer>>) -> Result<Vec<Linearized<ShardMapTracer>>, TraceError>,
>(
    function: F,
    primals: Vec<ShardMapTracer>,
) -> Result<
    (Vec<ShardMapTracer>, ryft_core::tracing_v2::LinearProgram<ShardMapTracer, Vec<ShardMapTracer>, Vec<ShardMapTracer>>),
    TraceError,
> {
    let (outputs, pushforward) = try_linearize_traced_shard_map_body(function, primals)?;
    Ok((outputs, pushforward.transpose()?))
}

fn apply_flat_traced_shard_map(
    body: FlatTracedShardMap,
    traced_inputs: Vec<ShardMapTracer>,
) -> Result<Vec<ShardMapTracer>, ShardMapTraceError> {
    JitTracer::apply_staged_op(
        traced_inputs.as_slice(),
        PrimitiveOp::Custom(std::sync::Arc::new(ShardMapOp::new(body.clone()))),
        body.global_output_types.iter().cloned().map(ShardMapTensor::new).collect::<Vec<_>>(),
    )
    .map_err(ShardMapTraceError::from)
}

fn replay_traced_xla_graph<
    GraphInput: ryft_core::parameters::Parameterized<ShardMapTensor>,
    GraphOutput: ryft_core::parameters::Parameterized<ShardMapTensor>,
    V: ReplayShardMapValue,
>(
    graph: &ryft_core::tracing_v2::Graph<
        PrimitiveOp<ShardMapTensor>,
        ShardMapTensor,
        GraphInput,
        GraphOutput,
    >,
    inputs: Vec<V>,
) -> Result<Vec<V>, ShardMapTraceError> {
    let mut values = vec![None; graph.atom_count()];
    for (atom_id, value) in graph.input_atoms().iter().copied().zip(inputs.iter().cloned()) {
        values[atom_id] = Some(value);
    }

    let mut equation_by_first_output = vec![None; graph.atom_count()];
    for (equation_index, equation) in graph.equations().iter().enumerate() {
        if let Some(first_output) = equation.outputs.first() {
            equation_by_first_output[*first_output] = Some(equation_index);
        }
    }

    for atom_id in 0..graph.atom_count() {
        let atom = graph.atom(atom_id).expect("atom IDs should be dense");
        match &atom.source {
            ryft_core::tracing_v2::AtomSource::Input => {}
            ryft_core::tracing_v2::AtomSource::Constant => {
                let seed_inputs = inputs.iter().cloned().chain(values.iter().flatten().cloned()).collect::<Vec<_>>();
                if seed_inputs.is_empty() {
                    return Err(ShardMapTraceError::TraceError(TraceError::EmptyParameterizedValue));
                }
                values[atom_id] = Some(V::lift_constant(&atom.example_value, seed_inputs.as_slice())?);
            }
            ryft_core::tracing_v2::AtomSource::Derived => {
                let Some(equation_index) = equation_by_first_output[atom_id] else {
                    continue;
                };
                let equation = &graph.equations()[equation_index];
                let input_values = equation
                    .inputs
                    .iter()
                    .map(|input| values[*input].clone().ok_or(TraceError::UnboundAtomId { id: *input }))
                    .collect::<Result<Vec<_>, _>>()?;
                let outputs = match &equation.op {
                    PrimitiveOp::Custom(custom_op) => {
                        if let Some(shard_map_op) = custom_op.as_any().downcast_ref::<ShardMapOp<ShardMapTensor>>() {
                            if shard_map_op.has_linear_state() {
                                return Err(ShardMapTraceError::TraceError(TraceError::HigherOrderOpFailure {
                                    op: "shard_map",
                                    message: "replaying one linear shard_map body is not supported".to_string(),
                                }));
                            }
                            V::apply_flat_body(shard_map_op.body().clone(), input_values)?
                        } else {
                            return Err(ShardMapTraceError::TraceError(TraceError::HigherOrderOpFailure {
                                op: "shard_map",
                                message: format!("replaying staged op '{}' is not supported", equation.op.name()),
                            }));
                        }
                    }
                    PrimitiveOp::Add => vec![input_values[0].clone() + input_values[1].clone()],
                    PrimitiveOp::Mul => vec![input_values[0].clone() * input_values[1].clone()],
                    PrimitiveOp::Neg => vec![-input_values[0].clone()],
                    PrimitiveOp::Sin => vec![input_values[0].clone().sin()],
                    PrimitiveOp::Cos => vec![input_values[0].clone().cos()],
                    PrimitiveOp::MatMul => vec![input_values[0].clone().matmul(input_values[1].clone())],
                    PrimitiveOp::MatrixTranspose => vec![input_values[0].clone().transpose_matrix()],
                    op => {
                        return Err(ShardMapTraceError::TraceError(TraceError::HigherOrderOpFailure {
                            op: "shard_map",
                            message: format!("replaying staged op '{}' is not supported", op.name()),
                        }));
                    }
                };
                for (output_atom, output_value) in equation.outputs.iter().copied().zip(outputs) {
                    values[output_atom] = Some(output_value);
                }
            }
        }
    }

    graph
        .outputs()
        .iter()
        .map(|output| {
            values[*output]
                .clone()
                .ok_or(ShardMapTraceError::TraceError(TraceError::UnboundAtomId { id: *output }))
        })
        .collect()
}

fn replay_flat_graph<V: ReplayShardMapValue>(
    body: &FlatTracedShardMap,
    inputs: Vec<V>,
) -> Result<Vec<V>, ShardMapTraceError> {
    replay_traced_xla_graph(body.compiled.graph(), inputs)
}

fn make_linear_shard_map(
    body: &FlatTracedShardMap,
    captured_global_primals: Vec<ShardMapTracer>,
) -> Result<ShardMapOp<ShardMapTracer>, ShardMapTraceError> {
    let linear_bodies = trace_linear_shard_map_bodies(body)?;
    let transpose_mode = match factorize_transpose_shard_map_body(&linear_bodies.pullback)? {
        Some(factorized) => LinearShardMapEvalMode::FactorizedTranspose(factorized),
        None => LinearShardMapEvalMode::Body(linear_bodies.pullback.clone()),
    };
    Ok(ShardMapOp::new_linear(
        body.clone(),
        captured_global_primals,
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

    let (_, pushforward_compiled): (
        Vec<ShardMapTensor>,
        ryft_core::tracing_v2::CompiledFunction<ShardMapTensor, Vec<ShardMapTensor>, Vec<ShardMapTensor>>,
    ) = ryft_core::tracing_v2::try_jit(
        {
            let body = body.clone();
            move |combined_inputs: Vec<ShardMapTracer>| -> Result<Vec<ShardMapTracer>, TraceError> {
                let local_primals = combined_inputs[..local_input_count].to_vec();
                let local_tangents = combined_inputs[local_input_count..].to_vec();
                let (_, pushforward_program): (
                    Vec<ShardMapTracer>,
                    ryft_core::tracing_v2::LinearProgram<ShardMapTracer, Vec<ShardMapTracer>, Vec<ShardMapTracer>>,
                ) = try_linearize_traced_shard_map_body(
                    {
                        let body = body.clone();
                        move |replay_inputs: Vec<Linearized<ShardMapTracer>>| {
                            replay_flat_graph(&body, replay_inputs).map_err(trace_error_from_shard_map)
                        }
                    },
                    local_primals,
                )?;
                pushforward_program.call(local_tangents)
            }
        },
        pushforward_local_input_types.iter().cloned().map(ShardMapTensor::new).collect::<Vec<_>>(),
    )?;

    let (_, pullback_compiled): (
        Vec<ShardMapTensor>,
        ryft_core::tracing_v2::CompiledFunction<ShardMapTensor, Vec<ShardMapTensor>, Vec<ShardMapTensor>>,
    ) = ryft_core::tracing_v2::try_jit(
        {
            let body = body.clone();
            move |combined_inputs: Vec<ShardMapTracer>| -> Result<Vec<ShardMapTracer>, TraceError> {
                let local_primals = combined_inputs[..local_input_count].to_vec();
                let local_output_cotangents = combined_inputs[local_input_count..].to_vec();
                let (_, pullback_program): (
                    Vec<ShardMapTracer>,
                    ryft_core::tracing_v2::LinearProgram<ShardMapTracer, Vec<ShardMapTracer>, Vec<ShardMapTracer>>,
                ) = try_transpose_traced_shard_map_body(
                    {
                        let body = body.clone();
                        move |replay_inputs: Vec<Linearized<ShardMapTracer>>| {
                            replay_flat_graph(&body, replay_inputs).map_err(trace_error_from_shard_map)
                        }
                    },
                    local_primals,
                )?;
                pullback_program.call(local_output_cotangents)
            }
        },
        pullback_local_input_types.iter().cloned().map(ShardMapTensor::new).collect::<Vec<_>>(),
    )?;

    Ok(LinearShardMapBodies {
        pushforward: FlatTracedShardMap::from_parts(
            pushforward_shard_map,
            pushforward_global_input_types,
            pushforward_local_input_types,
            body.global_output_types.clone(),
            body.local_output_types.clone(),
            ryft_core::tracing_v2::CompiledFunction::from_graph(
                pushforward_compiled.graph().clone_with_structures::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(
                    vec![ryft_core::parameters::Placeholder; local_input_count * 2],
                    vec![ryft_core::parameters::Placeholder; local_output_count],
                ),
            ),
        ),
        pullback: FlatTracedShardMap::from_parts(
            pullback_shard_map,
            pullback_global_input_types,
            pullback_local_input_types,
            body.global_input_types.clone(),
            body.local_input_types.clone(),
            ryft_core::tracing_v2::CompiledFunction::from_graph(
                pullback_compiled.graph().clone_with_structures::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(
                    vec![ryft_core::parameters::Placeholder; local_input_count + local_output_count],
                    vec![ryft_core::parameters::Placeholder; local_input_count],
                ),
            ),
        ),
    })
}

/// Applies one linearized shard-map body to already-traced values.
pub(crate) fn apply_linearized_flat_shard_map(
    body: FlatTracedShardMap,
    traced_inputs: Vec<Linearized<ShardMapTracer>>,
) -> Result<Vec<Linearized<ShardMapTracer>>, ShardMapTraceError> {
    let traced_primals = traced_inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
    let traced_tangents = traced_inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
    let primal_outputs = apply_flat_traced_shard_map(body.clone(), traced_primals.clone())?;
    let tangent_outputs = LinearTerm::apply_staged_op(
        traced_tangents.as_slice(),
        PrimitiveOp::Custom(std::sync::Arc::new(make_linear_shard_map(&body, traced_primals)?)),
        body.global_output_types.len(),
    )?;
    Ok(primal_outputs
        .into_iter()
        .zip(tangent_outputs)
        .map(|(primal, tangent)| Linearized { primal, tangent })
        .collect::<Vec<_>>())
}

impl ReplayShardMapValue for ShardMapTracer {
    fn lift_constant(constant: &ShardMapTensor, inputs: &[Self]) -> Result<Self, TraceError> {
        let exemplar = inputs.first().ok_or(TraceError::EmptyParameterizedValue)?;
        let builder = exemplar.builder_handle();
        let staging_error = exemplar.staging_error_handle();
        let atom = builder.borrow_mut().add_constant(constant.clone());
        Ok(JitTracer::from_staged_parts(constant.clone(), atom, builder, staging_error))
    }

    fn apply_flat_body(body: FlatTracedShardMap, inputs: Vec<Self>) -> Result<Vec<Self>, ShardMapTraceError> {
        apply_flat_traced_shard_map(body, inputs)
    }
}

impl ReplayShardMapValue for Linearized<ShardMapTracer> {
    fn lift_constant(constant: &ShardMapTensor, inputs: &[Self]) -> Result<Self, TraceError> {
        let exemplar = inputs.first().ok_or(TraceError::EmptyParameterizedValue)?;
        let primal =
            <ShardMapTracer as ReplayShardMapValue>::lift_constant(constant, std::slice::from_ref(&exemplar.primal))?;
        let zero = primal.zero_like();
        let linear_builder = exemplar.tangent.builder_handle();
        let tangent_atom = linear_builder.borrow_mut().add_constant(zero);
        let tangent = LinearTerm::from_staged_parts(tangent_atom, linear_builder);
        Ok(Linearized { primal, tangent })
    }

    fn apply_flat_body(body: FlatTracedShardMap, inputs: Vec<Self>) -> Result<Vec<Self>, ShardMapTraceError> {
        apply_linearized_flat_shard_map(body, inputs)
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
    traced: FlatTracedShardMap,
    traced_inputs: Vec<ShardMapTracer>,
    output_structure: Output::ParameterStructure,
) -> Result<Output, ShardMapTraceError> {
    let staged_outputs = JitTracer::apply_staged_op(
        traced_inputs.as_slice(),
        PrimitiveOp::Custom(std::sync::Arc::new(ShardMapOp::new(traced.clone()))),
        traced.global_output_types.iter().cloned().map(ShardMapTensor::new).collect::<Vec<_>>(),
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
        traced_inputs.parameters().map(Typed::tpe).collect::<Vec<_>>(),
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
        let traced = trace_flat_shard_map::<F, Input::To<ArrayType>, Output>(
            function,
            global_input_types,
            mesh,
            global_in_specs,
            out_specs,
            manual_axes,
            check_vma,
        )?;
        apply_traced_shard_map(traced, traced_inputs, output_structure)
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
        let global_input_primals = Input::To::<ShardMapTracer>::from_parameters(
            input_structure.clone(),
            traced_inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>(),
        )?;
        let global_input_types = Input::To::<ArrayType>::from_parameters(
            input_structure,
            global_input_primals.parameters().map(Typed::tpe).collect::<Vec<_>>(),
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
        let staged_outputs = apply_linearized_flat_shard_map(traced, traced_inputs)?;
        Ok(Output::To::<Linearized<ShardMapTracer>>::from_parameters(output_structure, staged_outputs)?)
    }
}
