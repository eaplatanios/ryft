//! StableHLO/Shardy lowering helpers for traced XLA programs.

use std::collections::HashMap;

#[cfg(feature = "ndarray")]
use ndarray::Array2;
use ryft_mlir::dialects::{func, shardy, stable_hlo};
use ryft_mlir::{
    Attribute, Block, BlockRef, Context as MlirContext, DenseElementsAttributeRef, Location, LocationRef, Operation,
    Region, Size as MlirSize, Type, TypeAndAttributes, TypeRef, Value, ValueRef,
};

use crate::parameters::Parameterized;
use crate::sharding::{LogicalMesh, ShardingError};
use crate::tracing_v2::{
    AtomSource, Graph, ProgramOpRef, StagedOpRef, TraceValue,
    operations::{FlatTracedVMap, LinearShardMapEvalMode, ShardMapOp, VMapOp},
};
use crate::types::{ArrayType, DataType, Shape, Size, Typed};

use super::shard_map::{ShardMap, ShardMapConstantKind, ShardMapError, ShardMapTensor};
use super::sharding::ShardingContext;

/// Error type for StableHLO/Shardy lowering.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum LoweringError {
    /// Underlying shard-map error returned while building manual-computation attributes.
    #[error("{0}")]
    ShardMapError(#[from] ShardMapError),

    /// Underlying sharding error returned while building mesh or sharding attributes.
    #[error("{0}")]
    ShardingError(#[from] ShardingError),

    /// Error returned when a lowered function name is empty or contains whitespace.
    #[error("invalid function name '{function_name}' used during XLA lowering")]
    InvalidFunctionName { function_name: String },

    /// Error returned when lowering encounters a traced tensor type that MLIR rejects.
    #[error("invalid tensor type '{array_type}' used during XLA lowering")]
    InvalidTensorType { array_type: ArrayType },

    /// Error returned when lowering encounters a staged op that does not yet have StableHLO support.
    #[error("unsupported staged op '{op}' during XLA lowering")]
    UnsupportedOp { op: String },

    /// Error returned when lowering encounters a constant value that it does not know how to build.
    #[error("unsupported traced constant at atom %{atom_id} during XLA lowering")]
    UnsupportedConstant { atom_id: usize },

    /// Error returned when lowering encounters a type that does not have StableHLO support yet.
    #[error("unsupported data type '{data_type}' during XLA lowering")]
    UnsupportedDataType { data_type: DataType },

    /// Error returned when lowering needs a staged value that was never assigned.
    #[error("missing lowered value for staged atom %{atom_id}")]
    MissingAtomValue { atom_id: usize },

    /// Error returned when MLIR rejects the constructed dense-elements attribute.
    #[error("invalid dense elements attribute for data type '{data_type}' during XLA lowering")]
    InvalidDenseElementsAttribute { data_type: DataType },

    /// Error returned when the constructed MLIR module fails verification.
    #[error("constructed MLIR module failed verification during XLA lowering")]
    MlirVerificationFailure,

    /// Error returned when one traced XLA graph mixes shard maps from incompatible meshes.
    #[error("traced XLA lowering requires all nested shard maps to use compatible logical meshes")]
    IncompatibleNestedMeshes,

    /// Error returned when simplifying a staged program prior to lowering fails.
    #[error("failed to simplify staged XLA program before lowering: {message}")]
    SimplificationFailure { message: String },
}

/// Lowering mode used for plain `tracing_v2` MLIR emission.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum PlainMlirLoweringMode {
    /// Lower the graph exactly as traced.
    Unpacked,

    /// Lower one packed `vmap` body graph with the provided lane count.
    Packed { lane_count: usize },
}

/// Lowering helper passed to op-owned plain StableHLO lowering hooks.
pub(crate) struct PlainMlirLowerer<'b, 'c: 'b, 't: 'c> {
    /// Owning block receiving the lowered operations.
    pub(crate) block: BlockRef<'b, 'c, 't>,

    /// MLIR context owning the block and created operations.
    pub(crate) context: &'c MlirContext<'t>,

    /// Shared MLIR location used for emitted operations.
    pub(crate) location: LocationRef<'c, 't>,
}

impl<'b, 'c: 'b, 't: 'c> PlainMlirLowerer<'b, 'c, 't> {
    /// Lowers one tensor type inside this lowering context.
    pub(crate) fn lower_tensor_type(
        &self,
        array_type: &ArrayType,
    ) -> Result<ryft_mlir::TensorTypeRef<'c, 't>, LoweringError> {
        lower_tensor_type(array_type, self.context, self.location)
    }

    /// Lowers one plain traced literal value inside this lowering context.
    pub(crate) fn lower_literal_value<V>(&mut self, value: &V) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
    where
        V: MlirLowerableValue,
    {
        lower_literal_value(value, &mut self.block, self.context, self.location)
    }

    /// Lowers one packed literal value inside this lowering context.
    pub(crate) fn lower_packed_literal_value<V>(
        &mut self,
        value: &V,
        packed_output_type: &ArrayType,
    ) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
    where
        V: MlirLowerableValue,
    {
        lower_packed_literal_value(value, packed_output_type, &mut self.block, self.context, self.location)
    }

    /// Lowers one nested `vmap` op inside this lowering context.
    pub(crate) fn lower_vmap<V>(
        &mut self,
        vmap_op: &VMapOp<V>,
        input_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
    where
        V: MlirLowerableValue,
    {
        lower_vmap_results(
            vmap_op.body(),
            lower_vmap_mode(vmap_op),
            input_values,
            &mut self.block,
            self.context,
            self.location,
        )
    }
}

/// Lowering helper passed to op-owned traced XLA MLIR lowering hooks.
pub(crate) struct ShardMapMlirLowerer<'b, 'c: 'b, 't: 'c> {
    /// Owning block receiving the lowered operations.
    pub(crate) block: BlockRef<'b, 'c, 't>,

    /// MLIR context owning the block and created operations.
    pub(crate) context: &'c MlirContext<'t>,

    /// Shared MLIR location used for emitted operations.
    pub(crate) location: LocationRef<'c, 't>,
}

impl<'b, 'c: 'b, 't: 'c> ShardMapMlirLowerer<'b, 'c, 't> {
    /// Lowers one tensor type inside this lowering context.
    pub(crate) fn lower_tensor_type(
        &self,
        array_type: &ArrayType,
    ) -> Result<ryft_mlir::TensorTypeRef<'c, 't>, LoweringError> {
        lower_tensor_type(array_type, self.context, self.location)
    }

    /// Lowers one nested `vmap` op inside this lowering context.
    pub(crate) fn lower_vmap<V>(
        &mut self,
        vmap_op: &VMapOp<V>,
        input_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
    where
        V: MlirLowerableValue,
    {
        lower_vmap_results(
            vmap_op.body(),
            lower_vmap_mode(vmap_op),
            input_values,
            &mut self.block,
            self.context,
            self.location,
        )
    }

    /// Lowers one nested Shardy manual computation operation inside this lowering context.
    pub(crate) fn lower_manual_computation<GraphInput, GraphOutput>(
        &mut self,
        outer_inputs: &[ValueRef<'b, 'c, 't>],
        shard_map: &ShardMap,
        graph: &Graph<StagedOpRef<ShardMapTensor>, ShardMapTensor, GraphInput, GraphOutput>,
        local_input_types: &[ArrayType],
        global_output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
    where
        GraphInput: Parameterized<ShardMapTensor>,
        GraphOutput: Parameterized<ShardMapTensor>,
    {
        lower_manual_computation(
            &mut self.block,
            outer_inputs,
            shard_map,
            graph,
            local_input_types,
            global_output_types,
            self.context,
            self.location,
        )
    }

    /// Lowers one linear shard-map evaluation mode inside this lowering context.
    pub(crate) fn lower_linear_shard_map_eval_mode(
        &mut self,
        eval_mode: &LinearShardMapEvalMode,
        input_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_linear_shard_map_eval_mode(eval_mode, input_values, &mut self.block, self.context, self.location)
    }
}

/// Lowers a traced shard-map program to a textual StableHLO/Shardy MLIR module.
pub(crate) fn to_mlir_module<Input, Output, GraphInput, GraphOutput, S>(
    shard_map: &ShardMap,
    graph: &Graph<StagedOpRef<ShardMapTensor>, ShardMapTensor, GraphInput, GraphOutput>,
    global_input_types: &Input,
    local_input_types: &Input,
    global_output_types: &Output,
    _local_output_types: &Output,
    function_name: S,
) -> Result<String, LoweringError>
where
    Input: Parameterized<ArrayType>,
    Output: Parameterized<ArrayType>,
    GraphInput: Parameterized<ShardMapTensor>,
    GraphOutput: Parameterized<ShardMapTensor>,
    S: AsRef<str>,
{
    let function_name = normalize_function_name(function_name.as_ref())?;
    let global_input_types = global_input_types.parameters().cloned().collect::<Vec<_>>();
    let local_input_types = local_input_types.parameters().cloned().collect::<Vec<_>>();
    let global_output_types = global_output_types.parameters().cloned().collect::<Vec<_>>();

    let context = MlirContext::new();
    let location = context.unknown_location();
    let module = context.module(location);

    let global_input_tensor_types = global_input_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, &context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let global_output_tensor_types = global_output_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, &context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let mesh_operation = shard_map.mesh().to_shardy_mesh(location);
    module.body().append_operation(mesh_operation);

    let function_arguments = global_input_tensor_types
        .iter()
        .zip(shard_map.in_shardings().iter())
        .map(|(tensor_type, sharding)| {
            let sharding = sharding.to_shardy_tensor_sharding(&context, ShardingContext::ExplicitSharding);
            Ok(TypeAndAttributes {
                r#type: tensor_type.as_ref(),
                attributes: Some(HashMap::from([("sdy.sharding".into(), sharding.as_ref())])),
            })
        })
        .collect::<Result<Vec<_>, LoweringError>>()?;
    let function_results = global_output_tensor_types
        .iter()
        .zip(shard_map.out_shardings().iter())
        .map(|(tensor_type, sharding)| {
            let sharding = sharding.to_shardy_tensor_sharding(&context, ShardingContext::ExplicitSharding);
            Ok(TypeAndAttributes {
                r#type: tensor_type.as_ref(),
                attributes: Some(HashMap::from([("sdy.sharding".into(), sharding.as_ref())])),
            })
        })
        .collect::<Result<Vec<_>, LoweringError>>()?;

    module.body().append_operation({
        let function_block = context.block(
            global_input_tensor_types
                .iter()
                .map(|tensor_type| (*tensor_type, location))
                .collect::<Vec<_>>()
                .as_slice(),
        );
        let outer_inputs = (0..global_input_tensor_types.len())
            .map(|index| function_block.argument(index).expect("function block arguments should exist").as_ref())
            .collect::<Vec<_>>();
        let mut function_block_ref = function_block.as_ref();
        let manual_results = lower_manual_computation(
            &mut function_block_ref,
            outer_inputs.as_slice(),
            shard_map,
            graph,
            local_input_types.as_slice(),
            global_output_types.as_slice(),
            &context,
            location.as_ref(),
        )?;
        function_block_ref.append_operation(func::r#return(manual_results.as_slice(), location));

        func::func(
            function_name.as_str(),
            func::FuncAttributes { arguments: function_arguments, results: function_results, ..Default::default() },
            function_block.into(),
            location,
        )
    });

    if !module.verify() {
        return Err(LoweringError::MlirVerificationFailure);
    }

    Ok(module.to_string())
}

/// Lowers an arbitrary traced XLA graph to a textual StableHLO/Shardy MLIR module.
pub(crate) fn to_mlir_module_for_graph<Input, Output, GraphInput, GraphOutput, S>(
    graph: &Graph<StagedOpRef<ShardMapTensor>, ShardMapTensor, GraphInput, GraphOutput>,
    global_input_types: &Input,
    global_output_types: &Output,
    function_name: S,
) -> Result<String, LoweringError>
where
    Input: Parameterized<ArrayType>,
    Output: Parameterized<ArrayType>,
    GraphInput: Parameterized<ShardMapTensor>,
    GraphOutput: Parameterized<ShardMapTensor>,
    S: AsRef<str>,
{
    let function_name = normalize_function_name(function_name.as_ref())?;
    let global_input_types = global_input_types.parameters().cloned().collect::<Vec<_>>();
    let global_output_types = global_output_types.parameters().cloned().collect::<Vec<_>>();

    let context = MlirContext::new();
    let location = context.unknown_location();
    let module = context.module(location);

    if let Some(mesh) = collect_nested_shard_map_mesh(graph, None)? {
        let mesh_operation = mesh.to_shardy_mesh(location);
        module.body().append_operation(mesh_operation);
    }

    let global_input_tensor_types = global_input_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, &context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let global_output_tensor_types = global_output_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, &context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let function_arguments = global_input_tensor_types
        .iter()
        .map(|tensor_type| TypeAndAttributes { r#type: tensor_type.as_ref(), attributes: None })
        .collect::<Vec<_>>();
    let function_results = global_output_tensor_types
        .iter()
        .map(|tensor_type| TypeAndAttributes { r#type: tensor_type.as_ref(), attributes: None })
        .collect::<Vec<_>>();

    module.body().append_operation({
        let function_block = context.block(
            global_input_tensor_types
                .iter()
                .map(|tensor_type| (*tensor_type, location))
                .collect::<Vec<_>>()
                .as_slice(),
        );
        {
            let mut function_block_ref = function_block.as_ref();
            let outputs = lower_graph_outputs(graph, &mut function_block_ref, &context, location.as_ref())?;
            function_block_ref.append_operation(func::r#return(outputs.as_slice(), location));
        }
        func::func(
            function_name.as_str(),
            func::FuncAttributes { arguments: function_arguments, results: function_results, ..Default::default() },
            function_block.into(),
            location,
        )
    });

    if !module.verify() {
        return Err(LoweringError::MlirVerificationFailure);
    }
    Ok(module.to_string())
}

/// Value type that can be materialized as a StableHLO dense constant during benchmark lowering.
pub(crate) trait MlirLowerableValue: TraceValue + Typed<ArrayType> + Clone + 'static {
    /// Builds a dense-elements attribute containing this value.
    fn to_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError>;

    /// Builds a scalar dense-elements attribute when this value can be represented as a scalar splat.
    #[inline]
    fn to_scalar_dense_elements_attribute<'c, 't>(
        &self,
        _tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        _context: &'c MlirContext<'t>,
    ) -> Result<Option<DenseElementsAttributeRef<'c, 't>>, LoweringError> {
        Ok(None)
    }
}

impl MlirLowerableValue for f64 {
    fn to_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError> {
        context
            .dense_f64_elements_attribute(tensor_type, std::slice::from_ref(self))
            .and_then(|attribute| attribute.cast::<DenseElementsAttributeRef>())
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type: DataType::F64 })
    }

    fn to_scalar_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<Option<DenseElementsAttributeRef<'c, 't>>, LoweringError> {
        Ok(Some(self.to_dense_elements_attribute(tensor_type, context)?))
    }
}

#[cfg(feature = "ndarray")]
impl MlirLowerableValue for Array2<f64> {
    fn to_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError> {
        let elements = self.iter().copied().collect::<Vec<_>>();
        context
            .dense_f64_elements_attribute(tensor_type, elements.as_slice())
            .and_then(|attribute| attribute.cast::<DenseElementsAttributeRef>())
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type: DataType::F64 })
    }

    fn to_scalar_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<Option<DenseElementsAttributeRef<'c, 't>>, LoweringError> {
        if self.shape() == [1, 1] {
            return Ok(Some(
                context
                    .dense_f64_elements_attribute(tensor_type, std::slice::from_ref(&self[(0, 0)]))
                    .and_then(|attribute| attribute.cast::<DenseElementsAttributeRef>())
                    .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type: DataType::F64 })?,
            ));
        }
        Ok(None)
    }
}

impl MlirLowerableValue for ShardMapTensor {
    fn to_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError> {
        let constant_kind = self.constant_kind().ok_or(LoweringError::UnsupportedConstant { atom_id: 0 })?;
        lower_constant_elements_attribute(self.r#type().data_type, tensor_type, constant_kind, context)
    }

    fn to_scalar_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<Option<DenseElementsAttributeRef<'c, 't>>, LoweringError> {
        let Some(constant_kind) = self.constant_kind() else {
            return Ok(None);
        };
        Ok(Some(lower_constant_elements_attribute(self.r#type().data_type, tensor_type, constant_kind, context)?))
    }
}

/// Lowers a plain traced `tracing_v2` graph to a textual StableHLO MLIR module.
#[cfg(any(test, feature = "benchmarking"))]
#[allow(dead_code)]
pub(crate) fn to_mlir_module_for_plain_graph<V, Input, Output, S>(
    graph: &Graph<ProgramOpRef<V>, V, Input, Output>,
    function_name: S,
) -> Result<String, LoweringError>
where
    V: MlirLowerableValue,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
    S: AsRef<str>,
{
    let function_name = normalize_function_name(function_name.as_ref())?;
    let context = MlirContext::new();
    let location = context.unknown_location();
    let module = context.module(location);

    let input_tensor_types = graph
        .input_atoms()
        .iter()
        .map(|atom_id| {
            let input_atom = graph.atom(*atom_id).expect("graph input atoms should exist");
            lower_tensor_type(&input_atom.abstract_value, &context, location)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let output_tensor_types = graph
        .outputs()
        .iter()
        .map(|atom_id| {
            let output_atom = graph.atom(*atom_id).expect("graph output atoms should exist");
            lower_tensor_type(&output_atom.abstract_value, &context, location)
        })
        .collect::<Result<Vec<_>, _>>()?;

    module.body().append_operation({
        let function_block = context.block(
            input_tensor_types.iter().map(|tensor_type| (*tensor_type, location)).collect::<Vec<_>>().as_slice(),
        );
        {
            let mut function_block_ref = function_block.as_ref();
            let outputs = lower_plain_graph_outputs(graph, &mut function_block_ref, &context, location.as_ref())?;
            function_block_ref.append_operation(func::r#return(outputs.as_slice(), location));
        }
        func::func(
            function_name.as_str(),
            func::FuncAttributes {
                arguments: input_tensor_types
                    .iter()
                    .map(|tensor_type| TypeAndAttributes { r#type: tensor_type.as_ref(), attributes: None })
                    .collect(),
                results: output_tensor_types
                    .iter()
                    .map(|tensor_type| TypeAndAttributes { r#type: tensor_type.as_ref(), attributes: None })
                    .collect(),
                ..Default::default()
            },
            function_block.into(),
            location,
        )
    });

    if !module.verify() {
        return Err(LoweringError::MlirVerificationFailure);
    }

    Ok(module.to_string())
}

fn collect_nested_shard_map_mesh<GraphInput, GraphOutput>(
    graph: &Graph<StagedOpRef<ShardMapTensor>, ShardMapTensor, GraphInput, GraphOutput>,
    existing: Option<LogicalMesh>,
) -> Result<Option<LogicalMesh>, LoweringError>
where
    GraphInput: Parameterized<ShardMapTensor>,
    GraphOutput: Parameterized<ShardMapTensor>,
{
    let mut mesh = existing;
    for equation in graph.equations() {
        if let Some(shard_map_op) = equation.op.as_any().downcast_ref::<ShardMapOp<ShardMapTensor>>() {
            if let Some(eval_mode) = shard_map_op.eval_mode() {
                mesh = collect_nested_linear_shard_map_mesh(eval_mode, mesh)?;
            } else {
                mesh = Some(match mesh.take() {
                    Some(existing_mesh) => merge_logical_meshes(&existing_mesh, shard_map_op.body().shard_map.mesh())?,
                    None => shard_map_op.body().shard_map.mesh().clone(),
                });
                mesh = collect_nested_shard_map_mesh(shard_map_op.body().compiled.graph(), mesh)?;
            }
        }
    }
    Ok(mesh)
}

/// Collects nested logical meshes referenced by one linear shard-map evaluation mode.
fn collect_nested_linear_shard_map_mesh(
    eval_mode: &LinearShardMapEvalMode,
    existing: Option<LogicalMesh>,
) -> Result<Option<LogicalMesh>, LoweringError> {
    match eval_mode {
        LinearShardMapEvalMode::Body(body) => {
            let mesh = Some(match existing {
                Some(existing_mesh) => merge_logical_meshes(&existing_mesh, body.shard_map.mesh())?,
                None => body.shard_map.mesh().clone(),
            });
            collect_nested_shard_map_mesh(body.compiled.graph(), mesh)
        }
        LinearShardMapEvalMode::FactorizedTranspose(factorized) => {
            let mesh = collect_nested_shard_map_mesh(factorized.residual_body.compiled.graph(), existing)?;
            collect_nested_shard_map_mesh(factorized.apply_body.compiled.graph(), mesh)
        }
    }
}

fn merge_logical_meshes(existing: &LogicalMesh, incoming: &LogicalMesh) -> Result<LogicalMesh, LoweringError> {
    let mut merged_axes = existing.axes.clone();
    for incoming_axis in &incoming.axes {
        match existing.axis_size(incoming_axis.name.as_str()) {
            Some(existing_size) if existing_size != incoming_axis.size => {
                return Err(LoweringError::IncompatibleNestedMeshes);
            }
            Some(_) => {}
            None => merged_axes.push(incoming_axis.clone()),
        }
    }
    LogicalMesh::new(merged_axes).map_err(LoweringError::from)
}

/// Controls how traced `vmap` bodies are packed and unpacked during MLIR lowering.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum VMapLoweringMode {
    /// Lower ordinary staged `vmap` by packing lanes with broadcast-plus-concatenate and unpacking with reshape.
    Forward,

    /// Lower transposed linear `vmap` using the transpose-friendly pad-and-reduce structure JAX emits.
    Transpose,
}

/// Maps the canonical traced `vmap` op to the lowering-specific packing mode.
fn lower_vmap_mode<V>(vmap_op: &VMapOp<V>) -> VMapLoweringMode
where
    V: TraceValue,
{
    if vmap_op.has_transpose_body() { VMapLoweringMode::Transpose } else { VMapLoweringMode::Forward }
}

/// Returns the static dimensions for one tensor type.
fn static_dimensions(array_type: &ArrayType) -> Result<Vec<usize>, LoweringError> {
    array_type
        .shape
        .dimensions
        .iter()
        .map(|size| match size {
            Size::Static(value) => Ok(*value),
            Size::Dynamic(_) => Err(LoweringError::InvalidTensorType { array_type: array_type.clone() }),
        })
        .collect()
}

/// Returns the tensor type obtained by prepending one leading batch dimension.
fn packed_array_type(array_type: &ArrayType, lane_count: usize) -> ArrayType {
    let mut dimensions = Vec::with_capacity(array_type.shape.dimensions.len() + 1);
    dimensions.push(Size::Static(lane_count));
    dimensions.extend(array_type.shape.dimensions.iter().cloned());
    ArrayType::new(array_type.data_type, Shape::new(dimensions), None)
}

/// Returns the tensor type obtained by prepending a leading axis of size one.
fn singleton_packed_array_type(array_type: &ArrayType) -> ArrayType {
    packed_array_type(array_type, 1)
}

/// Returns the broadcast dimensions used when prepending one leading axis without permuting existing axes.
fn leading_axis_broadcast_dimensions(rank: usize) -> Vec<usize> {
    (1..=rank).collect::<Vec<_>>()
}

/// Returns the start indices for slicing one packed lane.
fn packed_lane_start_indices(rank: usize, lane_index: usize) -> Vec<usize> {
    std::iter::once(lane_index).chain(std::iter::repeat_n(0, rank)).collect::<Vec<_>>()
}

/// Returns the limit indices for slicing one packed lane.
fn packed_lane_limit_indices(array_type: &ArrayType, lane_index: usize) -> Result<Vec<usize>, LoweringError> {
    Ok(std::iter::once(lane_index + 1).chain(static_dimensions(array_type)?).collect::<Vec<_>>())
}

/// Lowers one scalar constant for the provided data type and constant kind.
fn lower_scalar_constant<'b, 'c: 'b, 't: 'c, B, L>(
    data_type: DataType,
    constant_kind: ShardMapConstantKind,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    L: Location<'c, 't> + Copy,
{
    let scalar_tensor_type =
        context.tensor_type(lower_element_type(data_type, context)?, &[], None, location).ok_or_else(|| {
            LoweringError::InvalidTensorType { array_type: ArrayType::new(data_type, Shape::new(Vec::new()), None) }
        })?;
    let elements = lower_constant_elements_attribute(data_type, scalar_tensor_type, constant_kind, context)?;
    let constant = block.append_operation(stable_hlo::constant(elements, location));
    Ok(constant.result(0).expect("stablehlo.constant should return one result").as_ref())
}

/// Lowers one zero constant matching the provided tensor type.
fn lower_zero_for_array_type<'b, 'c: 'b, 't: 'c, B, L>(
    array_type: &ArrayType,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    L: Location<'c, 't> + Copy,
{
    let scalar_zero =
        lower_scalar_constant(array_type.data_type, ShardMapConstantKind::Zero, block, context, location)?;
    if array_type.shape.dimensions.is_empty() {
        return Ok(scalar_zero);
    }

    let tensor_type = lower_tensor_type(array_type, context, location)?;
    let broadcast = block.append_operation(stable_hlo::broadcast(scalar_zero, tensor_type, &[], location));
    Ok(broadcast.result(0).expect("stablehlo.broadcast should return one result").as_ref())
}

/// Lowers one literal value and broadcasts it to a packed tensor type when needed.
fn lower_packed_literal_value<'b, 'c: 'b, 't: 'c, B, V, L>(
    value: &V,
    packed_type: &ArrayType,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    V: MlirLowerableValue,
    L: Location<'c, 't> + Copy,
{
    let lowered_value = lower_literal_value(value, block, context, location)?;
    if value.tpe() == *packed_type {
        return Ok(lowered_value);
    }

    let lane_count = match packed_type.shape.dimensions.first() {
        Some(Size::Static(value)) => *value,
        _ => return Err(LoweringError::InvalidTensorType { array_type: packed_type.clone() }),
    };
    if packed_array_type(&value.tpe(), lane_count) != *packed_type {
        return Err(LoweringError::InvalidTensorType { array_type: packed_type.clone() });
    }

    if !value.tpe().shape.dimensions.is_empty() {
        let singleton_tensor_type = lower_tensor_type(&singleton_packed_array_type(&value.tpe()), context, location)?;
        let singleton = block.append_operation(stable_hlo::broadcast(
            lowered_value,
            singleton_tensor_type,
            leading_axis_broadcast_dimensions(value.tpe().shape.dimensions.len()).as_slice(),
            location,
        ));
        let singleton_value = singleton.result(0).expect("stablehlo.broadcast should return one result").as_ref();
        if lane_count == 1 {
            return Ok(singleton_value);
        }

        let tensor_type = lower_tensor_type(packed_type, context, location)?;
        let broadcast_dimensions = (0..=value.tpe().shape.dimensions.len()).collect::<Vec<_>>();
        let packed = block.append_operation(stable_hlo::broadcast(
            singleton_value,
            tensor_type,
            broadcast_dimensions.as_slice(),
            location,
        ));
        return Ok(packed.result(0).expect("stablehlo.broadcast should return one result").as_ref());
    }

    let tensor_type = lower_tensor_type(packed_type, context, location)?;
    let broadcast = block.append_operation(stable_hlo::broadcast(
        lowered_value,
        tensor_type,
        leading_axis_broadcast_dimensions(value.tpe().shape.dimensions.len()).as_slice(),
        location,
    ));
    Ok(broadcast.result(0).expect("stablehlo.broadcast should return one result").as_ref())
}

/// Packs one set of per-lane inputs into one leading-axis batched tensor using forward `vmap` semantics.
fn lower_vmap_forward_pack<'b, 'c: 'b, 't: 'c, B, L>(
    lane_values: &[ValueRef<'b, 'c, 't>],
    lane_type: &ArrayType,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    L: Location<'c, 't> + Copy,
{
    debug_assert!(!lane_values.is_empty());
    let singleton_tensor_type = lower_tensor_type(&singleton_packed_array_type(lane_type), context, location)?;
    let broadcast_dimensions = leading_axis_broadcast_dimensions(lane_type.shape.dimensions.len());
    let packed_lanes = lane_values
        .iter()
        .map(|lane_value| {
            let broadcast = block.append_operation(stable_hlo::broadcast(
                *lane_value,
                singleton_tensor_type,
                broadcast_dimensions.as_slice(),
                location,
            ));
            Ok(broadcast.result(0).expect("stablehlo.broadcast should return one result").as_ref())
        })
        .collect::<Result<Vec<_>, LoweringError>>()?;
    if packed_lanes.len() == 1 {
        return Ok(packed_lanes[0]);
    }
    let concatenate = block.append_operation(stable_hlo::concatenate(packed_lanes.as_slice(), 0, location));
    Ok(concatenate.result(0).expect("stablehlo.concatenate should return one result").as_ref())
}

/// Packs one set of per-lane cotangents into one leading-axis batched tensor using transpose-friendly padding.
fn lower_vmap_transpose_pack<'b, 'c: 'b, 't: 'c, B, L>(
    lane_values: &[ValueRef<'b, 'c, 't>],
    lane_type: &ArrayType,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    L: Location<'c, 't> + Copy,
{
    debug_assert!(!lane_values.is_empty());
    let rank = lane_type.shape.dimensions.len();
    let singleton_tensor_type = lower_tensor_type(&singleton_packed_array_type(lane_type), context, location)?;
    let broadcast_dimensions = leading_axis_broadcast_dimensions(rank);
    let padding_value =
        lower_scalar_constant(lane_type.data_type, ShardMapConstantKind::Zero, block, context, location)?;
    let mut padded_lanes = Vec::with_capacity(lane_values.len());
    for (lane_index, lane_value) in lane_values.iter().enumerate() {
        let singleton = block.append_operation(stable_hlo::broadcast(
            *lane_value,
            singleton_tensor_type,
            broadcast_dimensions.as_slice(),
            location,
        ));
        let edge_padding_low =
            std::iter::once(lane_index as i64).chain(std::iter::repeat_n(0, rank)).collect::<Vec<_>>();
        let edge_padding_high = std::iter::once((lane_values.len() - lane_index - 1) as i64)
            .chain(std::iter::repeat_n(0, rank))
            .collect::<Vec<_>>();
        let interior_padding = std::iter::repeat_n(0usize, rank + 1).collect::<Vec<_>>();
        let padded = block.append_operation(stable_hlo::pad(
            singleton.result(0).expect("stablehlo.broadcast should return one result").as_ref(),
            padding_value,
            edge_padding_low.as_slice(),
            edge_padding_high.as_slice(),
            interior_padding.as_slice(),
            location,
        ));
        padded_lanes.push(padded.result(0).expect("stablehlo.pad should return one result").as_ref());
    }

    let mut accumulated = padded_lanes[0];
    for padded_lane in padded_lanes.into_iter().skip(1) {
        let add = block.append_operation(stable_hlo::add(accumulated, padded_lane, location));
        accumulated = add.result(0).expect("stablehlo.add should return one result").as_ref();
    }
    Ok(accumulated)
}

/// Unpacks one leading-axis batched tensor into one result per lane using forward `vmap` semantics.
fn lower_vmap_forward_unpack<'b, 'c: 'b, 't: 'c, B, L>(
    packed_value: ValueRef<'b, 'c, 't>,
    lane_type: &ArrayType,
    lane_count: usize,
    block: &mut B,
    _context: &'c MlirContext<'t>,
    location: L,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    L: Location<'c, 't> + Copy,
{
    let rank = lane_type.shape.dimensions.len();
    let lane_shape = static_dimensions(lane_type)?;
    let strides = std::iter::repeat_n(1usize, rank + 1).collect::<Vec<_>>();
    (0..lane_count)
        .map(|lane_index| {
            let slice = block.append_operation(stable_hlo::slice(
                packed_value,
                packed_lane_start_indices(rank, lane_index).as_slice(),
                packed_lane_limit_indices(lane_type, lane_index)?.as_slice(),
                strides.as_slice(),
                location,
            ));
            let reshape = block.append_operation(stable_hlo::reshape(
                slice.result(0).expect("stablehlo.slice should return one result").as_ref(),
                lane_shape.as_slice(),
                location,
            ));
            Ok(reshape.result(0).expect("stablehlo.reshape should return one result").as_ref())
        })
        .collect()
}

/// Unpacks one leading-axis batched tensor into one result per lane using transpose-friendly reductions.
fn lower_vmap_transpose_unpack<'b, 'c: 'b, 't: 'c, B, L>(
    packed_value: ValueRef<'b, 'c, 't>,
    lane_type: &ArrayType,
    lane_count: usize,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    L: Location<'c, 't> + Copy,
{
    let rank = lane_type.shape.dimensions.len();
    let strides = std::iter::repeat_n(1usize, rank + 1).collect::<Vec<_>>();
    let reduction_type = lower_tensor_type(lane_type, context, location)?;
    (0..lane_count)
        .map(|lane_index| {
            let slice = block.append_operation(stable_hlo::slice(
                packed_value,
                packed_lane_start_indices(rank, lane_index).as_slice(),
                packed_lane_limit_indices(lane_type, lane_index)?.as_slice(),
                strides.as_slice(),
                location,
            ));
            let zero = lower_zero_for_array_type(lane_type, block, context, location)?;
            let mut computation = context.region();
            let mut computation_block = context.block(&[(reduction_type, location), (reduction_type, location)]);
            let add = computation_block.append_operation(stable_hlo::add(
                computation_block.argument(0).expect("reduction lhs should exist").as_ref(),
                computation_block.argument(1).expect("reduction rhs should exist").as_ref(),
                location,
            ));
            computation_block.append_operation(stable_hlo::r#return(
                &[add.result(0).expect("stablehlo.add should return one result").as_ref()],
                location,
            ));
            computation.append_block(computation_block);
            let reduce = block.append_operation(stable_hlo::reduce(
                &[slice.result(0).expect("stablehlo.slice should return one result").as_ref()],
                &[zero],
                &[0],
                computation.into(),
                location,
            ));
            Ok(reduce.result(0).expect("stablehlo.reduce should return one result").as_ref())
        })
        .collect()
}

/// Lowers one packed `vmap` body graph whose inputs and outputs already carry a leading batch axis.
fn lower_packed_program_outputs<'b, 'c: 'b, 't: 'c, B, V, L>(
    graph: &Graph<ProgramOpRef<V>, V, Vec<V>, Vec<V>>,
    packed_inputs: &[ValueRef<'b, 'c, 't>],
    lane_count: usize,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    V: MlirLowerableValue,
    L: Location<'c, 't> + Copy,
{
    fn resolve_packed_atom_value<'b, 'c: 'b, 't: 'c, B, V, L>(
        graph: &Graph<ProgramOpRef<V>, V, Vec<V>, Vec<V>>,
        atom_values: &[Option<ValueRef<'b, 'c, 't>>],
        atom_id: usize,
        lane_count: usize,
        block: &mut B,
        context: &'c MlirContext<'t>,
        location: L,
    ) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
    where
        B: Block<'b, 'c, 't>,
        V: MlirLowerableValue,
        L: Location<'c, 't> + Copy,
    {
        if let Some(value) = atom_values[atom_id] {
            return Ok(value);
        }

        let atom = graph.atom(atom_id).ok_or(LoweringError::MissingAtomValue { atom_id })?;
        match &atom.source {
            AtomSource::Constant => lower_packed_literal_value(
                &atom.example_value,
                &packed_array_type(&atom.abstract_value, lane_count),
                block,
                context,
                location,
            ),
            _ => Err(LoweringError::MissingAtomValue { atom_id }),
        }
    }

    let mut atom_values = vec![None; graph.atom_count()];
    for (input_index, atom_id) in graph.input_atoms().iter().copied().enumerate() {
        atom_values[atom_id] = Some(packed_inputs[input_index]);
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
            AtomSource::Input => {}
            AtomSource::Constant => {}
            AtomSource::Derived => {
                let Some(equation_index) = equation_by_first_output[atom_id] else {
                    continue;
                };
                let equation = &graph.equations()[equation_index];
                let inputs = equation
                    .inputs
                    .iter()
                    .map(|input| {
                        resolve_packed_atom_value(
                            graph,
                            atom_values.as_slice(),
                            *input,
                            lane_count,
                            block,
                            context,
                            location,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let mut block_ref = block.as_ref();
                let lowered_outputs = lower_packed_plain_equation(
                    graph,
                    equation_index,
                    inputs.as_slice(),
                    lane_count,
                    &mut block_ref,
                    context,
                    location.as_ref(),
                )?;
                for (output_atom, lowered_output) in equation.outputs.iter().copied().zip(lowered_outputs.into_iter()) {
                    atom_values[output_atom] = Some(lowered_output);
                }
            }
        }
    }

    graph
        .outputs()
        .iter()
        .map(|output| {
            resolve_packed_atom_value(graph, atom_values.as_slice(), *output, lane_count, block, context, location)
        })
        .collect::<Result<Vec<_>, _>>()
}

/// Lowers one higher-order `vmap` op by explicitly packing inputs, lowering the packed body, and unpacking outputs.
fn lower_vmap_results<'b, 'c: 'b, 't: 'c, B, V, L>(
    body: &FlatTracedVMap<V>,
    mode: VMapLoweringMode,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    V: MlirLowerableValue,
    L: Location<'c, 't> + Copy,
{
    let lane_count = body.lane_count();
    let logical_input_count = body.input_types().len();
    let logical_output_count = body.output_types().len();
    debug_assert_eq!(input_values.len(), body.total_input_count());

    let packed_inputs = (0..logical_input_count)
        .map(|input_index| {
            let lanes = input_values.chunks(logical_input_count).map(|chunk| chunk[input_index]).collect::<Vec<_>>();
            match mode {
                VMapLoweringMode::Forward => lower_vmap_forward_pack(
                    lanes.as_slice(),
                    &body.input_types()[input_index],
                    block,
                    context,
                    location,
                ),
                VMapLoweringMode::Transpose => lower_vmap_transpose_pack(
                    lanes.as_slice(),
                    &body.input_types()[input_index],
                    block,
                    context,
                    location,
                ),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    let packed_outputs = lower_packed_program_outputs(
        body.compiled().program().graph(),
        packed_inputs.as_slice(),
        lane_count,
        block,
        context,
        location,
    )?;
    debug_assert_eq!(packed_outputs.len(), logical_output_count);

    let unpacked_by_output = packed_outputs
        .iter()
        .zip(body.output_types().iter())
        .map(|(packed_output, output_type)| match mode {
            VMapLoweringMode::Forward => {
                lower_vmap_forward_unpack(*packed_output, output_type, lane_count, block, context, location)
            }
            VMapLoweringMode::Transpose => {
                lower_vmap_transpose_unpack(*packed_output, output_type, lane_count, block, context, location)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut results = Vec::with_capacity(body.total_output_count());
    for lane_index in 0..lane_count {
        for output_index in 0..logical_output_count {
            results.push(unpacked_by_output[output_index][lane_index]);
        }
    }
    Ok(results)
}

/// Lowers one plain traced graph to values inside a block.
#[cfg(any(test, feature = "benchmarking"))]
#[allow(dead_code)]
fn lower_plain_graph_outputs<'b, 'c: 'b, 't: 'c, V, Input, Output>(
    graph: &Graph<ProgramOpRef<V>, V, Input, Output>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    V: MlirLowerableValue,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    let mut atom_values = vec![None; graph.atom_count()];
    for (input_index, atom_id) in graph.input_atoms().iter().copied().enumerate() {
        atom_values[atom_id] = Some(block.argument(input_index).expect("body block arguments should exist").as_ref());
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
            AtomSource::Input => {}
            AtomSource::Constant => {
                atom_values[atom_id] = Some(lower_literal_value(&atom.example_value, block, context, location)?);
            }
            AtomSource::Derived => {
                let Some(equation_index) = equation_by_first_output[atom_id] else {
                    continue;
                };
                let equation = &graph.equations()[equation_index];
                let inputs = equation
                    .inputs
                    .iter()
                    .map(|input| atom_values[*input].ok_or(LoweringError::MissingAtomValue { atom_id: *input }))
                    .collect::<Result<Vec<_>, _>>()?;
                let lowered_outputs =
                    lower_plain_equation(graph, equation_index, inputs.as_slice(), block, context, location)?;
                for (output_atom, lowered_output) in equation.outputs.iter().copied().zip(lowered_outputs.into_iter()) {
                    atom_values[output_atom] = Some(lowered_output);
                }
            }
        }
    }

    graph
        .outputs()
        .iter()
        .map(|output| atom_values[*output].ok_or(LoweringError::MissingAtomValue { atom_id: *output }))
        .collect::<Result<Vec<_>, _>>()
}

/// Lowers one traced graph to values inside a block.
fn lower_graph_outputs<'b, 'c: 'b, 't: 'c, GraphInput, GraphOutput>(
    graph: &Graph<StagedOpRef<ShardMapTensor>, ShardMapTensor, GraphInput, GraphOutput>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    GraphInput: Parameterized<ShardMapTensor>,
    GraphOutput: Parameterized<ShardMapTensor>,
{
    let mut atom_values = vec![None; graph.atom_count()];
    for (input_index, atom_id) in graph.input_atoms().iter().copied().enumerate() {
        atom_values[atom_id] = Some(block.argument(input_index).expect("body block arguments should exist").as_ref());
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
            AtomSource::Input => {}
            AtomSource::Constant => {
                atom_values[atom_id] = Some(lower_constant(atom_id, &atom.example_value, block, context, location)?);
            }
            AtomSource::Derived => {
                let Some(equation_index) = equation_by_first_output[atom_id] else {
                    continue;
                };
                let equation = &graph.equations()[equation_index];
                let inputs = equation
                    .inputs
                    .iter()
                    .map(|input| atom_values[*input].ok_or(LoweringError::MissingAtomValue { atom_id: *input }))
                    .collect::<Result<Vec<_>, _>>()?;
                let lowered_outputs =
                    lower_equation(graph, equation_index, inputs.as_slice(), block, context, location)?;
                for (output_atom, lowered_output) in equation.outputs.iter().copied().zip(lowered_outputs.into_iter()) {
                    atom_values[output_atom] = Some(lowered_output);
                }
            }
        }
    }

    graph
        .outputs()
        .iter()
        .map(|output| atom_values[*output].ok_or(LoweringError::MissingAtomValue { atom_id: *output }))
        .collect::<Result<Vec<_>, _>>()
}

/// Lowers one `sdy.manual_computation` operation, including its nested body graph.
fn lower_manual_computation<'b, 'c: 'b, 't: 'c, GraphInput, GraphOutput>(
    block: &mut BlockRef<'b, 'c, 't>,
    outer_inputs: &[ValueRef<'b, 'c, 't>],
    shard_map: &ShardMap,
    graph: &Graph<StagedOpRef<ShardMapTensor>, ShardMapTensor, GraphInput, GraphOutput>,
    local_input_types: &[ArrayType],
    global_output_types: &[ArrayType],
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    GraphInput: Parameterized<ShardMapTensor>,
    GraphOutput: Parameterized<ShardMapTensor>,
{
    let local_input_tensor_types = local_input_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let global_output_tensor_types = global_output_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, context, location))
        .collect::<Result<Vec<_>, _>>()?;

    let mut body_region = context.region();
    let body_block = context.block(
        local_input_tensor_types
            .iter()
            .map(|tensor_type| (*tensor_type, location))
            .collect::<Vec<_>>()
            .as_slice(),
    );
    {
        let mut body_block_ref = body_block.as_ref();
        let body_outputs = lower_graph_outputs(graph, &mut body_block_ref, context, location.as_ref())?;
        body_block_ref.append_operation(shardy::r#return(body_outputs.as_slice(), location));
    }
    body_region.append_block(body_block);

    let manual_computation = block.append_operation(shardy::manual_computation(
        outer_inputs,
        global_output_tensor_types.as_slice(),
        shard_map.to_shardy_in_shardings(context),
        shard_map.to_shardy_out_shardings(context),
        shard_map.to_shardy_manual_axes(context),
        body_region,
        location,
    ));
    Ok(manual_computation.results().map(|result| result.as_ref()).collect::<Vec<_>>())
}

/// Lowers one linear shard-map evaluation mode and returns its resulting values.
fn lower_linear_shard_map_eval_mode<'b, 'c: 'b, 't: 'c>(
    eval_mode: &LinearShardMapEvalMode,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    match eval_mode {
        LinearShardMapEvalMode::Body(body) => {
            let simplified_body = body
                .simplified()
                .map_err(|error| LoweringError::SimplificationFailure { message: error.to_string() })?;
            lower_manual_computation(
                block,
                input_values,
                &simplified_body.shard_map,
                simplified_body.compiled.graph(),
                simplified_body.local_input_types.as_slice(),
                simplified_body.global_output_types.as_slice(),
                context,
                location,
            )
        }
        LinearShardMapEvalMode::FactorizedTranspose(factorized) => {
            let residual_body = factorized
                .residual_body
                .simplified()
                .map_err(|error| LoweringError::SimplificationFailure { message: error.to_string() })?;
            let residual_results = lower_manual_computation(
                block,
                &input_values[..residual_body.global_input_types.len()],
                &residual_body.shard_map,
                residual_body.compiled.graph(),
                residual_body.local_input_types.as_slice(),
                residual_body.global_output_types.as_slice(),
                context,
                location,
            )?;
            let apply_body = factorized
                .apply_body
                .simplified()
                .map_err(|error| LoweringError::SimplificationFailure { message: error.to_string() })?;
            let apply_inputs = input_values
                .iter()
                .copied()
                .take(apply_body.global_input_types.len() - residual_results.len())
                .chain(residual_results)
                .collect::<Vec<_>>();
            lower_manual_computation(
                block,
                apply_inputs.as_slice(),
                &apply_body.shard_map,
                apply_body.compiled.graph(),
                apply_body.local_input_types.as_slice(),
                apply_body.global_output_types.as_slice(),
                context,
                location,
            )
        }
    }
}

/// Lowers one concrete traced value to a StableHLO constant operation and returns its result value.
fn lower_literal_value<'b, 'c: 'b, 't: 'c, B, V, L>(
    value: &V,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    V: MlirLowerableValue,
    L: Location<'c, 't> + Copy,
{
    let value_type = value.tpe();
    if !value_type.shape.dimensions.is_empty() {
        let scalar_tensor_type = context
            .tensor_type(lower_element_type(value_type.data_type, context)?, &[], None, location)
            .ok_or_else(|| LoweringError::InvalidTensorType {
                array_type: ArrayType::new(value_type.data_type, crate::types::Shape::new(Vec::new()), None),
            })?;
        if let Some(scalar_elements) = value.to_scalar_dense_elements_attribute(scalar_tensor_type, context)? {
            let scalar_constant = block.append_operation(stable_hlo::constant(scalar_elements, location));
            let tensor_type = lower_tensor_type(&value_type, context, location)?;
            let broadcast = block.append_operation(stable_hlo::broadcast(
                scalar_constant.result(0).unwrap().as_ref(),
                tensor_type,
                &[],
                location,
            ));
            return Ok(broadcast.result(0).expect("stablehlo.broadcast should return one result").as_ref());
        }
    }

    let tensor_type = lower_tensor_type(&value_type, context, location)?;
    let elements = value.to_dense_elements_attribute(tensor_type, context)?;
    let constant = block.append_operation(stable_hlo::constant(elements, location));
    Ok(constant.result(0).expect("stablehlo.constant should return one result").as_ref())
}

/// Lowers a traced constant atom to a StableHLO constant operation and returns its result value.
fn lower_constant<'b, 'c: 'b, 't: 'c, B, L>(
    atom_id: usize,
    value: &ShardMapTensor,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    L: Location<'c, 't> + Copy,
{
    let constant_kind = value.constant_kind().ok_or(LoweringError::UnsupportedConstant { atom_id })?;
    let tensor_type = lower_tensor_type(value.r#type(), context, location)?;
    if !value.r#type().shape.dimensions.is_empty() {
        let scalar_tensor_type = context
            .tensor_type(lower_element_type(value.r#type().data_type, context)?, &[], None, location)
            .ok_or_else(|| LoweringError::InvalidTensorType {
                array_type: ArrayType::new(value.r#type().data_type, crate::types::Shape::new(Vec::new()), None),
            })?;
        let scalar_elements =
            lower_constant_elements_attribute(value.r#type().data_type, scalar_tensor_type, constant_kind, context)?;
        let scalar_constant = block.append_operation(stable_hlo::constant(scalar_elements, location));
        let broadcast = block.append_operation(stable_hlo::broadcast(
            scalar_constant.result(0).unwrap().as_ref(),
            tensor_type,
            &[],
            location,
        ));
        return Ok(broadcast.result(0).expect("stablehlo.broadcast should return one result").as_ref());
    }
    let elements = lower_constant_elements_attribute(value.r#type().data_type, tensor_type, constant_kind, context)?;
    let constant = block.append_operation(stable_hlo::constant(elements, location));
    Ok(constant.result(0).expect("stablehlo.constant should return one result").as_ref())
}

/// Lowers one traced equation from a plain `tracing_v2` program.
#[cfg(any(test, feature = "benchmarking"))]
#[allow(dead_code)]
fn lower_plain_equation<'b, 'c: 'b, 't: 'c, V, Input, Output>(
    graph: &Graph<ProgramOpRef<V>, V, Input, Output>,
    equation_index: usize,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    V: MlirLowerableValue,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    let equation = &graph.equations()[equation_index];
    let output_types = equation
        .outputs
        .iter()
        .map(|output| graph.atom(*output).expect("equation output should exist").abstract_value.clone())
        .collect::<Vec<_>>();
    let mut lowerer = PlainMlirLowerer { block: *block, context, location };
    equation
        .op
        .lower_plain_mlir(input_values, output_types.as_slice(), PlainMlirLoweringMode::Unpacked, &mut lowerer)
}

/// Lowers one equation inside a packed `vmap` body graph.
fn lower_packed_plain_equation<'b, 'c: 'b, 't: 'c, V>(
    graph: &Graph<ProgramOpRef<V>, V, Vec<V>, Vec<V>>,
    equation_index: usize,
    input_values: &[ValueRef<'b, 'c, 't>],
    lane_count: usize,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    V: MlirLowerableValue,
{
    let equation = &graph.equations()[equation_index];
    let output_types = equation
        .outputs
        .iter()
        .map(|output| {
            packed_array_type(&graph.atom(*output).expect("equation output should exist").abstract_value, lane_count)
        })
        .collect::<Vec<_>>();
    let mut lowerer = PlainMlirLowerer { block: *block, context, location };
    equation.op.lower_plain_mlir(
        input_values,
        output_types.as_slice(),
        PlainMlirLoweringMode::Packed { lane_count },
        &mut lowerer,
    )
}

/// Lowers one traced equation to the corresponding StableHLO operation and returns its result value.
fn lower_equation<'b, 'c: 'b, 't: 'c, GraphInput, GraphOutput>(
    graph: &Graph<StagedOpRef<ShardMapTensor>, ShardMapTensor, GraphInput, GraphOutput>,
    equation_index: usize,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    GraphInput: Parameterized<ShardMapTensor>,
    GraphOutput: Parameterized<ShardMapTensor>,
{
    let equation = &graph.equations()[equation_index];
    let output_types = equation
        .outputs
        .iter()
        .map(|output| graph.atom(*output).expect("equation output should exist").abstract_value.clone())
        .collect::<Vec<_>>();
    let mut lowerer = ShardMapMlirLowerer { block: *block, context, location };
    equation.op.lower_shard_map_mlir(input_values, output_types.as_slice(), &mut lowerer)
}

/// Normalizes a user-provided MLIR symbol name.
fn normalize_function_name(function_name: &str) -> Result<String, LoweringError> {
    let function_name = function_name.trim();
    if function_name.is_empty() || function_name.chars().any(char::is_whitespace) {
        return Err(LoweringError::InvalidFunctionName { function_name: function_name.to_string() });
    }
    Ok(function_name.strip_prefix('@').unwrap_or(function_name).to_string())
}

/// Lowers an [`ArrayType`] to a typed MLIR tensor type.
fn lower_tensor_type<'c, 't, L: Location<'c, 't>>(
    array_type: &ArrayType,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ryft_mlir::TensorTypeRef<'c, 't>, LoweringError> {
    let element_type = lower_element_type(array_type.data_type, context)?;
    let dimensions = array_type
        .shape
        .dimensions
        .iter()
        .map(|size| match size {
            Size::Static(value) => MlirSize::Static(*value),
            Size::Dynamic(_) => MlirSize::Dynamic,
        })
        .collect::<Vec<_>>();
    context
        .tensor_type(element_type, dimensions.as_slice(), None, location)
        .ok_or_else(|| LoweringError::InvalidTensorType { array_type: array_type.clone() })
}

/// Lowers one [`DataType`] to the corresponding MLIR element type.
fn lower_element_type<'c, 't>(
    data_type: DataType,
    context: &'c MlirContext<'t>,
) -> Result<TypeRef<'c, 't>, LoweringError> {
    Ok(match data_type {
        DataType::Token => return Err(LoweringError::UnsupportedDataType { data_type }),
        DataType::Boolean => context.signless_integer_type(1).as_ref(),
        DataType::I1 => context.signless_integer_type(1).as_ref(),
        DataType::I2 => context.signless_integer_type(2).as_ref(),
        DataType::I4 => context.signless_integer_type(4).as_ref(),
        DataType::I8 => context.signless_integer_type(8).as_ref(),
        DataType::I16 => context.signless_integer_type(16).as_ref(),
        DataType::I32 => context.signless_integer_type(32).as_ref(),
        DataType::I64 => context.signless_integer_type(64).as_ref(),
        DataType::U1 => context.unsigned_integer_type(1).as_ref(),
        DataType::U2 => context.unsigned_integer_type(2).as_ref(),
        DataType::U4 => context.unsigned_integer_type(4).as_ref(),
        DataType::U8 => context.unsigned_integer_type(8).as_ref(),
        DataType::U16 => context.unsigned_integer_type(16).as_ref(),
        DataType::U32 => context.unsigned_integer_type(32).as_ref(),
        DataType::U64 => context.unsigned_integer_type(64).as_ref(),
        DataType::BF16 => context.bfloat16_type().as_ref(),
        DataType::F16 => context.float16_type().as_ref(),
        DataType::F32 => context.float32_type().as_ref(),
        DataType::F64 => context.float64_type().as_ref(),
        DataType::F4E2M1FN => context.float4e2m1fn_type().as_ref(),
        DataType::F8E3M4 => context.float8e3m4_type().as_ref(),
        DataType::F8E4M3 => context.float8e4m3_type().as_ref(),
        DataType::F8E4M3FN => context.float8e4m3fn_type().as_ref(),
        DataType::F8E4M3FNUZ => context.float8e4m3fnuz_type().as_ref(),
        DataType::F8E4M3B11FNUZ => context.float8e4m3b11fnuz_type().as_ref(),
        DataType::F8E5M2 => context.float8e5m2_type().as_ref(),
        DataType::F8E5M2FNUZ => context.float8e5m2fnuz_type().as_ref(),
        DataType::F8E8M0FNU => context.float8e8m0fnu_type().as_ref(),
        DataType::C64 => context.complex_type(context.float32_type()).as_ref(),
        DataType::C128 => context.complex_type(context.float64_type()).as_ref(),
    })
}

/// Builds the dense-elements attribute for one traced splat constant.
fn lower_constant_elements_attribute<'c, 't>(
    data_type: DataType,
    tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
    constant_kind: ShardMapConstantKind,
    context: &'c MlirContext<'t>,
) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError> {
    let integer_value = match constant_kind {
        ShardMapConstantKind::Zero => 0,
        ShardMapConstantKind::One => 1,
    };
    let float_value = integer_value as f64;

    match data_type {
        DataType::Boolean => context
            .splatted_dense_attribute_elements_attribute(tensor_type, context.boolean_attribute(integer_value != 0))
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::I1 | DataType::I2 | DataType::I4 | DataType::I8 | DataType::I16 | DataType::I32 | DataType::I64 => {
            context
                .splatted_dense_attribute_elements_attribute(
                    tensor_type,
                    context.integer_attribute(
                        context.signless_integer_type(signed_integer_width(data_type)?),
                        integer_value,
                    ),
                )
                .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type })
        }
        DataType::U1 | DataType::U2 | DataType::U4 | DataType::U8 | DataType::U16 | DataType::U32 | DataType::U64 => {
            context
                .splatted_dense_attribute_elements_attribute(
                    tensor_type,
                    context.integer_attribute(
                        context.unsigned_integer_type(unsigned_integer_width(data_type)?),
                        integer_value,
                    ),
                )
                .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type })
        }
        DataType::BF16 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.bfloat16_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F16 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float16_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F32 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float32_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F64 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float64_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F4E2M1FN => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float4e2m1fn_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E3M4 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e3m4_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E4M3 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e4m3_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E4M3FN => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e4m3fn_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E4M3FNUZ => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e4m3fnuz_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E4M3B11FNUZ => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e4m3b11fnuz_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E5M2 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e5m2_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E5M2FNUZ => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e5m2fnuz_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E8M0FNU => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e8m0fnu_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::Token | DataType::C64 | DataType::C128 => Err(LoweringError::UnsupportedDataType { data_type }),
    }
}

/// Returns the bit width of a signed integer [`DataType`].
fn signed_integer_width(data_type: DataType) -> Result<usize, LoweringError> {
    Ok(match data_type {
        DataType::I1 => 1,
        DataType::I2 => 2,
        DataType::I4 => 4,
        DataType::I8 => 8,
        DataType::I16 => 16,
        DataType::I32 => 32,
        DataType::I64 => 64,
        _ => return Err(LoweringError::UnsupportedDataType { data_type }),
    })
}

/// Returns the bit width of an unsigned integer [`DataType`].
fn unsigned_integer_width(data_type: DataType) -> Result<usize, LoweringError> {
    Ok(match data_type {
        DataType::U1 => 1,
        DataType::U2 => 2,
        DataType::U4 => 4,
        DataType::U8 => 8,
        DataType::U16 => 16,
        DataType::U32 => 32,
        DataType::U64 => 64,
        _ => return Err(LoweringError::UnsupportedDataType { data_type }),
    })
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    #[cfg(feature = "ndarray")]
    use ndarray::{Array2, arr2};

    use crate::sharding::{MeshAxis, MeshAxisType};
    use crate::tracing_v2::{FloatExt, MatrixOps, OneLike, ZeroLike};
    use crate::types::Shape;

    use super::super::shard_map::{TracedShardMap, shard_map as traced_shard_map};
    use crate::sharding::LogicalMesh;

    use super::super::sharding::{PartitionDimension, PartitionSpec};
    use super::*;

    fn test_manual_mesh(axis_name: &str, axis_size: usize) -> LogicalMesh {
        LogicalMesh::new(vec![MeshAxis::new(axis_name, axis_size, MeshAxisType::Manual).unwrap()]).unwrap()
    }

    fn test_vector_type(length: usize) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(length)]), None)
    }

    fn test_matrix_type(rows: usize, cols: usize) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(rows), Size::Static(cols)]), None)
    }

    fn lower_traced_module(
        traced: &TracedShardMap<ArrayType, ArrayType>,
        function_name: &str,
    ) -> Result<String, super::super::shard_map::ShardMapTraceError> {
        traced.to_mlir_module(function_name)
    }

    #[cfg(feature = "ndarray")]
    fn bilinear_matmul<M>(inputs: (M, M)) -> M
    where
        M: MatrixOps,
    {
        inputs.0.matmul(inputs.1)
    }

    #[test]
    fn test_to_mlir_module_renders_a_full_add_module() {
        let global_input_type = test_vector_type(8);
        let traced: TracedShardMap<ArrayType, ArrayType> = traced_shard_map(
            |x| x.clone() + x,
            global_input_type,
            test_manual_mesh("x", 4),
            PartitionSpec::new(vec![PartitionDimension::sharded("x")]),
            PartitionSpec::new(vec![PartitionDimension::sharded("x")]),
        )
        .unwrap();

        assert_eq!(
            lower_traced_module(&traced, "main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = stablehlo.add %arg1, %arg1 : tensor<2xf32>
                      sdy.return %1 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_to_mlir_module_renders_constants_and_supported_ops() {
        let global_input_type = test_matrix_type(4, 4);
        let traced: TracedShardMap<ArrayType, ArrayType> = traced_shard_map(
            |x| {
                let product = x.clone().transpose_matrix().matmul(x);
                let waveform = (-product).cos().sin();
                (waveform.clone() * waveform.one_like()) + waveform.zero_like()
            },
            global_input_type,
            test_manual_mesh("x", 2),
            PartitionSpec::new(vec![PartitionDimension::sharded("x"), PartitionDimension::unsharded()]),
            PartitionSpec::new(vec![PartitionDimension::sharded("x"), PartitionDimension::unsharded()]),
        )
        .unwrap();

        assert_eq!(
            lower_traced_module(&traced, "kernel").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @kernel(%arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}, {}]>] out_shardings=[<@mesh, [{"x"}, {}]>] manual_axes={"x"} (%arg1: tensor<2x4xf32>) {
                      %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<2x4xf32>) -> tensor<4x2xf32>
                      %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x2xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
                      %3 = stablehlo.negate %2 : tensor<4x4xf32>
                      %4 = stablehlo.cosine %3 : tensor<4x4xf32>
                      %5 = stablehlo.sine %4 : tensor<4x4xf32>
                      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
                      %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x4xf32>
                      %7 = stablehlo.multiply %5, %6 : tensor<4x4xf32>
                      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                      %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<4x4xf32>
                      %9 = stablehlo.add %7, %8 : tensor<4x4xf32>
                      sdy.return %9 : tensor<4x4xf32>
                    } : (tensor<4x4xf32>) -> tensor<8x4xf32>
                    return %0 : tensor<8x4xf32>
                  }
                }
            "#}
        );
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_to_mlir_module_for_plain_graph_renders_transposed_matrix_pullback_factors() {
        let left = arr2(&[[1.0f64, 2.0], [3.0, 4.0]]);
        let right = arr2(&[[5.0f64, 6.0], [7.0, 8.0]]);
        let (_, pullback): (
            Array2<f64>,
            crate::tracing_v2::LinearProgram<Array2<f64>, Array2<f64>, (Array2<f64>, Array2<f64>)>,
        ) = crate::tracing_v2::vjp(bilinear_matmul, (left, right)).unwrap();

        assert_eq!(
            to_mlir_module_for_plain_graph(pullback.program().graph(), "main").unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x2xf64>) -> (tensor<2x2xf64>, tensor<2x2xf64>) {
                    %cst = stablehlo.constant dense<[[5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]]> : tensor<2x2xf64>
                    %0 = stablehlo.dot_general %arg0, %cst, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
                    %cst_0 = stablehlo.constant dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>
                    %1 = stablehlo.dot_general %arg0, %cst_0, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
                    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<2x2xf64>) -> tensor<2x2xf64>
                    return %0, %2 : tensor<2x2xf64>, tensor<2x2xf64>
                  }
                }
            "#}
        );
    }
}
