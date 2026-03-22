//! Manual SPMD metadata for Shardy `sdy.manual_computation`.
//!
//! This module provides a small `ryft-core::xla` representation of the metadata carried by
//! JAX's `shard_map`: a logical mesh, per-input and per-output shardings, and the set of mesh
//! axes that are handled manually inside the computation body.
//!
//! The scope here is intentionally narrow. This module does **not** trace Rust closures or
//! compile executable bodies. Instead, it models the part of `shard_map` that already fits the
//! current `ryft-core::xla` layer:
//!
//! - validating manual-axis usage against a [`LogicalMesh`],
//! - deriving body-local shapes from global shapes, and
//! - rendering the Shardy attributes needed by `sdy.manual_computation`.
//!
//! # Relationship to existing sharding types
//!
//! `ShardMap` accepts [`PartitionSpec`] values as its public input surface, but internally it
//! lowers them to [`NamedSharding`] values.
//!
//! `ShardMap` derives its manual axes from the mesh itself: every axis whose type is
//! [`Manual`](super::sharding::MeshAxisType::Manual) is treated as manual for the
//! `sdy.manual_computation` region.
//!
//! This matters because Shardy requires each manual axis to be made explicit in every
//! `in_shardings` / `out_shardings` entry: a manual axis must either shard a dimension or be
//! explicitly listed as replicated. `PartitionSpec` alone cannot represent explicit replicated
//! axes, but [`NamedSharding`] can. As a result, `ShardMap` automatically promotes unused manual
//! axes into each internal [`NamedSharding`]'s replicated-axis list.
//!
//! # Shardy correspondence
//!
//! The public helpers on [`ShardMap`] render the three attributes attached to
//! `sdy.manual_computation`:
//!
//! | `ShardMap` data      | Shardy attribute    |
//! | -------------------- | ------------------- |
//! | input shardings      | `in_shardings=[...]`  |
//! | output shardings     | `out_shardings=[...]` |
//! | manual mesh axes     | `manual_axes={...}`   |
//!
//! Refer to the [Shardy compiler API documentation](https://openxla.org/shardy/compiler_api) and
//! the [Shardy dialect documentation](https://openxla.org/shardy/sdy_dialect) for the IR-level
//! semantics of manual computation regions.

use std::collections::HashSet;

use thiserror::Error;

use super::sharding::{LogicalMesh, NamedSharding, PartitionDimension, PartitionSpec, ShardingError};

/// Error type for [`ShardMap`] construction, local-shape derivation, and Shardy rendering.
#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum ShardMapError {
    /// Underlying error returned by the mesh/sharding layer.
    #[error("{0}")]
    ShardingError(#[from] ShardingError),

    /// Error returned when a mesh used for `ShardMap` has no manual axes.
    #[error("shard_map requires at least one mesh axis with type manual")]
    MeshHasNoManualAxes,

    /// Error returned when an input index does not exist.
    #[error("input index {input_index} is out of range for {input_count} input sharding(s)")]
    InvalidInputIndex { input_index: usize, input_count: usize },

    /// Error returned when an output index does not exist.
    #[error("output index {output_index} is out of range for {output_count} output sharding(s)")]
    InvalidOutputIndex { output_index: usize, output_count: usize },

    /// Error returned when a partitioned dimension uses a free axis more major than a manual axis.
    #[error(
        "{value_kind} sharding #{value_index} dimension #{dimension} uses free axis '{free_axis_name}' \
         more major than manual axis '{manual_axis_name}'"
    )]
    ManualAxisMustPrecedeFreeAxis {
        value_kind: &'static str,
        value_index: usize,
        dimension: usize,
        free_axis_name: String,
        manual_axis_name: String,
    },

    /// Error returned when a provided global shape rank does not match the sharding rank.
    #[error(
        "{value_kind} sharding #{value_index} has rank {partition_rank}, but the provided shape \
         has rank {shape_rank}"
    )]
    RankMismatch { value_kind: &'static str, value_index: usize, partition_rank: usize, shape_rank: usize },

    /// Error returned when a manual axis would require padding in the local body shape.
    #[error(
        "{value_kind} sharding #{value_index} dimension #{dimension} has size {dimension_size}, \
         which is not divisible by manual partition count {manual_partition_count}"
    )]
    ManualAxisIntroducesPadding {
        value_kind: &'static str,
        value_index: usize,
        dimension: usize,
        dimension_size: usize,
        manual_partition_count: usize,
    },
}

/// Metadata describing one manual SPMD computation over a mesh.
///
/// A `ShardMap` stores the mesh plus the validated per-input and per-output shardings.
///
/// The manual axes are not stored separately; they are always derived from the mesh axes whose
/// type is [`Manual`](super::sharding::MeshAxisType::Manual).
///
/// The public constructors accept [`PartitionSpec`] values because that is the natural
/// JAX-facing surface. Internally, those partition specs are converted to [`NamedSharding`]
/// values so that manual axes omitted from a partition spec can be made explicit as replicated
/// axes in the Shardy lowering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardMap {
    mesh: LogicalMesh,
    in_shardings: Vec<NamedSharding>,
    out_shardings: Vec<NamedSharding>,
}

impl ShardMap {
    /// Creates a `ShardMap` whose manual axes are derived from the mesh.
    ///
    /// Every mesh axis with type [`Manual`](super::sharding::MeshAxisType::Manual) is treated as
    /// manual inside the body. The constructor returns [`ShardMapError::MeshHasNoManualAxes`] if
    /// the mesh contains no manual axes.
    ///
    /// # Parameters
    ///
    ///   - `mesh`: Logical mesh that the manual computation is defined over.
    ///   - `in_specs`: Per-input partition specs for the global inputs.
    ///   - `out_specs`: Per-output partition specs for the global outputs.
    pub fn new(
        mesh: LogicalMesh,
        in_specs: Vec<PartitionSpec>,
        out_specs: Vec<PartitionSpec>,
    ) -> Result<Self, ShardMapError> {
        let in_shardings = build_named_shardings(&mesh, in_specs, "input")?;
        let out_shardings = build_named_shardings(&mesh, out_specs, "output")?;
        Ok(Self { mesh, in_shardings, out_shardings })
    }

    /// Returns the logical mesh of this manual computation.
    pub fn mesh(&self) -> &LogicalMesh {
        &self.mesh
    }

    /// Returns the validated per-input shardings.
    pub fn in_shardings(&self) -> &[NamedSharding] {
        self.in_shardings.as_slice()
    }

    /// Returns the validated per-output shardings.
    pub fn out_shardings(&self) -> &[NamedSharding] {
        self.out_shardings.as_slice()
    }

    /// Returns the manual mesh axes in mesh order.
    pub fn manual_axes(&self) -> Vec<&str> {
        self.mesh.manual_axes()
    }

    /// Returns the local body shape for input `input_index`.
    ///
    /// The returned shape is the tensor shape seen inside the manual computation body for the
    /// corresponding global input. Only manual axes reduce the local shape; free axes remain
    /// global from the body's point of view.
    ///
    /// # Parameters
    ///
    ///   - `input_index`: Index of the input sharding to use.
    ///   - `global_shape`: Global input shape associated with that input.
    pub fn local_input_shape(&self, input_index: usize, global_shape: &[usize]) -> Result<Vec<usize>, ShardMapError> {
        let sharding = self
            .in_shardings
            .get(input_index)
            .ok_or(ShardMapError::InvalidInputIndex { input_index, input_count: self.in_shardings.len() })?;
        local_shape_for_sharding(sharding, global_shape, "input", input_index)
    }

    /// Returns the local body shape for output `output_index`.
    ///
    /// # Parameters
    ///
    ///   - `output_index`: Index of the output sharding to use.
    ///   - `global_shape`: Global output shape associated with that output.
    pub fn local_output_shape(&self, output_index: usize, global_shape: &[usize]) -> Result<Vec<usize>, ShardMapError> {
        let sharding = self
            .out_shardings
            .get(output_index)
            .ok_or(ShardMapError::InvalidOutputIndex { output_index, output_count: self.out_shardings.len() })?;
        local_shape_for_sharding(sharding, global_shape, "output", output_index)
    }

    /// Renders the Shardy `in_shardings=[...]` attribute payload.
    ///
    /// The returned string is suitable for direct insertion into an `sdy.manual_computation`
    /// operation.
    ///
    /// # Parameters
    ///
    ///   - `mesh_symbol_name`: Symbol name used by the surrounding `sdy.mesh` declaration.
    pub fn to_shardy_in_shardings_attribute<S: AsRef<str>>(
        &self,
        mesh_symbol_name: S,
    ) -> Result<String, ShardMapError> {
        render_shardy_sharding_list(self.in_shardings.as_slice(), mesh_symbol_name)
    }

    /// Renders the Shardy `out_shardings=[...]` attribute payload.
    ///
    /// # Parameters
    ///
    ///   - `mesh_symbol_name`: Symbol name used by the surrounding `sdy.mesh` declaration.
    pub fn to_shardy_out_shardings_attribute<S: AsRef<str>>(
        &self,
        mesh_symbol_name: S,
    ) -> Result<String, ShardMapError> {
        render_shardy_sharding_list(self.out_shardings.as_slice(), mesh_symbol_name)
    }

    /// Renders the Shardy `manual_axes={...}` attribute payload.
    pub fn to_shardy_manual_axes_attribute(&self) -> String {
        let manual_axes = self.manual_axes();
        render_shardy_axes(manual_axes.as_slice())
    }

    /// Renders the three Shardy attributes attached to `sdy.manual_computation`.
    ///
    /// # Parameters
    ///
    ///   - `mesh_symbol_name`: Symbol name used by the surrounding `sdy.mesh` declaration.
    pub fn to_shardy_manual_computation_attributes<S: AsRef<str>>(
        &self,
        mesh_symbol_name: S,
    ) -> Result<String, ShardMapError> {
        let in_shardings = self.to_shardy_in_shardings_attribute(mesh_symbol_name.as_ref())?;
        let out_shardings = self.to_shardy_out_shardings_attribute(mesh_symbol_name.as_ref())?;
        Ok(format!(
            "in_shardings={in_shardings} out_shardings={out_shardings} manual_axes={}",
            self.to_shardy_manual_axes_attribute()
        ))
    }
}

fn build_named_shardings(
    mesh: &LogicalMesh,
    partition_specs: Vec<PartitionSpec>,
    value_kind: &'static str,
) -> Result<Vec<NamedSharding>, ShardMapError> {
    let manual_axes = manual_axes_from_mesh(mesh)?;
    let manual_axis_names = manual_axes.iter().map(String::as_str).collect::<HashSet<_>>();
    partition_specs
        .into_iter()
        .enumerate()
        .map(|(value_index, partition_spec)| {
            validate_manual_axis_order(&partition_spec, &manual_axis_names, value_kind, value_index)?;
            let used_axes = used_axes_in_partition_spec(&partition_spec);
            let replicated_axes =
                manual_axes.iter().filter(|axis_name| !used_axes.contains(axis_name.as_str())).cloned().collect();
            Ok(NamedSharding::with_extra_axes(mesh.clone(), partition_spec, replicated_axes, Vec::new())?)
        })
        .collect()
}

fn manual_axes_from_mesh(mesh: &LogicalMesh) -> Result<Vec<String>, ShardMapError> {
    let manual_axes = mesh.manual_axes().into_iter().map(ToString::to_string).collect::<Vec<_>>();
    if manual_axes.is_empty() {
        return Err(ShardMapError::MeshHasNoManualAxes);
    }
    Ok(manual_axes)
}

fn validate_manual_axis_order(
    partition_spec: &PartitionSpec,
    manual_axes: &HashSet<&str>,
    value_kind: &'static str,
    value_index: usize,
) -> Result<(), ShardMapError> {
    for (dimension, partition_dimension) in partition_spec.dimensions().iter().enumerate() {
        if let PartitionDimension::Sharded(axis_names) = partition_dimension {
            let mut first_free_axis: Option<&str> = None;
            for axis_name in axis_names {
                if manual_axes.contains(axis_name.as_str()) {
                    if let Some(free_axis_name) = first_free_axis {
                        return Err(ShardMapError::ManualAxisMustPrecedeFreeAxis {
                            value_kind,
                            value_index,
                            dimension,
                            free_axis_name: free_axis_name.to_string(),
                            manual_axis_name: axis_name.clone(),
                        });
                    }
                } else if first_free_axis.is_none() {
                    first_free_axis = Some(axis_name.as_str());
                }
            }
        }
    }
    Ok(())
}

fn used_axes_in_partition_spec(partition_spec: &PartitionSpec) -> HashSet<&str> {
    let mut used_axes = HashSet::new();
    for partition_dimension in partition_spec.dimensions() {
        if let PartitionDimension::Sharded(axis_names) = partition_dimension {
            for axis_name in axis_names {
                used_axes.insert(axis_name.as_str());
            }
        }
    }
    used_axes
}

fn local_shape_for_sharding(
    sharding: &NamedSharding,
    global_shape: &[usize],
    value_kind: &'static str,
    value_index: usize,
) -> Result<Vec<usize>, ShardMapError> {
    let partition_spec = sharding.partition_spec();
    if partition_spec.rank() != global_shape.len() {
        return Err(ShardMapError::RankMismatch {
            value_kind,
            value_index,
            partition_rank: partition_spec.rank(),
            shape_rank: global_shape.len(),
        });
    }

    let manual_axis_names = sharding.mesh().manual_axes().into_iter().collect::<HashSet<_>>();
    let mut local_shape = Vec::with_capacity(global_shape.len());
    for (dimension, (partition_dimension, dimension_size)) in
        partition_spec.dimensions().iter().zip(global_shape.iter().copied()).enumerate()
    {
        let manual_partition_count = match partition_dimension {
            PartitionDimension::Sharded(axis_names) => axis_names
                .iter()
                .filter(|axis_name| manual_axis_names.contains(axis_name.as_str()))
                .try_fold(1usize, |partition_count, axis_name| {
                    let axis_size = sharding
                        .mesh()
                        .axis_size(axis_name)
                        .ok_or_else(|| ShardingError::UnknownMeshAxis { axis_name: axis_name.clone() })?;
                    partition_count.checked_mul(axis_size).ok_or_else(|| ShardingError::Overflow {
                        context: format!(
                            "computing manual partition count for {value_kind} sharding \
                                 #{value_index} dimension #{dimension}"
                        ),
                    })
                })?,
            PartitionDimension::Unsharded | PartitionDimension::Unconstrained => 1,
        };

        if dimension_size % manual_partition_count != 0 {
            return Err(ShardMapError::ManualAxisIntroducesPadding {
                value_kind,
                value_index,
                dimension,
                dimension_size,
                manual_partition_count,
            });
        }

        local_shape.push(dimension_size / manual_partition_count);
    }
    Ok(local_shape)
}

fn render_shardy_sharding_list<S: AsRef<str>>(
    shardings: &[NamedSharding],
    mesh_symbol_name: S,
) -> Result<String, ShardMapError> {
    let mut result = String::from("[");
    for (sharding_index, sharding) in shardings.iter().enumerate() {
        if sharding_index > 0 {
            result.push_str(", ");
        }
        result.push_str(stripped_shardy_tensor_sharding(sharding, mesh_symbol_name.as_ref())?.as_str());
    }
    result.push(']');
    Ok(result)
}

fn stripped_shardy_tensor_sharding<S: AsRef<str>>(
    sharding: &NamedSharding,
    mesh_symbol_name: S,
) -> Result<String, ShardMapError> {
    let mesh_symbol_name = normalize_mesh_symbol_name(mesh_symbol_name.as_ref())?;
    let mut result = format!(
        "<@{mesh_symbol_name}, {}",
        render_manual_computation_dimensions(sharding.mesh(), sharding.partition_spec())
    );

    if !sharding.replicated_axes().is_empty() {
        result.push_str(", replicated={");
        for (axis_index, axis_name) in sharding.replicated_axes().iter().enumerate() {
            if axis_index > 0 {
                result.push_str(", ");
            }
            result.push('"');
            result.push_str(escape_shardy_string(axis_name).as_str());
            result.push('"');
        }
        result.push('}');
    }

    if !sharding.unreduced_axes().is_empty() {
        result.push_str(", unreduced={");
        for (axis_index, axis_name) in sharding.unreduced_axes().iter().enumerate() {
            if axis_index > 0 {
                result.push_str(", ");
            }
            result.push('"');
            result.push_str(escape_shardy_string(axis_name).as_str());
            result.push('"');
        }
        result.push('}');
    }

    result.push('>');
    Ok(result)
}

fn render_manual_computation_dimensions(mesh: &LogicalMesh, partition_spec: &PartitionSpec) -> String {
    let manual_axis_names = mesh.manual_axes().into_iter().collect::<HashSet<_>>();
    let has_free_axes = mesh.axes().len() > manual_axis_names.len();

    let mut result = String::from("[");
    for (dimension_index, partition_dimension) in partition_spec.dimensions().iter().enumerate() {
        if dimension_index > 0 {
            result.push_str(", ");
        }

        match partition_dimension {
            PartitionDimension::Unsharded => {
                if has_free_axes {
                    result.push_str("{?}");
                } else {
                    result.push_str("{}");
                }
            }
            PartitionDimension::Sharded(axis_names) => {
                let contains_free_axis =
                    axis_names.iter().any(|axis_name| !manual_axis_names.contains(axis_name.as_str()));
                result.push('{');
                for (axis_index, axis_name) in axis_names.iter().enumerate() {
                    if axis_index > 0 {
                        result.push_str(", ");
                    }
                    result.push('"');
                    result.push_str(escape_shardy_string(axis_name).as_str());
                    result.push('"');
                }
                if contains_free_axis {
                    result.push_str(", ?");
                }
                result.push('}');
            }
            PartitionDimension::Unconstrained => result.push_str("{?}"),
        }
    }
    result.push(']');
    result
}

fn render_shardy_axes<A: AsRef<str>>(axis_names: &[A]) -> String {
    let mut result = String::from("{");
    for (axis_index, axis_name) in axis_names.iter().enumerate() {
        if axis_index > 0 {
            result.push_str(", ");
        }
        result.push('"');
        result.push_str(escape_shardy_string(axis_name.as_ref()).as_str());
        result.push('"');
    }
    result.push('}');
    result
}

fn escape_shardy_string(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn normalize_mesh_symbol_name(mesh_symbol_name: &str) -> Result<String, ShardingError> {
    let mesh_symbol_name = mesh_symbol_name.trim();
    if mesh_symbol_name.is_empty() {
        return Err(ShardingError::EmptyMeshSymbolName);
    }

    let mesh_symbol_name = mesh_symbol_name.strip_prefix('@').unwrap_or(mesh_symbol_name);
    if mesh_symbol_name.is_empty() || mesh_symbol_name.chars().any(char::is_whitespace) {
        return Err(ShardingError::InvalidMeshSymbolName { mesh_symbol_name: mesh_symbol_name.to_string() });
    }

    Ok(mesh_symbol_name.to_string())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;
    use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
    use ryft_pjrt::{BufferType, ClientOptions, CpuClientOptions, Program, load_cpu_plugin};

    use super::*;
    use crate::types::data_types::DataType;
    use crate::xla::arrays::Array;
    use crate::xla::sharding::{
        DeviceMesh, MeshAxis, MeshAxisType, MeshDevice, PartitionDimension, PartitionSpec, ShardingContext,
    };

    fn test_logical_mesh_2x2() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::with_type("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::with_type("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap()
    }

    fn test_logical_mesh_data_model() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::with_type("data", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("model", 4).unwrap(),
        ])
        .unwrap()
    }

    fn test_logical_mesh_without_manual_axes() -> LogicalMesh {
        LogicalMesh::new(vec![MeshAxis::new("x", 2).unwrap(), MeshAxis::new("y", 2).unwrap()]).unwrap()
    }

    fn test_spmd_compilation_options(partition_count: usize) -> CompilationOptions {
        CompilationOptions {
            argument_layouts: Vec::new(),
            parameter_is_tupled_arguments: false,
            executable_build_options: Some(ExecutableCompilationOptions {
                device_ordinal: -1,
                replica_count: 1,
                partition_count: partition_count as i64,
                use_spmd_partitioning: true,
                use_shardy_partitioner: true,
                ..Default::default()
            }),
            compile_portable_executable: false,
            profile_version: 0,
            serialized_multi_slice_configuration: Vec::new(),
            environment_option_overrides: HashMap::new(),
            target_config: None,
            allow_in_place_mlir_modification: false,
            matrix_unit_operand_precision: Precision::Default as i32,
        }
    }

    fn f32_values_to_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * size_of::<f32>());
        for value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        bytes
    }

    fn two_f32s_from_bytes(bytes: &[u8]) -> [f32; 2] {
        assert_eq!(bytes.len(), 2 * size_of::<f32>());
        let first = f32::from_ne_bytes(bytes[..size_of::<f32>()].try_into().unwrap());
        let second = f32::from_ne_bytes(bytes[size_of::<f32>()..].try_into().unwrap());
        [first, second]
    }

    #[test]
    fn test_shard_map_uses_manual_axes_from_mesh() {
        let shard_map = ShardMap::new(
            test_logical_mesh_2x2(),
            vec![PartitionSpec::new(vec![PartitionDimension::sharded("x")])],
            vec![PartitionSpec::new(vec![PartitionDimension::sharded("x")])],
        )
        .unwrap();

        assert_eq!(shard_map.manual_axes(), vec!["x", "y"]);
        assert_eq!(shard_map.in_shardings()[0].replicated_axes(), &["y".to_string()]);
        assert_eq!(shard_map.out_shardings()[0].replicated_axes(), &["y".to_string()]);
    }

    #[test]
    fn test_shard_map_rejects_mesh_without_manual_axes() {
        let result = ShardMap::new(test_logical_mesh_without_manual_axes(), Vec::new(), Vec::new());

        assert_eq!(result, Err(ShardMapError::MeshHasNoManualAxes));
    }

    #[test]
    fn test_shard_map_rejects_free_axis_before_manual_axis() {
        let result = ShardMap::new(
            test_logical_mesh_data_model(),
            vec![PartitionSpec::new(vec![PartitionDimension::sharded_by(["model", "data"])])],
            Vec::new(),
        );

        assert_eq!(
            result,
            Err(ShardMapError::ManualAxisMustPrecedeFreeAxis {
                value_kind: "input",
                value_index: 0,
                dimension: 0,
                free_axis_name: "model".to_string(),
                manual_axis_name: "data".to_string(),
            })
        );
    }

    #[test]
    fn test_shard_map_local_input_shape_for_all_manual_axes() {
        let shard_map = ShardMap::new(
            test_logical_mesh_2x2(),
            vec![PartitionSpec::new(vec![PartitionDimension::sharded_by(["x", "y"])])],
            Vec::new(),
        )
        .unwrap();

        assert_eq!(shard_map.local_input_shape(0, &[16]).unwrap(), vec![4]);
    }

    #[test]
    fn test_shard_map_local_input_shape_for_mixed_manual_and_free_axes() {
        let shard_map = ShardMap::new(
            test_logical_mesh_data_model(),
            vec![PartitionSpec::new(vec![PartitionDimension::sharded_by(["data", "model"])])],
            Vec::new(),
        )
        .unwrap();

        assert_eq!(shard_map.local_input_shape(0, &[16]).unwrap(), vec![8]);
    }

    #[test]
    fn test_shard_map_local_output_shape() {
        let shard_map = ShardMap::new(
            test_logical_mesh_data_model(),
            Vec::new(),
            vec![PartitionSpec::new(vec![PartitionDimension::sharded("data"), PartitionDimension::unsharded()])],
        )
        .unwrap();

        assert_eq!(shard_map.local_output_shape(0, &[32, 8]).unwrap(), vec![16, 8]);
    }

    #[test]
    fn test_shard_map_local_shape_rejects_padding_from_manual_axes() {
        let shard_map = ShardMap::new(
            LogicalMesh::new(vec![MeshAxis::with_type("x", 3, MeshAxisType::Manual).unwrap()]).unwrap(),
            vec![PartitionSpec::new(vec![PartitionDimension::sharded("x")])],
            Vec::new(),
        )
        .unwrap();

        assert_eq!(
            shard_map.local_input_shape(0, &[10]),
            Err(ShardMapError::ManualAxisIntroducesPadding {
                value_kind: "input",
                value_index: 0,
                dimension: 0,
                dimension_size: 10,
                manual_partition_count: 3,
            })
        );
    }

    #[test]
    fn test_shard_map_local_shape_rejects_rank_mismatch() {
        let shard_map = ShardMap::new(
            test_logical_mesh_2x2(),
            vec![PartitionSpec::new(vec![PartitionDimension::sharded("x")])],
            Vec::new(),
        )
        .unwrap();

        assert_eq!(
            shard_map.local_input_shape(0, &[8, 4]),
            Err(ShardMapError::RankMismatch { value_kind: "input", value_index: 0, partition_rank: 1, shape_rank: 2 })
        );
    }

    #[test]
    fn test_shard_map_local_shape_rejects_invalid_indices() {
        let shard_map = ShardMap::new(
            test_logical_mesh_2x2(),
            vec![PartitionSpec::new(vec![PartitionDimension::sharded("x")])],
            vec![PartitionSpec::new(vec![PartitionDimension::sharded("x")])],
        )
        .unwrap();

        assert_eq!(
            shard_map.local_input_shape(1, &[8]),
            Err(ShardMapError::InvalidInputIndex { input_index: 1, input_count: 1 })
        );
        assert_eq!(
            shard_map.local_output_shape(1, &[8]),
            Err(ShardMapError::InvalidOutputIndex { output_index: 1, output_count: 1 })
        );
    }

    #[test]
    fn test_shard_map_renders_in_shardings_attribute() {
        let shard_map = ShardMap::new(
            test_logical_mesh_2x2(),
            vec![PartitionSpec::new(vec![PartitionDimension::sharded("x")])],
            Vec::new(),
        )
        .unwrap();

        assert_eq!(
            shard_map.to_shardy_in_shardings_attribute("mesh").unwrap(),
            r#"[<@mesh, [{"x"}], replicated={"y"}>]"#
        );
    }

    #[test]
    fn test_shard_map_renders_free_axes_as_open_dimension_shardings() {
        let shard_map = ShardMap::new(
            test_logical_mesh_data_model(),
            vec![PartitionSpec::new(vec![PartitionDimension::sharded_by(["data", "model"])])],
            Vec::new(),
        )
        .unwrap();

        assert_eq!(shard_map.to_shardy_in_shardings_attribute("mesh").unwrap(), r#"[<@mesh, [{"data", "model", ?}]>]"#);
    }

    #[test]
    fn test_shard_map_renders_out_shardings_attribute() {
        let shard_map = ShardMap::new(
            LogicalMesh::new(vec![
                MeshAxis::with_type("x", 2, MeshAxisType::Manual).unwrap(),
                MeshAxis::new("y", 2).unwrap(),
            ])
            .unwrap(),
            Vec::new(),
            vec![PartitionSpec::new(vec![PartitionDimension::unsharded()])],
        )
        .unwrap();

        assert_eq!(
            shard_map.to_shardy_out_shardings_attribute("mesh").unwrap(),
            r#"[<@mesh, [{?}], replicated={"x"}>]"#
        );
    }

    #[test]
    fn test_shard_map_renders_manual_axes_attribute() {
        let shard_map = ShardMap::new(
            LogicalMesh::new(vec![
                MeshAxis::new("x", 2).unwrap(),
                MeshAxis::with_type("y", 2, MeshAxisType::Manual).unwrap(),
            ])
            .unwrap(),
            Vec::new(),
            Vec::new(),
        )
        .unwrap();

        assert_eq!(shard_map.manual_axes(), vec!["y"]);
        assert_eq!(shard_map.to_shardy_manual_axes_attribute(), r#"{"y"}"#);
    }

    #[test]
    fn test_shard_map_renders_manual_computation_attributes() {
        let shard_map = ShardMap::new(
            test_logical_mesh_data_model(),
            vec![PartitionSpec::new(vec![PartitionDimension::sharded("data")])],
            vec![PartitionSpec::new(vec![PartitionDimension::sharded("data")])],
        )
        .unwrap();

        assert_eq!(
            shard_map.to_shardy_manual_computation_attributes("mesh").unwrap(),
            r#"in_shardings=[<@mesh, [{"data"}]>] out_shardings=[<@mesh, [{"data"}]>] manual_axes={"data"}"#
        );
    }

    #[test]
    fn test_shard_map_manual_computation_executes_end_to_end_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) }))
            .expect("failed to create 4-device CPU client");
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 4);

        let mesh_devices = client_devices
            .iter()
            .map(|device| MeshDevice::new(device.id().unwrap(), device.process_index().unwrap()))
            .collect::<Vec<_>>();
        let device_mesh =
            DeviceMesh::new(vec![MeshAxis::with_type("x", 4, MeshAxisType::Manual).unwrap()], mesh_devices).unwrap();

        let partition_spec = PartitionSpec::new(vec![PartitionDimension::sharded("x")]);
        let shard_map = ShardMap::new(
            device_mesh.logical_mesh().clone(),
            vec![partition_spec.clone()],
            vec![partition_spec.clone()],
        )
        .unwrap();
        assert_eq!(shard_map.local_input_shape(0, &[8]).unwrap(), vec![2]);
        assert_eq!(shard_map.local_output_shape(0, &[8]).unwrap(), vec![2]);

        let input_sharding = shard_map.in_shardings()[0]
            .to_shardy_tensor_sharding_attribute("mesh", ShardingContext::ExplicitSharding)
            .unwrap();
        let output_sharding = shard_map.out_shardings()[0]
            .to_shardy_tensor_sharding_attribute("mesh", ShardingContext::ExplicitSharding)
            .unwrap();
        let manual_computation_attributes = shard_map.to_shardy_manual_computation_attributes("mesh").unwrap();
        let mesh_operation = shard_map.mesh().to_shardy_mesh_operation("mesh").unwrap();

        let mlir_program = format!(
            r#"
                module {{
                    {mesh_operation}
                    func.func @main(
                        %arg0: tensor<8xf32> {{sdy.sharding = {input_sharding}}}
                    ) -> (tensor<8xf32> {{sdy.sharding = {output_sharding}}}) {{
                        %0 = sdy.manual_computation(%arg0) {manual_computation_attributes} (%arg1: tensor<2xf32>) {{
                            %1 = stablehlo.add %arg1, %arg1 : tensor<2xf32>
                            sdy.return %1 : tensor<2xf32>
                        }} : (tensor<8xf32>) -> tensor<8xf32>
                        return %0 : tensor<8xf32>
                    }}
                }}
            "#
        );

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let shard_values = [device_index as f32 * 2.0 + 1.0, device_index as f32 * 2.0 + 2.0];
                client
                    .buffer(
                        f32_values_to_bytes(&shard_values).as_slice(),
                        BufferType::F32,
                        [2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let input_array =
            Array::from_sharding(vec![8], DataType::F32, device_mesh, partition_spec, input_buffers).unwrap();
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(4)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 4);
        let expected_values_by_device = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                (device.id().unwrap(), [device_index as f32 * 4.0 + 2.0, device_index as f32 * 4.0 + 4.0])
            })
            .collect::<HashMap<_, _>>();
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();

        let execute_arguments =
            Array::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), 0, None, Some(file!()), None, None)
            .unwrap();

        assert_eq!(outputs.len(), execution_device_ids.len());
        for (output, device_id) in outputs.into_iter().zip(execution_device_ids.iter().copied()) {
            output.done.r#await().unwrap();
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(
                two_f32s_from_bytes(output_bytes.as_slice()),
                *expected_values_by_device.get(&device_id).unwrap()
            );
        }
    }
}
