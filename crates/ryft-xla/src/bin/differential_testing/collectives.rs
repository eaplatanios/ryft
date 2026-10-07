//! Executes the shared collective matrix on real CPU PJRT participants.

use std::collections::BTreeMap;
use std::error::Error;

use serde::Deserialize;

use ryft_core::{
    ArrayType, AxisIndex, BatchAxis, Broadcast, ConvertElementType, Differentiate, LogicalMesh, MeshAxis, MeshAxisType,
    ParallelAllGather, ParallelAllGatherOutputVariance, ParallelAllToAll, ParallelPermute, ParallelReduce,
    ParallelSumScatter, ParallelVary, ProgramError, ReductionKind, Sharding, ShardingDimension, Typed, Value,
    ValueProjection, batch,
};
use ryft_pjrt::{ClientOptions, CpuClientOptions, load_cpu_plugin};
use ryft_xla::FromPjrt;
use ryft_xla::experimental::{ShardMapTracer, TracedXlaProgram, shard_map, trace};

use super::{DifferentialObservation, SCHEMA, execute_collective_inputs};

/// One finite collective experiment shared with JAX and the independent host reference.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CollectiveCase {
    /// Stable observation identifier.
    pub(super) id: String,

    /// Collective operation to execute.
    operation: String,

    /// Number of CPU PJRT participants.
    participants: usize,

    /// Physical input shape on each participant, including a leading batch axis for batching cases.
    local_shape: Vec<usize>,

    /// Row-major mesh dimensions; omitted for a one-dimensional mesh.
    #[serde(default)]
    mesh_shape: Vec<usize>,

    /// Named mesh axis addressed by the operation.
    #[serde(default = "default_axis_name")]
    axis_name: String,

    /// Logical tensor axis for gathering or scattering.
    #[serde(default)]
    tensor_axis: usize,

    /// Logical tensor axis consumed by all-to-all.
    #[serde(default)]
    split_axis: usize,

    /// Logical tensor axis receiving all-to-all chunks.
    #[serde(default)]
    concatenation_axis: usize,

    /// Whether an existing tensor dimension is tiled rather than inserted or removed.
    #[serde(default)]
    tiled: bool,

    /// Ordered participant groups within the named axis.
    #[serde(default)]
    groups: Option<Vec<Vec<usize>>>,

    /// Source-to-target routing pairs for permutation.
    #[serde(default)]
    pairs: Vec<(usize, usize)>,

    /// Source participant at each shuffle destination.
    #[serde(default)]
    permutation: Vec<usize>,

    /// Reduction kind for parallel reduction.
    #[serde(default)]
    reduction: Option<String>,

    /// Transform executed inside the manual mesh region.
    #[serde(default = "default_transform")]
    transform: String,

    /// Whether every device receives the same logically replicated input.
    #[serde(default)]
    replicated_input: bool,
}

/// Default one-dimensional mesh axis.
fn default_axis_name() -> String {
    "x".into()
}

/// Default direct execution policy.
fn default_transform() -> String {
    "primal".into()
}

/// Reads the same manifest used by the Python comparison harness.
pub(super) fn registry() -> Result<Vec<CollectiveCase>, Box<dyn Error>> {
    let cases: Vec<CollectiveCase> = serde_json::from_str(include_str!("collective_cases.json"))?;
    for (index, case) in cases.iter().enumerate() {
        if cases[..index].iter().any(|previous| previous.id == case.id) {
            return Err(format!("duplicate collective case `{}`", case.id).into());
        }
        if !matches!(
            case.operation.as_str(),
            "all_gather"
                | "sum_scatter"
                | "all_to_all"
                | "swap_axes"
                | "permute"
                | "shuffle"
                | "reduce"
                | "vary"
                | "axis_index"
        ) || !matches!(case.transform.as_str(), "primal" | "batch" | "jvp" | "vjp")
            || !matches!(case.reduction.as_deref(), None | Some("sum" | "product" | "mean" | "min" | "max"))
        {
            return Err(format!("unknown operation, transform, or reduction in `{}`", case.id).into());
        }
        if case.participants == 0 || case.local_shape.is_empty() || case.local_shape.contains(&0) {
            return Err(format!("invalid participant count or local shape in `{}`", case.id).into());
        }
    }
    Ok(cases)
}

impl CollectiveCase {
    /// Applies the selected operation to primal, batch, or differentiation tracers through the same capability API.
    fn apply<V>(&self, input: V) -> Result<V, ProgramError>
    where
        V: ParallelAllGather<ArrayType>
            + ParallelAllToAll<ArrayType>
            + ParallelPermute<ArrayType>
            + ParallelReduce<ArrayType>
            + ParallelSumScatter<ArrayType>
            + ParallelVary<ArrayType>,
    {
        let mut options = if self.tiled { ryft_core::CollectiveOptions::tiled() } else { Default::default() };
        if let Some(groups) = &self.groups {
            options = options.with_axis_index_groups(groups.clone());
        }
        match self.operation.as_str() {
            "all_gather" => input.parallel_all_gather_with_options(
                &self.axis_name,
                self.tensor_axis,
                options,
                ParallelAllGatherOutputVariance::Varying,
            ),
            "sum_scatter" => input.parallel_sum_scatter_with_options(&self.axis_name, self.tensor_axis, options),
            "all_to_all" => input.parallel_all_to_all_with_options(
                &self.axis_name,
                self.split_axis,
                self.concatenation_axis,
                options,
            ),
            "swap_axes" => match &self.groups {
                Some(groups) => {
                    input.parallel_swap_axes_with_axis_index_groups(&self.axis_name, self.tensor_axis, groups.clone())
                }
                None => input.parallel_swap_axes(&self.axis_name, self.tensor_axis),
            },
            "permute" => input.parallel_permute(&self.axis_name, self.pairs.clone()),
            "shuffle" => input.parallel_shuffle(&self.axis_name, &self.permutation),
            "vary" => input.parallel_vary(&self.axis_name),
            "reduce" => {
                let kind = match self.reduction.as_deref().unwrap_or("sum") {
                    "sum" => ReductionKind::Sum,
                    "product" => ReductionKind::Product,
                    "mean" => ReductionKind::Mean,
                    "min" => ReductionKind::Min,
                    "max" => ReductionKind::Max,
                    kind => panic!("unknown manifest reduction `{kind}`"),
                };
                match &self.groups {
                    Some(groups) => input.parallel_reduce_with_axis_index_groups(kind, &self.axis_name, groups.clone()),
                    None => input.parallel_reduce(kind, &self.axis_name),
                }
            }
            operation => panic!("unknown manifest operation `{operation}`"),
        }
    }

    /// Stages an actual transform within the mesh binder so named collectives survive to XLA execution.
    fn transformed(&self, input: ShardMapTracer, seed: ShardMapTracer) -> Result<Vec<ShardMapTracer>, ProgramError> {
        match self.transform.as_str() {
            "primal" if self.operation == "axis_index" => {
                let index = input.domain().axis_index(&self.axis_name)?;
                let index = index.convert_element_type(ryft_core::DataType::F32)?;
                let index_sharding = index
                    .r#type()
                    .sharding()
                    .unwrap()
                    .with_broadcasted_dimensions(input.r#type().rank(), &[])
                    .map_err(ryft_core::TypeError::from)?;
                let output_type =
                    input.r#type().into_owned().with_sharding(index_sharding).map_err(ryft_core::TypeError::from)?;
                Ok(vec![index.broadcast(output_type, &[])?])
            }
            "primal" => Ok(vec![self.apply(input)?]),
            "batch" => {
                // The leading physical dimension is the unrelated mapped axis. Collective tensor axes refer to
                // the logical slice shape; batching restores the mapped axis at position zero after the collective.
                let output = batch(|input| self.apply(input), input, BatchAxis::new(0), BatchAxis::new(0), None)?;
                Ok(vec![output])
            }
            "jvp" => {
                let (primal, tangent) =
                    input.clone().into_value().domain().differentiate_at(input.into_value()).jvp(
                        seed.into_value(),
                        |input| {
                            let input = ValueProjection::<ArrayType>::into_projected(input)?;
                            Ok(self.apply(input)?.into_value())
                        },
                    )?;
                Ok(vec![
                    ValueProjection::<ArrayType>::into_projected(primal)?,
                    ValueProjection::<ArrayType>::into_projected(tangent)?,
                ])
            }
            "vjp" => {
                let (primal, pullback) =
                    input.clone().into_value().domain().differentiate_at(input.into_value()).vjp(|input| {
                        let input = ValueProjection::<ArrayType>::into_projected(input)?;
                        Ok(self.apply(input)?.into_value())
                    })?;
                let primal = ValueProjection::<ArrayType>::into_projected(primal)?;
                // A reduction's primal boundary is invariant on the addressed mesh axis. Its replicated outputs
                // contribute all participant seeds to that invariant cotangent before entering the local pullback.
                let seed = if self.operation == "reduce" {
                    match &self.groups {
                        Some(groups) => seed.parallel_reduce_with_axis_index_groups(
                            ReductionKind::Sum,
                            &self.axis_name,
                            groups.clone(),
                        )?,
                        None => seed.parallel_reduce(ReductionKind::Sum, &self.axis_name)?,
                    }
                } else {
                    seed
                };
                let cotangent = pullback.apply(seed.into_value())?;
                Ok(vec![primal, ValueProjection::<ArrayType>::into_projected(cotangent)?])
            }
            transform => panic!("unknown manifest transform `{transform}`"),
        }
    }

    /// Computes the physical result shape from the collective contract without using backend inference.
    fn output_shape(&self, mesh_shape: &[usize]) -> Vec<usize> {
        let axis_position = if self.axis_name == "y" { 1 } else { 0 };
        let group_size = self.groups.as_ref().map_or(mesh_shape[axis_position], |groups| groups[0].len());
        let mut shape = self.local_shape.clone();
        let offset = usize::from(self.transform == "batch");
        match self.operation.as_str() {
            "all_gather" if self.tiled => shape[self.tensor_axis + offset] *= group_size,
            "all_gather" => shape.insert(self.tensor_axis + offset, group_size),
            "sum_scatter" if self.tiled => shape[self.tensor_axis + offset] /= group_size,
            "sum_scatter" => {
                shape.remove(self.tensor_axis + offset);
            }
            "all_to_all" if self.tiled => {
                shape[self.split_axis + offset] /= group_size;
                shape[self.concatenation_axis + offset] *= group_size;
            }
            "all_to_all" => {
                shape.remove(self.split_axis + offset);
                shape.insert(self.concatenation_axis + offset, group_size);
            }
            _ => {}
        }
        shape
    }

    /// Compiles and executes this case using one CPU device per logical participant.
    pub(super) fn emit(&self) -> Result<DifferentialObservation, Box<dyn Error>> {
        let mesh_shape = if self.mesh_shape.is_empty() { vec![self.participants] } else { self.mesh_shape.clone() };
        if mesh_shape.iter().product::<usize>() != self.participants || mesh_shape.len() > 2 {
            return Err(format!("invalid mesh shape in `{}`", self.id).into());
        }
        let mesh = LogicalMesh::new(
            mesh_shape
                .iter()
                .zip(["x", "y"])
                .map(|(size, name)| MeshAxis::new(name, *size, MeshAxisType::Manual))
                .collect::<Result<Vec<_>, _>>()?,
        )?;
        let dimension = if self.replicated_input {
            ShardingDimension::replicated()
        } else {
            ShardingDimension::sharded(mesh.axes().iter().map(|axis| axis.name()))
        };
        let mut dimensions = vec![ShardingDimension::replicated(); self.local_shape.len()];
        dimensions[0] = dimension;
        let input_sharding = Sharding::new(mesh.clone(), dimensions)?;
        let output_count = if self.transform == "jvp" || self.transform == "vjp" { 2 } else { 1 };
        let output_shape = self.output_shape(&mesh_shape);
        let sharding_for_shape = |shape: &[usize]| -> Result<Sharding, Box<dyn Error>> {
            let mut dimensions = vec![ShardingDimension::replicated(); shape.len()];
            dimensions[0] = ShardingDimension::sharded(mesh.axes().iter().map(|axis| axis.name()));
            Ok(Sharding::new(mesh.clone(), dimensions)?)
        };
        let output_sharding = sharding_for_shape(&output_shape)?;
        let mut output_shardings = vec![output_sharding.clone()];
        if output_count == 2 {
            output_shardings.push(if self.transform == "vjp" {
                sharding_for_shape(&self.local_shape)?
            } else {
                output_sharding
            });
        }
        let mut global_shape = self.local_shape.clone();
        if !self.replicated_input {
            global_shape[0] *= self.participants;
        }
        let seed_shape = if self.transform == "vjp" { output_shape.clone() } else { self.local_shape.clone() };
        let seed_sharding = sharding_for_shape(&seed_shape)?;
        let mut seed_global_shape = seed_shape.clone();
        seed_global_shape[0] *= self.participants;
        let stablehlo = if output_count == 1 {
            let traced: TracedXlaProgram<ArrayType, Vec<ArrayType>> = trace(
                |input: ShardMapTracer| {
                    shard_map::<_, _, Vec<ArrayType>, _>(
                        |input: ShardMapTracer| self.transformed(input.clone(), input).unwrap(),
                        input,
                        mesh.clone(),
                        input_sharding.clone(),
                        output_shardings.clone(),
                    )
                    .unwrap()
                },
                ArrayType::new_static(ryft_core::DataType::F32, global_shape.clone()),
            )?;
            traced.to_mlir_module("main")?
        } else {
            let traced: TracedXlaProgram<(ArrayType, ArrayType), Vec<ArrayType>> = trace(
                |inputs: (ShardMapTracer, ShardMapTracer)| {
                    shard_map::<_, _, Vec<ArrayType>, _>(
                        |(input, seed): (ShardMapTracer, ShardMapTracer)| self.transformed(input, seed).unwrap(),
                        inputs,
                        mesh.clone(),
                        (input_sharding.clone(), seed_sharding.clone()),
                        output_shardings.clone(),
                    )
                    .unwrap()
                },
                (
                    ArrayType::new_static(ryft_core::DataType::F32, global_shape.clone()),
                    ArrayType::new_static(ryft_core::DataType::F32, seed_global_shape.clone()),
                ),
            )?;
            traced.to_mlir_module("main")?
        };
        let plugin = load_cpu_plugin()?;
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions {
            device_count: Some(self.participants),
            ..Default::default()
        }))?;
        let devices = client
            .addressable_devices()?
            .iter()
            .map(ryft_core::Device::from_pjrt)
            .collect::<Result<Vec<_>, _>>()?;
        let device_mesh = ryft_core::DeviceMesh::new(mesh, devices)?;
        let count = self.local_shape.iter().product::<usize>();
        let values = (0..self.participants)
            .map(|participant| {
                let start = if self.replicated_input { 0 } else { participant * count };
                (1..=count).map(|offset| (start + offset) as f32).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let seed_count = seed_shape.iter().product::<usize>();
        let seeds = (0..self.participants)
            .map(|participant| {
                (0..seed_count)
                    .map(|offset| {
                        let index = participant * seed_count + offset;
                        if self.transform == "jvp" { (index % 5 + 1) as f32 } else { (index + 1) as f32 }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let local_shape = self.local_shape.iter().map(|extent| *extent as u64).collect::<Vec<_>>();
        let seed_local_shape = seed_shape.iter().map(|extent| *extent as u64).collect::<Vec<_>>();
        let mut expected_output_shapes = vec![output_shape.clone()];
        if output_count == 2 {
            expected_output_shapes.push(if self.transform == "vjp" { self.local_shape.clone() } else { output_shape });
        }
        let mut inputs = vec![(input_sharding, global_shape.as_slice(), local_shape.as_slice(), values.as_slice())];
        if output_count == 2 {
            inputs.push((seed_sharding, seed_global_shape.as_slice(), seed_local_shape.as_slice(), seeds.as_slice()));
        }
        let outputs = execute_collective_inputs(&client, device_mesh, &stablehlo, &inputs, &expected_output_shapes)?;
        let mut observations = BTreeMap::from([("primal", outputs.iter().map(|device| device[0].clone()).collect())]);
        if output_count == 2 {
            let name = if self.transform == "jvp" { "tangent" } else { "cotangent" };
            observations.insert(name, outputs.iter().map(|device| device[1].clone()).collect());
        }
        Ok(DifferentialObservation {
            schema: SCHEMA,
            case_id: self.id.clone(),
            observations,
            staging: None,
            stablehlo: Some(stablehlo),
        })
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_registry() {
        let cases = registry().unwrap();
        assert_eq!(cases.first().unwrap().id, "collective_gather_1_untiled");
        assert!(cases.iter().any(|case| case.participants == 8));
        assert!(cases.iter().any(|case| case.mesh_shape == [2, 2]));
        for transform in ["primal", "batch", "jvp", "vjp"] {
            assert!(cases.iter().any(|case| case.transform == transform));
        }
    }

    #[test]
    fn test_collective_case_output_shape() {
        let cases = registry().unwrap();
        let gather = cases.iter().find(|case| case.id == "collective_gather_4_untiled").unwrap();
        assert_eq!(gather.output_shape(&[4]), vec![2, 4, 3]);
        let gather = cases.iter().find(|case| case.id == "collective_gather_tiled_axis_1").unwrap();
        assert_eq!(gather.output_shape(&[4]), vec![2, 12]);
    }
}
