//! Executes real two-process CUDA collectives over the distributed PJRT runtime.

#[cfg(feature = "cuda-13")]
use std::collections::BTreeMap;
use std::error::Error;
#[cfg(feature = "cuda-13")]
use std::sync::Arc;
#[cfg(feature = "cuda-13")]
use std::time::Duration;

use serde::Deserialize;

#[cfg(any(test, feature = "cuda-13"))]
use ryft_core::{
    ArrayType, CollectiveOptions, DataType, LogicalMesh, MeshAxis, MeshAxisType, ParallelAllGather,
    ParallelAllGatherOutputVariance, ParallelAllToAll, ParallelRaggedAllToAll, ParallelReduce, ReductionKind, Sharding,
    ShardingDimension,
};
#[cfg(feature = "cuda-13")]
use ryft_pjrt::{
    BufferType, Client, ClientOptions, DistributedRuntimeClientOptions, DistributedRuntimeServiceOptions,
    ExecutionDeviceInputs, ExecutionInput, GpuClientOptions, GpuMemoryAllocator, GpuPlatform, KeyValueStore, Program,
    load_cuda_13_plugin,
};
#[cfg(feature = "cuda-13")]
use ryft_xla::DistributedRuntime;
#[cfg(any(test, feature = "cuda-13"))]
use ryft_xla::experimental::{ShardMapTracer, TracedXlaProgram, shard_map, trace};

use super::DifferentialObservation;
#[cfg(feature = "cuda-13")]
use super::{SCHEMA, collective_compilation_options, f32_values_from_bytes};

/// One real two-rank exchange shared with JAX and the independent host reference.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DistributedCollectiveCase {
    /// Stable observation identifier.
    pub(super) id: String,

    /// Collective primitive to execute.
    operation: String,

    /// Global participant count.
    participants: usize,

    /// Physical operand buffer shape on each rank.
    local_shape: Vec<usize>,

    /// Physical output buffer shape on each rank.
    output_shape: Vec<usize>,

    /// Runtime metadata element type for ragged exchanges.
    #[serde(default)]
    metadata_type: Option<String>,

    /// Sender-owned operand row offsets, indexed by rank then destination.
    #[serde(default)]
    input_offsets: Vec<Vec<u64>>,

    /// Sender-owned row counts, indexed by rank then destination.
    #[serde(default)]
    send_sizes: Vec<Vec<u64>>,

    /// Sender-owned destination row offsets, indexed by rank then destination.
    #[serde(default)]
    output_offsets: Vec<Vec<u64>>,

    /// Receiver-owned row counts, indexed by rank then source.
    #[serde(default)]
    receive_sizes: Vec<Vec<u64>>,
}

/// Reads the shared manifest and rejects invalid or locally degenerate distributed experiments.
pub(super) fn registry() -> Result<Vec<DistributedCollectiveCase>, Box<dyn Error>> {
    let cases: Vec<DistributedCollectiveCase> =
        serde_json::from_str(include_str!("distributed_collective_cases.json"))?;
    for (index, case) in cases.iter().enumerate() {
        if cases[..index].iter().any(|previous| previous.id == case.id)
            || case.participants != 2
            || case.local_shape.is_empty()
            || case.output_shape.is_empty()
            || case.local_shape.contains(&0)
            || case.output_shape.contains(&0)
            || !matches!(case.operation.as_str(), "sum" | "all_gather" | "all_to_all" | "ragged_all_to_all")
        {
            return Err(format!("invalid distributed collective descriptor `{}`", case.id).into());
        }
        if case.operation == "ragged_all_to_all" {
            if !matches!(case.metadata_type.as_deref(), Some("i32" | "u64"))
                || case.local_shape[1..] != case.output_shape[1..]
                || [&case.input_offsets, &case.send_sizes, &case.output_offsets, &case.receive_sizes]
                    .iter()
                    .any(|metadata| metadata.len() != 2 || metadata.iter().any(|values| values.len() != 2))
            {
                return Err(format!("invalid distributed ragged metadata in `{}`", case.id).into());
            }
            // Metadata retain the sender's destination indexing. Validate both ranks together so destination bounds,
            // transposed receive sizes, and disjoint writes are established before launching accelerator work.
            for sender in 0..2 {
                for receiver in 0..2 {
                    let size = case.send_sizes[sender][receiver];
                    if size != case.receive_sizes[receiver][sender]
                        || case.input_offsets[sender][receiver]
                            .checked_add(size)
                            .is_none_or(|end| end > case.local_shape[0] as u64)
                        || case.output_offsets[sender][receiver]
                            .checked_add(size)
                            .is_none_or(|end| end > case.output_shape[0] as u64)
                    {
                        return Err(format!("invalid distributed ragged transfer in `{}`", case.id).into());
                    }
                }
            }
            for receiver in 0..2 {
                let first = case.output_offsets[0][receiver];
                let second = case.output_offsets[1][receiver];
                if case.send_sizes[0][receiver] > 0
                    && case.send_sizes[1][receiver] > 0
                    && first < second + case.send_sizes[1][receiver]
                    && second < first + case.send_sizes[0][receiver]
                {
                    return Err(format!("overlapping distributed ragged receives in `{}`", case.id).into());
                }
            }
        }
    }
    Ok(cases)
}

impl DistributedCollectiveCase {
    /// Lowers one global SPMD module identically on both ranks, preserving runtime ragged metadata inputs.
    #[cfg(any(test, feature = "cuda-13"))]
    fn stage(&self) -> Result<String, Box<dyn Error>> {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual)?])?;
        let sharding = |shape: &[usize]| -> Result<Sharding, Box<dyn Error>> {
            let mut dimensions = vec![ShardingDimension::replicated(); shape.len()];
            dimensions[0] = ShardingDimension::sharded(["x"]);
            Ok(Sharding::new(mesh.clone(), dimensions)?)
        };
        let mut shapes = vec![self.local_shape.clone()];
        if self.operation == "ragged_all_to_all" {
            shapes.push(self.output_shape.clone());
            shapes.extend(vec![vec![2]; 4]);
        }
        let types = shapes
            .iter()
            .enumerate()
            .map(|(index, shape)| {
                let data_type = if index < 2 {
                    DataType::F32
                } else if self.metadata_type.as_deref() == Some("u64") {
                    DataType::U64
                } else {
                    DataType::I32
                };
                let mut global_shape = shape.clone();
                global_shape[0] *= 2;
                ArrayType::new_static(data_type, global_shape)
            })
            .collect::<Vec<_>>();
        let input_shardings = shapes.iter().map(|shape| sharding(shape)).collect::<Result<Vec<_>, _>>()?;
        let output_sharding = sharding(&self.output_shape)?;
        let traced: TracedXlaProgram<Vec<ArrayType>, ArrayType> = trace(
            |inputs: Vec<ShardMapTracer>| {
                shard_map::<_, _, ArrayType, _>(
                    |inputs: Vec<ShardMapTracer>| match self.operation.as_str() {
                        "sum" => inputs[0].parallel_reduce(ReductionKind::Sum, "x").unwrap(),
                        "all_gather" => inputs[0]
                            .parallel_all_gather_with_options(
                                "x",
                                0,
                                CollectiveOptions::tiled(),
                                ParallelAllGatherOutputVariance::Varying,
                            )
                            .unwrap(),
                        "all_to_all" => {
                            inputs[0].parallel_all_to_all_with_options("x", 0, 1, CollectiveOptions::tiled()).unwrap()
                        }
                        "ragged_all_to_all" => inputs[0]
                            .parallel_ragged_all_to_all("x", &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5])
                            .unwrap(),
                        _ => unreachable!(),
                    },
                    inputs,
                    mesh.clone(),
                    input_shardings.clone(),
                    output_sharding.clone(),
                )
                .unwrap()
            },
            types,
        )?;
        Ok(traced.to_mlir_module("main")?)
    }

    /// Compiles with two global partitions, but supplies only this process's one addressable CUDA buffer family.
    #[cfg(feature = "cuda-13")]
    fn emit(&self, client: &Client<'_>, rank: usize) -> Result<DifferentialObservation, Box<dyn Error>> {
        let stablehlo = self.stage()?;
        let devices = client.addressable_devices()?;
        let count = self.local_shape.iter().product::<usize>();
        let floating = |shape: &[usize], values: Vec<f32>| {
            (
                values.into_iter().flat_map(f32::to_ne_bytes).collect::<Vec<_>>(),
                BufferType::F32,
                shape.iter().map(|extent| *extent as u64).collect::<Vec<_>>(),
            )
        };
        let mut raw_inputs =
            vec![floating(&self.local_shape, (1..=count).map(|index| (rank * count + index) as f32).collect())];
        if self.operation == "ragged_all_to_all" {
            let count = self.output_shape.iter().product::<usize>();
            raw_inputs.push(floating(
                &self.output_shape,
                (0..count).map(|index| (100 + rank * count + index) as f32).collect(),
            ));
            for metadata in [&self.input_offsets, &self.send_sizes, &self.output_offsets, &self.receive_sizes] {
                let values = &metadata[rank];
                let (bytes, buffer_type) = if self.metadata_type.as_deref() == Some("u64") {
                    (values.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<_>>(), BufferType::U64)
                } else {
                    let values = values.iter().map(|value| i32::try_from(*value)).collect::<Result<Vec<_>, _>>()?;
                    (values.into_iter().flat_map(i32::to_ne_bytes).collect::<Vec<_>>(), BufferType::I32)
                };
                raw_inputs.push((bytes, buffer_type, vec![2]));
            }
        }
        let inputs = raw_inputs
            .into_iter()
            .map(|(bytes, buffer_type, shape)| {
                Ok(ExecutionInput {
                    buffer: Arc::new(client.buffer(
                        bytes.as_slice(),
                        buffer_type,
                        shape,
                        None,
                        devices[0].clone(),
                        None,
                    )?),
                    donatable: false,
                })
            })
            .collect::<Result<Vec<_>, ryft_pjrt::Error>>()?;
        let executable = client
            .compile(&Program::Mlir { bytecode: stablehlo.as_bytes().to_vec() }, &collective_compilation_options(2))?;
        if executable.addressable_devices()?.len() != 1 {
            return Err(format!("rank {rank} executable does not have exactly one local device").into());
        }
        let outputs = executable
            .execute(
                vec![ExecutionDeviceInputs { inputs: &inputs, ..Default::default() }],
                Vec::new(),
                0,
                None,
                Some(file!()),
                None,
                None,
            )?
            .block_until_ready()?;
        if outputs.len() != 1 || outputs[0].outputs.len() != 1 {
            return Err(format!("rank {rank} returned an unexpected output count in `{}`", self.id).into());
        }
        let buffer = &outputs[0].outputs[0];
        let expected_shape = self.output_shape.iter().map(|extent| *extent as u64).collect::<Vec<_>>();
        if buffer.element_type()? != BufferType::F32 || buffer.dimensions()? != expected_shape.as_slice() {
            return Err(format!("rank {rank} output in `{}` has incorrect shape or element type", self.id).into());
        }
        let bytes = buffer.copy_to_host(None)?.r#await()?;
        if bytes.len() != self.output_shape.iter().product::<usize>() * size_of::<f32>() {
            return Err(format!("rank {rank} output in `{}` has incorrect byte count", self.id).into());
        }
        Ok(DifferentialObservation {
            schema: SCHEMA,
            case_id: self.id.clone(),
            observations: BTreeMap::from([("primal", vec![f32_values_from_bytes(&bytes)])]),
            staging: None,
            stablehlo: Some(stablehlo),
        })
    }
}

/// Runs all selected experiments in one bounded distributed-runtime lifetime and returns only this rank's records.
pub(super) fn run_worker(
    rank: usize,
    coordinator: &str,
    case_ids: &[&str],
) -> Result<Vec<DifferentialObservation>, Box<dyn Error>> {
    if rank >= 2 {
        return Err(format!("distributed worker rank must be `0` or `1`, got {rank}").into());
    }
    if coordinator.is_empty() {
        return Err("distributed workers require a coordinator address".into());
    }
    let cases = registry()?;
    let selected = if case_ids.is_empty() {
        cases.iter().collect::<Vec<_>>()
    } else {
        case_ids
            .iter()
            .map(|id| {
                cases
                    .iter()
                    .find(|case| case.id == *id)
                    .ok_or_else(|| format!("unknown distributed collective case `{id}`"))
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    #[cfg(not(feature = "cuda-13"))]
    {
        let _ = selected;
        Err("suite `cuda-distributed-collectives` requires the `cuda-13` Cargo feature".into())
    }
    #[cfg(feature = "cuda-13")]
    {
        let timeout = Duration::from_secs(30);
        let plugin = load_cuda_13_plugin()?;
        let runtime = DistributedRuntime::initialize_with_options(
            &plugin,
            coordinator,
            rank as u32,
            DistributedRuntimeServiceOptions {
                num_nodes: 2,
                heartbeat_timeout: timeout,
                cluster_register_timeout: timeout,
                shutdown_timeout: timeout,
            },
            DistributedRuntimeClientOptions {
                node_id: rank as u32,
                rpc_timeout: timeout,
                initialization_timeout: timeout,
                shutdown_timeout: timeout,
                heartbeat_timeout: timeout,
                missed_heartbeat_callback: Some(Box::new(|error| {
                    if let Some(error) = error {
                        eprintln!("distributed heartbeat failed: {error}");
                    }
                })),
                ..Default::default()
            },
        )?;
        // The PJRT client borrows the runtime's key-value store. Drop all executable/device buffers and the client
        // before collectively shutting down the coordination client, keeping rank zero's service alive throughout.
        let records = {
            let client = runtime.create_client(
                &plugin,
                ClientOptions::GPU(GpuClientOptions {
                    platform: Some(GpuPlatform::CUDA),
                    visible_devices: Some(vec![0]),
                    node_id: Some(rank),
                    node_count: Some(2),
                    allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                    collective_memory_size: Some(32 * 1024 * 1024),
                    abort_collectives_on_failure: true,
                    ..Default::default()
                }),
            )?;
            if client.process_index()? != rank
                || client.devices()?.len() != 2
                || client.addressable_devices()?.len() != 1
                || client.addressable_devices()?[0].process_index()? != rank
            {
                return Err(
                    format!("rank {rank} requires two global CUDA devices and exactly one local CUDA device").into()
                );
            }
            let mut records = Vec::new();
            for case in selected {
                eprintln!("rank {rank}: executing `{}`", case.id);
                records.push(
                    case.emit(&client, rank)
                        .map_err(|error| format!("rank {rank} case `{}` failed: {error}", case.id))?,
                );
            }
            records
        };
        let key = format!("ryft/differential-workers/completed/{rank}");
        runtime.key_value_store().put(key.as_bytes(), b"ready")?;
        for participant in 0..2 {
            let key = format!("ryft/differential-workers/completed/{participant}");
            if runtime.key_value_store().get(key.as_bytes(), timeout)? != b"ready" {
                return Err(format!("rank {participant} reported an invalid completion record").into());
            }
        }
        runtime.key_value_store().client().shutdown()?;
        Ok(records)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_registry() {
        let cases = registry().unwrap();
        assert!(cases.iter().all(|case| case.participants == 2));
        for operation in ["sum", "all_gather", "all_to_all", "ragged_all_to_all"] {
            assert!(cases.iter().any(|case| case.operation == operation));
        }
    }

    #[test]
    fn test_distributed_collective_case_stage() {
        for case in registry().unwrap() {
            let module = case.stage().unwrap();
            assert!(module.contains("sdy.mesh @mesh = <[\"x\"=2]>"), "{}", case.id);
            assert!(
                module.contains(match case.operation.as_str() {
                    "sum" => "all_reduce",
                    "all_gather" => "all_gather",
                    "all_to_all" => "all_to_all",
                    _ => "ragged_all_to_all",
                }),
                "{}",
                case.id
            );
        }
    }

    #[cfg(not(feature = "cuda-13"))]
    #[test]
    fn test_run_worker_requires_cuda_feature() {
        assert_eq!(
            run_worker(0, "127.0.0.1:1", &[]).unwrap_err().to_string(),
            "suite `cuda-distributed-collectives` requires the `cuda-13` Cargo feature",
        );
    }

    #[test]
    fn test_run_worker_rejects_invalid_rank() {
        assert_eq!(
            run_worker(2, "127.0.0.1:1", &[]).unwrap_err().to_string(),
            "distributed worker rank must be `0` or `1`, got 2"
        );
    }
}
