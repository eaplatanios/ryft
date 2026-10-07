//! Executes explicitly selected ragged exchanges on the real CUDA backend, including on one GPU.

#[cfg(feature = "cuda-13")]
use std::collections::BTreeMap;
use std::error::Error;
#[cfg(feature = "cuda-13")]
use std::sync::Arc;

use serde::Deserialize;

#[cfg(any(test, feature = "cuda-13"))]
use ryft_core::{
    ArrayType, BatchAxis, BatchAxisSpecification, DataType, Differentiate, LogicalMesh, MeshAxis, MeshAxisType,
    ParallelRaggedAllToAll, ProgramError, Sharding, ShardingDimension, Value, ValueProjection, batch, shard_map,
};
#[cfg(feature = "cuda-13")]
use ryft_pjrt::{
    BufferType, ClientOptions, ExecutionDeviceInputs, ExecutionInput, GpuClientOptions, GpuMemoryAllocator,
    GpuPlatform, Program, load_cuda_13_plugin,
};
#[cfg(any(test, feature = "cuda-13"))]
use ryft_xla::experimental::{TracedXlaProgram, XlaArrayTracer, trace};

use super::DifferentialObservation;
#[cfg(feature = "cuda-13")]
use super::{SCHEMA, collective_compilation_options, f32_values_from_bytes};

/// One accelerator experiment shared with JAX and the independent host reference.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RaggedCase {
    /// Stable observation identifier.
    pub(super) id: String,

    /// Logical participant count; the single-GPU suite requires one.
    participants: usize,

    /// Physical operand shape, including a leading mapped dimension for batching.
    local_shape: Vec<usize>,

    /// Physical seeded output shape.
    output_shape: Vec<usize>,

    /// Runtime metadata element type.
    metadata_type: String,

    /// Starting operand row for each transfer.
    input_offsets: Vec<u64>,

    /// Number of operand rows sent in each transfer.
    send_sizes: Vec<u64>,

    /// Starting result row for each transfer.
    output_offsets: Vec<u64>,

    /// Number of result rows received in each transfer.
    receive_sizes: Vec<u64>,

    /// Direct execution or a transform staged inside the manual mesh.
    transform: String,
}

/// Reads and validates the manifest without loading a CUDA plugin.
pub(super) fn registry() -> Result<Vec<RaggedCase>, Box<dyn Error>> {
    let cases: Vec<RaggedCase> = serde_json::from_str(include_str!("ragged_cases.json"))?;
    for (index, case) in cases.iter().enumerate() {
        if cases[..index].iter().any(|previous| previous.id == case.id) {
            return Err(format!("duplicate CUDA collective case `{}`", case.id).into());
        }
        if case.participants != 1
            || case.local_shape.is_empty()
            || case.output_shape.is_empty()
            || case.local_shape.contains(&0)
            || case.output_shape.contains(&0)
            || !matches!(case.metadata_type.as_str(), "i32" | "u64")
            || !matches!(case.transform.as_str(), "primal" | "batch" | "jvp" | "vjp")
            || case.input_offsets.is_empty()
            || [case.send_sizes.len(), case.output_offsets.len(), case.receive_sizes.len()]
                .iter()
                .any(|length| *length != case.input_offsets.len())
        {
            return Err(format!("invalid CUDA collective descriptor `{}`", case.id).into());
        }
        let batch_axis = usize::from(case.transform == "batch");
        if case.local_shape.len() <= batch_axis
            || case.output_shape.len() <= batch_axis
            || case.local_shape[batch_axis + 1..] != case.output_shape[batch_axis + 1..]
            || (batch_axis == 1
                && (case.local_shape[0] != case.output_shape[0] || case.input_offsets.len() % case.local_shape[0] != 0))
        {
            return Err(format!("invalid CUDA collective shapes in `{}`", case.id).into());
        }
        for transfer in 0..case.input_offsets.len() {
            if case.send_sizes[transfer] != case.receive_sizes[transfer]
                || case.input_offsets[transfer]
                    .checked_add(case.send_sizes[transfer])
                    .is_none_or(|end| end > case.local_shape[batch_axis] as u64)
                || case.output_offsets[transfer]
                    .checked_add(case.receive_sizes[transfer])
                    .is_none_or(|end| end > case.output_shape[batch_axis] as u64)
            {
                return Err(format!("invalid single-participant transfer in `{}`", case.id).into());
            }
        }
    }
    Ok(cases)
}

impl RaggedCase {
    /// Metadata shape for ordinary transfers or an unrelated leading batch dimension.
    #[cfg(any(test, feature = "cuda-13"))]
    fn metadata_shape(&self) -> Vec<usize> {
        if self.transform == "batch" {
            vec![self.local_shape[0], self.input_offsets.len() / self.local_shape[0]]
        } else {
            vec![self.input_offsets.len()]
        }
    }

    /// Stages a transform while retaining all four metadata arrays as runtime arguments.
    #[cfg(any(test, feature = "cuda-13"))]
    fn transformed(&self, inputs: Vec<XlaArrayTracer>) -> Result<Vec<XlaArrayTracer>, ProgramError> {
        match self.transform.as_str() {
            "primal" => Ok(vec![
                inputs[0]
                    .parallel_ragged_all_to_all("x", &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5])?,
            ]),
            "batch" => {
                // The leading physical axis maps independent exchanges. The unrelated named axis merges transfers
                // and rebases row offsets, exercising the GPU's widened `u64` metadata path.
                let output = batch(
                    |inputs: Vec<_>| {
                        inputs[0]
                            .parallel_ragged_all_to_all("x", &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5])
                    },
                    inputs,
                    vec![BatchAxis::new(0); 6],
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("y"),
                )?;
                Ok(vec![output])
            }
            "jvp" => {
                // Both floating data inputs are active; integer transfer metadata are fixed runtime captures.
                let active = vec![inputs[0].clone().into_value(), inputs[1].clone().into_value()];
                let captures = inputs[2..6].iter().cloned().map(XlaArrayTracer::into_value).collect::<Vec<_>>();
                let (primal, tangent) =
                    inputs[0].clone().into_value().domain().differentiate_at(active).with_captures(captures).jvp(
                        vec![inputs[6].clone().into_value(), inputs[7].clone().into_value()],
                        |active, captures| {
                            let active = active
                                .into_iter()
                                .map(ValueProjection::<ArrayType>::into_projected)
                                .collect::<Result<Vec<_>, _>>()?;
                            let captures = captures
                                .into_iter()
                                .map(ValueProjection::<ArrayType>::into_projected)
                                .collect::<Result<Vec<_>, _>>()?;
                            Ok(active[0]
                                .parallel_ragged_all_to_all(
                                    "x",
                                    &active[1],
                                    &captures[0],
                                    &captures[1],
                                    &captures[2],
                                    &captures[3],
                                )?
                                .into_value())
                        },
                    )?;
                Ok(vec![
                    ValueProjection::<ArrayType>::into_projected(primal)?,
                    ValueProjection::<ArrayType>::into_projected(tangent)?,
                ])
            }
            "vjp" => {
                let active = vec![inputs[0].clone().into_value(), inputs[1].clone().into_value()];
                let captures = inputs[2..6].iter().cloned().map(XlaArrayTracer::into_value).collect::<Vec<_>>();
                let (primal, pullback) =
                    inputs[0].clone().into_value().domain().differentiate_at(active).with_captures(captures).vjp(
                        |active, captures| {
                            let active = active
                                .into_iter()
                                .map(ValueProjection::<ArrayType>::into_projected)
                                .collect::<Result<Vec<_>, _>>()?;
                            let captures = captures
                                .into_iter()
                                .map(ValueProjection::<ArrayType>::into_projected)
                                .collect::<Result<Vec<_>, _>>()?;
                            Ok(active[0]
                                .parallel_ragged_all_to_all(
                                    "x",
                                    &active[1],
                                    &captures[0],
                                    &captures[1],
                                    &captures[2],
                                    &captures[3],
                                )?
                                .into_value())
                        },
                    )?;
                let cotangents = pullback.apply(inputs[6].clone().into_value())?;
                Ok(vec![
                    ValueProjection::<ArrayType>::into_projected(primal)?,
                    ValueProjection::<ArrayType>::into_projected(cotangents[0].clone())?,
                    ValueProjection::<ArrayType>::into_projected(cotangents[1].clone())?,
                ])
            }
            _ => unreachable!(),
        }
    }

    /// Lowers the finite experiment; this is also available for CPU-only staging coverage.
    #[cfg(any(test, feature = "cuda-13"))]
    fn stage(&self) -> Result<(String, Vec<Vec<usize>>), Box<dyn Error>> {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Manual)?])?;
        let sharding = |shape: &[usize]| -> Result<Sharding, Box<dyn Error>> {
            let mut dimensions = vec![ShardingDimension::replicated(); shape.len()];
            dimensions[0] = ShardingDimension::sharded(["x"]);
            Ok(Sharding::new(mesh.clone(), dimensions)?)
        };
        let mut shapes = vec![self.local_shape.clone(), self.output_shape.clone()];
        shapes.extend(vec![self.metadata_shape(); 4]);
        if self.transform == "jvp" {
            shapes.extend([self.local_shape.clone(), self.output_shape.clone()]);
        } else if self.transform == "vjp" {
            shapes.push(self.output_shape.clone());
        }
        let mut output_shapes = vec![self.output_shape.clone()];
        if self.transform == "jvp" {
            output_shapes.push(self.output_shape.clone());
        } else if self.transform == "vjp" {
            output_shapes.extend([self.local_shape.clone(), self.output_shape.clone()]);
        }
        let metadata_type = if self.metadata_type == "u64" { DataType::U64 } else { DataType::I32 };
        let types = shapes
            .iter()
            .enumerate()
            .map(|(index, shape)| {
                ArrayType::new_static(
                    if (2..6).contains(&index) { metadata_type } else { DataType::F32 },
                    shape.clone(),
                )
            })
            .collect::<Vec<_>>();
        let input_shardings = shapes.iter().map(|shape| sharding(shape)).collect::<Result<Vec<_>, _>>()?;
        let output_shardings = output_shapes.iter().map(|shape| sharding(shape)).collect::<Result<Vec<_>, _>>()?;
        let traced: TracedXlaProgram<Vec<ArrayType>, Vec<ArrayType>> = trace(
            |inputs: Vec<XlaArrayTracer>| {
                shard_map(
                    |inputs| self.transformed(inputs).unwrap(),
                    inputs,
                    mesh.clone(),
                    input_shardings.clone(),
                    output_shardings.clone(),
                )
                .unwrap()
            },
            types,
        )?;
        Ok((traced.to_mlir_module("main")?, output_shapes))
    }

    /// Executes this case exclusively through the CUDA plugin rather than selecting an available default backend.
    pub(super) fn emit(&self) -> Result<DifferentialObservation, Box<dyn Error>> {
        #[cfg(not(feature = "cuda-13"))]
        {
            Err("suite `cuda-collectives` requires the `cuda-13` Cargo feature".into())
        }
        #[cfg(feature = "cuda-13")]
        {
            let (stablehlo, output_shapes) = self.stage()?;
            let plugin = load_cuda_13_plugin()?;
            let client = plugin.client(ClientOptions::GPU(GpuClientOptions {
                platform: Some(GpuPlatform::CUDA),
                allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                ..Default::default()
            }))?;
            let devices = client.addressable_devices()?;
            if devices.len() != 1 {
                return Err(format!(
                    "suite `cuda-collectives` expects one addressable CUDA device, got {}",
                    devices.len()
                )
                .into());
            }
            let mut raw_inputs = Vec::new();
            let count = self.local_shape.iter().product::<usize>();
            let output_count = self.output_shape.iter().product::<usize>();
            let floating = |shape: &[usize], values: Vec<f32>| {
                (
                    values.into_iter().flat_map(f32::to_ne_bytes).collect::<Vec<_>>(),
                    BufferType::F32,
                    shape.iter().map(|extent| *extent as u64).collect::<Vec<_>>(),
                )
            };
            raw_inputs.push(floating(&self.local_shape, (1..=count).map(|value| value as f32).collect()));
            raw_inputs
                .push(floating(&self.output_shape, (0..output_count).map(|value| (100 + value) as f32).collect()));
            for values in [&self.input_offsets, &self.send_sizes, &self.output_offsets, &self.receive_sizes] {
                let (bytes, buffer_type) = if self.metadata_type == "u64" {
                    (values.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<_>>(), BufferType::U64)
                } else {
                    let values = values.iter().map(|value| i32::try_from(*value)).collect::<Result<Vec<_>, _>>()?;
                    (values.into_iter().flat_map(i32::to_ne_bytes).collect::<Vec<_>>(), BufferType::I32)
                };
                raw_inputs.push((
                    bytes,
                    buffer_type,
                    self.metadata_shape().iter().map(|extent| *extent as u64).collect(),
                ));
            }
            if self.transform == "jvp" {
                raw_inputs.push(floating(&self.local_shape, (0..count).map(|index| (index % 5 + 1) as f32).collect()));
                raw_inputs.push(floating(
                    &self.output_shape,
                    (0..output_count).map(|index| (index % 3 + 1) as f32).collect(),
                ));
            } else if self.transform == "vjp" {
                raw_inputs.push(floating(&self.output_shape, (1..=output_count).map(|index| index as f32).collect()));
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
            let executable = client.compile(
                &Program::Mlir { bytecode: stablehlo.as_bytes().to_vec() },
                &collective_compilation_options(1),
            )?;
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
            if outputs.len() != 1 || outputs[0].outputs.len() != output_shapes.len() {
                return Err(format!("unexpected CUDA result count in `{}`", self.id).into());
            }
            let mut observations = BTreeMap::new();
            let names: &[&'static str] = match self.transform.as_str() {
                "jvp" => &["primal", "tangent"],
                "vjp" => &["primal", "operand_cotangent", "seed_cotangent"],
                _ => &["primal"],
            };
            for ((buffer, shape), name) in outputs[0].outputs.iter().zip(&output_shapes).zip(names) {
                let expected = shape.iter().map(|extent| *extent as u64).collect::<Vec<_>>();
                if buffer.element_type()? != BufferType::F32 || buffer.dimensions()? != expected.as_slice() {
                    return Err(
                        format!("CUDA output `{name}` in `{}` has incorrect shape or element type", self.id).into()
                    );
                }
                let bytes = buffer.copy_to_host(None)?.r#await()?;
                if bytes.len() != shape.iter().product::<usize>() * size_of::<f32>() {
                    return Err(format!("CUDA output `{name}` in `{}` has incorrect byte count", self.id).into());
                }
                observations.insert(*name, vec![f32_values_from_bytes(&bytes)]);
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
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_registry() {
        let cases = registry().unwrap();
        assert!(cases.iter().all(|case| case.participants == 1));
        for transform in ["primal", "batch", "jvp", "vjp"] {
            assert!(cases.iter().any(|case| case.transform == transform));
        }
        for metadata_type in ["i32", "u64"] {
            assert!(cases.iter().any(|case| case.metadata_type == metadata_type));
        }
    }

    #[test]
    fn test_ragged_case_stage() {
        for case in registry().unwrap() {
            let (module, shapes) = case.stage().unwrap();
            assert!(module.contains("ragged_all_to_all"), "{}", case.id);
            assert_eq!(shapes[0], case.output_shape);
            assert_eq!(
                shapes.len(),
                if case.transform == "vjp" {
                    3
                } else if case.transform == "jvp" {
                    2
                } else {
                    1
                }
            );
        }
    }
}
