//! General XLA program tracing: the [`trace`] entry point stages an arbitrary function over [`XlaArrayTracer`]s in a
//! fresh root tracing context of [`XlaDomain`], and its [`TracedXlaProgram`] result reports the statistics of the
//! staged program and lowers it to a StableHLO/Shardy MLIR module through the [`lowering`] module. [`TraceError`]
//! reports the failures of both stages.
//!
//! Traced functions may compose `shard_map` with other transforms (e.g., differentiation or batching). The logical
//! semantics of `shard_map` (boundary metadata, validation, closure tracing, and transform rules) are owned by the
//! [`shard_map`](mod@ryft_core::operations::sharding::shard_map) module of `ryft-core`, and its manual computations
//! lower to Shardy `sdy.manual_computation` operations in the [`lowering`] module, which also lowers
//! [`TracedShardMap`](ryft_core::TracedShardMap)s traced for [`XlaDomain`] directly through
//! [`to_mlir_module`](crate::experimental::lowering::to_mlir_module).

use std::cell::RefCell;

use ryft_core::{
    ArrayIrType, ArrayType, DomainTracingContext, ParameterError, Parameterized, ParameterizedFamily, ProgramError,
    ProgramStatistics, ProjectedValue, Sharding, ValueProjection,
};
use thiserror::Error;

use crate::experimental::domains::{XlaDomain, XlaTracer};
use crate::experimental::ops::{XlaConstant, XlaProgram};

use super::lowering::LoweringError;

/// Error type for tracing XLA programs (including programs that stage `shard_map` boundaries) and lowering them to
/// StableHLO/Shardy MLIR. It wraps the underlying core and lowering errors without flattening them into messages, so
/// callers can match on the original error. Shard-map construction and tracing functions (e.g.,
/// [`ryft_core::shard_map()`]) report the logical [`ShardMapError`](ryft_core::ShardMapError) themselves, and a
/// shard-map error that a transform rule raises while a program is traced or lowered arrives as
/// [`ProgramError::Custom`], from which [`ProgramError::downcast_custom`] recovers it.
#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum TraceError {
    /// Underlying program error returned while staging or simplifying a traced program.
    #[error("{0}")]
    Program(#[from] ProgramError),

    /// Underlying parameter-structure error returned while reparameterizing traced values.
    #[error("{0}")]
    Parameter(#[from] ParameterError),

    /// Error returned while lowering a traced program to StableHLO/Shardy MLIR.
    #[error("{0}")]
    Lowering(#[from] LoweringError),
}

/// Array value that closures traced through [`trace`] receive and return: a [`DomainTracer`](ryft_core::DomainTracer)
/// of [`XlaDomain`] projected onto its array member. It is also the [`ShardMapTracer`](ryft_core::ShardMapTracer) of
/// [`XlaDomain`], so `shard_map` bodies traced in XLA programs (e.g., through [`ryft_core::shard_map()`] on values of
/// type [`XlaArrayTracer`], or through [`ryft_core::trace_shard_map`] for [`XlaDomain`]) receive the same values. This
/// alias is public so that callers (e.g., tooling binaries) can annotate traced closure parameters.
pub type XlaArrayTracer = ProjectedValue<ArrayType, XlaTracer<'static>>;

/// Stages an arbitrary traced XLA function over global array types.
///
/// This is the general XLA tracing entry point used when callers want to compose `shard_map` with differentiation
/// transforms and then lower the resulting whole program to StableHLO/Shardy MLIR. `function` is traced once, in a
/// fresh root tracing context of [`XlaDomain`] that binds no named axes, over [`XlaArrayTracer`]s of
/// `global_input_types`, and the traced program is simplified.
///
/// # Parameters
///
///   - `function`: Function to trace over global XLA values.
///   - `global_input_types`: Global input array types passed to the traced function.
///
/// # Errors
///
/// Returns [`TraceError::Program`] when tracing or simplifying the program fails (e.g., because `function` stages an
/// invalid operation or registers a capture, which the fresh root context discards), and [`TraceError::Parameter`]
/// when the traced inputs or outputs cannot be restructured into `Input` or `Output`.
pub fn trace<F, Input, Output>(
    function: F,
    global_input_types: Input,
) -> Result<TracedXlaProgram<Input, Output>, TraceError>
where
    Input: Parameterized<ArrayType>,
    Input::Family: ParameterizedFamily<XlaConstant> + ParameterizedFamily<XlaArrayTracer>,
    Output: Parameterized<ArrayType>,
    Output::Family: ParameterizedFamily<XlaConstant> + ParameterizedFamily<XlaArrayTracer>,
    F: FnOnce(Input::To<XlaArrayTracer>) -> Output::To<XlaArrayTracer>,
{
    let input_structure = global_input_types.parameter_structure();
    let output_structure = RefCell::new(None);
    let (output_types, program) = DomainTracingContext::<XlaDomain<'static>>::trace(
        |inputs: Vec<XlaTracer<'static>>| {
            let inputs = inputs
                .into_iter()
                .map(|input| ValueProjection::<ArrayType>::into_projected(input).map_err(ProgramError::from))
                .collect::<Result<Vec<_>, _>>()?;
            let outputs = function(Input::To::<XlaArrayTracer>::from_parameters(input_structure.clone(), inputs)?);
            output_structure.replace(Some(outputs.parameter_structure()));
            Ok(outputs.into_parameters().map(ProjectedValue::into_value).collect::<Vec<_>>())
        },
        global_input_types.parameters().cloned().map(ArrayIrType::Array).collect::<Vec<_>>(),
    )?;
    let output_structure = output_structure.into_inner().ok_or_else(|| {
        ProgramError::MalformedProgram("XLA tracing completed without recording its output structure".to_string())
    })?;
    let global_output_types = Output::from_parameters(
        output_structure.clone(),
        output_types
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProgramError::from)?,
    )?;
    let program = program
        .simplified()?
        .restructured::<Input::To<XlaConstant>, Output::To<XlaConstant>>(input_structure, output_structure)?;
    Ok(TracedXlaProgram { global_input_types, global_output_types, program })
}

/// Traced XLA program backed by a staged [`Program`](ryft_core::Program), as returned by [`trace`].
pub struct TracedXlaProgram<Input: Parameterized<ArrayType>, Output: Parameterized<ArrayType>>
where
    Input::Family: ParameterizedFamily<XlaConstant>,
    Output::Family: ParameterizedFamily<XlaConstant>,
{
    /// Global input types supplied to the traced function.
    global_input_types: Input,

    /// Global output types inferred by tracing the function.
    global_output_types: Output,

    /// Staged traced XLA program specialized to abstract array leaves.
    program: XlaProgram<Input::To<XlaConstant>, Output::To<XlaConstant>>,
}

impl<Input: Parameterized<ArrayType>, Output: Parameterized<ArrayType>> TracedXlaProgram<Input, Output>
where
    Input::Family: ParameterizedFamily<XlaConstant>,
    Output::Family: ParameterizedFamily<XlaConstant>,
{
    /// Returns backend-neutral structural statistics for the staged traced XLA program backing this handle. The
    /// statistics describe the program as [`trace`] stores it, which it has already simplified (refer to
    /// [`Program::simplified`](ryft_core::Program::simplified)): instructions that reach no program output are not
    /// counted unless they have observable effects, while the boundary pruning of operations with attached regions
    /// (refer to [`Program::into_pruned`](ryft_core::Program::into_pruned)), which lowering applies, is not reflected
    /// in the counts. Refer to the documentation of [`Program::statistics`](ryft_core::Program::statistics) for the
    /// precise semantics of the reported statistics.
    pub fn statistics(&self) -> ProgramStatistics {
        self.program.statistics()
    }

    /// Returns the traced global input types.
    pub fn global_input_types(&self) -> &Input {
        &self.global_input_types
    }

    /// Returns the traced global output types.
    pub fn global_output_types(&self) -> &Output {
        &self.global_output_types
    }

    /// Renders a full StableHLO/Shardy MLIR module for this traced XLA program, without signature shardings (refer to
    /// [`Self::to_mlir_module_with_signature_shardings`]).
    ///
    /// # Parameters
    ///
    ///   - `function_name`: Symbol name to use for the outer `func.func`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::to_mlir_module_with_signature_shardings`].
    pub fn to_mlir_module<S: AsRef<str>>(&self, function_name: S) -> Result<String, TraceError> {
        self.to_mlir_module_with_signature_shardings(function_name, None, None)
    }

    /// Same as [`Self::to_mlir_module`] but additionally attaches `sdy.sharding` attributes to the function's arguments
    /// and/or results when shardings are provided. This is what the XLA SPMD partitioner reads to drive boundary
    /// slicing of per-device output buffers, including for shapes whose dimensions are not divisible by the partition
    /// count (e.g., shape `[5]` on 2 partitions producing `[3]` + `[2]`).
    ///
    /// The traced program is pruned before lowering (refer to
    /// [`Program::into_pruned`](ryft_core::Program::into_pruned)), which drops dead outputs of region-carrying
    /// instructions together with the body computation that only they use (e.g., the dead primal outputs of a
    /// forward-mode `shard_map` whose residuals feed a gradient), and is then simplified. A `shard_map` whose ordinary
    /// outputs are all dead but whose body has observable effects (e.g., prints with the
    /// [`DeviceOrderedIo`](ryft_core::EffectClass::DeviceOrderedIo) effect) is kept with no ordinary outputs and lowers
    /// to a manual computation that only threads its effect token.
    ///
    /// # Parameters
    ///
    ///   - `function_name`: Symbol name to use for the outer `func.func`.
    ///   - `argument_shardings`: Optional shardings to attach to each function argument. Must have the same length as
    ///     the global input types, or be `None`.
    ///   - `result_shardings`: Optional shardings to attach to each function result. Must have the same length as the
    ///     global output types, or be `None`.
    ///
    /// # Errors
    ///
    /// Returns [`TraceError::Program`] when pruning or simplifying the traced program fails, and
    /// [`TraceError::Lowering`] when lowering it fails (e.g., for an invalid function name, mismatched signature
    /// shardings, unresolved references or state, or operations that the backend cannot lower).
    pub fn to_mlir_module_with_signature_shardings<S: AsRef<str>>(
        &self,
        function_name: S,
        argument_shardings: Option<&[Sharding]>,
        result_shardings: Option<&[Sharding]>,
    ) -> Result<String, TraceError> {
        let simplified_program = self.program.clone().into_pruned()?.simplified()?;
        super::lowering::to_mlir_module_for_program(
            &simplified_program,
            &[],
            &self.global_input_types,
            &self.global_output_types,
            function_name,
            argument_shardings,
            result_shardings,
        )
        .map_err(TraceError::from)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Array, BatchAxis, BatchAxisSpecification, CollectiveOptions, ConstrainSharding, Context, DataType, Device,
        DeviceMesh, Differentiate, Dimension, EffectClass, LogicalMesh, MeshAxis, MeshAxisType, Mul, NamedAxis,
        ParallelAllGather, ParallelAllGatherOutputVariance, ParallelRaggedAllToAll, ParallelVary, Placeholder, Print,
        Reduce, ReductionKind, RegionRole, Reshape, Reshard, Shape, ShardMap, ShardMapContext, ShardMapOperation,
        Sharding, ShardingDimension, Sin, StagingContext, Typed, Value, batch, shard_map, shard_map_in_context,
        shard_map_with_options,
    };
    use ryft_pjrt::{BufferType, ClientOptions, CpuClientOptions, Program, load_cpu_plugin};
    #[cfg(feature = "cuda-13")]
    use ryft_pjrt::{GpuClientOptions, GpuMemoryAllocator, GpuPlatform, load_cuda_13_plugin};

    use crate::experimental::lowering::tests::{
        static_sharded_array_type, test_sharding, test_spmd_compilation_options,
    };
    use crate::experimental::ops::{XlaOperation, XlaProgramBuilder};
    use crate::tests::{values_from_bytes, values_to_bytes};
    use crate::{FromPjrt, XlaArray, XlaSession};

    use super::*;

    /// Executes `local.parallel_reduce(kind, "x")` inside a `shard_map` over the manual axis `"x"` of four CPU devices,
    /// on the global vector of `4 · shard_size` elements of `data_type` whose device shards hold the raw `shards`, and
    /// returns the raw bytes of every device's output shard in device order.
    fn execute_parallel_reduce_on_cpu(
        kind: ReductionKind,
        data_type: DataType,
        buffer_type: BufferType,
        shard_size: usize,
        shards: [Vec<u8>; 4],
    ) -> Vec<Vec<u8>> {
        use ryft_core::ParallelReduce;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = XlaSession::new(&client).domain();
        let devices = client.addressable_devices().unwrap();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let device_mesh =
            DeviceMesh::new(mesh.clone(), devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect())
                .unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let sharding = sharding.clone();
                move |x: XlaArrayTracer| {
                    shard_map(
                        |local_x: XlaArrayTracer| local_x.parallel_reduce(kind, "x").unwrap(),
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .unwrap()
                }
            },
            ArrayType::new_static(data_type, [4 * shard_size]),
        )
        .unwrap();
        let program = Program::Mlir { bytecode: traced.to_mlir_module("main").unwrap().into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(4)).unwrap();
        let buffers = devices
            .iter()
            .zip(shards)
            .map(|(device, shard)| {
                client
                    .buffer(shard.as_slice(), buffer_type, [shard_size as u64], None, device.clone(), None)
                    .unwrap()
            })
            .collect();
        let input = XlaArray::from_addressable_buffers(
            &domain,
            static_sharded_array_type(data_type, &[4 * shard_size], sharding),
            device_mesh,
            buffers,
        )
        .unwrap();
        let device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let arguments = XlaArray::into_execute_arguments(vec![input], &device_ids).unwrap();
        executable
            .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap()
            .into_iter()
            .map(|output| output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap())
            .collect()
    }

    /// Traces a program over a sharded `f64[4]` input that invokes a `shard_map` over the two-device manual mesh, whose
    /// body prints its local shard with the `DeviceOrderedIo` effect and whose only output is dead, and that returns
    /// its input unchanged.
    fn traced_effectful_dead_shard_map() -> (TracedXlaProgram<ArrayType, ArrayType>, Sharding) {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let traced = trace(
            {
                let sharding = sharding.clone();
                move |x: XlaArrayTracer| {
                    let _printed = shard_map(
                        |local_x: XlaArrayTracer| {
                            local_x.print_with_effect_class("body", EffectClass::DeviceOrderedIo).unwrap()
                        },
                        x.clone(),
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .unwrap();
                    x
                }
            },
            ArrayType::new_static(DataType::F64, [4]),
        )
        .unwrap();
        (traced, sharding)
    }

    #[test]
    fn test_trace() {
        let traced: TracedXlaProgram<(ArrayType, ArrayType), ArrayType> = trace(
            |(x, y): (XlaArrayTracer, XlaArrayTracer)| x.sin().unwrap() * y,
            (ArrayType::new_static(DataType::F32, [4]), ArrayType::new_static(DataType::F32, [4])),
        )
        .unwrap();
        assert_eq!(
            traced.global_input_types(),
            &(ArrayType::new_static(DataType::F32, [4]), ArrayType::new_static(DataType::F32, [4])),
        );
        assert_eq!(traced.global_output_types(), &ArrayType::new_static(DataType::F32, [4]));
        assert_eq!(
            traced.program.to_string(),
            indoc! {"
                lambda %0:f32[4], %1:f32[4] .
                let %2:f32[4] = sin %0
                    %3:f32[4] = mul %2 %1
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_trace_reshard_renders_mlir() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]);
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));

        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let sharding = sharding.clone();
                move |x: XlaArrayTracer| {
                    x.sin().unwrap().reshard(&sharding).expect("reshard should stage on traced XLA values")
                }
            },
            global_input_type.clone(),
        )
        .unwrap();

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
                    %0 = stablehlo.sine %arg0 : tensor<8xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{"x"}]> : tensor<8xf32>
                    return %1 : tensor<8xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_trace_reshard_then_reshape_renders_mlir() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let sharding = sharding.clone();
                move |x: XlaArrayTracer| {
                    x.reshard(&sharding)
                        .expect("reshard should stage before reshape")
                        .reshape(Shape::new(vec![Dimension::Static(1), Dimension::Static(8), Dimension::Static(1)]))
                        .unwrap()
                }
            },
            input_type,
        )
        .unwrap();

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32>) -> tensor<1x8x1xf32> {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{"x"}]> : tensor<8xf32>
                    %1 = stablehlo.reshape %0 : (tensor<8xf32>) -> tensor<1x8x1xf32>
                    return %1 : tensor<1x8x1xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_trace_constrain_sharding_merges_tracked_placement_into_mlir() {
        // The input is tracked as sharded over the explicit axis and the constraint places it over the auto axis, so
        // the emitted constraint must carry both placements rather than the constraint alone, which would contradict
        // the tracked type by marking the explicit axis replicated.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let tracked = test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]);
        let constraint = test_sharding(&mesh, vec![ShardingDimension::sharded(["a"])], vec![]);
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
            .with_sharding(tracked.clone())
            .unwrap();

        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let constraint = constraint.clone();
                move |x: XlaArrayTracer| {
                    x.constrain_sharding(&constraint).expect("constraint should stage on traced XLA values")
                }
            },
            global_input_type.clone(),
        )
        .unwrap();

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2, "a"=2]>
                  func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{"x", "a"}]> : tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#},
        );
    }

    /// Verifies that [`TracedXlaProgram::statistics`] delegates to the stored (simplified) staged program. The asserted
    /// numbers are cross-checked against `traced.program.to_string()`, which renders the entry region as one
    /// `shard_map` instruction attaching a single-`sin` body region.
    #[test]
    fn test_traced_xla_program_statistics() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = mesh.clone();
                move |x: XlaArrayTracer| {
                    shard_map(
                        |local_x: XlaArrayTracer| local_x.sin().unwrap(),
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .unwrap()
                }
            },
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)])),
        )
        .unwrap();

        let statistics = traced.statistics();
        assert_eq!(statistics.region_count(), 2);

        let body = &statistics.regions()[0];
        assert_eq!(body.input_count(), 1);
        assert_eq!(body.output_count(), 1);
        assert_eq!(body.instruction_count(), 1);
        assert_eq!(body.constant_count(), 0);
        assert_eq!(body.operation_counts(), &BTreeMap::from([("sin", 1usize)]));
        assert_eq!(body.maximum_output_dependency_depth(), 1);
        assert_eq!(body.attached_regions(), &[]);

        let entry = statistics.entry_region_statistics();
        assert_eq!(entry.input_count(), 1);
        assert_eq!(entry.output_count(), 1);
        assert_eq!(entry.instruction_count(), 1);
        assert_eq!(entry.operation_counts(), &BTreeMap::from([("shard_map", 1usize)]));
        assert_eq!(entry.maximum_output_dependency_depth(), 1);
        assert_eq!(entry.attached_regions().len(), 1);
        let edge = &entry.attached_regions()[0];
        assert_eq!(edge.instruction_index(), 0);
        assert_eq!(edge.operation(), "shard_map");
        assert_eq!(edge.region_slot(), "body");
        assert_eq!(edge.region_role(), RegionRole::Computation);
        assert_eq!(edge.region_index(), 0);
        assert_eq!(edge.label(), "shard_map.body");
    }

    /// Pruning keeps a `shard_map` whose ordinary results are all dead when its body has observable effects, with no
    /// ordinary results, so the lowered manual computation only threads the effect token of its body.
    #[test]
    fn test_traced_xla_program_to_mlir_module_with_signature_shardings_keeps_effectful_dead_shard_maps() {
        let (traced, sharding) = traced_effectful_dead_shard_map();
        assert_eq!(
            traced
                .to_mlir_module_with_signature_shardings(
                    "main",
                    Some(std::slice::from_ref(&sharding)),
                    Some(std::slice::from_ref(&sharding)),
                )
                .unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<4xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1 = sdy.manual_computation(%arg0, %0) in_shardings=[<@mesh, [{"x"}]>, <@mesh, []>] out_shardings=[<@mesh, []>] manual_axes={"x"} (%arg1: tensor<2xf64>, %arg2: !stablehlo.token) {
                      %2 = stablehlo.custom_call @ryft.print(%arg1, %arg2) {api_version = 4 : i32, backend_config = {label = "body"}, has_side_effect = true} : (tensor<2xf64>, !stablehlo.token) -> !stablehlo.token
                      sdy.return %2 : !stablehlo.token
                    } : (tensor<4xf64>, !stablehlo.token) -> !stablehlo.token
                    return %arg0 : tensor<4xf64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_traced_xla_program_effectful_dead_shard_map_executes_on_cpu() {
        use ryft_pjrt::{ExecutionDeviceInputs, ExecutionInput};

        use crate::experimental::debugging::{ensure_print_handler_registered, with_captured_prints};

        let (traced, sharding) = traced_effectful_dead_shard_map();
        let module = traced
            .to_mlir_module_with_signature_shardings(
                "main",
                Some(std::slice::from_ref(&sharding)),
                Some(std::slice::from_ref(&sharding)),
            )
            .unwrap();
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        ensure_print_handler_registered(&client).unwrap();
        let executable = client
            .compile(&Program::Mlir { bytecode: module.into_bytes() }, &test_spmd_compilation_options(2))
            .unwrap();
        let shards = [vec![1.5f64, 2.5], vec![3.5f64, 4.5]];
        let inputs = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .zip(&shards)
            .map(|(device, shard)| {
                let bytes = values_to_bytes(shard.as_slice());
                vec![ExecutionInput {
                    buffer: std::sync::Arc::new(
                        client.buffer(bytes.as_slice(), BufferType::F64, [2u64], None, device.clone(), None).unwrap(),
                    ),
                    donatable: false,
                }]
            })
            .collect::<Vec<_>>();
        let (outputs, mut lines) = with_captured_prints(|| {
            executable
                .execute(
                    inputs.iter().map(|inputs| ExecutionDeviceInputs { inputs, ..Default::default() }).collect(),
                    Vec::new(),
                    0,
                    None,
                    Some(file!()),
                    None,
                    None,
                )
                .unwrap()
                .block_until_ready()
                .unwrap()
        });

        // Every device returns its own shard and prints it exactly once, although nothing uses the `shard_map` output.
        assert_eq!(outputs.len(), 2);
        for (output, shard) in outputs.iter().zip(&shards) {
            assert_eq!(output.outputs.len(), 1);
            let bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(&values_from_bytes::<f64>(&bytes), shard);
        }
        lines.sort();
        assert_eq!(lines, vec!["body: [1.5, 2.5]".to_string(), "body: [3.5, 4.5]".to_string()]);
    }

    #[test]
    fn test_parallel_ragged_all_to_all_target_aware_lowering_rejects_cpu() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]);
        let input_types = vec![
            ArrayType::new_static(DataType::F32, [6]),
            ArrayType::new_static(DataType::F32, [8]),
            ArrayType::new_static(DataType::I32, [4]),
            ArrayType::new_static(DataType::I32, [4]),
            ArrayType::new_static(DataType::I32, [4]),
            ArrayType::new_static(DataType::I32, [4]),
        ];
        let traced: TracedXlaProgram<Vec<ArrayType>, ArrayType> = trace(
            {
                let shardings = vec![sharding.clone(); 6];
                let output_sharding = sharding;
                move |inputs: Vec<XlaArrayTracer>| {
                    shard_map(
                        |inputs: Vec<XlaArrayTracer>| {
                            inputs[0]
                                .parallel_ragged_all_to_all(
                                    "x", &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5],
                                )
                                .unwrap()
                        },
                        inputs,
                        mesh.clone(),
                        shardings.clone(),
                        output_sharding.clone(),
                    )
                    .unwrap()
                }
            },
            input_types,
        )
        .unwrap();

        assert!(matches!(
            crate::experimental::lowering::lower_mlir_module_for_program(
                &traced.program,
                &[],
                &traced.global_input_types,
                &traced.global_output_types,
                "main",
                None,
                None,
                Some("cpu"),
            ),
            Err(crate::experimental::lowering::LoweringError::Tracing(
                ProgramError::UnsupportedOperation { message },
            )) if message == "`parallel_ragged_all_to_all` is not supported by the XLA CPU backend",
        ));
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_parallel_ragged_all_to_all_forward_and_overlapping_send_transpose_execute_on_cuda() {
        let plugin = load_cuda_13_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::GPU(GpuClientOptions {
                platform: Some(GpuPlatform::CUDA),
                allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                ..Default::default()
            }))
            .unwrap();
        let domain = XlaSession::new(&client).domain();
        let client_devices = client.addressable_devices().unwrap();
        if client_devices.len() < 2 {
            return;
        }
        let client_devices = &client_devices[..2];
        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let sharding =
            Sharding::new(device_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_types = vec![
            ArrayType::new_static(DataType::I32, [6]),
            ArrayType::new_static(DataType::I32, [8]),
            ArrayType::new_static(DataType::I32, [4]),
            ArrayType::new_static(DataType::I32, [4]),
            ArrayType::new_static(DataType::I32, [4]),
            ArrayType::new_static(DataType::I32, [4]),
        ];
        let traced: TracedXlaProgram<Vec<ArrayType>, ArrayType> = trace(
            {
                let mesh = device_mesh.logical_mesh().clone();
                let shardings = vec![sharding.clone(); 6];
                let output_sharding = sharding.clone();
                move |inputs: Vec<XlaArrayTracer>| {
                    shard_map(
                        |inputs: Vec<XlaArrayTracer>| {
                            inputs[0]
                                .parallel_ragged_all_to_all(
                                    "x", &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5],
                                )
                                .unwrap()
                        },
                        inputs,
                        mesh.clone(),
                        shardings.clone(),
                        output_sharding.clone(),
                    )
                    .unwrap()
                }
            },
            input_types,
        )
        .unwrap();
        let mlir_program = traced.to_mlir_module("main").unwrap();
        let local_inputs = [
            [[1_i32, 2, 2].as_slice(), [3_i32, 4, 0].as_slice()],
            [[0_i32, 0, 0, 0].as_slice(), [0_i32, 0, 0, 0].as_slice()],
            [[0_i32, 1].as_slice(), [0_i32, 1].as_slice()],
            [[1_i32, 2].as_slice(), [1_i32, 1].as_slice()],
            [[0_i32, 0].as_slice(), [1_i32, 2].as_slice()],
            [[1_i32, 1].as_slice(), [2_i32, 1].as_slice()],
        ];
        let input_arrays = local_inputs
            .iter()
            .zip([6, 8, 4, 4, 4, 4])
            .map(|(values, global_extent)| {
                let buffers = client_devices
                    .iter()
                    .zip(values)
                    .map(|(device, values)| {
                        client
                            .buffer(
                                values_to_bytes::<i32>(values).as_slice(),
                                BufferType::I32,
                                [values.len() as u64],
                                None,
                                device.clone(),
                                None,
                            )
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                XlaArray::from_addressable_buffers(
                    &domain,
                    static_sharded_array_type(DataType::I32, &[global_extent], sharding.clone()),
                    device_mesh.clone(),
                    buffers,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let executable = client
            .compile(&Program::Mlir { bytecode: mlir_program.into_bytes() }, &test_spmd_compilation_options(2))
            .unwrap();
        let execution_device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let execute_arguments =
            XlaArray::into_execute_arguments(input_arrays, execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();

        let expected = client_devices
            .iter()
            .zip([[1_i32, 3, 0, 0], [2_i32, 2, 4, 0]])
            .map(|(device, values)| (device.id().unwrap(), values))
            .collect::<HashMap<_, _>>();
        for (output, device_id) in outputs.into_iter().zip(execution_device_ids.iter().copied()) {
            let bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(values_from_bytes::<i32>(bytes.as_slice()), *expected.get(&device_id).unwrap());
        }

        let gradient_input_types = vec![
            ArrayType::new_static(DataType::F32, [6]),
            ArrayType::new_static(DataType::F32, [8]),
            ArrayType::new_static(DataType::I32, [4]),
            ArrayType::new_static(DataType::I32, [4]),
            ArrayType::new_static(DataType::I32, [4]),
            ArrayType::new_static(DataType::I32, [4]),
        ];
        let gradient: TracedXlaProgram<Vec<ArrayType>, ArrayType> = trace(
            {
                let mesh = device_mesh.logical_mesh().clone();
                let shardings = vec![sharding.clone(); 6];
                let output_sharding = sharding.clone();
                move |inputs: Vec<XlaArrayTracer>| {
                    shard_map(
                        |inputs: Vec<XlaArrayTracer>| {
                            let context = inputs[0].domain();
                            context
                                .differentiate_at(inputs[0].clone())
                                .with_captures((
                                    inputs[1].clone(),
                                    inputs[2].clone(),
                                    inputs[3].clone(),
                                    inputs[4].clone(),
                                    inputs[5].clone(),
                                ))
                                .gradient(
                                    |operand, (output, input_offsets, send_sizes, output_offsets, receive_sizes)| {
                                        operand
                                            .parallel_ragged_all_to_all(
                                                "x",
                                                &output,
                                                &input_offsets,
                                                &send_sizes,
                                                &output_offsets,
                                                &receive_sizes,
                                            )
                                            .map(|result| result.reduce(&[0], ReductionKind::Sum).unwrap())
                                    },
                                )
                                .unwrap()
                        },
                        inputs,
                        mesh.clone(),
                        shardings.clone(),
                        output_sharding.clone(),
                    )
                    .unwrap()
                }
            },
            gradient_input_types,
        )
        .unwrap();
        let gradient_program = gradient.to_mlir_module("main").unwrap();
        let local_gradient_data =
            [[vec![10.0_f32, 11.0, 12.0], vec![20.0_f32, 21.0, 22.0]], [vec![0.0_f32; 4], vec![0.0_f32; 4]]];
        let mut gradient_inputs = local_gradient_data
            .iter()
            .zip([6, 8])
            .map(|(values, global_extent)| {
                let buffers = client_devices
                    .iter()
                    .zip(values)
                    .map(|(device, values)| {
                        client
                            .buffer(
                                values_to_bytes::<f32>(values).as_slice(),
                                BufferType::F32,
                                [values.len() as u64],
                                None,
                                device.clone(),
                                None,
                            )
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                XlaArray::from_addressable_buffers(
                    &domain,
                    static_sharded_array_type(DataType::F32, &[global_extent], sharding.clone()),
                    device_mesh.clone(),
                    buffers,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let local_gradient_metadata = [
            [vec![0_i32, 0], vec![0_i32, 0]],
            [vec![1_i32, 1], vec![1_i32, 1]],
            [vec![0_i32, 0], vec![1_i32, 1]],
            [vec![1_i32, 1], vec![1_i32, 1]],
        ];
        gradient_inputs.extend(local_gradient_metadata.iter().map(|values| {
            let buffers = client_devices
                .iter()
                .zip(values)
                .map(|(device, values)| {
                    client
                        .buffer(
                            values_to_bytes::<i32>(values).as_slice(),
                            BufferType::I32,
                            [values.len() as u64],
                            None,
                            device.clone(),
                            None,
                        )
                        .unwrap()
                })
                .collect::<Vec<_>>();
            XlaArray::from_addressable_buffers(
                &domain,
                static_sharded_array_type(DataType::I32, &[4], sharding.clone()),
                device_mesh.clone(),
                buffers,
            )
            .unwrap()
        }));
        let gradient_executable = client
            .compile(&Program::Mlir { bytecode: gradient_program.into_bytes() }, &test_spmd_compilation_options(2))
            .unwrap();
        let gradient_device_ids = gradient_executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let gradient_arguments =
            XlaArray::into_execute_arguments(gradient_inputs, gradient_device_ids.as_slice()).unwrap();
        let gradients = gradient_executable
            .execute(gradient_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();
        for gradient in gradients {
            let bytes = gradient.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(values_from_bytes::<f32>(bytes.as_slice()), vec![2.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn test_shard_map_parallel_sum_lowers_to_all_reduce_and_executes_on_cpu() {
        use ryft_core::{ParallelReduce, ReductionKind};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .expect("failed to create 4-device CPU client");
        let domain = XlaSession::new(&client).domain();
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 4);

        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let sharding =
            Sharding::new(device_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));

        // A sum `parallel_reduce` over the manual mesh axis `"x"` resolves against the seeded body trace and lowers to
        // a `stablehlo.all_reduce` whose replica group spans the four devices along `"x"`, so every shard receives the
        // elementwise sum of all four local shards.
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = device_mesh.logical_mesh().clone();
                let sharding = sharding.clone();
                move |x: XlaArrayTracer| {
                    shard_map(
                        |local_x: XlaArrayTracer| local_x.parallel_reduce(ReductionKind::Sum, "x").unwrap(),
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .expect("shard_map with a sum parallel_reduce should trace")
                }
            },
            global_input_type,
        )
        .unwrap();

        let mlir_program = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            mlir_program,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %2 : tensor<f32>
                      }) : (tensor<2xf32>) -> tensor<2xf32>
                      sdy.return %1 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#},
        );

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let shard_values = [device_index as f32 * 2.0 + 1.0, device_index as f32 * 2.0 + 2.0];
                client
                    .buffer(
                        values_to_bytes::<f32>(&shard_values).as_slice(),
                        BufferType::F32,
                        [2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array = XlaArray::from_addressable_buffers(
            &domain,
            static_sharded_array_type(DataType::F32, &[8], sharding),
            device_mesh,
            input_buffers,
        )
        .unwrap();
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(4)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 4);
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        let execute_arguments =
            XlaArray::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();

        // Shards are [1, 2], [3, 4], [5, 6], [7, 8], so every device receives [1+3+5+7, 2+4+6+8] = [16, 20].
        assert_eq!(outputs.len(), execution_device_ids.len());
        for output in outputs {
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values: [f32; 2] = values_from_bytes::<f32>(output_bytes.as_slice()).try_into().unwrap();
            assert_eq!(values, [16.0, 20.0]);
        }
    }

    #[test]
    fn test_shard_map_parallel_reduce_kinds_execute_on_cpu() {
        // Every primitive kind lowers to a `stablehlo.all_reduce` with the scalar combiner of the matching `reduce`, so
        // every device receives the elementwise reduction of the four local shards.
        let f32_shards = [[1.0f32, -2.0], [3.0, 4.0], [-5.0, 0.5], [2.0, 8.0]].map(|shard| values_to_bytes(&shard));
        for (kind, expected) in [
            (ReductionKind::Sum, [1.0f32, 10.5]),
            (ReductionKind::Product, [-30.0, -32.0]),
            (ReductionKind::Mean, [0.25, 2.625]),
            (ReductionKind::Max, [3.0, 8.0]),
            (ReductionKind::Min, [-5.0, -2.0]),
        ] {
            for output in execute_parallel_reduce_on_cpu(kind, DataType::F32, BufferType::F32, 2, f32_shards.clone()) {
                assert_eq!(values_from_bytes::<f32>(&output), expected, "{kind}");
            }
        }
        let i32_shards = [[1i32, -2], [3, 4], [-5, 7], [2, 8]].map(|shard| values_to_bytes(&shard));
        for (kind, expected) in [
            (ReductionKind::Sum, [1i32, 17]),
            (ReductionKind::Product, [-30, -448]),
            (ReductionKind::Max, [3, 8]),
            (ReductionKind::Min, [-5, -2]),
        ] {
            for output in execute_parallel_reduce_on_cpu(kind, DataType::I32, BufferType::I32, 2, i32_shards.clone()) {
                assert_eq!(values_from_bytes::<i32>(&output), expected, "{kind}");
            }
        }
        let boolean_shards = [[true, false, false], [true, true, false], [true, false, false], [true, false, false]]
            .map(|shard| values_to_bytes(&shard));
        for (kind, expected) in [(ReductionKind::Any, [true, true, false]), (ReductionKind::All, [true, false, false])]
        {
            for output in execute_parallel_reduce_on_cpu(
                kind,
                DataType::Boolean,
                BufferType::Predicate,
                3,
                boolean_shards.clone(),
            ) {
                assert_eq!(values_from_bytes::<bool>(&output), expected, "{kind}");
            }
        }
    }

    #[test]
    fn test_shard_map_parallel_log_sum_exp_executes_on_cpu() {
        // A logarithmic sum of exponentials over a manual mesh axis is composed from a mesh maximum and a mesh sum of
        // the shifted exponentials, so it stays finite for large inputs and matches the single-device reduction for
        // infinite inputs. Columns hold large equal inputs, all `-∞`, one `+∞`, mixed magnitudes, and one NaN.
        let shards = [
            [1000.0f32, f32::NEG_INFINITY, f32::INFINITY, -1000.0, 0.0],
            [1000.0, f32::NEG_INFINITY, 1.0, 0.0, f32::NAN],
            [1000.0, f32::NEG_INFINITY, 2.0, 1.0, 0.0],
            [1000.0, f32::NEG_INFINITY, 3.0, 2.0, 0.0],
        ]
        .map(|shard| values_to_bytes(&shard));
        let mixed = (0f64.exp() + 1f64.exp() + 2f64.exp()).ln() as f32;
        for output in
            execute_parallel_reduce_on_cpu(ReductionKind::LogSumExp, DataType::F32, BufferType::F32, 5, shards)
        {
            let values = values_from_bytes::<f32>(&output);
            assert!((values[0] - (1000.0 + 4f32.ln())).abs() <= 1e-4, "{values:?}");
            assert_eq!(values[1..3], [f32::NEG_INFINITY, f32::INFINITY]);
            assert!((values[3] - mixed).abs() <= 1e-6, "{values:?}");
            assert!(values[4].is_nan(), "{values:?}");
        }

        // Complex inputs are shifted by the maximum of their real components, so large real components stay finite
        // and the imaginary components rotate every term before the terms are summed.
        let shards = [[1000.0f32, 0.0], [1000.0, std::f32::consts::FRAC_PI_2], [1000.0, 0.0], [1000.0, 0.0]]
            .map(|shard| values_to_bytes(&shard));
        let expected = num_complex::Complex64::new(3.0, 1.0).ln() + 1000.0;
        for output in
            execute_parallel_reduce_on_cpu(ReductionKind::LogSumExp, DataType::C64, BufferType::C64, 1, shards)
        {
            let values = values_from_bytes::<f32>(&output);
            assert!((values[0] as f64 - expected.re).abs() <= 1e-4, "{values:?}");
            assert!((values[1] as f64 - expected.im).abs() <= 1e-6, "{values:?}");
        }
    }

    #[test]
    fn test_shard_map_parallel_log_sum_exp_gradient_executes_on_cpu() {
        use ryft_core::ParallelReduce;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = XlaSession::new(&client).domain();
        let devices = client.addressable_devices().unwrap();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let device_mesh =
            DeviceMesh::new(mesh.clone(), devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect())
                .unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        // The gradient of a logarithmic sum of exponentials is the softmax of its inputs. The composed mesh form
        // differentiates through its mesh sum alone, because its mesh maximum only shifts the exponents.
        let traced: TracedXlaProgram<ArrayType, Vec<ArrayType>> = trace(
            {
                let replicated = replicated.clone();
                let sharded = sharded.clone();
                move |x: XlaArrayTracer| {
                    let (value, gradient) = x
                        .clone()
                        .into_value()
                        .domain()
                        .differentiate_at(x.into_value())
                        .value_and_gradient(|x| {
                            let x = ValueProjection::<ArrayType>::into_projected(x)?;
                            Ok(shard_map(
                                |local_x: XlaArrayTracer| {
                                    local_x
                                        .reduce(&[0], ReductionKind::Sum)
                                        .unwrap()
                                        .parallel_reduce(ReductionKind::LogSumExp, "x")
                                        .unwrap()
                                },
                                x,
                                mesh.clone(),
                                sharded.clone(),
                                replicated.clone(),
                            )
                            .unwrap()
                            .into_value())
                        })
                        .unwrap();
                    vec![
                        ValueProjection::<ArrayType>::into_projected(value).unwrap(),
                        ValueProjection::<ArrayType>::into_projected(gradient).unwrap(),
                    ]
                }
            },
            ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap(),
        )
        .unwrap();
        let program = traced
            .to_mlir_module_with_signature_shardings(
                "main",
                Some(std::slice::from_ref(&sharded)),
                Some(&[replicated, sharded.clone()]),
            )
            .unwrap();
        let executable = client
            .compile(&Program::Mlir { bytecode: program.into_bytes() }, &test_spmd_compilation_options(4))
            .unwrap();
        let inputs = [1000.0f32, 1001.0, 1002.0, f32::NEG_INFINITY];
        let buffers = devices
            .iter()
            .zip(inputs)
            .map(|(device, input)| {
                client
                    .buffer(values_to_bytes(&[input]).as_slice(), BufferType::F32, [1], None, device.clone(), None)
                    .unwrap()
            })
            .collect();
        let input = XlaArray::from_addressable_buffers(
            &domain,
            static_sharded_array_type(DataType::F32, &[4], sharded),
            device_mesh,
            buffers,
        )
        .unwrap();
        let device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let arguments = XlaArray::into_execute_arguments(vec![input], &device_ids).unwrap();
        let outputs = executable
            .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();
        let normalizer = 0f64.exp() + 1f64.exp() + 2f64.exp();
        let softmax = [0f64.exp() / normalizer, 1f64.exp() / normalizer, 2f64.exp() / normalizer, 0.0];
        assert_eq!(outputs.len(), 4);
        for (output, expected_gradient) in outputs.into_iter().zip(softmax) {
            let values = output
                .outputs
                .iter()
                .map(|buffer| values_from_bytes::<f32>(&buffer.copy_to_host(None).unwrap().r#await().unwrap())[0])
                .collect::<Vec<_>>();
            assert!((values[0] as f64 - (1000.0 + normalizer.ln())).abs() <= 1e-4, "{values:?}");
            assert!((values[1] as f64 - expected_gradient).abs() <= 1e-6, "{values:?}");
        }
    }

    #[test]
    fn test_shard_map_invariant_all_gather_gradient_executes_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = XlaSession::new(&client).domain();
        let devices = client.addressable_devices().unwrap();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let device_mesh =
            DeviceMesh::new(mesh.clone(), devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect())
                .unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        // The gradient of `sum(g * g)`, where `g` is the invariant all-gather of `x`, is `2 * x`. Linearizing the
        // all-gather keeps its gathered extent as a first-class dimension residual, which crosses the boundary between
        // the primal and the tangent `shard_map`s.
        let traced: TracedXlaProgram<ArrayType, Vec<ArrayType>> = trace(
            {
                let replicated = replicated.clone();
                let sharded = sharded.clone();
                move |x: XlaArrayTracer| {
                    let (value, gradient) = x
                        .clone()
                        .into_value()
                        .domain()
                        .differentiate_at(x.into_value())
                        .value_and_gradient(|x| {
                            let x = ValueProjection::<ArrayType>::into_projected(x)?;
                            Ok(shard_map(
                                |local_x: XlaArrayTracer| {
                                    let gathered = local_x
                                        .parallel_all_gather_with_options(
                                            "x",
                                            0,
                                            CollectiveOptions::tiled(),
                                            ParallelAllGatherOutputVariance::Invariant,
                                        )
                                        .unwrap();
                                    gathered.mul(&gathered).unwrap().reduce(&[0], ReductionKind::Sum).unwrap()
                                },
                                x,
                                mesh.clone(),
                                sharded.clone(),
                                replicated.clone(),
                            )
                            .unwrap()
                            .into_value())
                        })
                        .unwrap();
                    vec![
                        ValueProjection::<ArrayType>::into_projected(value).unwrap(),
                        ValueProjection::<ArrayType>::into_projected(gradient).unwrap(),
                    ]
                }
            },
            ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap(),
        )
        .unwrap();
        let program = traced
            .to_mlir_module_with_signature_shardings(
                "main",
                Some(std::slice::from_ref(&sharded)),
                Some(&[replicated, sharded.clone()]),
            )
            .unwrap();
        // Recovering the dimension residual in the tangent body checks its bounds with a runtime assertion.
        crate::experimental::assertions::ensure_assertion_handler_registered(&client).unwrap();
        let executable = client
            .compile(&Program::Mlir { bytecode: program.into_bytes() }, &test_spmd_compilation_options(4))
            .unwrap();
        let inputs = [1.0f32, 2.0, 3.0, 4.0];
        let buffers = devices
            .iter()
            .zip(inputs)
            .map(|(device, input)| {
                client
                    .buffer(values_to_bytes(&[input]).as_slice(), BufferType::F32, [1], None, device.clone(), None)
                    .unwrap()
            })
            .collect();
        let input = XlaArray::from_addressable_buffers(
            &domain,
            static_sharded_array_type(DataType::F32, &[4], sharded),
            device_mesh,
            buffers,
        )
        .unwrap();
        let device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let arguments = XlaArray::into_execute_arguments(vec![input], &device_ids).unwrap();
        let outputs = executable
            .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();
        assert_eq!(outputs.len(), 4);
        for (output, input) in outputs.into_iter().zip(inputs) {
            let values = output
                .outputs
                .iter()
                .map(|buffer| values_from_bytes::<f32>(&buffer.copy_to_host(None).unwrap().r#await().unwrap())[0])
                .collect::<Vec<_>>();
            assert_eq!(values, vec![30.0, 2.0 * input]);
        }
    }

    #[test]
    fn test_shard_map_checked_variation_gradients_execute_on_cpu() {
        use ryft_core::{Fill, ParallelReduce, ReductionKind};

        for device_count in [1, 2] {
            let plugin = load_cpu_plugin().unwrap();
            let client = plugin
                .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(device_count), ..Default::default() }))
                .unwrap();
            let domain = XlaSession::new(&client).domain();
            let client_devices = client.addressable_devices().unwrap();
            assert_eq!(client_devices.len(), device_count);
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", device_count, MeshAxisType::Manual).unwrap()]).unwrap();
            let device_mesh = DeviceMesh::new(
                mesh.clone(),
                client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
            )
            .unwrap();
            let replicated = Sharding::replicated(mesh.clone(), 0);
            let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
            let traced: TracedXlaProgram<Vec<ArrayType>, Vec<ArrayType>> = trace(
                {
                    let replicated = replicated.clone();
                    let sharded = sharded.clone();
                    move |inputs: Vec<XlaArrayTracer>| {
                        let (value, gradient) = inputs[0]
                            .clone()
                            .into_value()
                            .domain()
                            .differentiate_at((inputs[0].clone().into_value(), inputs[1].clone().into_value()))
                            .value_and_gradient(|(shared, weights)| {
                                let shared = ValueProjection::<ArrayType>::into_projected(shared)?;
                                let weights = ValueProjection::<ArrayType>::into_projected(weights)?;
                                Ok(shard_map(
                                    |(shared, weights): (XlaArrayTracer, XlaArrayTracer)| {
                                        (shared.clone() * shared * weights)
                                            .reduce(&[0], ReductionKind::Sum)
                                            .unwrap()
                                            .parallel_reduce(ReductionKind::Sum, "x")
                                            .unwrap()
                                    },
                                    (shared, weights),
                                    mesh.clone(),
                                    (replicated.clone(), sharded.clone()),
                                    replicated.clone(),
                                )
                                .unwrap()
                                .into_value())
                            })
                            .unwrap();
                        let varying_gradient = inputs[1]
                            .clone()
                            .into_value()
                            .domain()
                            .differentiate_at(inputs[1].clone().into_value())
                            .gradient(|varying| {
                                let varying = ValueProjection::<ArrayType>::into_projected(varying)?;
                                Ok(shard_map(
                                    |varying: XlaArrayTracer| {
                                        (varying.clone() * varying)
                                            .reduce(&[0], ReductionKind::Sum)
                                            .unwrap()
                                            .parallel_reduce(ReductionKind::Sum, "x")
                                            .unwrap()
                                    },
                                    varying,
                                    mesh.clone(),
                                    sharded.clone(),
                                    replicated.clone(),
                                )
                                .unwrap()
                                .into_value())
                            })
                            .unwrap();
                        let second_derivative = inputs[0]
                            .clone()
                            .into_value()
                            .domain()
                            .differentiate_at(inputs[0].clone().into_value())
                            .with_captures(inputs[1].clone().into_value())
                            .gradient(|shared, weights| {
                                Ok(shared
                                    .domain()
                                    .differentiate_at(shared.clone())
                                    .with_captures(weights)
                                    .gradient(|shared, weights| {
                                        let shared = ValueProjection::<ArrayType>::into_projected(shared)?;
                                        let weights = ValueProjection::<ArrayType>::into_projected(weights)?;
                                        Ok(shard_map(
                                            |(shared, weights): (XlaArrayTracer, XlaArrayTracer)| {
                                                (shared.clone() * shared * weights)
                                                    .reduce(&[0], ReductionKind::Sum)
                                                    .unwrap()
                                                    .parallel_reduce(ReductionKind::Sum, "x")
                                                    .unwrap()
                                            },
                                            (shared, weights),
                                            mesh.clone(),
                                            (replicated.clone(), sharded.clone()),
                                            replicated.clone(),
                                        )
                                        .unwrap()
                                        .into_value())
                                    })
                                    .unwrap())
                            })
                            .unwrap();
                        let (_, pullback) = inputs[0]
                            .clone()
                            .into_value()
                            .domain()
                            .differentiate_at((inputs[0].clone().into_value(), inputs[1].clone().into_value()))
                            .vjp(|(shared, weights)| {
                                let shared = ValueProjection::<ArrayType>::into_projected(shared)?;
                                let weights = ValueProjection::<ArrayType>::into_projected(weights)?;
                                Ok(shard_map(
                                    |(shared, weights): (XlaArrayTracer, XlaArrayTracer)| {
                                        shared.clone() * shared * weights
                                    },
                                    (shared, weights),
                                    mesh.clone(),
                                    (replicated.clone(), sharded.clone()),
                                    sharded.clone(),
                                )
                                .unwrap()
                                .into_value())
                            })
                            .unwrap();
                        let seeded_gradients = pullback.apply(inputs[1].clone().into_value()).unwrap();
                        let (_, directional_derivative) = inputs[0]
                            .clone()
                            .into_value()
                            .domain()
                            .differentiate_at((inputs[0].clone().into_value(), inputs[1].clone().into_value()))
                            .jvp(
                                (inputs[0].clone().into_value(), inputs[1].clone().into_value()),
                                |(shared, weights)| {
                                    let shared = ValueProjection::<ArrayType>::into_projected(shared)?;
                                    let weights = ValueProjection::<ArrayType>::into_projected(weights)?;
                                    Ok(shard_map(
                                        |(shared, weights): (XlaArrayTracer, XlaArrayTracer)| {
                                            shared.clone() * shared * weights
                                        },
                                        (shared, weights),
                                        mesh.clone(),
                                        (replicated.clone(), sharded.clone()),
                                        sharded.clone(),
                                    )
                                    .unwrap()
                                    .into_value())
                                },
                            )
                            .unwrap();
                        let inner_gradient = shard_map(
                            |(shared, weights): (XlaArrayTracer, XlaArrayTracer)| {
                                let derivative = shared
                                    .clone()
                                    .into_value()
                                    .domain()
                                    .differentiate_at(shared.into_value())
                                    .with_captures(weights.into_value())
                                    .gradient(|shared, weights| {
                                        let shared = ValueProjection::<ArrayType>::into_projected(shared)?;
                                        let weights = ValueProjection::<ArrayType>::into_projected(weights)?;
                                        Ok((shared.clone() * shared * weights)
                                            .reduce(&[0], ReductionKind::Sum)
                                            .unwrap()
                                            .into_value())
                                    })
                                    .unwrap();
                                ValueProjection::<ArrayType>::into_projected(derivative).unwrap()
                            },
                            (inputs[0].clone(), inputs[1].clone()),
                            mesh.clone(),
                            (replicated.clone(), sharded.clone()),
                            replicated.clone(),
                        )
                        .unwrap();
                        let constant_sum = shard_map(
                            |input: XlaArrayTracer| {
                                let constant: XlaArrayTracer = input.domain().fill(&input.r#type(), 3.0_f32).unwrap();
                                constant.parallel_reduce(ReductionKind::Sum, "x").unwrap()
                            },
                            inputs[0].clone(),
                            mesh.clone(),
                            replicated.clone(),
                            replicated.clone(),
                        )
                        .unwrap();
                        vec![
                            ValueProjection::<ArrayType>::into_projected(value).unwrap(),
                            ValueProjection::<ArrayType>::into_projected(gradient.0).unwrap(),
                            ValueProjection::<ArrayType>::into_projected(gradient.1).unwrap(),
                            constant_sum,
                            ValueProjection::<ArrayType>::into_projected(varying_gradient).unwrap(),
                            ValueProjection::<ArrayType>::into_projected(second_derivative).unwrap(),
                            ValueProjection::<ArrayType>::into_projected(seeded_gradients.0).unwrap(),
                            ValueProjection::<ArrayType>::into_projected(seeded_gradients.1).unwrap(),
                            ValueProjection::<ArrayType>::into_projected(directional_derivative).unwrap(),
                            inner_gradient,
                        ]
                    }
                },
                vec![
                    ArrayType::scalar(DataType::F32),
                    ArrayType::new_static(DataType::F32, [device_count]).with_sharding(sharded.clone()).unwrap(),
                ],
            )
            .unwrap();
            let program = traced
                .to_mlir_module_with_signature_shardings(
                    "main",
                    Some(&[replicated.clone(), sharded.clone()]),
                    Some(&[
                        replicated.clone(),
                        replicated.clone(),
                        sharded.clone(),
                        replicated.clone(),
                        sharded.clone(),
                        replicated.clone(),
                        replicated.clone(),
                        sharded.clone(),
                        sharded.clone(),
                        replicated.clone(),
                    ]),
                )
                .unwrap();
            let executable = client
                .compile(
                    &Program::Mlir { bytecode: program.into_bytes() },
                    &test_spmd_compilation_options(device_count),
                )
                .unwrap();
            let shared_buffers = client_devices
                .iter()
                .map(|device| {
                    client.buffer(&3.0_f32.to_ne_bytes(), BufferType::F32, [], None, device.clone(), None).unwrap()
                })
                .collect();
            let weight_buffers = client_devices
                .iter()
                .zip([2.0_f32, 5.0])
                .map(|(device, weight)| {
                    client.buffer(&weight.to_ne_bytes(), BufferType::F32, [1], None, device.clone(), None).unwrap()
                })
                .collect();
            let inputs = vec![
                XlaArray::from_addressable_buffers(
                    &domain,
                    static_sharded_array_type(DataType::F32, &[], replicated),
                    device_mesh.clone(),
                    shared_buffers,
                )
                .unwrap(),
                XlaArray::from_addressable_buffers(
                    &domain,
                    static_sharded_array_type(DataType::F32, &[device_count], sharded),
                    device_mesh,
                    weight_buffers,
                )
                .unwrap(),
            ];
            let device_ids = executable
                .addressable_devices()
                .unwrap()
                .iter()
                .map(|device| device.id().unwrap())
                .collect::<Vec<_>>();
            let arguments = XlaArray::into_execute_arguments(inputs, &device_ids).unwrap();
            let outputs = executable
                .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
                .unwrap()
                .block_until_ready()
                .unwrap();
            assert_eq!(outputs.len(), device_count);
            let expected_varying_gradients = client_devices
                .iter()
                .zip([4.0, 10.0])
                .map(|(device, gradient)| (device.id().unwrap(), gradient))
                .collect::<HashMap<_, _>>();
            let total_weight = if device_count == 1 { 2.0 } else { 7.0 };
            let squared_weight_sum = if device_count == 1 { 4.0 } else { 29.0 };
            let mut output_pairing = 0.0;
            let mut input_pairing = 0.0;
            for (device_index, (output, device_id)) in outputs.into_iter().zip(device_ids).enumerate() {
                let expected_varying_gradient = expected_varying_gradients[&device_id];
                assert_eq!(output.outputs.len(), 10);
                let values = output
                    .outputs
                    .iter()
                    .map(|buffer| values_from_bytes::<f32>(&buffer.copy_to_host(None).unwrap().r#await().unwrap()))
                    .collect::<Vec<_>>();
                let weight = expected_varying_gradient / 2.0;
                output_pairing += weight * values[8][0];
                input_pairing += weight * values[7][0];
                if device_index == 0 {
                    input_pairing += 3.0 * values[6][0];
                }
                // The shared derivative sums distinct contributions; each varying weight receives only its local one.
                assert_eq!(
                    values,
                    vec![
                        vec![9.0 * total_weight],
                        vec![6.0 * total_weight],
                        vec![9.0],
                        vec![3.0 * device_count as f32],
                        vec![expected_varying_gradient],
                        vec![2.0 * total_weight],
                        vec![6.0 * squared_weight_sum],
                        vec![9.0 * weight],
                        vec![27.0 * weight],
                        vec![6.0 * total_weight],
                    ],
                );
            }
            // Count the invariant shared input once; pair varying inputs/outputs over their logical shards.
            assert_eq!(input_pairing, output_pairing);
            assert_eq!(output_pairing, 27.0 * squared_weight_sum);
        }
    }

    #[test]
    fn test_shard_map_gather_varying_indices_gradient_executes_on_cpu() {
        use ryft_core::{Gather, GatherMode, ParallelReduce, ReductionKind};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let domain = XlaSession::new(&client).domain();
        let devices = client.addressable_devices().unwrap();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let device_mesh =
            DeviceMesh::new(mesh.clone(), devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect())
                .unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_types = vec![
            ArrayType::new_static(DataType::F32, [3]),
            ArrayType::new_static(DataType::I32, [4]).with_sharding(sharded.clone()).unwrap(),
            ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap(),
        ];
        let traced: TracedXlaProgram<Vec<ArrayType>, Vec<ArrayType>> = trace(
            {
                let mesh = mesh.clone();
                let replicated = replicated.clone();
                let sharded = sharded.clone();
                move |inputs: Vec<XlaArrayTracer>| {
                    let (output, pullback) = inputs[0]
                        .clone()
                        .into_value()
                        .domain()
                        .differentiate_at(inputs[0].clone().into_value())
                        .with_captures(inputs[1].clone().into_value())
                        .vjp(|data, indices| {
                            let data = ValueProjection::<ArrayType>::into_projected(data)?;
                            let indices = ValueProjection::<ArrayType>::into_projected(indices)?;
                            Ok(shard_map(
                                |(data, indices): (XlaArrayTracer, XlaArrayTracer)| {
                                    data.gather_axis(&indices, 0, GatherMode::Clip).unwrap()
                                },
                                (data, indices),
                                mesh.clone(),
                                (replicated.clone(), sharded.clone()),
                                sharded.clone(),
                            )
                            .unwrap()
                            .into_value())
                        })
                        .unwrap();
                    let gradient = pullback.apply(inputs[2].clone().into_value()).unwrap();
                    let maxima = shard_map(
                        |input: XlaArrayTracer| {
                            let maximum = input.parallel_reduce(ReductionKind::Max, "x").unwrap();
                            let varying = maximum.parallel_vary("x").unwrap();
                            (maximum, varying)
                        },
                        inputs[2].clone(),
                        mesh.clone(),
                        sharded.clone(),
                        (replicated.clone(), sharded.clone()),
                    )
                    .unwrap();
                    vec![
                        ValueProjection::<ArrayType>::into_projected(output).unwrap(),
                        ValueProjection::<ArrayType>::into_projected(gradient).unwrap(),
                        maxima.0,
                        maxima.1,
                    ]
                }
            },
            input_types,
        )
        .unwrap();
        let program = traced
            .to_mlir_module_with_signature_shardings(
                "main",
                Some(&[replicated.clone(), sharded.clone(), sharded.clone()]),
                Some(&[sharded.clone(), replicated.clone(), replicated.clone(), sharded.clone()]),
            )
            .unwrap();
        let executable = client
            .compile(&Program::Mlir { bytecode: program.into_bytes() }, &test_spmd_compilation_options(2))
            .unwrap();
        let data_buffers = devices
            .iter()
            .map(|device| {
                client
                    .buffer(
                        values_to_bytes(&[11.0_f32, 13.0, 17.0]).as_slice(),
                        BufferType::F32,
                        [3],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect();
        let index_buffers = devices
            .iter()
            .zip([[0_i32, 1], [0, 2]])
            .map(|(device, indices)| {
                client
                    .buffer(values_to_bytes(&indices).as_slice(), BufferType::I32, [2], None, device.clone(), None)
                    .unwrap()
            })
            .collect();
        let seed_buffers = devices
            .iter()
            .zip([[2.0_f32, 3.0], [5.0, 7.0]])
            .map(|(device, seeds)| {
                client
                    .buffer(values_to_bytes(&seeds).as_slice(), BufferType::F32, [2], None, device.clone(), None)
                    .unwrap()
            })
            .collect();
        let inputs = vec![
            XlaArray::from_addressable_buffers(
                &domain,
                static_sharded_array_type(DataType::F32, &[3], replicated),
                device_mesh.clone(),
                data_buffers,
            )
            .unwrap(),
            XlaArray::from_addressable_buffers(
                &domain,
                static_sharded_array_type(DataType::I32, &[4], sharded.clone()),
                device_mesh.clone(),
                index_buffers,
            )
            .unwrap(),
            XlaArray::from_addressable_buffers(
                &domain,
                static_sharded_array_type(DataType::F32, &[4], sharded),
                device_mesh,
                seed_buffers,
            )
            .unwrap(),
        ];
        let expected = devices
            .iter()
            .zip([[11.0_f32, 13.0], [11.0, 17.0]])
            .map(|(device, values)| (device.id().unwrap(), values))
            .collect::<HashMap<_, _>>();
        let device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let arguments = XlaArray::into_execute_arguments(inputs, &device_ids).unwrap();
        let outputs = executable
            .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();
        assert_eq!(outputs.len(), 2);
        for (output, device_id) in outputs.into_iter().zip(device_ids) {
            let values = output
                .outputs
                .iter()
                .map(|buffer| values_from_bytes::<f32>(&buffer.copy_to_host(None).unwrap().r#await().unwrap()))
                .collect::<Vec<_>>();
            // Index zero occurs on both devices; its unequal seed contributions must be added exactly once.
            assert_eq!(
                values,
                vec![expected[&device_id].to_vec(), vec![7.0, 3.0, 7.0], vec![5.0, 7.0], vec![5.0, 7.0]],
            );
        }
    }

    #[test]
    fn test_shard_map_sort_varying_keys_jvp_executes_on_cpu() {
        use ryft_core::{Sort, SortDirection};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let domain = XlaSession::new(&client).domain();
        let devices = client.addressable_devices().unwrap();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let device_mesh =
            DeviceMesh::new(mesh.clone(), devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect())
                .unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_types = vec![
            ArrayType::new_static(DataType::F32, [2]),
            ArrayType::new_static(DataType::I32, [4]).with_sharding(sharded.clone()).unwrap(),
            ArrayType::new_static(DataType::F32, [2]),
        ];
        let traced: TracedXlaProgram<Vec<ArrayType>, Vec<ArrayType>> = trace(
            {
                let mesh = mesh.clone();
                let replicated = replicated.clone();
                let sharded = sharded.clone();
                move |inputs: Vec<XlaArrayTracer>| {
                    let (output, tangent) = inputs[0]
                        .clone()
                        .into_value()
                        .domain()
                        .differentiate_at(inputs[0].clone().into_value())
                        .with_captures(inputs[1].clone().into_value())
                        .jvp(inputs[2].clone().into_value(), |data, indices| {
                            let data = ValueProjection::<ArrayType>::into_projected(data)?;
                            let indices = ValueProjection::<ArrayType>::into_projected(indices)?;
                            Ok(shard_map(
                                |(data, indices): (XlaArrayTracer, XlaArrayTracer)| {
                                    Sort::sort(&[indices, data], 0, SortDirection::Ascending).unwrap().remove(1)
                                },
                                (data, indices),
                                mesh.clone(),
                                (replicated.clone(), sharded.clone()),
                                sharded.clone(),
                            )
                            .unwrap()
                            .into_value())
                        })
                        .unwrap();
                    vec![
                        ValueProjection::<ArrayType>::into_projected(output).unwrap(),
                        ValueProjection::<ArrayType>::into_projected(tangent).unwrap(),
                    ]
                }
            },
            input_types,
        )
        .unwrap();
        let program = traced
            .to_mlir_module_with_signature_shardings(
                "main",
                Some(&[replicated.clone(), sharded.clone(), replicated.clone()]),
                Some(&[sharded.clone(), sharded.clone()]),
            )
            .unwrap();
        let executable = client
            .compile(&Program::Mlir { bytecode: program.into_bytes() }, &test_spmd_compilation_options(2))
            .unwrap();
        let data_buffers = devices
            .iter()
            .map(|device| {
                client
                    .buffer(
                        values_to_bytes(&[11.0_f32, 13.0]).as_slice(),
                        BufferType::F32,
                        [2],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect();
        let index_buffers = devices
            .iter()
            .zip([[2_i32, 1], [1, 2]])
            .map(|(device, indices)| {
                client
                    .buffer(values_to_bytes(&indices).as_slice(), BufferType::I32, [2], None, device.clone(), None)
                    .unwrap()
            })
            .collect();
        let seed_buffers = devices
            .iter()
            .zip([[2.0_f32, 3.0], [2.0, 3.0]])
            .map(|(device, seeds)| {
                client
                    .buffer(values_to_bytes(&seeds).as_slice(), BufferType::F32, [2], None, device.clone(), None)
                    .unwrap()
            })
            .collect();
        let inputs = vec![
            XlaArray::from_addressable_buffers(
                &domain,
                static_sharded_array_type(DataType::F32, &[2], replicated.clone()),
                device_mesh.clone(),
                data_buffers,
            )
            .unwrap(),
            XlaArray::from_addressable_buffers(
                &domain,
                static_sharded_array_type(DataType::I32, &[4], sharded.clone()),
                device_mesh.clone(),
                index_buffers,
            )
            .unwrap(),
            XlaArray::from_addressable_buffers(
                &domain,
                static_sharded_array_type(DataType::F32, &[2], replicated),
                device_mesh,
                seed_buffers,
            )
            .unwrap(),
        ];
        let expected = devices
            .iter()
            .zip([[13.0_f32, 11.0], [11.0, 13.0]])
            .map(|(device, values)| (device.id().unwrap(), values))
            .collect::<HashMap<_, _>>();
        let expected_tangents = devices
            .iter()
            .zip([[3.0_f32, 2.0], [2.0, 3.0]])
            .map(|(device, values)| (device.id().unwrap(), values))
            .collect::<HashMap<_, _>>();
        let device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let arguments = XlaArray::into_execute_arguments(inputs, &device_ids).unwrap();
        let outputs = executable
            .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();
        assert_eq!(outputs.len(), 2);
        for (output, device_id) in outputs.into_iter().zip(device_ids) {
            let values = output
                .outputs
                .iter()
                .map(|buffer| values_from_bytes::<f32>(&buffer.copy_to_host(None).unwrap().r#await().unwrap()))
                .collect::<Vec<_>>();
            // The invariant passenger and its tangent follow each device's distinct key permutation.
            assert_eq!(values, vec![expected[&device_id].to_vec(), expected_tangents[&device_id].to_vec()]);
        }
    }

    #[test]
    fn test_shard_map_attention_varying_keys_gradient_executes_on_cpu() {
        use ryft_core::operations::attention::{
            AttentionConfiguration, AttentionImplementation, AttentionInputs, differentiable_dot_product_attention,
        };
        use ryft_core::{ArrayOperation, DomainTracer, EagerContext};

        use crate::experimental::ops::XlaArrayConstant;

        type AttentionDomain = EagerContext<XlaArrayConstant, ArrayOperation<XlaArrayConstant>>;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let domain = XlaSession::new(&client).domain();
        let devices = client.addressable_devices().unwrap();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let device_mesh =
            DeviceMesh::new(mesh.clone(), devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect())
                .unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 4);
        let sharded = Sharding::new(
            mesh.clone(),
            vec![
                ShardingDimension::sharded(["x"]),
                ShardingDimension::Replicated,
                ShardingDimension::Replicated,
                ShardingDimension::Replicated,
            ],
        )
        .unwrap();
        let query_type = ArrayType::new_static(DataType::F32, [1, 1, 1, 1]).with_sharding(replicated.clone()).unwrap();
        let key_type = ArrayType::new_static(DataType::F32, [2, 2, 1, 1]).with_sharding(sharded.clone()).unwrap();
        let value_type = ArrayType::new_static(DataType::F32, [1, 2, 1, 1]).with_sharding(replicated.clone()).unwrap();
        let seed_type = ArrayType::new_static(DataType::F32, [2, 1, 1, 1]).with_sharding(sharded.clone()).unwrap();
        let input_types = vec![query_type.clone(), key_type.clone(), value_type.clone(), seed_type.clone()];
        let in_shardings = vec![replicated.clone(), sharded.clone(), replicated.clone(), sharded.clone()];
        let out_shardings = vec![sharded.clone(), replicated.clone()];
        let boundary = ShardMap::new(mesh.clone(), in_shardings, out_shardings, vec![]).unwrap();
        let local_input_types = input_types
            .iter()
            .enumerate()
            .map(|(index, input)| boundary.local_input_type(index, input).unwrap())
            .collect::<Vec<_>>();
        // Trace the custom function in its homogeneous universe, then lift the complete program. Whole-program
        // promotion preserves the call's primal region and converts its retained rules into the lifted family.
        let (_, body) = DomainTracingContext::<AttentionDomain>::trace_with_named_axes(
            |inputs: Vec<DomainTracer<AttentionDomain>>| {
                let function = differentiable_dot_product_attention::<AttentionDomain>(
                    AttentionConfiguration::new().with_implementation(AttentionImplementation::Portable),
                );
                let (output, pullback) = inputs[0]
                    .domain()
                    .differentiate_at(AttentionInputs::new(inputs[0].clone(), inputs[1].clone(), inputs[2].clone()))
                    .vjp(|inputs| function.call(inputs))
                    .unwrap();
                let gradient = pullback.apply(inputs[3].clone()).unwrap();
                // The query component of the full VJP is the partial derivative with respect to the query;
                // making key/value cotangents available does not change this component.
                Ok(vec![output, gradient.query])
            },
            local_input_types,
            vec![("x".to_string(), NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2 })],
        )
        .unwrap();
        let body = body.into_flat_program().into_unprojected::<XlaConstant, XlaOperation>().unwrap();
        let operation = ShardMapOperation::from_program(
            &body,
            input_types.iter().cloned().map(ArrayIrType::Array).collect(),
            boundary,
        )
        .unwrap();
        let traced: TracedXlaProgram<Vec<ArrayType>, Vec<ArrayType>> = trace(
            move |inputs: Vec<XlaArrayTracer>| {
                let inputs = inputs.into_iter().map(ProjectedValue::into_value).collect::<Vec<_>>();
                inputs[0]
                    .domain()
                    .bind(XlaOperation::ShardMap(Box::new(operation.clone())), vec![body.clone()], &inputs)
                    .unwrap()
                    .into_iter()
                    .map(|output| ValueProjection::<ArrayType>::into_projected(output).unwrap())
                    .collect()
            },
            input_types,
        )
        .unwrap();
        let program = traced
            .to_mlir_module_with_signature_shardings(
                "main",
                Some(&[replicated.clone(), sharded.clone(), replicated.clone(), sharded.clone()]),
                Some(&[sharded, replicated]),
            )
            .unwrap();
        let executable = client
            .compile(&Program::Mlir { bytecode: program.into_bytes() }, &test_spmd_compilation_options(2))
            .unwrap();
        let inputs = vec![
            XlaArray::from_host_buffer(&domain, query_type, device_mesh.clone(), &values_to_bytes(&[0.0_f32])).unwrap(),
            XlaArray::from_host_buffer(
                &domain,
                key_type,
                device_mesh.clone(),
                &values_to_bytes(&[0.0_f32, 1.0, 0.0, 3.0]),
            )
            .unwrap(),
            XlaArray::from_host_buffer(&domain, value_type, device_mesh.clone(), &values_to_bytes(&[2.0_f32, 6.0]))
                .unwrap(),
            XlaArray::from_host_buffer(&domain, seed_type, device_mesh, &values_to_bytes(&[2.0_f32, 5.0])).unwrap(),
        ];
        let device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let arguments = XlaArray::into_execute_arguments(inputs, &device_ids).unwrap();
        let outputs = executable
            .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();
        assert_eq!(outputs.len(), 2);
        for output in outputs {
            let values = output
                .outputs
                .iter()
                .map(|buffer| values_from_bytes::<f32>(&buffer.copy_to_host(None).unwrap().r#await().unwrap()))
                .collect::<Vec<_>>();
            // At query zero the attention weights are equal. The query derivatives are 1 and 3, so the distinct
            // output seeds give one shared cotangent of 2*1 + 5*3 = 17, rather than either local contribution.
            assert_eq!(values, vec![vec![4.0], vec![17.0]]);
        }
    }

    #[test]
    fn test_shard_map_nested_variation_gradients_execute_on_cpu() {
        use ryft_core::{ParallelReduce, ReductionKind};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = XlaSession::new(&client).domain();
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 4);
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let device_mesh = DeviceMesh::new(
            mesh.clone(),
            client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let outer = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let inner = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
        let global = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x", "y"])]).unwrap();
        let traced: TracedXlaProgram<Vec<ArrayType>, Vec<ArrayType>> = trace(
            {
                let replicated = replicated.clone();
                move |inputs: Vec<XlaArrayTracer>| {
                    let (value, gradient) = inputs[0]
                        .clone()
                        .into_value()
                        .domain()
                        .differentiate_at((inputs[0].clone().into_value(), inputs[1].clone().into_value()))
                        .value_and_gradient(|(shared, weights)| {
                            let shared = ValueProjection::<ArrayType>::into_projected(shared)?;
                            let weights = ValueProjection::<ArrayType>::into_projected(weights)?;
                            Ok(shard_map_with_options(
                                |inputs: (XlaArrayTracer, XlaArrayTracer)| {
                                    shard_map_with_options(
                                        |(shared, weights): (XlaArrayTracer, XlaArrayTracer)| {
                                            (shared.clone() * shared * weights.clone() * weights)
                                                .reduce(&[0], ReductionKind::Sum)
                                                .unwrap()
                                                .parallel_reduce(ReductionKind::Sum, "y")
                                                .unwrap()
                                        },
                                        inputs,
                                        mesh.clone(),
                                        (replicated.clone(), inner.clone()),
                                        replicated.clone(),
                                        vec!["y".to_string()],
                                    )
                                    .unwrap()
                                    .parallel_reduce(ReductionKind::Sum, "x")
                                    .unwrap()
                                },
                                (shared, weights),
                                mesh.clone(),
                                (replicated.clone(), outer.clone()),
                                replicated.clone(),
                                vec!["x".to_string()],
                            )
                            .unwrap()
                            .into_value())
                        })
                        .unwrap();
                    vec![
                        ValueProjection::<ArrayType>::into_projected(value).unwrap(),
                        ValueProjection::<ArrayType>::into_projected(gradient.0).unwrap(),
                        ValueProjection::<ArrayType>::into_projected(gradient.1).unwrap(),
                    ]
                }
            },
            vec![ArrayType::scalar(DataType::F32), ArrayType::new_static(DataType::F32, [4])],
        )
        .unwrap();
        let program = traced
            .to_mlir_module_with_signature_shardings(
                "main",
                Some(&[replicated.clone(), global.clone()]),
                Some(&[replicated.clone(), replicated.clone(), global.clone()]),
            )
            .unwrap();
        let executable = client
            .compile(&Program::Mlir { bytecode: program.into_bytes() }, &test_spmd_compilation_options(4))
            .unwrap();
        let shared_buffers = client_devices
            .iter()
            .map(|device| {
                client.buffer(&3.0_f32.to_ne_bytes(), BufferType::F32, [], None, device.clone(), None).unwrap()
            })
            .collect();
        let weight_buffers = client_devices
            .iter()
            .zip([1.0_f32, 2.0, 3.0, 4.0])
            .map(|(device, weight)| {
                client.buffer(&weight.to_ne_bytes(), BufferType::F32, [1], None, device.clone(), None).unwrap()
            })
            .collect();
        let inputs = vec![
            XlaArray::from_addressable_buffers(
                &domain,
                static_sharded_array_type(DataType::F32, &[], replicated),
                device_mesh.clone(),
                shared_buffers,
            )
            .unwrap(),
            XlaArray::from_addressable_buffers(
                &domain,
                static_sharded_array_type(DataType::F32, &[4], global),
                device_mesh,
                weight_buffers,
            )
            .unwrap(),
        ];
        let device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let arguments = XlaArray::into_execute_arguments(inputs, &device_ids).unwrap();
        let outputs = executable
            .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();
        assert_eq!(outputs.len(), 4);
        let expected_weight_gradients = client_devices
            .iter()
            .zip([18.0, 36.0, 54.0, 72.0])
            .map(|(device, gradient)| (device.id().unwrap(), gradient))
            .collect::<HashMap<_, _>>();
        for (output, device_id) in outputs.into_iter().zip(device_ids) {
            let values = output
                .outputs
                .iter()
                .map(|buffer| values_from_bytes::<f32>(&buffer.copy_to_host(None).unwrap().r#await().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(values, vec![vec![270.0], vec![180.0], vec![expected_weight_gradients[&device_id]]]);
        }
    }

    #[test]
    fn test_shard_map_mixed_variation_gradients_execute_on_cpu() {
        use ryft_core::{ParallelReduce, ReductionKind};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = XlaSession::new(&client).domain();
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 4);
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let device_mesh = DeviceMesh::new(
            mesh.clone(),
            client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let shared_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["z"])]).unwrap();
        let global =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["z"])])
                .unwrap();
        let traced: TracedXlaProgram<Vec<ArrayType>, Vec<ArrayType>> = trace(
            {
                let replicated = replicated.clone();
                let shared_sharding = shared_sharding.clone();
                let global = global.clone();
                move |inputs: Vec<XlaArrayTracer>| {
                    let (value, gradient) = inputs[0]
                        .clone()
                        .into_value()
                        .domain()
                        .differentiate_at((inputs[0].clone().into_value(), inputs[1].clone().into_value()))
                        .value_and_gradient(|(shared, weights)| {
                            let shared = ValueProjection::<ArrayType>::into_projected(shared)?;
                            let weights = ValueProjection::<ArrayType>::into_projected(weights)?;
                            Ok(shard_map_with_options(
                                |(shared, weights): (XlaArrayTracer, XlaArrayTracer)| {
                                    (shared.clone() * shared * weights)
                                        .reduce(&[0, 1], ReductionKind::Sum)
                                        .unwrap()
                                        .parallel_reduce(ReductionKind::Sum, "x")
                                        .unwrap()
                                },
                                (shared, weights),
                                mesh.clone(),
                                (shared_sharding.clone(), global.clone()),
                                replicated.clone(),
                                vec!["x".to_string()],
                            )
                            .unwrap()
                            .into_value())
                        })
                        .unwrap();
                    vec![
                        ValueProjection::<ArrayType>::into_projected(value).unwrap(),
                        ValueProjection::<ArrayType>::into_projected(gradient.0).unwrap(),
                        ValueProjection::<ArrayType>::into_projected(gradient.1).unwrap(),
                    ]
                }
            },
            vec![
                ArrayType::new_static(DataType::F32, [2]).with_sharding(shared_sharding.clone()).unwrap(),
                ArrayType::new_static(DataType::F32, [2, 2]).with_sharding(global.clone()).unwrap(),
            ],
        )
        .unwrap();
        let program = traced
            .to_mlir_module_with_signature_shardings(
                "main",
                Some(&[shared_sharding.clone(), global.clone()]),
                Some(&[replicated.clone(), shared_sharding.clone(), global.clone()]),
            )
            .unwrap();
        let executable = client
            .compile(&Program::Mlir { bytecode: program.into_bytes() }, &test_spmd_compilation_options(4))
            .unwrap();
        let shared_buffers = client_devices
            .iter()
            .zip([3.0_f32, 4.0, 3.0, 4.0])
            .map(|(device, shared)| {
                client.buffer(&shared.to_ne_bytes(), BufferType::F32, [1], None, device.clone(), None).unwrap()
            })
            .collect();
        let weight_buffers = client_devices
            .iter()
            .zip([2.0_f32, 3.0, 5.0, 7.0])
            .map(|(device, weight)| {
                client.buffer(&weight.to_ne_bytes(), BufferType::F32, [1, 1], None, device.clone(), None).unwrap()
            })
            .collect();
        let inputs = vec![
            XlaArray::from_addressable_buffers(
                &domain,
                static_sharded_array_type(DataType::F32, &[2], shared_sharding),
                device_mesh.clone(),
                shared_buffers,
            )
            .unwrap(),
            XlaArray::from_addressable_buffers(
                &domain,
                static_sharded_array_type(DataType::F32, &[2, 2], global),
                device_mesh,
                weight_buffers,
            )
            .unwrap(),
        ];
        let device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let arguments = XlaArray::into_execute_arguments(inputs, &device_ids).unwrap();
        let outputs = executable
            .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();
        assert_eq!(outputs.len(), 4);
        let expected_gradients = client_devices
            .iter()
            .zip([(42.0, 9.0), (80.0, 16.0), (42.0, 9.0), (80.0, 16.0)])
            .map(|(device, gradient)| (device.id().unwrap(), gradient))
            .collect::<HashMap<_, _>>();
        for (output, device_id) in outputs.into_iter().zip(device_ids) {
            let values = output
                .outputs
                .iter()
                .map(|buffer| values_from_bytes::<f32>(&buffer.copy_to_host(None).unwrap().r#await().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(
                values,
                vec![vec![223.0], vec![expected_gradients[&device_id].0], vec![expected_gradients[&device_id].1]],
            );
        }
    }

    #[test]
    fn test_shard_map_varying_while_predicate_gradient_executes_on_cpu() {
        use ryft_core::{
            ArrayOperation, CompareOperation, ComparisonDirection, ConstantOperation, MulOperation, ReduceOperation,
            WhileOperation,
        };

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = XlaSession::new(&client).domain();
        let devices = client.addressable_devices().unwrap();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let device_mesh =
            DeviceMesh::new(mesh.clone(), devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect())
                .unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        // Each device squares its shard while the shard is below `10`, so the loop predicate varies over `x` and the
        // devices run different numbers of iterations: inputs `1.5`, `2`, `3`, and `20` take three, two, two, and zero
        // iterations. Reverse mode stores per-device residual stacks and validity masks of the same variation.
        let traced: TracedXlaProgram<ArrayType, Vec<ArrayType>> = trace(
            {
                let sharded = sharded.clone();
                move |x: XlaArrayTracer| {
                    let (value, gradient) = x
                        .clone()
                        .into_value()
                        .domain()
                        .differentiate_at(x.into_value())
                        .value_and_gradient(|x| {
                            let x = ValueProjection::<ArrayType>::into_projected(x)?;
                            Ok(shard_map(
                                |local_x: XlaArrayTracer| {
                                    let local_type = local_x.r#type().into_owned();
                                    // The limit is a non-differentiable constant, so it is created with the
                                    // variation of the state that it is compared against.
                                    let limit_sharding =
                                        Sharding::replicated(local_type.sharding().unwrap().mesh().clone(), 0)
                                            .with_varying_manual_axes(["x"])
                                            .unwrap();
                                    let limit_type =
                                        ArrayType::scalar(DataType::F32).with_sharding(limit_sharding).unwrap();
                                    let condition = {
                                        let mut builder = XlaProgramBuilder::new();
                                        let state = builder.add_input(ArrayIrType::Array(local_type.clone()));
                                        let state = builder
                                            .add_instruction(
                                                ReduceOperation::new(vec![0], ReductionKind::Sum),
                                                Vec::new(),
                                                vec![state],
                                                None,
                                            )
                                            .unwrap()[0];
                                        let limit = Array::from_elements(limit_type, &[10f32]).unwrap();
                                        let limit = builder
                                            .add_instruction(
                                                ConstantOperation::new(limit),
                                                Vec::new(),
                                                Vec::new(),
                                                None,
                                            )
                                            .unwrap()[0];
                                        let predicate = builder
                                            .add_instruction(
                                                XlaOperation::Array(ArrayOperation::Compare(CompareOperation::new(
                                                    ComparisonDirection::LessThan,
                                                ))),
                                                Vec::new(),
                                                vec![state, limit],
                                                None,
                                            )
                                            .unwrap()[0];
                                        builder.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap()
                                    };
                                    let body = {
                                        let mut builder = XlaProgramBuilder::new();
                                        let state = builder.add_input(ArrayIrType::Array(local_type));
                                        let next_state = builder
                                            .add_instruction(MulOperation::new(), Vec::new(), vec![state, state], None)
                                            .unwrap()[0];
                                        builder.build(vec![next_state], vec![Placeholder], vec![Placeholder]).unwrap()
                                    };
                                    let context = local_x.value().context().clone();
                                    let mut outputs = context
                                        .stage_operation(
                                            XlaOperation::While(WhileOperation::new().with_iteration_bound(4).unwrap()),
                                            vec![condition, body],
                                            &[local_x.into_value()],
                                        )
                                        .unwrap();
                                    ValueProjection::<ArrayType>::into_projected(outputs.remove(0)).unwrap()
                                },
                                x,
                                mesh.clone(),
                                sharded.clone(),
                                sharded.clone(),
                            )
                            .unwrap()
                            .reduce(&[0], ReductionKind::Sum)
                            .unwrap()
                            .into_value())
                        })
                        .unwrap();
                    vec![
                        ValueProjection::<ArrayType>::into_projected(value).unwrap(),
                        ValueProjection::<ArrayType>::into_projected(gradient).unwrap(),
                    ]
                }
            },
            ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap(),
        )
        .unwrap();
        let program = traced
            .to_mlir_module_with_signature_shardings(
                "main",
                Some(std::slice::from_ref(&sharded)),
                Some(&[replicated, sharded.clone()]),
            )
            .unwrap();
        let executable = client
            .compile(&Program::Mlir { bytecode: program.into_bytes() }, &test_spmd_compilation_options(4))
            .unwrap();
        let inputs = [1.5f32, 2.0, 3.0, 20.0];
        let buffers = devices
            .iter()
            .zip(inputs)
            .map(|(device, input)| {
                client
                    .buffer(values_to_bytes(&[input]).as_slice(), BufferType::F32, [1], None, device.clone(), None)
                    .unwrap()
            })
            .collect();
        let input = XlaArray::from_addressable_buffers(
            &domain,
            static_sharded_array_type(DataType::F32, &[4], sharded),
            device_mesh,
            buffers,
        )
        .unwrap();
        let device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let arguments = XlaArray::into_execute_arguments(vec![input], &device_ids).unwrap();
        let outputs = executable
            .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();

        // Each shard computes `x^(2^k)` after `k` iterations, whose derivative is `2^k x^(2^k - 1)`.
        let expected_gradients = devices
            .iter()
            .map(|device| device.id().unwrap())
            .zip([8.0 * 1.5f32.powi(7), 4.0 * 2f32.powi(3), 4.0 * 3f32.powi(3), 1.0])
            .collect::<HashMap<_, _>>();
        assert_eq!(outputs.len(), 4);
        for (output, device_id) in outputs.into_iter().zip(device_ids) {
            let values = output
                .outputs
                .iter()
                .map(|buffer| values_from_bytes::<f32>(&buffer.copy_to_host(None).unwrap().r#await().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(values, vec![vec![1.5f32.powi(8) + 16.0 + 81.0 + 20.0], vec![expected_gradients[&device_id]]]);
        }
    }

    #[test]
    fn test_shard_map_custom_call_lowers_inside_manual_region_and_executes_on_cpu() {
        use ryft_core::{CustomCall, CustomCallOperation};

        use crate::tests::{ADD_ONE_CUSTOM_CALL_TARGET, ensure_add_one_handler_registered};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .expect("failed to create 2-device CPU client");
        let domain = XlaSession::new(&client).domain();
        ensure_add_one_handler_registered(&client).unwrap();
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 2);

        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let sharding =
            Sharding::new(device_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));

        // The custom call executes per shard inside the manual region: its declared output type is the local
        // shard type, and the lowered `stablehlo.custom_call` appears inside the `sdy.manual_computation` body.
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = device_mesh.logical_mesh().clone();
                let sharding = sharding.clone();
                move |x: XlaArrayTracer| {
                    shard_map(
                        |local_x: XlaArrayTracer| {
                            let operation = CustomCallOperation::new(
                                ADD_ONE_CUSTOM_CALL_TARGET,
                                vec![local_x.r#type().into_owned()],
                            );
                            CustomCall::custom_call(&operation, std::slice::from_ref(&local_x)).unwrap().remove(0)
                        },
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .expect("shard_map with a custom call should trace")
                }
            },
            global_input_type,
        )
        .unwrap();

        let mlir_program = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            mlir_program,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = stablehlo.custom_call @ryft.test.add_one(%arg1) {api_version = 4 : i32, backend_config = {}} : (tensor<2xf32>) -> tensor<2xf32>
                      sdy.return %1 : tensor<2xf32>
                    } : (tensor<4xf32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let shard_values = [device_index as f32 * 2.0 + 1.0, device_index as f32 * 2.0 + 2.0];
                client
                    .buffer(
                        values_to_bytes::<f32>(&shard_values).as_slice(),
                        BufferType::F32,
                        [2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array = XlaArray::from_addressable_buffers(
            &domain,
            static_sharded_array_type(DataType::F32, &[4], sharding),
            device_mesh,
            input_buffers,
        )
        .unwrap();
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(2)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 2);
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        let execute_arguments =
            XlaArray::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();

        // Shards are [1, 2] and [3, 4], so the kernel produces [2, 3] and [4, 5] respectively.
        assert_eq!(outputs.len(), execution_device_ids.len());
        for (device_index, output) in outputs.into_iter().enumerate() {
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values: [f32; 2] = values_from_bytes::<f32>(output_bytes.as_slice()).try_into().unwrap();
            assert_eq!(values, [device_index as f32 * 2.0 + 2.0, device_index as f32 * 2.0 + 3.0]);
        }
    }

    #[test]
    fn test_shard_map_parallel_mean_lowers_to_all_reduce_with_axis_size_division() {
        use ryft_core::{ParallelReduce, ReductionKind};

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));

        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = mesh.clone();
                let sharding = sharding.clone();
                move |x: XlaArrayTracer| {
                    shard_map(
                        |local_x: XlaArrayTracer| local_x.parallel_reduce(ReductionKind::Mean, "x").unwrap(),
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .expect("shard_map with a mean parallel_reduce should trace")
                }
            },
            global_input_type,
        )
        .unwrap();

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %4 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %4 : tensor<f32>
                      }) : (tensor<2xf32>) -> tensor<2xf32>
                      %cst = stablehlo.constant dense<4.000000e+00> : tensor<f32>
                      %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2xf32>
                      %3 = stablehlo.divide %1, %2 : tensor<2xf32>
                      sdy.return %3 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_shard_map_grouped_parallel_mean_preserves_group_order_and_uses_group_divisor() {
        use ryft_core::{ParallelReduce, ReductionKind};

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let sharding = sharding.clone();
                move |x: XlaArrayTracer| {
                    shard_map(
                        |local_x: XlaArrayTracer| {
                            local_x
                                .parallel_reduce_with_axis_index_groups(
                                    ReductionKind::Mean,
                                    "x",
                                    vec![vec![0, 2], vec![3, 1]],
                                )
                                .unwrap()
                        },
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .unwrap()
                }
            },
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)])),
        )
        .unwrap();

        let module = traced.to_mlir_module("main").unwrap();
        assert!(module.contains("replica_groups = dense<[[0, 2], [3, 1]]> : tensor<2x2xi64>"), "{module}",);
        assert!(module.contains("stablehlo.constant dense<2.000000e+00> : tensor<f32>"), "{module}");
    }

    #[test]
    fn test_batch_inside_shard_map_forwards_mesh_collective_to_all_reduce() {
        use ryft_core::{Batch, BatchAxis, BatchAxisSpecification, ParallelReduce, ReductionKind};

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));

        // A named `batch` level inside the shard_map body binds `"b"`, while the sum `parallel_reduce` names the *mesh*
        // axis `"x"`: the batching rule forwards the collective through the batch level to the seeded base trace (which
        // binds `"x"`), so it lands in the body program on the batched physical value and lowers to the same
        // `all_reduce` as a direct sum `parallel_reduce` — resolution composes across binder kinds.
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = mesh.clone();
                let sharding = sharding.clone();
                move |x: XlaArrayTracer| {
                    shard_map(
                        |local_x: XlaArrayTracer| {
                            let context = local_x.domain();
                            let summed: XlaArrayTracer = Batch::batch(
                                &context,
                                |item| item.parallel_reduce(ReductionKind::Sum, "x"),
                                local_x,
                                BatchAxis::new(0),
                                BatchAxis::new(0),
                                BatchAxisSpecification::named("b"),
                            )
                            .expect("vmap inside shard_map should trace");
                            summed
                        },
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .expect("shard_map with a vmapped sum parallel_reduce should trace")
                }
            },
            global_input_type,
        )
        .unwrap();

        // The module is identical to the direct sum `parallel_reduce` module: the batch level forwarded the mesh
        // collective untouched.
        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %2 : tensor<f32>
                      }) : (tensor<2xf32>) -> tensor<2xf32>
                      sdy.return %1 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_batched_shard_map_matches_per_item_results_on_cpu() {
        use ryft_core::{ArrayIrBatchingPolicy, BatchingContext, ParallelReduce, ShardMapTracer};

        // Batching a `shard_map` that sums its local shards across the four devices along `x` over a leading axis of
        // three items yields, for every item, what the unbatched `shard_map` yields for that item alone. The batch
        // dimension is an unpartitioned leading dimension of the boundary, so every device holds all three items of its
        // shard.
        type BatchedShardMapTracer =
            ShardMapTracer<BatchingContext<DomainTracingContext<XlaDomain<'static>>, ArrayIrBatchingPolicy>>;
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = XlaSession::new(&client).domain();
        let devices = client.addressable_devices().unwrap();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let device_mesh =
            DeviceMesh::new(mesh.clone(), devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect())
                .unwrap();
        let item_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let batched_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
                .unwrap();
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = mesh.clone();
                let item_sharding = item_sharding.clone();
                move |x: XlaArrayTracer| {
                    let output = batch(
                        |item| {
                            let item = ValueProjection::<ArrayType>::into_projected(item)?;
                            let output = shard_map(
                                |local: BatchedShardMapTracer| local.parallel_reduce(ReductionKind::Sum, "x").unwrap(),
                                item,
                                mesh.clone(),
                                item_sharding.clone(),
                                item_sharding.clone(),
                            )?;
                            Ok(output.into_value())
                        },
                        x.into_value(),
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        BatchAxisSpecification::default(),
                    )
                    .unwrap();
                    ValueProjection::<ArrayType>::into_projected(output).unwrap()
                }
            },
            ArrayType::new_static(DataType::F32, [3, 8]),
        )
        .unwrap();
        let mlir_program = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            mlir_program,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<3x8xf32>) -> tensor<3x8xf32> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {"x"}]>] out_shardings=[<@mesh, [{}, {"x"}]>] manual_axes={"x"} (%arg1: tensor<3x2xf32>) {
                      %1 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %2 : tensor<f32>
                      }) : (tensor<3x2xf32>) -> tensor<3x2xf32>
                      sdy.return %1 : tensor<3x2xf32>
                    } : (tensor<3x8xf32>) -> tensor<3x8xf32>
                    return %0 : tensor<3x8xf32>
                  }
                }
            "#},
        );

        // Item `b` of the global `f32[3, 8]` input holds `8 · b + j` at column `j`, so device `d` holds columns
        // `2 · d` and `2 · d + 1` of every item.
        let shard = |item: usize, device: usize| [(8 * item + 2 * device) as f32, (8 * item + 2 * device + 1) as f32];
        let buffers = devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let values = (0..3).flat_map(|item| shard(item, device_index)).collect::<Vec<_>>();
                client
                    .buffer(
                        values_to_bytes::<f32>(&values).as_slice(),
                        BufferType::F32,
                        [3u64, 2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect();
        let input = XlaArray::from_addressable_buffers(
            &domain,
            static_sharded_array_type(DataType::F32, &[3, 8], batched_sharding),
            device_mesh,
            buffers,
        )
        .unwrap();
        let executable = client
            .compile(&Program::Mlir { bytecode: mlir_program.into_bytes() }, &test_spmd_compilation_options(4))
            .unwrap();
        let device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let arguments = XlaArray::into_execute_arguments(vec![input], &device_ids).unwrap();
        let outputs = executable
            .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap()
            .into_iter()
            .map(|output| {
                values_from_bytes::<f32>(output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap().as_slice())
            })
            .collect::<Vec<_>>();
        for item in 0..3 {
            let item_outputs = execute_parallel_reduce_on_cpu(
                ReductionKind::Sum,
                DataType::F32,
                BufferType::F32,
                2,
                std::array::from_fn(|device| values_to_bytes::<f32>(&shard(item, device))),
            );
            for (device_outputs, item_output) in outputs.iter().zip(item_outputs) {
                assert_eq!(
                    &device_outputs[2 * item..2 * item + 2],
                    values_from_bytes::<f32>(item_output.as_slice()).as_slice(),
                );
            }
        }
    }

    #[test]
    fn test_spmd_batched_shard_map_matches_per_item_results_on_cpu() {
        use ryft_core::{ArrayIrBatchingPolicy, BatchingContext, ParallelReduce, ShardMapTracer};

        // Batching a `shard_map` over the manual axes `x` and `y` of four CPU devices, whose body sums its local shards
        // along `x` only, over a leading axis of four items that the input places on `y` (the analogue of JAX's
        // `spmd_axis_name`) makes `y` free in the batched `shard_map`: every device holds the two items of its
        // coordinate along `y`, and the result for every item is what the unbatched `shard_map` yields for that item.
        type BatchedShardMapTracer =
            ShardMapTracer<BatchingContext<DomainTracingContext<XlaDomain<'static>>, ArrayIrBatchingPolicy>>;
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = XlaSession::new(&client).domain();
        let devices = client.addressable_devices().unwrap();
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let caller_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let device_mesh = DeviceMesh::new(
            caller_mesh.clone(),
            devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let item_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let batched_sharding =
            Sharding::new(caller_mesh, vec![ShardingDimension::sharded(["y"]), ShardingDimension::sharded(["x"])])
                .unwrap();
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = mesh.clone();
                let item_sharding = item_sharding.clone();
                move |x: XlaArrayTracer| {
                    let output = batch(
                        |item| {
                            let item = ValueProjection::<ArrayType>::into_projected(item)?;
                            let output = shard_map(
                                |local: BatchedShardMapTracer| local.parallel_reduce(ReductionKind::Sum, "x").unwrap(),
                                item,
                                mesh.clone(),
                                item_sharding.clone(),
                                item_sharding.clone(),
                            )?;
                            Ok(output.into_value())
                        },
                        x.into_value(),
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        BatchAxisSpecification::default(),
                    )
                    .unwrap();
                    ValueProjection::<ArrayType>::into_projected(output).unwrap()
                }
            },
            static_sharded_array_type(DataType::F32, &[4, 8], batched_sharding.clone()),
        )
        .unwrap();
        let mlir_program = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            mlir_program,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2, "y"=2]>
                  func.func @main(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"y", ?}, {"x"}]>] out_shardings=[<@mesh, [{"y", ?}, {"x"}]>] manual_axes={"x"} (%arg1: tensor<4x4xf32>) {
                      %1 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %2 : tensor<f32>
                      }) : (tensor<4x4xf32>) -> tensor<4x4xf32>
                      sdy.return %1 : tensor<4x4xf32>
                    } : (tensor<4x8xf32>) -> tensor<4x8xf32>
                    return %0 : tensor<4x8xf32>
                  }
                }
            "#},
        );

        // Item `b` of the global `f32[4, 8]` input holds `8 · b + j` at column `j`, so the device at mesh coordinates
        // `(x, y)` (device `2 · x + y`) holds columns `4 · x` to `4 · x + 3` of items `2 · y` and `2 · y + 1`.
        let shard = |device: usize| {
            let (x, y) = (device / 2, device % 2);
            (2 * y..2 * y + 2)
                .flat_map(|item| (0..4).map(move |column| (8 * item + 4 * x + column) as f32))
                .collect::<Vec<_>>()
        };
        let buffers = devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                client
                    .buffer(
                        values_to_bytes::<f32>(&shard(device_index)).as_slice(),
                        BufferType::F32,
                        [2u64, 4u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect();
        let input = XlaArray::from_addressable_buffers(
            &domain,
            static_sharded_array_type(DataType::F32, &[4, 8], batched_sharding),
            device_mesh,
            buffers,
        )
        .unwrap();
        let executable = client
            .compile(&Program::Mlir { bytecode: mlir_program.into_bytes() }, &test_spmd_compilation_options(4))
            .unwrap();
        let device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let arguments = XlaArray::into_execute_arguments(vec![input], &device_ids).unwrap();
        let outputs = executable
            .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap()
            .into_iter()
            .map(|output| {
                values_from_bytes::<f32>(output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap().as_slice())
            })
            .collect::<Vec<_>>();

        // The unbatched `shard_map` for item `b` sums the two column blocks of that item along `x`, so its output holds
        // `(8 · b + c) + (8 · b + 4 + c) = 16 · b + 2 · c + 4` at column `c` of every block. The device at mesh
        // coordinates `(x, y)` holds the output columns `4 · x` to `4 · x + 3` of items `2 · y` and `2 · y + 1`.
        for (device_index, device_outputs) in outputs.iter().enumerate() {
            let y = device_index % 2;
            let expected = (2 * y..2 * y + 2)
                .flat_map(|item| (0..4).map(move |column| (16 * item + 2 * column + 4) as f32))
                .collect::<Vec<_>>();
            assert_eq!(device_outputs, &expected);
        }
    }

    #[test]
    fn test_parallel_reduce_inside_condition_inside_shard_map_lowers_to_all_reduce() {
        use ryft_core::{ConditionOperation, ParallelReduceOperation, ReductionKind};

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));

        // The sum `parallel_reduce` sits inside a `condition` branch inside the shard_map body, so it lowers through
        // the nested control-flow path: the threaded collective lowering state resolves the manual mesh axis inside the
        // `stablehlo.if` region and emits the same `all_reduce` as a body-level sum `parallel_reduce` would.
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = mesh.clone();
                let sharding = sharding.clone();
                move |x: XlaArrayTracer| {
                    shard_map(
                        |local_x: XlaArrayTracer| {
                            let local_type = local_x.r#type().into_owned();
                            let parallel_sum_branch = {
                                let mut builder = XlaProgramBuilder::new();
                                let input = builder.add_input(ArrayIrType::Array(local_type.clone()));
                                let output = builder
                                    .add_instruction(
                                        ParallelReduceOperation::new(ReductionKind::Sum, "x".to_string()),
                                        Vec::new(),
                                        vec![input],
                                        None,
                                    )
                                    .unwrap()[0];
                                builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
                            };
                            let identity_branch = {
                                let mut builder = XlaProgramBuilder::new();
                                let input = builder.add_input(ArrayIrType::Array(local_type));
                                builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
                            };
                            let context = local_x.value().context().clone();
                            let predicate = context.lift(XlaConstant::Boolean(true)).unwrap();
                            let mut outputs = context
                                .stage_operation(
                                    XlaOperation::Condition(ConditionOperation::new()),
                                    vec![parallel_sum_branch, identity_branch],
                                    &[predicate, local_x.into_value()],
                                )
                                .unwrap();
                            ValueProjection::<ArrayType>::into_projected(outputs.remove(0))
                                .expect("condition output should remain an array")
                        },
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .expect("shard_map with a sum parallel_reduce inside a condition should trace")
                }
            },
            global_input_type,
        )
        .unwrap();

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %c = stablehlo.constant dense<true> : tensor<i1>
                      %1 = "stablehlo.if"(%c) ({
                        %2 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
                        ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                          %3 = stablehlo.add %arg2, %arg3 : tensor<f32>
                          stablehlo.return %3 : tensor<f32>
                        }) : (tensor<2xf32>) -> tensor<2xf32>
                        stablehlo.return %2 : tensor<2xf32>
                      }, {
                        stablehlo.return %arg1 : tensor<2xf32>
                      }) : (tensor<i1>) -> tensor<2xf32>
                      sdy.return %1 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_ordinary_parallel_log_sum_exp_inside_shard_map_is_rejected_by_lowering() {
        use ryft_core::{ParallelReduceOperation, ReductionKind};

        // The `ParallelReduce` capability composes a logarithmic sum over a manual mesh axis from primitive mesh
        // reductions, so only a hand-staged ordinary reduction can reach the lowering, which has no combiner for it.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            move |x: XlaArrayTracer| {
                shard_map(
                    |local_x: XlaArrayTracer| {
                        let context = local_x.value().context().clone();
                        let mut outputs = context
                            .stage_operation(
                                ParallelReduceOperation::new(ReductionKind::LogSumExp, "x".to_string()),
                                Vec::new(),
                                &[local_x.into_value()],
                            )
                            .unwrap();
                        ValueProjection::<ArrayType>::into_projected(outputs.remove(0)).unwrap()
                    },
                    x,
                    mesh.clone(),
                    sharding.clone(),
                    sharding.clone(),
                )
                .unwrap()
            },
            ArrayType::new_static(DataType::F32, [8]),
        )
        .unwrap();
        assert_eq!(
            traced.to_mlir_module("main").unwrap_err().to_string(),
            "`parallel_reduce` with kind `log_sum_exp` has no all-reduce combiner; stage it through the \
             `ParallelReduce` capability, which composes it from primitive mesh reductions",
        );
    }

    #[test]
    fn test_two_shard_maps_with_collectives_receive_unique_channel_ids() {
        use ryft_core::{ParallelReduce, ReductionKind};

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));

        // Two manual regions in one module each emit a channeled `all_reduce`: the module-scoped channel allocator
        // hands out distinct handles (1 and 2), which XLA requires across a module.
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = mesh.clone();
                let sharding = sharding.clone();
                move |x: XlaArrayTracer| {
                    let first = shard_map(
                        |local_x: XlaArrayTracer| local_x.parallel_reduce(ReductionKind::Sum, "x").unwrap(),
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .expect("first shard_map should trace");
                    shard_map(
                        |local_x: XlaArrayTracer| local_x.parallel_reduce(ReductionKind::Sum, "x").unwrap(),
                        first,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .expect("second shard_map should trace")
                }
            },
            global_input_type,
        )
        .unwrap();

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %2 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %3 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %3 : tensor<f32>
                      }) : (tensor<2xf32>) -> tensor<2xf32>
                      sdy.return %2 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %2 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %3 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %3 : tensor<f32>
                      }) : (tensor<2xf32>) -> tensor<2xf32>
                      sdy.return %2 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %1 : tensor<8xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_shard_map_axis_index_lowers_to_partition_id_coordinate_and_executes_on_cpu() {
        use ryft_core::{AxisIndex, Broadcast};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .expect("failed to create 4-device CPU client");
        let domain = XlaSession::new(&client).domain();
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 4);

        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let sharding =
            Sharding::new(device_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(4)]));

        // `axis_index("x")` gives each device its own coordinate along the manual mesh axis `"x"`, added to the local
        // shard. The single-axis mesh has unit stride and full-size axis, so the coordinate is just `partition_id`.
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = device_mesh.logical_mesh().clone();
                let sharding = sharding.clone();
                move |x: XlaArrayTracer| {
                    shard_map(
                        |local_x: XlaArrayTracer| {
                            // `axis_index` is a scalar per device; broadcast it to the shard shape so the add has
                            // shape-congruent operands (StableHLO has no implicit broadcasting).
                            let local_type = local_x.r#type().into_owned();
                            let index = local_x.domain().axis_index("x").unwrap();
                            let index_sharding = index
                                .r#type()
                                .sharding()
                                .unwrap()
                                .with_broadcasted_dimensions(local_type.rank(), &[])
                                .unwrap();
                            let index_type = local_type.with_sharding(index_sharding).unwrap();
                            let index = index.broadcast(index_type, &[]).unwrap();
                            local_x + index
                        },
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .expect("shard_map with axis_index should trace")
                }
            },
            global_input_type,
        )
        .unwrap();

        let mlir_program = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            mlir_program,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<4xui64>) -> tensor<4xui64> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<1xui64>) {
                      %1 = stablehlo.partition_id : tensor<ui32>
                      %2 = stablehlo.convert %1 : (tensor<ui32>) -> tensor<ui64>
                      %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<ui64>) -> tensor<1xui64>
                      %4 = stablehlo.add %arg1, %3 : tensor<1xui64>
                      sdy.return %4 : tensor<1xui64>
                    } : (tensor<4xui64>) -> tensor<4xui64>
                    return %0 : tensor<4xui64>
                  }
                }
            "#},
        );

        let input_buffers = client_devices
            .iter()
            .map(|device| {
                client
                    .buffer(
                        values_to_bytes::<u64>(&[10u64]).as_slice(),
                        BufferType::U64,
                        [1u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array = XlaArray::from_addressable_buffers(
            &domain,
            static_sharded_array_type(DataType::U64, &[4], sharding),
            device_mesh,
            input_buffers,
        )
        .unwrap();
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(4)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        let execute_arguments =
            XlaArray::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();

        // Each device d holds shard [10] and adds its coordinate d, so device d outputs [10 + d].
        assert_eq!(outputs.len(), execution_device_ids.len());
        for (device_index, output) in outputs.into_iter().enumerate() {
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values: [u64; 1] = values_from_bytes::<u64>(output_bytes.as_slice()).try_into().unwrap();
            assert_eq!(values, [10 + device_index as u64]);
        }
    }

    #[test]
    fn test_shard_map_in_context_without_inputs_executes_on_cpu() {
        use ryft_core::AxisIndex;

        // JAX's `test_axis_index`: a body without inputs reads the coordinate of each device along `x` through its
        // context, and the output sharding tiles the coordinates along `x`, so the global output is `[0, 1, 2, 3]`. The
        // `shard_map` lowers to an `sdy.manual_computation` without operands. The traced program takes an unused input
        // only because its closure receives no context otherwise.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = XlaSession::new(&client).domain();
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 4);
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let device_mesh = DeviceMesh::new(
            mesh.clone(),
            client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let sharded = sharded.clone();
                move |x: XlaArrayTracer| {
                    shard_map_in_context(
                        x.value().context(),
                        |context: &ShardMapContext<DomainTracingContext<XlaDomain<'static>>>, ()| {
                            context.axis_index("x").unwrap().reshape([1]).unwrap()
                        },
                        (),
                        mesh,
                        (),
                        sharded,
                        Vec::new(),
                    )
                    .unwrap()
                }
            },
            ArrayType::scalar(DataType::F32),
        )
        .unwrap();
        let program = traced
            .to_mlir_module_with_signature_shardings("main", Some(&[replicated.clone()]), Some(&[sharded]))
            .unwrap();
        assert_eq!(
            program,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, [], replicated={"x"}>}) -> (tensor<4xui64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation() in_shardings=[] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} () {
                      %1 = stablehlo.partition_id : tensor<ui32>
                      %2 = stablehlo.convert %1 : (tensor<ui32>) -> tensor<ui64>
                      %3 = stablehlo.reshape %2 : (tensor<ui64>) -> tensor<1xui64>
                      sdy.return %3 : tensor<1xui64>
                    } : () -> tensor<4xui64>
                    return %0 : tensor<4xui64>
                  }
                }
            "#},
        );

        let executable = client
            .compile(&Program::Mlir { bytecode: program.into_bytes() }, &test_spmd_compilation_options(4))
            .unwrap();
        let input_buffers = client_devices
            .iter()
            .map(|device| client.buffer(&0f32.to_ne_bytes(), BufferType::F32, [], None, device.clone(), None).unwrap())
            .collect();
        let input = XlaArray::from_addressable_buffers(
            &domain,
            static_sharded_array_type(DataType::F32, &[], replicated),
            device_mesh,
            input_buffers,
        )
        .unwrap();
        let device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let arguments = XlaArray::into_execute_arguments(vec![input], &device_ids).unwrap();
        let outputs = executable
            .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();

        // Device `d` holds shard `d` of the output, which is its own coordinate along `x`.
        let values = outputs
            .into_iter()
            .flat_map(|output| {
                assert_eq!(output.outputs.len(), 1);
                values_from_bytes::<u64>(&output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap())
            })
            .collect::<Vec<_>>();
        assert_eq!(values, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_shard_map_axis_index_of_major_mesh_axis_lowers_to_divide_and_remainder() {
        use ryft_core::{AxisIndex, Broadcast};

        // A 2x2 mesh: `axis_index("x")` addresses the major axis (row-major stride 2, size 2), so the device
        // coordinate is `(partition_id / 2) % 2` — exercising both the divide and the remainder that a single-axis
        // mesh skips.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x", "y"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(4)]));

        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = mesh.clone();
                let sharding = sharding.clone();
                move |x: XlaArrayTracer| {
                    shard_map(
                        |local_x: XlaArrayTracer| {
                            let local_type = local_x.r#type().into_owned();
                            let index = local_x.domain().axis_index("x").unwrap();
                            let index_sharding = index
                                .r#type()
                                .sharding()
                                .unwrap()
                                .with_broadcasted_dimensions(local_type.rank(), &[])
                                .unwrap();
                            let index_type = local_type.with_sharding(index_sharding).unwrap();
                            index.broadcast(index_type, &[]).unwrap()
                        },
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .expect("shard_map with axis_index of the major axis should trace")
                }
            },
            global_input_type,
        )
        .unwrap();

        // The body reads only the shape of its input, so pruning drops the input from the manual computation.
        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2, "y"=2]>
                  func.func @main(%arg0: tensor<4xui64>) -> tensor<4xui64> {
                    %0 = sdy.manual_computation() in_shardings=[] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x", "y"} () {
                      %1 = stablehlo.partition_id : tensor<ui32>
                      %2 = stablehlo.convert %1 : (tensor<ui32>) -> tensor<ui64>
                      %c = stablehlo.constant dense<2> : tensor<ui64>
                      %3 = stablehlo.divide %2, %c : tensor<ui64>
                      %c_0 = stablehlo.constant dense<2> : tensor<ui64>
                      %4 = stablehlo.remainder %3, %c_0 : tensor<ui64>
                      %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<ui64>) -> tensor<1xui64>
                      sdy.return %5 : tensor<1xui64>
                    } : () -> tensor<4xui64>
                    return %0 : tensor<4xui64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_shard_map_parallel_all_gather_lowers_and_executes_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .expect("failed to create 2-device CPU client");
        let domain = XlaSession::new(&client).domain();
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 2);

        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let sharding =
            Sharding::new(device_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));

        // `parallel_all_gather` over the manual mesh axis `"x"` extends the local shard from `f32[2]` to the full
        // `f32[4]` concatenation on every device. Its output still varies along the manual axis for VMA purposes (a
        // replicated output sharding is rejected with `OutputVaryingAlongUntiledManualAxis`, mirroring JAX's vma
        // tracking), so the output stays sharded over `"x"`, giving the global `f32[8]` concatenation of the per-device
        // gathers. The staged collective lowers to a channeled `stablehlo.all_gather` over the two devices along `"x"`.
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = device_mesh.logical_mesh().clone();
                let sharding = sharding.clone();
                move |x: XlaArrayTracer| {
                    shard_map(
                        |local_x: XlaArrayTracer| {
                            local_x
                                .parallel_all_gather_with_options(
                                    "x",
                                    0,
                                    CollectiveOptions::tiled(),
                                    ParallelAllGatherOutputVariance::Varying,
                                )
                                .unwrap()
                        },
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .expect("shard_map with parallel_all_gather should trace")
                }
            },
            global_input_type,
        )
        .unwrap();

        let mlir_program = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            mlir_program,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<8xf32> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %c = stablehlo.constant dense<2> : tensor<i64>
                      %c_0 = stablehlo.constant dense<2> : tensor<i64>
                      %1 = stablehlo.multiply %c, %c_0 : tensor<i64>
                      %2 = "stablehlo.all_gather"(%arg1) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> : (tensor<2xf32>) -> tensor<4xf32>
                      sdy.return %2 : tensor<4xf32>
                    } : (tensor<4xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#},
        );

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let shard_values = [device_index as f32 * 2.0 + 1.0, device_index as f32 * 2.0 + 2.0];
                client
                    .buffer(
                        values_to_bytes::<f32>(&shard_values).as_slice(),
                        BufferType::F32,
                        [2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array = XlaArray::from_addressable_buffers(
            &domain,
            static_sharded_array_type(DataType::F32, &[4], sharding),
            device_mesh,
            input_buffers,
        )
        .unwrap();
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(2)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 2);
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        let execute_arguments =
            XlaArray::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();

        // Shards are [1, 2] and [3, 4], so every device receives the full concatenation [1, 2, 3, 4].
        assert_eq!(outputs.len(), execution_device_ids.len());
        for output in outputs {
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values: [f32; 4] = values_from_bytes::<f32>(output_bytes.as_slice()).try_into().unwrap();
            assert_eq!(values, [1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn test_shard_map_untiled_parallel_all_gather_lowers_rank_insertion() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .expect("failed to create 2-device CPU client");
        let domain = XlaSession::new(&client).domain();
        let client_devices = client.addressable_devices().unwrap();
        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let mesh = device_mesh.logical_mesh().clone();
        let input_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let output_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
                .unwrap();
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let input_sharding = input_sharding.clone();
                let output_sharding = output_sharding.clone();
                move |x: XlaArrayTracer| {
                    shard_map(
                        |local_x: XlaArrayTracer| {
                            local_x
                                .parallel_all_gather_with_options(
                                    "x",
                                    0,
                                    CollectiveOptions::default(),
                                    ParallelAllGatherOutputVariance::Varying,
                                )
                                .unwrap()
                        },
                        x,
                        mesh.clone(),
                        input_sharding.clone(),
                        output_sharding.clone(),
                    )
                    .unwrap()
                }
            },
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)])),
        )
        .unwrap();

        let module = traced.to_mlir_module("main").unwrap();
        assert!(module.contains("stablehlo.broadcast_in_dim"), "{module}");
        assert!(
            module.contains("(tensor<2xf32>) -> tensor<1x2xf32>")
                && module.contains("(tensor<1x2xf32>) -> tensor<2x2xf32>"),
            "{module}",
        );

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let values = [device_index as f32 * 2.0 + 1.0, device_index as f32 * 2.0 + 2.0];
                client
                    .buffer(
                        values_to_bytes(values.as_slice()).as_slice(),
                        BufferType::F32,
                        [2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array = XlaArray::from_addressable_buffers(
            &domain,
            static_sharded_array_type(DataType::F32, &[4], input_sharding),
            device_mesh,
            input_buffers,
        )
        .unwrap();
        let executable = client
            .compile(&Program::Mlir { bytecode: module.into_bytes() }, &test_spmd_compilation_options(2))
            .unwrap();
        let execution_device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let execute_arguments =
            XlaArray::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();
        for output in outputs {
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(values_from_bytes::<f32>(output_bytes.as_slice()), vec![1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn test_shard_map_grouped_parallel_all_gather_preserves_group_order_across_mesh_coordinates() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let sharding = sharding.clone();
                move |x: XlaArrayTracer| {
                    shard_map(
                        |local_x: XlaArrayTracer| {
                            local_x
                                .parallel_all_gather_with_options(
                                    "x",
                                    0,
                                    CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]),
                                    ParallelAllGatherOutputVariance::Varying,
                                )
                                .unwrap()
                        },
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .unwrap()
                }
            },
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)])),
        )
        .unwrap();

        let module = traced.to_mlir_module("main").unwrap();
        assert!(
            module.contains("replica_groups = dense<[[0, 4], [6, 2], [1, 5], [7, 3]]> : tensor<4x2xi64>"),
            "{module}",
        );
        assert!(module.contains("(tensor<2xf32>) -> tensor<4xf32>"), "{module}");
    }

    #[test]
    fn test_shard_map_grouped_shape_changing_collectives_execute_on_cpu() {
        use ryft_core::{ParallelAllToAll, ParallelSumScatter};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .expect("failed to create 4-device CPU client");
        let domain = XlaSession::new(&client).domain();
        let client_devices = client.addressable_devices().unwrap();
        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let mesh = device_mesh.logical_mesh().clone();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let traced: TracedXlaProgram<ArrayType, (ArrayType, ArrayType, ArrayType)> = trace(
            {
                let sharding = sharding.clone();
                move |x: XlaArrayTracer| {
                    shard_map(
                        |local_x: XlaArrayTracer| {
                            let options =
                                CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]);
                            (
                                local_x
                                    .parallel_all_gather_with_options(
                                        "x",
                                        0,
                                        options.clone(),
                                        ParallelAllGatherOutputVariance::Varying,
                                    )
                                    .unwrap(),
                                local_x.clone().parallel_sum_scatter_with_options("x", 0, options.clone()).unwrap(),
                                local_x.parallel_all_to_all_with_options("x", 0, 0, options).unwrap(),
                            )
                        },
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        (sharding.clone(), sharding.clone(), sharding.clone()),
                    )
                    .unwrap()
                }
            },
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(16)])),
        )
        .unwrap();

        let module = traced.to_mlir_module("main").unwrap();
        assert_eq!(module.matches("replica_groups = dense<[[0, 2], [3, 1]]> : tensor<2x2xi64>").count(), 3, "{module}");

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let values = (0..4).map(|offset| (device_index * 4 + offset) as f32).collect::<Vec<_>>();
                client
                    .buffer(
                        values_to_bytes(values.as_slice()).as_slice(),
                        BufferType::F32,
                        [4u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array = XlaArray::from_addressable_buffers(
            &domain,
            static_sharded_array_type(DataType::F32, &[16], sharding),
            device_mesh,
            input_buffers,
        )
        .unwrap();
        let executable = client
            .compile(&Program::Mlir { bytecode: module.into_bytes() }, &test_spmd_compilation_options(4))
            .unwrap();
        let execution_device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let execute_arguments =
            XlaArray::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();

        let expected_gather = [
            vec![0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0],
            vec![12.0, 13.0, 14.0, 15.0, 4.0, 5.0, 6.0, 7.0],
            vec![0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0],
            vec![12.0, 13.0, 14.0, 15.0, 4.0, 5.0, 6.0, 7.0],
        ];
        let expected_scatter = [vec![8.0, 10.0], vec![20.0, 22.0], vec![12.0, 14.0], vec![16.0, 18.0]];
        let expected_parallel_all_to_all = [
            vec![0.0, 1.0, 8.0, 9.0],
            vec![14.0, 15.0, 6.0, 7.0],
            vec![2.0, 3.0, 10.0, 11.0],
            vec![12.0, 13.0, 4.0, 5.0],
        ];
        for (device_index, output) in outputs.into_iter().enumerate() {
            assert_eq!(output.outputs.len(), 3);
            let actual = output
                .outputs
                .into_iter()
                .map(|buffer| {
                    let bytes = buffer.copy_to_host(None).unwrap().r#await().unwrap();
                    values_from_bytes::<f32>(bytes.as_slice())
                })
                .collect::<Vec<_>>();
            assert_eq!(
                actual,
                vec![
                    expected_gather[device_index].clone(),
                    expected_scatter[device_index].clone(),
                    expected_parallel_all_to_all[device_index].clone(),
                ],
            );
        }
    }

    #[test]
    fn test_shard_map_parallel_sum_scatter_lowers_and_executes_on_cpu() {
        use ryft_core::{CollectiveOptions, ParallelSumScatter};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .expect("failed to create 2-device CPU client");
        let domain = XlaSession::new(&client).domain();
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 2);

        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let sharding =
            Sharding::new(device_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));

        // `parallel_sum_scatter` over the manual mesh axis `"x"` sums the two local `f32[4]` shards elementwise and
        // scatters the sum, leaving each device with its own `f32[2]` chunk, so the sharded global output is `f32[4]`.
        // The staged collective lowers to a channeled `stablehlo.reduce_scatter` with a sum reduction.
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = device_mesh.logical_mesh().clone();
                let sharding = sharding.clone();
                move |x: XlaArrayTracer| {
                    shard_map(
                        |local_x: XlaArrayTracer| {
                            local_x.parallel_sum_scatter_with_options("x", 0, CollectiveOptions::tiled()).unwrap()
                        },
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .expect("shard_map with parallel_sum_scatter should trace")
                }
            },
            global_input_type,
        )
        .unwrap();

        let mlir_program = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            mlir_program,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<8xf32>) -> tensor<4xf32> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
                      %c = stablehlo.constant dense<4> : tensor<i64>
                      %c_0 = stablehlo.constant dense<2> : tensor<i64>
                      %1 = stablehlo.divide %c, %c_0 : tensor<i64>
                      %2 = "stablehlo.reduce_scatter"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, scatter_dimension = 0 : i64, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %3 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %3 : tensor<f32>
                      }) : (tensor<4xf32>) -> tensor<2xf32>
                      sdy.return %2 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let scale = if device_index == 0 { 1.0f32 } else { 10.0f32 };
                let shard_values = [scale, scale * 2.0, scale * 3.0, scale * 4.0];
                client
                    .buffer(
                        values_to_bytes::<f32>(&shard_values).as_slice(),
                        BufferType::F32,
                        [4u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array = XlaArray::from_addressable_buffers(
            &domain,
            static_sharded_array_type(DataType::F32, &[8], sharding),
            device_mesh,
            input_buffers,
        )
        .unwrap();
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(2)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 2);
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        let execute_arguments =
            XlaArray::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();

        // Shards are [1, 2, 3, 4] and [10, 20, 30, 40], so the elementwise sum is [11, 22, 33, 44]: device 0
        // receives the first chunk [11, 22] and device 1 the second chunk [33, 44].
        assert_eq!(outputs.len(), execution_device_ids.len());
        let expected_values_by_device = [[11.0f32, 22.0], [33.0, 44.0]];
        for (device_index, output) in outputs.into_iter().enumerate() {
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values: [f32; 2] = values_from_bytes::<f32>(output_bytes.as_slice()).try_into().unwrap();
            assert_eq!(values, expected_values_by_device[device_index]);
        }
    }

    #[test]
    fn test_shard_map_untiled_parallel_sum_scatter_lowers_rank_removal() {
        use ryft_core::{CollectiveOptions, ParallelSumScatter};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .expect("failed to create 2-device CPU client");
        let domain = XlaSession::new(&client).domain();
        let client_devices = client.addressable_devices().unwrap();
        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let mesh = device_mesh.logical_mesh().clone();
        let input_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap();
        let output_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let input_sharding = input_sharding.clone();
                let output_sharding = output_sharding.clone();
                move |x: XlaArrayTracer| {
                    shard_map(
                        |local_x: XlaArrayTracer| {
                            local_x
                                .parallel_sum_scatter_with_options(
                                    "x",
                                    1,
                                    CollectiveOptions::default().with_axis_index_groups(vec![vec![1, 0]]),
                                )
                                .unwrap()
                        },
                        x,
                        mesh.clone(),
                        input_sharding.clone(),
                        output_sharding.clone(),
                    )
                    .unwrap()
                }
            },
            ArrayType::new_static(DataType::F32, [6, 2]),
        )
        .unwrap();

        let module = traced.to_mlir_module("main").unwrap();
        // Group order chooses the recipient of each column: device 1 receives column 0, and device 0 column 1.
        // Untiled scatter removes dimension 1 after the native reduce-scatter leaves it with extent one.
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<6x2xf32>) -> tensor<6xf32> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}, {}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<3x2xf32>) {
                      %1 = "stablehlo.reduce_scatter"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[1, 0]]> : tensor<1x2xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %3 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %3 : tensor<f32>
                      }) : (tensor<3x2xf32>) -> tensor<3x1xf32>
                      %2 = stablehlo.reshape %1 : (tensor<3x1xf32>) -> tensor<3xf32>
                      sdy.return %2 : tensor<3xf32>
                    } : (tensor<6x2xf32>) -> tensor<6xf32>
                    return %0 : tensor<6xf32>
                  }
                }
            "#},
        );

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let scale = if device_index == 0 { 1.0f32 } else { 10.0f32 };
                let values = [scale, scale * 2.0, scale * 3.0, scale * 4.0, scale * 5.0, scale * 6.0];
                client
                    .buffer(
                        values_to_bytes(values.as_slice()).as_slice(),
                        BufferType::F32,
                        [3u64, 2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array = XlaArray::from_addressable_buffers(
            &domain,
            static_sharded_array_type(DataType::F32, &[6, 2], input_sharding),
            device_mesh,
            input_buffers,
        )
        .unwrap();
        let executable = client
            .compile(&Program::Mlir { bytecode: module.into_bytes() }, &test_spmd_compilation_options(2))
            .unwrap();
        let execution_device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let execute_arguments =
            XlaArray::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();
        assert_eq!(outputs.len(), 2);
        let expected = [vec![22.0f32, 44.0, 66.0], vec![11.0f32, 33.0, 55.0]];
        for (output, expected) in outputs.into_iter().zip(expected) {
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(values_from_bytes::<f32>(output_bytes.as_slice()), expected);
        }
    }

    #[test]
    fn test_shard_map_parallel_permute_lowers_and_executes_on_cpu() {
        use ryft_core::ParallelPermute;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .expect("failed to create 2-device CPU client");
        let domain = XlaSession::new(&client).domain();
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 2);

        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let sharding =
            Sharding::new(device_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));

        // `parallel_permute` over the manual mesh axis `"x"` with the rotation pairs [(0, 1), (1, 0)] swaps the two
        // local shards without changing their shapes. The staged collective lowers to a channeled
        // `stablehlo.collective_permute` with the axis-local pairs expanded to global device pairs.
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = device_mesh.logical_mesh().clone();
                let sharding = sharding.clone();
                move |x: XlaArrayTracer| {
                    shard_map(
                        |local_x: XlaArrayTracer| local_x.parallel_permute("x", vec![(0, 1), (1, 0)]).unwrap(),
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .expect("shard_map with parallel_permute should trace")
                }
            },
            global_input_type,
        )
        .unwrap();

        let mlir_program = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            mlir_program,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = "stablehlo.collective_permute"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<[[0, 1], [1, 0]]> : tensor<2x2xi64>}> : (tensor<2xf32>) -> tensor<2xf32>
                      sdy.return %1 : tensor<2xf32>
                    } : (tensor<4xf32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let shard_values = [device_index as f32 * 2.0 + 1.0, device_index as f32 * 2.0 + 2.0];
                client
                    .buffer(
                        values_to_bytes::<f32>(&shard_values).as_slice(),
                        BufferType::F32,
                        [2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array = XlaArray::from_addressable_buffers(
            &domain,
            static_sharded_array_type(DataType::F32, &[4], sharding.clone()),
            device_mesh.clone(),
            input_buffers,
        )
        .unwrap();
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(2)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 2);
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        let execute_arguments =
            XlaArray::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();

        // Shards are [1, 2] and [3, 4]; the rotation swaps them, so device 0 receives [3, 4] and device 1 [1, 2].
        assert_eq!(outputs.len(), execution_device_ids.len());
        let expected_values_by_device = [[3.0f32, 4.0], [1.0, 2.0]];
        for (device_index, output) in outputs.into_iter().enumerate() {
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values: [f32; 2] = values_from_bytes::<f32>(output_bytes.as_slice()).try_into().unwrap();
            assert_eq!(values, expected_values_by_device[device_index]);
        }

        // Partial and empty permutations give every untargeted participant a zero-filled shard.
        for (pairs, expected_values_by_device) in
            [(vec![(0, 1)], [[0.0f32, 0.0], [1.0, 2.0]]), (Vec::new(), [[0.0f32, 0.0], [0.0, 0.0]])]
        {
            let mesh = device_mesh.logical_mesh().clone();
            let sharding = sharding.clone();
            let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
                move |input: XlaArrayTracer| {
                    shard_map(
                        |local_input: XlaArrayTracer| local_input.parallel_permute("x", pairs.clone()).unwrap(),
                        input,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .unwrap()
                },
                ArrayType::new_static(DataType::F32, [4]),
            )
            .unwrap();
            let program = Program::Mlir { bytecode: traced.to_mlir_module("main").unwrap().into_bytes() };
            let executable = client.compile(&program, &test_spmd_compilation_options(2)).unwrap();
            assert_eq!(
                executable
                    .addressable_devices()
                    .unwrap()
                    .iter()
                    .map(|device| device.id().unwrap())
                    .collect::<Vec<_>>(),
                execution_device_ids,
            );
            let outputs = executable
                .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
                .unwrap()
                .block_until_ready()
                .unwrap();
            assert_eq!(outputs.len(), expected_values_by_device.len());
            for (output, expected) in outputs.into_iter().zip(expected_values_by_device) {
                assert_eq!(output.outputs.len(), 1);
                let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
                assert_eq!(values_from_bytes::<f32>(output_bytes.as_slice()), expected);
            }
        }
    }

    #[test]
    fn test_shard_map_parallel_permute_rejects_axis_size_mismatch() {
        use ryft_core::ParallelPermuteOperation;

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        // Both operations are valid in isolation, but neither records the size of the enclosing manual axis.
        // The larger size also exercises the pair that previously indexed beyond its replica group.
        for (axis_size, pairs) in [(1, vec![(0, 0)]), (3, vec![(2, 0)])] {
            let mesh = mesh.clone();
            let sharding = sharding.clone();
            let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
                move |input: XlaArrayTracer| {
                    shard_map(
                        |local_input: XlaArrayTracer| {
                            local_input
                                .domain()
                                .bind(
                                    ParallelPermuteOperation::new("x".to_string(), axis_size, pairs.clone()),
                                    Vec::new(),
                                    &[local_input],
                                )
                                .unwrap()
                                .remove(0)
                        },
                        input,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .unwrap()
                },
                ArrayType::new_static(DataType::F32, [4]),
            )
            .unwrap();
            assert!(matches!(
                traced.to_mlir_module("main"),
                Err(TraceError::Lowering(error))
                    if error.to_string() == format!(
                        "encountered malformed program: collective over axis `x` records size {axis_size}, \
                         but the enclosing mesh axis has size 2",
                    ),
            ));
        }
    }

    #[test]
    fn test_shard_map_parallel_all_to_all_lowers_and_executes_on_cpu() {
        use ryft_core::{CollectiveOptions, ParallelAllToAll};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .expect("failed to create 2-device CPU client");
        let domain = XlaSession::new(&client).domain();
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 2);

        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let sharding =
            Sharding::new(device_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));

        // `parallel_all_to_all` over the manual mesh axis `"x"` with split and concat both at axis 0 keeps the local
        // `f32[4]` shape: each device splits its shard into two chunks, keeps its own chunk, and receives the peer's
        // matching chunk. The staged collective lowers to a channeled `stablehlo.all_to_all`.
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = device_mesh.logical_mesh().clone();
                let sharding = sharding.clone();
                move |x: XlaArrayTracer| {
                    shard_map(
                        |local_x: XlaArrayTracer| {
                            local_x.parallel_all_to_all_with_options("x", 0, 0, CollectiveOptions::tiled()).unwrap()
                        },
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .expect("shard_map with parallel_all_to_all should trace")
                }
            },
            global_input_type,
        )
        .unwrap();

        let mlir_program = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            mlir_program,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
                      %1 = "stablehlo.all_to_all"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, split_count = 2 : i64, split_dimension = 0 : i64}> {use_global_device_ids} : (tensor<4xf32>) -> tensor<4xf32>
                      sdy.return %1 : tensor<4xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#},
        );

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let base = device_index as f32 * 4.0;
                let shard_values = [base + 1.0, base + 2.0, base + 3.0, base + 4.0];
                client
                    .buffer(
                        values_to_bytes::<f32>(&shard_values).as_slice(),
                        BufferType::F32,
                        [4u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array = XlaArray::from_addressable_buffers(
            &domain,
            static_sharded_array_type(DataType::F32, &[8], sharding),
            device_mesh,
            input_buffers,
        )
        .unwrap();
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(2)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 2);
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        let execute_arguments =
            XlaArray::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();

        // Shards are [1, 2, 3, 4] and [5, 6, 7, 8]: the exchange leaves device 0 with the two first halves
        // [1, 2, 5, 6] and device 1 with the two second halves [3, 4, 7, 8].
        assert_eq!(outputs.len(), execution_device_ids.len());
        let expected_values_by_device = [[1.0f32, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]];
        for (device_index, output) in outputs.into_iter().enumerate() {
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values: [f32; 4] = values_from_bytes::<f32>(output_bytes.as_slice()).try_into().unwrap();
            assert_eq!(values, expected_values_by_device[device_index]);
        }
    }

    #[test]
    fn test_shard_map_untiled_parallel_all_to_all_lowers_rank_exchange() {
        use ryft_core::ParallelAllToAll;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .expect("failed to create 2-device CPU client");
        let domain = XlaSession::new(&client).domain();
        let client_devices = client.addressable_devices().unwrap();
        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let mesh = device_mesh.logical_mesh().clone();
        let input_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
                .unwrap();
        let output_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap();
        let traced: TracedXlaProgram<ArrayType, (ArrayType, ArrayType)> = trace(
            {
                let input_sharding = input_sharding.clone();
                let output_sharding = output_sharding.clone();
                move |x: XlaArrayTracer| {
                    shard_map(
                        |local_x: XlaArrayTracer| {
                            // Removing split axis 0 puts the gathered participant axis at output axis 1.
                            // The inverse exchanges axis 1 back into output axis 0.
                            let exchanged = local_x.parallel_all_to_all("x", 0, 1).unwrap();
                            let recovered = exchanged.parallel_all_to_all("x", 1, 0).unwrap();
                            (exchanged, recovered)
                        },
                        x,
                        mesh.clone(),
                        input_sharding.clone(),
                        (output_sharding.clone(), input_sharding.clone()),
                    )
                    .unwrap()
                }
            },
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(6)])),
        )
        .unwrap();

        let module = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<2x6xf32>) -> (tensor<6x2xf32>, tensor<2x6xf32>) {
                    %0:2 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {"x"}]>] out_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"x"}]>] manual_axes={"x"} (%arg1: tensor<2x3xf32>) {
                      %1 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<2x3xf32>) -> tensor<2x3x1xf32>
                      %2 = "stablehlo.all_to_all"(%1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 2 : i64, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, split_count = 2 : i64, split_dimension = 0 : i64}> {use_global_device_ids} : (tensor<2x3x1xf32>) -> tensor<1x3x2xf32>
                      %3 = stablehlo.reshape %2 : (tensor<1x3x2xf32>) -> tensor<3x2xf32>
                      %4 = stablehlo.broadcast_in_dim %3, dims = [1, 2] : (tensor<3x2xf32>) -> tensor<1x3x2xf32>
                      %5 = "stablehlo.all_to_all"(%4) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, split_count = 2 : i64, split_dimension = 2 : i64}> {use_global_device_ids} : (tensor<1x3x2xf32>) -> tensor<2x3x1xf32>
                      %6 = stablehlo.reshape %5 : (tensor<2x3x1xf32>) -> tensor<2x3xf32>
                      sdy.return %3, %6 : tensor<3x2xf32>, tensor<2x3xf32>
                    } : (tensor<2x6xf32>) -> (tensor<6x2xf32>, tensor<2x6xf32>)
                    return %0#0, %0#1 : tensor<6x2xf32>, tensor<2x6xf32>
                  }
                }
            "#},
        );

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let offset = device_index as f32 * 9.0;
                let values = [offset + 1.0, offset + 2.0, offset + 3.0, offset + 4.0, offset + 5.0, offset + 6.0];
                client
                    .buffer(
                        values_to_bytes(values.as_slice()).as_slice(),
                        BufferType::F32,
                        [2u64, 3u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array = XlaArray::from_addressable_buffers(
            &domain,
            static_sharded_array_type(DataType::F32, &[2, 6], input_sharding),
            device_mesh,
            input_buffers,
        )
        .unwrap();
        let executable = client
            .compile(&Program::Mlir { bytecode: module.into_bytes() }, &test_spmd_compilation_options(2))
            .unwrap();
        let execution_device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let execute_arguments =
            XlaArray::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();
        let expected_exchange = [vec![1.0f32, 10.0, 2.0, 11.0, 3.0, 12.0], vec![4.0f32, 13.0, 5.0, 14.0, 6.0, 15.0]];
        let expected_recovery = [vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![10.0f32, 11.0, 12.0, 13.0, 14.0, 15.0]];
        assert_eq!(outputs.len(), 2);
        for (device_index, output) in outputs.into_iter().enumerate() {
            assert_eq!(output.outputs.len(), 2);
            assert_eq!(output.outputs[0].dimensions().unwrap(), vec![3, 2]);
            assert_eq!(output.outputs[1].dimensions().unwrap(), vec![2, 3]);
            let exchanged_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let recovered_bytes = output.outputs[1].copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(values_from_bytes::<f32>(exchanged_bytes.as_slice()), expected_exchange[device_index]);
            assert_eq!(values_from_bytes::<f32>(recovered_bytes.as_slice()), expected_recovery[device_index]);
        }
    }
}
