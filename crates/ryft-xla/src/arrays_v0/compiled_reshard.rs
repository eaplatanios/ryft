use std::collections::HashMap;

use ryft_core::sharding::{DeviceMesh, MeshAxisType, Sharding, ShardingDimension};
use ryft_core::types::ArrayType;
use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions};

use crate::experimental::domains::XlaDomain;
use crate::experimental::shard_map::{ShardMapTracer, TracedXlaProgram, trace, with_sharding_constraint};
use crate::{Array, ArrayError, CompilationContext, Error as XlaError};

/// Performs the compiled-XLA resharding path for [`Array::to_placement`](crate::Array::to_placement).
///
/// This is the `ryft` analogue of how JAX's `jax.device_put(arr, new_sharding)` lowers a reshard
/// of a committed `jax.Array`: an `identity(x) = x` program annotated with input and output
/// `with_sharding_constraint` ops, lowered to StableHLO + Shardy, compiled, and executed entirely
/// on device.
///
/// Three cases are supported:
///
///   - **Same-mesh reshard** (source and destination [`DeviceMesh`]es are equal): one compiled
///     identity program with both input and output sharding constraints.
///   - **Replicated cross-mesh reshard** (source is fully replicated, destination is a different
///     [`DeviceMesh`]): source buffers are first physically placed on every destination device via
///     intra-host D2D copies, then the same-mesh path reshards the resulting replicated
///     intermediate to `dst_sharding`.
///   - **Sharded cross-mesh reshard** (source is sharded, destination is a different
///     [`DeviceMesh`]): a compiled SPMD all-gather first replicates the source on `src_mesh`,
///     then the replicated cross-mesh path broadcasts the result to `dst_mesh` and applies the
///     final `dst_sharding`.
///
/// Unsupported requests return typed [`ArrayError`] variants:
///
///   - [`ArrayError::UnsupportedMeshAxisType`] when the destination mesh has any non-`Auto` axis.
///   - [`ArrayError::NonAddressableDestinationDevice`] when a destination device is on a
///     different process than the wrapped [`Client`](ryft_pjrt::Client).
///   - [`ArrayError::MissingAddressableShardForMove`] when the source has no addressable shard.
///
/// PJRT compile or execute failures propagate as `ArrayError::Error(...)` wrapping the underlying
/// [`crate::Error`].
///
/// # Parameters
///
///   - `source`: [`Array`] to reshard.
///   - `context`: [`CompilationContext`] used both as the PJRT client wrapper and as the
///     executable cache.
///   - `dst_mesh`: Destination [`DeviceMesh`].
///   - `dst_sharding`: Destination [`Sharding`].
pub(crate) fn reshard<'o>(
    source: &Array<'o>,
    context: &CompilationContext<'o>,
    dst_mesh: &DeviceMesh,
    dst_sharding: &Sharding,
) -> Result<Array<'o>, ArrayError> {
    reshard_with_donation(source, context, dst_mesh, dst_sharding, false)
}

/// Same as [`reshard`] but allows the caller to opt into donating the source array's input
/// buffers to the compiled SPMD program. With `donate=true`, PJRT may reuse the source's
/// device-side memory for output buffers; the caller must guarantee the source is not read again
/// after this call. [`Array::to_device`](crate::Array::to_device) sets `donate=true` because it
/// consumes its `self`.
pub(crate) fn reshard_with_donation<'o>(
    source: &Array<'o>,
    context: &CompilationContext<'o>,
    dst_mesh: &DeviceMesh,
    dst_sharding: &Sharding,
    donate: bool,
) -> Result<Array<'o>, ArrayError> {
    if let Some(axis) = dst_mesh.logical_mesh().axes().iter().find(|axis| axis.r#type() != MeshAxisType::Auto) {
        return Err(ArrayError::UnsupportedMeshAxisType {
            axis_name: axis.name().to_string(),
            axis_type: axis.r#type(),
        });
    }

    // The compiled path requires every source shard to be addressable from the current process.
    // Remote shards on other processes are not yet supported. Surface the first non-addressable
    // source shard as a concrete error rather than letting `XlaDomain::execute` fail downstream.
    for shard in source.shards() {
        if shard.buffer().is_none() {
            return Err(ArrayError::MissingAddressableShardForMove {
                shard_index: shard.index(),
                device_id: shard.device().id(),
            });
        }
    }

    let src_mesh = source.mesh();
    if &src_mesh == dst_mesh {
        try_same_mesh(source, context, dst_mesh, dst_sharding, donate)
    } else if is_fully_replicated(source.sharding()) {
        try_replicated_cross_mesh(source, context, dst_mesh, dst_sharding)
    } else {
        try_sharded_cross_mesh(source, context, &src_mesh, dst_mesh, dst_sharding)
    }
}

/// Overlays the SPMD partitioning fields required by the compiled reshard onto a base
/// [`CompilationOptions`] template (typically [`CompilationContext::base_options`]). The base
/// template is preserved field-by-field; only `replica_count`, `partition_count`, and the SPMD /
/// Shardy partitioner flags are overwritten with mesh-derived values.
fn spmd_compilation_options(base: &CompilationOptions, partition_count: usize) -> CompilationOptions {
    let mut options = base.clone();
    let exec_options = options.executable_build_options.get_or_insert_with(ExecutableCompilationOptions::default);
    if exec_options.device_ordinal == 0 {
        // `0` is the protobuf default but PJRT expects `-1` to mean "use the default device". Only
        // overwrite when the base template hasn't been customized.
        exec_options.device_ordinal = -1;
    }
    exec_options.replica_count = 1;
    exec_options.partition_count = partition_count as i64;
    exec_options.use_spmd_partitioning = true;
    exec_options.use_shardy_partitioner = true;
    options
}

fn is_fully_replicated(sharding: &Sharding) -> bool {
    sharding.dimensions().iter().all(|dim| matches!(dim, ShardingDimension::Replicated))
        && sharding.unreduced_axes().is_empty()
        && sharding.reduced_manual_axes().is_empty()
        && sharding.varying_manual_axes().is_empty()
}

/// Runs the compiled identity-with-sharding-constraints program against `dst_mesh`. Assumes the
/// source array already lives on `dst_mesh`. If `donate` is true, the source's input buffers are
/// marked donatable so PJRT can reuse their memory for output buffers; the caller must guarantee
/// that the source's buffer is not read after this call.
fn try_same_mesh<'o>(
    source: &Array<'o>,
    context: &CompilationContext<'o>,
    dst_mesh: &DeviceMesh,
    dst_sharding: &Sharding,
    donate: bool,
) -> Result<Array<'o>, ArrayError> {
    let element_type = source.data_type();
    let shape = source.shape();
    let src_sharding = source.sharding().clone();
    let dst_sharding = dst_sharding.clone();

    // Build the bare input ArrayType (no sharding). The traced body materializes the source
    // sharding as a leading `sdy.sharding_constraint` op, which is what the SPMD partitioner
    // reads as the input layout. The destination sharding is materialized as a second
    // `sdy.sharding_constraint` op on the returned value.
    let bare_input_type = ArrayType::new(element_type, shape.clone().into(), None, None).map_err(XlaError::from)?;

    let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
        {
            let src_sharding = src_sharding.clone();
            let dst_sharding = dst_sharding.clone();
            move |x: ShardMapTracer| {
                let constrained_src = with_sharding_constraint(x, src_sharding.clone())
                    .expect("source sharding has the same rank as the source array");
                with_sharding_constraint(constrained_src, dst_sharding.clone())
                    .expect("destination sharding has the same rank as the source array")
            }
        },
        bare_input_type,
    )
    .map_err(|error| ArrayError::CompiledReshardInternalError { message: format!("trace failed: {error}") })?;

    // SPMD requires explicit `replica_count` and `partition_count` plus the Shardy partitioner
    // flag. `partition_count` is the number of devices the compiled program will run across.
    let compilation_options = spmd_compilation_options(context.base_options(), dst_mesh.devices().len());
    let domain = XlaDomain::with_compilation_options(context.client(), dst_mesh.clone(), compilation_options);
    // PJRT requires the entry function to be named `main`.
    let mlir = domain
        .lower(&traced, "main")
        .map_err(|error| ArrayError::CompiledReshardInternalError { message: format!("lower failed: {error}") })?;
    let executable = context.compile(&mlir, domain.compilation_options()).map_err(XlaError::from)?;

    let output_type = ArrayType::new(element_type, shape.into(), None, Some(dst_sharding)).map_err(XlaError::from)?;
    let outputs = domain
        .execute_with_donation(&executable, vec![source.clone()], &[donate], &[output_type])
        .map_err(|error| ArrayError::CompiledReshardInternalError { message: format!("execute failed: {error}") })?;

    outputs.into_iter().next().ok_or_else(|| ArrayError::CompiledReshardInternalError {
        message: "compiled reshard produced no outputs".to_string(),
    })
}

/// Broadcasts a fully-replicated `source` onto every device in `dst_mesh` via intra-host D2D
/// copies, then defers to [`try_same_mesh`] to perform the on-`dst_mesh` reshard.
fn try_replicated_cross_mesh<'o>(
    source: &Array<'o>,
    context: &CompilationContext<'o>,
    dst_mesh: &DeviceMesh,
    dst_sharding: &Sharding,
) -> Result<Array<'o>, ArrayError> {
    let client = context.client();
    let addressable_devices = client.addressable_devices().map_err(XlaError::from)?;
    let mut dst_device_by_id = HashMap::with_capacity(addressable_devices.len());
    for device in addressable_devices {
        let id = device.id().map_err(XlaError::from)?;
        dst_device_by_id.insert(id, device);
    }

    // A fully-replicated source has the entire array on every src_mesh device. Pick any
    // addressable source buffer to use as the broadcast source for dst-only devices.
    let any_source_buffer = source.addressable_shards().find_map(|shard| shard.buffer().cloned()).ok_or_else(|| {
        ArrayError::MissingAddressableShardForMove {
            shard_index: 0,
            device_id: source.shards().first().map(|shard| shard.device().id()).unwrap_or(0),
        }
    })?;

    let mut buffers = Vec::with_capacity(dst_mesh.devices().len());
    for dst_device in dst_mesh.devices() {
        let dst_device_id = dst_device.id();
        let pjrt_device = dst_device_by_id.get(&dst_device_id).ok_or(ArrayError::NonAddressableDestinationDevice {
            device_id: dst_device_id,
            process_index: dst_device.process_index(),
        })?;
        // Place a fresh PJRT buffer on every destination device. `copy_to_device` on the source's
        // own device returns a fresh independent buffer, so we get a clean ownership chain in
        // both the reuse and broadcast cases.
        let buffer = any_source_buffer.copy_to_device(pjrt_device.clone()).map_err(XlaError::from)?;
        buffers.push(buffer);
    }

    let shape = source.shape();
    let element_type = source.data_type();
    let replicated_on_dst = Sharding::replicated(dst_mesh.logical_mesh().clone(), shape.rank());
    let intermediate_type =
        ArrayType::new(element_type, shape.into(), None, Some(replicated_on_dst)).map_err(XlaError::from)?;
    let intermediate = Array::from_addressable_buffers(intermediate_type, dst_mesh.clone(), buffers)?;

    // The intermediate is owned exclusively by this function and is never observed by callers.
    // Donating its buffers lets PJRT reuse their memory for the output of the final SPMD reshard.
    try_same_mesh(&intermediate, context, dst_mesh, dst_sharding, true)
}

/// Reshards a sharded source array onto a different destination mesh.
///
/// Composes two compiled SPMD programs:
///
///   1. An all-gather on `src_mesh` that produces a fully-replicated intermediate (every device
///      in `src_mesh` ends up with the full logical array).
///   2. [`try_replicated_cross_mesh`] then broadcasts that intermediate to `dst_mesh` and
///      compiles the final reshard from "replicated on `dst_mesh`" to `dst_sharding`.
///
/// Both compiled programs go through `context`'s executable cache, so repeated calls with the
/// same input/output shardings pay the compile cost once.
fn try_sharded_cross_mesh<'o>(
    source: &Array<'o>,
    context: &CompilationContext<'o>,
    src_mesh: &DeviceMesh,
    dst_mesh: &DeviceMesh,
    dst_sharding: &Sharding,
) -> Result<Array<'o>, ArrayError> {
    let shape = source.shape();
    let replicated_on_src = Sharding::replicated(src_mesh.logical_mesh().clone(), shape.rank());
    // Source is owned externally, so we do not donate it during the all-gather. The gathered
    // intermediate is owned by this function and is donated to the subsequent broadcast/reshard
    // step inside `try_replicated_cross_mesh`.
    let gathered = try_same_mesh(source, context, src_mesh, &replicated_on_src, false)?;
    try_replicated_cross_mesh(&gathered, context, dst_mesh, dst_sharding)
}
