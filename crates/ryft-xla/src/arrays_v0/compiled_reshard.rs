use std::collections::HashMap;

use ryft_core::sharding::{DeviceMesh, MeshAxisType, Sharding, ShardingDimension};
use ryft_core::types::ArrayType;
use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions};

use crate::experimental::domains::XlaDomain;
use crate::experimental::shard_map::{ShardMapTracer, TracedXlaProgram, trace, with_sharding_constraint};
use crate::{Array, ArrayError, CompilationContext, Error as XlaError};

/// Tries the compiled-XLA resharding path for [`Array::to_placement`](crate::Array::to_placement).
///
/// On success, returns `Some(array)` carrying the resharded array. On a soft decline (any guard
/// fails, or the trace/lower/compile/execute pipeline raises an error), returns `Ok(None)` so
/// the caller can fall through to the host-materialization slow path. The compiled path is the
/// `ryft` analogue of how JAX's `jax.device_put(arr, new_sharding)` lowers a reshard of a
/// committed `jax.Array`: an `identity(x) = x` program annotated with input and output
/// `with_sharding_constraint` ops, lowered to StableHLO + Shardy, compiled, and executed.
///
/// Two cases are supported:
///
///   - **Same-mesh reshard** (source and destination [`DeviceMesh`]es are equal): one compiled
///     identity program with both input and output sharding constraints.
///   - **Replicated cross-mesh reshard** (source is fully replicated, destination is a different
///     [`DeviceMesh`]): source buffers are first physically placed on every destination device via
///     intra-host D2D copies, then the same-mesh path resards the resulting replicated
///     intermediate to `dst_sharding`. Cross-host destinations (devices on other processes) are
///     not yet supported and fall through to the host path.
///
/// Other guards:
///
///   - All [`MeshAxisType`]s of the destination mesh must be [`MeshAxisType::Auto`], so the SPMD
///     partitioner can plan the reshard collectives.
///   - Sharded cross-mesh reshards are not yet supported and fall through to the host path.
///
/// # Parameters
///
///   - `source`: [`Array`] to reshard.
///   - `context`: [`CompilationContext`] used both as the PJRT client wrapper and as the
///     executable cache.
///   - `dst_mesh`: Destination [`DeviceMesh`].
///   - `dst_sharding`: Destination [`Sharding`].
pub(crate) fn try_compiled_reshard<'o>(
    source: &Array<'o>,
    context: &CompilationContext<'o>,
    dst_mesh: &DeviceMesh,
    dst_sharding: &Sharding,
) -> Result<Option<Array<'o>>, ArrayError> {
    if dst_mesh.logical_mesh().axes().iter().any(|axis| axis.r#type() != MeshAxisType::Auto) {
        return Ok(None);
    }

    let src_mesh = source.mesh();
    if &src_mesh == dst_mesh {
        try_same_mesh(source, context, dst_mesh, dst_sharding)
    } else if is_fully_replicated(source.sharding()) {
        try_replicated_cross_mesh(source, context, dst_mesh, dst_sharding)
    } else {
        // Sharded cross-mesh reshards require an additional compiled all-gather on the source
        // mesh before the broadcast step. That is deferred until a use case exercises it.
        Ok(None)
    }
}

/// Builds the [`CompilationOptions`] required for compiling SPMD-partitioned programs against a
/// `partition_count`-sized mesh using the Shardy partitioner.
fn spmd_compilation_options(partition_count: usize) -> CompilationOptions {
    CompilationOptions {
        executable_build_options: Some(ExecutableCompilationOptions {
            device_ordinal: -1,
            replica_count: 1,
            partition_count: partition_count as i64,
            use_spmd_partitioning: true,
            use_shardy_partitioner: true,
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn is_fully_replicated(sharding: &Sharding) -> bool {
    sharding.dimensions().iter().all(|dim| matches!(dim, ShardingDimension::Replicated))
        && sharding.unreduced_axes().is_empty()
        && sharding.reduced_manual_axes().is_empty()
        && sharding.varying_manual_axes().is_empty()
}

/// Runs the compiled identity-with-sharding-constraints program against `dst_mesh`. Assumes the
/// source array already lives on `dst_mesh`.
fn try_same_mesh<'o>(
    source: &Array<'o>,
    context: &CompilationContext<'o>,
    dst_mesh: &DeviceMesh,
    dst_sharding: &Sharding,
) -> Result<Option<Array<'o>>, ArrayError> {
    let element_type = source.data_type();
    let shape = source.shape();
    let src_sharding = source.sharding().clone();
    let dst_sharding = dst_sharding.clone();

    // Build the bare input ArrayType (no sharding). The traced body materializes the source
    // sharding as a leading `sdy.sharding_constraint` op, which is what the SPMD partitioner
    // reads as the input layout. The destination sharding is materialized as a second
    // `sdy.sharding_constraint` op on the returned value.
    let bare_input_type = ArrayType::new(element_type, shape.clone().into(), None, None).map_err(XlaError::from)?;

    let traced_result: Result<TracedXlaProgram<ArrayType, ArrayType>, _> = trace(
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
    );
    let traced = match traced_result {
        Ok(traced) => traced,
        Err(_) => return Ok(None),
    };

    // SPMD requires explicit `replica_count` and `partition_count` plus the Shardy partitioner
    // flag. `partition_count` is the number of devices the compiled program will run across.
    let compilation_options = spmd_compilation_options(dst_mesh.devices().len());
    let domain = XlaDomain::with_compilation_options(context.client(), dst_mesh.clone(), compilation_options);
    // PJRT requires the entry function to be named `main`.
    let mlir = match domain.lower(&traced, "main") {
        Ok(mlir) => mlir,
        Err(_) => return Ok(None),
    };
    let executable = match context.compile(&mlir, domain.compilation_options()) {
        Ok(executable) => executable,
        Err(_) => return Ok(None),
    };

    let output_type = ArrayType::new(element_type, shape.into(), None, Some(dst_sharding)).map_err(XlaError::from)?;
    let outputs = match domain.execute(&executable, vec![source.clone()], &[output_type]) {
        Ok(outputs) => outputs,
        Err(_) => return Ok(None),
    };

    Ok(outputs.into_iter().next())
}

/// Broadcasts a fully-replicated `source` onto every device in `dst_mesh` via intra-host D2D
/// copies, then defers to [`try_same_mesh`] to perform the on-`dst_mesh` reshard. Returns
/// `Ok(None)` if any destination device is non-addressable on the current process or any PJRT
/// transfer raises an error.
fn try_replicated_cross_mesh<'o>(
    source: &Array<'o>,
    context: &CompilationContext<'o>,
    dst_mesh: &DeviceMesh,
    dst_sharding: &Sharding,
) -> Result<Option<Array<'o>>, ArrayError> {
    let client = context.client();
    let addressable_devices = match client.addressable_devices() {
        Ok(devices) => devices,
        Err(_) => return Ok(None),
    };
    let mut dst_device_by_id = HashMap::with_capacity(addressable_devices.len());
    for device in addressable_devices {
        let id = match device.id() {
            Ok(id) => id,
            Err(_) => return Ok(None),
        };
        dst_device_by_id.insert(id, device);
    }

    // A fully-replicated source has the entire array on every src_mesh device. Pick any
    // addressable source buffer to use as the broadcast source for dst-only devices.
    let any_source_buffer_arc = source.addressable_shards().find_map(|shard| shard.buffer().cloned());
    let any_source_buffer = match any_source_buffer_arc {
        Some(arc) => arc,
        None => return Ok(None),
    };

    let mut buffers = Vec::with_capacity(dst_mesh.devices().len());
    for dst_device in dst_mesh.devices() {
        let dst_device_id = dst_device.id();
        let pjrt_device = match dst_device_by_id.get(&dst_device_id) {
            Some(device) => device,
            None => return Ok(None),
        };
        // Place a fresh PJRT buffer on every destination device. `copy_to_device` on the source's
        // own device returns a fresh independent buffer, so we get a clean ownership chain in
        // both the reuse and broadcast cases.
        let buffer = match any_source_buffer.copy_to_device(pjrt_device.clone()) {
            Ok(buffer) => buffer,
            Err(_) => return Ok(None),
        };
        buffers.push(buffer);
    }

    let shape = source.shape();
    let element_type = source.data_type();
    let replicated_on_dst = Sharding::replicated(dst_mesh.logical_mesh().clone(), shape.rank());
    let intermediate_type =
        ArrayType::new(element_type, shape.into(), None, Some(replicated_on_dst)).map_err(XlaError::from)?;
    let intermediate = match Array::from_addressable_buffers(intermediate_type, dst_mesh.clone(), buffers) {
        Ok(array) => array,
        Err(_) => return Ok(None),
    };

    try_same_mesh(&intermediate, context, dst_mesh, dst_sharding)
}
