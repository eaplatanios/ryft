use std::collections::HashMap;

use ryft_core::compilation::CompilationDomain;
use ryft_core::contexts::Domain;
use ryft_core::sharding::{DeviceMesh, MeshAxisType, Sharding, ShardingDimension};
use ryft_core::types::ArrayType;
use ryft_pjrt::extensions::cross_host_transfers::{CrossHostTransferKey, GlobalDeviceId};
use ryft_pjrt::{Buffer, DeviceId};

use crate::arrays_v0::transfers::{cross_host_global_device_id, exact_shard_transfer_key};
use crate::experimental::domains::{XlaDomain, XlaDomainError, XlaOptions, XlaTracer};
use crate::{Array, ArrayError, Error as XlaError, ToPjrt};

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
///   - `engine`: [`XlaDomain`] providing the PJRT client and the compile-program cache.
///   - `dst_mesh`: Destination [`DeviceMesh`].
///   - `dst_sharding`: Destination [`Sharding`].
pub(crate) fn reshard<'o>(
    source: &Array<'o>,
    engine: &XlaDomain<'o>,
    dst_mesh: &DeviceMesh,
    dst_sharding: &Sharding,
) -> Result<Array<'o>, ArrayError> {
    reshard_with_donation(source, engine, dst_mesh, dst_sharding, false)
}

/// Same as [`reshard`] but allows the caller to opt into donating the source array's input
/// buffers to the compiled SPMD program. With `donate=true`, PJRT may reuse the source's
/// device-side memory for output buffers; the caller must guarantee the source is not read again
/// after this call. [`Array::into_placement`](crate::Array::into_placement) sets `donate=true`
/// because it consumes its `self`.
pub(crate) fn reshard_with_donation<'o>(
    source: &Array<'o>,
    engine: &XlaDomain<'o>,
    dst_mesh: &DeviceMesh,
    dst_sharding: &Sharding,
    donate: bool,
) -> Result<Array<'o>, ArrayError> {
    // `Manual` axes are managed explicitly by the user (e.g. inside `shard_map`) and cannot be
    // planned by the SPMD partitioner from the top level. `Auto` and `Explicit` axes both go
    // through SPMD partitioning, so allow both.
    if let Some(axis) = dst_mesh.logical_mesh().axes().iter().find(|axis| axis.r#type() == MeshAxisType::Manual) {
        return Err(ArrayError::UnsupportedMeshAxisType {
            axis_name: axis.name().to_string(),
            axis_type: axis.r#type(),
        });
    }

    // Source shards on the current process must carry a local buffer (a missing one indicates a
    // real misconfiguration). Shards on other processes are normal — they're served by the
    // cross-host transfers extension when the compiled path needs them.
    let client_process_index = engine.client().process_index().map_err(XlaError::from)?;
    for shard in source.shards() {
        if shard.device().process_index() == client_process_index && shard.buffer().is_none() {
            return Err(ArrayError::MissingAddressableShardForMove {
                shard_index: shard.index(),
                device_id: shard.device().id(),
            });
        }
    }

    let src_mesh = source.mesh();
    if &src_mesh == dst_mesh {
        try_same_mesh(source, engine, dst_mesh, dst_sharding, donate)
    } else if is_fully_replicated(source.sharding()) {
        try_replicated_cross_mesh(source, engine, dst_mesh, dst_sharding)
    } else {
        try_sharded_cross_mesh(source, engine, &src_mesh, dst_mesh, dst_sharding)
    }
}

fn is_fully_replicated(sharding: &Sharding) -> bool {
    sharding.dimensions().iter().all(|dim| matches!(dim, ShardingDimension::Replicated))
        && sharding.unreduced_axes().is_empty()
        && sharding.reduced_axes().is_empty()
        && sharding.varying_manual_axes().is_empty()
}

/// Runs the compiled identity-with-sharding-constraints program against `dst_mesh`. Assumes the
/// source array already lives on `dst_mesh`. If `donate` is true, the source's input buffers are
/// marked donatable so PJRT can reuse their memory for output buffers; the caller must guarantee
/// that the source's buffer is not read after this call.
///
/// The compilation cache is keyed by the complete lowered computation, including its input/output sharding
/// annotations and PJRT options.
fn try_same_mesh<'o>(
    source: &Array<'o>,
    engine: &XlaDomain<'o>,
    dst_mesh: &DeviceMesh,
    dst_sharding: &Sharding,
    donate: bool,
) -> Result<Array<'o>, ArrayError> {
    let element_type = source.data_type();
    let shape = source.shape();
    let src_sharding = source.sharding().clone();
    let dst_sharding = dst_sharding.clone();

    // Build the bare input type for tracing (no sharding); the in/out shardings are attached as
    // `sdy.sharding` attributes via `XlaOptions::{in_shardings, out_shardings}`.
    let bare_input_type = ArrayType::new(element_type, shape.clone().into());

    let xla_options = XlaOptions {
        mesh: dst_mesh.clone(),
        in_shardings: Some(vec![src_sharding.clone()]),
        out_shardings: Some(vec![dst_sharding.clone()]),
        donation_flags: vec![donate],
    };

    let (_output_types_tree, program): (ArrayType, _) =
        XlaDomain::<'o>::trace(|x: XlaTracer<'o>| Ok(x), bare_input_type.clone()).map_err(|error| {
            ArrayError::CompiledReshardInternalError { message: format!("tracing failed: {error}") }
        })?;
    let program = program.into_flat_program();
    let lowered = CompilationDomain::lower(engine, &program, 0, &xla_options).map_err(|error| match error {
        XlaDomainError::Array(array_error) => array_error,
        other => ArrayError::CompiledReshardInternalError { message: format!("lowering failed: {other}") },
    })?;
    let cache_key =
        CompilationDomain::compilation_key(engine, &lowered, &xla_options).map_err(|error| match error {
            XlaDomainError::Array(array_error) => array_error,
            other => {
                ArrayError::CompiledReshardInternalError { message: format!("cache-key construction failed: {other}") }
            }
        })?;

    let cache = engine.cache().expect("XlaDomain always exposes a compile cache");
    let compiled = cache
        .get_or_compile(engine, cache_key, || CompilationDomain::compile(engine, &lowered, &xla_options))
        .map_err(|error| match error {
            XlaDomainError::Array(array_error) => array_error,
            other => ArrayError::CompiledReshardInternalError { message: format!("{other}") },
        })?;

    let inputs = vec![source.clone()];
    let outputs = CompilationDomain::execute(engine, &compiled, inputs).map_err(|error| match error {
        XlaDomainError::Array(array_error) => array_error,
        other => ArrayError::CompiledReshardInternalError { message: format!("execute failed: {other}") },
    })?;

    outputs.into_iter().next().ok_or_else(|| ArrayError::CompiledReshardInternalError {
        message: "compiled reshard produced no outputs".to_string(),
    })
}

/// Broadcasts a fully-replicated `source` onto every device in `dst_mesh` via intra-host D2D
/// copies, then defers to [`try_same_mesh`] to perform the on-`dst_mesh` reshard.
fn try_replicated_cross_mesh<'o>(
    source: &Array<'o>,
    engine: &XlaDomain<'o>,
    dst_mesh: &DeviceMesh,
    dst_sharding: &Sharding,
) -> Result<Array<'o>, ArrayError> {
    let client = engine.client();
    let client_process_index = client.process_index().map_err(XlaError::from)?;
    let addressable_devices = client.addressable_devices().map_err(XlaError::from)?;
    let mut dst_device_by_id = HashMap::with_capacity(addressable_devices.len());
    for device in addressable_devices {
        let id = device.id().map_err(XlaError::from)?;
        dst_device_by_id.insert(id, device);
    }

    // For a fully-replicated source, the data is present on every src_mesh device. Pick a
    // canonical sender — the first src_mesh device — to drive all cross-host sends from this
    // process. Receiving processes use the same device id as their cross-host source.
    let canonical_src_device = source
        .mesh()
        .devices()
        .first()
        .copied()
        .ok_or_else(|| ArrayError::MissingAddressableShardForMove { shard_index: 0, device_id: 0 })?;
    let canonical_src_global_id = cross_host_global_device_id(canonical_src_device.id())?;
    let this_process_is_canonical_sender = canonical_src_device.process_index() == client_process_index;
    let local_source_buffer = source.addressable_shards().find_map(|shard| shard.buffer().cloned());

    // Decide each destination device's transfer plan. Local destinations are served by
    // `copy_to_device` when this process owns a source buffer, or by `cross_host_receive_buffers`
    // when the source is on another process. Remote destinations are served by
    // `cross_host_send_buffers` from the canonical-sender process and ignored elsewhere.
    let mut local_buffers: HashMap<DeviceId, Buffer<'o>> = HashMap::new();
    let mut send_buffers: Vec<&Buffer<'o>> = Vec::new();
    let mut send_dst_devices: Vec<GlobalDeviceId> = Vec::new();
    let mut send_transfer_keys: Vec<CrossHostTransferKey> = Vec::new();
    // Receives must be grouped by destination device for the cross-host API.
    let mut receive_plans_by_device: HashMap<DeviceId, Vec<(GlobalDeviceId, CrossHostTransferKey)>> = HashMap::new();

    let dst_count = dst_mesh.devices().len();
    for (dst_index, dst_device) in dst_mesh.devices().iter().enumerate() {
        let dst_device_id = dst_device.id();
        let dst_local = dst_device.process_index() == client_process_index;
        let transfer_key = exact_shard_transfer_key(0, dst_index, dst_count)?;
        let dst_global_id = cross_host_global_device_id(dst_device_id)?;

        if dst_local {
            let pjrt_device =
                dst_device_by_id.get(&dst_device_id).ok_or(ArrayError::NonAddressableDestinationDevice {
                    device_id: dst_device_id,
                    process_index: dst_device.process_index(),
                })?;
            if let Some(local_source) = local_source_buffer.as_ref() {
                let buffer = local_source.copy_to_device(pjrt_device.clone()).map_err(XlaError::from)?;
                local_buffers.insert(dst_device_id, buffer);
            } else {
                receive_plans_by_device
                    .entry(dst_device_id)
                    .or_default()
                    .push((canonical_src_global_id, transfer_key));
            }
        } else if this_process_is_canonical_sender {
            let local_source = local_source_buffer.as_ref().ok_or(ArrayError::MissingAddressableShardForMove {
                shard_index: 0,
                device_id: canonical_src_device.id(),
            })?;
            send_buffers.push(local_source.as_ref());
            send_dst_devices.push(dst_global_id);
            send_transfer_keys.push(transfer_key);
        }
        // Otherwise: remote destination served by some other process; nothing to do here.
    }

    // If any cross-host transfer is required, verify the PJRT plugin exposes the extension. On
    // backends without it, return a clear `NonAddressableDestinationDevice` error rather than
    // letting the send/receive call fail later with a generic `Unimplemented`.
    if !send_buffers.is_empty() || !receive_plans_by_device.is_empty() {
        if client.cross_host_transfers_extension().is_err() {
            let blocking_dst = dst_mesh
                .devices()
                .iter()
                .find(|device| device.process_index() != client_process_index)
                .expect("cross-host transfer planned but no remote destination found");
            return Err(ArrayError::NonAddressableDestinationDevice {
                device_id: blocking_dst.id(),
                process_index: blocking_dst.process_index(),
            });
        }
    }

    // Issue cross-host sends, if any.
    if !send_buffers.is_empty() {
        client
            .cross_host_send_buffers(
                send_buffers.as_slice(),
                send_dst_devices.as_slice(),
                send_transfer_keys.as_slice(),
            )
            .map_err(XlaError::from)?;
    }

    // Issue cross-host receives, if any. Each call covers one destination device, matching the
    // existing fast-path pattern in `transfers.rs`.
    let element_type_pjrt = source.data_type().to_pjrt();
    let shape_i64 = source.shape().as_slice().iter().map(|&size| size as i64).collect::<Vec<_>>();
    let mut receive_device_ids = receive_plans_by_device.keys().copied().collect::<Vec<_>>();
    receive_device_ids.sort_unstable();
    for dst_device_id in receive_device_ids {
        let plans = receive_plans_by_device.get(&dst_device_id).expect("receive plans present after sort");
        let pjrt_device = dst_device_by_id.get(&dst_device_id).ok_or(ArrayError::NonAddressableDestinationDevice {
            device_id: dst_device_id,
            process_index: client_process_index,
        })?;
        let element_types = plans.iter().map(|_| element_type_pjrt).collect::<Vec<_>>();
        let dimensions = plans.iter().map(|_| shape_i64.as_slice()).collect::<Vec<_>>();
        let source_devices = plans.iter().map(|(src_global_id, _)| *src_global_id).collect::<Vec<_>>();
        let transfer_keys = plans.iter().map(|(_, key)| *key).collect::<Vec<_>>();
        let received = client
            .cross_host_receive_buffers(
                element_types.as_slice(),
                dimensions.as_slice(),
                pjrt_device,
                source_devices.as_slice(),
                transfer_keys.as_slice(),
            )
            .map_err(XlaError::from)?;
        for buffer in received {
            local_buffers.insert(dst_device_id, buffer);
        }
    }

    // Assemble per-device buffers in `dst_mesh` device order. Only locally addressable destinations
    // contribute to the intermediate Array; remote destinations are handled by their owning
    // processes via parallel calls to this function.
    let mut buffers = Vec::with_capacity(local_buffers.len());
    for dst_device in dst_mesh.devices() {
        if let Some(buffer) = local_buffers.remove(&dst_device.id()) {
            buffers.push(buffer);
        }
    }

    let shape = source.shape();
    let element_type = source.data_type();
    let replicated_on_dst = Sharding::replicated(dst_mesh.logical_mesh().clone(), shape.rank());
    let intermediate_type = ArrayType::new(element_type, shape.into())
        .with_sharding(replicated_on_dst)
        .map_err(XlaError::from)?;
    let intermediate = Array::from_addressable_buffers(client, intermediate_type, dst_mesh.clone(), buffers)?;

    // The intermediate is owned exclusively by this function and is never observed by callers.
    // Donating its buffers lets PJRT reuse their memory for the output of the final SPMD reshard.
    try_same_mesh(&intermediate, engine, dst_mesh, dst_sharding, true)
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
/// Both compiled programs go through `cache`'s LRU, so repeated calls with the same input/output
/// shardings pay the compile cost once.
fn try_sharded_cross_mesh<'o>(
    source: &Array<'o>,
    engine: &XlaDomain<'o>,
    src_mesh: &DeviceMesh,
    dst_mesh: &DeviceMesh,
    dst_sharding: &Sharding,
) -> Result<Array<'o>, ArrayError> {
    let shape = source.shape();
    let replicated_on_src = Sharding::replicated(src_mesh.logical_mesh().clone(), shape.rank());
    // Source is owned externally, so we do not donate it during the all-gather. The gathered
    // intermediate is owned by this function and is donated to the subsequent broadcast/reshard
    // step inside `try_replicated_cross_mesh`.
    let gathered = try_same_mesh(source, engine, src_mesh, &replicated_on_src, false)?;
    try_replicated_cross_mesh(&gathered, engine, dst_mesh, dst_sharding)
}
