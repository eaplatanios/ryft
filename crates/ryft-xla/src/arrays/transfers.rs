use super::*;

/// Deterministic exact-shard transfer plan for one [`Array::to_placement`] call on the current process.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExactShardPutPlan {
    /// Local destination shards that can be satisfied via intra-host device-to-device copies.
    pub(crate) local_copies: Vec<LocalShardCopyPlan>,

    /// Cross-host sends that this process must initiate for remote destination shards.
    pub(crate) cross_host_sends: Vec<CrossHostShardSendPlan>,

    /// Cross-host receives that this process must initiate for local destination shards.
    pub(crate) cross_host_receives: Vec<CrossHostShardReceivePlan>,
}

/// One exact-shard local device-to-device copy in an [`ExactShardPutPlan`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LocalShardCopyPlan {
    /// Source shard index in the source array.
    pub(crate) source_shard_index: ShardIndex,

    /// Source device ID.
    pub(crate) source_device_id: DeviceId,

    /// Destination shard index in the destination array.
    pub(crate) destination_shard_index: ShardIndex,

    /// Destination device ID.
    pub(crate) destination_device_id: DeviceId,
}

/// One exact-shard cross-host send in an [`ExactShardPutPlan`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CrossHostShardSendPlan {
    /// Source shard index in the source array.
    pub(crate) source_shard_index: ShardIndex,

    /// Source device ID.
    pub(crate) source_device_id: DeviceId,

    /// Destination shard index in the destination array.
    pub(crate) destination_shard_index: ShardIndex,

    /// Destination device ID.
    pub(crate) destination_device_id: DeviceId,

    /// Deterministic transfer rendezvous key shared with the matching receive.
    pub(crate) transfer_key: CrossHostTransferKey,
}

/// One exact-shard cross-host receive in an [`ExactShardPutPlan`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CrossHostShardReceivePlan {
    /// Source shard index in the source array.
    pub(crate) source_shard_index: ShardIndex,

    /// Source device ID.
    pub(crate) source_device_id: DeviceId,

    /// Destination shard index in the destination array.
    pub(crate) destination_shard_index: ShardIndex,

    /// Destination device ID.
    pub(crate) destination_device_id: DeviceId,

    /// Logical destination shard shape.
    pub(crate) destination_shape: Vec<usize>,

    /// Deterministic transfer rendezvous key shared with the matching send.
    pub(crate) transfer_key: CrossHostTransferKey,
}

/// Returns the deterministic exact-shard cross-host transfer key for one source/destination pair.
fn exact_shard_transfer_key(
    source_shard_index: ShardIndex,
    destination_shard_index: ShardIndex,
    destination_shard_count: usize,
) -> Result<CrossHostTransferKey, ArrayError> {
    let transfer_key = source_shard_index
        .checked_mul(destination_shard_count)
        .and_then(|key| key.checked_add(destination_shard_index))
        .ok_or(ArrayError::CrossHostTransferKeyTooLarge { source_shard_index, destination_shard_index })?;
    i64::try_from(transfer_key)
        .map_err(|_| ArrayError::CrossHostTransferKeyTooLarge { source_shard_index, destination_shard_index })
}

/// Returns the PJRT cross-host global device ID for `device_id`.
fn cross_host_global_device_id(device_id: DeviceId) -> Result<GlobalDeviceId, ArrayError> {
    i32::try_from(device_id).map_err(|_| ArrayError::CrossHostTransferDeviceIdTooLarge { device_id })
}

/// Returns the PJRT cross-host shape for one destination shard.
fn cross_host_shape(plan: &CrossHostShardReceivePlan) -> Result<Vec<i64>, ArrayError> {
    plan.destination_shape
        .iter()
        .enumerate()
        .map(|(dimension, &size)| {
            i64::try_from(size).map_err(|_| ArrayError::CrossHostTransferShapeDimensionTooLarge {
                shard_index: plan.destination_shard_index,
                dimension,
                size,
            })
        })
        .collect()
}

/// Returns the preferred exact matching source shard for `destination_shard`.
///
/// Preference is deterministic and depends only on global shard metadata so that every process
/// chooses the same source shard for a given destination shard:
/// 1. prefer a source shard on the destination process,
/// 2. then prefer the same device ID as the destination shard, and
/// 3. finally break ties by device ID and shard index.
fn preferred_exact_source_shard<'a, 'o>(
    source_shards: &[&'a ArrayShard<'o>],
    destination_shard: &ShardDescriptor,
) -> &'a ArrayShard<'o> {
    source_shards
        .iter()
        .min_by_key(|source_shard| {
            (
                source_shard.device.process_index != destination_shard.device.process_index,
                source_shard.device.id != destination_shard.device.id,
                source_shard.device.id,
                source_shard.index,
            )
        })
        .copied()
        .expect("preferred exact source shard selection requires at least one candidate")
}

/// Plans exact whole-shard moves for one [`Array::to_placement`] call on the current process.
///
/// Returns `Ok(None)` when any destination shard requires repartitioning or concatenating multiple
/// source shards, which means the exact-shard fast path cannot satisfy the requested sharding.
pub(crate) fn plan_exact_shard_put<'o>(
    array: &Array<'o>,
    client_process_index: usize,
    global_shape: &[usize],
    mesh: &DeviceMesh,
    sharding: &Sharding,
) -> Result<Option<ExactShardPutPlan>, ArrayError> {
    let mut source_shards_by_slices = HashMap::<Vec<Range<usize>>, Vec<&ArrayShard<'o>>>::new();
    for shard in array.shards() {
        source_shards_by_slices.entry(shard.slice.clone()).or_default().push(shard);
    }

    let destination_layout = ShardLayout::new(global_shape, mesh, sharding)?;
    let destination_shards = destination_layout.descriptors();
    let destination_shard_count = destination_shards.len();
    let mut plan =
        ExactShardPutPlan { local_copies: Vec::new(), cross_host_sends: Vec::new(), cross_host_receives: Vec::new() };
    for destination_shard in destination_shards {
        let source_shards = match source_shards_by_slices.get(&destination_shard.slice) {
            Some(source_shards) => source_shards,
            None => return Ok(None),
        };
        let source_shard = preferred_exact_source_shard(source_shards.as_slice(), destination_shard);
        let source_process_index = source_shard.device.process_index;
        let destination_process_index = destination_shard.device.process_index;

        if destination_process_index == client_process_index {
            if source_process_index == client_process_index {
                if !source_shard.buffer.is_some() {
                    return Ok(None);
                }
                plan.local_copies.push(LocalShardCopyPlan {
                    source_shard_index: source_shard.index,
                    source_device_id: source_shard.device.id,
                    destination_shard_index: destination_shard.index,
                    destination_device_id: destination_shard.device.id,
                });
            } else {
                plan.cross_host_receives.push(CrossHostShardReceivePlan {
                    source_shard_index: source_shard.index,
                    source_device_id: source_shard.device.id,
                    destination_shard_index: destination_shard.index,
                    destination_device_id: destination_shard.device.id,
                    destination_shape: destination_shard.shape(),
                    transfer_key: exact_shard_transfer_key(
                        source_shard.index,
                        destination_shard.index,
                        destination_shard_count,
                    )?,
                });
            }
        } else if source_process_index == client_process_index {
            if !source_shard.buffer.is_some() {
                return Ok(None);
            }
            plan.cross_host_sends.push(CrossHostShardSendPlan {
                source_shard_index: source_shard.index,
                source_device_id: source_shard.device.id,
                destination_shard_index: destination_shard.index,
                destination_device_id: destination_shard.device.id,
                transfer_key: exact_shard_transfer_key(
                    source_shard.index,
                    destination_shard.index,
                    destination_shard_count,
                )?,
            });
        }
    }

    Ok(Some(plan))
}

/// Tries to build the destination local shard buffers via exact whole-shard transfers.
///
/// This fast path succeeds only when every destination shard addressable by `client` already
/// exists as one source shard with the exact same logical slices. Local source shards are copied
/// directly between devices, and remote source shards are transferred with the PJRT cross-host
/// transfers extension when it is available. When the destination requires repartitioning or the
/// cross-host extension is unavailable for a needed remote move, the function returns `Ok(None)`
/// so that [`Array::to_placement`] can fall back to the dense host path.
pub(crate) fn copy_addressable_destination_shards_from_exact_source_shards<'o>(
    array: &Array<'o>,
    client: &'o Client<'_>,
    global_shape: &[usize],
    mesh: &DeviceMesh,
    sharding: &Sharding,
) -> Result<Option<Vec<Buffer<'o>>>, ArrayError> {
    let client_process_index = client.process_index()?;
    let plan = match plan_exact_shard_put(array, client_process_index, global_shape, mesh, sharding)? {
        Some(plan) => plan,
        None => return Ok(None),
    };
    let needs_cross_host_transfers = !plan.cross_host_sends.is_empty() || !plan.cross_host_receives.is_empty();
    if needs_cross_host_transfers {
        match client.cross_host_transfers_extension() {
            Ok(_) => {}
            Err(PjrtError::Unimplemented { .. }) => return Ok(None),
            Err(error) => return Err(error.into()),
        }
    }

    let addressable_devices = client.addressable_devices()?;
    let mut addressable_device_by_id = HashMap::with_capacity(addressable_devices.len());
    for device in addressable_devices {
        addressable_device_by_id.insert(device.id()?, device);
    }

    if !plan.cross_host_sends.is_empty() {
        let mut send_buffers = Vec::with_capacity(plan.cross_host_sends.len());
        let mut destination_devices = Vec::with_capacity(plan.cross_host_sends.len());
        let mut transfer_keys = Vec::with_capacity(plan.cross_host_sends.len());
        for send_plan in &plan.cross_host_sends {
            let source_buffer = array
                .addressable_shard_for_device(send_plan.source_device_id)
                .ok_or(ArrayError::MissingAddressableShardForMove {
                    shard_index: send_plan.source_shard_index,
                    device_id: send_plan.source_device_id,
                })?
                .buffer
                .as_deref()
                .expect("addressable shard lookups should always return a local buffer");
            send_buffers.push(source_buffer);
            destination_devices.push(cross_host_global_device_id(send_plan.destination_device_id)?);
            transfer_keys.push(send_plan.transfer_key);
        }
        let _send_events = client.cross_host_send_buffers(
            send_buffers.as_slice(),
            destination_devices.as_slice(),
            transfer_keys.as_slice(),
        )?;
    }

    let mut addressable_buffers = Vec::new();
    for local_copy_plan in &plan.local_copies {
        let source_buffer = array
            .addressable_shard_for_device(local_copy_plan.source_device_id)
            .ok_or(ArrayError::MissingAddressableShardForMove {
                shard_index: local_copy_plan.source_shard_index,
                device_id: local_copy_plan.source_device_id,
            })?
            .buffer
            .as_deref()
            .expect("addressable shard lookups should always return a local buffer");
        let destination_device = addressable_device_by_id.get(&local_copy_plan.destination_device_id).ok_or(
            ArrayError::MissingClientDeviceForLocalMeshDevice {
                device_id: local_copy_plan.destination_device_id,
                process_index: client_process_index,
            },
        )?;
        if local_copy_plan.source_device_id == local_copy_plan.destination_device_id {
            addressable_buffers.push(source_buffer.bitcast(ryft_pjrt::BufferSpecification {
                element_type: source_buffer.element_type()?,
                dimensions: source_buffer.dimensions()?,
                #[allow(deprecated)]
                layout: source_buffer.layout()?,
            })?);
        } else {
            addressable_buffers.push(source_buffer.copy_to_device(destination_device.clone())?);
        }
    }

    let mut receive_plans_by_device = HashMap::<DeviceId, Vec<&CrossHostShardReceivePlan>>::new();
    for receive_plan in &plan.cross_host_receives {
        receive_plans_by_device.entry(receive_plan.destination_device_id).or_default().push(receive_plan);
    }
    let mut receive_device_ids = receive_plans_by_device.keys().copied().collect::<Vec<_>>();
    receive_device_ids.sort_unstable();
    for receive_device_id in receive_device_ids {
        let receive_plans = receive_plans_by_device
            .get(&receive_device_id)
            .expect("grouped receive plans should exist for every grouped destination device");
        let destination_device = addressable_device_by_id.get(&receive_device_id).ok_or(
            ArrayError::MissingClientDeviceForLocalMeshDevice {
                device_id: receive_device_id,
                process_index: client_process_index,
            },
        )?;
        let element_types = receive_plans.iter().map(|_| array.element_type().to_pjrt()).collect::<Vec<_>>();
        let dimensions = receive_plans
            .iter()
            .map(|receive_plan| cross_host_shape(receive_plan))
            .collect::<Result<Vec<_>, _>>()?;
        let dimension_slices = dimensions.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let source_devices = receive_plans
            .iter()
            .map(|receive_plan| cross_host_global_device_id(receive_plan.source_device_id))
            .collect::<Result<Vec<_>, _>>()?;
        let transfer_keys = receive_plans.iter().map(|receive_plan| receive_plan.transfer_key).collect::<Vec<_>>();
        let received_buffers = client.cross_host_receive_buffers(
            element_types.as_slice(),
            dimension_slices.as_slice(),
            destination_device,
            source_devices.as_slice(),
            transfer_keys.as_slice(),
        )?;
        addressable_buffers.extend(received_buffers);
    }

    Ok(Some(addressable_buffers))
}
