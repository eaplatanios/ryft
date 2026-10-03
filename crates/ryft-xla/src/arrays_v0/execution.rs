use std::sync::Arc;

use crate::Array;

use super::*;

pub struct ExecuteArguments<'o> {
    /// Addressable devices in the same order as [`Self::inputs_by_device`].
    addressable_device_ids: Arc<[DeviceId]>,

    /// Execution inputs grouped by addressable device.
    inputs_by_device: Vec<Vec<ExecutionInput<'o>>>,
}

impl<'o> ExecuteArguments<'o> {
    /// Returns addressable device IDs corresponding to [`Self::inputs_by_device`].
    pub fn addressable_device_ids(&self) -> &[DeviceId] {
        &self.addressable_device_ids
    }

    /// Returns execution inputs grouped by device.
    pub fn inputs_by_device(&self) -> &[Vec<ExecutionInput<'o>>] {
        self.inputs_by_device.as_slice()
    }

    /// Creates PJRT `ExecutionDeviceInputs` in the same device order as [`Self::addressable_device_ids`].
    pub fn as_execution_device_inputs<'l>(&'l self) -> Vec<ExecutionDeviceInputs<'o, 'l>> {
        self.inputs_by_device.iter().map(|inputs| ExecutionDeviceInputs::from(inputs.as_slice())).collect()
    }

    /// Appends one non-donatable hidden input per addressable device. The execution boundary validates the device
    /// assignment before calling this function; hidden tokens never pass through numerical array construction.
    pub(crate) fn push_input_buffers(&mut self, buffers: Vec<Arc<Buffer<'o>>>) {
        assert_eq!(buffers.len(), self.inputs_by_device.len());
        for (inputs, buffer) in self.inputs_by_device.iter_mut().zip(buffers) {
            inputs.push(ExecutionInput::from(buffer));
        }
    }

    /// Creates execution arguments from `arrays`, honoring `donation_flags` only for arrays whose shard buffers are
    /// uniquely owned. A requested donation is silently downgraded to a non-donating input when any shard buffer is
    /// shared (for example, by another clone of the array or by a retained materialization cache); callers that need
    /// to observe the effective decision can inspect [`ExecutionInput::donatable`] on the result.
    pub(crate) fn from_arrays_with_donation(
        arrays: Vec<Array<'o>>,
        addressable_device_ids: Arc<[DeviceId]>,
        donation_flags: &[bool],
    ) -> Result<Self, ArrayError> {
        if donation_flags.len() != arrays.len() {
            return Err(ArrayError::DonationFlagCountMismatch {
                expected_count: arrays.len(),
                actual_count: donation_flags.len(),
            });
        }
        for (position, &device_id) in addressable_device_ids.iter().enumerate() {
            if addressable_device_ids[..position].contains(&device_id) {
                return Err(ArrayError::DuplicateExecutionDeviceId { device_id });
            }
        }

        // Every array must have exactly one buffer on each addressable device and none elsewhere. Its shards are
        // looked up by device in constant time, and counting its buffers detects buffers on other devices.
        let mut inputs_by_device = (0..addressable_device_ids.len())
            .map(|_| Vec::with_capacity(arrays.len()))
            .collect::<Vec<Vec<ExecutionInput<'o>>>>();
        for (array_index, (array, requested)) in arrays.iter().zip(donation_flags).enumerate() {
            let donatable = *requested && array.has_unique_shard_buffers();
            for (device_inputs, &device_id) in inputs_by_device.iter_mut().zip(addressable_device_ids.iter()) {
                let buffer = array
                    .device_shard(device_id)
                    .and_then(ArrayShard::buffer)
                    .ok_or(ArrayError::MissingArrayShardForDevice { array_index, device_id })?;
                device_inputs.push(ExecutionInput { buffer: Arc::clone(buffer), donatable });
            }
            if array.shards().iter().filter(|shard| shard.buffer().is_some()).count() != addressable_device_ids.len() {
                let device_id = array
                    .shards()
                    .iter()
                    .filter(|shard| shard.buffer().is_some())
                    .map(|shard| shard.device().id())
                    .find(|device_id| !addressable_device_ids.contains(device_id))
                    .unwrap();
                return Err(ArrayError::UnexpectedArrayShardDevice { array_index, device_id });
            }
        }

        // Dropping the consumed arrays leaves the execution inputs as the only owners of donated buffers.
        drop(arrays);
        Ok(Self { addressable_device_ids, inputs_by_device })
    }
}

impl std::fmt::Debug for ExecuteArguments<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let input_counts = self.inputs_by_device.iter().map(Vec::len).collect::<Vec<_>>();
        formatter
            .debug_struct("ExecuteArguments")
            .field("addressable_device_ids", &self.addressable_device_ids)
            .field("input_counts_per_device", &input_counts)
            .finish()
    }
}
