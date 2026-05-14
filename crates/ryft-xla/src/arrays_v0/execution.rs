use super::*;

pub struct ExecuteArguments<'o> {
    /// Addressable devices in the same order as [`Self::inputs_by_device`].
    addressable_device_ids: Vec<DeviceId>,

    /// Execution inputs grouped by addressable device.
    inputs_by_device: Vec<Vec<ExecutionInput<'o>>>,
}

impl<'o> ExecuteArguments<'o> {
    /// Returns addressable device IDs corresponding to [`Self::inputs_by_device`].
    pub fn addressable_device_ids(&self) -> &[DeviceId] {
        self.addressable_device_ids.as_slice()
    }

    /// Returns execution inputs grouped by device.
    pub fn inputs_by_device(&self) -> &[Vec<ExecutionInput<'o>>] {
        self.inputs_by_device.as_slice()
    }

    /// Creates PJRT `ExecutionDeviceInputs` in the same device order as [`Self::addressable_device_ids`].
    pub fn as_execution_device_inputs<'l>(&'l self) -> Vec<ExecutionDeviceInputs<'o, 'l>> {
        self.inputs_by_device.iter().map(|inputs| ExecutionDeviceInputs::from(inputs.as_slice())).collect()
    }

    pub(crate) fn from_arrays_with_donation(
        arrays: Vec<Array<'o>>,
        addressable_device_ids: &[DeviceId],
        donation_flags: &[bool],
    ) -> Result<Self, ArrayError> {
        if donation_flags.len() != arrays.len() {
            return Err(ArrayError::DonationFlagCountMismatch {
                expected_count: arrays.len(),
                actual_count: donation_flags.len(),
            });
        }

        let mut seen_devices = HashSet::with_capacity(addressable_device_ids.len());
        for &device_id in addressable_device_ids {
            if !seen_devices.insert(device_id) {
                return Err(ArrayError::DuplicateExecutionDeviceId { device_id });
            }
        }

        let mut buffers_by_array =
            arrays.into_iter().map(Array::into_addressable_buffers_by_device).collect::<Vec<_>>();

        let mut inputs_by_device = Vec::with_capacity(addressable_device_ids.len());
        for &device_id in addressable_device_ids {
            let mut device_inputs = Vec::with_capacity(buffers_by_array.len());
            for (array_index, array_buffers_by_device) in buffers_by_array.iter_mut().enumerate() {
                let buffer = array_buffers_by_device
                    .remove(&device_id)
                    .ok_or(ArrayError::MissingArrayShardForDevice { array_index, device_id })?;
                device_inputs.push(ExecutionInput { buffer, donatable: donation_flags[array_index] });
            }
            inputs_by_device.push(device_inputs);
        }

        for (array_index, array_buffers_by_device) in buffers_by_array.iter().enumerate() {
            if let Some(device_id) = array_buffers_by_device.keys().next().copied() {
                return Err(ArrayError::UnexpectedArrayShardDevice { array_index, device_id });
            }
        }

        Ok(Self { addressable_device_ids: addressable_device_ids.to_vec(), inputs_by_device })
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
