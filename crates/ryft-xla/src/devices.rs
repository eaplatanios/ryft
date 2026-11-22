//! PJRT device conversion and memory placement. A `ryft-core` [`Device`] records the device identifier and owning
//! process index. Transfers use the live PJRT device to resolve a [`Memory`] placement to a native memory handle.

use ryft_core::{Device, Memory};
use ryft_pjrt::{Device as PjrtDevice, Memory as PjrtMemory};

use crate::{ArrayError, Error, FromPjrt};

impl FromPjrt<PjrtDevice<'_>> for Device {
    type Output = Result<Self, ryft_pjrt::Error>;

    #[inline]
    fn from_pjrt(value: PjrtDevice<'_>) -> Self::Output {
        Self::from_pjrt(&value)
    }
}

impl FromPjrt<&PjrtDevice<'_>> for Device {
    // Converts a PJRT device into the coordinates of a `ryft-core` `Device` (i.e., its device identifier and its
    // process index), without the PJRT client that owns it. Device coordinates make placements and programs portable
    // across clients, so a converted device names those coordinates rather than that PJRT device: every client with a
    // device at the same coordinates resolves it to its own device, and a `Session` resolves every `Device` that it is
    // given against its own client. Converting the device of another client (e.g., device 0 of a second CPU client) and
    // placing a value through a session therefore places it on the device of the session's client with those
    // coordinates, not on the device of the other client. A session rejects a device of its own process that its client
    // does not address (`ArrayError::NonAddressableDevice`), and the devices of other processes are resolved by the
    // processes that own them. Arrays cannot move between PJRT clients through this frontend, where JAX's `device_put`
    // can transfer an array to a device of another platform.

    type Output = Result<Self, ryft_pjrt::Error>;

    #[inline]
    fn from_pjrt(value: &PjrtDevice<'_>) -> Self::Output {
        Ok(Self::new(value.id()?, value.process_index()?))
    }
}

/// Resolves `ryft-core` memory placements on PJRT devices for XLA uploads and copies.
pub(crate) trait PjrtDeviceExtension {
    /// Returns the PJRT memory corresponding to `memory` on this device. [`Memory::Device`] selects the device's
    /// default memory. [`Memory::Host`] selects its `pinned_host` or `unpinned_host` memory.
    ///
    /// # Errors
    ///
    /// Returns [`ArrayError::UnsupportedMemory`] through [`Error::Array`] if the requested host memory kind is
    /// unavailable. PJRT lookup failures are returned through [`Error::Pjrt`].
    fn memory(&self, memory: Memory) -> Result<PjrtMemory<'_>, Error>;
}

impl PjrtDeviceExtension for PjrtDevice<'_> {
    fn memory(&self, memory: Memory) -> Result<PjrtMemory<'_>, Error> {
        match memory {
            Memory::Device => Ok(self.default_memory()?),
            Memory::Host { .. } => {
                let kind = memory.placement_kind();
                for candidate in self.addressable_memories()? {
                    if candidate.kind()? == kind {
                        return Ok(candidate);
                    }
                }
                Err(ArrayError::UnsupportedMemory { device_id: self.id()?, memory }.into())
            }
        }
    }
}

/// Extends [`Memory`] with its XLA placement kind.
pub(crate) trait MemoryExtension {
    /// Returns the XLA memory kind used by `mhlo.memory_kind` signature attributes and the `_xla_buffer_placement`
    /// frontend attribute on `annotate_device_placement` custom calls. PJRT uses the same names for host memories and
    /// resolves [`Memory::Device`] through the device's default memory.
    fn placement_kind(self) -> &'static str;
}

impl MemoryExtension for Memory {
    #[inline]
    fn placement_kind(self) -> &'static str {
        match self {
            Self::Device => "device",
            Self::Host { pinned: true } => "pinned_host",
            Self::Host { pinned: false } => "unpinned_host",
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::tests::{cpu_client, execution_client};

    use super::*;

    #[test]
    fn test_device_from_pjrt() {
        let client = execution_client();
        let devices = client.addressable_devices().unwrap();
        let expected = Device::new(devices[0].id().unwrap(), devices[0].process_index().unwrap());
        assert_eq!(Device::from_pjrt(&devices[0]), Ok(expected));
        assert_eq!(Device::from_pjrt(devices[0].clone()), Ok(expected));
    }

    #[test]
    fn test_pjrt_device_memory() {
        let devices = cpu_client(1).addressable_devices().unwrap();
        let device = &devices[0];
        assert_eq!(device.memory(Memory::Device).unwrap(), device.default_memory().unwrap());
        assert_eq!(device.memory(Memory::Host { pinned: true }).unwrap().kind().unwrap(), "pinned_host");
        assert_eq!(device.memory(Memory::Host { pinned: false }).unwrap().kind().unwrap(), "unpinned_host");
    }

    #[test]
    fn test_memory_placement_kind() {
        assert_eq!(Memory::Device.placement_kind(), "device");
        assert_eq!(Memory::Host { pinned: true }.placement_kind(), "pinned_host");
        assert_eq!(Memory::Host { pinned: false }.placement_kind(), "unpinned_host");
    }
}
