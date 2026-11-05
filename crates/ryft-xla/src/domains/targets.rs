use std::collections::{BTreeMap, HashSet};

use ryft_core::{Device, DeviceId, DeviceMesh};
use ryft_pjrt::{Client, Value as PjrtValue};

use crate::domains::compilation::CompilationError;
use crate::{Error, FromPjrt};

// TODO(eaplatanios): Review this module.

/// Owned snapshot of the platform, plugin, and devices used for compilation and execution by a
/// [`Session`](crate::Session).
///
/// Lowering reads the platform and device capabilities to select implementations. Compiled-program metadata records
/// target facts for cache identity and compatibility checks when loading persisted executables. Execution also uses
/// the captured process and device addressability to validate placements for reference state and ordered effects.
///
/// Create a snapshot with [`Self::from_client`], or obtain the one retained by a session through
/// [`Session::target`](crate::Session::target). All strings, attributes, and device descriptions are owned, so a
/// snapshot can be inspected or cloned independently of the client. Loading executables, allocating buffers, and
/// submitting work still require the live client and session resources.
///
/// Keeping these facts separate gives lowering a fixed target description within each session and leaves room for
/// future descriptions obtained from a PJRT topology for compilation without the target hardware. The current
/// constructor captures facts from a live client.
#[derive(Clone, Debug)]
pub struct Target {
    /// Platform name reported by the capturing client (e.g., `"cpu"` or `"cuda"`).
    platform_name: String,

    /// Opaque platform implementation version reported by the capturing client.
    platform_version: String,

    /// PJRT C API version exposed by the client's plugin.
    pjrt_version: ryft_pjrt::Version,

    /// Availability of the PJRT FFI extension when the snapshot was captured.
    has_ffi_extension: bool,

    /// Owned plugin attributes, ordered by attribute name.
    attributes: BTreeMap<String, PjrtValue>,

    /// Process index reported by the capturing client.
    process_index: usize,

    /// All devices visible to the client, including remote devices, in the captured enumeration order.
    devices: Vec<TargetDevice>,
}

/// Owned snapshot of one device's identity, capabilities, and addressability within a [`Target`].
///
/// The embedded [`Device`] supplies the backend-neutral identity used in meshes. The remaining fields capture the
/// PJRT device kind, typed capabilities, and addressability from that client. A [`TargetDevice`] retains these facts
/// without a live [`ryft_pjrt::Device`] handle, so reading them requires no further runtime queries. Devices visible
/// on another process can be included even when the capturing client cannot address them.
#[derive(Clone, Debug)]
pub struct TargetDevice {
    /// Backend-neutral device identity and owning process, used to match mesh entries.
    device: Device,

    /// Implementation-defined PJRT device kind, used in hardware compatibility checks.
    kind: String,

    /// Owned device attributes, ordered by name, including hardware capabilities exposed by the plugin.
    attributes: BTreeMap<String, PjrtValue>,

    /// Whether the capturing client listed this device among its addressable devices.
    addressable: bool,
}

impl Target {
    /// Captures a live PJRT [`Client`]'s platform, plugin capabilities, and all visible devices.
    ///
    /// Device order follows [`Client::devices`], and addressability is determined by membership in
    /// [`Client::addressable_devices`]. All returned metadata is owned and retains no client or device handles.
    /// An unavailable PJRT FFI extension is recorded as `false` in [`Self::has_ffi_extension`].
    ///
    /// # Errors
    ///
    /// Returns the first [`ryft_pjrt::Error`] encountered while querying platform information, plugin attributes,
    /// process identity, or device descriptions and addressability.
    pub fn from_client(client: &Client<'_>) -> Result<Self, ryft_pjrt::Error> {
        let addressable_device_ids =
            client.addressable_devices()?.iter().map(|device| device.id()).collect::<Result<HashSet<_>, _>>()?;
        let devices = client
            .devices()?
            .iter()
            .map(|device| {
                let device_facts = Device::from_pjrt(device)?;
                Ok(TargetDevice {
                    device: device_facts,
                    kind: device.kind()?.into_owned(),
                    attributes: device
                        .attributes()?
                        .iter()
                        .map(|(name, value)| (name.clone(), value.clone()))
                        .collect(),
                    addressable: addressable_device_ids.contains(&device_facts.id()),
                })
            })
            .collect::<Result<Vec<_>, ryft_pjrt::Error>>()?;
        Ok(Self {
            platform_name: client.platform_name()?.into_owned(),
            platform_version: client.platform_version()?.into_owned(),
            pjrt_version: client.version(),
            has_ffi_extension: client.ffi_extension().is_ok(),
            attributes: client.attributes()?.iter().map(|(name, value)| (name.clone(), value.clone())).collect(),
            process_index: client.process_index()?,
            devices,
        })
    }

    /// Returns the captured PJRT platform name (e.g., `"cpu"` or `"cuda"`).
    #[inline]
    pub fn platform_name(&self) -> &str {
        self.platform_name.as_str()
    }

    /// Returns the captured platform implementation version, an opaque string used in executable compatibility checks.
    #[inline]
    pub fn platform_version(&self) -> &str {
        self.platform_version.as_str()
    }

    /// Returns the PJRT C API version exposed by the capturing client's plugin.
    #[inline]
    pub fn pjrt_version(&self) -> ryft_pjrt::Version {
        self.pjrt_version
    }

    /// Returns whether the PJRT FFI extension was available when this snapshot was captured.
    #[inline]
    pub fn has_ffi_extension(&self) -> bool {
        self.has_ffi_extension
    }

    /// Returns the captured plugin attributes, ordered by attribute name.
    #[inline]
    pub fn attributes(&self) -> &BTreeMap<String, PjrtValue> {
        &self.attributes
    }

    /// Returns the process index reported by the client that this snapshot describes.
    #[inline]
    pub fn process_index(&self) -> usize {
        self.process_index
    }

    /// Returns all captured devices in the client's enumeration order, including devices on other processes.
    /// Use [`TargetDevice::is_addressable`] to determine which devices the capturing client can address.
    #[inline]
    pub fn devices(&self) -> &[TargetDevice] {
        self.devices.as_slice()
    }

    /// Returns the captured device with identifier `device_id`, or [`None`] if that identifier was not visible to
    /// the client. The device can belong to another process and need not be addressable by the capturing client.
    pub fn device(&self, device_id: DeviceId) -> Option<&TargetDevice> {
        self.devices.iter().find(|device| device.device.id() == device_id)
    }

    /// Returns the captured PJRT device kinds for `mesh`, preserving mesh order and matching devices by identifier.
    /// These kinds contribute to the metadata of lowered programs and compatibility checks for persisted executables.
    ///
    /// # Errors
    ///
    /// Returns [`CompilationError::InvalidPersistentExecutable`] wrapped in [`Error::Compilation`] if a mesh device
    /// identifier is absent from the snapshot.
    pub(super) fn mesh_device_kinds(&self, mesh: &DeviceMesh) -> Result<Vec<String>, Error> {
        mesh.devices()
            .iter()
            .map(|mesh_device| {
                self.device(mesh_device.id()).map(|device| device.kind.clone()).ok_or_else(|| {
                    Error::Compilation(CompilationError::InvalidPersistentExecutable {
                        reason: format!("device {} is not visible to the live client", mesh_device.id()),
                    })
                })
            })
            .collect()
    }

    /// Returns whether each device in `mesh` has this target's process index and identifies a captured, addressable
    /// device. An unknown device identifier or a different process index in the mesh makes the result `false`.
    /// Execution with reference state or ordered effects uses this check to restrict placement to one addressable
    /// process.
    pub fn is_fully_addressable_single_process_mesh(&self, mesh: &DeviceMesh) -> bool {
        mesh.devices().iter().all(|mesh_device| {
            mesh_device.process_index() == self.process_index
                && self.device(mesh_device.id()).is_some_and(|device| device.addressable)
        })
    }
}

impl TargetDevice {
    /// Returns the backend-neutral identity and owning process used to place this device in a [`DeviceMesh`].
    #[inline]
    pub fn device(&self) -> Device {
        self.device
    }

    /// Returns the captured, implementation-defined PJRT device kind used in hardware compatibility checks.
    #[inline]
    pub fn kind(&self) -> &str {
        self.kind.as_str()
    }

    /// Returns the captured device attributes, ordered by attribute name. Hardware capability attributes depend on
    /// the PJRT plugin and the device kind.
    #[inline]
    pub fn attributes(&self) -> &BTreeMap<String, PjrtValue> {
        &self.attributes
    }

    /// Returns whether the capturing client listed this device in [`Client::addressable_devices`]. This is captured
    /// addressability relative to that client, independent of where the snapshot is later inspected.
    #[inline]
    pub fn is_addressable(&self) -> bool {
        self.addressable
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::{ArrayIrType, LogicalMesh, MeshAxis, MeshAxisType, NegOperation, Placeholder};
    use ryft_pjrt::{ClientOptions, CpuClientOptions, load_cpu_plugin};

    use crate::domains::options::Options;
    use crate::domains::sessions::Session;
    use crate::domains::tests::*;
    use crate::experimental::ops::{FlatProgram, Operation, ProgramBuilder};

    use super::*;

    #[test]
    fn test_target_from_client() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
        let target = domain.target().unwrap();
        assert!(std::ptr::eq(target, domain.session().target().unwrap()));
        assert_eq!(target.platform_name(), client.platform_name().unwrap());
        assert_eq!(target.platform_version(), client.platform_version().unwrap());
        assert_eq!(target.pjrt_version(), client.version());
        assert_eq!(target.has_ffi_extension(), client.ffi_extension().is_ok());
        assert_eq!(target.process_index(), client.process_index().unwrap());
        let devices = client.devices().unwrap();
        assert_eq!(target.devices().len(), devices.len());
        for (target_device, device) in target.devices().iter().zip(devices.iter()) {
            assert_eq!(target_device.device(), Device::from_pjrt(device).unwrap());
            assert_eq!(target_device.kind(), device.kind().unwrap());
            assert!(target_device.is_addressable());
        }
        assert!(target.device(usize::MAX).is_none());

        // Lowering records the target facts, and a mesh over the target's devices is fully addressable.
        let mesh = domain_mesh(&client, "x", 2);
        assert!(target.is_fully_addressable_single_process_mesh(&mesh));
        let remote_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(mesh.devices()[0].id(), target.process_index() + 1)],
        )
        .unwrap();
        assert!(!target.is_fully_addressable_single_process_mesh(&remote_mesh));
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(ArrayIrType::Array(replicated_vector_type(&mesh, 2)));
        let output = builder
            .add_instruction(Operation::Array(NegOperation::new().into()), Vec::new(), vec![input], None)
            .unwrap()
            .to_vec();
        let program: FlatProgram = builder.build(output, vec![Placeholder], vec![Placeholder]).unwrap();
        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        assert_eq!(lowered.platform_name.as_ref(), target.platform_name());
        assert_eq!(lowered.platform_version.as_ref(), target.platform_version());
        let device_kinds = devices.iter().map(|device| device.kind().unwrap().into_owned()).collect::<Vec<_>>();
        assert_eq!(lowered.device_kinds.as_ref(), device_kinds.as_slice());
    }
}
