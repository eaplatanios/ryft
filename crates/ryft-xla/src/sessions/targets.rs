use std::collections::{BTreeMap, HashSet};

use ryft_core::{Device, DeviceId, DeviceMesh};
use ryft_pjrt::{Client, Value as PjrtValue, Version as PjrtVersion};

use crate::{CompilationError, Error, FromPjrt};

/// Snapshot of the platform, plugin, and devices used for compilation and execution by a [`Session`](crate::Session).
/// Lowering reads the platform and device capabilities to select implementations. The metadata of compiled programs
/// records target facts for cache identity and compatibility checks when loading persisted executables. Execution also
/// uses the captured process and device addressability to validate placements for reference state and ordered effects.
///
/// Create a snapshot with [`Self::from_client`], or obtain the one retained by a session through
/// [`Session::target`](crate::Session::target). All strings, attributes, and device descriptions are owned, so a
/// snapshot can be inspected or cloned independently of the client. Loading executables, allocating buffers, and
/// submitting work still require the live client and session resources. Keeping these facts separate gives lowering
/// a fixed target description within each session and leaves room for future descriptions obtained from a PJRT
/// topology for compilation without the target hardware. The current constructor captures facts from a live client.
#[derive(Clone, Debug)]
pub struct Target {
    /// Platform name reported by the capturing client (e.g., `"cpu"` or `"cuda"`).
    platform_name: String,

    /// Platform implementation version reported by the capturing client.
    platform_version: String,

    /// PJRT C API version exposed by the client's plugin.
    pjrt_version: PjrtVersion,

    /// Availability of the PJRT FFI extension when the snapshot was captured.
    has_ffi_extension: bool,

    /// Owned PJRT plugin attributes, ordered by attribute name.
    attributes: BTreeMap<String, PjrtValue>,

    /// Process index reported by the capturing client.
    process_index: usize,

    /// All devices visible to the client, including remote devices, in the captured enumeration order.
    devices: Vec<TargetDevice>,
}

impl Target {
    /// Creates a new [`Target`] by capturing a live PJRT [`Client`]'s platform, plugin capabilities, and all visible
    /// devices. Device order follows [`Client::devices`], and addressability is determined by membership in
    /// [`Client::addressable_devices`]. All returned metadata is owned and retains no client or device handles.
    /// An unavailable PJRT FFI extension is recorded as `false` in [`Self::has_ffi_extension`].
    ///
    /// # Errors
    ///
    /// Returns the first PJRT error, as [`Error::Pjrt`], encountered while querying platform information,
    /// plugin attributes, process identity, or device descriptions and addressability.
    pub fn from_client(client: &Client<'_>) -> Result<Self, Error> {
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
            .collect::<Result<Vec<_>, Error>>()?;
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
    pub fn pjrt_version(&self) -> PjrtVersion {
        self.pjrt_version
    }

    /// Returns whether the PJRT FFI extension was available when this snapshot was captured.
    #[inline]
    pub fn has_ffi_extension(&self) -> bool {
        self.has_ffi_extension
    }

    /// Returns the captured PJRT plugin attributes, ordered by attribute name.
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
    #[inline]
    pub fn device(&self, device_id: DeviceId) -> Option<&TargetDevice> {
        self.devices.iter().find(|device| device.device.id() == device_id)
    }

    /// Returns the captured PJRT device kinds for `mesh`, preserving mesh order and matching devices by identifier.
    /// These kinds contribute to the metadata of lowered programs and compatibility checks for persisted executables.
    ///
    /// # Errors
    ///
    /// Returns [`CompilationError::InvalidPersistentExecutable`] if a mesh device identifier is absent from the
    /// snapshot.
    pub fn mesh_device_kinds(&self, mesh: &DeviceMesh) -> Result<Vec<String>, CompilationError> {
        mesh.devices()
            .iter()
            .map(|mesh_device| {
                self.device(mesh_device.id()).map(|device| device.kind.clone()).ok_or_else(|| {
                    CompilationError::InvalidPersistentExecutable {
                        reason: format!("device {} is not visible to the live client", mesh_device.id()),
                    }
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

    /// Returns whether this target's process participates in executions over `mesh`. Every device in `mesh` must
    /// be a captured device with the process index that the mesh assigns it, the devices of this process must be
    /// addressable while those of other processes are not, and at least one device must belong to this process. An
    /// unknown device identifier, a mismatched process index, or a mesh entirely of other processes makes the result
    /// `false`. Execution with reference state uses this check, because each process binds, executes, and commits only
    /// its addressable shards of a reference, as the stateful execution protocol in [`execution`](crate::execution)
    /// describes.
    pub fn is_participating_mesh(&self, mesh: &DeviceMesh) -> bool {
        let mut participates = false;
        let consistent = mesh.devices().iter().all(|mesh_device| {
            self.device(mesh_device.id()).is_some_and(|device| {
                let local = mesh_device.process_index() == self.process_index;
                participates |= local;
                device.device == *mesh_device && device.addressable == local
            })
        });
        consistent && participates
    }
}

/// Snapshot of one device's identity, capabilities, and addressability within a [`Target`]. The embedded [`Device`]
/// supplies the identity used in meshes, which is independent of the backend. The remaining fields capture the PJRT
/// device kind, typed capabilities, and addressability from that client. A [`TargetDevice`] retains these facts without
/// a live [`ryft_pjrt::Device`] handle, so reading them requires no further runtime queries. Devices visible on other
/// processes can be included even when the capturing client cannot address them.
#[derive(Clone, Debug)]
pub struct TargetDevice {
    /// Device identity and owning process, independent of the backend, used to match mesh entries.
    device: Device,

    /// PJRT device kind, defined by the implementation and used in hardware compatibility checks.
    kind: String,

    /// PJRT device attributes, ordered by name, including hardware capabilities exposed by the plugin.
    attributes: BTreeMap<String, PjrtValue>,

    /// Whether the capturing client listed this device among its addressable devices.
    addressable: bool,
}

impl TargetDevice {
    /// Returns the identity and owning process, independent of the backend, used to place this device in a
    /// [`DeviceMesh`].
    #[inline]
    pub fn device(&self) -> Device {
        self.device
    }

    /// Returns the captured PJRT device kind, which the implementation defines and hardware compatibility checks use.
    #[inline]
    pub fn kind(&self) -> &str {
        self.kind.as_str()
    }

    /// Returns the captured PJRT device attributes, ordered by attribute name. Hardware capability attributes depend on
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

    use ryft_core::{LogicalMesh, MeshAxis, MeshAxisType, NegOperation, Placeholder};

    use crate::sessions::tests::{domain_mesh, replicated_vector_type};
    use crate::tests::cpu_client;
    use crate::{FlatProgram, Operation, Options, ProgramBuilder, Session, Type};

    use super::*;

    /// Builds a target of process 0 over `devices`, each given as its identifier, owning process, and kind.
    /// A device is addressable when it belongs to process 0.
    fn target(devices: &[(DeviceId, usize, &str)]) -> Target {
        Target {
            platform_name: "test".into(),
            platform_version: "test".into(),
            pjrt_version: PjrtVersion { major: 0, minor: 0 },
            has_ffi_extension: false,
            attributes: BTreeMap::new(),
            process_index: 0,
            devices: devices
                .iter()
                .map(|&(id, process_index, kind)| TargetDevice {
                    device: Device::new(id, process_index),
                    kind: kind.into(),
                    attributes: BTreeMap::new(),
                    addressable: process_index == 0,
                })
                .collect(),
        }
    }

    /// Builds a mesh with one manual axis over `devices`.
    fn mesh(devices: Vec<Device>) -> DeviceMesh {
        DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", devices.len(), MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap()
    }

    #[test]
    fn test_target_from_client() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let target = session.target().unwrap();
        assert!(std::ptr::eq(target, session.clone().target().unwrap()));
        assert_eq!(target.platform_name(), client.platform_name().unwrap());
        assert_eq!(target.platform_version(), client.platform_version().unwrap());
        assert_eq!(target.pjrt_version(), client.version());
        assert_eq!(target.has_ffi_extension(), client.ffi_extension().is_ok());
        assert_eq!(target.attributes(), &client.attributes().unwrap().clone().into_iter().collect::<BTreeMap<_, _>>());
        assert_eq!(target.process_index(), client.process_index().unwrap());
        let devices = client.devices().unwrap();
        assert_eq!(target.devices().len(), devices.len());
        for (target_device, device) in target.devices().iter().zip(devices.iter()) {
            assert_eq!(target_device.device(), Device::from_pjrt(device).unwrap());
            assert_eq!(target_device.kind(), device.kind().unwrap());
            assert_eq!(
                target_device.attributes(),
                &device.attributes().unwrap().clone().into_iter().collect::<BTreeMap<_, _>>(),
            );
            assert!(target_device.is_addressable());
        }

        // Lowering over a mesh of the captured devices records the target facts.
        let mesh = domain_mesh(client, "x", 2);
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(Type::Array(replicated_vector_type(&mesh, 2)));
        let output = builder
            .add_instruction(Operation::Array(NegOperation::new().into()), Vec::new(), vec![input], None)
            .unwrap()
            .to_vec();
        let program: FlatProgram = builder.build(output, vec![Placeholder], vec![Placeholder]).unwrap();
        let lowered = session.lower_program(&program, 0, &Options::new(mesh)).unwrap();
        assert_eq!(lowered.platform_name(), target.platform_name());
        assert_eq!(lowered.platform_version(), target.platform_version());
        let device_kinds = devices.iter().map(|device| device.kind().unwrap().into_owned()).collect::<Vec<_>>();
        assert_eq!(lowered.device_kinds(), device_kinds.as_slice());
    }

    #[test]
    fn test_target_mesh_device_kinds() {
        let target = target(&[(7, 0, "first"), (9, 0, "second")]);
        assert_eq!(
            target.mesh_device_kinds(&mesh(vec![Device::new(9, 0), Device::new(7, 0)])).unwrap(),
            vec!["second", "first"],
        );
        assert!(matches!(
            target.mesh_device_kinds(&mesh(vec![Device::new(usize::MAX, 0)])),
            Err(CompilationError::InvalidPersistentExecutable { reason })
                if reason == format!("device {} is not visible to the live client", usize::MAX),
        ));
    }

    #[test]
    fn test_target_is_fully_addressable_single_process_mesh() {
        let target = target(&[(0, 0, "cpu"), (1, 0, "cpu"), (2, 1, "cpu")]);
        let (first, second, remote) = (Device::new(0, 0), Device::new(1, 0), Device::new(2, 1));
        assert!(target.is_fully_addressable_single_process_mesh(&mesh(vec![first, second])));

        // Devices of other processes, devices that the mesh attributes to another process, and unknown devices are
        // not addressable by this process.
        assert!(!target.is_fully_addressable_single_process_mesh(&mesh(vec![first, remote])));
        assert!(!target.is_fully_addressable_single_process_mesh(&mesh(vec![Device::new(0, 1)])));
        assert!(!target.is_fully_addressable_single_process_mesh(&mesh(vec![Device::new(usize::MAX, 0)])));
    }

    #[test]
    fn test_target_is_participating_mesh() {
        let target = target(&[(0, 0, "cpu"), (2048, 1, "cpu")]);
        let (local, remote) = (Device::new(0, 0), Device::new(2048, 1));
        assert!(target.is_participating_mesh(&mesh(vec![local])));
        assert!(target.is_participating_mesh(&mesh(vec![local, remote])));

        // This process does not participate in a mesh of only remote devices, nor in meshes that misattribute a device
        // to a process or name an unknown device.
        assert!(!target.is_participating_mesh(&mesh(vec![remote])));
        assert!(!target.is_participating_mesh(&mesh(vec![local, Device::new(2048, 0)])));
        assert!(!target.is_participating_mesh(&mesh(vec![local, Device::new(usize::MAX, 1)])));
    }

    #[test]
    fn test_target_device() {
        // Devices of other processes are found too, while unknown identifiers are not.
        let target = target(&[(7, 0, "local"), (9, 1, "remote")]);
        assert_eq!(target.device(7).map(TargetDevice::kind), Some("local"));
        assert_eq!(target.device(9).map(TargetDevice::device), Some(Device::new(9, 1)));
        assert_eq!(target.device(9).map(TargetDevice::is_addressable), Some(false));
        assert!(target.device(8).is_none());
    }
}
