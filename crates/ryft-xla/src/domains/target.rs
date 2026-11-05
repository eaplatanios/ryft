//! Captured PJRT target and device capabilities.

use std::collections::{BTreeMap, HashSet};

use ryft_core::{Device, DeviceId, DeviceMesh};
use ryft_pjrt::{Client, Value as PjrtValue};

use crate::domains::persistence::persistent_error;
use crate::{Error, FromPjrt};

/// Compile target of a [`Session`](crate::Session): the platform, plugin, and device facts that lowering,
/// compilation keys, and persisted-executable validation read, as opposed to the runtime resources (loading,
/// execution, buffers, and kernel runtimes) that require the live PJRT [`Client`].
///
/// Keeping these facts separate from the client makes the client-independence of lowering explicit: everything that
/// lowering reads besides the program and its [`Options`](crate::Options) is fixed per session, which is what
/// lets compiled programs be cached per session. It is also the seam for compiling ahead of time without devices,
/// where the same facts would come from a PJRT topology description instead of a live client.
#[derive(Clone, Debug)]
pub struct Target {
    /// PJRT platform name (e.g., `"cpu"` or `"cuda"`).
    platform_name: String,

    /// PJRT platform implementation version.
    platform_version: String,

    /// PJRT C API version of the plugin.
    pjrt_version: ryft_pjrt::Version,

    /// Whether the plugin exposes the PJRT FFI extension.
    has_ffi_extension: bool,

    /// Plugin attributes, in deterministic name order.
    attributes: BTreeMap<String, PjrtValue>,

    /// Index of the current process.
    process_index: usize,

    /// Every device of the target, in the client's enumeration order.
    devices: Vec<TargetDevice>,
}

/// Facts about one device of an [`Target`].
#[derive(Clone, Debug)]
pub struct TargetDevice {
    /// Identity and owning process of the device.
    device: Device,

    /// PJRT device kind, including the implementation's architecture description.
    kind: String,

    /// Typed device attributes, including compute capability when the plugin exposes it.
    attributes: BTreeMap<String, PjrtValue>,

    /// Whether the device is addressable from the current process.
    addressable: bool,
}

impl Target {
    /// Reads the target facts of a live PJRT [`Client`].
    ///
    /// # Errors
    ///
    /// Returns the [`ryft_pjrt::Error`] of the first failed platform, plugin, or device query.
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

    /// Returns the PJRT platform name (e.g., `"cpu"` or `"cuda"`).
    #[inline]
    pub fn platform_name(&self) -> &str {
        self.platform_name.as_str()
    }

    /// Returns the PJRT platform implementation version.
    #[inline]
    pub fn platform_version(&self) -> &str {
        self.platform_version.as_str()
    }

    /// Returns the PJRT C API version of the plugin.
    #[inline]
    pub fn pjrt_version(&self) -> ryft_pjrt::Version {
        self.pjrt_version
    }

    /// Returns whether the plugin exposes the PJRT FFI extension.
    #[inline]
    pub fn has_ffi_extension(&self) -> bool {
        self.has_ffi_extension
    }

    /// Returns the plugin attributes, in deterministic name order.
    #[inline]
    pub fn attributes(&self) -> &BTreeMap<String, PjrtValue> {
        &self.attributes
    }

    /// Returns the index of the current process.
    #[inline]
    pub fn process_index(&self) -> usize {
        self.process_index
    }

    /// Returns every device of the target, in the client's enumeration order.
    #[inline]
    pub fn devices(&self) -> &[TargetDevice] {
        self.devices.as_slice()
    }

    /// Returns the device with the provided identifier, if the target has one.
    pub fn device(&self, device_id: DeviceId) -> Option<&TargetDevice> {
        self.devices.iter().find(|device| device.device.id() == device_id)
    }

    /// Returns the kinds of the devices of `mesh`, in mesh order, matching devices by identifier. Lowered and
    /// persisted programs record these kinds so that executables are only reused on equivalent hardware.
    pub(super) fn mesh_device_kinds(&self, mesh: &DeviceMesh) -> Result<Vec<String>, Error> {
        mesh.devices()
            .iter()
            .map(|mesh_device| {
                self.device(mesh_device.id()).map(|device| device.kind.clone()).ok_or_else(|| {
                    persistent_error(format!("device {} is not visible to the live client", mesh_device.id()))
                })
            })
            .collect()
    }

    /// Returns whether every device of `mesh` is owned by the current process and addressable from it.
    pub fn is_fully_addressable_single_process_mesh(&self, mesh: &DeviceMesh) -> bool {
        mesh.devices().iter().all(|mesh_device| {
            mesh_device.process_index() == self.process_index
                && self.device(mesh_device.id()).is_some_and(|device| device.addressable)
        })
    }
}

impl TargetDevice {
    /// Returns the identity and owning process of this device.
    #[inline]
    pub fn device(&self) -> Device {
        self.device
    }

    /// Returns the PJRT device kind, including the implementation's architecture description.
    #[inline]
    pub fn kind(&self) -> &str {
        self.kind.as_str()
    }

    /// Returns the typed device attributes.
    #[inline]
    pub fn attributes(&self) -> &BTreeMap<String, PjrtValue> {
        &self.attributes
    }

    /// Returns whether this device is addressable from the current process.
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
