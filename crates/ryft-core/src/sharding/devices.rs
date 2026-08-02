use std::collections::HashSet;

use crate::sharding::ShardingError;
use crate::sharding::meshes::{LogicalMesh, MeshAxisType};

/// Type alias used to represent [`Device`] IDs, which are unique among devices of the same type (e.g., CPUs, GPUs)
/// and, on multi-host environments, are also unique across all devices and all hosts.
pub type DeviceId = usize;

/// Type alias used to represent process indices in multi-process/multi-host environments.
pub type ProcessIndex = usize;

/// Device that belongs to a mesh topology. This type separates global device identity that is described by a
/// [`DeviceId`], from host/process ownership, that is described by a [`ProcessIndex`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Device {
    /// Globally (i.e., across all hosts/processes) unique [`DeviceId`].
    id: DeviceId,

    /// Index of the process that owns this device. In single-host setups, this will always be set to `0`. In multi-host
    /// setups it determines _addressability_. That is, a _shard_ of an array that is located on some device `d` is
    /// _addressable_ from a process with index `p` if and only if `d.process_index == p`.
    process_index: ProcessIndex,
}

impl Device {
    /// Creates a new [`Device`].
    #[inline]
    pub fn new(id: DeviceId, process_index: ProcessIndex) -> Self {
        Self { id, process_index }
    }

    /// Returns globally (i.e., across all hosts/processes) unique [`DeviceId`] of this [`Device`].
    #[inline]
    pub fn id(&self) -> DeviceId {
        self.id
    }

    /// Returns the index of the process that owns this device. In single-host setups, this will always be set to `0`.
    /// In multi-host setups it determines _addressability_. That is, a _shard_ of an array that is located on some
    /// device `d` is _addressable_ from a process with index `p` if and only if `d.process_index == p`.
    #[inline]
    pub fn process_index(&self) -> ProcessIndex {
        self.process_index
    }
}

/// Mesh of devices used by sharding layouts. A [`DeviceMesh`] organizes a set of [`Device`]s into a [`LogicalMesh`].
/// Devices are stored in **row-major order** with respect to the [`MeshAxis`] list (e.g., for a two-dimensional mesh
/// with axes `("data"=4, "model"=2)`, the device at mesh coordinate `(i, j)` has linear index `i * 2 + j`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceMesh {
    /// Logical mesh topology that defines the names, sizes, and types of the mesh axes.
    pub(crate) logical_mesh: LogicalMesh,

    /// Physical devices laid out in row-major order with respect to [`Self::logical_mesh`].
    pub(crate) devices: Vec<Device>,
}

impl DeviceMesh {
    /// Creates a new [`DeviceMesh`].
    #[inline]
    pub fn new(logical_mesh: LogicalMesh, devices: Vec<Device>) -> Result<Self, ShardingError> {
        let expected_device_count = logical_mesh.device_count();
        if devices.len() != expected_device_count {
            return Err(ShardingError::DeviceCountMismatch { expected: expected_device_count, actual: devices.len() });
        }

        let mut seen_device_ids = HashSet::with_capacity(devices.len());
        for device in &devices {
            if !seen_device_ids.insert(device.id) {
                return Err(ShardingError::DuplicateDeviceId { id: device.id });
            }
        }

        Ok(Self { logical_mesh, devices })
    }

    /// Returns the logical mesh topology that defines the names, sizes, and types of the mesh axes.
    #[inline]
    pub fn logical_mesh(&self) -> &LogicalMesh {
        &self.logical_mesh
    }

    /// Returns the physical devices laid out in row-major order with respect to [`Self::logical_mesh`].
    #[inline]
    pub fn devices(&self) -> &[Device] {
        &self.devices
    }

    /// Returns the rank (i.e., number of axes) of this [`DeviceMesh`].
    #[inline]
    pub fn rank(&self) -> usize {
        self.logical_mesh.rank()
    }

    /// Returns the size of the [`MeshAxis`] in this [`DeviceMesh`] with the provided name, if such an axis exists.
    #[inline]
    pub fn axis_size<S: AsRef<str>>(&self, axis_name: S) -> Option<usize> {
        self.logical_mesh.axis_size(axis_name)
    }

    /// Returns the type of the [`MeshAxis`] in this [`DeviceMesh`] with the provided name, if such an axis exists.
    #[inline]
    pub fn axis_type<S: AsRef<str>>(&self, axis_name: S) -> Option<MeshAxisType> {
        self.logical_mesh.axis_type(axis_name)
    }

    /// Returns the total number of devices that the topology defined by this [`DeviceMesh`] contains.
    #[inline]
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Returns the mesh coordinates of the [`Device`] at the provided index, if valid.
    #[inline]
    pub fn device_coordinates(&self, device_index: usize) -> Option<Vec<usize>> {
        (device_index < self.devices.len()).then(|| {
            let axis_sizes = self.logical_mesh.axes.iter().map(|axis| axis.size).collect::<Vec<_>>();
            let mut coordinates = vec![0usize; axis_sizes.len()];
            let mut index = device_index;
            for (axis_index, axis_size) in axis_sizes.iter().enumerate().rev() {
                coordinates[axis_index] = index % axis_size;
                index /= axis_size;
            }
            coordinates
        })
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::sharding::meshes::MeshAxis;

    use super::*;

    #[test]
    fn test_device_mesh() {
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let devices = vec![Device::new(0, 0), Device::new(1, 0), Device::new(2, 1), Device::new(3, 1)];
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap();
        assert_eq!(&mesh.logical_mesh, &logical_mesh);
        assert_eq!(&mesh.devices, &devices);
        assert_eq!(mesh.rank(), 2);
        assert_eq!(mesh.axis_size("x"), Some(2));
        assert_eq!(mesh.axis_size("y"), Some(2));
        assert_eq!(mesh.axis_size("z"), None);
        assert_eq!(mesh.axis_type("x"), Some(MeshAxisType::Auto));
        assert_eq!(mesh.axis_type("y"), Some(MeshAxisType::Manual));
        assert_eq!(mesh.axis_type("z"), None);
        assert_eq!(mesh.device_count(), 4);
        assert_eq!(mesh.device_coordinates(0), Some(vec![0, 0]));
        assert_eq!(mesh.device_coordinates(1), Some(vec![0, 1]));
        assert_eq!(mesh.device_coordinates(2), Some(vec![1, 0]));
        assert_eq!(mesh.device_coordinates(3), Some(vec![1, 1]));
        assert_eq!(mesh.device_coordinates(4), None);

        assert!(matches!(
            DeviceMesh::new(logical_mesh.clone(), vec![Device::new(0, 0), Device::new(1, 0), Device::new(2, 1)],),
            Err(ShardingError::DeviceCountMismatch { expected: 4, actual: 3 }),
        ));
        assert!(matches!(
            DeviceMesh::new(
                logical_mesh.clone(),
                vec![Device::new(0, 0), Device::new(0, 0), Device::new(1, 1), Device::new(2, 1)],
            ),
            Err(ShardingError::DuplicateDeviceId { id }) if id == 0,
        ));
    }
}
