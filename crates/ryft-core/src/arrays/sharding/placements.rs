use std::fmt::Display;

use ryft_macros::Parameter;

use crate::arrays::sharding::ShardingError;
use crate::arrays::sharding::meshes::DeviceMesh;
use crate::arrays::sharding::shardings::Sharding;
use crate::arrays::types::layouts::Layout;
use crate::arrays::types::memories::Memory;
use crate::macros::check_sharding;
use crate::parameters::Parameter;

/// Resolved placement of an array on concrete devices. A [`Placement`] combines a [`DeviceMesh`], a [`Sharding`] over
/// that mesh, and optionally the [`Memory`] space and storage [`Layout`] of the array. It is to [`DeviceMesh`] what
/// [`Sharding`] is to [`LogicalMesh`](crate::LogicalMesh) (i.e., it names the devices that hold the shards of an array,
/// while a sharding only describes how the array is partitioned over a mesh topology). The sharding is a runtime
/// sharding, so it may place dimensions over [`MeshAxisType::Auto`](crate::MeshAxisType::Auto) mesh axes, which array
/// types cannot record (refer to [`Sharding::without_auto_axes`]).
///
/// A [`Placement`] plays two roles. As the target of a [`PlaceOperation`](crate::PlaceOperation), it says where a
/// value must be moved, and an omitted memory or layout keeps the storage that the value already has. As the source
/// of such an operation, it is informative and says where the value is expected to live when the operation is
/// created. Transposition uses it to move the cotangent back to those devices, and type inference checks it against
/// what the type of the value tracks, while backends that execute the operation on a concrete value move the value
/// from wherever it actually lives without comparing that placement with the source.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct Placement {
    /// [`DeviceMesh`] whose devices hold the shards of the array.
    mesh: DeviceMesh,

    /// [`Sharding`] of the array over [`Self::mesh`].
    sharding: Sharding,

    /// Optional storage [`Layout`] of the array.
    layout: Option<Layout>,

    /// Optional [`Memory`] space of the array.
    memory: Option<Memory>,
}

impl Placement {
    /// Creates a new [`Placement`] of an array that is sharded according to `sharding` over the devices of `mesh`,
    /// without any [`Layout`] or [`Memory`] information.
    ///
    /// # Errors
    ///
    /// Returns [`ShardingError::MeshMismatch`] if the logical mesh of `sharding` does not have the same topology as the
    /// logical mesh of `mesh` (refer to [`LogicalMesh::has_same_topology`](crate::LogicalMesh::has_same_topology)).
    #[inline]
    pub fn new(mesh: DeviceMesh, sharding: Sharding) -> Result<Self, ShardingError> {
        check_sharding!(mesh, sharding);
        Ok(Self { mesh, sharding, layout: None, memory: None })
    }

    /// Returns this [`Placement`] with its storage [`Layout`] set to `layout`.
    #[inline]
    pub fn with_layout(mut self, layout: Layout) -> Self {
        self.layout = Some(layout);
        self
    }

    /// Returns this [`Placement`] with its [`Memory`] space set to `memory`.
    #[inline]
    pub fn with_memory(mut self, memory: Memory) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Returns the [`DeviceMesh`] whose devices hold the shards of the array.
    #[inline]
    pub fn mesh(&self) -> &DeviceMesh {
        &self.mesh
    }

    /// Returns the [`Sharding`] of the array over [`Self::mesh`]. Its logical mesh has the same topology as that of
    /// [`Self::mesh`], although the two may type their axes differently.
    #[inline]
    pub fn sharding(&self) -> &Sharding {
        &self.sharding
    }

    /// Returns the storage [`Layout`] of the array, or [`None`] if this [`Placement`] does not specify one.
    #[inline]
    pub fn layout(&self) -> Option<&Layout> {
        self.layout.as_ref()
    }

    /// Returns the [`Memory`] space of the array, or [`None`] if this [`Placement`] does not specify one.
    #[inline]
    pub fn memory(&self) -> Option<Memory> {
        self.memory
    }

    /// Consumes this placement and returns its mesh, sharding, layout, and memory.
    #[inline]
    pub fn into_parts(self) -> (DeviceMesh, Sharding, Option<Layout>, Option<Memory>) {
        (self.mesh, self.sharding, self.layout, self.memory)
    }
}

impl Display for Placement {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The device mesh is rendered as its logical mesh followed by its device IDs, which are globally unique,
        // so that the rendering identifies the devices along with how the sharding partitions the array over them.
        write!(
            formatter,
            "{{mesh<{}>, devices=[{}], sharding={}",
            self.mesh.logical_mesh(),
            self.mesh.devices().iter().map(|device| device.id().to_string()).collect::<Vec<_>>().join(", "),
            self.sharding,
        )?;
        if let Some(memory) = self.memory {
            write!(formatter, ", memory={memory}")?;
        }
        if let Some(layout) = &self.layout {
            write!(formatter, ", layout={layout}")?;
        }
        write!(formatter, "}}")
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use pretty_assertions::assert_eq;

    use crate::arrays::sharding::meshes::{Device, LogicalMesh, MeshAxis, MeshAxisType};
    use crate::arrays::sharding::shardings::ShardingDimension;
    use crate::arrays::types::layouts::TiledLayout;

    use super::*;

    #[test]
    fn test_placement() {
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![Device::new(0, 0), Device::new(1, 0)],
        )
        .unwrap();

        // The sharding may type the axes of the device mesh differently (e.g., as auto), as long as the topology
        // agrees.
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let sharding = Sharding::new(auto_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        let placement = Placement::new(mesh.clone(), sharding.clone()).unwrap();
        assert_eq!(placement.mesh(), &mesh);
        assert_eq!(placement.sharding(), &sharding);
        assert_eq!(placement.layout(), None);
        assert_eq!(placement.memory(), None);
        assert_eq!(
            placement.to_string(),
            "{mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:auto]>, [{'x'}]}}",
        );

        // A sharding over a mesh with a different topology is rejected.
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        assert_eq!(
            Placement::new(mesh.clone(), Sharding::replicated(other_mesh.clone(), 1)),
            Err(ShardingError::MeshMismatch { expected: mesh.logical_mesh().clone(), actual: other_mesh }),
        );
    }

    #[test]
    fn test_placement_with_layout() {
        let mesh = DeviceMesh::new(LogicalMesh::new(vec![]).unwrap(), vec![Device::new(3, 1)]).unwrap();
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let placement = Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 2))
            .unwrap()
            .with_layout(layout.clone())
            .with_memory(Memory::Device);
        assert_eq!(placement.layout(), Some(&layout));
        assert_eq!(placement.memory(), Some(Memory::Device));
        assert_eq!(
            placement.to_string(),
            format!("{{mesh<[]>, devices=[3], sharding={{mesh<[]>, [{{}}, {{}}]}}, memory=Device, layout={layout}}}"),
        );
    }

    #[test]
    fn test_placement_with_memory() {
        let mesh = DeviceMesh::new(LogicalMesh::new(vec![]).unwrap(), vec![Device::new(3, 1)]).unwrap();
        let placement = Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap()
            .with_memory(Memory::Host { pinned: true });
        assert_eq!(placement.layout(), None);
        assert_eq!(placement.memory(), Some(Memory::Host { pinned: true }));
        assert_eq!(placement.to_string(), "{mesh<[]>, devices=[3], sharding={mesh<[]>, []}, memory=Host[Pinned]}");
    }

    #[test]
    fn test_placement_into_parts() {
        let mesh = DeviceMesh::new(LogicalMesh::new(vec![]).unwrap(), vec![Device::new(3, 1)]).unwrap();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 2);
        let placement = Placement::new(mesh.clone(), sharding.clone()).unwrap();
        assert_eq!(placement.clone().into_parts(), (mesh.clone(), sharding.clone(), None, None));

        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let memory = Memory::Host { pinned: true };
        let placement = placement.with_layout(layout.clone()).with_memory(memory);
        assert_eq!(placement.into_parts(), (mesh, sharding, Some(layout), Some(memory)));
    }

    #[test]
    fn test_placement_equality_and_hash() {
        let mesh = DeviceMesh::new(LogicalMesh::new(vec![]).unwrap(), vec![Device::new(0, 0)]).unwrap();
        let other_mesh = DeviceMesh::new(LogicalMesh::new(vec![]).unwrap(), vec![Device::new(1, 0)]).unwrap();
        let placement = Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 0)).unwrap();
        let other_placement =
            Placement::new(other_mesh.clone(), Sharding::replicated(other_mesh.logical_mesh().clone(), 0)).unwrap();
        let pinned = placement.clone().with_memory(Memory::Host { pinned: true });
        assert_eq!(placement, placement.clone());
        assert_ne!(placement, other_placement);
        assert_ne!(placement, pinned);
        let placements = HashSet::from([placement.clone(), other_placement.clone()]);
        assert!(placements.contains(&placement));
        assert!(placements.contains(&other_placement));
        assert!(!placements.contains(&pinned));
    }
}
