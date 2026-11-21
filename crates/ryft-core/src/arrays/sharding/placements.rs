use std::fmt::Display;

use ryft_macros::Parameter;

use crate::arrays::sharding::ShardingError;
use crate::arrays::sharding::meshes::DeviceMesh;
use crate::arrays::sharding::shardings::{Sharding, ShardingDimension};
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
///
/// A [`Placement`] can also describe a batch item inside a batched function, which observes the placement of the packed
/// value with its batch dimension removed. Such a batch item does not live on the whole mesh when its batch is split
/// along mesh axes, because each device then holds only some of the batch items, and no sharding of the batch item
/// alone can say that. Such a placement therefore also records the sharding of the batch dimension of each batched
/// value that encloses it (refer to [`Self::batch_shardings`]), so that batching rules can put that sharding back when
/// they lift the placement of a batch item to the placement of its packed value. That record describes the enclosing
/// batches rather than the array itself, and it is empty for an ordinary placement. It takes part in equality and
/// hashing, so two placements of batch items whose batches are distributed differently are different.
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

    /// Sharding of the batch dimension of each batched value that encloses the array, from the outermost to the
    /// innermost one. Refer to [`Self::batch_shardings`] for more information.
    batch_shardings: Vec<ShardingDimension>,
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
        Ok(Self { mesh, sharding, layout: None, memory: None, batch_shardings: Vec::new() })
    }

    /// Returns this [`Placement`] with its sharding replaced by `sharding`, keeping its mesh, [`Layout`], [`Memory`],
    /// and recorded batch shardings (refer to [`Self::batch_shardings`]).
    ///
    /// # Errors
    ///
    /// Returns [`ShardingError::MeshMismatch`] if the logical mesh of `sharding` does not have the same topology as the
    /// logical mesh of [`Self::mesh`], and the errors of [`Self::with_batch_sharding`] if a recorded batch sharding
    /// cannot be put back into `sharding` (e.g., because `sharding` already uses one of its axes).
    pub fn with_sharding(mut self, sharding: Sharding) -> Result<Self, ShardingError> {
        check_sharding!(self.mesh, sharding);
        if !self.batch_shardings.is_empty() {
            sharding.with_dimensions(sharding.dimensions().iter().chain(&self.batch_shardings).cloned().collect())?;
        }
        self.sharding = sharding;
        Ok(self)
    }

    /// Returns this [`Placement`] with its storage [`Layout`] set to `layout`.
    #[inline]
    pub fn with_layout(mut self, layout: Layout) -> Self {
        self.layout = Some(layout);
        self
    }

    /// Returns this [`Placement`] without a storage [`Layout`], keeping everything else.
    #[inline]
    pub fn without_layout(mut self) -> Self {
        self.layout = None;
        self
    }

    /// Returns this [`Placement`] with its [`Memory`] space set to `memory`.
    #[inline]
    pub fn with_memory(mut self, memory: Memory) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Returns this [`Placement`] without a [`Memory`] space, keeping everything else.
    #[inline]
    pub fn without_memory(mut self) -> Self {
        self.memory = None;
        self
    }

    /// Returns this [`Placement`] with `batch_sharding` recorded as the sharding of the batch dimension of one more
    /// batched value that encloses the array, which becomes the innermost one (refer to [`Self::batch_shardings`]).
    /// Batched values use this function when they report the placement of their batch items, which is the placement
    /// of their packed value with the batch dimension removed, so that the placement still says how the batch is
    /// distributed.
    ///
    /// # Errors
    ///
    /// Returns [`ShardingError::UnknownMeshAxisName`] if `batch_sharding` names an axis that the mesh of
    /// [`Self::sharding`] does not define, [`ShardingError::DuplicateMeshAxisName`] if it names an axis that a
    /// dimension of [`Self::sharding`], its unreduced or reduced axes, or another recorded batch dimension already
    /// uses, and [`ShardingError::EmptySharding`] if it is sharded along no axes. These are the checks that
    /// [`Sharding::with_dimensions`] makes when the batch dimension is put back into [`Self::sharding`],
    /// so every recorded batch sharding can be put back.
    pub fn with_batch_sharding(mut self, batch_sharding: ShardingDimension) -> Result<Self, ShardingError> {
        let dimensions = self
            .sharding
            .dimensions()
            .iter()
            .chain(&self.batch_shardings)
            .chain(std::iter::once(&batch_sharding))
            .cloned()
            .collect();
        self.sharding.with_dimensions(dimensions)?;
        self.batch_shardings.push(batch_sharding);
        Ok(self)
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

    /// Returns the sharding of the batch dimension of each batched value that encloses the array, from the outermost
    /// to the innermost one. A batch item does not live on the whole mesh when its batch is split along mesh axes, and
    /// this record is what says so, because [`Self::sharding`] describes only the batch item itself. Each recorded
    /// sharding names only axes of the mesh of [`Self::sharding`] that none of its dimensions, none of its unreduced or
    /// reduced axes, and no other recorded sharding uses. The record is empty for an ordinary placement (i.e., one that
    /// does not describe a batch item).
    #[inline]
    pub fn batch_shardings(&self) -> &[ShardingDimension] {
        &self.batch_shardings
    }

    /// Removes the innermost recorded batch sharding (refer to [`Self::batch_shardings`]) and returns it, or returns
    /// [`None`] if this placement records none. A batching rule uses this function to lift the placement of a batch
    /// item to the placement of the packed value that holds the batch, which no longer has that enclosing batch.
    #[inline]
    pub fn pop_batch_sharding(&mut self) -> Option<ShardingDimension> {
        self.batch_shardings.pop()
    }

    /// Consumes this placement and returns its mesh, sharding, layout, and memory. The recorded batch shardings (refer
    /// to [`Self::batch_shardings`]) are dropped, because they describe the batches that enclose the array rather than
    /// the array itself.
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
        if !self.batch_shardings.is_empty() {
            let batch_shardings = self.batch_shardings.iter().map(ToString::to_string).collect::<Vec<_>>();
            write!(formatter, ", batch_shardings=[{}]", batch_shardings.join(", "))?;
        }
        write!(formatter, "}}")
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use pretty_assertions::assert_eq;

    use crate::arrays::sharding::meshes::{Device, LogicalMesh, MeshAxis, MeshAxisType};
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
    fn test_placement_with_sharding() {
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let devices = (0..4).map(|id| Device::new(id, 0)).collect();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices).unwrap();
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        let replicated = Sharding::replicated(logical_mesh.clone(), 1);
        let split = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
        let placement = Placement::new(mesh.clone(), replicated)
            .unwrap()
            .with_layout(layout.clone())
            .with_memory(Memory::Host { pinned: true })
            .with_batch_sharding(ShardingDimension::sharded(["x"]))
            .unwrap();

        // Only the sharding changes, so the layout, the memory, and the recorded batch shardings are kept.
        let placed = placement.clone().with_sharding(split.clone()).unwrap();
        assert_eq!(placed.mesh(), &mesh);
        assert_eq!(placed.sharding(), &split);
        assert_eq!(placed.layout(), Some(&layout));
        assert_eq!(placed.memory(), Some(Memory::Host { pinned: true }));
        assert_eq!(placed.batch_shardings(), &[ShardingDimension::sharded(["x"])]);

        // A sharding that uses the axis of a recorded batch sharding could not have that batch put back.
        let conflicting = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        assert_eq!(
            placement.clone().with_sharding(conflicting),
            Err(ShardingError::DuplicateMeshAxisName { name: "x".to_string() }),
        );

        // A sharding over a mesh with a different topology is rejected.
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("z", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        assert_eq!(
            placement.with_sharding(Sharding::replicated(other_mesh.clone(), 1)),
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
    fn test_placement_without_layout() {
        let mesh = DeviceMesh::new(LogicalMesh::new(vec![]).unwrap(), vec![Device::new(3, 1)]).unwrap();
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let placement = Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 2))
            .unwrap()
            .with_memory(Memory::Host { pinned: true })
            .with_batch_sharding(ShardingDimension::Replicated)
            .unwrap();

        // Only the layout is dropped, so the memory and the recorded batch shardings are kept.
        assert_eq!(placement.clone().with_layout(layout).without_layout(), placement);
        assert_eq!(placement.clone().without_layout(), placement);
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
    fn test_placement_without_memory() {
        let mesh = DeviceMesh::new(LogicalMesh::new(vec![]).unwrap(), vec![Device::new(3, 1)]).unwrap();
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let placement = Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 2))
            .unwrap()
            .with_layout(layout)
            .with_batch_sharding(ShardingDimension::Replicated)
            .unwrap();

        // Only the memory is dropped, so the layout and the recorded batch shardings are kept.
        assert_eq!(placement.clone().with_memory(Memory::Device).without_memory(), placement);
        assert_eq!(placement.clone().without_memory(), placement);
    }

    #[test]
    fn test_placement_with_batch_sharding() {
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), (0..4).map(|id| Device::new(id, 0)).collect()).unwrap();
        let placement = Placement::new(mesh.clone(), Sharding::replicated(logical_mesh.clone(), 1)).unwrap();
        assert!(placement.batch_shardings().is_empty());

        // Batch shardings are recorded from the outermost to the innermost enclosing batch, and they are rendered
        // last, only when there are any.
        let placement = placement
            .with_memory(Memory::Device)
            .with_batch_sharding(ShardingDimension::sharded(["x"]))
            .unwrap()
            .with_batch_sharding(ShardingDimension::Replicated)
            .unwrap();
        assert_eq!(placement.batch_shardings(), &[ShardingDimension::sharded(["x"]), ShardingDimension::Replicated]);
        assert_eq!(placement.memory(), Some(Memory::Device));
        assert_eq!(
            placement.to_string(),
            "{mesh<['x'=2:auto, 'y'=2:explicit]>, devices=[0, 1, 2, 3], \
             sharding={mesh<['x'=2:auto, 'y'=2:explicit]>, [{}]}, memory=Device, batch_shardings=[{'x'}, {}]}",
        );

        // A recorded batch sharding must be one that can be put back into the sharding of the placement, so it can
        // only name axes of its mesh that neither the sharding nor another recorded batch sharding uses.
        assert_eq!(
            placement.clone().with_batch_sharding(ShardingDimension::sharded(["z"])),
            Err(ShardingError::UnknownMeshAxisName { name: "z".to_string() }),
        );
        assert_eq!(
            placement.clone().with_batch_sharding(ShardingDimension::sharded(["x"])),
            Err(ShardingError::DuplicateMeshAxisName { name: "x".to_string() }),
        );
        let sharded = Placement::new(
            mesh.clone(),
            Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap(),
        )
        .unwrap();
        assert_eq!(
            sharded.with_batch_sharding(ShardingDimension::sharded(["y"])),
            Err(ShardingError::DuplicateMeshAxisName { name: "y".to_string() }),
        );

        // Axes that the sharding holds as unreduced or reduced are in use as well, so they cannot shard a batch.
        let reduction_state = Sharding::replicated(logical_mesh, 1)
            .with_unreduced_axes(["x"])
            .unwrap()
            .with_reduced_axes(["y"])
            .unwrap();
        let reduced = Placement::new(mesh, reduction_state).unwrap();
        assert_eq!(
            reduced.clone().with_batch_sharding(ShardingDimension::sharded(["x"])),
            Err(ShardingError::DuplicateMeshAxisName { name: "x".to_string() }),
        );
        assert_eq!(
            reduced.with_batch_sharding(ShardingDimension::sharded(["y"])),
            Err(ShardingError::DuplicateMeshAxisName { name: "y".to_string() }),
        );
    }

    #[test]
    fn test_placement_pop_batch_sharding() {
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(0, 0), Device::new(1, 0)],
        )
        .unwrap();
        let ordinary = Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 1)).unwrap();
        let mut placement = ordinary
            .clone()
            .with_batch_sharding(ShardingDimension::sharded(["x"]))
            .unwrap()
            .with_batch_sharding(ShardingDimension::Replicated)
            .unwrap();

        // The innermost batch sharding is removed first, and an ordinary placement records none.
        assert_eq!(placement.pop_batch_sharding(), Some(ShardingDimension::Replicated));
        assert_eq!(placement.batch_shardings(), &[ShardingDimension::sharded(["x"])]);
        assert_eq!(placement.pop_batch_sharding(), Some(ShardingDimension::sharded(["x"])));
        assert_eq!(placement, ordinary);
        assert_eq!(placement.pop_batch_sharding(), None);
        assert_eq!(placement, ordinary);
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
        assert_eq!(
            placement.clone().into_parts(),
            (mesh.clone(), sharding.clone(), Some(layout.clone()), Some(memory))
        );

        // The recorded batch shardings describe the enclosing batches rather than the array, so they are dropped.
        let placement = placement.with_batch_sharding(ShardingDimension::Replicated).unwrap();
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
        let batch_item = placement.clone().with_batch_sharding(ShardingDimension::Replicated).unwrap();
        assert_eq!(placement, placement.clone());
        assert_eq!(batch_item, batch_item.clone());
        assert_ne!(placement, other_placement);
        assert_ne!(placement, pinned);
        assert_ne!(placement, batch_item);
        let placements = HashSet::from([placement.clone(), other_placement.clone()]);
        assert!(placements.contains(&placement));
        assert!(placements.contains(&other_placement));
        assert!(!placements.contains(&pinned));
        assert!(!placements.contains(&batch_item));
    }
}
