use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

use ryft_core::{DeviceMesh, MeshDevice, MeshDeviceId, Sharding, ShardingDimension, ShardingError, StaticShape};
use ryft_pjrt::Buffer;

use crate::arrays::ArrayError;

/// Row-major ordinal index of an [`ArrayShard`] within a [`DeviceMesh`]. Shard indices are assigned using the same
/// row-major ordering as [`DeviceMesh::devices`]. This gives all processes a stable way to refer to the same global
/// shard without depending on whether that shard is locally addressable or not.
pub type ShardIndex = usize;

/// Placement and slice metadata for an [`ArrayShard`]. A [`ShardDescriptor`] is intentionally independent of any local
/// [`Buffer`]. It describes which [`MeshDevice`] owns the [`ArrayShard`] and which slice of the underlying
/// [`Array`](crate::Array) that shard represents. [`ArrayShard`]s pair this metadata with optional addressable buffers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardDescriptor {
    /// [`ShardIndex`] of this [`ShardDescriptor`]. This index is stable across processes and matches this descriptor's
    /// position in [`ShardLayout::descriptors`].
    pub index: ShardIndex,

    /// [`MeshDevice`] that owns the shard. Note that ownership does not imply local addressability. A shard is local
    /// to a process when [`MeshDevice::process_index`] matches that process.
    pub device: MeshDevice,

    /// Per-dimension global index ranges covered by the shard. This vector contains one [`Range`] per array dimension.
    /// Ranges use normal Rust semantics: the start is inclusive and the end is exclusive. Replicated and unconstrained
    /// dimensions span the full dimension range (i.e., `0..dimension_size`) while sharded dimensions span the
    /// contiguous partition selected by this shard's [`MeshDevice`] coordinates in the underlying [`DeviceMesh`].
    pub slice: Vec<Range<usize>>,
}

impl ShardDescriptor {
    /// Returns the local [`StaticShape`] of the shard described by this [`ShardDescriptor`]. The returned shape is
    /// derived purely from static metadata; it does not inspect any device buffers and is valid for both addressable
    /// and non-addressable shards.
    #[inline]
    pub fn shape(&self) -> StaticShape {
        StaticShape::new(self.slice.iter().map(|slice| slice.len()).collect())
    }
}

/// [`ShardLayout`] that is obtained by applying a [`Sharding`] to a [`StaticShape`] over a [`DeviceMesh`].
/// [`ShardLayout`] is the bridge between type-level device placement metadata and runtime array metadata. It expands
/// the logical sharding specification into one [`ShardDescriptor`] per [`MeshDevice`] and also records a device-ID
/// lookup table for routing local [`Buffer`]s back to their global shard descriptors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardLayout {
    /// [`ShardDescriptor`]s for all global shards in row-major [`DeviceMesh`] order. The position of each descriptor in
    /// this vector is the descriptor's [`ShardDescriptor::index`].
    pub descriptors: Vec<ShardDescriptor>,

    /// Lookup table mapping [`MeshDeviceId`]s to their corresponding [`ShardIndex`]es. This is used when a [`Buffer`]
    /// reports a [`MeshDeviceId`] and the [`Array`](crate::Array) constructor needs to find the global shard metadata
    /// that buffer should satisfy.
    pub shard_index_by_device: HashMap<MeshDeviceId, ShardIndex>,

    /// Private marker that prevents external struct-literal construction. The other fields of this struct are public
    /// for inspection, but construction must go through [`ShardLayout::new`] so that mesh, rank, and sharding-axis
    /// validation is enforced.
    _private: (),
}

impl ShardLayout {
    /// Constructs a new [`ShardLayout`] by applying the provided [`Sharding`] to the provided [`StaticShape`]
    /// over the provided [`DeviceMesh`]. The returned layout contains one [`ShardDescriptor`] per [`MeshDevice`].
    /// For each array dimension, replicated and unconstrained dimensions map every device to the full dimension range,
    /// while sharded dimensions are split across the product of the referenced mesh axes. If the dimension size is not
    /// evenly divisible by the partition count, earlier partitions receive one extra element.
    ///
    /// # Parameters
    ///
    ///   - `global_shape`: [`StaticShape`] of the [`Array`](crate::Array) being sharded/partitioned.
    ///   - `mesh`: [`DeviceMesh`] whose row-major device order determines shard indices.
    ///   - `sharding`: Logical [`Sharding`] specification to apply to `global_shape`.
    pub fn new(shape: &StaticShape, mesh: &DeviceMesh, sharding: &Sharding) -> Result<Self, ArrayError> {
        if mesh.logical_mesh != sharding.mesh {
            return Err(ShardingError::MeshMismatch {
                expected: mesh.logical_mesh.clone(),
                actual: sharding.mesh.clone(),
            }
            .into());
        }

        let sharding_rank = sharding.rank();
        let array_rank = shape.rank();
        if sharding_rank != array_rank {
            return Err(ShardingError::ShardingRankMismatch { sharding_rank, array_rank }.into());
        }

        let dimensions = shape.as_slice();

        // Resolve each sharded array dimension to mesh-axis indices. This keeps the inner per-device loop
        // allocation-free with respect to axis-name lookup, and it also defensively revalidates the provided `sharding`
        // because its fields are public and could have been mutated after construction.
        let mut used_axis_names = HashSet::new();
        let dimension_axis_indices = sharding
            .dimensions
            .iter()
            .enumerate()
            .map(|(dimension, sharding_dimension)| -> Result<Vec<usize>, ArrayError> {
                let ShardingDimension::Sharded(axis_names) = sharding_dimension else {
                    return Ok(Vec::new());
                };
                if axis_names.is_empty() {
                    return Err(ShardingError::EmptySharding { dimension }.into());
                }

                let mut axis_indices = Vec::with_capacity(axis_names.len());
                for axis_name in axis_names {
                    let axis_index = mesh
                        .logical_mesh
                        .axis_indices
                        .get(axis_name.as_str())
                        .copied()
                        .ok_or_else(|| ShardingError::UnknownMeshAxisName { name: axis_name.clone() })?;
                    if !used_axis_names.insert(axis_name.as_str()) {
                        return Err(ShardingError::DuplicateMeshAxisName { name: axis_name.clone() }.into());
                    }
                    axis_indices.push(axis_index);
                }
                Ok(axis_indices)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut descriptors = Vec::with_capacity(mesh.device_count());
        let mut shard_index_by_device = HashMap::with_capacity(mesh.device_count());
        for (index, mesh_device) in mesh.devices.iter().copied().enumerate() {
            // `DeviceMesh` stores devices in row-major mesh order, so the loop index is also the global shard index.
            // Convert it back into mesh coordinates so sharded dimensions can pick the partition owned by this device.
            let device_coordinates =
                mesh.device_coordinates(index).expect("mesh coordinate should exist for valid mesh device index");

            let mut slices = Vec::with_capacity(dimensions.len());
            for (dimension, dimension_size) in dimensions.iter().copied().enumerate() {
                let axis_indices = &dimension_axis_indices[dimension];
                let range = if axis_indices.is_empty() {
                    // Replicated and unconstrained dimensions both map every device to the full dimension range.
                    0..dimension_size
                } else {
                    // Linearize this device's coordinates across the dimension's sharding axes. The resulting
                    // partition index selects one contiguous chunk along the global array dimension.
                    let mut partition_index = 0usize;
                    let mut partition_count = 1usize;
                    for &axis_index in axis_indices {
                        let axis_size = mesh.logical_mesh.axes[axis_index].size;
                        let axis_coordinate = device_coordinates[axis_index];

                        partition_index = partition_index * axis_size + axis_coordinate;
                        partition_count *= axis_size;
                    }

                    let base_size = dimension_size / partition_count;
                    let remainder = dimension_size % partition_count;
                    let extra_before = partition_index.min(remainder);

                    // Split uneven dimensions by assigning one extra element to the earliest partitions.
                    let start = partition_index * base_size + extra_before;
                    let size = base_size + usize::from(partition_index < remainder);
                    start..start + size
                };
                slices.push(range);
            }

            shard_index_by_device.insert(mesh_device.id, index);
            descriptors.push(ShardDescriptor { index, device: mesh_device, slice: slices });
        }

        Ok(Self { descriptors, shard_index_by_device, _private: () })
    }
}

/// Runtime view of one global shard of an [`Array`](crate::arrays::Array).
///
/// An [`ArrayShard`] always carries global shard metadata through [`ArrayShard::descriptor`]. It carries a PJRT
/// [`Buffer`] only when the owning device is addressable from the current process. This lets an
/// [`Array`](crate::arrays::Array) describe the full global layout while storing local buffers only for the shards this
/// process can read, copy, or execute with directly.
///
/// [`ArrayShard`] holds its [`Buffer`] inside an [`Arc`] so that cloning an [`Array`](crate::arrays::Array) can
/// cheaply share addressable PJRT handles with other array instances. The last [`Arc`] dropped releases the underlying
/// PJRT buffer via [`Buffer`]'s [`Drop`] implementation. This mirrors the reference-counted array pattern that IFRT
/// uses above PJRT.
#[derive(Clone)]
pub struct ArrayShard<'o> {
    /// Global metadata for this shard.
    ///
    /// The descriptor is present for every shard, including shards whose buffers are not addressable from this process.
    pub descriptor: ShardDescriptor,

    /// Reference-counted local PJRT buffer for this shard, if addressable from the current process.
    ///
    /// `None` means the shard is owned by a remote process. Cloning an [`ArrayShard`] clones the [`Arc`] and does not
    /// copy device memory.
    pub buffer: Option<Arc<Buffer<'o>>>,
}

impl<'o> ArrayShard<'o> {
    /// Returns this shard's global row-major shard index.
    #[inline]
    pub fn index(&self) -> ShardIndex {
        self.descriptor.index
    }

    /// Returns the mesh device that owns this shard.
    #[inline]
    pub fn device(&self) -> MeshDevice {
        self.descriptor.device
    }

    /// Returns the per-dimension global index ranges covered by this shard.
    #[inline]
    pub fn slice(&self) -> &[Range<usize>] {
        self.descriptor.slice.as_slice()
    }

    /// Returns the static local shape covered by this shard.
    #[inline]
    pub fn shape(&self) -> StaticShape {
        self.descriptor.shape()
    }

    /// Returns `true` when this shard has a local PJRT buffer addressable from the current process.
    #[inline]
    pub fn is_addressable(&self) -> bool {
        self.buffer.is_some()
    }
}

impl Debug for ArrayShard<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let device = self.device();
        formatter
            .debug_struct("ArrayShard")
            .field("index", &self.index())
            .field("device_id", &device.id)
            .field("process_index", &device.process_index)
            .field("shape", &self.shape())
            .field("is_addressable", &self.is_addressable())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::{DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, MeshDevice, Sharding, ShardingError};

    use super::*;

    fn test_logical_mesh_2x2() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap()
    }

    fn test_device_mesh_2x2() -> DeviceMesh {
        let devices = vec![MeshDevice::new(0, 0), MeshDevice::new(1, 0), MeshDevice::new(2, 1), MeshDevice::new(3, 1)];
        DeviceMesh::new(test_logical_mesh_2x2(), devices).unwrap()
    }

    fn test_static_shape(dimensions: &[usize]) -> StaticShape {
        StaticShape::new(dimensions.to_vec())
    }

    fn shard_for_device(layout: &ShardLayout, device_id: MeshDeviceId) -> &ShardDescriptor {
        let shard_index = layout
            .shard_index_by_device
            .get(&device_id)
            .copied()
            .expect("device should have a shard descriptor");
        &layout.descriptors[shard_index]
    }

    fn shard_indices_for_process(shards: &[ShardDescriptor], process_index: usize) -> Vec<ShardIndex> {
        shards
            .iter()
            .filter_map(|shard| (shard.device.process_index == process_index).then_some(shard.index))
            .collect()
    }

    #[test]
    fn test_shard_descriptor_shape() {
        let descriptor = ShardDescriptor { index: 3, device: MeshDevice::new(7, 2), slice: vec![2..5, 0..4] };

        assert_eq!(descriptor.shape(), test_static_shape(&[3, 4]));
    }

    #[test]
    fn test_array_shard_accessors_and_debug() {
        let descriptor = ShardDescriptor { index: 3, device: MeshDevice::new(7, 2), slice: vec![2..5, 0..4] };
        let shard = ArrayShard { descriptor, buffer: None };

        assert_eq!(shard.index(), 3);
        assert_eq!(shard.device(), MeshDevice::new(7, 2));
        assert_eq!(shard.slice(), &[2..5, 0..4]);
        assert_eq!(shard.shape(), test_static_shape(&[3, 4]));
        assert!(!shard.is_addressable());
        assert_eq!(
            format!("{shard:?}"),
            concat!(
                "ArrayShard { index: 3, device_id: 7, process_index: 2, ",
                "shape: StaticShape { dimensions: [3, 4] }, is_addressable: false }",
            ),
        );
    }

    #[test]
    fn test_shard_layout_rank_mismatch() {
        let logical_mesh = test_logical_mesh_2x2();
        let mesh = test_device_mesh_2x2();
        let sharding =
            Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])])
                .unwrap();

        assert!(matches!(
            ShardLayout::new(&test_static_shape(&[8usize]), &mesh, &sharding),
            Err(ArrayError::ShardingError(ShardingError::ShardingRankMismatch { sharding_rank: 2, array_rank: 1 })),
        ));
    }

    #[test]
    fn test_shard_layout_rejects_invalid_sharded_axes() {
        let mesh = test_device_mesh_2x2();

        let mut empty_axis_sharding = Sharding::replicated(mesh.logical_mesh.clone(), 1);
        empty_axis_sharding.dimensions[0] = ShardingDimension::sharded(Vec::<String>::new());
        assert!(matches!(
            ShardLayout::new(&test_static_shape(&[8]), &mesh, &empty_axis_sharding),
            Err(ArrayError::ShardingError(ShardingError::EmptySharding { dimension: 0 })),
        ));

        let mut unknown_axis_sharding = Sharding::replicated(mesh.logical_mesh.clone(), 1);
        unknown_axis_sharding.dimensions[0] = ShardingDimension::sharded(["z"]);
        assert!(matches!(
            ShardLayout::new(&test_static_shape(&[8]), &mesh, &unknown_axis_sharding),
            Err(ArrayError::ShardingError(ShardingError::UnknownMeshAxisName { name })) if name == "z",
        ));

        let mut duplicate_axis_sharding = Sharding::replicated(mesh.logical_mesh.clone(), 2);
        duplicate_axis_sharding.dimensions[0] = ShardingDimension::sharded(["x"]);
        duplicate_axis_sharding.dimensions[1] = ShardingDimension::sharded(["x"]);
        assert!(matches!(
            ShardLayout::new(&test_static_shape(&[8, 8]), &mesh, &duplicate_axis_sharding),
            Err(ArrayError::ShardingError(ShardingError::DuplicateMeshAxisName { name })) if name == "x",
        ));
    }

    #[test]
    fn test_shard_layout_unconstrained_is_ignored() {
        let logical_mesh = test_logical_mesh_2x2();
        let mesh = test_device_mesh_2x2();
        let sharding =
            Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::unconstrained()])
                .unwrap();
        let layout = ShardLayout::new(&test_static_shape(&[8, 6]), &mesh, &sharding).unwrap();

        let shard0 = shard_for_device(&layout, 0);
        let shard3 = shard_for_device(&layout, 3);

        assert_eq!(shard0.slice[0], 0..4);
        assert_eq!(shard0.slice[1], 0..6);
        assert_eq!(shard3.slice[0], 4..8);
        assert_eq!(shard3.slice[1], 0..6);
        assert_eq!(shard0.shape(), test_static_shape(&[4, 6]));
        assert_eq!(shard3.shape(), test_static_shape(&[4, 6]));
    }

    #[test]
    fn test_shard_layout_even_2d_partitioning() {
        let logical_mesh = test_logical_mesh_2x2();
        let mesh = test_device_mesh_2x2();
        let sharding =
            Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])])
                .unwrap();
        let layout = ShardLayout::new(&test_static_shape(&[8, 6]), &mesh, &sharding).unwrap();

        let shard0 = shard_for_device(&layout, 0);
        let shard3 = shard_for_device(&layout, 3);

        assert_eq!(shard0.shape(), test_static_shape(&[4, 3]));
        assert_eq!(shard0.slice[0], 0..4);
        assert_eq!(shard0.slice[1], 0..3);
        assert_eq!(shard3.shape(), test_static_shape(&[4, 3]));
        assert_eq!(shard3.slice[0], 4..8);
        assert_eq!(shard3.slice[1], 3..6);
    }

    #[test]
    fn test_shard_layout_uneven_partitioning() {
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let devices = vec![MeshDevice::new(0, 0), MeshDevice::new(1, 0)];
        let mesh = DeviceMesh::new(logical_mesh, devices).unwrap();
        let sharding = Sharding::new(mesh.logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let layout = ShardLayout::new(&test_static_shape(&[5]), &mesh, &sharding).unwrap();

        let shard0 = shard_for_device(&layout, 0);
        let shard1 = shard_for_device(&layout, 1);

        assert_eq!(shard0.shape(), test_static_shape(&[3]));
        assert_eq!(shard0.slice[0], 0..3);
        assert_eq!(shard1.shape(), test_static_shape(&[2]));
        assert_eq!(shard1.slice[0], 3..5);
    }

    #[test]
    fn test_shard_layout_multi_axis_single_dimension_partitioning() {
        let logical_mesh = test_logical_mesh_2x2();
        let mesh = test_device_mesh_2x2();
        let sharding =
            Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x".to_string(), "y".to_string()])]).unwrap();
        let layout = ShardLayout::new(&test_static_shape(&[10]), &mesh, &sharding).unwrap();

        assert_eq!(shard_for_device(&layout, 0).slice[0], 0..3);
        assert_eq!(shard_for_device(&layout, 1).slice[0], 3..6);
        assert_eq!(shard_for_device(&layout, 2).slice[0], 6..8);
        assert_eq!(shard_for_device(&layout, 3).slice[0], 8..10);
    }

    #[test]
    fn test_shard_layout_process_filtering() {
        let logical_mesh = test_logical_mesh_2x2();
        let mesh = test_device_mesh_2x2();
        let sharding =
            Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])])
                .unwrap();
        let layout = ShardLayout::new(&test_static_shape(&[8, 6]), &mesh, &sharding).unwrap();

        assert_eq!(shard_indices_for_process(&layout.descriptors, 0), vec![0, 1]);
        assert_eq!(shard_indices_for_process(&layout.descriptors, 1), vec![2, 3]);
        assert_eq!(shard_indices_for_process(&layout.descriptors, 42), Vec::<usize>::new());
    }

    #[test]
    fn test_shard_layout_mesh_mismatch_reports_expected_and_actual_meshes() {
        let logical_mesh = test_logical_mesh_2x2();
        let mesh = test_device_mesh_2x2();
        let actual = LogicalMesh::new(vec![MeshAxis::new("z", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let sharding = Sharding::new(actual.clone(), vec![ShardingDimension::sharded(["z"])]).unwrap();

        assert!(matches!(
            ShardLayout::new(&test_static_shape(&[8]), &mesh, &sharding),
            Err(ArrayError::ShardingError(ShardingError::MeshMismatch { expected, actual: a }))
                if expected == logical_mesh && a == actual,
        ));
    }
}
