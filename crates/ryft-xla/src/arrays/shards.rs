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
    /// Refer to the documentation of [`Self::index`] for information on this field.
    index: ShardIndex,

    /// Refer to the documentation of [`Self::device`] for information on this field.
    device: MeshDevice,

    /// Refer to the documentation of [`Self::slice`] for information on this field.
    slice: Vec<Range<usize>>,
}

impl ShardDescriptor {
    /// Creates a new [`ShardDescriptor`].
    #[inline]
    pub fn new(index: ShardIndex, device: MeshDevice, slice: Vec<Range<usize>>) -> Self {
        Self { index, device, slice }
    }

    /// Returns the [`ShardIndex`] of this [`ShardDescriptor`]. This index is stable across processes and matches this
    /// descriptor's position in [`ShardLayout::descriptors`].
    #[inline]
    pub fn index(&self) -> ShardIndex {
        self.index
    }

    /// Returns the [`MeshDevice`] that owns the shard. Note that ownership does not imply local addressability.
    /// A shard is local to a process when [`MeshDevice::process_index`] matches that process.
    #[inline]
    pub fn device(&self) -> MeshDevice {
        self.device
    }

    /// Returns the per-dimension global index ranges covered by the shard. This vector contains one [`Range`] per array
    /// dimension. Ranges use normal Rust semantics: the start is inclusive and the end is exclusive. Replicated and
    /// unconstrained dimensions span the full dimension range (i.e., `0..dimension_size`) while sharded dimensions span
    /// the contiguous partition selected by this shard's [`MeshDevice`] coordinates in the underlying [`DeviceMesh`].
    #[inline]
    pub fn slice(&self) -> &[Range<usize>] {
        self.slice.as_slice()
    }

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
    /// Refer to the documentation of [`Self::descriptors`] for information on this field.
    descriptors: Vec<ShardDescriptor>,

    /// Refer to the documentation of [`Self::shard_index_by_device`] for information on this field.
    shard_index_by_device: HashMap<MeshDeviceId, ShardIndex>,

    /// Private marker that prevents external struct-literal construction. Construction must go through
    /// [`ShardLayout::new`] so that mesh, rank, and sharding-axis validation is enforced.
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
    ///   - `shape`: [`StaticShape`] of the [`Array`](crate::Array) being sharded/partitioned.
    ///   - `mesh`: [`DeviceMesh`] whose row-major device order determines shard indices.
    ///   - `sharding`: Logical [`Sharding`] specification to apply to `shape`.
    pub fn new(shape: &StaticShape, mesh: &DeviceMesh, sharding: &Sharding) -> Result<Self, ArrayError> {
        if mesh.logical_mesh() != sharding.mesh() {
            return Err(ShardingError::MeshMismatch {
                expected: mesh.logical_mesh().clone(),
                actual: sharding.mesh().clone(),
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
        // allocation-free with respect to axis-name lookup and centralizes defensive validation before any shard
        // descriptors are built.
        let mut used_axis_names = HashSet::new();
        let dimension_axis_indices = sharding
            .dimensions()
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
                        .logical_mesh()
                        .axis_index(axis_name.as_str())
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
        for (index, mesh_device) in mesh.devices().iter().copied().enumerate() {
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
                        let axis_size = mesh.logical_mesh().axes()[axis_index].size();
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

            shard_index_by_device.insert(mesh_device.id(), index);
            descriptors.push(ShardDescriptor::new(index, mesh_device, slices));
        }

        Ok(Self { descriptors, shard_index_by_device, _private: () })
    }

    /// Returns the [`ShardDescriptor`]s for all global shards in row-major [`DeviceMesh`] order. The position of each
    /// descriptor in this vector is the descriptor's [`ShardDescriptor::index`].
    #[inline]
    pub fn descriptors(&self) -> &[ShardDescriptor] {
        &self.descriptors
    }

    /// Returns the lookup table mapping [`MeshDeviceId`]s to their corresponding [`ShardIndex`]es. This is used when a
    /// [`Buffer`] reports a [`MeshDeviceId`] and the [`Array`](crate::Array) constructor needs to find the global shard
    /// metadata that buffer should satisfy.
    #[inline]
    pub fn shard_index_by_device(&self) -> &HashMap<MeshDeviceId, ShardIndex> {
        &self.shard_index_by_device
    }

    /// Consumes this [`ShardLayout`] and returns its [`ShardDescriptor`]s and device-to-shard lookup table.
    #[inline]
    pub(crate) fn into_parts(self) -> (Vec<ShardDescriptor>, HashMap<MeshDeviceId, ShardIndex>) {
        (self.descriptors, self.shard_index_by_device)
    }
}

/// Shard of an [`Array`](crate::Array). [`ArrayShard`]s always carry global shard metadata through
/// [`ArrayShard::descriptor`]. They also carry a PJRT [`Buffer`] when the owning device is addressable from the current
/// process (otherwise [`ArrayShard::buffer`] is set to `None`). This lets an [`Array`](crate::Array) describe its full
/// global layout while storing local buffers only for the shards that the current process can read directly, without
/// moving data around. Note that, [`ArrayShard`]s holds their [`Buffer`]s inside [`Arc`]s so that cloning an array can
/// share addressable PJRT [`Buffer`]s with other array instances. The last [`Arc`] dropped releases the underlying PJRT
/// buffer via [`Buffer`]'s [`Drop`] implementation.
#[derive(Clone)]
pub struct ArrayShard<'o> {
    /// Refer to the documentation of [`Self::descriptor`] for information on this field.
    descriptor: ShardDescriptor,

    /// Refer to the documentation of [`Self::buffer`] for information on this field.
    buffer: Option<Arc<Buffer<'o>>>,
}

impl<'o> ArrayShard<'o> {
    /// Creates a new [`ArrayShard`].
    #[inline]
    pub fn new(descriptor: ShardDescriptor, buffer: Option<Arc<Buffer<'o>>>) -> Self {
        Self { descriptor, buffer }
    }

    /// Returns the [`ShardDescriptor`] of this [`ArrayShard`], which is defined and provided irrespective of whether
    /// this shard is addressable from the current process or not.
    #[inline]
    pub fn descriptor(&self) -> &ShardDescriptor {
        &self.descriptor
    }

    /// Returns the [`Buffer`] underlying this [`ArrayShard`]. This is `None` if the shard is not addressable
    /// from the current process.
    #[inline]
    pub fn buffer(&self) -> Option<&Arc<Buffer<'o>>> {
        self.buffer.as_ref()
    }

    /// Returns this [`ArrayShard`]'s global index.
    #[inline]
    pub fn index(&self) -> ShardIndex {
        self.descriptor.index()
    }

    /// Returns the [`MeshDevice`] that owns this [`ArrayShard`].
    #[inline]
    pub fn device(&self) -> MeshDevice {
        self.descriptor.device()
    }

    /// Returns the [`ShardDescriptor::slice`] covered by this [`ArrayShard`].
    #[inline]
    pub fn slice(&self) -> &[Range<usize>] {
        self.descriptor.slice()
    }

    /// Returns the local [`StaticShape`] of this [`ArrayShard`].
    #[inline]
    pub fn shape(&self) -> StaticShape {
        self.descriptor.shape()
    }

    /// Returns `true` if this shard is addressable from the current process.
    #[inline]
    pub fn is_addressable(&self) -> bool {
        self.buffer.is_some()
    }

    /// Consumes this shard and returns its [`ShardDescriptor`] and addressable [`Buffer`], if any.
    #[inline]
    pub(crate) fn into_parts(self) -> (ShardDescriptor, Option<Arc<Buffer<'o>>>) {
        (self.descriptor, self.buffer)
    }
}

impl Debug for ArrayShard<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let device = self.device();
        formatter
            .debug_struct("ArrayShard")
            .field("index", &self.index())
            .field("device_id", &device.id())
            .field("process_index", &device.process_index())
            .field("shape", &self.shape())
            .field("is_addressable", &self.is_addressable())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;

    use ryft_core::{
        DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, MeshDevice, MeshDeviceId, Sharding, ShardingDimension,
        ShardingError, StaticShape,
    };

    use crate::arrays::ArrayError;
    use crate::tests::device_mesh_2x2;

    use super::{ArrayShard, ShardDescriptor, ShardIndex, ShardLayout};

    fn static_shape(dimensions: &[usize]) -> StaticShape {
        StaticShape::new(dimensions.to_vec())
    }

    fn shard_index_by_device_2x2() -> HashMap<MeshDeviceId, ShardIndex> {
        HashMap::from([(0, 0), (1, 1), (2, 2), (3, 3)])
    }

    #[test]
    fn test_shard_descriptor_accessors() {
        let device = MeshDevice::new(7, 2);
        let descriptor = ShardDescriptor::new(3, device, vec![2..5, 0..4]);

        assert_eq!(descriptor.index(), 3);
        assert_eq!(descriptor.device(), device);
        assert_eq!(descriptor.slice(), [2..5, 0..4].as_slice());
        assert_eq!(descriptor.shape(), static_shape(&[3, 4]));
    }

    #[test]
    fn test_array_shard_accessors() {
        let descriptor = ShardDescriptor::new(3, MeshDevice::new(7, 2), vec![2..5, 0..4]);
        let shard = ArrayShard::new(descriptor.clone(), None);

        assert_eq!(shard.descriptor(), &descriptor);
        assert!(shard.buffer().is_none());
        assert_eq!(shard.index(), 3);
        assert_eq!(shard.device(), MeshDevice::new(7, 2));
        assert_eq!(shard.slice(), [2..5, 0..4].as_slice());
        assert_eq!(shard.shape(), static_shape(&[3, 4]));
        assert!(!shard.is_addressable());

        let (actual_descriptor, buffer) = shard.into_parts();
        assert_eq!(actual_descriptor, descriptor);
        assert!(buffer.is_none());
    }

    #[test]
    fn test_array_shard_debug() {
        let descriptor = ShardDescriptor::new(3, MeshDevice::new(7, 2), vec![2..5, 0..4]);
        let shard = ArrayShard::new(descriptor, None);

        assert_eq!(
            format!("{shard:?}"),
            concat!(
                "ArrayShard { index: 3, device_id: 7, process_index: 2, ",
                "shape: StaticShape { dimensions: [3, 4] }, is_addressable: false }",
            ),
        );
    }

    #[test]
    fn test_shard_layout_evenly_partitions_two_dimensions() {
        let mesh = device_mesh_2x2();
        let shape = static_shape(&[8, 6]);
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
        )
        .unwrap();

        assert_eq!(
            ShardLayout::new(&shape, &mesh, &sharding),
            Ok(ShardLayout {
                descriptors: vec![
                    ShardDescriptor::new(0, MeshDevice::new(0, 0), vec![0..4, 0..3]),
                    ShardDescriptor::new(1, MeshDevice::new(1, 0), vec![0..4, 3..6]),
                    ShardDescriptor::new(2, MeshDevice::new(2, 1), vec![4..8, 0..3]),
                    ShardDescriptor::new(3, MeshDevice::new(3, 1), vec![4..8, 3..6]),
                ],
                shard_index_by_device: shard_index_by_device_2x2(),
                _private: (),
            }),
        );
    }

    #[test]
    fn test_shard_layout_keeps_replicated_and_unconstrained_dimensions_full_size() {
        let mesh = device_mesh_2x2();
        let shape = static_shape(&[8, 6, 5]);
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![
                ShardingDimension::sharded(["x"]),
                ShardingDimension::replicated(),
                ShardingDimension::unconstrained(),
            ],
        )
        .unwrap();

        assert_eq!(
            ShardLayout::new(&shape, &mesh, &sharding),
            Ok(ShardLayout {
                descriptors: vec![
                    ShardDescriptor::new(0, MeshDevice::new(0, 0), vec![0..4, 0..6, 0..5]),
                    ShardDescriptor::new(1, MeshDevice::new(1, 0), vec![0..4, 0..6, 0..5]),
                    ShardDescriptor::new(2, MeshDevice::new(2, 1), vec![4..8, 0..6, 0..5]),
                    ShardDescriptor::new(3, MeshDevice::new(3, 1), vec![4..8, 0..6, 0..5]),
                ],
                shard_index_by_device: shard_index_by_device_2x2(),
                _private: (),
            }),
        );
    }

    #[test]
    fn test_shard_layout_partitions_uneven_dimension() {
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![MeshDevice::new(0, 0), MeshDevice::new(1, 0)]).unwrap();
        let shape = static_shape(&[5]);
        let sharding = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();

        assert_eq!(
            ShardLayout::new(&shape, &mesh, &sharding),
            Ok(ShardLayout {
                descriptors: vec![
                    ShardDescriptor::new(0, MeshDevice::new(0, 0), vec![0..3]),
                    ShardDescriptor::new(1, MeshDevice::new(1, 0), vec![3..5]),
                ],
                shard_index_by_device: HashMap::from([(0, 0), (1, 1)]),
                _private: (),
            }),
        );
    }

    #[test]
    fn test_shard_layout_partitions_single_dimension_over_multiple_mesh_axes() {
        let mesh = device_mesh_2x2();
        let shape = static_shape(&[10]);
        let sharding =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x", "y"])]).unwrap();

        assert_eq!(
            ShardLayout::new(&shape, &mesh, &sharding),
            Ok(ShardLayout {
                descriptors: vec![
                    ShardDescriptor::new(0, MeshDevice::new(0, 0), vec![0..3]),
                    ShardDescriptor::new(1, MeshDevice::new(1, 0), vec![3..6]),
                    ShardDescriptor::new(2, MeshDevice::new(2, 1), vec![6..8]),
                    ShardDescriptor::new(3, MeshDevice::new(3, 1), vec![8..10]),
                ],
                shard_index_by_device: shard_index_by_device_2x2(),
                _private: (),
            }),
        );
    }

    #[test]
    fn test_shard_layout_into_parts() {
        let mesh = device_mesh_2x2();
        let shape = static_shape(&[8, 6]);
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::replicated()],
        )
        .unwrap();
        let layout = ShardLayout::new(&shape, &mesh, &sharding).unwrap();

        assert_eq!(
            layout.into_parts(),
            (
                vec![
                    ShardDescriptor::new(0, MeshDevice::new(0, 0), vec![0..8, 0..6]),
                    ShardDescriptor::new(1, MeshDevice::new(1, 0), vec![0..8, 0..6]),
                    ShardDescriptor::new(2, MeshDevice::new(2, 1), vec![0..8, 0..6]),
                    ShardDescriptor::new(3, MeshDevice::new(3, 1), vec![0..8, 0..6]),
                ],
                shard_index_by_device_2x2(),
            ),
        );
    }

    #[test]
    fn test_shard_layout_rejects_mesh_mismatch() {
        let mesh = device_mesh_2x2();
        let expected_mesh = mesh.logical_mesh().clone();
        let actual_mesh = LogicalMesh::new(vec![MeshAxis::new("z", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let shape = static_shape(&[8]);
        let sharding = Sharding::new(actual_mesh.clone(), vec![ShardingDimension::sharded(["z"])]).unwrap();

        let error = ShardLayout::new(&shape, &mesh, &sharding).unwrap_err();
        assert_eq!(
            error,
            ArrayError::ShardingError(ShardingError::MeshMismatch {
                expected: expected_mesh.clone(),
                actual: actual_mesh.clone(),
            }),
        );
        let expected_message = format!("mesh mismatch; expected '{expected_mesh:?}' but got '{actual_mesh:?}'");
        assert_eq!(error.to_string(), expected_message);
    }

    #[test]
    fn test_shard_layout_rejects_rank_mismatch() {
        let mesh = device_mesh_2x2();
        let shape = static_shape(&[8]);
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
        )
        .unwrap();

        let error = ShardLayout::new(&shape, &mesh, &sharding).unwrap_err();
        assert_eq!(
            error,
            ArrayError::ShardingError(ShardingError::ShardingRankMismatch { sharding_rank: 2, array_rank: 1 }),
        );
        assert_eq!(error.to_string(), "sharding rank (2) does not match array rank (1)");
    }
}
