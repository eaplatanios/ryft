use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

use ryft_core::{DeviceMesh, MeshDevice, MeshDeviceId, Sharding, ShardingDimension, ShardingError, StaticShape};
use ryft_pjrt::Buffer;

use crate::arrays::ArrayError;

/// Row-major ordinal of a global shard within a device mesh.
pub type ShardIndex = usize;

/// Pure metadata for one global shard of an [`Array`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardDescriptor {
    /// Global (ordinal) index of this shard in a row-major device mesh ordering.
    pub index: ShardIndex,

    /// [`MeshDevice`] that owns this shard.
    pub device: MeshDevice,

    /// Per-dimension ranges describing the portion of the corresponding global array that this shard corresponds to.
    /// Each range describes which contiguous range of elements along a single array dimension this shard owns. For a
    /// replicated dimension, the slice spans the full extent of that dimension, `[0, dimension_size)`. For a sharded
    /// dimension, the slice covers the partition assigned to a specific device based on its mesh coordinates.
    pub slice: Vec<Range<usize>>,
}

impl ShardDescriptor {
    /// Logical shape of this shard, derived from the per-dimension [`ShardDescriptor::slice`] ranges.
    #[inline]
    pub fn shape(&self) -> StaticShape {
        StaticShape::new(self.slice.iter().map(|slice| slice.len()).collect())
    }
}

/// Shard descriptors and lookup tables implied by a shape, mesh, and sharding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardLayout {
    /// Descriptors for all global shards in mesh order.
    pub descriptors: Vec<ShardDescriptor>,

    /// Lookup table from device id to the corresponding descriptor index.
    pub shard_index_by_device: HashMap<MeshDeviceId, ShardIndex>,

    /// Private marker that prevents external struct-literal construction.
    _private: (),
}

impl ShardLayout {
    /// Computes one [`ShardDescriptor`] per mesh device for the provided static global shape and [`Sharding`].
    pub fn new(global_shape: &StaticShape, mesh: &DeviceMesh, sharding: &Sharding) -> Result<Self, ArrayError> {
        if mesh.logical_mesh != sharding.mesh {
            return Err(ShardingError::MeshMismatch {
                expected: mesh.logical_mesh.clone(),
                actual: sharding.mesh.clone(),
            }
            .into());
        }

        let partition_rank = sharding.rank();
        let array_rank = global_shape.rank();
        if partition_rank != array_rank {
            return Err(ShardingError::ShardingRankMismatch { sharding_rank: partition_rank, array_rank }.into());
        }

        let global_dimensions = global_shape.as_slice();
        let mut descriptors = Vec::with_capacity(mesh.device_count());
        let mut shard_index_by_device = HashMap::with_capacity(mesh.device_count());
        for (index, mesh_device) in mesh.devices.iter().copied().enumerate() {
            let device_coordinates =
                mesh.device_coordinates(index).expect("mesh coordinate should exist for valid mesh device index");

            let mut slices = Vec::with_capacity(global_dimensions.len());
            for (dimension, dimension_size) in global_dimensions.iter().copied().enumerate() {
                let range = match &sharding.dimensions[dimension] {
                    ShardingDimension::Replicated => 0..dimension_size,
                    ShardingDimension::Sharded(axis_names) => {
                        let mut partition_index = 0usize;
                        let mut partition_count = 1usize;
                        for axis_name in axis_names {
                            let axis_index = mesh
                                .logical_mesh
                                .axis_indices
                                .get(axis_name.as_str())
                                .copied()
                                .expect("sharding mesh axes should be validated before building shard slices");
                            let axis_size = mesh.logical_mesh.axes[axis_index].size;
                            let axis_coordinate = device_coordinates[axis_index];

                            partition_index = partition_index * axis_size + axis_coordinate;
                            partition_count *= axis_size;
                        }

                        let base_size = dimension_size / partition_count;
                        let remainder = dimension_size % partition_count;
                        let extra_before = partition_index.min(remainder);

                        let start = partition_index * base_size + extra_before;
                        let size = base_size + usize::from(partition_index < remainder);
                        start..start + size
                    }
                    ShardingDimension::Unconstrained => 0..dimension_size,
                };
                slices.push(range);
            }

            shard_index_by_device.insert(mesh_device.id, index);
            descriptors.push(ShardDescriptor { index, device: mesh_device, slice: slices });
        }

        Ok(Self { descriptors, shard_index_by_device, _private: () })
    }
}

/// One global shard of an [`Array`].
///
/// Each shard corresponds to one device in a [`DeviceMesh`] and describes the portion of the global
/// array that that device holds. When [`buffer`](Self::buffer) is `Some(_)`, the shard is
/// addressable from the current process and corresponds to one entry in JAX's
/// `array.addressable_shards`.
///
/// [`ArrayShard`] holds its [`Buffer`] inside an [`Arc`] so that [`Array::clone`] can cheaply
/// share addressable PJRT handles with other [`Array`] instances. The last [`Arc`] dropped
/// releases the underlying PJRT buffer via [`Buffer`]'s [`Drop`] implementation. This mirrors
/// the reference-counted array pattern that IFRT uses above PJRT.
#[derive(Clone)]
pub struct ArrayShard<'o> {
    /// Pure global shard metadata.
    pub descriptor: ShardDescriptor,

    /// Reference-counted local PJRT buffer for this shard, or `None` if the shard is not
    /// addressable from the current process. Cloning an [`ArrayShard`] bumps the [`Arc`]
    /// refcount rather than copying device memory.
    pub buffer: Option<Arc<Buffer<'o>>>,
}

impl<'o> ArrayShard<'o> {
    /// Global (ordinal) index of this shard in row-major device mesh order.
    #[inline]
    pub fn index(&self) -> ShardIndex {
        self.descriptor.index
    }

    /// [`MeshDevice`] that owns this shard.
    #[inline]
    pub fn device(&self) -> MeshDevice {
        self.descriptor.device
    }

    /// Per-dimension ranges describing the global array slice owned by this shard.
    #[inline]
    pub fn slice(&self) -> &[Range<usize>] {
        self.descriptor.slice.as_slice()
    }

    /// Logical shape of this shard, derived from the per-dimension [`ShardDescriptor::slice`] ranges.
    #[inline]
    pub fn shape(&self) -> StaticShape {
        self.descriptor.shape()
    }

    /// Returns whether this shard has a local PJRT buffer addressable from the current process.
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
