use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

use ryft_core::{
    Device, DeviceId, DeviceMesh, Sharding, ShardingDimension, ShardingError, StaticShape, check_sharding,
};
use ryft_pjrt::Buffer;

use crate::Error;
use crate::arrays::ArrayError;

/// Shard of an [`Array`](crate::Array). [`ArrayShard`]s always carry global shard metadata through
/// [`ArrayShard::descriptor`]. They also carry either a PJRT [`Buffer`] when ordinary data is addressable from the
/// current process, a bufferless marker for the unique value of [`DataType::Zero`](ryft_core::DataType::Zero), or
/// metadata only when the shard belongs to another process. This lets an array describe its full global layout
/// without allocating meaningless zero-space buffers. Ordinary addressable buffers remain stored inside [`Arc`]s
/// so that cloned arrays can share them.
#[derive(Clone)]
pub struct ArrayShard {
    /// Refer to the documentation of [`Self::descriptor`] for information on this field.
    descriptor: ShardDescriptor,

    /// Physical storage state for this [`ArrayShard`].
    storage: ArrayShardStorage,
}

impl ArrayShard {
    /// Creates a new [`ArrayShard`] described by `descriptor`. `buffer` holds the shard's storage when the shard is
    /// addressable from the current process, and [`None`] creates a metadata-only remote shard.
    #[inline]
    pub fn new(descriptor: ShardDescriptor, buffer: Option<Arc<Buffer<'static>>>) -> Self {
        let storage = buffer.map_or(ArrayShardStorage::Remote, ArrayShardStorage::Buffer);
        Self { descriptor, storage }
    }

    /// Creates a process-local bufferless [`DataType::Zero`](ryft_core::DataType::Zero)-valued [`ArrayShard`].
    #[inline]
    pub fn zero(descriptor: ShardDescriptor) -> Self {
        Self { descriptor, storage: ArrayShardStorage::Zero }
    }

    /// Returns the [`ShardDescriptor`] of this [`ArrayShard`], which is defined and provided irrespective of whether
    /// this shard is addressable from the current process or not.
    #[inline]
    pub fn descriptor(&self) -> &ShardDescriptor {
        &self.descriptor
    }

    /// Returns the [`ArrayShardStorage`] of this [`ArrayShard`], which is what the current process holds for it.
    #[inline]
    pub fn storage(&self) -> &ArrayShardStorage {
        &self.storage
    }

    /// Returns the [`Buffer`] underlying this [`ArrayShard`]. This is `None` if the shard is remote or is an
    /// addressable bufferless [`DataType::Zero`](ryft_core::DataType::Zero)-valued shard.
    #[inline]
    pub fn buffer(&self) -> Option<&Arc<Buffer<'static>>> {
        match &self.storage {
            ArrayShardStorage::Buffer(buffer) => Some(buffer),
            ArrayShardStorage::Remote | ArrayShardStorage::Zero => None,
        }
    }

    /// Returns this [`ArrayShard`]'s [`ShardDescriptor::index`].
    #[inline]
    pub fn index(&self) -> ShardIndex {
        self.descriptor.index()
    }

    /// Returns this [`ArrayShard`]'s [`ShardDescriptor::device`].
    #[inline]
    pub fn device(&self) -> Device {
        self.descriptor.device()
    }

    /// Returns this [`ArrayShard`]'s [`ShardDescriptor::slice`].
    #[inline]
    pub fn slice(&self) -> &[Range<usize>] {
        self.descriptor.slice()
    }

    /// Returns this [`ArrayShard`]'s [`ShardDescriptor::shape`].
    #[inline]
    pub fn shape(&self) -> StaticShape {
        self.descriptor.shape()
    }

    /// Returns `true` if this shard is addressable from the current process.
    #[inline]
    pub fn is_addressable(&self) -> bool {
        !matches!(self.storage, ArrayShardStorage::Remote)
    }

    /// Copies the bytes that the device of this [`ArrayShard`] holds to the host and returns them in dense row-major
    /// order, irrespective of the physical layout of the shard's [`Buffer`]. This function blocks the current thread
    /// until the copy completes. Bufferless [`DataType::Zero`](ryft_core::DataType::Zero) shards and shards with no
    /// elements produce no bytes and issue no copy.
    ///
    /// These bytes are a tile of the [`Array`](crate::Array) that this shard belongs to, unless the [`Sharding`] of
    /// that array has unreduced axes, in which case they are the separate value that this device holds for that tile
    /// (e.g., a summand). This function also observes only this shard's buffer. Use
    /// [`Array::copy_to_host`](crate::Array::copy_to_host) to read the whole value
    /// of an array and to surface the errors of the whole execution that produced it.
    ///
    /// Note that a shard belongs to no session, so the copy is not registered with one. It is complete when this
    /// function returns, and [`Session::synchronize`](crate::Session::synchronize) neither waits for it nor reports
    /// its failure.
    ///
    /// # Errors
    ///
    /// Returns [`ArrayError::MissingShardBuffer`] when this shard lives in another process. PJRT transfer failures
    /// are returned unchanged.
    pub fn copy_to_host_bytes(&self) -> Result<Vec<u8>, Error> {
        let buffer = match &self.storage {
            ArrayShardStorage::Buffer(buffer) => buffer,
            ArrayShardStorage::Zero => return Ok(Vec::new()),
            ArrayShardStorage::Remote => {
                return Err(Error::from(ArrayError::MissingShardBuffer {
                    shard_index: self.index(),
                    device_id: self.device().id(),
                }));
            }
        };

        // Avoid asking PJRT for a zero byte copy of a dynamically sized buffer, whose size query fallback can expose
        // the bound allocation and its trailer instead of the empty logical value.
        let shape = self.shape();
        if shape.as_slice().contains(&0) {
            return Ok(Vec::new());
        }

        // Request row-major bytes explicitly, because the device buffer may have another physical layout.
        let copy = buffer.copy_to_host(Some(ryft_pjrt::Layout::dense_major_to_minor(shape.rank())))?;
        Ok(copy.r#await()?)
    }
}

impl Debug for ArrayShard {
    #[inline]
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

/// Physical storage state of an [`ArrayShard`], which [`ArrayShard::storage`] returns.
#[derive(Clone)]
pub enum ArrayShardStorage {
    /// The shard belongs to another process and has no process-local storage.
    Remote,

    /// The shard is process-local but represents the unique value of
    /// [`DataType::Zero`](ryft_core::DataType::Zero) without an underlying buffer.
    Zero,

    /// The shard is process-local and backed by a PJRT [`Buffer`].
    Buffer(Arc<Buffer<'static>>),
}

// TODO(eaplatanios): Review from here onwards.

/// Row-major ordinal index of an [`ArrayShard`] within a [`DeviceMesh`]. Shard indices are assigned using the same
/// row-major ordering as [`DeviceMesh::devices`]. This gives all processes a stable way to refer to the same global
/// shard without depending on whether that shard is locally addressable or not.
pub type ShardIndex = usize;

/// Device ownership and slice metadata for an [`ArrayShard`]. A [`ShardDescriptor`] is intentionally independent of
/// any local [`Buffer`]. It describes which [`Device`] owns the [`ArrayShard`] and which slice of the underlying
/// [`Array`](crate::Array) that shard represents. [`ArrayShard`]s pair this metadata with optional addressable
/// buffers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardDescriptor {
    /// Refer to the documentation of [`Self::index`] for information on this field.
    index: ShardIndex,

    /// Refer to the documentation of [`Self::device`] for information on this field.
    device: Device,

    /// Refer to the documentation of [`Self::slice`] for information on this field. It is shared, so cloning a
    /// descriptor (e.g., into every output [`Array`](crate::Array) of a compiled program) does not allocate.
    slice: Arc<[Range<usize>]>,
}

impl ShardDescriptor {
    /// Creates a new [`ShardDescriptor`].
    #[inline]
    pub fn new(index: ShardIndex, device: Device, slice: Vec<Range<usize>>) -> Self {
        Self { index, device, slice: slice.into() }
    }

    /// Returns the [`ShardIndex`] of this [`ShardDescriptor`]. This index is stable across processes and matches this
    /// descriptor's position in [`ShardLayout::descriptors`].
    #[inline]
    pub fn index(&self) -> ShardIndex {
        self.index
    }

    /// Returns the [`Device`] that owns the shard. Note that ownership does not imply local addressability.
    /// A shard is local to a process when [`Device::process_index`] matches that process.
    #[inline]
    pub fn device(&self) -> Device {
        self.device
    }

    /// Returns the per-dimension global index ranges covered by the shard. This vector contains one [`Range`] per array
    /// dimension. Ranges use normal Rust semantics, so the start is inclusive and the end is exclusive. Replicated and
    /// unconstrained dimensions span the full dimension range (i.e., `0..dimension_size`) while sharded dimensions span
    /// the contiguous partition selected by this shard's [`Device`] coordinates in the underlying [`DeviceMesh`].
    #[inline]
    pub fn slice(&self) -> &[Range<usize>] {
        &self.slice
    }

    /// Returns the local [`StaticShape`] of the shard described by this [`ShardDescriptor`]. The returned shape is
    /// derived purely from static metadata. It does not inspect any device buffers and is valid for both addressable
    /// and non-addressable shards.
    #[inline]
    pub fn shape(&self) -> StaticShape {
        StaticShape::new(self.slice.iter().map(|slice| slice.len()).collect())
    }

    /// Returns the outermost dimension `d` such that, in a row-major buffer representing the storage of an array with
    /// the provided [`StaticShape`], the elements covered by this [`ShardDescriptor`]'s [`Self::slice`] over dimensions
    /// `d..rank` form a single contiguous span, together with the element count of that contiguous block.
    ///
    /// Trailing dimensions whose slice fully covers the corresponding `shape` extent are absorbed into the contiguous
    /// block. The innermost dimension is always part of the block since its element span at fixed outer indices is
    /// already contiguous in row-major order. Replicated layouts and layouts sharded only along the outermost axis
    /// therefore collapse to a single block, while layouts sharded along an inner axis stop the collapse early. Rank-0
    /// shards trivially return `(0, 1)` since their singular element is its own contiguous block. A returned dimension
    /// of `0` therefore means that the whole shard is one contiguous span, which lets
    /// [`Array::from_host_buffer`](crate::Array::from_host_buffer) upload such a shard from the dense host buffer
    /// without byte strides. Otherwise, each block can be copied with one [`Vec::extend_from_slice`] call.
    ///
    /// Consider the following two shard slices over the same `shape = [4, 6]`:
    ///
    /// ```text
    ///   slice = [1..3, 0..6]            slice = [0..4, 2..5]
    ///   returns (0, 12)                 returns (1, 3)
    ///
    ///   . . . . . .                     . . X X X .
    ///   X X X X X X                     . . X X X .
    ///   X X X X X X                     . . X X X .
    ///   . . . . . .                     . . X X X .
    ///
    ///   one contiguous block            four contiguous blocks
    ///   of 12 elements                  of 3 elements each
    /// ```
    ///
    /// The left slice fully covers the inner dimension (i.e., `0..6` over a host extent of `6`), and so the outer
    /// dimension is absorbed into the block, and the whole shard is one contiguous span. The right slice does not cover
    /// the inner dimension, and so each row of the slice becomes its own contiguous block of `3` elements.
    ///
    /// # Parameters
    ///
    ///   - `shape`: [`StaticShape`] of the contiguous buffer that this [`ShardDescriptor`]'s [`Self::slice`] refers to.
    ///     Must have the same rank as [`Self::slice`].
    ///
    /// # Errors
    ///
    /// Returns [`ArrayError::ShardSliceRankMismatch`] if `shape` has a different rank than [`Self::slice`],
    /// or [`Error::SizeLimitExceeded`] if the absorbed block element count overflows [`usize`].
    pub fn contiguous_inner_block(&self, shape: &StaticShape) -> Result<(usize, usize), Error> {
        let rank = self.slice.len();

        if shape.rank() != rank {
            return Err(Error::from(ArrayError::ShardSliceRankMismatch { shape_rank: shape.rank(), slice_rank: rank }));
        }

        if rank == 0 {
            return Ok((0, 1));
        }

        let mut dimension = rank - 1;
        let mut element_count = self.slice[dimension].len();
        while dimension > 0 && self.slice[dimension].len() == shape[dimension] {
            dimension -= 1;
            element_count =
                element_count.checked_mul(self.slice[dimension].len()).ok_or_else(|| Error::SizeLimitExceeded {
                    message: format!(
                        "contiguous shard block element count for shard slice `{:?}` and shape `{shape}` exceeds the \
                         maximum allowed size of {}",
                        self.slice,
                        usize::MAX,
                    ),
                })?;
        }
        Ok((dimension, element_count))
    }
}

/// [`ShardLayout`] that is obtained by applying a [`Sharding`] to a [`StaticShape`] over a [`DeviceMesh`].
/// [`ShardLayout`] is the bridge between type-level device placement metadata and runtime array metadata. It expands
/// the logical sharding specification into one [`ShardDescriptor`] per [`Device`], in the row-major device order that
/// a [`ShardIndex`] refers to, and also records a device ID lookup table for routing local [`Buffer`]s back to their
/// global shard descriptors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardLayout {
    /// Refer to the documentation of [`Self::descriptors`] for information on this field.
    descriptors: Vec<ShardDescriptor>,

    /// Refer to the documentation of [`Self::shard_index_by_device`] for information on this field.
    shard_index_by_device: HashMap<DeviceId, ShardIndex>,
}

impl ShardLayout {
    /// Constructs a new [`ShardLayout`] by applying the provided [`Sharding`] to the provided [`StaticShape`]
    /// over the provided [`DeviceMesh`]. The returned layout contains one [`ShardDescriptor`] per [`Device`].
    /// For each array dimension, replicated and unconstrained dimensions map every device to the full dimension range,
    /// while sharded dimensions are split across the product of the referenced mesh axes. If the dimension size is not
    /// evenly divisible by the partition count, earlier partitions receive one extra element.
    ///
    /// # Parameters
    ///
    ///   - `shape`: [`StaticShape`] of the [`Array`](crate::Array) being sharded.
    ///   - `mesh`: [`DeviceMesh`] whose row-major device order determines shard indices.
    ///   - `sharding`: Logical [`Sharding`] specification to apply to `shape`.
    pub fn new(shape: &StaticShape, mesh: &DeviceMesh, sharding: &Sharding) -> Result<Self, Error> {
        check_sharding!(mesh, sharding);

        let sharding_rank = sharding.rank();
        let array_rank = shape.rank();
        if sharding_rank != array_rank {
            return Err(ShardingError::ShardingRankMismatch { sharding_rank, array_rank }.into());
        }

        let dimensions = shape.as_slice();

        // Resolve each sharded array dimension to mesh axis indices. This keeps axis name lookups and their allocations
        // out of the inner per-device loop and centralizes defensive validation before any shard descriptors are
        // built.
        let mut used_axis_names = HashSet::new();
        let dimension_axis_indices = sharding
            .dimensions()
            .iter()
            .enumerate()
            .map(|(dimension, sharding_dimension)| -> Result<Vec<usize>, Error> {
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
        for (index, device) in mesh.devices().iter().copied().enumerate() {
            // `DeviceMesh` stores devices in row-major mesh order, so the loop index is also the global shard index.
            // Convert it back into mesh coordinates so sharded dimensions can pick the partition owned by this device.
            let device_coordinates = mesh.device_coordinates(index).unwrap();

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

            shard_index_by_device.insert(device.id(), index);
            descriptors.push(ShardDescriptor::new(index, device, slices));
        }

        Ok(Self { descriptors, shard_index_by_device })
    }

    /// Returns the [`ShardDescriptor`]s for all global shards in row-major [`DeviceMesh`] order. The position of each
    /// descriptor in this vector is the descriptor's [`ShardDescriptor::index`].
    #[inline]
    pub fn descriptors(&self) -> &[ShardDescriptor] {
        &self.descriptors
    }

    /// Returns the lookup table mapping [`DeviceId`]s to their corresponding [`ShardIndex`]es. This is used when a
    /// [`Buffer`] reports a [`DeviceId`] and the [`Array`](crate::Array) constructor needs to find the global shard
    /// metadata that buffer should satisfy.
    #[inline]
    pub fn shard_index_by_device(&self) -> &HashMap<DeviceId, ShardIndex> {
        &self.shard_index_by_device
    }

    /// Consumes this [`ShardLayout`] and returns its [`ShardDescriptor`]s and its lookup table from devices to shards.
    #[inline]
    pub(crate) fn into_parts(self) -> (Vec<ShardDescriptor>, HashMap<DeviceId, ShardIndex>) {
        (self.descriptors, self.shard_index_by_device)
    }
}

// Per-axis tile ranges are lists that often hold a single range, which this lint mistakes for a range to collect.
#[allow(clippy::single_range_in_vec_init)]
#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::{ArrayType, DataType, Error as CoreError, LogicalMesh, MeshAxis, MeshAxisType};

    use crate::Session;
    use crate::arrays::Array;
    use crate::sessions::tests::domain_mesh;
    use crate::tests::{
        cpu_client, device_mesh_2x2, logical_mesh_2x2, replicated_array, values_from_bytes, values_to_bytes,
    };

    use super::*;

    #[test]
    fn test_array_shard_new() {
        let descriptor = ShardDescriptor::new(3, Device::new(7, 2), vec![2..5, 0..4]);
        let shard = ArrayShard::new(descriptor.clone(), None);

        assert_eq!(shard.descriptor(), &descriptor);
        assert!(shard.buffer().is_none());
        assert_eq!(shard.index(), 3);
        assert_eq!(shard.device(), Device::new(7, 2));
        assert_eq!(shard.slice(), [2..5, 0..4].as_slice());
        assert_eq!(shard.shape(), StaticShape::new(vec![3, 4]));
        assert!(!shard.is_addressable());
    }

    #[test]
    fn test_array_shard_storage() {
        // A shard without a buffer lives in another process, and a zero shard is local without holding a buffer.
        let descriptor = ShardDescriptor::new(0, Device::new(0, 0), vec![0..2]);
        assert!(matches!(ArrayShard::new(descriptor.clone(), None).storage(), ArrayShardStorage::Remote));
        assert!(matches!(ArrayShard::zero(descriptor).storage(), ArrayShardStorage::Zero));

        // A shard of an uploaded array holds the buffer that `ArrayShard::buffer` returns.
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let array = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.]));
        let shard = &array.shards()[0];
        assert!(matches!(
            shard.storage(),
            ArrayShardStorage::Buffer(buffer) if Arc::ptr_eq(buffer, shard.buffer().unwrap()),
        ));
    }

    #[test]
    fn test_array_shard_copy_to_host() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);

        // A buffer-backed shard holds only its own tile of the array.
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4]),
            mesh.clone(),
            Some(sharding),
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        assert_eq!(values_from_bytes::<f32>(&array.shards()[1].copy_to_host_bytes().unwrap()), vec![3f32, 4.]);

        // Bufferless zero shards and buffer-backed shards with no elements produce no bytes.
        let zero = replicated_array(&session, &mesh, DataType::Zero, &[3], &[]);
        assert!(zero.shards()[0].buffer().is_none());
        assert_eq!(zero.shards()[0].copy_to_host_bytes().unwrap(), Vec::<u8>::new());
        let empty = replicated_array(&session, &mesh, DataType::F32, &[0], &[]);
        assert!(empty.shards()[0].buffer().is_some());
        assert_eq!(empty.shards()[0].copy_to_host_bytes().unwrap(), Vec::<u8>::new());

        // A remote shard has no buffer in the current process.
        let remote = ArrayShard::new(ShardDescriptor::new(3, Device::new(7, 2), vec![2..5, 0..4]), None);
        let error = remote.copy_to_host_bytes().unwrap_err();
        assert!(matches!(
            &error,
            Error::Array(error) if matches!(error.as_ref(),
                ArrayError::MissingShardBuffer { shard_index: 3, device_id: 7 },
            ),
        ));
        assert_eq!(error.to_string(), "array shard #3 on device 7 has no addressable buffer");
    }

    #[test]
    fn test_array_shard_debug() {
        let descriptor = ShardDescriptor::new(3, Device::new(7, 2), vec![2..5, 0..4]);
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
    fn test_shard_descriptor_new() {
        let device = Device::new(7, 2);
        let descriptor = ShardDescriptor::new(3, device, vec![2..5, 0..4]);

        assert_eq!(descriptor.index(), 3);
        assert_eq!(descriptor.device(), device);
        assert_eq!(descriptor.slice(), [2..5, 0..4].as_slice());
        assert_eq!(descriptor.shape(), StaticShape::new(vec![3, 4]));
    }

    #[test]
    fn test_shard_descriptor_contiguous_inner_block() {
        let device = Device::new(7, 2);
        let descriptor = ShardDescriptor::new(3, device, vec![2..5, 0..4]);

        // The inner dimension is not fully covered by the host extent, and so the block stops at the innermost
        // dimension and only spans `slice[1].len()` elements.
        assert_eq!(descriptor.contiguous_inner_block(&StaticShape::new(vec![8, 5])).unwrap(), (1, 4));

        // When the inner dimension is fully covered, the block absorbs the outer dimension and spans
        // `slice[0].len() * slice[1].len()` elements.
        assert_eq!(descriptor.contiguous_inner_block(&StaticShape::new(vec![8, 4])).unwrap(), (0, 12));

        // A shape whose rank does not match the descriptor's slice rank reports a rank mismatch error.
        let error = descriptor.contiguous_inner_block(&StaticShape::new(vec![8, 4, 2])).unwrap_err();
        assert!(matches!(
            &error,
            Error::Array(error) if matches!(error.as_ref(),
                ArrayError::ShardSliceRankMismatch { shape_rank: 3, slice_rank: 2 },
            ),
        ));
        assert_eq!(error.to_string(), "shape rank 3 does not match shard slice rank 2");

        // A rank-1 descriptor always reports a single block over its slice.
        let descriptor = ShardDescriptor::new(0, device, vec![2..7]);
        assert_eq!(descriptor.contiguous_inner_block(&StaticShape::new(vec![10])).unwrap(), (0, 5));

        // A rank-3 descriptor whose two innermost dimensions are fully covered collapses to a single block
        // over all three dimensions.
        let descriptor = ShardDescriptor::new(0, device, vec![1..2, 0..3, 0..4]);
        assert_eq!(descriptor.contiguous_inner_block(&StaticShape::new(vec![5, 3, 4])).unwrap(), (0, 12));

        // A scalar (i.e., rank-0) descriptor trivially reports a single block with one element.
        let descriptor = ShardDescriptor::new(0, device, vec![]);
        assert_eq!(descriptor.contiguous_inner_block(&StaticShape::new(vec![])).unwrap(), (0, 1));
    }

    #[test]
    fn test_shard_layout_new_evenly_partitions_two_dimensions() {
        let mesh = device_mesh_2x2();
        let shape = StaticShape::new(vec![8, 6]);
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
        )
        .unwrap();

        assert_eq!(
            ShardLayout::new(&shape, &mesh, &sharding).unwrap(),
            ShardLayout {
                descriptors: vec![
                    ShardDescriptor::new(0, Device::new(0, 0), vec![0..4, 0..3]),
                    ShardDescriptor::new(1, Device::new(1, 0), vec![0..4, 3..6]),
                    ShardDescriptor::new(2, Device::new(2, 1), vec![4..8, 0..3]),
                    ShardDescriptor::new(3, Device::new(3, 1), vec![4..8, 3..6]),
                ],
                shard_index_by_device: HashMap::from([(0, 0), (1, 1), (2, 2), (3, 3)]),
            },
        );
    }

    #[test]
    fn test_shard_layout_new_keeps_replicated_and_unconstrained_dimensions_full_size() {
        let mesh = device_mesh_2x2();
        let shape = StaticShape::new(vec![8, 6, 5]);
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
            ShardLayout::new(&shape, &mesh, &sharding).unwrap(),
            ShardLayout {
                descriptors: vec![
                    ShardDescriptor::new(0, Device::new(0, 0), vec![0..4, 0..6, 0..5]),
                    ShardDescriptor::new(1, Device::new(1, 0), vec![0..4, 0..6, 0..5]),
                    ShardDescriptor::new(2, Device::new(2, 1), vec![4..8, 0..6, 0..5]),
                    ShardDescriptor::new(3, Device::new(3, 1), vec![4..8, 0..6, 0..5]),
                ],
                shard_index_by_device: HashMap::from([(0, 0), (1, 1), (2, 2), (3, 3)]),
            },
        );
    }

    #[test]
    fn test_shard_layout_new_partitions_uneven_dimension() {
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![Device::new(0, 0), Device::new(1, 0)]).unwrap();
        let shape = StaticShape::new(vec![5]);
        let sharding = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();

        assert_eq!(
            ShardLayout::new(&shape, &mesh, &sharding).unwrap(),
            ShardLayout {
                descriptors: vec![
                    ShardDescriptor::new(0, Device::new(0, 0), vec![0..3]),
                    ShardDescriptor::new(1, Device::new(1, 0), vec![3..5]),
                ],
                shard_index_by_device: HashMap::from([(0, 0), (1, 1)]),
            },
        );
    }

    #[test]
    fn test_shard_layout_new_partitions_single_dimension_over_multiple_mesh_axes() {
        let mesh = device_mesh_2x2();
        let shape = StaticShape::new(vec![10]);
        let sharding =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x", "y"])]).unwrap();

        assert_eq!(
            ShardLayout::new(&shape, &mesh, &sharding).unwrap(),
            ShardLayout {
                descriptors: vec![
                    ShardDescriptor::new(0, Device::new(0, 0), vec![0..3]),
                    ShardDescriptor::new(1, Device::new(1, 0), vec![3..6]),
                    ShardDescriptor::new(2, Device::new(2, 1), vec![6..8]),
                    ShardDescriptor::new(3, Device::new(3, 1), vec![8..10]),
                ],
                shard_index_by_device: HashMap::from([(0, 0), (1, 1), (2, 2), (3, 3)]),
            },
        );
    }

    #[test]
    fn test_shard_layout_new_rejects_mesh_mismatch() {
        let mesh = device_mesh_2x2();
        let expected_mesh = mesh.logical_mesh().clone();
        let actual_mesh = LogicalMesh::new(vec![MeshAxis::new("z", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let shape = StaticShape::new(vec![8]);
        let sharding = Sharding::new(actual_mesh.clone(), vec![ShardingDimension::sharded(["z"])]).unwrap();

        let error = ShardLayout::new(&shape, &mesh, &sharding).unwrap_err();
        assert!(matches!(
            &error,
            Error::Core(core_error) if matches!(
                core_error.as_ref(),
                CoreError::Sharding(ShardingError::MeshMismatch { expected, actual })
                    if expected == &expected_mesh && actual == &actual_mesh,
            ),
        ));
        let expected_message = format!("mesh mismatch; expected `{expected_mesh:?}` but got `{actual_mesh:?}`");
        assert_eq!(error.to_string(), expected_message);
    }

    #[test]
    fn test_shard_layout_new_rejects_rank_mismatch() {
        let mesh = device_mesh_2x2();
        let shape = StaticShape::new(vec![8]);
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
        )
        .unwrap();

        let error = ShardLayout::new(&shape, &mesh, &sharding).unwrap_err();
        assert!(matches!(
            &error,
            Error::Core(core_error) if matches!(
                core_error.as_ref(),
                CoreError::Sharding(ShardingError::ShardingRankMismatch { sharding_rank: 2, array_rank: 1 }),
            ),
        ));
        assert_eq!(error.to_string(), "sharding rank (2) does not match array rank (1)");
    }

    #[test]
    fn test_shard_layout_descriptors() {
        let mesh = device_mesh_2x2();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
        let layout = ShardLayout::new(&StaticShape::new(vec![4]), &mesh, &sharding).unwrap();

        assert_eq!(
            layout.descriptors(),
            [
                ShardDescriptor::new(0, Device::new(0, 0), vec![0..2]),
                ShardDescriptor::new(1, Device::new(1, 0), vec![2..4]),
                ShardDescriptor::new(2, Device::new(2, 1), vec![0..2]),
                ShardDescriptor::new(3, Device::new(3, 1), vec![2..4]),
            ]
            .as_slice(),
        );
    }

    #[test]
    fn test_shard_layout_shard_index_by_device() {
        let mesh = DeviceMesh::new(
            logical_mesh_2x2(),
            vec![Device::new(5, 0), Device::new(3, 0), Device::new(9, 1), Device::new(1, 1)],
        )
        .unwrap();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let layout = ShardLayout::new(&StaticShape::new(vec![4]), &mesh, &sharding).unwrap();

        // Shard indices follow the row-major device order of the mesh rather than the device identifiers.
        assert_eq!(layout.shard_index_by_device(), &HashMap::from([(5, 0), (3, 1), (9, 2), (1, 3)]));
    }

    #[test]
    fn test_shard_layout_into_parts() {
        let mesh = device_mesh_2x2();
        let shape = StaticShape::new(vec![8, 6]);
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
                    ShardDescriptor::new(0, Device::new(0, 0), vec![0..8, 0..6]),
                    ShardDescriptor::new(1, Device::new(1, 0), vec![0..8, 0..6]),
                    ShardDescriptor::new(2, Device::new(2, 1), vec![0..8, 0..6]),
                    ShardDescriptor::new(3, Device::new(3, 1), vec![0..8, 0..6]),
                ],
                HashMap::from([(0, 0), (1, 1), (2, 2), (3, 3)]),
            ),
        );
    }
}
