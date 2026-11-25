use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

use ryft_core::{
    ArrayType, DataType, Device, DeviceId, DeviceMesh, Layout, LayoutError, Memory, Sharding, ShardingDimension,
    ShardingError, StaticShape, check_sharding,
};
use ryft_pjrt::{Buffer, Client, Error as PjrtError, Event};

use crate::arrays::{Array, ArrayError};
use crate::devices::MemoryExtension;
use crate::types::DataTypeExtension;
use crate::{Error, FromPjrt};

/// Shard of an [`Array`]. [`ArrayShard`]s always carry global shard metadata through [`ArrayShard::descriptor`]. They
/// also carry either a PJRT [`Buffer`] when ordinary data is addressable from the current process, a bufferless marker
/// for the unique value of [`DataType::Zero`], or metadata only when the current process holds no buffer for the shard
/// (refer to [`ArrayShardStorage::Remote`]). This lets an array describe its full global layout without allocating
/// meaningless zero-space buffers. Ordinary addressable buffers remain stored inside [`Arc`]s so that cloned
/// arrays can share them.
#[derive(Clone)]
pub struct ArrayShard {
    /// Refer to the documentation of [`Self::descriptor`] for information on this field.
    descriptor: ShardDescriptor,

    /// Physical storage state for this [`ArrayShard`].
    storage: ArrayShardStorage,
}

impl ArrayShard {
    /// Creates a new [`ArrayShard`] described by `descriptor`. `buffer` holds the shard's storage when the shard is
    /// addressable from the current process, and [`None`] creates a metadata-only remote shard. Nothing checks that
    /// `buffer` lives on the device of `descriptor` and holds its shape and data type, so only the constructors of
    /// [`Array`], which validate every buffer against the descriptor of its shard first (refer to
    /// [`Array::from_addressable_buffers`]), create shards. A public caller imports buffers
    /// through those constructors and reads the shards of the resulting array.
    #[inline]
    pub(crate) fn new(descriptor: ShardDescriptor, buffer: Option<Arc<Buffer<'static>>>) -> Self {
        let storage = buffer.map_or(ArrayShardStorage::Remote, ArrayShardStorage::Buffer);
        Self { descriptor, storage }
    }

    /// Creates a process-local bufferless [`DataType::Zero`]-valued [`ArrayShard`]. Note that nothing checks that the
    /// device of `descriptor` belongs to the current process, so only the constructors of [`Array`], which do, create
    /// such shards.
    #[inline]
    pub(crate) fn zero(descriptor: ShardDescriptor) -> Self {
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
    /// addressable bufferless [`DataType::Zero`]-valued shard. Note that a clone of the returned [`Arc`] shares the
    /// buffer, and so an execution that is asked to donate the array of this shard passes it without donation while the
    /// clone exists. A [`Weak`](std::sync::Weak) handle of it does not prevent the donation, and it no longer upgrades
    /// once the execution has taken the buffer for donation.
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

    /// Returns `true` if the current process holds the storage of this shard (i.e., a [`Buffer`] or the bufferless
    /// [`DataType::Zero`] value). This reports storage presence, not device ownership. A shard on a device of the
    /// current process whose buffer was not supplied (e.g., by a sparse import through
    /// [`Array::from_addressable_buffers`]) is not addressable either, while [`Device::process_index`]
    /// of [`Self::device`] reports which process owns the device.
    #[inline]
    pub fn is_addressable(&self) -> bool {
        !matches!(self.storage, ArrayShardStorage::Remote)
    }

    /// Copies the bytes that the device of this [`ArrayShard`] holds to the host and returns them in dense row-major
    /// order, irrespective of the physical layout of the shard's [`Buffer`]. This function blocks the current thread
    /// until the copy completes. Bufferless [`DataType::Zero`] shards and shards with no elements produce no bytes
    /// and issue no copy. A shard with no elements still waits for its buffer, so that a failure of the work that
    /// produced that buffer (e.g., a failed host to device transfer) surfaces here as it does for the readback of
    /// the whole array.
    ///
    /// These bytes are a tile of the [`Array`] that this shard belongs to, unless the [`Sharding`] of that array
    /// has unreduced axes, in which case they are the separate value that this device holds for that tile (e.g., a
    /// summand). This function also observes only this shard's buffer. Use [`Array::copy_to_host`] to read the whole
    /// value of an array and to surface the errors of the whole execution that produced it.
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
        read_shard_to_host(buffer, &self.shape(), Event::r#await)
    }

    /// Consumes this [`ArrayShard`] and returns its [`Buffer`], which is `None` in the cases of [`Self::buffer`].
    #[inline]
    pub(crate) fn into_buffer(self) -> Option<Arc<Buffer<'static>>> {
        match self.storage {
            ArrayShardStorage::Buffer(buffer) => Some(buffer),
            ArrayShardStorage::Remote | ArrayShardStorage::Zero => None,
        }
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

/// Physical storage state of an [`ArrayShard`], which [`ArrayShard::storage`] returns. It describes what the current
/// process holds for the shard, not which process owns the device of the shard, which [`Device::process_index`] of
/// [`ShardDescriptor::device`] states. In particular, a shard on a device of the current process can still have no
/// storage here.
#[derive(Clone)]
pub enum ArrayShardStorage {
    /// The current process holds no buffer for the shard. Despite its name, this variant describes the absence of
    /// storage rather than a device of another process. It is the case for every shard on a device of another process,
    /// and also for a shard on a device of the current process whose buffer was not supplied (e.g., an array imported
    /// from the buffers of some of the local devices only, whose other local replicas make it readable).
    Remote,

    /// The shard is process-local but represents the unique value of [`DataType::Zero`] without an underlying buffer.
    Zero,

    /// The shard is process-local and backed by a PJRT [`Buffer`].
    Buffer(Arc<Buffer<'static>>),
}

/// Row-major ordinal index of an [`ArrayShard`] within a [`DeviceMesh`]. Shard indices are assigned using the same
/// row-major ordering as [`DeviceMesh::devices`]. This gives all processes a stable way to refer to the same global
/// shard without depending on whether that shard is locally addressable or not.
pub type ShardIndex = usize;

/// Device ownership and slice metadata for an [`ArrayShard`]. A [`ShardDescriptor`] is intentionally independent
/// of any local [`Buffer`]. It describes which [`Device`] owns the [`ArrayShard`] and which slice of the underlying
/// [`Array`] that shard represents. [`ArrayShard`]s pair this metadata with optional addressable buffers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardDescriptor {
    /// Refer to the documentation of [`Self::index`] for information on this field.
    index: ShardIndex,

    /// Refer to the documentation of [`Self::device`] for information on this field.
    device: Device,

    /// Refer to the documentation of [`Self::slice`] for information on this field. It is shared, so cloning
    /// a descriptor (e.g., into every output [`Array`] of a compiled program) does not allocate.
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

    /// Describes how the elements that this [`ShardDescriptor`]'s [`Self::slice`] covers are laid out in a row-major
    /// buffer that stores an array of the provided [`StaticShape`], as a sequence of equally sized contiguous blocks.
    /// Returns the pair `(dimension, element_count)`, where:
    ///
    ///   - `dimension` is the outermost dimension `d` for which the slice over the dimensions `d..rank` is one
    ///     contiguous span of the buffer once the indices of the dimensions `0..d` are fixed. Each choice of those
    ///     outer indices within the slice therefore selects one block.
    ///   - `element_count` is the number of elements in each such block, which is the product of the slice lengths
    ///     over the dimensions `d..rank`.
    ///
    /// A dimension whose inner slices cover their full extents is part of the block. A dimension that selects one
    /// index can also join the block, since it adds no gaps between elements. An empty slice returns `(0, 0)`.
    /// Rank-0 shards return `(0, 1)` since their single element is its own contiguous block. A returned dimension
    /// of `0` therefore means that the whole shard is one contiguous span. Host uploads can read such a shard from
    /// the dense host buffer without byte strides, and host readback can write one contribution of such a tile
    /// directly to the dense destination. Otherwise, each block can be copied with one [`Vec::extend_from_slice`] call.
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
    /// dimension is absorbed into the block, and the whole shard is one contiguous span. The right slice does not
    /// cover the inner dimension, and so each row of the slice becomes its own contiguous block of `3` elements.
    ///
    /// # Parameters
    ///
    ///   - `shape`: [`StaticShape`] of the contiguous buffer that this [`ShardDescriptor`]'s [`Self::slice`] refers to.
    ///     Its rank must match [`Self::slice`], and each slice range must lie within its corresponding extent.
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

        if self.slice.iter().any(Range::is_empty) {
            return Ok((0, 0));
        }

        let mut dimension = rank;
        let mut element_count = 1usize;
        let mut covers_inner_extents = true;
        for (slice, &extent) in self.slice.iter().zip(shape.as_slice()).rev() {
            let length = slice.len();
            if length > 1 && !covers_inner_extents {
                break;
            }
            dimension -= 1;
            covers_inner_extents &= length == extent;
            element_count = element_count.checked_mul(length).ok_or_else(|| Error::SizeLimitExceeded {
                message: format!(
                    "contiguous shard block element count for shard slice `{:?}` and shape `{}` exceeds the \
                     maximum allowed size of {}",
                    self.slice,
                    shape,
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
    /// Constructs a new [`ShardLayout`] by applying the provided [`Sharding`] to the provided [`StaticShape`] over the
    /// provided [`DeviceMesh`]. The returned layout contains one [`ShardDescriptor`] per [`Device`]. For each array
    /// dimension, replicated and unconstrained dimensions map every device to the full dimension range, while sharded
    /// dimensions are split across the product of the referenced mesh axes. If the dimension size is not evenly
    /// divisible by the partition count, earlier partitions receive one extra element.
    ///
    /// # Parameters
    ///
    ///   - `shape`: [`StaticShape`] of the [`Array`] being sharded.
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
        // out of the inner per-device loop and centralizes defensive validation before any shard descriptors are built.
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

    /// Returns the [`ShardDescriptor`]s for all global shards in row-major [`DeviceMesh`] order.
    /// The position of each descriptor in this vector is the descriptor's [`ShardDescriptor::index`].
    #[inline]
    pub fn descriptors(&self) -> &[ShardDescriptor] {
        &self.descriptors
    }

    /// Returns the lookup table mapping [`DeviceId`]s to their corresponding [`ShardIndex`]es. This is used when a
    /// [`Buffer`] reports a [`DeviceId`] and the [`Array`] constructor needs to find the global shard metadata that
    /// buffer should satisfy.
    #[inline]
    pub fn shard_index_by_device(&self) -> &HashMap<DeviceId, ShardIndex> {
        &self.shard_index_by_device
    }

    /// Consumes this [`ShardLayout`] and returns its [`ShardDescriptor`]s and its lookup table from devices to shards.
    #[inline]
    pub fn into_parts(self) -> (Vec<ShardDescriptor>, HashMap<DeviceId, ShardIndex>) {
        (self.descriptors, self.shard_index_by_device)
    }
}

/// Identity of the contribution held by a shard. Replicas of a contribution share this identity and must hold the
/// same bytes. This definition is shared by the functions that substitute one shard for another or decide that an
/// array holds the same bytes along a mesh axis. Host readback reads one replica per contribution (refer to
/// [`Array::copy_to_host_bytes`]), and partial imports require equal bytes for a contribution's replicas (refer to
/// [`Array::from_addressable_buffers`] and [`Array::from_host_shards`]). Placements in [`crate::arrays::resharding`]
/// use the same rules to find a local replica for a missing buffer, move shards or regions by slice (refer to
/// [`Sharding::holds_one_value_per_slice`]) or by contribution, restrict an array to coordinate zero along the axes
/// of an introduced sum (refer to [`Sharding::distinguishes_shards_along`]), and choose the process of a host round
/// trip. A sharding makes the shards of one array differ in these ways:
///
///   - **Slice:** A dimension that is split along some mesh axes gives the devices at different coordinates along those
///     axes different slices.
///   - **Pending Reduction:** The devices at different coordinates along an unreduced axis hold different contributions
///     to the pending reduction of one slice, such as different summands.
///   - **Manual Variation:** The devices at different coordinates along a varying manual axis (refer to
///     [`Sharding::varying_manual_axes`]) can hold different values of one slice. Along a varying axis that also splits
///     a dimension, the slices already tell those devices apart, so the variation is only between tiles, while along
///     any other varying axis it is within a tile.
///
/// Nothing else that a sharding records makes the shards differ: the devices along a reduced axis all hold the result
/// of the completed reduction, the devices along an axis that neither splits a dimension nor carries reduction state or
/// variation hold replicas, an unconstrained dimension is stored replicated, and the memory, the layout, and the data
/// type are those of the whole array.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct ShardContribution {
    /// Slice of the shard in global array coordinates.
    slice: Vec<Range<usize>>,

    /// Ordinal of the contribution of the shard to the pending reduction of its slice, which is the row-major index
    /// of the coordinates of its device along the unreduced axes of the sharding in mesh order (i.e., a mixed-radix
    /// number whose digits are those coordinates), and zero without unreduced axes. The host readback combines the
    /// contributions of one tile in ascending lexicographic order of those coordinates, which is the order of these
    /// ordinals, so two placements with the same unreduced kind and the same number of contributions correspond by
    /// ordinal, whatever the names, sizes, and order of their unreduced axes.
    unreduced_ordinal: usize,

    /// Coordinates of the device of the shard along the varying manual axes of the sharding, in mesh order unless
    /// [`ShardContribution::ordered`] supplies other axes or another order.
    varying_manual_coordinates: Vec<usize>,
}

impl ShardContribution {
    /// Returns the contribution that `shard`, a shard of a placement on `mesh` with `sharding`, holds.
    pub(super) fn new(mesh: &DeviceMesh, sharding: &Sharding, shard: &ShardDescriptor) -> Self {
        let axes = mesh.logical_mesh().axes();
        let along = |selected: &BTreeSet<String>| {
            axes.iter().map(|axis| axis.name()).filter(|name| selected.contains(*name)).collect::<Vec<_>>()
        };
        Self::ordered(mesh, sharding, shard, &along(sharding.varying_manual_axes()))
    }

    /// Returns the contribution that `shard`, a shard of a placement on `mesh` with `sharding`, holds, with its
    /// coordinates along the mesh axes named `varying_manual_axes` in the order of that list rather than in mesh order.
    /// This lets the contributions of placements on two meshes whose varying manual axes correspond in another order
    /// be compared.
    pub(super) fn ordered<S: AsRef<str>>(
        mesh: &DeviceMesh,
        sharding: &Sharding,
        shard: &ShardDescriptor,
        varying_manual_axes: &[S],
    ) -> Self {
        let coordinates = mesh.device_coordinates(shard.index()).unwrap_or_default();
        let axes = mesh.logical_mesh().axes();
        let unreduced_ordinal = axes
            .iter()
            .zip(&coordinates)
            .filter(|(axis, _)| sharding.unreduced_axes().contains(axis.name()))
            .fold(0, |ordinal, (axis, coordinate)| ordinal * axis.size() + coordinate);
        let varying_manual_coordinates = varying_manual_axes
            .iter()
            .filter_map(|name| axes.iter().position(|axis| axis.name() == name.as_ref()))
            .filter_map(|index| coordinates.get(index).copied())
            .collect();
        Self { slice: shard.slice().to_vec(), unreduced_ordinal, varying_manual_coordinates }
    }

    /// Returns which array of the sequence of arrays that the contributions of a placement form the shard belongs to
    /// (i.e., its unreduced ordinal and its coordinates along the varying manual axes), which the shards with other
    /// slices of the same array share.
    pub(super) fn member(&self) -> (usize, &[usize]) {
        (self.unreduced_ordinal, self.varying_manual_coordinates.as_slice())
    }

    /// Consumes this contribution and returns its slice, unreduced ordinal, and varying manual coordinates.
    pub(super) fn into_parts(self) -> (Vec<Range<usize>>, usize, Vec<usize>) {
        (self.slice, self.unreduced_ordinal, self.varying_manual_coordinates)
    }
}

impl Array {
    /// Returns the [`ShardContribution`] that identifies the contribution held by `shard` of this array.
    pub(super) fn shard_contribution(&self, shard: &ShardDescriptor) -> ShardContribution {
        ShardContribution::new(self.mesh(), self.sharding(), shard)
    }
}

/// Reads the shard with `shape` that `buffer` holds back to the host in dense row-major order, irrespective of the
/// physical layout of `buffer`. `wait` waits for the copy to the host and returns its bytes (e.g., by awaiting its
/// [`Event`], or through the work tracking of a session). This is the one reader of a single shard, which
/// [`ArrayShard::copy_to_host_bytes`] and the host round trips of placements (refer to [`crate::arrays::resharding`])
/// use alike.
///
/// A shard without elements issues no copy, because PJRT can answer a zero byte copy of a bounded-dynamic buffer with
/// its bound allocation and its trailer rather than the empty logical value. This function then only waits for the
/// readiness of `buffer`, so that a failure of the work that produced it (e.g., a failed host to device transfer)
/// still surfaces, as it does for a shard with elements through its copy.
pub(crate) fn read_shard_to_host(
    buffer: &Buffer<'_>,
    shape: &StaticShape,
    wait: impl FnOnce(Event<Vec<u8>>) -> Result<Vec<u8>, PjrtError>,
) -> Result<Vec<u8>, Error> {
    if shape.as_slice().contains(&0) {
        buffer.ready()?.r#await()?;
        return Ok(Vec::new());
    }
    let host_layout = ryft_pjrt::Layout::dense_major_to_minor(shape.rank());
    Ok(wait(buffer.copy_to_host(Some(host_layout))?)?)
}

/// Validates one addressable `buffer` of an array of type `r#type` whose shards are described by `descriptors` and
/// `shard_index_by_device`, returning the identifier of the device that owns it. The buffer must be owned by `client`,
/// live on a device of the mesh in the process that the mesh expects, live in the memory that `r#type` requests, and
/// have the physical element type and the shard shape implied by `r#type`. When `validate_zero_carriers` is set, the
/// buffer of a zero-space array must also contain only false bits.
pub(super) fn validate_addressable_buffer(
    client: &Client<'_>,
    r#type: &ArrayType,
    descriptors: &[ShardDescriptor],
    shard_index_by_device: &HashMap<DeviceId, ShardIndex>,
    buffer: &Buffer<'_>,
    validate_zero_carriers: bool,
) -> Result<DeviceId, Error> {
    let buffer_device = buffer.device()?;
    let device = Device::from_pjrt(&buffer_device)?;
    let device_id = device.id();
    if !buffer.is_owned_by(client) {
        return Err(PjrtError::invalid_argument(format!(
            "the session's client does not own the addressable shard buffer for device {}",
            device.id(),
        ))
        .into());
    }

    let shard_index = shard_index_by_device
        .get(&device_id)
        .ok_or_else(|| Error::from(ArrayError::DeviceNotInMesh { device_id }))?;
    let descriptor = descriptors.get(*shard_index).unwrap();

    // Validate that each buffer is owned by the process expected for the corresponding mesh device.
    let process_index = device.process_index();
    if process_index != descriptor.device().process_index() {
        return Err(Error::from(ArrayError::DeviceProcessIndexMismatch {
            device_id,
            expected: descriptor.device().process_index(),
            actual: process_index,
        }));
    }

    // The element kind of the buffer is validated before any query of its storage or dimensions, which only a
    // buffer that holds an array can answer (the CPU plugin aborts when asked for the dimensions of a token buffer).
    // Zero-space values and one-bit integers use predicate carriers in this execution path. Converting a carrier back
    // to a `DataType` would incorrectly relabel those logical arrays as Boolean. A buffer type without a `DataType`
    // is rejected by that conversion, and a token buffer is rejected here, because effect tokens are not arrays.
    let buffer_type = buffer.element_type()?;
    let expected_buffer_type = r#type.data_type().executable_buffer_type();
    let data_type =
        if buffer_type == expected_buffer_type { r#type.data_type() } else { DataType::from_pjrt(buffer_type)? };
    if data_type == DataType::Token {
        return Err(Error::from(ArrayError::TokenArray));
    }

    // A type's placement must describe the actual buffer, not merely relabel its storage. Device placement
    // means the plugin's default memory, whose kind is platform-dependent (e.g., CPU uses host memory).
    let buffer_memory = buffer.memory()?;
    let expected_memory = r#type.memory();
    let mismatched_kind = match expected_memory {
        Memory::Device => match buffer_memory.compare(&buffer_device.default_memory()?) {
            Ok(()) => None,
            Err(Some(kind)) => Some(kind),
            Err(None) => Some(buffer_memory.kind()?),
        },
        Memory::Host { .. } => {
            // Host placement accepts any memory space of the requested kind. `device.memory(...)`
            // selects one addressable space for allocation.
            let kind = buffer_memory.kind()?;
            (kind != expected_memory.placement_kind()).then_some(kind)
        }
    };

    if let Some(kind) = mismatched_kind {
        return Err(Error::from(ArrayError::BufferMemoryMismatch {
            device_id,
            expected: expected_memory,
            actual: kind.into_owned(),
        }));
    }

    if let Some(expected) = r#type.layout() {
        let actual = Layout::from_pjrt(buffer.specification()?.layout)?;
        if &actual != expected {
            return Err(LayoutError::InvalidLayout {
                message: format!(
                    "buffer on device {device_id} uses layout `{actual}` but its type declares `{expected}`",
                ),
            }
            .into());
        }
    }

    // Validate the concrete buffer type and shape against the physical representation of the logical shard type.
    let shape = StaticShape::new(
        buffer
            .unpadded_dimensions()?
            .iter()
            .map(|size| {
                usize::try_from(*size).map_err(|_| Error::SizeLimitExceeded {
                    message: format!(
                        "buffer dimension size {} exceeds the maximum allowed size of {}",
                        size,
                        usize::MAX,
                    ),
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
    );

    if buffer_type != expected_buffer_type && matches!(r#type.data_type(), DataType::I1 | DataType::U1) {
        return Err(Error::from(ArrayError::BufferTypeMismatch {
            expected: ArrayType::new(DataType::Boolean, shape.clone().into()),
            actual: ArrayType::new(data_type, shape.into()),
        }));
    }

    let array_type = ArrayType::new(data_type, shape.into());
    let expected_array_type = ArrayType::new(r#type.data_type(), descriptor.shape().into());
    if array_type != expected_array_type {
        return Err(Error::from(ArrayError::BufferTypeMismatch { expected: expected_array_type, actual: array_type }));
    }

    // Only the logical elements of the carrier are read, so a buffer of a shard without elements is only awaited and
    // never answered with its bound allocation (refer to `read_shard_to_host`).
    if validate_zero_carriers
        && r#type.data_type().is_zero()
        && read_shard_to_host(buffer, &descriptor.shape(), |event| event.r#await())?
            .iter()
            .any(|value| *value != 0)
    {
        return Err(Error::from(ArrayError::NonCanonicalZeroBuffer { device_id }));
    }

    Ok(device_id)
}

// Per-axis tile ranges are lists that often hold a single range, which this lint mistakes for a range to collect.
#[allow(clippy::single_range_in_vec_init)]
#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::{
        ArrayType, DataType, Dimension, DimensionBounds, DimensionVariable, Error as CoreError, LogicalMesh, MeshAxis,
        MeshAxisType, Shape,
    };
    use ryft_pjrt::{BufferSpecification, BufferType};

    use crate::arrays::Array;
    use crate::devices::PjrtDeviceExtension;
    use crate::sessions::tests::domain_mesh;
    use crate::tests::{
        cpu_client, device_mesh_2x2, execution_client, logical_mesh_2x2, replicated_array, values_from_bytes,
        values_to_bytes,
    };
    use crate::{CompileTracer, CompiledFunction, Session, compile};

    use super::*;

    // Returns a device buffer that holds the `f32` values `[1, 2]`, for shards that need process-local storage.
    fn vector_buffer() -> Arc<Buffer<'static>> {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let array = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.]));
        Arc::clone(array.shards()[0].buffer().unwrap())
    }

    #[test]
    fn test_array_shard_new() {
        // Without a buffer, the shard is a remote shard that only carries its descriptor.
        let descriptor = ShardDescriptor::new(3, Device::new(7, 2), vec![0..2]);
        let shard = ArrayShard::new(descriptor.clone(), None);
        assert_eq!(shard.descriptor(), &descriptor);
        assert!(matches!(shard.storage(), ArrayShardStorage::Remote));

        // With a buffer, the shard holds that buffer as its process-local storage.
        let buffer = vector_buffer();
        let shard = ArrayShard::new(descriptor.clone(), Some(Arc::clone(&buffer)));
        assert_eq!(shard.descriptor(), &descriptor);
        assert!(matches!(shard.storage(), ArrayShardStorage::Buffer(storage) if Arc::ptr_eq(storage, &buffer)));
    }

    #[test]
    fn test_array_shard_zero() {
        let descriptor = ShardDescriptor::new(3, Device::new(7, 2), vec![0..2]);
        let shard = ArrayShard::zero(descriptor.clone());
        assert_eq!(shard.descriptor(), &descriptor);
        assert!(matches!(shard.storage(), ArrayShardStorage::Zero));
    }

    #[test]
    fn test_array_shard_descriptor() {
        let descriptor = ShardDescriptor::new(3, Device::new(7, 2), vec![2..5, 0..4]);
        assert_eq!(ArrayShard::new(descriptor.clone(), None).descriptor(), &descriptor);
        assert_eq!(ArrayShard::zero(descriptor.clone()).descriptor(), &descriptor);
    }

    #[test]
    fn test_array_shard_storage() {
        let descriptor = ShardDescriptor::new(0, Device::new(0, 0), vec![0..2]);
        let buffer = vector_buffer();
        assert!(matches!(ArrayShard::new(descriptor.clone(), None).storage(), ArrayShardStorage::Remote));
        assert!(matches!(ArrayShard::zero(descriptor.clone()).storage(), ArrayShardStorage::Zero));
        assert!(matches!(
            ArrayShard::new(descriptor, Some(Arc::clone(&buffer))).storage(),
            ArrayShardStorage::Buffer(storage) if Arc::ptr_eq(storage, &buffer),
        ));
    }

    #[test]
    fn test_array_shard_buffer() {
        // Only a shard with process-local storage in a buffer has one, so remote and zero shards have none.
        let descriptor = ShardDescriptor::new(0, Device::new(0, 0), vec![0..2]);
        let buffer = vector_buffer();
        assert!(ArrayShard::new(descriptor.clone(), None).buffer().is_none());
        assert!(ArrayShard::zero(descriptor.clone()).buffer().is_none());
        let shard = ArrayShard::new(descriptor, Some(Arc::clone(&buffer)));
        assert!(Arc::ptr_eq(shard.buffer().unwrap(), &buffer));
    }

    #[test]
    fn test_array_shard_index() {
        let shard = ArrayShard::new(ShardDescriptor::new(3, Device::new(7, 2), vec![2..5, 0..4]), None);
        assert_eq!(shard.index(), 3);
    }

    #[test]
    fn test_array_shard_device() {
        let shard = ArrayShard::new(ShardDescriptor::new(3, Device::new(7, 2), vec![2..5, 0..4]), None);
        assert_eq!(shard.device(), Device::new(7, 2));
    }

    #[test]
    fn test_array_shard_slice() {
        let shard = ArrayShard::new(ShardDescriptor::new(3, Device::new(7, 2), vec![2..5, 0..4]), None);
        assert_eq!(shard.slice(), [2..5, 0..4].as_slice());
    }

    #[test]
    fn test_array_shard_shape() {
        let shard = ArrayShard::new(ShardDescriptor::new(3, Device::new(7, 2), vec![2..5, 0..4]), None);
        assert_eq!(shard.shape(), StaticShape::new(vec![3, 4]));
    }

    #[test]
    fn test_array_shard_is_addressable() {
        // A shard is addressable when the current process holds it, whether or not it needs a buffer.
        let descriptor = ShardDescriptor::new(0, Device::new(0, 0), vec![0..2]);
        assert!(!ArrayShard::new(descriptor.clone(), None).is_addressable());
        assert!(ArrayShard::zero(descriptor.clone()).is_addressable());
        assert!(ArrayShard::new(descriptor, Some(vector_buffer())).is_addressable());
    }

    #[test]
    fn test_array_shard_copy_to_host_bytes() {
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

        // A shard with no elements issues no copy, but it still waits for its buffer, so it reports a failure of the
        // transfer that produced that buffer, as the readback of its whole array does.
        let device = client.addressable_devices().unwrap().remove(0);
        let specification = BufferSpecification::new(BufferType::U8, [0u64]);
        let manager = client
            .host_to_device_transfer_manager(vec![specification], device.default_memory().unwrap())
            .unwrap();
        let failed = Array::from_addressable_buffers(
            &session,
            ArrayType::new_static(DataType::U8, [0]),
            DeviceMesh::new(LogicalMesh::single_device(), vec![Device::from_pjrt(&device).unwrap()]).unwrap(),
            None,
            vec![manager.retrieve_buffer(0).unwrap()],
        )
        .unwrap();
        manager.set_error(0, PjrtError::aborted("transfer failed")).unwrap();
        assert!(matches!(
            failed.shards()[0].copy_to_host_bytes(),
            Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "transfer failed",
        ));
        assert!(matches!(
            failed.copy_to_host_bytes(),
            Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "transfer failed",
        ));

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
        assert_eq!(
            ShardDescriptor::new(3, Device::new(7, 2), vec![2..5, 0..4]),
            ShardDescriptor { index: 3, device: Device::new(7, 2), slice: Arc::from([2..5, 0..4]) },
        );
    }

    #[test]
    fn test_shard_descriptor_index() {
        assert_eq!(ShardDescriptor::new(3, Device::new(7, 2), vec![2..5, 0..4]).index(), 3);
    }

    #[test]
    fn test_shard_descriptor_device() {
        assert_eq!(ShardDescriptor::new(3, Device::new(7, 2), vec![2..5, 0..4]).device(), Device::new(7, 2));
    }

    #[test]
    fn test_shard_descriptor_slice() {
        assert_eq!(ShardDescriptor::new(3, Device::new(7, 2), vec![2..5, 0..4]).slice(), [2..5, 0..4].as_slice());

        // A rank-0 shard has no dimensions to slice.
        assert_eq!(ShardDescriptor::new(0, Device::new(7, 2), vec![]).slice(), [].as_slice());
    }

    #[test]
    fn test_shard_descriptor_shape() {
        // The extent of each dimension is the length of its range, so an empty range yields an extent of zero.
        let descriptor = ShardDescriptor::new(3, Device::new(7, 2), vec![2..5, 0..4, 6..6]);
        assert_eq!(descriptor.shape(), StaticShape::new(vec![3, 4, 0]));
        assert_eq!(ShardDescriptor::new(0, Device::new(7, 2), vec![]).shape(), StaticShape::new(vec![]));
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

        let shape = StaticShape::new(vec![4, 6]);
        let single_row = ShardDescriptor::new(0, device, vec![1..2, 2..5]);
        assert_eq!(single_row.contiguous_inner_block(&shape).unwrap(), (0, 3));
        let empty = ShardDescriptor::new(0, device, vec![1..1, 2..5]);
        assert_eq!(empty.contiguous_inner_block(&shape).unwrap(), (0, 0));

        // The singleton dimension joins the block without closing the gap between consecutive rows.
        let separated = ShardDescriptor::new(0, device, vec![1..3, 0..1, 2..5]);
        assert_eq!(separated.contiguous_inner_block(&StaticShape::new(vec![4, 1, 6])).unwrap(), (1, 3));

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

        // A block whose element count does not fit in a `usize` is rejected instead of wrapping around.
        let descriptor = ShardDescriptor::new(0, device, vec![0..usize::MAX, 0..2]);
        let error = descriptor.contiguous_inner_block(&StaticShape::new(vec![usize::MAX, 2])).unwrap_err();
        let message = format!(
            "contiguous shard block element count for shard slice `[0..{max}, 0..2]` and shape `[{max}, 2]` exceeds \
             the maximum allowed size of {max}",
            max = usize::MAX,
        );
        assert!(matches!(&error, Error::SizeLimitExceeded { message: actual } if actual == &message));
    }

    #[test]
    fn test_shard_layout_new() {
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

    #[test]
    fn test_array_shard_contribution() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let contributions = |array: &Array| {
            let contribution = |shard: &ArrayShard| array.shard_contribution(shard.descriptor()).into_parts();
            array.shards().iter().map(contribution).collect::<Vec<_>>()
        };

        // The devices along `x` hold different slices, and those along the unreduced `y` hold the contributions
        // to the pending reduction of one slice, which the ordinal tells apart.
        let mesh = DeviceMesh::new(logical_mesh_2x2(), devices.clone()).unwrap();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])])
            .unwrap()
            .with_unreduced_axes(["y"])
            .unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let bytes = values_to_bytes(&[1f32, 2., 3., 4.]);
        let array = Array::from_host_buffer(&session, r#type, mesh, Some(sharding), bytes).unwrap();
        assert_eq!(
            contributions(&array),
            [(vec![0..2], 0, vec![]), (vec![0..2], 1, vec![]), (vec![2..4], 0, vec![]), (vec![2..4], 1, vec![])],
        );

        // The devices along a varying manual axis that splits no dimension hold values of one slice that can differ,
        // which their coordinates along that axis tell apart.
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices[..2].to_vec()).unwrap();
        let sharding = Sharding::replicated(logical_mesh, 1).with_varying_manual_axes(["m"]).unwrap();
        let r#type = ArrayType::new_static(DataType::U8, [1]);
        let shards = [(devices[0].id(), [1u8].as_slice()), (devices[1].id(), [2u8].as_slice())];
        let array = Array::from_host_shards(&session, r#type, mesh, Some(sharding), shards).unwrap();
        assert_eq!(contributions(&array), [(vec![0..1], 0, vec![0]), (vec![0..1], 0, vec![1])]);
    }

    #[test]
    fn test_validate_addressable_buffer() {
        let client = cpu_client(1);
        let pjrt_device = client.addressable_devices().unwrap().remove(0);
        let device = Device::from_pjrt(&pjrt_device).unwrap();
        let device_id = device.id();
        let r#type = ArrayType::scalar(DataType::I32);
        let bytes = 7i32.to_ne_bytes();
        let buffer = client.buffer(bytes.as_slice(), BufferType::I32, [], None, pjrt_device.clone(), None).unwrap();
        let shard_index_by_device = HashMap::from([(device_id, 0)]);
        let descriptors = [ShardDescriptor::new(0, device, Vec::new())];
        assert_eq!(
            validate_addressable_buffer(client, &r#type, &descriptors, &shard_index_by_device, &buffer, true).unwrap(),
            device_id,
        );

        // Device placement requires the default memory. A host buffer reports its actual kind on mismatch.
        let pinned_memory = pjrt_device.memory(Memory::Host { pinned: true }).unwrap();
        let pinned_buffer = client.buffer(bytes.as_slice(), BufferType::I32, [], None, pinned_memory, None).unwrap();
        assert!(matches!(
            validate_addressable_buffer(client, &r#type, &descriptors, &shard_index_by_device, &pinned_buffer, true),
            Err(Error::Array(error)) if matches!(error.as_ref(),
                ArrayError::BufferMemoryMismatch { device_id: actual_device_id, expected: Memory::Device, actual }
                    if *actual_device_id == device_id && actual == "pinned_host",
            ),
        ));

        // A matching device ID still requires the process index declared by the mesh.
        let process_index = device.process_index();
        let other_process_index = process_index + 1;
        let descriptors = [ShardDescriptor::new(0, Device::new(device_id, other_process_index), Vec::new())];
        let error = validate_addressable_buffer(client, &r#type, &descriptors, &shard_index_by_device, &buffer, true)
            .unwrap_err();
        assert!(matches!(
            &error,
            Error::Array(error) if matches!(error.as_ref(),
                ArrayError::DeviceProcessIndexMismatch { device_id: actual_device_id, expected, actual }
                    if *actual_device_id == device_id && *expected == other_process_index && *actual == process_index,
            ),
        ));
        assert_eq!(
            error.to_string(),
            format!("device {device_id} reports process index {process_index}, but expected {other_process_index}"),
        );

        // A token buffer holds no array, so it is rejected by its element type before anything asks for its
        // dimensions, which the CPU plugin answers by aborting.
        let descriptors = [ShardDescriptor::new(0, device, Vec::new())];
        let token = client.buffer([0u8; 0].as_slice(), BufferType::Token, [], None, pjrt_device.clone(), None).unwrap();
        assert_eq!(
            validate_addressable_buffer(client, &r#type, &descriptors, &shard_index_by_device, &token, true)
                .map_err(|error| error.to_string()),
            Err("tokens are created by effectful operations and cannot be constructed, uploaded, or placed as arrays"
                .to_string()),
        );
    }

    #[test]
    fn test_validate_addressable_buffer_reads_only_logical_zero_carrier_elements() {
        // A bounded function of a Boolean `[n, 3]` input at `n = 0` returns a buffer of logical shape `[0, 3]` whose
        // raw host copy holds its bound allocation (29 bytes on the CPU plugin). Importing that buffer as an empty
        // array of the zero data type reads only its logical elements, of which there are none, so the import succeeds.
        // Before, the import copied and judged the whole bound allocation. The CPU plugin zeroes that padding, so this
        // test pins the empty import, not a rejection that nonzero padding caused.
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![device]).unwrap();
        let extent = DimensionVariable::new("n", DimensionBounds::new(0, Some(7)).unwrap());
        let bounded_type = ArrayType::new(DataType::Boolean, Shape::new(vec![extent.into(), Dimension::Static(3)]));
        let negation: CompiledFunction<ArrayType, ArrayType> =
            compile(|x: CompileTracer| !x, bounded_type, &session, mesh.clone()).unwrap();
        let empty_type = ArrayType::new_static(DataType::Boolean, [0, 3]);
        let input = Array::from_host_buffer(&session, empty_type, Some(mesh.clone()), None, Vec::new()).unwrap();
        let output = negation.executable().execute(&session, input).unwrap();
        let buffer = Arc::clone(output.shards()[0].buffer().unwrap());
        drop(output);
        let buffer = Arc::try_unwrap(buffer).ok().unwrap();
        assert!(!buffer.copy_to_host(None).unwrap().r#await().unwrap().is_empty());

        let zero_type = ArrayType::new_static(DataType::Zero, [0, 3]);
        let array = Array::from_addressable_buffers(&session, zero_type, mesh, None, vec![buffer]).unwrap();
        assert_eq!((array.shape(), array.copy_to_host_bytes().unwrap()), (StaticShape::new(vec![0, 3]), Vec::new()));
    }
}
