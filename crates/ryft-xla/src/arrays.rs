use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

use ryft_core::{
    ArrayType, DataType, Device, DeviceId, DeviceMesh, Parameter, Sharding, ShardingDimension, ShardingError,
    StaticShape, Typed, check_sharding,
};
use ryft_macros::Parameter;
use ryft_pjrt::Buffer;

use crate::{Error, FromPjrt};

/// Distributed array with a global [`StaticShape`] and element [`DataType`] as well as [`Sharding`] information.
/// An [`Array`] represents one logical [`ArrayType`] whose elements may be split or replicated across the multiple
/// devices in a [`DeviceMesh`], potentially spanning multiple nodes or processes. The global array is described by
/// its type and sharding metadata, while each physical piece of that global array is represented by an [`ArrayShard`].
/// [`Array::shards`] is a global list: it contains one [`ShardDescriptor`] for each device that participates in the
/// array placement, and not only for the devices that are visible to the current process. In a single-process setup,
/// every shard is normally _addressable_ because the local [`Client`](ryft_pjrt::Client) can directly access every
/// backing [`Buffer`]. In a multi-device or multi-node setup, the same logical array can span devices owned by other
/// processes. Shards on the current process's devices are addressable and carry local [`Buffer`]s; shards on remote
/// devices are non-addressable and carry only metadata such as their global [`ShardIndex`], [`DeviceId`], and
/// [`ArrayType`]. Keeping both addressable and non-addressable shards in the same [`Array`] lets local code reason
/// about the complete global placement while only transferring, executing with, or materializing buffers that this
/// process can access directly. This distinction is what allows array movement, execution argument assembly, and
/// cross-host transfers to preserve the full global sharding contract without requiring every process to own every
/// shard buffer.
#[derive(Clone, Parameter)]
pub struct Array<'o> {
    // TODO(eaplatanios): Make these fields private.
    /// [`ArrayType`] of this [`Array`].
    pub(crate) r#type: ArrayType,

    /// [`ArrayShard`]s that make up this [`Array`].
    pub(crate) shards: Vec<ArrayShard<'o>>,

    /// Lookup table mapping [`ryft_pjrt::DeviceId`]s to their corresponding [`ShardIndex`]es
    /// (indexing into [`Self::shards`]).
    pub(crate) shard_index_by_device: HashMap<ryft_pjrt::DeviceId, ShardIndex>,
}

impl<'o> Array<'o> {
    /// Creates an [`Array`] of type `r#type` from the provided _addressable_ [`Buffer`]s using the provided concrete
    /// [`DeviceMesh`] to determine what [`ArrayShard`]s make up the array and which ones correspond to the addressable
    /// buffers. The provided [`ArrayType`] describes the global logical array. Its [`Shape`](ryft_core::Shape) must be
    /// static, and it must normally contain [`Sharding`] metadata whose logical mesh matches `mesh`. As a convenience
    /// for unsharded arrays, callers may omit the sharding information only when exactly one addressable buffer is
    /// provided; in that case the array is treated as replicated over the provided `mesh`.
    ///
    /// Each [`Buffer`] is assigned to a global shard based on the [`DeviceId`] of its owning device. This function will
    /// return an [`Error::MultipleBuffersOnDevice`] if there are multiple buffers provided that are owned by the same
    /// device. It will also return an [`Error::DeviceNotInMesh`] if there are buffer whose device is not present in
    /// `mesh`, an [`Error::NonAddressableDevice`] for buffers whose device belongs to a different process than the
    /// corresponding mesh device, and an [`Error::BufferTypeMismatch`] for buffers whose data type or static shape
    /// does not match the shard type derived from `r#type`, `mesh`, and the effective sharding. Shards that do not
    /// have a local/addressable buffer are retained as non-addressable [`ArrayShard`]s.
    ///
    /// # Parameters
    ///
    ///   - `r#type`: Global [`ArrayType`] for the new [`Array`].
    ///   - `mesh`: [`DeviceMesh`] that is used to determine the [`ArrayShard`] placement.
    ///   - `buffers`: [`Buffer`]s for the [`ArrayShard`]s that are addressable from the current process.
    pub fn from_addressable_buffers(
        r#type: ArrayType,
        mesh: DeviceMesh,
        buffers: Vec<Buffer<'o>>,
    ) -> Result<Self, Error> {
        // Normalize the provided `r#type` before deriving shard placement. A single buffer with no sharding metadata
        // is treated as an unsharded replicated array over the caller-provided mesh.
        let r#type = match r#type.sharding() {
            Some(_) => r#type,
            None if buffers.len() != 1 => {
                return Err(Error::MissingSharding);
            }
            None => ArrayType::new(
                r#type.data_type(),
                r#type.shape().clone(),
                r#type.layout().cloned(),
                Some(Sharding::replicated(mesh.logical_mesh().clone(), r#type.shape().rank())),
            )?,
        };

        // Compute the global [`ShardLayout`] implied by the normalized array type and concrete device mesh.
        let shape = r#type.static_shape().ok_or_else(|| Error::DynamicShape { shape: r#type.shape().clone() })?;
        let sharding = r#type.sharding().ok_or(Error::MissingSharding)?;
        let layout = ShardLayout::new(&shape, &mesh, sharding)?;
        let (descriptors, shard_index_by_device) = layout.into_parts();

        // Index the provided [`Buffer`]s by device while rejecting duplicate local buffers for the same device.
        let mut buffers_by_device = HashMap::with_capacity(buffers.len());
        for buffer in buffers {
            let device = buffer.device()?;
            let device_id = device.id()?;
            if buffers_by_device.contains_key(&device_id) {
                return Err(Error::MultipleBuffersOnDevice { device_id });
            }

            let shard_index = shard_index_by_device.get(&device_id).ok_or(Error::DeviceNotInMesh { device_id })?;
            let descriptor = descriptors.get(*shard_index).unwrap();

            // Validate that each buffer is owned by the process expected for the corresponding mesh device.
            let process_index = device.process_index()?;
            if process_index != descriptor.device().process_index() {
                return Err(Error::DeviceProcessIndexMismatch {
                    device_id,
                    expected_process_index: descriptor.device().process_index(),
                    actual_process_index: process_index,
                });
            }

            // Validate the concrete buffer type against the shard type that the layout assigns to this device.
            let data_type = DataType::from_pjrt(buffer.element_type()?)?;
            let shape = StaticShape::new(
                buffer
                    .dimensions()?
                    .iter()
                    .map(|size| usize::try_from(*size).map_err(|_| Error::SizeLimitExceeded { size: *size }))
                    .collect::<Result<Vec<_>, _>>()?,
            );
            let array_type = ArrayType::new(data_type, shape.into(), None, None)?;
            let expected_array_type = ArrayType::new(r#type.data_type(), descriptor.shape().into(), None, None)?;
            if array_type != expected_array_type {
                return Err(Error::BufferTypeMismatch { expected: expected_array_type, actual: array_type });
            }

            buffers_by_device.insert(device_id, Arc::new(buffer));
        }

        // Materialize the global shard list with local buffers attached to the shards addressable from this process.
        let shards = descriptors
            .into_iter()
            .map(|descriptor| {
                let buffer = buffers_by_device.remove(&descriptor.device().id());
                ArrayShard::new(descriptor, buffer)
            })
            .collect::<Vec<_>>();

        Ok(Self { r#type, shards, shard_index_by_device })
    }

    /// Returns the [`DataType`] of the elements stored in this [`Array`].
    #[inline]
    pub fn data_type(&self) -> DataType {
        self.r#type.data_type()
    }

    /// Returns the global [`StaticShape`] of this [`Array`].
    #[inline]
    pub fn shape(&self) -> StaticShape {
        self.r#type
            .static_shape()
            .expect("runtime arrays should only be constructed from array types with static shapes")
    }

    /// Returns the [`ArrayShard`]s that make up this [`Array`].
    pub fn shards(&self) -> &[ArrayShard<'o>] {
        self.shards.as_slice()
    }

    /// Returns an [`Iterator`] over the _addressable_ [`ArrayShard`]s of this [`Array`].
    pub fn addressable_shards(&self) -> impl Iterator<Item = &ArrayShard<'o>> {
        self.shards.iter().filter(|shard| shard.is_addressable())
    }

    /// Returns the [`ArrayShard`] of this [`Array`] that is placed on the device with the provided
    /// [`DeviceId`], if such a shard exists.
    pub fn device_shard(&self, device_id: DeviceId) -> Option<&ArrayShard<'o>> {
        self.shard_index_by_device.get(&device_id).and_then(|index| self.shards.get(*index))
    }

    /// Returns the _addressable_ [`ArrayShard`] of this [`Array`] that is placed on the device with the provided
    /// [`DeviceId`], if such a shard exists.
    pub fn addressable_device_shard(&self, device_id: DeviceId) -> Option<&ArrayShard<'o>> {
        self.device_shard(device_id).filter(|shard| shard.is_addressable())
    }
}

impl Debug for Array<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Array").field("type", &self.r#type).field("shards", &self.shards()).finish()
    }
}

impl Typed<ArrayType> for Array<'_> {
    fn r#type(&self) -> Cow<'_, ArrayType> {
        Cow::Borrowed(&self.r#type)
    }
}

/// Shard of an [`Array`]. [`ArrayShard`]s always carry global shard metadata through [`ArrayShard::descriptor`].
/// They also carry a PJRT [`Buffer`] when the owning device is addressable from the current process (otherwise
/// [`ArrayShard::buffer`] is set to `None`). This lets an [`Array`] describe its full global layout while storing
/// local buffers only for the shards that the current process can read directly, without moving data around. Note
/// that, [`ArrayShard`]s holds their [`Buffer`]s inside [`Arc`]s so that cloning an array can share addressable
/// PJRT [`Buffer`]s with other array instances. The last [`Arc`] dropped releases the underlying PJRT buffer via
/// [`Buffer`]'s [`Drop`] implementation.
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

    /// Returns the [`Device`] that owns this [`ArrayShard`].
    #[inline]
    pub fn device(&self) -> Device {
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

/// Row-major ordinal index of an [`ArrayShard`] within a [`DeviceMesh`]. Shard indices are assigned using the same
/// row-major ordering as [`DeviceMesh::devices`]. This gives all processes a stable way to refer to the same global
/// shard without depending on whether that shard is locally addressable or not.
pub type ShardIndex = usize;

/// Device ownership and slice metadata for an [`ArrayShard`]. A [`ShardDescriptor`] is intentionally independent of any
/// local [`Buffer`]. It describes which [`Device`] owns the [`ArrayShard`] and which slice of the underlying [`Array`]
/// that shard represents. [`ArrayShard`]s pair this metadata with optional addressable buffers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardDescriptor {
    /// Refer to the documentation of [`Self::index`] for information on this field.
    index: ShardIndex,

    /// Refer to the documentation of [`Self::device`] for information on this field.
    device: Device,

    /// Refer to the documentation of [`Self::slice`] for information on this field.
    slice: Vec<Range<usize>>,
}

impl ShardDescriptor {
    /// Creates a new [`ShardDescriptor`].
    #[inline]
    pub fn new(index: ShardIndex, device: Device, slice: Vec<Range<usize>>) -> Self {
        Self { index, device, slice }
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
    /// dimension. Ranges use normal Rust semantics: the start is inclusive and the end is exclusive. Replicated and
    /// unconstrained dimensions span the full dimension range (i.e., `0..dimension_size`) while sharded dimensions span
    /// the contiguous partition selected by this shard's [`Device`] coordinates in the underlying [`DeviceMesh`].
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
/// the logical sharding specification into one [`ShardDescriptor`] per [`Device`] and also records a device-ID
/// lookup table for routing local [`Buffer`]s back to their global shard descriptors.
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
    ///   - `shape`: [`StaticShape`] of the [`Array`] being sharded/partitioned.
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

        // Resolve each sharded array dimension to mesh-axis indices. This keeps the inner per-device loop
        // allocation-free with respect to axis-name lookup and centralizes defensive validation before any shard
        // descriptors are built.
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
            let device_coordinates =
                mesh.device_coordinates(index).expect("mesh coordinate should exist for valid device index");

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
    /// [`Buffer`] reports a [`DeviceId`] and the [`Array`] constructor needs to find the global shard metadata that
    /// buffer should satisfy.
    #[inline]
    pub fn shard_index_by_device(&self) -> &HashMap<DeviceId, ShardIndex> {
        &self.shard_index_by_device
    }

    /// Consumes this [`ShardLayout`] and returns its [`ShardDescriptor`]s and device-to-shard lookup table.
    #[inline]
    pub(crate) fn into_parts(self) -> (Vec<ShardDescriptor>, HashMap<DeviceId, ShardIndex>) {
        (self.descriptors, self.shard_index_by_device)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;

    use ryft_core::{
        Device, DeviceMesh, Error as CoreError, LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension,
        ShardingError, StaticShape,
    };

    use crate::Error;
    use crate::tests::device_mesh_2x2;

    use super::{ArrayShard, ShardDescriptor, ShardLayout};

    // TODO(eaplatanios):
    //  - `test_array`:
    //     - `Array::from_addressable_buffers`
    //     - `Array::r#type`
    //     - `Array::data_type`
    //     - `Array::shape`
    //     - `Array::shards`
    //     - `Array::addressable_shards`
    //     - `Array::device_shard`
    //     - `Array::addressable_device_shard`
    //  - `test_array_debug`

    #[test]
    fn test_array_shard() {
        let descriptor = ShardDescriptor::new(3, Device::new(7, 2), vec![2..5, 0..4]);
        let shard = ArrayShard::new(descriptor.clone(), None);

        assert_eq!(shard.descriptor(), &descriptor);
        assert!(shard.buffer().is_none());
        assert_eq!(shard.index(), 3);
        assert_eq!(shard.device(), Device::new(7, 2));
        assert_eq!(shard.slice(), [2..5, 0..4].as_slice());
        assert_eq!(shard.shape(), StaticShape::new(vec![3, 4]));
        assert!(!shard.is_addressable());
        assert_eq!(shard.descriptor, descriptor);
        assert!(shard.buffer.is_none());
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
    fn test_shard_descriptor() {
        let device = Device::new(7, 2);
        let descriptor = ShardDescriptor::new(3, device, vec![2..5, 0..4]);

        assert_eq!(descriptor.index(), 3);
        assert_eq!(descriptor.device(), device);
        assert_eq!(descriptor.slice(), [2..5, 0..4].as_slice());
        assert_eq!(descriptor.shape(), StaticShape::new(vec![3, 4]));
    }

    #[test]
    fn test_shard_layout_evenly_partitions_two_dimensions() {
        let mesh = device_mesh_2x2();
        let shape = StaticShape::new(vec![8, 6]);
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
        )
        .unwrap();

        assert_eq!(
            ShardLayout::new(&shape, &mesh, &sharding),
            Ok(ShardLayout {
                descriptors: vec![
                    ShardDescriptor::new(0, Device::new(0, 0), vec![0..4, 0..3]),
                    ShardDescriptor::new(1, Device::new(1, 0), vec![0..4, 3..6]),
                    ShardDescriptor::new(2, Device::new(2, 1), vec![4..8, 0..3]),
                    ShardDescriptor::new(3, Device::new(3, 1), vec![4..8, 3..6]),
                ],
                shard_index_by_device: HashMap::from([(0, 0), (1, 1), (2, 2), (3, 3)]),
            }),
        );
    }

    #[test]
    fn test_shard_layout_keeps_replicated_and_unconstrained_dimensions_full_size() {
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
            ShardLayout::new(&shape, &mesh, &sharding),
            Ok(ShardLayout {
                descriptors: vec![
                    ShardDescriptor::new(0, Device::new(0, 0), vec![0..4, 0..6, 0..5]),
                    ShardDescriptor::new(1, Device::new(1, 0), vec![0..4, 0..6, 0..5]),
                    ShardDescriptor::new(2, Device::new(2, 1), vec![4..8, 0..6, 0..5]),
                    ShardDescriptor::new(3, Device::new(3, 1), vec![4..8, 0..6, 0..5]),
                ],
                shard_index_by_device: HashMap::from([(0, 0), (1, 1), (2, 2), (3, 3)]),
            }),
        );
    }

    #[test]
    fn test_shard_layout_partitions_uneven_dimension() {
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![Device::new(0, 0), Device::new(1, 0)]).unwrap();
        let shape = StaticShape::new(vec![5]);
        let sharding = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();

        assert_eq!(
            ShardLayout::new(&shape, &mesh, &sharding),
            Ok(ShardLayout {
                descriptors: vec![
                    ShardDescriptor::new(0, Device::new(0, 0), vec![0..3]),
                    ShardDescriptor::new(1, Device::new(1, 0), vec![3..5]),
                ],
                shard_index_by_device: HashMap::from([(0, 0), (1, 1)]),
            }),
        );
    }

    #[test]
    fn test_shard_layout_partitions_single_dimension_over_multiple_mesh_axes() {
        let mesh = device_mesh_2x2();
        let shape = StaticShape::new(vec![10]);
        let sharding =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x", "y"])]).unwrap();

        assert_eq!(
            ShardLayout::new(&shape, &mesh, &sharding),
            Ok(ShardLayout {
                descriptors: vec![
                    ShardDescriptor::new(0, Device::new(0, 0), vec![0..3]),
                    ShardDescriptor::new(1, Device::new(1, 0), vec![3..6]),
                    ShardDescriptor::new(2, Device::new(2, 1), vec![6..8]),
                    ShardDescriptor::new(3, Device::new(3, 1), vec![8..10]),
                ],
                shard_index_by_device: HashMap::from([(0, 0), (1, 1), (2, 2), (3, 3)]),
            }),
        );
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
    fn test_shard_layout_rejects_mesh_mismatch() {
        let mesh = device_mesh_2x2();
        let expected_mesh = mesh.logical_mesh().clone();
        let actual_mesh = LogicalMesh::new(vec![MeshAxis::new("z", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let shape = StaticShape::new(vec![8]);
        let sharding = Sharding::new(actual_mesh.clone(), vec![ShardingDimension::sharded(["z"])]).unwrap();

        let error = ShardLayout::new(&shape, &mesh, &sharding).unwrap_err();
        assert_eq!(
            error,
            Error::CoreError(CoreError::Sharding(ShardingError::MeshMismatch {
                expected: expected_mesh.clone(),
                actual: actual_mesh.clone(),
            })),
        );
        let expected_message = format!("mesh mismatch; expected '{expected_mesh:?}' but got '{actual_mesh:?}'");
        assert_eq!(error.to_string(), expected_message);
    }

    #[test]
    fn test_shard_layout_rejects_rank_mismatch() {
        let mesh = device_mesh_2x2();
        let shape = StaticShape::new(vec![8]);
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
        )
        .unwrap();

        let error = ShardLayout::new(&shape, &mesh, &sharding).unwrap_err();
        assert_eq!(
            error,
            Error::CoreError(CoreError::Sharding(ShardingError::ShardingRankMismatch {
                sharding_rank: 2,
                array_rank: 1,
            })),
        );
        assert_eq!(error.to_string(), "sharding rank (2) does not match array rank (1)");
    }
}
