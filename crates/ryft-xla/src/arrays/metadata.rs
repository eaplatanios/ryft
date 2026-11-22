use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use ryft_core::{ArrayType, DeviceId, DeviceMesh, Sharding, StaticShape, TypeError};

use crate::Error;
use crate::arrays::ArrayError;
use crate::arrays::shards::{ShardDescriptor, ShardIndex, ShardLayout};

/// Type and runtime placement of an [`Array`](crate::arrays::Array), independent of its device buffers. It is immutable
/// and shared. An array, its clones, the [`LiveArray`](crate::LiveArray)s and [`RetainedArray`](crate::RetainedArray)s
/// created from it, and the other arrays that a compiled program produces for the same output all refer to one
/// [`ArrayMetadata`], so sharing it costs a single reference count rather than one for each of its fields.
///
/// The metadata describes a value, and it is not a type. It contains the [`ArrayType`] of the array, and it adds what
/// that type deliberately omits (i.e., the concrete devices of the mesh and the placement over auto mesh axes).
#[derive(Debug)]
pub(crate) struct ArrayMetadata {
    /// [`ArrayType`] of the array. Its sharding is [`Self::sharding`] without auto mesh axes, because types never
    /// record placement over those axes.
    r#type: ArrayType,

    /// Runtime placement of the array over all mesh axes, including the auto mesh axes that its type omits.
    sharding: Sharding,

    /// Lookup table that maps each [`DeviceId`] to the [`ShardIndex`] of the shard that the device holds. It depends
    /// only on the placement of the array.
    shard_index_by_device: HashMap<DeviceId, ShardIndex>,

    /// Concrete [`DeviceMesh`] that the array is placed on.
    mesh: DeviceMesh,
}

impl ArrayMetadata {
    /// Creates the [`ArrayMetadata`] of an array of type `r#type` that is placed on `mesh` according to `sharding`, and
    /// returns it together with the [`ShardDescriptor`]s of the shards of that array, in mesh order. When `sharding` is
    /// [`None`], the array is placed according to the sharding of `r#type`, or replicated over `mesh` when `r#type` has
    /// no sharding. The type in the metadata carries the placement without its auto mesh axes, because types never
    /// record placement over those axes.
    ///
    /// This is the only way to create metadata, so the type, the placement, and the device index of every
    /// [`ArrayMetadata`] agree with the one [`ShardLayout`] that they were derived from.
    ///
    /// # Errors
    ///
    /// Returns [`ArrayError::PlacementTypeMismatch`] if `sharding` disagrees with the sharding of `r#type` over
    /// the non-auto mesh axes, a [`TypeError`] if the layout of `r#type` does not address the shape of each shard,
    /// [`ArrayError::DynamicShape`] if `r#type` does not have a static shape, and the errors of [`ShardLayout::new`]
    /// if the placement is not valid on `mesh`.
    pub(crate) fn new(
        r#type: ArrayType,
        sharding: Option<Sharding>,
        mesh: DeviceMesh,
    ) -> Result<(Self, Vec<ShardDescriptor>), Error> {
        let (r#type, sharding) = resolve_placement(r#type, sharding, &mesh)?;
        let shape = r#type
            .static_shape()
            .ok_or_else(|| Error::from(ArrayError::DynamicShape { shape: r#type.shape().clone() }))?;
        let (descriptors, shard_index_by_device) = ShardLayout::new(&shape, &mesh, &sharding)?.into_parts();
        Ok((Self { r#type, sharding, shard_index_by_device, mesh }, descriptors))
    }

    /// Returns the [`ArrayType`] of the array.
    #[inline]
    pub(crate) fn r#type(&self) -> &ArrayType {
        &self.r#type
    }

    /// Returns the runtime placement of the array over all mesh axes, including the auto mesh axes that its type omits.
    #[inline]
    pub(crate) fn sharding(&self) -> &Sharding {
        &self.sharding
    }

    /// Returns the lookup table that maps each [`DeviceId`] to the [`ShardIndex`] of the shard that the device holds.
    #[inline]
    pub(crate) fn shard_index_by_device(&self) -> &HashMap<DeviceId, ShardIndex> {
        &self.shard_index_by_device
    }

    /// Returns the concrete [`DeviceMesh`] that the array is placed on.
    #[inline]
    pub(crate) fn mesh(&self) -> &DeviceMesh {
        &self.mesh
    }
}

/// Placement of one output of a compiled program, computed once and reused by
/// [`Array::from_execution_output`](crate::Array::from_execution_output) when it builds an output from the
/// returned buffers. Every array built from one plan shares its [`ArrayMetadata`].
pub(crate) struct ArrayOutputPlan {
    /// Refer to the documentation of [`Self::metadata`] for information on this field.
    metadata: Arc<ArrayMetadata>,

    /// Refer to the documentation of [`Self::descriptors`] for information on this field.
    descriptors: Vec<ShardDescriptor>,

    /// Refer to the documentation of [`Self::buffer_positions`] for information on this field.
    buffer_positions: Vec<Option<usize>>,
}

impl ArrayOutputPlan {
    /// Creates the plan of an output of type `r#type` placed on `mesh` according to `sharding`, whose buffers an
    /// execution returns in the order of `addressable_device_ids`. When `sharding` is [`None`], the output is placed
    /// according to the sharding of `r#type`, or replicated when `r#type` has no sharding.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if `r#type` does not have a static shape, if `sharding` disagrees with the sharding of
    /// `r#type`, or if the placement is not valid on `mesh`.
    pub(crate) fn new(
        r#type: ArrayType,
        sharding: Option<Sharding>,
        mesh: &DeviceMesh,
        addressable_device_ids: &[DeviceId],
    ) -> Result<Self, Error> {
        let (metadata, descriptors) = ArrayMetadata::new(r#type, sharding, mesh.clone())?;
        let mut buffer_positions = vec![None; descriptors.len()];
        let shard_index_by_device = metadata.shard_index_by_device();
        for (position, device_id) in addressable_device_ids.iter().enumerate() {
            if let Some(&shard_index) = shard_index_by_device.get(device_id) {
                // Preserve the first buffer position for each device.
                buffer_positions[shard_index].get_or_insert(position);
            }
        }
        Ok(Self { metadata: Arc::new(metadata), descriptors, buffer_positions })
    }

    /// Returns the shared metadata for this output, including its static type and runtime placement.
    #[inline]
    pub(super) fn metadata(&self) -> &Arc<ArrayMetadata> {
        &self.metadata
    }

    /// Returns the shard descriptors of the output in mesh order.
    #[inline]
    pub(super) fn descriptors(&self) -> &[ShardDescriptor] {
        &self.descriptors
    }

    /// Returns the position of each shard's buffer among the executable's addressable devices, or [`None`] for
    /// a shard that is not addressable from the current process. Entries follow the order of [`Self::descriptors`].
    #[inline]
    pub(super) fn buffer_positions(&self) -> &[Option<usize>] {
        &self.buffer_positions
    }
}

/// Returns the [`ArrayType`] and runtime placement of an array of type `r#type` that is constructed on `mesh` with
/// the optional placement `sharding`. Types never record placement over auto mesh axes, so the returned type carries
/// the placement without its auto mesh axes. Without `sharding`, the placement is the sharding of `r#type` if present
/// and replication over `mesh` otherwise.
///
/// The layout of `r#type`, if any, is the physical layout of each device shard, so it must address the shape of every
/// shard of the placement (refer to [`validate_shard_layout`]). Every constructor of an array resolves its placement
/// here before it looks at buffers or at the process that it runs in, so every process and every constructor reject
/// such a layout alike, including the constructors of arrays that need no buffer (e.g., of
/// [`DataType::Zero`](ryft_core::DataType::Zero)) and the processes that hold no shard of an array.
///
/// # Errors
///
/// Returns [`ArrayError::PlacementTypeMismatch`] if `sharding` disagrees with the sharding of `r#type` over the
/// non-auto mesh axes, the errors of [`ShardLayout::new`] if the type has a layout and the placement is not valid on
/// `mesh`, and the [`TypeError`] of [`validate_shard_layout`] if the layout of `r#type` does not address the shape of
/// a shard.
pub(super) fn resolve_placement(
    r#type: ArrayType,
    sharding: Option<Sharding>,
    mesh: &DeviceMesh,
) -> Result<(ArrayType, Sharding), Error> {
    let sharding = match (sharding, r#type.sharding()) {
        (Some(sharding), Some(type_sharding)) if *type_sharding != sharding.without_auto_axes() => {
            return Err(Error::from(ArrayError::PlacementTypeMismatch {
                type_sharding: type_sharding.clone(),
                sharding,
            }));
        }
        (Some(sharding), _) => sharding,
        (None, Some(type_sharding)) => type_sharding.clone(),
        (None, None) => Sharding::replicated(mesh.logical_mesh().clone(), r#type.rank()),
    };

    let r#type = r#type.with_sharding(sharding.without_auto_axes())?;
    if r#type.layout().is_some() {
        match r#type.static_shape() {
            Some(shape) => {
                let layout = ShardLayout::new(&shape, mesh, &sharding)?;
                validate_shard_layout(&r#type, layout.descriptors().iter().map(ShardDescriptor::shape))?;
            }
            None => {
                // A type without a static shape has no shards, and the constructors reject it right
                // after this function.
                r#type.validate_layout()?
            }
        }
    }

    Ok((r#type, sharding))
}

/// Validates the layout of `r#type`, if any, against `shard_shapes`, which are the shapes of the device shards of an
/// array of that type. A layout of an array describes the physical storage of each of its device shards, and never of
/// the global value, because uploads give every shard buffer that layout, placements convert every shard buffer to it,
/// and a readback reads every shard buffer through it. The layout must therefore address the shape of every shard
/// (refer to [`ArrayType::validate_layout`]), which is checked once for each distinct shard shape.
///
///   - **Tiled Layouts:** A tiled layout is a permutation of the dimensions with tiles, which is valid for the shape
///     of a shard exactly when it is valid for the global shape, because both have the same rank.
///   - **Strided Layouts:** A strided layout holds byte strides, which must keep the elements of the shape that they
///     address apart, so they depend on the extents of a shard. For example, the strides `[8, 4]` describe each dense
///     `f32[4, 2]` column shard of a global `f32[4, 4]` value, while they would overlap the rows of the global shape.
///
/// A layout of the wrong rank addresses no shard, so every constructor and every process rejects it, also when its
/// array has no shard on this process.
///
/// # Errors
///
/// Returns the [`TypeError`] of [`ArrayType::validate_layout`] for the first shard shape that the layout does not
/// address.
pub(super) fn validate_shard_layout(
    r#type: &ArrayType,
    shard_shapes: impl IntoIterator<Item = StaticShape>,
) -> Result<(), TypeError> {
    let Some(layout) = r#type.layout() else {
        return Ok(());
    };

    let mut validated = HashSet::new();
    for shape in shard_shapes {
        if validated.insert(shape.clone()) {
            ArrayType::new(r#type.data_type(), shape.into()).with_layout(layout.clone()).validate_layout()?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::{
        DataType, Device, Dimension, DimensionBounds, DimensionVariable, Error as CoreError, Layout, LogicalMesh,
        MeshAxis, MeshAxisType, Shape, ShardingDimension, StaticShape, StridedLayout, TiledLayout, TypeError,
    };

    use crate::tests::{device_mesh_2x2, logical_mesh_2x2};

    use super::*;

    #[test]
    fn test_array_metadata() {
        let mesh = device_mesh_2x2();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [4]);

        // The metadata agrees with the shard layout of the placement, whose descriptors are returned with it.
        // Its type records none of the placement over the auto mesh axes.
        let (metadata, descriptors) = ArrayMetadata::new(r#type.clone(), Some(sharding.clone()), mesh.clone()).unwrap();
        let layout = ShardLayout::new(&StaticShape::new(vec![4]), &mesh, &sharding).unwrap();
        assert_eq!(metadata.r#type(), &r#type.clone().with_sharding(sharding.without_auto_axes()).unwrap());
        assert_eq!(metadata.sharding(), &sharding);
        assert_eq!(metadata.shard_index_by_device(), layout.shard_index_by_device());
        assert_eq!(metadata.mesh(), &mesh);
        assert_eq!(descriptors, layout.descriptors());

        // Without a placement, an array of a type without sharding is replicated over the mesh.
        let (metadata, _) = ArrayMetadata::new(r#type, None, mesh.clone()).unwrap();
        assert_eq!(metadata.sharding(), &Sharding::replicated(mesh.logical_mesh().clone(), 1));

        // A type without a static shape has no shard layout.
        let dynamic = DimensionVariable::new("dynamic", DimensionBounds::non_negative(Some(10)).unwrap());
        let dynamic_shape = Shape::new(vec![Dimension::Dynamic(dynamic)]);
        assert!(matches!(
            ArrayMetadata::new(ArrayType::new(DataType::F32, dynamic_shape.clone()), None, mesh),
            Err(Error::Array(error)) if matches!(error.as_ref(),
                ArrayError::DynamicShape { shape } if shape == &dynamic_shape,
            ),
        ));

        // A placement that disagrees with the sharding of the type over the non-auto mesh axes is rejected.
        let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let replicated = Sharding::replicated(explicit_mesh.clone(), 1);
        let sharded = Sharding::new(explicit_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(replicated.clone()).unwrap();
        let device_mesh = DeviceMesh::new(explicit_mesh, vec![Device::new(0, 0), Device::new(1, 0)]).unwrap();
        assert!(matches!(
            ArrayMetadata::new(replicated_type, Some(sharded.clone()), device_mesh),
            Err(Error::Array(error)) if matches!(error.as_ref(),
                ArrayError::PlacementTypeMismatch { type_sharding, sharding }
                    if type_sharding == &replicated && sharding == &sharded,
            ),
        ));
    }

    #[test]
    fn test_array_metadata_rejects_invalid_layout() {
        // A layout that does not address the shape of the type is rejected whatever the data type, including for the
        // zero data type, whose arrays have no buffer that would reject it, and for a mesh whose devices all belong to
        // another process.
        let rank_two = Layout::Strided(StridedLayout::new(vec![4, 4]));
        let remote_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(7, 3)],
        )
        .unwrap();
        for (data_type, mesh) in [(DataType::Zero, device_mesh_2x2()), (DataType::F32, remote_mesh)] {
            let r#type = ArrayType::new_static(data_type, [4]).with_layout(rank_two.clone());
            assert!(matches!(
                ArrayMetadata::new(r#type, None, mesh),
                Err(Error::Core(error))
                    if *error == CoreError::Type(TypeError::invalid("strided layout rank 2 does not match array rank 1")),
            ));
        }

        // A layout that addresses the shape is accepted.
        let r#type =
            ArrayType::new_static(DataType::F32, [4]).with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        let (metadata, _) = ArrayMetadata::new(r#type.clone(), None, device_mesh_2x2()).unwrap();
        assert_eq!(metadata.r#type().layout(), r#type.layout());
    }

    #[test]
    fn test_array_metadata_validates_layout_of_each_shard() {
        let mesh = device_mesh_2x2();
        let columns = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],
        )
        .unwrap();
        let strided = |strides: Vec<isize>| {
            ArrayType::new_static(DataType::F32, [4, 4]).with_layout(Layout::Strided(StridedLayout::new(strides)))
        };

        // The byte strides `[8, 4]` describe each dense `f32[4, 2]` column shard of a global `f32[4, 4]` value, while
        // they would overlap the rows of the global shape, so the shards accept them and a replicated value does not.
        let (metadata, descriptors) =
            ArrayMetadata::new(strided(vec![8, 4]), Some(columns.clone()), mesh.clone()).unwrap();
        assert_eq!(metadata.r#type().layout(), strided(vec![8, 4]).layout());
        assert!(descriptors.iter().all(|descriptor| descriptor.shape() == StaticShape::new(vec![4, 2])));
        let overlapping = "strided layout stride 8 on axis 0 is smaller than the 16-byte span occupied by more minor \
                           axes and may alias array elements";
        assert!(matches!(
            ArrayMetadata::new(strided(vec![8, 4]), None, mesh.clone()),
            Err(Error::Core(error)) if *error == CoreError::Type(TypeError::invalid(overlapping)),
        ));

        // Strides that overlap the elements of a shard are rejected for that shard.
        let overlapping = "strided layout stride 8 on axis 1 is smaller than the 16-byte span occupied by more minor \
                           axes and may alias array elements";
        assert!(matches!(
            ArrayMetadata::new(strided(vec![4, 8]), Some(columns), mesh),
            Err(Error::Core(error)) if *error == CoreError::Type(TypeError::invalid(overlapping)),
        ));
    }

    #[test]
    fn test_array_output_plan_new() {
        let mesh = DeviceMesh::new(
            logical_mesh_2x2(),
            vec![Device::new(5, 0), Device::new(3, 0), Device::new(9, 1), Device::new(1, 1)],
        )
        .unwrap();
        let devices = mesh.devices();

        // The executable returns buffers for process 1's devices in reverse mesh order.
        let addressable_device_ids = [devices[3].id(), devices[2].id()];
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let plan = ArrayOutputPlan::new(r#type.clone(), None, &mesh, &addressable_device_ids).unwrap();

        assert_eq!(plan.descriptors().iter().map(ShardDescriptor::device).collect::<Vec<_>>(), devices);
        assert_eq!(plan.buffer_positions(), &[None, None, Some(1), Some(0)]);

        // A device outside the mesh still occupies a buffer position. A repeated device uses its first buffer.
        let addressable_device_ids = [devices[3].id(), 42, devices[2].id(), devices[3].id()];
        let plan = ArrayOutputPlan::new(r#type, None, &mesh, &addressable_device_ids).unwrap();
        assert_eq!(plan.buffer_positions(), &[None, None, Some(2), Some(0)]);
    }

    #[test]
    fn test_validate_shard_layout() {
        let r#type = ArrayType::new_static(DataType::F32, [4, 4]);
        let shards = || [StaticShape::new(vec![4, 2]), StaticShape::new(vec![4, 2])];

        // A type without a layout and a layout that addresses every shard shape are accepted, and so is a tiled
        // layout, which is a permutation of the dimensions and is valid for every shape of its rank.
        assert_eq!(validate_shard_layout(&r#type, shards()), Ok(()));
        let strided = r#type.clone().with_layout(Layout::Strided(StridedLayout::new(vec![8, 4])));
        assert_eq!(validate_shard_layout(&strided, shards()), Ok(()));
        let column_major = r#type.clone().with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])));
        assert_eq!(validate_shard_layout(&column_major, shards()), Ok(()));

        // A layout must address every shard shape, including the shards of an uneven split, and its rank must match.
        let uneven = [StaticShape::new(vec![4, 2]), StaticShape::new(vec![4, 5])];
        assert_eq!(
            validate_shard_layout(&strided, uneven),
            Err(TypeError::invalid(
                "strided layout stride 8 on axis 0 is smaller than the 20-byte span occupied by more minor axes and \
                 may alias array elements",
            )),
        );
        let rank_one = r#type.with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        assert_eq!(
            validate_shard_layout(&rank_one, shards()),
            Err(TypeError::invalid("strided layout rank 1 does not match array rank 2")),
        );
    }
}
