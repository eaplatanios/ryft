use std::collections::HashMap;

use ryft_core::{ArrayType, DeviceId, DeviceMesh, Sharding};

use crate::arrays::shards::ShardIndex;

/// Type and runtime placement of an [`Array`](crate::Array), which is everything about an array that does not depend
/// on its device buffers. It is immutable and shared. An array, its clones, the [`LiveArray`](crate::LiveArray)s and
/// [`RetainedArray`](crate::RetainedArray)s created from it, and the other arrays that a compiled program produces for
/// the same output all refer to one [`ArrayMetadata`], so sharing it costs a single reference count rather than one for
/// each of its fields.
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
    /// Creates a new [`ArrayMetadata`]. The arguments must describe one placement (i.e., `sharding` must be
    /// over the logical mesh of `mesh` and agree with the sharding of `r#type` over the non-auto mesh axes, and
    /// `shard_index_by_device` must be the index of the [`ShardLayout`](crate::ShardLayout) that they imply). This
    /// function does not validate that, because its callers derive the arguments from a layout that they validated.
    #[inline]
    pub(crate) fn new(
        r#type: ArrayType,
        sharding: Sharding,
        shard_index_by_device: HashMap<DeviceId, ShardIndex>,
        mesh: DeviceMesh,
    ) -> Self {
        Self { r#type, sharding, shard_index_by_device, mesh }
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

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::DataType;

    use crate::tests::device_mesh_2x2;

    use super::*;

    #[test]
    fn test_array_metadata() {
        let mesh = device_mesh_2x2();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let r#type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding.clone()).unwrap();
        let shard_index_by_device = HashMap::from([(0, 0), (1, 1), (2, 2), (3, 3)]);
        let metadata =
            ArrayMetadata::new(r#type.clone(), sharding.clone(), shard_index_by_device.clone(), mesh.clone());
        assert_eq!(metadata.r#type(), &r#type);
        assert_eq!(metadata.sharding(), &sharding);
        assert_eq!(metadata.shard_index_by_device(), &shard_index_by_device);
        assert_eq!(metadata.mesh(), &mesh);
    }
}
