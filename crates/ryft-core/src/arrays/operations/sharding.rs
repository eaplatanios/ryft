//! Reference [`Array`] answers to the sharding operation family contracts.
//!
//! A concrete array is a single-device value. The sharding-constraint hint is untracked and therefore leaves both
//! its payload and its type unchanged.

use crate::arrays::arrays::Array;

// TODO(eaplatanios): Review this.

// The sharding-constraint hint is untracked: the output type (sharding included) is identical to the input, so the
// identity default is exactly the `ShardingConstraintOperation` interpretation contract for a concrete value.
impl crate::operations::sharding::ConstrainSharding for Array {}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::sharding::meshes::{LogicalMesh, MeshAxis, MeshAxisType};
    use crate::arrays::sharding::shardings::{Sharding, ShardingDimension};
    use crate::arrays::types::arrays::ArrayType;
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::memories::Memory;
    use crate::operations::{ConstrainSharding, TransferToMemory};
    use crate::programs::Typed;

    use super::*;

    #[test]
    fn test_array_type_metadata_operations() {
        // The sharding, memory, and tagging operations alter only the carried type (or nothing at all): the payload
        // of a concrete single-device array is host-resident metadata-free storage either way.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();

        // Memory transfers re-place the array by updating the memory carried by its type.
        let array = Array::vector(vec![1.0, 2.0]).unwrap();
        let transferred = array.transfer_to_memory(Memory::Host { pinned: true });
        assert_eq!(transferred.r#type().memory(), Memory::Host { pinned: true });
        assert_eq!(transferred.r#type().into_owned().with_memory(Memory::Device), array.r#type().into_owned());
        assert_eq!(transferred.storage_bytes(), array.storage_bytes());

        let input_sharding = Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["m"]).unwrap();
        let input = Array::from_f64s(
            ArrayType::new_static(DataType::F64, [2]).with_sharding(input_sharding).unwrap(),
            vec![1.0, 2.0],
        )
        .unwrap();
        let target = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();

        // The sharding-constraint hint is untracked, so constraining leaves the value (type included) unchanged.
        assert_eq!(input.constrain_sharding(&target), input);
    }
}
