//! Reference [`Array`] answers to the sharding operation family contracts.
//!
//! A concrete array is a single-device value, so both sharding operations leave its payload untouched. Resharding
//! still records the requested distribution metadata on the carried type, while the sharding-constraint hint is
//! untracked and therefore a complete identity.

use crate::arrays::arrays::Array;

// TODO(eaplatanios): Review this.

// An `Array` is a concrete single-device value, so resharding is a no-op on its payload. Its type still records the
// requested distribution metadata — mirroring the `ReshardOperation` type-inference rule, which carries the input's
// varying manual axes over to the target sharding — so interpreted programs preserve their declared boundaries
// exactly. The infallible capability signature makes an invalid target sharding a panic rather than an error, which
// the type-level validation performed before interpretation rules out for staged programs.
impl crate::operations::sharding::Reshard for Array {
    fn reshard(&self, sharding: &crate::arrays::Sharding) -> Self {
        let varying_manual_axes =
            self.r#type.sharding().map(|sharding| sharding.varying_manual_axes().clone()).unwrap_or_default();
        let sharding = sharding
            .clone()
            .with_varying_manual_axes(varying_manual_axes)
            .unwrap_or_else(|error| panic!("{error}"));
        let r#type = self.r#type.clone().with_sharding(sharding).unwrap_or_else(|error| panic!("{error}"));
        Self { r#type, bytes: self.bytes.clone() }
    }
}

// The sharding-constraint hint is untracked: the output type (sharding included) is identical to the input, so the
// identity default is exactly the `ShardingConstraintOperation` interpretation contract for a concrete value.
impl crate::operations::sharding::ConstrainSharding for Array {}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::array_type;
    use crate::arrays::sharding::meshes::{LogicalMesh, MeshAxis, MeshAxisType};
    use crate::arrays::sharding::shardings::{Sharding, ShardingDimension};
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::memories::Memory;
    use crate::operations::{ConstrainSharding, Reshard, TransferToMemory};
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
        let array = Array::vector(vec![1.0, 2.0]);
        let transferred = array.transfer_to_memory(Memory::Host { pinned: true });
        assert_eq!(transferred.r#type().memory(), Memory::Host { pinned: true });
        assert_eq!(transferred.r#type().into_owned().with_memory(Memory::Device), array.r#type().into_owned());
        assert_eq!(transferred.storage_bytes(), array.storage_bytes());

        // Resharding records the requested distribution metadata on the type, carrying the input's varying manual
        // axes over to the target sharding exactly like the `ReshardOperation` type-inference rule.
        let input_sharding = Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["m"]).unwrap();
        let input =
            Array::from_f64s(array_type(DataType::F64, &[2]).with_sharding(input_sharding).unwrap(), vec![1.0, 2.0]);
        let target = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        let resharded = input.reshard(&target);
        assert_eq!(resharded.r#type().sharding(), Some(&target.clone().with_varying_manual_axes(["m"]).unwrap()),);
        assert_eq!(resharded.storage_bytes(), input.storage_bytes());

        // The sharding-constraint hint is untracked, so constraining leaves the value (type included) unchanged.
        assert_eq!(input.constrain_sharding(&target), input);
    }
}
