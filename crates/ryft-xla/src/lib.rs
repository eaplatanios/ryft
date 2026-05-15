pub mod arrays;
pub mod arrays_v0;
pub mod errors;
pub mod experimental;
pub mod mlir;
pub mod pjrt;
pub mod sharding;
pub mod types;

pub use arrays::{Array, ArrayShard, ShardDescriptor, ShardIndex, ShardLayout};
pub use arrays_v0::ArrayError;
pub use errors::Error;
pub use mlir::ToMlir;
pub use pjrt::{FromPjrt, ToPjrt};

#[cfg(test)]
pub(crate) mod tests {
    use ryft_core::sharding::{Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType};

    pub(crate) fn logical_mesh_2x2() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap()
    }

    pub(crate) fn logical_mesh_3x2x1() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 3, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("z", 1, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap()
    }

    pub(crate) fn device_mesh_2x2() -> DeviceMesh {
        DeviceMesh::new(
            logical_mesh_2x2(),
            vec![Device::new(0, 0), Device::new(1, 0), Device::new(2, 1), Device::new(3, 1)],
        )
        .unwrap()
    }
}
