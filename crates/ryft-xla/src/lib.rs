pub mod arrays;
pub mod arrays_v0;
pub mod compilation;
pub mod distributed;
pub mod errors;
pub mod experimental;
pub mod jit;
pub mod mlir;
pub mod pjrt;
pub mod sharding;
pub mod telemetry;
pub mod types;

pub use arrays::{Array, ArrayShard, ShardDescriptor, ShardIndex, ShardLayout};
pub use arrays_v0::ArrayError;
pub use compilation::{CompilationContext, CompilationKey, FunctionFingerprint};
pub use distributed::DistributedRuntime;
pub use errors::Error;
pub use experimental::shard_map::{to, with_sharding_constraint};
pub use jit::{CompiledFunction, JitOptions, eval_shape, jit, jit_with_options};
pub use mlir::ToMlir;
pub use pjrt::{FromPjrt, ToPjrt};
pub use telemetry::live_array_count;

#[cfg(test)]
pub(crate) mod tests {
    use std::mem::MaybeUninit;

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

    pub(crate) fn values_to_bytes<V: Copy>(values: &[V]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(size_of_val(values));
        for value in values {
            let value_bytes = unsafe { std::slice::from_raw_parts(value as *const V as *const u8, size_of::<V>()) };
            bytes.extend_from_slice(value_bytes);
        }
        bytes
    }

    pub(crate) fn values_from_bytes<V: Copy>(bytes: &[u8]) -> Vec<V> {
        assert_eq!(bytes.len() % size_of::<V>(), 0);
        bytes
            .chunks_exact(size_of::<V>())
            .map(|chunk| {
                let mut value = MaybeUninit::<V>::uninit();
                unsafe {
                    std::ptr::copy_nonoverlapping(chunk.as_ptr(), value.as_mut_ptr() as *mut u8, size_of::<V>());
                    value.assume_init()
                }
            })
            .collect()
    }
}
