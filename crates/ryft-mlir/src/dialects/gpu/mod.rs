//! The `gpu` dialect provides middle-level abstractions for launching GPU kernels following a programming model similar
//! to that of CUDA or OpenCL. It provides abstractions for kernel invocations (and may eventually provide those for
//! device management) that are not present at the lower level (e.g., as LLVM IR intrinsics for GPUs). Its goal is to
//! abstract away device- and driver-specific manipulations for launching GPU kernels and provide a simple path towards
//! GPU execution from MLIR. It may be targeted, for example, by DSLs using MLIR.
//!
//! This dialect also abstracts away primitives commonly available in GPU code, such as with `gpu.thread_id`
//! (an [`Operation`](crate::Operation) that returns the ID of threads within a thread block/workgroup along a given
//! dimension). While the compilation pipelines documented below expect such code to live inside a `gpu.module` and
//! `gpu.func`, these intrinsic wrappers may be used outside of this context.
//!
//! Intrinsic-wrapping operations should not expect that they have a parent of type `gpu.func`. However, operations
//! that deal in compiling and launching GPU functions, like `gpu.launch_func` or `gpu.binary` may assume that the
//! dialect's full layering is being used.
//!
//! Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/GPU/) for more information.
use ryft_xla_sys::bindings::mlirGetDialectHandle__gpu__;

use crate::{DialectHandle, Error};

pub mod attributes;
pub mod operations;
pub mod passes;
pub mod types;

pub use attributes::*;
pub use operations::*;
pub use passes::*;
pub use types::*;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the `gpu` [`Dialect`](crate::Dialect).
    pub fn gpu() -> Result<Self, Error> {
        unsafe { Self::from_c_api(mlirGetDialectHandle__gpu__()) }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_gpu_dialect() {
        let handle = DialectHandle::gpu().unwrap();
        assert_eq!(handle.namespace().unwrap(), "gpu");

        // Check that registration works (both in the context and in a registry).
        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        // Check that loading works.
        let context = Context::new();
        let dialect_1 = context.load_dialect(handle).unwrap();
        assert_eq!(dialect_1.namespace().unwrap(), "gpu");

        // Check that comparison works.
        let dialect_2 = context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        assert_eq!(dialect_1, dialect_2);
    }
}
