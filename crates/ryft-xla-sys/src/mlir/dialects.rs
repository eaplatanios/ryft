pub mod affine;
pub mod arith;
pub mod bufferization;
pub mod builtin;
pub mod complex;
pub mod gpu;
pub mod llvm;
pub mod mosaic;
pub mod nvgpu;
pub mod shape;
pub mod sparse_tensor;
pub mod transform;
pub mod triton;
pub mod ub;

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use crate::bindings::{
        MlirDialectHandle, mlirContextCreateWithRegistry, mlirContextDestroy, mlirDialectHandleGetNamespace,
        mlirDialectHandleInsertDialect, mlirDialectHandleLoadDialect, mlirDialectHandleRegisterDialect,
        mlirDialectRegistryCreate, mlirDialectRegistryDestroy, mlirGetDialectHandle__math__,
        mlirGetDialectHandle__vector__,
    };

    use super::*;

    fn dialect_namespace(handle: MlirDialectHandle) -> String {
        let dialect_namespace = unsafe { mlirDialectHandleGetNamespace(handle) };
        let bytes =
            unsafe { std::slice::from_raw_parts(dialect_namespace.data.cast::<u8>(), dialect_namespace.length) };
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    #[test]
    fn test_common_compiler_dialect_handles() {
        let dialects = unsafe {
            [
                (mlirGetDialectHandle__math__(), "math"),
                (mlirGetDialectHandle__vector__(), "vector"),
                (complex::mlirGetDialectHandle__complex__(), "complex"),
                (ub::mlirGetDialectHandle__ub__(), "ub"),
                (bufferization::mlirGetDialectHandle__bufferization__(), "bufferization"),
            ]
        };

        let registry = unsafe { mlirDialectRegistryCreate() };
        assert_ne!(registry.ptr, std::ptr::null_mut::<c_void>());
        for (handle, expected_namespace) in dialects {
            assert_ne!(handle.ptr, std::ptr::null());
            assert_eq!(dialect_namespace(handle), expected_namespace);
            unsafe { mlirDialectHandleInsertDialect(handle, registry) };
        }

        let context = unsafe { mlirContextCreateWithRegistry(registry, false) };
        assert_ne!(context.ptr, std::ptr::null_mut());
        for (handle, _) in dialects {
            let dialect = unsafe { mlirDialectHandleLoadDialect(handle, context) };
            assert_ne!(dialect.ptr, std::ptr::null_mut());
            unsafe { mlirDialectHandleRegisterDialect(handle, context) };
        }

        unsafe {
            mlirContextDestroy(context);
            mlirDialectRegistryDestroy(registry);
        }
    }
}
