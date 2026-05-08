use ryft_xla_sys::bindings::MlirType;
use ryft_xla_sys::mlir::dialects::mosaic::gpu::{
    mlirMosaicGpuBarrierTypeGet, mlirMosaicGpuBarrierTypeGetOrdersTensorCore, mlirMosaicGpuIsABarrierType,
};

use crate::{Context, DialectHandle, Error, Type, mlir_subtype_trait_impls};

/// Mosaic GPU barrier [`Type`]. Barriers are used in shared memory to synchronize GPU threads, asynchronous transfers,
/// and optionally tensor-core operations.
#[derive(Copy, Clone)]
pub struct BarrierTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> Type<'c, 't> for BarrierTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null() && unsafe { mlirMosaicGpuIsABarrierType(handle) } {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected MLIR type handle"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirType {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(BarrierTypeRef<'c, 't> as Type, mlir_type = Type);

impl BarrierTypeRef<'_, '_> {
    /// Returns whether this barrier type can order tensor-core operations.
    pub fn orders_tensor_core(&self) -> bool {
        unsafe { mlirMosaicGpuBarrierTypeGetOrdersTensorCore(self.handle) }
    }
}

impl<'t> Context<'t> {
    /// Creates a new Mosaic GPU [`BarrierTypeRef`] owned by this [`Context`].
    pub fn mosaic_gpu_barrier_type<'c>(&'c self, orders_tensor_core: bool) -> Result<BarrierTypeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::mosaic_gpu()?)?;
        unsafe {
            BarrierTypeRef::from_c_api(mlirMosaicGpuBarrierTypeGet(*self.handle.borrow(), orders_tensor_core), self)
                .map_err(|_| Error::internal("MLIR returned an invalid Mosaic GPU barrier type"))
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::types::tests::{test_type_casting, test_type_display_and_debug};

    use super::*;

    #[test]
    fn test_barrier_type() {
        let context = Context::new();
        let barrier_type = context.mosaic_gpu_barrier_type(false).unwrap();
        assert_eq!(&context, barrier_type.context());
        assert_eq!(barrier_type.dialect().unwrap().namespace().unwrap(), "mosaic_gpu");
        assert_eq!(barrier_type.orders_tensor_core(), false);

        let tensor_core_barrier_type = context.mosaic_gpu_barrier_type(true).unwrap();
        assert_eq!(tensor_core_barrier_type.orders_tensor_core(), true);
        assert_ne!(barrier_type, tensor_core_barrier_type);
    }

    #[test]
    fn test_barrier_type_equality() {
        let context = Context::new();

        // Same types from the same context must be equal because they are "uniqued".
        let barrier_type_1 = context.mosaic_gpu_barrier_type(false).unwrap();
        let barrier_type_2 = context.mosaic_gpu_barrier_type(false).unwrap();
        assert_eq!(barrier_type_1, barrier_type_2);

        // Same types from different contexts must not be equal.
        let context = Context::new();
        let barrier_type_2 = context.mosaic_gpu_barrier_type(false).unwrap();
        assert_ne!(barrier_type_1, barrier_type_2);
    }

    #[test]
    fn test_barrier_type_display_and_debug() {
        let context = Context::new();
        let barrier_type = context.mosaic_gpu_barrier_type(false).unwrap();
        test_type_display_and_debug(barrier_type, "!mosaic_gpu.barrier");

        let tensor_core_barrier_type = context.mosaic_gpu_barrier_type(true).unwrap();
        test_type_display_and_debug(tensor_core_barrier_type, "!mosaic_gpu.barrier<orders_tensor_core = true>");
    }

    #[test]
    fn test_barrier_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        let barrier_type = context.mosaic_gpu_barrier_type(false).unwrap();
        assert_eq!(context.parse_type("!mosaic_gpu.barrier").unwrap(), barrier_type);

        let tensor_core_barrier_type = context.mosaic_gpu_barrier_type(true).unwrap();
        assert_eq!(
            context.parse_type("!mosaic_gpu.barrier<orders_tensor_core = true>").unwrap(),
            tensor_core_barrier_type
        );
    }

    #[test]
    fn test_barrier_type_casting() {
        let context = Context::new();
        let barrier_type = context.mosaic_gpu_barrier_type(false).unwrap();
        test_type_casting(barrier_type);
    }
}
