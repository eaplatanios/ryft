use ryft_xla_sys::bindings::{MlirType, mlirTypeIsAFloat6E2M3FN, mlirTypeIsAFloat6E3M2FN};
use ryft_xla_sys::mlir::dialects::mosaic::gpu::{
    mlirMosaicGpuB6x16P32TypeGet, mlirMosaicGpuB6x16P32TypeGetElementType, mlirMosaicGpuBarrierTypeGet,
    mlirMosaicGpuBarrierTypeGetOrdersTensorCore, mlirMosaicGpuIsAB6x16P32Type, mlirMosaicGpuIsABarrierType,
    mlirMosaicGpuIsAP2B6Type, mlirMosaicGpuP2B6TypeGet, mlirMosaicGpuP2B6TypeGetElementType,
};

use crate::{Context, DialectHandle, Error, Type, TypeRef, mlir_subtype_trait_impls};

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

/// Mosaic GPU packed [`Type`]. Packs sixteen 6-bit values with 32 padding bits into a 128-bit element.
#[derive(Copy, Clone)]
pub struct B6x16P32TypeRef<'c, 't> {
    /// Handle that represents this type in the MLIR C API.
    handle: MlirType,

    /// Context that owns this type.
    context: &'c Context<'t>,
}

impl<'c, 't> B6x16P32TypeRef<'c, 't> {
    /// Returns the packed 6-bit element type.
    pub fn element_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        unsafe { TypeRef::from_c_api(mlirMosaicGpuB6x16P32TypeGetElementType(self.handle), self.context) }
    }
}

impl<'c, 't> Type<'c, 't> for B6x16P32TypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null() && unsafe { mlirMosaicGpuIsAB6x16P32Type(handle) } {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected a Mosaic GPU `b6x16_p32` type"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirType {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(B6x16P32TypeRef<'c, 't> as Type, mlir_type = Type);

/// Mosaic GPU packed [`Type`]. Stores a 6-bit value in the low bits of an 8-bit element.
#[derive(Copy, Clone)]
pub struct P2B6TypeRef<'c, 't> {
    /// Handle that represents this type in the MLIR C API.
    handle: MlirType,

    /// Context that owns this type.
    context: &'c Context<'t>,
}

impl<'c, 't> P2B6TypeRef<'c, 't> {
    /// Returns the packed 6-bit element type.
    pub fn element_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        unsafe { TypeRef::from_c_api(mlirMosaicGpuP2B6TypeGetElementType(self.handle), self.context) }
    }
}

impl<'c, 't> Type<'c, 't> for P2B6TypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null() && unsafe { mlirMosaicGpuIsAP2B6Type(handle) } {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected a Mosaic GPU `p2b6` type"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirType {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(P2B6TypeRef<'c, 't> as Type, mlir_type = Type);

impl<'t> Context<'t> {
    /// Creates a packed [`P2B6TypeRef`] with a 6-bit floating-point element type in this context.
    pub fn mosaic_gpu_p2b6_type<'c, T: Type<'c, 't>>(&'c self, element_type: T) -> Result<P2B6TypeRef<'c, 't>, Error> {
        if !std::ptr::eq(self, element_type.context()) {
            return Err(Error::invalid_argument("packed type and element type must use the same context"));
        }
        let handle = unsafe { element_type.to_c_api() };
        if !unsafe { mlirTypeIsAFloat6E2M3FN(handle) || mlirTypeIsAFloat6E3M2FN(handle) } {
            return Err(Error::invalid_argument("expected `f6E2M3FN` or `f6E3M2FN` element type"));
        }
        self.load_dialect(DialectHandle::mosaic_gpu()?)?;
        unsafe { P2B6TypeRef::from_c_api(mlirMosaicGpuP2B6TypeGet(*self.handle.borrow_mut(), handle), self) }
    }

    /// Creates a packed [`B6x16P32TypeRef`] with a 6-bit floating-point element type in this context.
    pub fn mosaic_gpu_b6x16_p32_type<'c, T: Type<'c, 't>>(
        &'c self,
        element_type: T,
    ) -> Result<B6x16P32TypeRef<'c, 't>, Error> {
        if !std::ptr::eq(self, element_type.context()) {
            return Err(Error::invalid_argument("packed type and element type must use the same context"));
        }
        let handle = unsafe { element_type.to_c_api() };
        if !unsafe { mlirTypeIsAFloat6E2M3FN(handle) || mlirTypeIsAFloat6E3M2FN(handle) } {
            return Err(Error::invalid_argument("expected `f6E2M3FN` or `f6E3M2FN` element type"));
        }
        self.load_dialect(DialectHandle::mosaic_gpu()?)?;
        unsafe { B6x16P32TypeRef::from_c_api(mlirMosaicGpuB6x16P32TypeGet(*self.handle.borrow_mut(), handle), self) }
    }

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
    fn test_context_mosaic_gpu_p2b6_type() {
        let context = Context::new();
        let element_type = context.float6e2m3fn_type();
        let packed_type = context.mosaic_gpu_p2b6_type(element_type).unwrap();
        assert_eq!(packed_type.element_type().unwrap(), element_type);
        assert_eq!(packed_type.context(), &context);
        assert_eq!(context.parse_type(&packed_type.to_string()).unwrap(), packed_type);
        assert!(matches!(
            context.mosaic_gpu_p2b6_type(context.float32_type()),
            Err(Error::InvalidArgument { message, .. })
                if message == "expected `f6E2M3FN` or `f6E3M2FN` element type",
        ));
        let other_context = Context::new();
        assert!(matches!(
            context.mosaic_gpu_p2b6_type(other_context.float6e2m3fn_type()),
            Err(Error::InvalidArgument { message, .. })
                if message == "packed type and element type must use the same context",
        ));
    }

    #[test]
    fn test_context_mosaic_gpu_b6x16_p32_type() {
        let context = Context::new();
        let element_type = context.float6e3m2fn_type();
        let packed_type = context.mosaic_gpu_b6x16_p32_type(element_type).unwrap();
        assert_eq!(packed_type.element_type().unwrap(), element_type);
        assert_eq!(packed_type.context(), &context);
        assert_eq!(context.parse_type(&packed_type.to_string()).unwrap(), packed_type);
        assert!(matches!(
            context.mosaic_gpu_b6x16_p32_type(context.float32_type()),
            Err(Error::InvalidArgument { message, .. })
                if message == "expected `f6E2M3FN` or `f6E3M2FN` element type",
        ));
        let other_context = Context::new();
        assert!(matches!(
            context.mosaic_gpu_b6x16_p32_type(other_context.float6e3m2fn_type()),
            Err(Error::InvalidArgument { message, .. })
                if message == "packed type and element type must use the same context",
        ));
    }

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
