use ryft_xla_sys::bindings::MlirType;
use ryft_xla_sys::mlir::dialects::mosaic::tpu::{
    mlirTpuDmaSemaphoreTypeGet, mlirTpuFloat8EXMYTypeGet, mlirTpuFloat8EXMYTypeGetUnderlyingType,
    mlirTpuIsADmaSemaphoreType, mlirTpuIsAFloat8EXMYType, mlirTpuIsASemaphoreType, mlirTpuSemaphoreTypeGet,
};

use crate::{Context, DialectHandle, Error, Type, TypeRef, mlir_subtype_trait_impls};

/// Mosaic TPU 8-bit EXMY floating-point [`Type`].
#[derive(Copy, Clone)]
pub struct Float8ExmyTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> Type<'c, 't> for Float8ExmyTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null() && unsafe { mlirTpuIsAFloat8EXMYType(handle) } {
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

mlir_subtype_trait_impls!(Float8ExmyTypeRef<'c, 't> as Type, mlir_type = Type);

impl<'c, 't> Float8ExmyTypeRef<'c, 't> {
    /// Returns the underlying EXMY floating-point type stored in the 8-bit container.
    pub fn underlying_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        unsafe {
            TypeRef::from_c_api(mlirTpuFloat8EXMYTypeGetUnderlyingType(self.handle), self.context)
                .map_err(|_| Error::internal("expected non-null Mosaic TPU float8 EXMY underlying type"))
        }
    }
}

/// Mosaic TPU semaphore [`Type`].
#[derive(Copy, Clone)]
pub struct SemaphoreTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> Type<'c, 't> for SemaphoreTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null() && unsafe { mlirTpuIsASemaphoreType(handle) } {
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

mlir_subtype_trait_impls!(SemaphoreTypeRef<'c, 't> as Type, mlir_type = Type);

/// Mosaic TPU DMA semaphore [`Type`].
#[derive(Copy, Clone)]
pub struct DmaSemaphoreTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> Type<'c, 't> for DmaSemaphoreTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null() && unsafe { mlirTpuIsADmaSemaphoreType(handle) } {
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

mlir_subtype_trait_impls!(DmaSemaphoreTypeRef<'c, 't> as Type, mlir_type = Type);

impl<'t> Context<'t> {
    /// Creates a new Mosaic TPU [`Float8ExmyTypeRef`] owned by this [`Context`].
    pub fn mosaic_tpu_float8_exmy_type<'c, T: Type<'c, 't>>(
        &'c self,
        underlying_type: T,
    ) -> Result<Float8ExmyTypeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::mosaic_tpu()?)?;
        unsafe {
            Float8ExmyTypeRef::from_c_api(
                mlirTpuFloat8EXMYTypeGet(*self.handle.borrow(), underlying_type.to_c_api()),
                self,
            )
            .map_err(|_| Error::invalid_argument("invalid arguments to `Context::mosaic_tpu_float8_exmy_type`"))
        }
    }

    /// Creates a new Mosaic TPU [`SemaphoreTypeRef`] owned by this [`Context`].
    pub fn mosaic_tpu_semaphore_type<'c>(&'c self) -> Result<SemaphoreTypeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::mosaic_tpu()?)?;
        unsafe {
            SemaphoreTypeRef::from_c_api(mlirTpuSemaphoreTypeGet(*self.handle.borrow()), self)
                .map_err(|_| Error::internal("MLIR returned an invalid Mosaic TPU semaphore type"))
        }
    }

    /// Creates a new Mosaic TPU [`DmaSemaphoreTypeRef`] owned by this [`Context`].
    pub fn mosaic_tpu_dma_semaphore_type<'c>(&'c self) -> Result<DmaSemaphoreTypeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::mosaic_tpu()?)?;
        unsafe {
            DmaSemaphoreTypeRef::from_c_api(mlirTpuDmaSemaphoreTypeGet(*self.handle.borrow()), self)
                .map_err(|_| Error::internal("MLIR returned an invalid Mosaic TPU DMA semaphore type"))
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::types::tests::{test_type_casting, test_type_display_and_debug};

    use super::*;

    #[test]
    fn test_float8_exmy_type() {
        let context = Context::new();
        let underlying_type = context.float8e4m3fn_type();
        let r#type = context.mosaic_tpu_float8_exmy_type(underlying_type).unwrap();
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.dialect().unwrap().namespace().unwrap(), "tpu");
        assert_eq!(r#type.underlying_type().unwrap(), underlying_type);
    }

    #[test]
    fn test_float8_exmy_type_equality() {
        let context = Context::new();
        let r#type_1 = context.mosaic_tpu_float8_exmy_type(context.float8e4m3fn_type()).unwrap();
        let r#type_2 = context.mosaic_tpu_float8_exmy_type(context.float8e4m3fn_type()).unwrap();
        assert_eq!(r#type_1, r#type_2);

        let r#type_2 = context.mosaic_tpu_float8_exmy_type(context.float8e5m2_type()).unwrap();
        assert_ne!(r#type_1, r#type_2);

        let context = Context::new();
        let r#type_2 = context.mosaic_tpu_float8_exmy_type(context.float8e4m3fn_type()).unwrap();
        assert_ne!(r#type_1, r#type_2);
    }

    #[test]
    fn test_float8_exmy_type_display_and_debug() {
        let context = Context::new();
        let r#type = context.mosaic_tpu_float8_exmy_type(context.float8e4m3fn_type()).unwrap();
        test_type_display_and_debug(r#type, "!tpu.float8_exmy<f8E4M3FN>");
    }

    #[test]
    fn test_float8_exmy_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_tpu().unwrap()).unwrap();
        let r#type = context.mosaic_tpu_float8_exmy_type(context.float8e4m3fn_type()).unwrap();
        assert_eq!(context.parse_type("!tpu.float8_exmy<f8E4M3FN>").unwrap(), r#type);
    }

    #[test]
    fn test_float8_exmy_type_casting() {
        let context = Context::new();
        let r#type = context.mosaic_tpu_float8_exmy_type(context.float8e4m3fn_type()).unwrap();
        test_type_casting(r#type);
    }

    #[test]
    fn test_semaphore_types() {
        let context = Context::new();
        let semaphore_type = context.mosaic_tpu_semaphore_type().unwrap();
        assert_eq!(&context, semaphore_type.context());
        assert_eq!(semaphore_type.dialect().unwrap().namespace().unwrap(), "tpu");

        let dma_semaphore_type = context.mosaic_tpu_dma_semaphore_type().unwrap();
        assert_eq!(&context, dma_semaphore_type.context());
        assert_eq!(dma_semaphore_type.dialect().unwrap().namespace().unwrap(), "tpu");
        assert_ne!(semaphore_type.as_ref(), dma_semaphore_type.as_ref());
    }

    #[test]
    fn test_semaphore_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.mosaic_tpu_semaphore_type().unwrap(), "!tpu.semaphore");
        test_type_display_and_debug(context.mosaic_tpu_dma_semaphore_type().unwrap(), "!tpu.dma_semaphore");
    }

    #[test]
    fn test_semaphore_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_tpu().unwrap()).unwrap();
        assert_eq!(context.parse_type("!tpu.semaphore").unwrap(), context.mosaic_tpu_semaphore_type().unwrap());
        assert_eq!(context.parse_type("!tpu.dma_semaphore").unwrap(), context.mosaic_tpu_dma_semaphore_type().unwrap());
    }

    #[test]
    fn test_semaphore_type_casting() {
        let context = Context::new();
        test_type_casting(context.mosaic_tpu_semaphore_type().unwrap());
        test_type_casting(context.mosaic_tpu_dma_semaphore_type().unwrap());
    }
}
