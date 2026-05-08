use ryft_xla_sys::bindings::MlirType;
use ryft_xla_sys::mlir::dialects::triton::tt::{
    mlirTritonTtPointerTypeGet, mlirTritonTtPointerTypeGetAddressSpace, mlirTritonTtPointerTypeGetPointeeType,
    mlirTritonTtTensorDescTypeGet, mlirTritonTtTensorDescTypeGetBlockType, mlirTritonTtTensorDescTypeGetDimSize,
    mlirTritonTtTensorDescTypeGetElementType, mlirTritonTtTensorDescTypeGetNumDims,
    mlirTritonTtTensorDescTypeGetSharedLayout, mlirTypeIsATritonTtPointerType, mlirTypeIsATritonTtTensorDescType,
};

use crate::{
    Attribute, AttributeRef, Context, DialectHandle, Error, Size, TensorTypeRef, Type, TypeRef,
    mlir_subtype_trait_impls,
};

/// Triton `tt` pointer [`Type`]. Pointer types represent addresses in a Triton address space and may point only to
/// scalar element types.
///
/// Refer to the [official Triton dialect documentation](https://triton-lang.org/main/dialects/TritonDialect.html)
/// for more information.
#[derive(Copy, Clone)]
pub struct PointerTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> PointerTypeRef<'c, 't> {
    /// Returns the pointee [`Type`] of this pointer.
    pub fn pointee_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        unsafe {
            TypeRef::from_c_api(mlirTritonTtPointerTypeGetPointeeType(self.handle), self.context)
                .map_err(|_| Error::internal("invalid `!tt.ptr` pointee type"))
        }
    }

    /// Returns the numeric Triton address space.
    pub fn address_space(&self) -> i32 {
        unsafe { mlirTritonTtPointerTypeGetAddressSpace(self.handle) }
    }
}

impl<'c, 't> Type<'c, 't> for PointerTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null() && unsafe { mlirTypeIsATritonTtPointerType(handle) } {
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

mlir_subtype_trait_impls!(PointerTypeRef<'c, 't> as Type, mlir_type = Type);

/// Triton `tt` tensor descriptor [`Type`]. Tensor descriptors represent tiled tensor memory access metadata and are
/// parameterized by a block shape, an element [`Type`], and an optional shared-memory layout attribute.
///
/// Refer to the [official Triton dialect documentation](https://triton-lang.org/main/dialects/TritonDialect.html)
/// for more information.
#[derive(Copy, Clone)]
pub struct TensorDescTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> TensorDescTypeRef<'c, 't> {
    /// Returns the block shape described by this descriptor.
    pub fn shape(&self) -> Vec<Size> {
        let dimension_count = unsafe { mlirTritonTtTensorDescTypeGetNumDims(self.handle) };
        (0..dimension_count)
            .map(|dimension| unsafe { Size::from_c_api(mlirTritonTtTensorDescTypeGetDimSize(self.handle, dimension)) })
            .collect()
    }

    /// Returns the element [`Type`] described by this descriptor.
    pub fn element_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        unsafe {
            TypeRef::from_c_api(mlirTritonTtTensorDescTypeGetElementType(self.handle), self.context)
                .map_err(|_| Error::internal("invalid `!tt.tensordesc` element type"))
        }
    }

    /// Returns the optional shared-memory layout attribute described by this descriptor.
    pub fn shared_layout(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        let handle = unsafe { mlirTritonTtTensorDescTypeGetSharedLayout(self.handle) };
        if handle.ptr.is_null() {
            Ok(None)
        } else {
            unsafe { AttributeRef::from_c_api(handle, self.context).map(Some) }
        }
    }

    /// Returns the ranked tensor block [`Type`] derived from this descriptor's shape and element type.
    pub fn block_type(&self) -> Result<TensorTypeRef<'c, 't>, Error> {
        unsafe {
            TensorTypeRef::from_c_api(mlirTritonTtTensorDescTypeGetBlockType(self.handle), self.context)
                .map_err(|_| Error::internal("invalid `!tt.tensordesc` block type"))
        }
    }
}

impl<'c, 't> Type<'c, 't> for TensorDescTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null() && unsafe { mlirTypeIsATritonTtTensorDescType(handle) } {
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

mlir_subtype_trait_impls!(TensorDescTypeRef<'c, 't> as Type, mlir_type = Type);

impl<'t> Context<'t> {
    /// Creates a new Triton `tt` [`PointerTypeRef`] owned by this [`Context`].
    pub fn triton_tt_pointer_type<'c, T: Type<'c, 't>>(
        &'c self,
        pointee_type: T,
        address_space: i32,
    ) -> Result<PointerTypeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::triton_tt()?)?;
        unsafe {
            PointerTypeRef::from_c_api(mlirTritonTtPointerTypeGet(pointee_type.to_c_api(), address_space), self)
                .map_err(|_| Error::invalid_argument("invalid arguments to `Context::triton_tt_pointer_type`"))
        }
    }

    /// Creates a new Triton `tt` [`TensorDescTypeRef`] owned by this [`Context`].
    pub fn triton_tt_tensor_desc_type<'c, T: Type<'c, 't>>(
        &'c self,
        shape: &[Size],
        element_type: T,
        shared_layout: Option<AttributeRef<'c, 't>>,
    ) -> Result<TensorDescTypeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::triton_tt()?)?;
        let dimensions = shape.iter().map(|dimension| unsafe { dimension.to_c_api() }).collect::<Vec<_>>();
        unsafe {
            TensorDescTypeRef::from_c_api(
                mlirTritonTtTensorDescTypeGet(
                    dimensions.as_ptr(),
                    dimensions.len().cast_signed(),
                    element_type.to_c_api(),
                    shared_layout.unwrap_or_else(|| self.null_attribute()).to_c_api(),
                ),
                self,
            )
            .map_err(|_| Error::invalid_argument("invalid arguments to `Context::triton_tt_tensor_desc_type`"))
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::Size;
    use crate::types::tests::{test_type_casting, test_type_display_and_debug};

    use super::*;

    #[test]
    fn test_pointer_type() {
        let context = Context::new();
        let pointer_type = context.triton_tt_pointer_type(context.float32_type(), 1).unwrap();
        assert_eq!(&context, pointer_type.context());
        assert_eq!(pointer_type.dialect().unwrap().namespace().unwrap(), "tt");
        assert_eq!(pointer_type.pointee_type().unwrap(), context.float32_type());
        assert_eq!(pointer_type.address_space(), 1);
    }

    #[test]
    fn test_pointer_type_equality() {
        let context = Context::new();

        // Same types from the same context must be equal because they are "uniqued".
        let pointer_type_1 = context.triton_tt_pointer_type(context.float32_type(), 1).unwrap();
        let pointer_type_2 = context.triton_tt_pointer_type(context.float32_type(), 1).unwrap();
        assert_eq!(pointer_type_1, pointer_type_2);

        // Different types from the same context must not be equal.
        let pointer_type_2 = context.triton_tt_pointer_type(context.float32_type(), 3).unwrap();
        assert_ne!(pointer_type_1, pointer_type_2);

        // Same types from different contexts must not be equal.
        let context = Context::new();
        let pointer_type_2 = context.triton_tt_pointer_type(context.float32_type(), 1).unwrap();
        assert_ne!(pointer_type_1, pointer_type_2);
    }

    #[test]
    fn test_pointer_type_display_and_debug() {
        let context = Context::new();
        let pointer_type = context.triton_tt_pointer_type(context.float32_type(), 1).unwrap();
        test_type_display_and_debug(pointer_type, "!tt.ptr<f32>");

        let pointer_type = context.triton_tt_pointer_type(context.float32_type(), 3).unwrap();
        test_type_display_and_debug(pointer_type, "!tt.ptr<f32, 3>");
    }

    #[test]
    fn test_pointer_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::triton_tt().unwrap()).unwrap();
        let pointer_type = context.triton_tt_pointer_type(context.float32_type(), 1).unwrap();
        assert_eq!(context.parse_type("!tt.ptr<f32>").unwrap(), pointer_type);
    }

    #[test]
    fn test_pointer_type_casting() {
        let context = Context::new();
        let pointer_type = context.triton_tt_pointer_type(context.float32_type(), 1).unwrap();
        test_type_casting(pointer_type);
    }

    #[test]
    fn test_tensor_desc_type() {
        let context = Context::new();
        let location = context.unknown_location();
        let shape = [Size::Static(16), Size::Static(32)];
        let block_type = context.tensor_type(context.float32_type(), &shape, None, location).unwrap();
        let tensor_desc_type = context.triton_tt_tensor_desc_type(&shape, context.float32_type(), None).unwrap();
        assert_eq!(&context, tensor_desc_type.context());
        assert_eq!(tensor_desc_type.dialect().unwrap().namespace().unwrap(), "tt");
        assert_eq!(tensor_desc_type.shape(), shape.to_vec());
        assert_eq!(tensor_desc_type.element_type().unwrap(), context.float32_type());
        assert_eq!(tensor_desc_type.shared_layout().unwrap(), None);
        assert_eq!(tensor_desc_type.block_type().unwrap(), block_type);
    }

    #[test]
    fn test_tensor_desc_type_equality() {
        let context = Context::new();
        let shape = [Size::Static(16), Size::Static(32)];

        // Same types from the same context must be equal because they are "uniqued".
        let tensor_desc_type_1 = context.triton_tt_tensor_desc_type(&shape, context.float32_type(), None).unwrap();
        let tensor_desc_type_2 = context.triton_tt_tensor_desc_type(&shape, context.float32_type(), None).unwrap();
        assert_eq!(tensor_desc_type_1, tensor_desc_type_2);

        // Different types from the same context must not be equal.
        let shape = [Size::Static(8), Size::Static(32)];
        let tensor_desc_type_2 = context.triton_tt_tensor_desc_type(&shape, context.float32_type(), None).unwrap();
        assert_ne!(tensor_desc_type_1, tensor_desc_type_2);

        // Same types from different contexts must not be equal.
        let context = Context::new();
        let shape = [Size::Static(16), Size::Static(32)];
        let tensor_desc_type_2 = context.triton_tt_tensor_desc_type(&shape, context.float32_type(), None).unwrap();
        assert_ne!(tensor_desc_type_1, tensor_desc_type_2);
    }

    #[test]
    fn test_tensor_desc_type_display_and_debug() {
        let context = Context::new();
        let shape = [Size::Static(16), Size::Static(32)];
        let tensor_desc_type = context.triton_tt_tensor_desc_type(&shape, context.float32_type(), None).unwrap();
        test_type_display_and_debug(tensor_desc_type, "!tt.tensordesc<16x32xf32>");
    }

    #[test]
    fn test_tensor_desc_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::triton_tt().unwrap()).unwrap();
        let shape = [Size::Static(16), Size::Static(32)];
        let tensor_desc_type = context.triton_tt_tensor_desc_type(&shape, context.float32_type(), None).unwrap();
        assert_eq!(context.parse_type("!tt.tensordesc<16x32xf32>").unwrap(), tensor_desc_type);
    }

    #[test]
    fn test_tensor_desc_type_casting() {
        let context = Context::new();
        let shape = [Size::Static(16), Size::Static(32)];
        let tensor_desc_type = context.triton_tt_tensor_desc_type(&shape, context.float32_type(), None).unwrap();
        test_type_casting(tensor_desc_type);
    }
}
