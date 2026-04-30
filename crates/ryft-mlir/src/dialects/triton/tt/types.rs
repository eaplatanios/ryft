use ryft_xla_sys::bindings::MlirType;
use ryft_xla_sys::mlir::dialects::triton::tt::{
    mlirTritonTtPointerTypeGet, mlirTritonTtPointerTypeGetAddressSpace, mlirTritonTtPointerTypeGetPointeeType,
    mlirTritonTtTensorDescTypeGet, mlirTritonTtTensorDescTypeGetBlockType, mlirTypeIsATritonTtPointerType,
    mlirTypeIsATritonTtTensorDescType,
};

use crate::{Context, DialectHandle, TensorTypeRef, Type, TypeRef, mlir_subtype_trait_impls};

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
    pub fn pointee_type(&self) -> TypeRef<'c, 't> {
        unsafe {
            TypeRef::from_c_api(mlirTritonTtPointerTypeGetPointeeType(self.handle), self.context)
                .expect("invalid `!tt.ptr` pointee type")
        }
    }

    /// Returns the numeric Triton address space.
    pub fn address_space(&self) -> i32 {
        unsafe { mlirTritonTtPointerTypeGetAddressSpace(self.handle) }
    }
}

impl<'c, 't> Type<'c, 't> for PointerTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirTypeIsATritonTtPointerType(handle) } {
            Some(Self { handle, context })
        } else {
            None
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

/// Triton `tt` tensor descriptor [`Type`]. Tensor descriptors represent tiled tensor memory access metadata.
///
/// The Triton version pinned by this repository models descriptors with a ranked tensor block type.
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
    /// Returns the ranked tensor block [`Type`] described by this descriptor.
    pub fn block_type(&self) -> TensorTypeRef<'c, 't> {
        unsafe {
            TensorTypeRef::from_c_api(mlirTritonTtTensorDescTypeGetBlockType(self.handle), self.context)
                .expect("invalid `!tt.tensordesc` block type")
        }
    }
}

impl<'c, 't> Type<'c, 't> for TensorDescTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirTypeIsATritonTtTensorDescType(handle) } {
            Some(Self { handle, context })
        } else {
            None
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
    ) -> PointerTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::triton_tt());
        unsafe {
            PointerTypeRef::from_c_api(mlirTritonTtPointerTypeGet(pointee_type.to_c_api(), address_space), self)
                .expect("invalid arguments to `Context::triton_tt_pointer_type`")
        }
    }

    /// Creates a new Triton `tt` [`TensorDescTypeRef`] owned by this [`Context`].
    pub fn triton_tt_tensor_desc_type<'c>(&'c self, block_type: TensorTypeRef<'c, 't>) -> TensorDescTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::triton_tt());
        unsafe {
            TensorDescTypeRef::from_c_api(mlirTritonTtTensorDescTypeGet(block_type.to_c_api()), self)
                .expect("invalid arguments to `Context::triton_tt_tensor_desc_type`")
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
        let pointer_type = context.triton_tt_pointer_type(context.float32_type(), 1);
        assert_eq!(&context, pointer_type.context());
        assert_eq!(pointer_type.dialect().namespace().unwrap(), "tt");
        assert_eq!(pointer_type.pointee_type(), context.float32_type());
        assert_eq!(pointer_type.address_space(), 1);
    }

    #[test]
    fn test_pointer_type_equality() {
        let context = Context::new();

        // Same types from the same context must be equal because they are "uniqued".
        let pointer_type_1 = context.triton_tt_pointer_type(context.float32_type(), 1);
        let pointer_type_2 = context.triton_tt_pointer_type(context.float32_type(), 1);
        assert_eq!(pointer_type_1, pointer_type_2);

        // Different types from the same context must not be equal.
        let pointer_type_2 = context.triton_tt_pointer_type(context.float32_type(), 3);
        assert_ne!(pointer_type_1, pointer_type_2);

        // Same types from different contexts must not be equal.
        let context = Context::new();
        let pointer_type_2 = context.triton_tt_pointer_type(context.float32_type(), 1);
        assert_ne!(pointer_type_1, pointer_type_2);
    }

    #[test]
    fn test_pointer_type_display_and_debug() {
        let context = Context::new();
        let pointer_type = context.triton_tt_pointer_type(context.float32_type(), 1);
        test_type_display_and_debug(pointer_type, "!tt.ptr<f32>");

        let pointer_type = context.triton_tt_pointer_type(context.float32_type(), 3);
        test_type_display_and_debug(pointer_type, "!tt.ptr<f32, 3>");
    }

    #[test]
    fn test_pointer_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::triton_tt());
        let pointer_type = context.triton_tt_pointer_type(context.float32_type(), 1);
        assert_eq!(context.parse_type("!tt.ptr<f32>").unwrap(), pointer_type);
    }

    #[test]
    fn test_pointer_type_casting() {
        let context = Context::new();
        let pointer_type = context.triton_tt_pointer_type(context.float32_type(), 1);
        test_type_casting(pointer_type);
    }

    #[test]
    fn test_tensor_desc_type() {
        let context = Context::new();
        let location = context.unknown_location();
        let block_type = context
            .tensor_type(context.float32_type(), &[Size::Static(16), Size::Static(32)], None, location)
            .unwrap();
        let tensor_desc_type = context.triton_tt_tensor_desc_type(block_type);
        assert_eq!(&context, tensor_desc_type.context());
        assert_eq!(tensor_desc_type.dialect().namespace().unwrap(), "tt");
        assert_eq!(tensor_desc_type.block_type(), block_type);
    }

    #[test]
    fn test_tensor_desc_type_equality() {
        let context = Context::new();
        let location = context.unknown_location();
        let block_type = context
            .tensor_type(context.float32_type(), &[Size::Static(16), Size::Static(32)], None, location)
            .unwrap();

        // Same types from the same context must be equal because they are "uniqued".
        let tensor_desc_type_1 = context.triton_tt_tensor_desc_type(block_type);
        let tensor_desc_type_2 = context.triton_tt_tensor_desc_type(block_type);
        assert_eq!(tensor_desc_type_1, tensor_desc_type_2);

        // Different types from the same context must not be equal.
        let block_type = context
            .tensor_type(context.float32_type(), &[Size::Static(8), Size::Static(32)], None, location)
            .unwrap();
        let tensor_desc_type_2 = context.triton_tt_tensor_desc_type(block_type);
        assert_ne!(tensor_desc_type_1, tensor_desc_type_2);

        // Same types from different contexts must not be equal.
        let context = Context::new();
        let location = context.unknown_location();
        let block_type = context
            .tensor_type(context.float32_type(), &[Size::Static(16), Size::Static(32)], None, location)
            .unwrap();
        let tensor_desc_type_2 = context.triton_tt_tensor_desc_type(block_type);
        assert_ne!(tensor_desc_type_1, tensor_desc_type_2);
    }

    #[test]
    fn test_tensor_desc_type_display_and_debug() {
        let context = Context::new();
        let location = context.unknown_location();
        let block_type = context
            .tensor_type(context.float32_type(), &[Size::Static(16), Size::Static(32)], None, location)
            .unwrap();
        let tensor_desc_type = context.triton_tt_tensor_desc_type(block_type);
        test_type_display_and_debug(tensor_desc_type, "!tt.tensordesc<tensor<16x32xf32>>");
    }

    #[test]
    fn test_tensor_desc_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::triton_tt());
        let location = context.unknown_location();
        let block_type = context
            .tensor_type(context.float32_type(), &[Size::Static(16), Size::Static(32)], None, location)
            .unwrap();
        let tensor_desc_type = context.triton_tt_tensor_desc_type(block_type);
        assert_eq!(context.parse_type("!tt.tensordesc<tensor<16x32xf32>>").unwrap(), tensor_desc_type);
    }

    #[test]
    fn test_tensor_desc_type_casting() {
        let context = Context::new();
        let location = context.unknown_location();
        let block_type = context
            .tensor_type(context.float32_type(), &[Size::Static(16), Size::Static(32)], None, location)
            .unwrap();
        let tensor_desc_type = context.triton_tt_tensor_desc_type(block_type);
        test_type_casting(tensor_desc_type);
    }
}
