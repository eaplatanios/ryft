use ryft_xla_sys::bindings::{MlirType, mlirNVGPUTensorMapDescriptorTypeGet, mlirTypeIsANVGPUTensorMapDescriptorType};
use ryft_xla_sys::mlir::dialects::nvgpu::{
    mlirNvgpuDeviceAsyncTokenTypeGet, mlirNvgpuMBarrierGroupTypeGet, mlirNvgpuMBarrierGroupTypeGetMemorySpace,
    mlirNvgpuMBarrierGroupTypeGetNumBarriers, mlirNvgpuMBarrierTokenTypeGet,
    mlirNvgpuTensorMapDescriptorTypeGetInterleave, mlirNvgpuTensorMapDescriptorTypeGetL2Promo,
    mlirNvgpuTensorMapDescriptorTypeGetOob, mlirNvgpuTensorMapDescriptorTypeGetSwizzle,
    mlirNvgpuTensorMapDescriptorTypeGetTensor, mlirNvgpuWarpgroupAccumulatorTypeGet,
    mlirNvgpuWarpgroupAccumulatorTypeGetFragmented, mlirNvgpuWarpgroupMatrixDescriptorTypeGet,
    mlirNvgpuWarpgroupMatrixDescriptorTypeGetTensor, mlirTypeIsANvgpuDeviceAsyncTokenType,
    mlirTypeIsANvgpuMBarrierGroupType, mlirTypeIsANvgpuMBarrierTokenType, mlirTypeIsANvgpuWarpgroupAccumulatorType,
    mlirTypeIsANvgpuWarpgroupMatrixDescriptorType,
};

use crate::{
    Attribute, AttributeRef, Context, DialectHandle, MemRefTypeRef, Type, VectorTypeRef, mlir_subtype_trait_impls,
};

use super::attributes::{TensorMapInterleaveKind, TensorMapL2PromoKind, TensorMapOobKind, TensorMapSwizzleKind};

macro_rules! nvgpu_unit_type {
    ($name:ident, $constructor:ident, $is_a:path, $get:path, $summary:literal, $mnemonic:literal) => {
        #[doc = $summary]
        ///
        /// Refer to the [official MLIR NVGPU dialect documentation](https://mlir.llvm.org/docs/Dialects/NVGPU/#types)
        /// for more information.
        #[derive(Copy, Clone)]
        pub struct $name<'c, 't> {
            /// Handle that represents this [`Type`] in the MLIR C API.
            handle: MlirType,

            /// [`Context`] that owns this [`Type`].
            context: &'c Context<'t>,
        }

        impl<'c, 't> Type<'c, 't> for $name<'c, 't> {
            unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
                if !handle.ptr.is_null() && unsafe { $is_a(handle) } {
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

        mlir_subtype_trait_impls!($name<'c, 't> as Type, mlir_type = Type);

        impl<'t> Context<'t> {
            #[doc = "Creates a new "]
            #[doc = $summary]
            #[doc = " owned by this [`Context`]."]
            pub fn $constructor<'c>(&'c self) -> $name<'c, 't> {
                self.load_dialect(DialectHandle::nvgpu());
                unsafe {
                    $name::from_c_api($get(*self.handle.borrow_mut()), self)
                        .expect(concat!("invalid arguments to `Context::", stringify!($constructor), "`"))
                }
            }
        }
    };
}

nvgpu_unit_type!(
    DeviceAsyncTokenTypeRef,
    nvgpu_device_async_token_type,
    mlirTypeIsANvgpuDeviceAsyncTokenType,
    mlirNvgpuDeviceAsyncTokenTypeGet,
    "NVGPU device asynchronous token [`Type`].",
    "!nvgpu.device.async.token"
);

/// NVGPU mbarrier group [`Type`]. It represents one or more 64-bit mbarrier objects in shared memory.
#[derive(Copy, Clone)]
pub struct MBarrierGroupTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> MBarrierGroupTypeRef<'c, 't> {
    /// Returns the memory space attribute for the represented mbarrier objects.
    pub fn memory_space(&self) -> AttributeRef<'c, 't> {
        unsafe {
            AttributeRef::from_c_api(mlirNvgpuMBarrierGroupTypeGetMemorySpace(self.handle), self.context)
                .expect("invalid `!nvgpu.mbarrier.group` memory space")
        }
    }

    /// Returns the number of mbarrier objects in this group.
    pub fn num_barriers(&self) -> u32 {
        unsafe { mlirNvgpuMBarrierGroupTypeGetNumBarriers(self.handle) }
    }
}

impl<'c, 't> Type<'c, 't> for MBarrierGroupTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirTypeIsANvgpuMBarrierGroupType(handle) } {
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

mlir_subtype_trait_impls!(MBarrierGroupTypeRef<'c, 't> as Type, mlir_type = Type);

nvgpu_unit_type!(
    MBarrierTokenTypeRef,
    nvgpu_mbarrier_token_type,
    mlirTypeIsANvgpuMBarrierTokenType,
    mlirNvgpuMBarrierTokenTypeGet,
    "NVGPU mbarrier token [`Type`].",
    "!nvgpu.mbarrier.token"
);

/// NVGPU tensor map descriptor [`Type`]. It describes Tensor Memory Access tiled memory metadata.
#[derive(Copy, Clone)]
pub struct TensorMapDescriptorTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> TensorMapDescriptorTypeRef<'c, 't> {
    /// Returns the memref type described by this tensor map descriptor.
    pub fn tensor(&self) -> MemRefTypeRef<'c, 't> {
        unsafe {
            MemRefTypeRef::from_c_api(mlirNvgpuTensorMapDescriptorTypeGetTensor(self.handle), self.context)
                .expect("invalid `!nvgpu.tensormap.descriptor` tensor type")
        }
    }

    /// Returns the tensor map swizzle kind.
    pub fn swizzle(&self) -> TensorMapSwizzleKind {
        TensorMapSwizzleKind::from_value(unsafe { mlirNvgpuTensorMapDescriptorTypeGetSwizzle(self.handle) })
            .expect("invalid `!nvgpu.tensormap.descriptor` swizzle kind")
    }

    /// Returns the tensor map L2 promotion kind.
    pub fn l2_promo(&self) -> TensorMapL2PromoKind {
        TensorMapL2PromoKind::from_value(unsafe { mlirNvgpuTensorMapDescriptorTypeGetL2Promo(self.handle) })
            .expect("invalid `!nvgpu.tensormap.descriptor` L2 promotion kind")
    }

    /// Returns the tensor map out-of-bounds fill kind.
    pub fn oob(&self) -> TensorMapOobKind {
        TensorMapOobKind::from_value(unsafe { mlirNvgpuTensorMapDescriptorTypeGetOob(self.handle) })
            .expect("invalid `!nvgpu.tensormap.descriptor` out-of-bounds fill kind")
    }

    /// Returns the tensor map interleave kind.
    pub fn interleave(&self) -> TensorMapInterleaveKind {
        TensorMapInterleaveKind::from_value(unsafe { mlirNvgpuTensorMapDescriptorTypeGetInterleave(self.handle) })
            .expect("invalid `!nvgpu.tensormap.descriptor` interleave kind")
    }
}

impl<'c, 't> Type<'c, 't> for TensorMapDescriptorTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirTypeIsANVGPUTensorMapDescriptorType(handle) } {
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

mlir_subtype_trait_impls!(TensorMapDescriptorTypeRef<'c, 't> as Type, mlir_type = Type);

/// NVGPU warpgroup matrix descriptor [`Type`]. It describes a shared-memory matrix operand for warpgroup MMA.
#[derive(Copy, Clone)]
pub struct WarpgroupMatrixDescriptorTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> WarpgroupMatrixDescriptorTypeRef<'c, 't> {
    /// Returns the memref type described by this warpgroup matrix descriptor.
    pub fn tensor(&self) -> MemRefTypeRef<'c, 't> {
        unsafe {
            MemRefTypeRef::from_c_api(mlirNvgpuWarpgroupMatrixDescriptorTypeGetTensor(self.handle), self.context)
                .expect("invalid `!nvgpu.warpgroup.descriptor` tensor type")
        }
    }
}

impl<'c, 't> Type<'c, 't> for WarpgroupMatrixDescriptorTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirTypeIsANvgpuWarpgroupMatrixDescriptorType(handle) } {
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

mlir_subtype_trait_impls!(WarpgroupMatrixDescriptorTypeRef<'c, 't> as Type, mlir_type = Type);

/// NVGPU warpgroup accumulator [`Type`]. It represents the distributed accumulator fragments owned by a warpgroup.
#[derive(Copy, Clone)]
pub struct WarpgroupAccumulatorTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> WarpgroupAccumulatorTypeRef<'c, 't> {
    /// Returns the fragmented vector type represented by this accumulator.
    pub fn fragmented(&self) -> VectorTypeRef<'c, 't> {
        unsafe {
            VectorTypeRef::from_c_api(mlirNvgpuWarpgroupAccumulatorTypeGetFragmented(self.handle), self.context)
                .expect("invalid `!nvgpu.warpgroup.accumulator` fragmented vector type")
        }
    }
}

impl<'c, 't> Type<'c, 't> for WarpgroupAccumulatorTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirTypeIsANvgpuWarpgroupAccumulatorType(handle) } {
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

mlir_subtype_trait_impls!(WarpgroupAccumulatorTypeRef<'c, 't> as Type, mlir_type = Type);

impl<'t> Context<'t> {
    /// Creates a new NVGPU [`MBarrierGroupTypeRef`] owned by this [`Context`].
    pub fn nvgpu_mbarrier_group_type<'c, A: Attribute<'c, 't>>(
        &'c self,
        memory_space: A,
        num_barriers: u32,
    ) -> MBarrierGroupTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::nvgpu());
        unsafe {
            MBarrierGroupTypeRef::from_c_api(
                mlirNvgpuMBarrierGroupTypeGet(*self.handle.borrow_mut(), memory_space.to_c_api(), num_barriers),
                self,
            )
            .expect("invalid arguments to `Context::nvgpu_mbarrier_group_type`")
        }
    }

    /// Creates a new NVGPU [`TensorMapDescriptorTypeRef`] owned by this [`Context`].
    pub fn nvgpu_tensor_map_descriptor_type<'c>(
        &'c self,
        tensor: MemRefTypeRef<'c, 't>,
        swizzle: TensorMapSwizzleKind,
        l2_promo: TensorMapL2PromoKind,
        oob: TensorMapOobKind,
        interleave: TensorMapInterleaveKind,
    ) -> TensorMapDescriptorTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::nvgpu());
        unsafe {
            TensorMapDescriptorTypeRef::from_c_api(
                mlirNVGPUTensorMapDescriptorTypeGet(
                    *self.handle.borrow_mut(),
                    tensor.to_c_api(),
                    swizzle.value() as std::os::raw::c_int,
                    l2_promo.value() as std::os::raw::c_int,
                    oob.value() as std::os::raw::c_int,
                    interleave.value() as std::os::raw::c_int,
                ),
                self,
            )
            .expect("invalid arguments to `Context::nvgpu_tensor_map_descriptor_type`")
        }
    }

    /// Creates a new NVGPU [`WarpgroupMatrixDescriptorTypeRef`] owned by this [`Context`].
    pub fn nvgpu_warpgroup_matrix_descriptor_type<'c>(
        &'c self,
        tensor: MemRefTypeRef<'c, 't>,
    ) -> WarpgroupMatrixDescriptorTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::nvgpu());
        unsafe {
            WarpgroupMatrixDescriptorTypeRef::from_c_api(
                mlirNvgpuWarpgroupMatrixDescriptorTypeGet(*self.handle.borrow_mut(), tensor.to_c_api()),
                self,
            )
            .expect("invalid arguments to `Context::nvgpu_warpgroup_matrix_descriptor_type`")
        }
    }

    /// Creates a new NVGPU [`WarpgroupAccumulatorTypeRef`] owned by this [`Context`].
    pub fn nvgpu_warpgroup_accumulator_type<'c>(
        &'c self,
        fragmented: VectorTypeRef<'c, 't>,
    ) -> WarpgroupAccumulatorTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::nvgpu());
        unsafe {
            WarpgroupAccumulatorTypeRef::from_c_api(
                mlirNvgpuWarpgroupAccumulatorTypeGet(*self.handle.borrow_mut(), fragmented.to_c_api()),
                self,
            )
            .expect("invalid arguments to `Context::nvgpu_warpgroup_accumulator_type`")
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::dialects::gpu::AddressSpace;
    use crate::types::tests::{test_type_casting, test_type_display_and_debug};
    use crate::{Size, VectorTypeDimension};

    use super::*;

    #[test]
    fn test_device_async_token_type() {
        let context = Context::new();
        let token_type = context.nvgpu_device_async_token_type();
        assert_eq!(&context, token_type.context());
        assert_eq!(token_type.dialect().namespace().unwrap(), "nvgpu");
    }

    #[test]
    fn test_device_async_token_type_equality() {
        let context = Context::new();
        let type_1 = context.nvgpu_device_async_token_type();
        let type_2 = context.nvgpu_device_async_token_type();
        assert_eq!(type_1, type_2);

        let context = Context::new();
        let type_2 = context.nvgpu_device_async_token_type();
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_device_async_token_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.nvgpu_device_async_token_type(), "!nvgpu.device.async.token");
    }

    #[test]
    fn test_device_async_token_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::nvgpu());
        let token_type = context.nvgpu_device_async_token_type();
        assert_eq!(context.parse_type("!nvgpu.device.async.token").unwrap(), token_type);
    }

    #[test]
    fn test_device_async_token_type_casting() {
        let context = Context::new();
        test_type_casting(context.nvgpu_device_async_token_type());
    }

    #[test]
    fn test_mbarrier_group_type() {
        let context = Context::new();
        let memory_space = context.gpu_address_space_attribute(AddressSpace::Workgroup);
        let group_type = context.nvgpu_mbarrier_group_type(memory_space, 4);
        assert_eq!(&context, group_type.context());
        assert_eq!(group_type.dialect().namespace().unwrap(), "nvgpu");
        assert_eq!(group_type.memory_space(), memory_space);
        assert_eq!(group_type.num_barriers(), 4);
    }

    #[test]
    fn test_mbarrier_group_type_equality() {
        let context = Context::new();
        let memory_space = context.gpu_address_space_attribute(AddressSpace::Workgroup);
        let type_1 = context.nvgpu_mbarrier_group_type(memory_space, 4);
        let type_2 = context.nvgpu_mbarrier_group_type(memory_space, 4);
        assert_eq!(type_1, type_2);

        let type_2 = context.nvgpu_mbarrier_group_type(memory_space, 2);
        assert_ne!(type_1, type_2);

        let context = Context::new();
        let memory_space = context.gpu_address_space_attribute(AddressSpace::Workgroup);
        let type_2 = context.nvgpu_mbarrier_group_type(memory_space, 4);
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_mbarrier_group_type_display_and_debug() {
        let context = Context::new();
        let memory_space = context.gpu_address_space_attribute(AddressSpace::Workgroup);
        let group_type = context.nvgpu_mbarrier_group_type(memory_space, 4);
        test_type_display_and_debug(
            group_type,
            "!nvgpu.mbarrier.group<memorySpace = #gpu.address_space<workgroup>, num_barriers = 4>",
        );
    }

    #[test]
    fn test_mbarrier_group_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::gpu());
        context.load_dialect(DialectHandle::nvgpu());
        let memory_space = context.gpu_address_space_attribute(AddressSpace::Workgroup);
        let group_type = context.nvgpu_mbarrier_group_type(memory_space, 4);
        assert_eq!(
            context
                .parse_type("!nvgpu.mbarrier.group<memorySpace = #gpu.address_space<workgroup>, num_barriers = 4>")
                .unwrap(),
            group_type
        );
    }

    #[test]
    fn test_mbarrier_group_type_casting() {
        let context = Context::new();
        let memory_space = context.gpu_address_space_attribute(AddressSpace::Workgroup);
        test_type_casting(context.nvgpu_mbarrier_group_type(memory_space, 4));
    }

    #[test]
    fn test_mbarrier_token_type() {
        let context = Context::new();
        let token_type = context.nvgpu_mbarrier_token_type();
        assert_eq!(&context, token_type.context());
        assert_eq!(token_type.dialect().namespace().unwrap(), "nvgpu");
    }

    #[test]
    fn test_mbarrier_token_type_equality() {
        let context = Context::new();
        let type_1 = context.nvgpu_mbarrier_token_type();
        let type_2 = context.nvgpu_mbarrier_token_type();
        assert_eq!(type_1, type_2);

        let context = Context::new();
        let type_2 = context.nvgpu_mbarrier_token_type();
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_mbarrier_token_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.nvgpu_mbarrier_token_type(), "!nvgpu.mbarrier.token");
    }

    #[test]
    fn test_mbarrier_token_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::nvgpu());
        let token_type = context.nvgpu_mbarrier_token_type();
        assert_eq!(context.parse_type("!nvgpu.mbarrier.token").unwrap(), token_type);
    }

    #[test]
    fn test_mbarrier_token_type_casting() {
        let context = Context::new();
        test_type_casting(context.nvgpu_mbarrier_token_type());
    }

    #[test]
    fn test_tensor_map_descriptor_type() {
        let context = Context::new();
        let location = context.unknown_location();
        let tensor = context
            .mem_ref_type(context.float32_type(), &[Size::Static(64), Size::Static(128)], None, None, location)
            .unwrap();
        let descriptor_type = context.nvgpu_tensor_map_descriptor_type(
            tensor,
            TensorMapSwizzleKind::Swizzle128B,
            TensorMapL2PromoKind::L2Promo64B,
            TensorMapOobKind::Zero,
            TensorMapInterleaveKind::None,
        );
        assert_eq!(&context, descriptor_type.context());
        assert_eq!(descriptor_type.dialect().namespace().unwrap(), "nvgpu");
        assert_eq!(descriptor_type.tensor(), tensor);
        assert_eq!(descriptor_type.swizzle(), TensorMapSwizzleKind::Swizzle128B);
        assert_eq!(descriptor_type.l2_promo(), TensorMapL2PromoKind::L2Promo64B);
        assert_eq!(descriptor_type.oob(), TensorMapOobKind::Zero);
        assert_eq!(descriptor_type.interleave(), TensorMapInterleaveKind::None);
    }

    #[test]
    fn test_tensor_map_descriptor_type_equality() {
        let context = Context::new();
        let location = context.unknown_location();
        let tensor = context.mem_ref_type(context.float32_type(), &[Size::Static(64)], None, None, location).unwrap();
        let type_1 = context.nvgpu_tensor_map_descriptor_type(
            tensor,
            TensorMapSwizzleKind::None,
            TensorMapL2PromoKind::None,
            TensorMapOobKind::Zero,
            TensorMapInterleaveKind::None,
        );
        let type_2 = context.nvgpu_tensor_map_descriptor_type(
            tensor,
            TensorMapSwizzleKind::None,
            TensorMapL2PromoKind::None,
            TensorMapOobKind::Zero,
            TensorMapInterleaveKind::None,
        );
        assert_eq!(type_1, type_2);

        let type_2 = context.nvgpu_tensor_map_descriptor_type(
            tensor,
            TensorMapSwizzleKind::Swizzle32B,
            TensorMapL2PromoKind::None,
            TensorMapOobKind::Zero,
            TensorMapInterleaveKind::None,
        );
        assert_ne!(type_1, type_2);

        let context = Context::new();
        let location = context.unknown_location();
        let tensor = context.mem_ref_type(context.float32_type(), &[Size::Static(64)], None, None, location).unwrap();
        let type_2 = context.nvgpu_tensor_map_descriptor_type(
            tensor,
            TensorMapSwizzleKind::None,
            TensorMapL2PromoKind::None,
            TensorMapOobKind::Zero,
            TensorMapInterleaveKind::None,
        );
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_tensor_map_descriptor_type_display_and_debug() {
        let context = Context::new();
        let location = context.unknown_location();
        let tensor = context.mem_ref_type(context.float32_type(), &[Size::Static(64)], None, None, location).unwrap();
        let descriptor_type = context.nvgpu_tensor_map_descriptor_type(
            tensor,
            TensorMapSwizzleKind::None,
            TensorMapL2PromoKind::None,
            TensorMapOobKind::Zero,
            TensorMapInterleaveKind::None,
        );
        test_type_display_and_debug(
            descriptor_type,
            "!nvgpu.tensormap.descriptor<tensor = memref<64xf32>, swizzle = none, l2promo = none, oob = zero, interleave = none>",
        );
    }

    #[test]
    fn test_tensor_map_descriptor_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::nvgpu());
        let location = context.unknown_location();
        let tensor = context.mem_ref_type(context.float32_type(), &[Size::Static(64)], None, None, location).unwrap();
        let descriptor_type = context.nvgpu_tensor_map_descriptor_type(
            tensor,
            TensorMapSwizzleKind::None,
            TensorMapL2PromoKind::None,
            TensorMapOobKind::Zero,
            TensorMapInterleaveKind::None,
        );
        assert_eq!(
            context
                .parse_type("!nvgpu.tensormap.descriptor<tensor = memref<64xf32>, swizzle = none, l2promo = none, oob = zero, interleave = none>")
                .unwrap(),
            descriptor_type
        );
    }

    #[test]
    fn test_tensor_map_descriptor_type_casting() {
        let context = Context::new();
        let location = context.unknown_location();
        let tensor = context.mem_ref_type(context.float32_type(), &[Size::Static(64)], None, None, location).unwrap();
        let descriptor_type = context.nvgpu_tensor_map_descriptor_type(
            tensor,
            TensorMapSwizzleKind::None,
            TensorMapL2PromoKind::None,
            TensorMapOobKind::Zero,
            TensorMapInterleaveKind::None,
        );
        test_type_casting(descriptor_type);
    }

    #[test]
    fn test_warpgroup_matrix_descriptor_type() {
        let context = Context::new();
        let location = context.unknown_location();
        let tensor = context.mem_ref_type(context.float16_type(), &[Size::Static(64)], None, None, location).unwrap();
        let descriptor_type = context.nvgpu_warpgroup_matrix_descriptor_type(tensor);
        assert_eq!(&context, descriptor_type.context());
        assert_eq!(descriptor_type.dialect().namespace().unwrap(), "nvgpu");
        assert_eq!(descriptor_type.tensor(), tensor);
    }

    #[test]
    fn test_warpgroup_matrix_descriptor_type_equality() {
        let context = Context::new();
        let location = context.unknown_location();
        let tensor = context.float16_type();
        let memref_1 = context.mem_ref_type(tensor, &[Size::Static(64)], None, None, location).unwrap();
        let memref_2 = context.mem_ref_type(tensor, &[Size::Static(128)], None, None, location).unwrap();
        let type_1 = context.nvgpu_warpgroup_matrix_descriptor_type(memref_1);
        let type_2 = context.nvgpu_warpgroup_matrix_descriptor_type(memref_1);
        assert_eq!(type_1, type_2);

        let type_2 = context.nvgpu_warpgroup_matrix_descriptor_type(memref_2);
        assert_ne!(type_1, type_2);

        let context = Context::new();
        let location = context.unknown_location();
        let memref = context.mem_ref_type(context.float16_type(), &[Size::Static(64)], None, None, location).unwrap();
        let type_2 = context.nvgpu_warpgroup_matrix_descriptor_type(memref);
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_warpgroup_matrix_descriptor_type_display_and_debug() {
        let context = Context::new();
        let location = context.unknown_location();
        let tensor = context.mem_ref_type(context.float16_type(), &[Size::Static(64)], None, None, location).unwrap();
        let descriptor_type = context.nvgpu_warpgroup_matrix_descriptor_type(tensor);
        test_type_display_and_debug(descriptor_type, "!nvgpu.warpgroup.descriptor<tensor = memref<64xf16>>");
    }

    #[test]
    fn test_warpgroup_matrix_descriptor_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::nvgpu());
        let location = context.unknown_location();
        let tensor = context.mem_ref_type(context.float16_type(), &[Size::Static(64)], None, None, location).unwrap();
        let descriptor_type = context.nvgpu_warpgroup_matrix_descriptor_type(tensor);
        assert_eq!(
            context.parse_type("!nvgpu.warpgroup.descriptor<tensor = memref<64xf16>>").unwrap(),
            descriptor_type
        );
    }

    #[test]
    fn test_warpgroup_matrix_descriptor_type_casting() {
        let context = Context::new();
        let location = context.unknown_location();
        let tensor = context.mem_ref_type(context.float16_type(), &[Size::Static(64)], None, None, location).unwrap();
        test_type_casting(context.nvgpu_warpgroup_matrix_descriptor_type(tensor));
    }

    #[test]
    fn test_warpgroup_accumulator_type() {
        let context = Context::new();
        let location = context.unknown_location();
        let fragmented =
            context.vector_type(context.float32_type(), &[VectorTypeDimension::Fixed(64)], location).unwrap();
        let accumulator_type = context.nvgpu_warpgroup_accumulator_type(fragmented);
        assert_eq!(&context, accumulator_type.context());
        assert_eq!(accumulator_type.dialect().namespace().unwrap(), "nvgpu");
        assert_eq!(accumulator_type.fragmented(), fragmented);
    }

    #[test]
    fn test_warpgroup_accumulator_type_equality() {
        let context = Context::new();
        let location = context.unknown_location();
        let vector_1 =
            context.vector_type(context.float32_type(), &[VectorTypeDimension::Fixed(64)], location).unwrap();
        let vector_2 =
            context.vector_type(context.float32_type(), &[VectorTypeDimension::Fixed(32)], location).unwrap();
        let type_1 = context.nvgpu_warpgroup_accumulator_type(vector_1);
        let type_2 = context.nvgpu_warpgroup_accumulator_type(vector_1);
        assert_eq!(type_1, type_2);

        let type_2 = context.nvgpu_warpgroup_accumulator_type(vector_2);
        assert_ne!(type_1, type_2);

        let context = Context::new();
        let location = context.unknown_location();
        let vector = context.vector_type(context.float32_type(), &[VectorTypeDimension::Fixed(64)], location).unwrap();
        let type_2 = context.nvgpu_warpgroup_accumulator_type(vector);
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_warpgroup_accumulator_type_display_and_debug() {
        let context = Context::new();
        let location = context.unknown_location();
        let fragmented =
            context.vector_type(context.float32_type(), &[VectorTypeDimension::Fixed(64)], location).unwrap();
        let accumulator_type = context.nvgpu_warpgroup_accumulator_type(fragmented);
        test_type_display_and_debug(accumulator_type, "!nvgpu.warpgroup.accumulator<fragmented = vector<64xf32>>");
    }

    #[test]
    fn test_warpgroup_accumulator_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::nvgpu());
        let location = context.unknown_location();
        let fragmented =
            context.vector_type(context.float32_type(), &[VectorTypeDimension::Fixed(64)], location).unwrap();
        let accumulator_type = context.nvgpu_warpgroup_accumulator_type(fragmented);
        assert_eq!(
            context.parse_type("!nvgpu.warpgroup.accumulator<fragmented = vector<64xf32>>").unwrap(),
            accumulator_type
        );
    }

    #[test]
    fn test_warpgroup_accumulator_type_casting() {
        let context = Context::new();
        let location = context.unknown_location();
        let fragmented =
            context.vector_type(context.float32_type(), &[VectorTypeDimension::Fixed(64)], location).unwrap();
        test_type_casting(context.nvgpu_warpgroup_accumulator_type(fragmented));
    }
}
