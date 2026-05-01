#![allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]

use crate::bindings::{MlirAttribute, MlirContext, MlirType};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum MlirNvgpuEnumAttribute {
    RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_TENSOR_MAP_SWIZZLE_KIND = 0,
    RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_TENSOR_MAP_L2_PROMO_KIND = 1,
    RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_TENSOR_MAP_OOB_KIND = 2,
    RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_TENSOR_MAP_INTERLEAVE_KIND = 3,
    RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_RCP_ROUNDING_MODE = 4,
}

unsafe extern "C" {
    pub fn mlirTypeIsANvgpuDeviceAsyncTokenType(r#type: MlirType) -> bool;
    pub fn mlirNvgpuDeviceAsyncTokenTypeGet(context: MlirContext) -> MlirType;

    pub fn mlirTypeIsANvgpuMBarrierGroupType(r#type: MlirType) -> bool;
    pub fn mlirNvgpuMBarrierGroupTypeGet(
        context: MlirContext,
        memory_space: MlirAttribute,
        num_barriers: u32,
    ) -> MlirType;
    pub fn mlirNvgpuMBarrierGroupTypeGetMemorySpace(r#type: MlirType) -> MlirAttribute;
    pub fn mlirNvgpuMBarrierGroupTypeGetNumBarriers(r#type: MlirType) -> u32;

    pub fn mlirTypeIsANvgpuMBarrierTokenType(r#type: MlirType) -> bool;
    pub fn mlirNvgpuMBarrierTokenTypeGet(context: MlirContext) -> MlirType;

    pub fn mlirNvgpuTensorMapDescriptorTypeGetTensor(r#type: MlirType) -> MlirType;
    pub fn mlirNvgpuTensorMapDescriptorTypeGetSwizzle(r#type: MlirType) -> u32;
    pub fn mlirNvgpuTensorMapDescriptorTypeGetL2Promo(r#type: MlirType) -> u32;
    pub fn mlirNvgpuTensorMapDescriptorTypeGetOob(r#type: MlirType) -> u32;
    pub fn mlirNvgpuTensorMapDescriptorTypeGetInterleave(r#type: MlirType) -> u32;

    pub fn mlirTypeIsANvgpuWarpgroupMatrixDescriptorType(r#type: MlirType) -> bool;
    pub fn mlirNvgpuWarpgroupMatrixDescriptorTypeGet(context: MlirContext, tensor: MlirType) -> MlirType;
    pub fn mlirNvgpuWarpgroupMatrixDescriptorTypeGetTensor(r#type: MlirType) -> MlirType;

    pub fn mlirTypeIsANvgpuWarpgroupAccumulatorType(r#type: MlirType) -> bool;
    pub fn mlirNvgpuWarpgroupAccumulatorTypeGet(context: MlirContext, fragmented: MlirType) -> MlirType;
    pub fn mlirNvgpuWarpgroupAccumulatorTypeGetFragmented(r#type: MlirType) -> MlirType;

    pub fn mlirAttributeIsANvgpuEnumAttr(attribute: MlirAttribute, kind: MlirNvgpuEnumAttribute) -> bool;
    pub fn mlirNvgpuEnumAttrGet(context: MlirContext, kind: MlirNvgpuEnumAttribute, value: u32) -> MlirAttribute;
    pub fn mlirNvgpuEnumAttrGetValue(attribute: MlirAttribute, kind: MlirNvgpuEnumAttribute) -> u32;
}
