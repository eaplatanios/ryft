#![allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]

use crate::bindings::{MlirAttribute, MlirContext, MlirDialectHandle, MlirDialectRegistry, MlirStringRef, MlirType};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum MlirMosaicGpuEnumAttribute {
    RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_DIMENSION = 0,
    RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_SWIZZLING_MODE = 1,
    RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_TMA_REDUCTION = 2,
    RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_OOB_FILL_MODE = 3,
    RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_MULTIMEM_LOAD_REDUCTION_TYPE = 4,
    RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_ATOMIC_OP_TYPE = 5,
}

unsafe extern "C" {
    pub fn mlirGetDialectHandle__mosaic_gpu__() -> MlirDialectHandle;
    pub fn mlirDialectRegistryInsertMosaicGpuInlinerExtensions(registry: MlirDialectRegistry);

    pub fn mlirMosaicGpuIsABarrierType(r#type: MlirType) -> bool;
    pub fn mlirMosaicGpuBarrierTypeGet(context: MlirContext, orders_tensor_core: bool) -> MlirType;
    pub fn mlirMosaicGpuBarrierTypeGetOrdersTensorCore(r#type: MlirType) -> bool;

    pub fn mlirMosaicGpuIsAWGStridedFragLayoutAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirMosaicGpuWGStridedFragLayoutAttrGet(
        context: MlirContext,
        shape: MlirAttribute,
        vector_size: i32,
    ) -> MlirAttribute;
    pub fn mlirMosaicGpuWGStridedFragLayoutAttrGetShape(attribute: MlirAttribute) -> MlirAttribute;
    pub fn mlirMosaicGpuWGStridedFragLayoutAttrGetVectorSize(attribute: MlirAttribute) -> i32;

    pub fn mlirMosaicGpuIsAWGSplatFragLayoutAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirMosaicGpuWGSplatFragLayoutAttrGet(context: MlirContext, shape: MlirAttribute) -> MlirAttribute;
    pub fn mlirMosaicGpuWGSplatFragLayoutAttrGetShape(attribute: MlirAttribute) -> MlirAttribute;

    pub fn mlirMosaicGpuIsAReplicatedAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirMosaicGpuReplicatedAttrGet(context: MlirContext, times: i32) -> MlirAttribute;
    pub fn mlirMosaicGpuReplicatedAttrGetTimes(attribute: MlirAttribute) -> i32;

    pub fn mlirMosaicGpuIsATiledLayoutAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirMosaicGpuTiledLayoutAttrGet(
        context: MlirContext,
        tiling: MlirAttribute,
        warp_dims: MlirAttribute,
        lane_dims: MlirAttribute,
        vector_dim: i32,
    ) -> MlirAttribute;
    pub fn mlirMosaicGpuTiledLayoutAttrGetTiling(attribute: MlirAttribute) -> MlirAttribute;
    pub fn mlirMosaicGpuTiledLayoutAttrGetWarpDims(attribute: MlirAttribute) -> MlirAttribute;
    pub fn mlirMosaicGpuTiledLayoutAttrGetLaneDims(attribute: MlirAttribute) -> MlirAttribute;
    pub fn mlirMosaicGpuTiledLayoutAttrGetVectorDim(attribute: MlirAttribute) -> i32;

    pub fn mlirAttributeIsAMosaicGpuEnumAttr(attribute: MlirAttribute, kind: MlirMosaicGpuEnumAttribute) -> bool;
    pub fn mlirMosaicGpuEnumAttrGet(
        context: MlirContext,
        kind: MlirMosaicGpuEnumAttribute,
        value: MlirStringRef,
    ) -> MlirAttribute;
    pub fn mlirMosaicGpuEnumAttrGetValue(attribute: MlirAttribute, kind: MlirMosaicGpuEnumAttribute) -> MlirStringRef;

    pub fn mlirMosaicGpuIsATileTransformAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirMosaicGpuTileTransformAttrGet(context: MlirContext, tiling: *mut i32, tiling_size: i32)
    -> MlirAttribute;
    pub fn mlirMosaicGpuTileTransformAttrGetTiling(attribute: MlirAttribute) -> MlirAttribute;

    pub fn mlirMosaicGpuIsATransposeTransformAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirMosaicGpuTransposeTransformAttrGet(
        context: MlirContext,
        permutation: *mut i32,
        permutation_size: i32,
    ) -> MlirAttribute;
    pub fn mlirMosaicGpuTransposeTransformAttrGetPermutation(attribute: MlirAttribute) -> MlirAttribute;

    pub fn mlirMosaicGpuIsASwizzleTransformAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirMosaicGpuSwizzleTransformAttrGet(context: MlirContext, swizzle: i32) -> MlirAttribute;
    pub fn mlirMosaicGpuSwizzleTransformAttrGetSwizzle(attribute: MlirAttribute) -> i32;

    pub fn mlirMosaicGpuIsACopyPartitionAttr(attribute: MlirAttribute) -> bool;

    pub fn mlirMosaicGpuIsACopyReplicatedAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirMosaicGpuCopyReplicatedAttrGet(context: MlirContext) -> MlirAttribute;

    pub fn mlirMosaicGpuIsACopyPartitionedAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirMosaicGpuCopyPartitionedAttrGet(context: MlirContext, axis: i32) -> MlirAttribute;
    pub fn mlirMosaicGpuCopyPartitionedAttrGetAxis(attribute: MlirAttribute) -> i32;

    pub fn mlirAttributeIsAMosaicGpuTmemAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirMosaicGpuTmemAttrGet(context: MlirContext) -> MlirAttribute;
}
