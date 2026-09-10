#![allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]

use crate::bindings::{
    MlirAttribute, MlirContext, MlirDialectHandle, MlirDialectRegistry, MlirStringRef, MlirType, MlirTypeID,
};

/// Version of the `stable_mosaic_gpu.version` MLIR bytecode schema supported by the pinned JAX serde pass.
pub const MOSAIC_GPU_SERDE_VERSION: i32 = 8;

/// Version of the serialized `MosaicGpuKernelProto` resource schema consumed by the pinned JAX runtime.
pub const MOSAIC_GPU_RESOURCE_SCHEMA_VERSION: i32 = 3;

/// XLA FFI target registered by the pinned JAX Mosaic GPU runtime.
pub const MOSAIC_GPU_FFI_TARGET: &str = "mosaic_gpu_v2";

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum MlirMosaicGpuEnumAttribute {
    RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_DIMENSION = 0,
    RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_SWIZZLING_MODE = 1,
    RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_TMA_REDUCTION = 2,
    RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_OOB_FILL_MODE = 3,
    RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_MULTIMEM_LOAD_REDUCTION_TYPE = 4,
    RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_ATOMIC_OP_TYPE = 5,
    RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_TMEM_LOAD_REDUCTION = 6,
}

unsafe extern "C" {
    pub fn mlirAttributeIsAMosaicGpuSmemClusterAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirMosaicGpuSmemClusterAttrGet(context: MlirContext) -> MlirAttribute;

    pub fn mlirGetDialectHandle__mosaic_gpu__() -> MlirDialectHandle;
    pub fn mlirDialectRegistryInsertMosaicGpuInlinerExtensions(registry: MlirDialectRegistry);
    pub fn mlirMosaicGpuRegisterSerdePass();

    pub fn mlirMosaicGpuIsAB6x16P32Type(r#type: MlirType) -> bool;
    pub fn mlirMosaicGpuB6x16P32TypeGet(context: MlirContext, element_type: MlirType) -> MlirType;
    pub fn mlirMosaicGpuB6x16P32TypeGetElementType(r#type: MlirType) -> MlirType;
    pub fn mlirMosaicGpuB6x16P32TypeGetTypeID() -> MlirTypeID;

    pub fn mlirMosaicGpuIsAP2B6Type(r#type: MlirType) -> bool;
    pub fn mlirMosaicGpuP2B6TypeGet(context: MlirContext, element_type: MlirType) -> MlirType;
    pub fn mlirMosaicGpuP2B6TypeGetElementType(r#type: MlirType) -> MlirType;
    pub fn mlirMosaicGpuP2B6TypeGetTypeID() -> MlirTypeID;

    pub fn mlirMosaicGpuIsABarrierType(r#type: MlirType) -> bool;
    pub fn mlirMosaicGpuBarrierTypeGet(context: MlirContext, orders_tensor_core: bool) -> MlirType;
    pub fn mlirMosaicGpuBarrierTypeGetOrdersTensorCore(r#type: MlirType) -> bool;
    pub fn mlirMosaicGpuBarrierTypeGetTypeID() -> MlirTypeID;

    pub fn mlirMosaicGpuIsAWGStridedFragLayoutAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirMosaicGpuWGStridedFragLayoutAttrGetTypeID() -> MlirTypeID;
    pub fn mlirMosaicGpuWGStridedFragLayoutAttrGet(
        context: MlirContext,
        shape: MlirAttribute,
        vector_size: i32,
    ) -> MlirAttribute;
    pub fn mlirMosaicGpuWGStridedFragLayoutAttrGetShape(attribute: MlirAttribute) -> MlirAttribute;
    pub fn mlirMosaicGpuWGStridedFragLayoutAttrGetVectorSize(attribute: MlirAttribute) -> i32;

    pub fn mlirMosaicGpuIsAWGSplatFragLayoutAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirMosaicGpuWGSplatFragLayoutAttrGetTypeID() -> MlirTypeID;
    pub fn mlirMosaicGpuWGSplatFragLayoutAttrGet(context: MlirContext, shape: MlirAttribute) -> MlirAttribute;
    pub fn mlirMosaicGpuWGSplatFragLayoutAttrGetShape(attribute: MlirAttribute) -> MlirAttribute;

    pub fn mlirMosaicGpuIsAReplicatedAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirMosaicGpuReplicatedAttrGetTypeID() -> MlirTypeID;
    pub fn mlirMosaicGpuReplicatedAttrGet(context: MlirContext, times: i32) -> MlirAttribute;
    pub fn mlirMosaicGpuReplicatedAttrGetTimes(attribute: MlirAttribute) -> i32;

    pub fn mlirMosaicGpuIsATiledLayoutAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirMosaicGpuTiledLayoutAttrGetTypeID() -> MlirTypeID;
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
    pub fn mlirMosaicGpuTileTransformAttrGetTypeID() -> MlirTypeID;
    pub fn mlirMosaicGpuTileTransformAttrGet(context: MlirContext, tiling: *mut i32, tiling_size: i32)
    -> MlirAttribute;
    pub fn mlirMosaicGpuTileTransformAttrGetTiling(attribute: MlirAttribute) -> MlirAttribute;

    pub fn mlirMosaicGpuIsASwizzleTransformAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirMosaicGpuSwizzleTransformAttrGetTypeID() -> MlirTypeID;
    pub fn mlirMosaicGpuSwizzleTransformAttrGet(context: MlirContext, swizzle: i32) -> MlirAttribute;
    pub fn mlirMosaicGpuSwizzleTransformAttrGetSwizzle(attribute: MlirAttribute) -> i32;

    pub fn mlirMosaicGpuIsACopyPartitionAttr(attribute: MlirAttribute) -> bool;

    pub fn mlirMosaicGpuIsACopyReplicatedAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirMosaicGpuCopyReplicatedAttrGetTypeID() -> MlirTypeID;
    pub fn mlirMosaicGpuCopyReplicatedAttrGet(context: MlirContext) -> MlirAttribute;

    pub fn mlirMosaicGpuIsACopyPartitionedAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirMosaicGpuCopyPartitionedAttrGetTypeID() -> MlirTypeID;
    pub fn mlirMosaicGpuCopyPartitionedAttrGet(context: MlirContext, axis: i32) -> MlirAttribute;
    pub fn mlirMosaicGpuCopyPartitionedAttrGetAxis(attribute: MlirAttribute) -> i32;

    pub fn mlirAttributeIsAMosaicGpuTmemAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirMosaicGpuTmemAttrGet(context: MlirContext) -> MlirAttribute;
}

#[cfg(test)]
mod tests {
    use crate::bindings::mlirTypeIDEqual;

    use super::*;

    #[test]
    fn test_mosaic_gpu_type_ids() {
        let type_ids = unsafe {
            [
                mlirMosaicGpuBarrierTypeGetTypeID(),
                mlirMosaicGpuCopyPartitionedAttrGetTypeID(),
                mlirMosaicGpuCopyReplicatedAttrGetTypeID(),
                mlirMosaicGpuReplicatedAttrGetTypeID(),
                mlirMosaicGpuSwizzleTransformAttrGetTypeID(),
                mlirMosaicGpuTiledLayoutAttrGetTypeID(),
                mlirMosaicGpuTileTransformAttrGetTypeID(),
                mlirMosaicGpuWGSplatFragLayoutAttrGetTypeID(),
                mlirMosaicGpuWGStridedFragLayoutAttrGetTypeID(),
            ]
        };
        for (index, type_id) in type_ids.iter().enumerate() {
            assert!(!type_id.ptr.is_null());
            for previous in &type_ids[..index] {
                assert!(!unsafe { mlirTypeIDEqual(*type_id, *previous) });
            }
        }
    }
}
