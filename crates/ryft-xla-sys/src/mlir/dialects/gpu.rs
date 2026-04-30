#![allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]

use crate::bindings::{MlirAffineMap, MlirAttribute, MlirContext, MlirStringRef, MlirType};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum MlirGpuSparseHandleType {
    RYFT_MLIR_GPU_SPARSE_DN_TENSOR_HANDLE_TYPE = 0,
    RYFT_MLIR_GPU_SPARSE_SP_MAT_HANDLE_TYPE = 1,
    RYFT_MLIR_GPU_SPARSE_SP_GEMM_OPERATION_HANDLE_TYPE = 2,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum MlirGpuEnumAttribute {
    RYFT_MLIR_GPU_ENUM_ATTRIBUTE_ADDRESS_SPACE = 0,
    RYFT_MLIR_GPU_ENUM_ATTRIBUTE_DIMENSION = 1,
    RYFT_MLIR_GPU_ENUM_ATTRIBUTE_ALL_REDUCE_OPERATION_KIND = 2,
    RYFT_MLIR_GPU_ENUM_ATTRIBUTE_SHUFFLE_MODE = 3,
    RYFT_MLIR_GPU_ENUM_ATTRIBUTE_MMA_ELEMENTWISE_OPERATION = 4,
    RYFT_MLIR_GPU_ENUM_ATTRIBUTE_PRUNE_2_TO_4_SPARSE_MATRIX_FLAG = 5,
    RYFT_MLIR_GPU_ENUM_ATTRIBUTE_MATRIX_TRANSPOSE_MODE = 6,
    RYFT_MLIR_GPU_ENUM_ATTRIBUTE_SP_GEMM_WORK_KIND = 7,
    RYFT_MLIR_GPU_ENUM_ATTRIBUTE_BROADCAST_TYPE = 8,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum MlirGpuMappingAttribute {
    RYFT_MLIR_GPU_MAPPING_ATTRIBUTE_BLOCK = 0,
    RYFT_MLIR_GPU_MAPPING_ATTRIBUTE_WARPGROUP = 1,
    RYFT_MLIR_GPU_MAPPING_ATTRIBUTE_WARP = 2,
    RYFT_MLIR_GPU_MAPPING_ATTRIBUTE_THREAD = 3,
    RYFT_MLIR_GPU_MAPPING_ATTRIBUTE_LANE = 4,
}

unsafe extern "C" {
    pub fn mlirTypeIsAGpuMmaMatrixType(r#type: MlirType) -> bool;
    pub fn mlirGpuMmaMatrixTypeGet(
        element_type: MlirType,
        shape: *const i64,
        shape_size: isize,
        operand: MlirStringRef,
    ) -> MlirType;
    pub fn mlirGpuMmaMatrixTypeGetNumDims(r#type: MlirType) -> isize;
    pub fn mlirGpuMmaMatrixTypeGetDimSize(r#type: MlirType, dimension: isize) -> i64;
    pub fn mlirGpuMmaMatrixTypeGetElementType(r#type: MlirType) -> MlirType;
    pub fn mlirGpuMmaMatrixTypeGetOperand(r#type: MlirType) -> MlirStringRef;

    pub fn mlirTypeIsAGpuSparseHandleType(r#type: MlirType, kind: MlirGpuSparseHandleType) -> bool;
    pub fn mlirGpuSparseHandleTypeGet(context: MlirContext, kind: MlirGpuSparseHandleType) -> MlirType;

    pub fn mlirAttributeIsAGpuKernelMetadataAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirGpuKernelMetadataAttrGet(
        name: MlirStringRef,
        function_type: MlirType,
        argument_attributes: MlirAttribute,
        metadata: MlirAttribute,
    ) -> MlirAttribute;

    pub fn mlirAttributeIsAGpuKernelTableAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirGpuKernelTableAttrGet(
        context: MlirContext,
        kernel_count: isize,
        kernels: *const MlirAttribute,
    ) -> MlirAttribute;

    pub fn mlirAttributeIsAGpuSelectObjectAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirGpuSelectObjectAttrGet(context: MlirContext, target: MlirAttribute) -> MlirAttribute;

    pub fn mlirAttributeIsAGpuEnumAttr(attribute: MlirAttribute, kind: MlirGpuEnumAttribute) -> bool;
    pub fn mlirGpuEnumAttrGet(context: MlirContext, kind: MlirGpuEnumAttribute, value: MlirStringRef) -> MlirAttribute;
    pub fn mlirGpuEnumAttrGetValue(attribute: MlirAttribute, kind: MlirGpuEnumAttribute) -> MlirStringRef;

    pub fn mlirAttributeIsAGpuMappingAttr(attribute: MlirAttribute, kind: MlirGpuMappingAttribute) -> bool;
    pub fn mlirGpuMappingAttrGet(
        context: MlirContext,
        kind: MlirGpuMappingAttribute,
        value: MlirStringRef,
    ) -> MlirAttribute;
    pub fn mlirGpuMappingAttrGetValue(attribute: MlirAttribute, kind: MlirGpuMappingAttribute) -> MlirStringRef;

    pub fn mlirAttributeIsAGpuMappingMaskAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirGpuMappingMaskAttrGet(context: MlirContext, mask: u64) -> MlirAttribute;
    pub fn mlirGpuMappingMaskAttrGetMask(attribute: MlirAttribute) -> u64;

    pub fn mlirAttributeIsAGpuMemorySpaceMappingAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirGpuMemorySpaceMappingAttrGet(context: MlirContext, address_space: MlirStringRef) -> MlirAttribute;
    pub fn mlirGpuMemorySpaceMappingAttrGetAddressSpace(attribute: MlirAttribute) -> MlirStringRef;

    pub fn mlirAttributeIsAGpuParallelLoopDimMappingAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirGpuParallelLoopDimMappingAttrGet(
        context: MlirContext,
        processor: MlirStringRef,
        map: MlirAffineMap,
        bound: MlirAffineMap,
    ) -> MlirAttribute;
    pub fn mlirGpuParallelLoopDimMappingAttrGetProcessor(attribute: MlirAttribute) -> MlirStringRef;
    pub fn mlirGpuParallelLoopDimMappingAttrGetMap(attribute: MlirAttribute) -> MlirAffineMap;
    pub fn mlirGpuParallelLoopDimMappingAttrGetBound(attribute: MlirAttribute) -> MlirAffineMap;
}
