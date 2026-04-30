#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "mlir-c/AffineMap.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

enum MlirGpuSparseHandleType {
  RYFT_MLIR_GPU_SPARSE_DN_TENSOR_HANDLE_TYPE = 0,
  RYFT_MLIR_GPU_SPARSE_SP_MAT_HANDLE_TYPE = 1,
  RYFT_MLIR_GPU_SPARSE_SP_GEMM_OPERATION_HANDLE_TYPE = 2,
};

enum MlirGpuEnumAttribute {
  RYFT_MLIR_GPU_ENUM_ATTRIBUTE_ADDRESS_SPACE = 0,
  RYFT_MLIR_GPU_ENUM_ATTRIBUTE_DIMENSION = 1,
  RYFT_MLIR_GPU_ENUM_ATTRIBUTE_ALL_REDUCE_OPERATION_KIND = 2,
  RYFT_MLIR_GPU_ENUM_ATTRIBUTE_SHUFFLE_MODE = 3,
  RYFT_MLIR_GPU_ENUM_ATTRIBUTE_MMA_ELEMENTWISE_OPERATION = 4,
  RYFT_MLIR_GPU_ENUM_ATTRIBUTE_PRUNE_2_TO_4_SPARSE_MATRIX_FLAG = 5,
  RYFT_MLIR_GPU_ENUM_ATTRIBUTE_MATRIX_TRANSPOSE_MODE = 6,
  RYFT_MLIR_GPU_ENUM_ATTRIBUTE_SP_GEMM_WORK_KIND = 7,
  RYFT_MLIR_GPU_ENUM_ATTRIBUTE_BROADCAST_TYPE = 8,
};

enum MlirGpuMappingAttribute {
  RYFT_MLIR_GPU_MAPPING_ATTRIBUTE_BLOCK = 0,
  RYFT_MLIR_GPU_MAPPING_ATTRIBUTE_WARPGROUP = 1,
  RYFT_MLIR_GPU_MAPPING_ATTRIBUTE_WARP = 2,
  RYFT_MLIR_GPU_MAPPING_ATTRIBUTE_THREAD = 3,
  RYFT_MLIR_GPU_MAPPING_ATTRIBUTE_LANE = 4,
};

bool mlirTypeIsAGpuMmaMatrixType(MlirType type);
MlirType mlirGpuMmaMatrixTypeGet(
    MlirType elementType,
    const int64_t *shape,
    intptr_t shapeSize,
    MlirStringRef operand);
intptr_t mlirGpuMmaMatrixTypeGetNumDims(MlirType type);
int64_t mlirGpuMmaMatrixTypeGetDimSize(MlirType type, intptr_t dimension);
MlirType mlirGpuMmaMatrixTypeGetElementType(MlirType type);
MlirStringRef mlirGpuMmaMatrixTypeGetOperand(MlirType type);

bool mlirTypeIsAGpuSparseHandleType(MlirType type, enum MlirGpuSparseHandleType kind);
MlirType mlirGpuSparseHandleTypeGet(MlirContext context, enum MlirGpuSparseHandleType kind);

bool mlirAttributeIsAGpuKernelMetadataAttr(MlirAttribute attribute);
MlirAttribute mlirGpuKernelMetadataAttrGet(
    MlirStringRef name,
    MlirType functionType,
    MlirAttribute argumentAttributes,
    MlirAttribute metadata);

bool mlirAttributeIsAGpuKernelTableAttr(MlirAttribute attribute);
MlirAttribute mlirGpuKernelTableAttrGet(
    MlirContext context,
    intptr_t kernelCount,
    const MlirAttribute *kernels);

bool mlirAttributeIsAGpuSelectObjectAttr(MlirAttribute attribute);
MlirAttribute mlirGpuSelectObjectAttrGet(MlirContext context, MlirAttribute target);

bool mlirAttributeIsAGpuEnumAttr(MlirAttribute attribute, enum MlirGpuEnumAttribute kind);
MlirAttribute mlirGpuEnumAttrGet(
    MlirContext context,
    enum MlirGpuEnumAttribute kind,
    MlirStringRef value);
MlirStringRef mlirGpuEnumAttrGetValue(MlirAttribute attribute, enum MlirGpuEnumAttribute kind);

bool mlirAttributeIsAGpuMappingAttr(MlirAttribute attribute, enum MlirGpuMappingAttribute kind);
MlirAttribute mlirGpuMappingAttrGet(
    MlirContext context,
    enum MlirGpuMappingAttribute kind,
    MlirStringRef value);
MlirStringRef mlirGpuMappingAttrGetValue(MlirAttribute attribute, enum MlirGpuMappingAttribute kind);

bool mlirAttributeIsAGpuMappingMaskAttr(MlirAttribute attribute);
MlirAttribute mlirGpuMappingMaskAttrGet(MlirContext context, uint64_t mask);
uint64_t mlirGpuMappingMaskAttrGetMask(MlirAttribute attribute);

bool mlirAttributeIsAGpuMemorySpaceMappingAttr(MlirAttribute attribute);
MlirAttribute mlirGpuMemorySpaceMappingAttrGet(MlirContext context, MlirStringRef addressSpace);
MlirStringRef mlirGpuMemorySpaceMappingAttrGetAddressSpace(MlirAttribute attribute);

bool mlirAttributeIsAGpuParallelLoopDimMappingAttr(MlirAttribute attribute);
MlirAttribute mlirGpuParallelLoopDimMappingAttrGet(
    MlirContext context,
    MlirStringRef processor,
    MlirAffineMap map,
    MlirAffineMap bound);
MlirStringRef mlirGpuParallelLoopDimMappingAttrGetProcessor(MlirAttribute attribute);
MlirAffineMap mlirGpuParallelLoopDimMappingAttrGetMap(MlirAttribute attribute);
MlirAffineMap mlirGpuParallelLoopDimMappingAttrGetBound(MlirAttribute attribute);

#ifdef __cplusplus
}
#endif
