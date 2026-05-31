#pragma once

#include "../../common.h"

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

RYFT_XLA_SYS_EXPORT bool mlirTypeIsAGpuMmaMatrixType(MlirType type);
RYFT_XLA_SYS_EXPORT MlirType mlirGpuMmaMatrixTypeGet(
    MlirType elementType,
    const int64_t *shape,
    intptr_t shapeSize,
    MlirStringRef operand);
RYFT_XLA_SYS_EXPORT intptr_t mlirGpuMmaMatrixTypeGetNumDims(MlirType type);
RYFT_XLA_SYS_EXPORT int64_t mlirGpuMmaMatrixTypeGetDimSize(MlirType type, intptr_t dimension);
RYFT_XLA_SYS_EXPORT MlirType mlirGpuMmaMatrixTypeGetElementType(MlirType type);
RYFT_XLA_SYS_EXPORT MlirStringRef mlirGpuMmaMatrixTypeGetOperand(MlirType type);

RYFT_XLA_SYS_EXPORT bool mlirTypeIsAGpuSparseHandleType(MlirType type, enum MlirGpuSparseHandleType kind);
RYFT_XLA_SYS_EXPORT MlirType mlirGpuSparseHandleTypeGet(MlirContext context, enum MlirGpuSparseHandleType kind);

RYFT_XLA_SYS_EXPORT bool mlirAttributeIsAGpuKernelMetadataAttr(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirGpuKernelMetadataAttrGet(
    MlirStringRef name,
    MlirType functionType,
    MlirAttribute argumentAttributes,
    MlirAttribute metadata);

RYFT_XLA_SYS_EXPORT bool mlirAttributeIsAGpuKernelTableAttr(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirGpuKernelTableAttrGet(
    MlirContext context,
    intptr_t kernelCount,
    const MlirAttribute *kernels);

RYFT_XLA_SYS_EXPORT bool mlirAttributeIsAGpuSelectObjectAttr(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirGpuSelectObjectAttrGet(MlirContext context, MlirAttribute target);

RYFT_XLA_SYS_EXPORT bool mlirAttributeIsAGpuEnumAttr(MlirAttribute attribute, enum MlirGpuEnumAttribute kind);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirGpuEnumAttrGet(
    MlirContext context,
    enum MlirGpuEnumAttribute kind,
    MlirStringRef value);
RYFT_XLA_SYS_EXPORT MlirStringRef mlirGpuEnumAttrGetValue(MlirAttribute attribute, enum MlirGpuEnumAttribute kind);

RYFT_XLA_SYS_EXPORT bool mlirAttributeIsAGpuMappingAttr(MlirAttribute attribute, enum MlirGpuMappingAttribute kind);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirGpuMappingAttrGet(
    MlirContext context,
    enum MlirGpuMappingAttribute kind,
    MlirStringRef value);
RYFT_XLA_SYS_EXPORT MlirStringRef mlirGpuMappingAttrGetValue(
    MlirAttribute attribute,
    enum MlirGpuMappingAttribute kind);

RYFT_XLA_SYS_EXPORT bool mlirAttributeIsAGpuMappingMaskAttr(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirGpuMappingMaskAttrGet(MlirContext context, uint64_t mask);
RYFT_XLA_SYS_EXPORT uint64_t mlirGpuMappingMaskAttrGetMask(MlirAttribute attribute);

RYFT_XLA_SYS_EXPORT bool mlirAttributeIsAGpuMemorySpaceMappingAttr(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirGpuMemorySpaceMappingAttrGet(MlirContext context, MlirStringRef addressSpace);
RYFT_XLA_SYS_EXPORT MlirStringRef mlirGpuMemorySpaceMappingAttrGetAddressSpace(MlirAttribute attribute);

RYFT_XLA_SYS_EXPORT bool mlirAttributeIsAGpuParallelLoopDimMappingAttr(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirGpuParallelLoopDimMappingAttrGet(
    MlirContext context,
    MlirStringRef processor,
    MlirAffineMap map,
    MlirAffineMap bound);
RYFT_XLA_SYS_EXPORT MlirStringRef mlirGpuParallelLoopDimMappingAttrGetProcessor(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT MlirAffineMap mlirGpuParallelLoopDimMappingAttrGetMap(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT MlirAffineMap mlirGpuParallelLoopDimMappingAttrGetBound(MlirAttribute attribute);

#ifdef __cplusplus
}
#endif
