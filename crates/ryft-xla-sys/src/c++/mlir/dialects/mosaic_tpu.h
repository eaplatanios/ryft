#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

enum MlirMosaicTpuEnumAttribute {
  RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_CORE_TYPE = 0,
  RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_PIPELINE_MODE = 1,
  RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_REVISIT_MODE = 2,
  RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_DIMENSION_SEMANTICS = 3,
  RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_CONTRACT_PRECISION = 4,
  RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_PACK_FORMAT = 5,
  RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_REDUCTION_KIND = 6,
  RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_ROUNDING_MODE = 7,
};

bool mlirTpuIsASemaphoreType(MlirType type);
MlirType mlirTpuSemaphoreTypeGet(MlirContext context);

bool mlirTpuIsADmaSemaphoreType(MlirType type);
MlirType mlirTpuDmaSemaphoreTypeGet(MlirContext context);

bool mlirAttributeIsAMosaicTpuEnumAttr(MlirAttribute attribute, enum MlirMosaicTpuEnumAttribute kind);
MlirAttribute mlirMosaicTpuEnumAttrGet(
    MlirContext context,
    enum MlirMosaicTpuEnumAttribute kind,
    MlirStringRef value);
MlirStringRef mlirMosaicTpuEnumAttrGetValue(MlirAttribute attribute, enum MlirMosaicTpuEnumAttribute kind);

bool mlirAttributeIsAMosaicTpuDotDimensionNumbersAttr(MlirAttribute attribute);
MlirAttribute mlirMosaicTpuDotDimensionNumbersAttrGet(
    MlirContext context,
    const int64_t *lhs_contracting_dims,
    intptr_t lhs_contracting_dims_size,
    const int64_t *rhs_contracting_dims,
    intptr_t rhs_contracting_dims_size,
    const int64_t *lhs_non_contracting_dims,
    intptr_t lhs_non_contracting_dims_size,
    const int64_t *rhs_non_contracting_dims,
    intptr_t rhs_non_contracting_dims_size,
    const int64_t *output_dim_order,
    intptr_t output_dim_order_size,
    const int64_t *lhs_batch_dims,
    intptr_t lhs_batch_dims_size,
    const int64_t *rhs_batch_dims,
    intptr_t rhs_batch_dims_size);
MlirAttribute mlirMosaicTpuDotDimensionNumbersAttrGetLhsContractingDims(MlirAttribute attribute);
MlirAttribute mlirMosaicTpuDotDimensionNumbersAttrGetRhsContractingDims(MlirAttribute attribute);
MlirAttribute mlirMosaicTpuDotDimensionNumbersAttrGetLhsNonContractingDims(MlirAttribute attribute);
MlirAttribute mlirMosaicTpuDotDimensionNumbersAttrGetRhsNonContractingDims(MlirAttribute attribute);
MlirAttribute mlirMosaicTpuDotDimensionNumbersAttrGetOutputDimOrder(MlirAttribute attribute);
MlirAttribute mlirMosaicTpuDotDimensionNumbersAttrGetLhsBatchDims(MlirAttribute attribute);
MlirAttribute mlirMosaicTpuDotDimensionNumbersAttrGetRhsBatchDims(MlirAttribute attribute);

bool mlirAttributeIsAMosaicTpuElementWindowAttr(MlirAttribute attribute);
MlirAttribute mlirMosaicTpuElementWindowAttrGet(
    MlirContext context,
    const int64_t *pad_low,
    intptr_t pad_low_size,
    const int64_t *pad_high,
    intptr_t pad_high_size);
MlirAttribute mlirMosaicTpuElementWindowAttrGetPadLow(MlirAttribute attribute);
MlirAttribute mlirMosaicTpuElementWindowAttrGetPadHigh(MlirAttribute attribute);

bool mlirAttributeIsAMosaicTpuVectorLayoutAttr(MlirAttribute attribute);
bool mlirAttributeIsAMosaicTpuTiledLayoutAttr(MlirAttribute attribute);

bool mlirAttributeIsAMosaicTpuMemorySpaceAttr(MlirAttribute attribute);
MlirAttribute mlirMosaicTpuMemorySpaceAttrGet(
    MlirContext context,
    MlirStringRef value,
    MlirStringRef core_type);
MlirStringRef mlirMosaicTpuMemorySpaceAttrGetValue(MlirAttribute attribute);
bool mlirMosaicTpuMemorySpaceAttrHasCoreType(MlirAttribute attribute);
MlirStringRef mlirMosaicTpuMemorySpaceAttrGetCoreType(MlirAttribute attribute);

#ifdef __cplusplus
}
#endif
