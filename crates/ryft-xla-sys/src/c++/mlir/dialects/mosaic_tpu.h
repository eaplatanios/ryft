#pragma once

#include "../../common.h"

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

RYFT_XLA_SYS_EXPORT bool mlirTpuIsASemaphoreType(MlirType type);
RYFT_XLA_SYS_EXPORT MlirType mlirTpuSemaphoreTypeGet(MlirContext context);

RYFT_XLA_SYS_EXPORT bool mlirTpuIsADmaSemaphoreType(MlirType type);
RYFT_XLA_SYS_EXPORT MlirType mlirTpuDmaSemaphoreTypeGet(MlirContext context);

RYFT_XLA_SYS_EXPORT bool mlirAttributeIsAMosaicTpuEnumAttr(
    MlirAttribute attribute,
    enum MlirMosaicTpuEnumAttribute kind);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirMosaicTpuEnumAttrGet(
    MlirContext context,
    enum MlirMosaicTpuEnumAttribute kind,
    MlirStringRef value);
RYFT_XLA_SYS_EXPORT MlirStringRef mlirMosaicTpuEnumAttrGetValue(
    MlirAttribute attribute,
    enum MlirMosaicTpuEnumAttribute kind);

RYFT_XLA_SYS_EXPORT bool mlirAttributeIsAMosaicTpuDotDimensionNumbersAttr(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirMosaicTpuDotDimensionNumbersAttrGet(
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
RYFT_XLA_SYS_EXPORT MlirAttribute mlirMosaicTpuDotDimensionNumbersAttrGetLhsContractingDims(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirMosaicTpuDotDimensionNumbersAttrGetRhsContractingDims(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirMosaicTpuDotDimensionNumbersAttrGetLhsNonContractingDims(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirMosaicTpuDotDimensionNumbersAttrGetRhsNonContractingDims(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirMosaicTpuDotDimensionNumbersAttrGetOutputDimOrder(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirMosaicTpuDotDimensionNumbersAttrGetLhsBatchDims(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirMosaicTpuDotDimensionNumbersAttrGetRhsBatchDims(MlirAttribute attribute);

RYFT_XLA_SYS_EXPORT bool mlirAttributeIsAMosaicTpuElementWindowAttr(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirMosaicTpuElementWindowAttrGet(
    MlirContext context,
    const int64_t *pad_low,
    intptr_t pad_low_size,
    const int64_t *pad_high,
    intptr_t pad_high_size);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirMosaicTpuElementWindowAttrGetPadLow(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirMosaicTpuElementWindowAttrGetPadHigh(MlirAttribute attribute);

RYFT_XLA_SYS_EXPORT bool mlirAttributeIsAMosaicTpuVectorLayoutAttr(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT bool mlirAttributeIsAMosaicTpuTiledLayoutAttr(MlirAttribute attribute);

RYFT_XLA_SYS_EXPORT bool mlirAttributeIsAMosaicTpuMemorySpaceAttr(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirMosaicTpuMemorySpaceAttrGet(
    MlirContext context,
    MlirStringRef value,
    MlirStringRef core_type);
RYFT_XLA_SYS_EXPORT MlirStringRef mlirMosaicTpuMemorySpaceAttrGetValue(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT bool mlirMosaicTpuMemorySpaceAttrHasCoreType(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT MlirStringRef mlirMosaicTpuMemorySpaceAttrGetCoreType(MlirAttribute attribute);

#ifdef __cplusplus
}
#endif
