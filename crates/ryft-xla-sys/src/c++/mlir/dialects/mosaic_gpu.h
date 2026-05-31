#pragma once

#include "../../common.h"

#include <stdbool.h>
#include <stdint.h>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

enum MlirMosaicGpuEnumAttribute {
  RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_DIMENSION = 0,
  RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_SWIZZLING_MODE = 1,
  RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_TMA_REDUCTION = 2,
  RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_OOB_FILL_MODE = 3,
  RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_MULTIMEM_LOAD_REDUCTION_TYPE = 4,
  RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_ATOMIC_OP_TYPE = 5,
};

RYFT_XLA_SYS_EXPORT bool mlirAttributeIsAMosaicGpuEnumAttr(
    MlirAttribute attribute,
    enum MlirMosaicGpuEnumAttribute kind);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirMosaicGpuEnumAttrGet(
    MlirContext context,
    enum MlirMosaicGpuEnumAttribute kind,
    MlirStringRef value);
RYFT_XLA_SYS_EXPORT MlirStringRef mlirMosaicGpuEnumAttrGetValue(
    MlirAttribute attribute,
    enum MlirMosaicGpuEnumAttribute kind);

RYFT_XLA_SYS_EXPORT bool mlirAttributeIsAMosaicGpuTmemAttr(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirMosaicGpuTmemAttrGet(MlirContext context);

#ifdef __cplusplus
}
#endif
