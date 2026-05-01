#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

enum MlirNvgpuEnumAttribute {
  RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_TENSOR_MAP_SWIZZLE_KIND = 0,
  RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_TENSOR_MAP_L2_PROMO_KIND = 1,
  RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_TENSOR_MAP_OOB_KIND = 2,
  RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_TENSOR_MAP_INTERLEAVE_KIND = 3,
  RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_RCP_ROUNDING_MODE = 4,
};

bool mlirTypeIsANvgpuDeviceAsyncTokenType(MlirType type);
MlirType mlirNvgpuDeviceAsyncTokenTypeGet(MlirContext context);

bool mlirTypeIsANvgpuMBarrierGroupType(MlirType type);
MlirType mlirNvgpuMBarrierGroupTypeGet(MlirContext context, MlirAttribute memory_space, uint32_t num_barriers);
MlirAttribute mlirNvgpuMBarrierGroupTypeGetMemorySpace(MlirType type);
uint32_t mlirNvgpuMBarrierGroupTypeGetNumBarriers(MlirType type);

bool mlirTypeIsANvgpuMBarrierTokenType(MlirType type);
MlirType mlirNvgpuMBarrierTokenTypeGet(MlirContext context);

MlirType mlirNvgpuTensorMapDescriptorTypeGetTensor(MlirType type);
uint32_t mlirNvgpuTensorMapDescriptorTypeGetSwizzle(MlirType type);
uint32_t mlirNvgpuTensorMapDescriptorTypeGetL2Promo(MlirType type);
uint32_t mlirNvgpuTensorMapDescriptorTypeGetOob(MlirType type);
uint32_t mlirNvgpuTensorMapDescriptorTypeGetInterleave(MlirType type);

bool mlirTypeIsANvgpuWarpgroupMatrixDescriptorType(MlirType type);
MlirType mlirNvgpuWarpgroupMatrixDescriptorTypeGet(MlirContext context, MlirType tensor);
MlirType mlirNvgpuWarpgroupMatrixDescriptorTypeGetTensor(MlirType type);

bool mlirTypeIsANvgpuWarpgroupAccumulatorType(MlirType type);
MlirType mlirNvgpuWarpgroupAccumulatorTypeGet(MlirContext context, MlirType fragmented);
MlirType mlirNvgpuWarpgroupAccumulatorTypeGetFragmented(MlirType type);

bool mlirAttributeIsANvgpuEnumAttr(MlirAttribute attribute, enum MlirNvgpuEnumAttribute kind);
MlirAttribute mlirNvgpuEnumAttrGet(MlirContext context, enum MlirNvgpuEnumAttribute kind, uint32_t value);
uint32_t mlirNvgpuEnumAttrGetValue(MlirAttribute attribute, enum MlirNvgpuEnumAttribute kind);

#ifdef __cplusplus
}
#endif
