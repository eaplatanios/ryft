#pragma once

#include "../../common.h"

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

RYFT_XLA_SYS_EXPORT bool mlirTypeIsANvgpuDeviceAsyncTokenType(MlirType type);
RYFT_XLA_SYS_EXPORT MlirType mlirNvgpuDeviceAsyncTokenTypeGet(MlirContext context);

RYFT_XLA_SYS_EXPORT bool mlirTypeIsANvgpuMBarrierGroupType(MlirType type);
RYFT_XLA_SYS_EXPORT MlirType mlirNvgpuMBarrierGroupTypeGet(
    MlirContext context,
    MlirAttribute memory_space,
    uint32_t num_barriers);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirNvgpuMBarrierGroupTypeGetMemorySpace(MlirType type);
RYFT_XLA_SYS_EXPORT uint32_t mlirNvgpuMBarrierGroupTypeGetNumBarriers(MlirType type);

RYFT_XLA_SYS_EXPORT bool mlirTypeIsANvgpuMBarrierTokenType(MlirType type);
RYFT_XLA_SYS_EXPORT MlirType mlirNvgpuMBarrierTokenTypeGet(MlirContext context);

RYFT_XLA_SYS_EXPORT MlirType mlirNvgpuTensorMapDescriptorTypeGetTensor(MlirType type);
RYFT_XLA_SYS_EXPORT uint32_t mlirNvgpuTensorMapDescriptorTypeGetSwizzle(MlirType type);
RYFT_XLA_SYS_EXPORT uint32_t mlirNvgpuTensorMapDescriptorTypeGetL2Promo(MlirType type);
RYFT_XLA_SYS_EXPORT uint32_t mlirNvgpuTensorMapDescriptorTypeGetOob(MlirType type);
RYFT_XLA_SYS_EXPORT uint32_t mlirNvgpuTensorMapDescriptorTypeGetInterleave(MlirType type);

RYFT_XLA_SYS_EXPORT bool mlirTypeIsANvgpuWarpgroupMatrixDescriptorType(MlirType type);
RYFT_XLA_SYS_EXPORT MlirType mlirNvgpuWarpgroupMatrixDescriptorTypeGet(MlirContext context, MlirType tensor);
RYFT_XLA_SYS_EXPORT MlirType mlirNvgpuWarpgroupMatrixDescriptorTypeGetTensor(MlirType type);

RYFT_XLA_SYS_EXPORT bool mlirTypeIsANvgpuWarpgroupAccumulatorType(MlirType type);
RYFT_XLA_SYS_EXPORT MlirType mlirNvgpuWarpgroupAccumulatorTypeGet(MlirContext context, MlirType fragmented);
RYFT_XLA_SYS_EXPORT MlirType mlirNvgpuWarpgroupAccumulatorTypeGetFragmented(MlirType type);

RYFT_XLA_SYS_EXPORT bool mlirAttributeIsANvgpuEnumAttr(MlirAttribute attribute, enum MlirNvgpuEnumAttribute kind);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirNvgpuEnumAttrGet(
    MlirContext context,
    enum MlirNvgpuEnumAttribute kind,
    uint32_t value);
RYFT_XLA_SYS_EXPORT uint32_t mlirNvgpuEnumAttrGetValue(MlirAttribute attribute, enum MlirNvgpuEnumAttribute kind);

#ifdef __cplusplus
}
#endif
