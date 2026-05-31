#pragma once

#include "../../common.h"

#include <stdbool.h>
#include <stdint.h>

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  MLIR_TRANSFORM_ENUM_ATTRIBUTE_FAILURE_PROPAGATION_MODE = 0,
  MLIR_TRANSFORM_ENUM_ATTRIBUTE_MATCH_CMP_I_PREDICATE = 1,
} MlirTransformEnumAttribute;

RYFT_XLA_SYS_EXPORT bool mlirTypeIsATransformAffineMapParamType(MlirType type);
RYFT_XLA_SYS_EXPORT MlirType mlirTransformAffineMapParamTypeGet(MlirContext context);

RYFT_XLA_SYS_EXPORT bool mlirTypeIsATransformTypeParamType(MlirType type);
RYFT_XLA_SYS_EXPORT MlirType mlirTransformTypeParamTypeGet(MlirContext context);

RYFT_XLA_SYS_EXPORT bool mlirAttributeIsATransformEnumAttr(MlirAttribute attribute, MlirTransformEnumAttribute kind);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirTransformEnumAttrGet(
    MlirContext context,
    MlirTransformEnumAttribute kind,
    uint32_t value);
RYFT_XLA_SYS_EXPORT uint32_t mlirTransformEnumAttrGetValue(MlirAttribute attribute, MlirTransformEnumAttribute kind);

RYFT_XLA_SYS_EXPORT bool mlirAttributeIsATransformParamOperandAttr(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirTransformParamOperandAttrGet(MlirContext context, MlirAttribute index);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirTransformParamOperandAttrGetIndex(MlirAttribute attribute);

#ifdef __cplusplus
}
#endif
