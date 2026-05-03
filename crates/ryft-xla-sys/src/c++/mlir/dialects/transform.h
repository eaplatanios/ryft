#pragma once

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

bool mlirTypeIsATransformAffineMapParamType(MlirType type);
MlirType mlirTransformAffineMapParamTypeGet(MlirContext context);

bool mlirTypeIsATransformTypeParamType(MlirType type);
MlirType mlirTransformTypeParamTypeGet(MlirContext context);

bool mlirAttributeIsATransformEnumAttr(MlirAttribute attribute, MlirTransformEnumAttribute kind);
MlirAttribute mlirTransformEnumAttrGet(MlirContext context, MlirTransformEnumAttribute kind, uint32_t value);
uint32_t mlirTransformEnumAttrGetValue(MlirAttribute attribute, MlirTransformEnumAttribute kind);

bool mlirAttributeIsATransformParamOperandAttr(MlirAttribute attribute);
MlirAttribute mlirTransformParamOperandAttrGet(MlirContext context, MlirAttribute index);
MlirAttribute mlirTransformParamOperandAttrGetIndex(MlirAttribute attribute);

#ifdef __cplusplus
}
#endif
