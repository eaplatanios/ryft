#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

bool mlirAttributeIsAArithAtomicRmwKindAttr(MlirAttribute attribute);
MlirAttribute mlirArithAtomicRmwKindAttrGet(MlirContext context, uint64_t value);
uint64_t mlirArithAtomicRmwKindAttrGetValue(MlirAttribute attribute);

bool mlirAttributeIsAArithFastMathFlagsAttr(MlirAttribute attribute);
MlirAttribute mlirArithFastMathFlagsAttrGet(MlirContext context, uint32_t value);
uint32_t mlirArithFastMathFlagsAttrGetValue(MlirAttribute attribute);

bool mlirAttributeIsAArithIntegerOverflowFlagsAttr(MlirAttribute attribute);
MlirAttribute mlirArithIntegerOverflowFlagsAttrGet(MlirContext context, uint32_t value);
uint32_t mlirArithIntegerOverflowFlagsAttrGetValue(MlirAttribute attribute);

bool mlirAttributeIsAArithRoundingModeAttr(MlirAttribute attribute);
MlirAttribute mlirArithRoundingModeAttrGet(MlirContext context, uint32_t value);
uint32_t mlirArithRoundingModeAttrGetValue(MlirAttribute attribute);

#ifdef __cplusplus
}
#endif
