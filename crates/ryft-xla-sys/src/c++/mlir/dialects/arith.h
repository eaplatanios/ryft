#pragma once

#include "../../common.h"

#include <stdbool.h>
#include <stdint.h>

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

RYFT_XLA_SYS_EXPORT bool mlirAttributeIsAArithAtomicRmwKindAttr(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirArithAtomicRmwKindAttrGet(MlirContext context, uint64_t value);
RYFT_XLA_SYS_EXPORT uint64_t mlirArithAtomicRmwKindAttrGetValue(MlirAttribute attribute);

RYFT_XLA_SYS_EXPORT bool mlirAttributeIsAArithFastMathFlagsAttr(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirArithFastMathFlagsAttrGet(MlirContext context, uint32_t value);
RYFT_XLA_SYS_EXPORT uint32_t mlirArithFastMathFlagsAttrGetValue(MlirAttribute attribute);

RYFT_XLA_SYS_EXPORT bool mlirAttributeIsAArithIntegerOverflowFlagsAttr(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirArithIntegerOverflowFlagsAttrGet(MlirContext context, uint32_t value);
RYFT_XLA_SYS_EXPORT uint32_t mlirArithIntegerOverflowFlagsAttrGetValue(MlirAttribute attribute);

RYFT_XLA_SYS_EXPORT bool mlirAttributeIsAArithRoundingModeAttr(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirArithRoundingModeAttrGet(MlirContext context, uint32_t value);
RYFT_XLA_SYS_EXPORT uint32_t mlirArithRoundingModeAttrGetValue(MlirAttribute attribute);

#ifdef __cplusplus
}
#endif
