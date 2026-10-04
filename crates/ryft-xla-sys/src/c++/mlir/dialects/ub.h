#pragma once

#include "../../common.h"

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(UB, ub);

RYFT_XLA_SYS_EXPORT bool mlirAttributeIsAUbPoisonAttr(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirUbPoisonAttrGet(MlirContext context);
RYFT_XLA_SYS_EXPORT MlirTypeID mlirUbPoisonAttrGetTypeID(void);

#ifdef __cplusplus
}
#endif
