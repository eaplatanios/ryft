#pragma once

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Bufferization, bufferization);

#ifdef __cplusplus
}
#endif
