#pragma once

#include "../../common.h"

#include <stdbool.h>

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

RYFT_XLA_SYS_EXPORT bool mlirTypeIsAToken(MlirType type);
RYFT_XLA_SYS_EXPORT MlirType mlirTokenTypeGet(MlirContext context);

#ifdef __cplusplus
}
#endif
