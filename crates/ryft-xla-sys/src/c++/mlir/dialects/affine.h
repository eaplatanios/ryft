#pragma once

#include "../../common.h"

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

RYFT_XLA_SYS_EXPORT MlirDialectHandle mlirGetDialectHandle__affine__();

#ifdef __cplusplus
}
#endif
