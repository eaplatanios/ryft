#pragma once

#include "../../common.h"

#include <stdbool.h>

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

RYFT_XLA_SYS_EXPORT bool mlirTypeIsAShapeShapeType(MlirType type);
RYFT_XLA_SYS_EXPORT MlirType mlirShapeShapeTypeGet(MlirContext context);

RYFT_XLA_SYS_EXPORT bool mlirTypeIsAShapeSizeType(MlirType type);
RYFT_XLA_SYS_EXPORT MlirType mlirShapeSizeTypeGet(MlirContext context);

RYFT_XLA_SYS_EXPORT bool mlirTypeIsAShapeValueShapeType(MlirType type);
RYFT_XLA_SYS_EXPORT MlirType mlirShapeValueShapeTypeGet(MlirContext context);

RYFT_XLA_SYS_EXPORT bool mlirTypeIsAShapeWitnessType(MlirType type);
RYFT_XLA_SYS_EXPORT MlirType mlirShapeWitnessTypeGet(MlirContext context);

#ifdef __cplusplus
}
#endif
