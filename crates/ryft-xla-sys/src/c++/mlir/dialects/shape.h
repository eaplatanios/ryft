#pragma once

#include <stdbool.h>

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

bool mlirTypeIsAShapeShapeType(MlirType type);
MlirType mlirShapeShapeTypeGet(MlirContext context);

bool mlirTypeIsAShapeSizeType(MlirType type);
MlirType mlirShapeSizeTypeGet(MlirContext context);

bool mlirTypeIsAShapeValueShapeType(MlirType type);
MlirType mlirShapeValueShapeTypeGet(MlirContext context);

bool mlirTypeIsAShapeWitnessType(MlirType type);
MlirType mlirShapeWitnessTypeGet(MlirContext context);

#ifdef __cplusplus
}
#endif
