#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MlirDialectHandle mlirGetDialectHandle__tt__();

enum MlirTritonTtEnumAttribute {
  MLIR_TRITON_TT_ENUM_ATTRIBUTE_CACHE_MODIFIER = 0,
  MLIR_TRITON_TT_ENUM_ATTRIBUTE_MEM_SEMANTIC = 1,
  MLIR_TRITON_TT_ENUM_ATTRIBUTE_EVICTION_POLICY = 2,
  MLIR_TRITON_TT_ENUM_ATTRIBUTE_PADDING_OPTION = 3,
  MLIR_TRITON_TT_ENUM_ATTRIBUTE_RMW_OP = 4,
  MLIR_TRITON_TT_ENUM_ATTRIBUTE_DESCRIPTOR_REDUCE_KIND = 5,
  MLIR_TRITON_TT_ENUM_ATTRIBUTE_MEM_SYNC_SCOPE = 6,
  MLIR_TRITON_TT_ENUM_ATTRIBUTE_PROGRAM_ID_DIM = 7,
  MLIR_TRITON_TT_ENUM_ATTRIBUTE_ROUNDING_MODE = 8,
  MLIR_TRITON_TT_ENUM_ATTRIBUTE_PROPAGATE_NAN = 9,
  MLIR_TRITON_TT_ENUM_ATTRIBUTE_INPUT_PRECISION = 10,
  MLIR_TRITON_TT_ENUM_ATTRIBUTE_SCALE_DOT_ELEM_TYPE = 11,
};

bool mlirTypeIsATritonTtPointerType(MlirType type);
MlirType mlirTritonTtPointerTypeGet(MlirType pointeeType, int32_t addressSpace);
MlirType mlirTritonTtPointerTypeGetPointeeType(MlirType type);
int32_t mlirTritonTtPointerTypeGetAddressSpace(MlirType type);

bool mlirTypeIsATritonTtTensorDescType(MlirType type);
MlirType mlirTritonTtTensorDescTypeGet(
    const int64_t *shape,
    intptr_t shapeSize,
    MlirType elementType,
    MlirAttribute sharedLayout);
intptr_t mlirTritonTtTensorDescTypeGetNumDims(MlirType type);
int64_t mlirTritonTtTensorDescTypeGetDimSize(MlirType type, intptr_t dimension);
MlirType mlirTritonTtTensorDescTypeGetElementType(MlirType type);
MlirAttribute mlirTritonTtTensorDescTypeGetSharedLayout(MlirType type);
MlirType mlirTritonTtTensorDescTypeGetBlockType(MlirType type);

bool mlirAttributeIsATritonTtEnumAttr(MlirAttribute attribute, enum MlirTritonTtEnumAttribute kind);
MlirAttribute mlirTritonTtEnumAttrGet(
    MlirContext context,
    enum MlirTritonTtEnumAttribute kind,
    MlirStringRef value);
MlirStringRef mlirTritonTtEnumAttrGetValue(MlirAttribute attribute, enum MlirTritonTtEnumAttribute kind);

#ifdef __cplusplus
}
#endif
