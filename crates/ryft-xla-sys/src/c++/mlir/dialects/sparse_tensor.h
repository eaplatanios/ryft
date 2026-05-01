#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "mlir-c/AffineMap.h"
#include "mlir-c/IR.h"
#include "mlir-c/Dialect/SparseTensor.h"

#ifdef __cplusplus
extern "C" {
#endif

enum MlirSparseTensorEnumAttribute {
  MLIR_SPARSE_TENSOR_ENUM_ATTRIBUTE_STORAGE_SPECIFIER_KIND = 0,
  MLIR_SPARSE_TENSOR_ENUM_ATTRIBUTE_SORT_KIND = 1,
  MLIR_SPARSE_TENSOR_ENUM_ATTRIBUTE_CRD_TRANS_DIRECTION = 2,
};

bool mlirAttributeIsASparseTensorDimSliceAttr(MlirAttribute attribute);
MlirAttribute mlirSparseTensorDimSliceAttrGet(
    MlirContext context,
    int64_t offset,
    int64_t size,
    int64_t stride);
int64_t mlirSparseTensorDimSliceAttrGetOffset(MlirAttribute attribute);
int64_t mlirSparseTensorDimSliceAttrGetSize(MlirAttribute attribute);
int64_t mlirSparseTensorDimSliceAttrGetStride(MlirAttribute attribute);

MlirAttribute mlirSparseTensorEncodingAttrGetWithDimSlices(
    MlirContext context,
    intptr_t levelRank,
    const MlirSparseTensorLevelType *levelTypes,
    MlirAffineMap dimensionToLevel,
    MlirAffineMap levelToDimension,
    int positionWidth,
    int coordinateWidth,
    MlirAttribute explicitValue,
    MlirAttribute implicitValue,
    intptr_t dimensionSliceCount,
    const MlirAttribute *dimensionSlices);
intptr_t mlirSparseTensorEncodingAttrGetDimSliceCount(MlirAttribute attribute);
MlirAttribute mlirSparseTensorEncodingAttrGetDimSlice(MlirAttribute attribute, intptr_t dimension);

bool mlirAttributeIsASparseTensorEnumAttr(
    MlirAttribute attribute,
    enum MlirSparseTensorEnumAttribute kind);
MlirAttribute mlirSparseTensorEnumAttrGet(
    MlirContext context,
    enum MlirSparseTensorEnumAttribute kind,
    uint32_t value);
uint32_t mlirSparseTensorEnumAttrGetValue(
    MlirAttribute attribute,
    enum MlirSparseTensorEnumAttribute kind);

bool mlirTypeIsASparseTensorStorageSpecifierType(MlirType type);
MlirType mlirSparseTensorStorageSpecifierTypeGet(MlirContext context, MlirAttribute encoding);
MlirAttribute mlirSparseTensorStorageSpecifierTypeGetEncoding(MlirType type);

bool mlirTypeIsASparseTensorIterSpaceType(MlirType type);
MlirType mlirSparseTensorIterSpaceTypeGet(
    MlirContext context,
    MlirAttribute encoding,
    uint64_t lowerLevel,
    uint64_t upperLevel);
MlirAttribute mlirSparseTensorIterSpaceTypeGetEncoding(MlirType type);
uint64_t mlirSparseTensorIterSpaceTypeGetLowerLevel(MlirType type);
uint64_t mlirSparseTensorIterSpaceTypeGetUpperLevel(MlirType type);

bool mlirTypeIsASparseTensorIteratorType(MlirType type);
MlirType mlirSparseTensorIteratorTypeGet(
    MlirContext context,
    MlirAttribute encoding,
    uint64_t lowerLevel,
    uint64_t upperLevel);
MlirAttribute mlirSparseTensorIteratorTypeGetEncoding(MlirType type);
uint64_t mlirSparseTensorIteratorTypeGetLowerLevel(MlirType type);
uint64_t mlirSparseTensorIteratorTypeGetUpperLevel(MlirType type);

#ifdef __cplusplus
}
#endif
