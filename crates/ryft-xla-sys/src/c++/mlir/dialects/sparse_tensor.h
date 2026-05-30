#pragma once

#include "../../common.h"

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

RYFT_XLA_SYS_EXPORT bool mlirAttributeIsASparseTensorDimSliceAttr(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirSparseTensorDimSliceAttrGet(
    MlirContext context,
    int64_t offset,
    int64_t size,
    int64_t stride);
RYFT_XLA_SYS_EXPORT int64_t mlirSparseTensorDimSliceAttrGetOffset(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT int64_t mlirSparseTensorDimSliceAttrGetSize(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT int64_t mlirSparseTensorDimSliceAttrGetStride(MlirAttribute attribute);

RYFT_XLA_SYS_EXPORT MlirAttribute mlirSparseTensorEncodingAttrGetWithDimSlices(
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
RYFT_XLA_SYS_EXPORT intptr_t mlirSparseTensorEncodingAttrGetDimSliceCount(MlirAttribute attribute);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirSparseTensorEncodingAttrGetDimSlice(MlirAttribute attribute, intptr_t dimension);

RYFT_XLA_SYS_EXPORT bool mlirAttributeIsASparseTensorEnumAttr(
    MlirAttribute attribute,
    enum MlirSparseTensorEnumAttribute kind);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirSparseTensorEnumAttrGet(
    MlirContext context,
    enum MlirSparseTensorEnumAttribute kind,
    uint32_t value);
RYFT_XLA_SYS_EXPORT uint32_t mlirSparseTensorEnumAttrGetValue(
    MlirAttribute attribute,
    enum MlirSparseTensorEnumAttribute kind);

RYFT_XLA_SYS_EXPORT bool mlirTypeIsASparseTensorStorageSpecifierType(MlirType type);
RYFT_XLA_SYS_EXPORT MlirType mlirSparseTensorStorageSpecifierTypeGet(MlirContext context, MlirAttribute encoding);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirSparseTensorStorageSpecifierTypeGetEncoding(MlirType type);

RYFT_XLA_SYS_EXPORT bool mlirTypeIsASparseTensorIterSpaceType(MlirType type);
RYFT_XLA_SYS_EXPORT MlirType mlirSparseTensorIterSpaceTypeGet(
    MlirContext context,
    MlirAttribute encoding,
    uint64_t lowerLevel,
    uint64_t upperLevel);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirSparseTensorIterSpaceTypeGetEncoding(MlirType type);
RYFT_XLA_SYS_EXPORT uint64_t mlirSparseTensorIterSpaceTypeGetLowerLevel(MlirType type);
RYFT_XLA_SYS_EXPORT uint64_t mlirSparseTensorIterSpaceTypeGetUpperLevel(MlirType type);

RYFT_XLA_SYS_EXPORT bool mlirTypeIsASparseTensorIteratorType(MlirType type);
RYFT_XLA_SYS_EXPORT MlirType mlirSparseTensorIteratorTypeGet(
    MlirContext context,
    MlirAttribute encoding,
    uint64_t lowerLevel,
    uint64_t upperLevel);
RYFT_XLA_SYS_EXPORT MlirAttribute mlirSparseTensorIteratorTypeGetEncoding(MlirType type);
RYFT_XLA_SYS_EXPORT uint64_t mlirSparseTensorIteratorTypeGetLowerLevel(MlirType type);
RYFT_XLA_SYS_EXPORT uint64_t mlirSparseTensorIteratorTypeGetUpperLevel(MlirType type);

#ifdef __cplusplus
}
#endif
