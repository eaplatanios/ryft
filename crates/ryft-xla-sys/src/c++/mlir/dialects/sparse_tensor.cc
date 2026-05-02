#include "sparse_tensor.h"

#include <cstddef>
#include <cstdint>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"

namespace {

template <typename AttributeT>
bool isAttribute(MlirAttribute attribute) {
  return attribute.ptr != nullptr && llvm::isa<AttributeT>(unwrap(attribute));
}

template <typename TypeT>
bool isType(MlirType type) {
  return type.ptr != nullptr && llvm::isa<TypeT>(unwrap(type));
}

template <typename AttributeT>
uint32_t getEnumAttributeValue(MlirAttribute attribute) {
  if (attribute.ptr == nullptr) {
    return 0;
  }
  auto typedAttribute = llvm::dyn_cast<AttributeT>(unwrap(attribute));
  if (!typedAttribute) {
    return 0;
  }
  return static_cast<uint32_t>(typedAttribute.getValue());
}

}  // namespace

bool mlirAttributeIsASparseTensorDimSliceAttr(MlirAttribute attribute) {
  return isAttribute<mlir::sparse_tensor::SparseTensorDimSliceAttr>(attribute);
}

MlirAttribute mlirSparseTensorDimSliceAttrGet(
    MlirContext context,
    int64_t offset,
    int64_t size,
    int64_t stride) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  return wrap(mlir::sparse_tensor::SparseTensorDimSliceAttr::get(unwrap(context), offset, size, stride));
}

int64_t mlirSparseTensorDimSliceAttrGetOffset(MlirAttribute attribute) {
  auto sliceAttribute = llvm::dyn_cast<mlir::sparse_tensor::SparseTensorDimSliceAttr>(unwrap(attribute));
  if (!sliceAttribute) {
    return mlir::sparse_tensor::SparseTensorDimSliceAttr::kDynamic;
  }
  return sliceAttribute.getOffset();
}

int64_t mlirSparseTensorDimSliceAttrGetSize(MlirAttribute attribute) {
  auto sliceAttribute = llvm::dyn_cast<mlir::sparse_tensor::SparseTensorDimSliceAttr>(unwrap(attribute));
  if (!sliceAttribute) {
    return mlir::sparse_tensor::SparseTensorDimSliceAttr::kDynamic;
  }
  return sliceAttribute.getSize();
}

int64_t mlirSparseTensorDimSliceAttrGetStride(MlirAttribute attribute) {
  auto sliceAttribute = llvm::dyn_cast<mlir::sparse_tensor::SparseTensorDimSliceAttr>(unwrap(attribute));
  if (!sliceAttribute) {
    return mlir::sparse_tensor::SparseTensorDimSliceAttr::kDynamic;
  }
  return sliceAttribute.getStride();
}

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
    const MlirAttribute *dimensionSlices) {
  if (context.ptr == nullptr || levelRank < 0 || (levelRank > 0 && levelTypes == nullptr) ||
      dimensionSliceCount < 0 || (dimensionSliceCount > 0 && dimensionSlices == nullptr)) {
    return {nullptr};
  }

  llvm::SmallVector<mlir::sparse_tensor::LevelType, 8> levelTypeValues;
  levelTypeValues.reserve(static_cast<size_t>(levelRank));
  for (intptr_t index = 0; index < levelRank; ++index) {
    levelTypeValues.push_back(static_cast<mlir::sparse_tensor::LevelType>(levelTypes[index]));
  }

  llvm::SmallVector<mlir::sparse_tensor::SparseTensorDimSliceAttr, 8> dimensionSliceValues;
  dimensionSliceValues.reserve(static_cast<size_t>(dimensionSliceCount));
  for (intptr_t index = 0; index < dimensionSliceCount; ++index) {
    auto slice = llvm::dyn_cast<mlir::sparse_tensor::SparseTensorDimSliceAttr>(unwrap(dimensionSlices[index]));
    if (!slice) {
      return {nullptr};
    }
    dimensionSliceValues.push_back(slice);
  }

  return wrap(mlir::sparse_tensor::SparseTensorEncodingAttr::get(
      unwrap(context),
      levelTypeValues,
      unwrap(dimensionToLevel),
      unwrap(levelToDimension),
      static_cast<unsigned>(positionWidth),
      static_cast<unsigned>(coordinateWidth),
      unwrap(explicitValue),
      unwrap(implicitValue),
      dimensionSliceValues));
}

intptr_t mlirSparseTensorEncodingAttrGetDimSliceCount(MlirAttribute attribute) {
  auto encodingAttribute = llvm::dyn_cast<mlir::sparse_tensor::SparseTensorEncodingAttr>(unwrap(attribute));
  if (!encodingAttribute) {
    return 0;
  }
  return static_cast<intptr_t>(encodingAttribute.getDimSlices().size());
}

MlirAttribute mlirSparseTensorEncodingAttrGetDimSlice(MlirAttribute attribute, intptr_t dimension) {
  auto encodingAttribute = llvm::dyn_cast<mlir::sparse_tensor::SparseTensorEncodingAttr>(unwrap(attribute));
  if (!encodingAttribute || dimension < 0 ||
      dimension >= static_cast<intptr_t>(encodingAttribute.getDimSlices().size())) {
    return {nullptr};
  }
  return wrap(encodingAttribute.getDimSlices()[static_cast<size_t>(dimension)]);
}

bool mlirAttributeIsASparseTensorEnumAttr(
    MlirAttribute attribute,
    enum MlirSparseTensorEnumAttribute kind) {
  switch (kind) {
    case MLIR_SPARSE_TENSOR_ENUM_ATTRIBUTE_STORAGE_SPECIFIER_KIND:
      return isAttribute<mlir::sparse_tensor::StorageSpecifierKindAttr>(attribute);
    case MLIR_SPARSE_TENSOR_ENUM_ATTRIBUTE_SORT_KIND:
      return isAttribute<mlir::sparse_tensor::SparseTensorSortKindAttr>(attribute);
    case MLIR_SPARSE_TENSOR_ENUM_ATTRIBUTE_CRD_TRANS_DIRECTION:
      return isAttribute<mlir::sparse_tensor::CrdTransDirectionKindAttr>(attribute);
  }
  return false;
}

MlirAttribute mlirSparseTensorEnumAttrGet(
    MlirContext context,
    enum MlirSparseTensorEnumAttribute kind,
    uint32_t value) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  switch (kind) {
    case MLIR_SPARSE_TENSOR_ENUM_ATTRIBUTE_STORAGE_SPECIFIER_KIND:
      if (auto enumValue = mlir::sparse_tensor::symbolizeStorageSpecifierKind(value)) {
        return wrap(mlir::sparse_tensor::StorageSpecifierKindAttr::get(unwrap(context), *enumValue));
      }
      return {nullptr};
    case MLIR_SPARSE_TENSOR_ENUM_ATTRIBUTE_SORT_KIND:
      if (auto enumValue = mlir::sparse_tensor::symbolizeSparseTensorSortKind(value)) {
        return wrap(mlir::sparse_tensor::SparseTensorSortKindAttr::get(unwrap(context), *enumValue));
      }
      return {nullptr};
    case MLIR_SPARSE_TENSOR_ENUM_ATTRIBUTE_CRD_TRANS_DIRECTION:
      if (auto enumValue = mlir::sparse_tensor::symbolizeCrdTransDirectionKind(value)) {
        return wrap(mlir::sparse_tensor::CrdTransDirectionKindAttr::get(unwrap(context), *enumValue));
      }
      return {nullptr};
  }
  return {nullptr};
}

uint32_t mlirSparseTensorEnumAttrGetValue(
    MlirAttribute attribute,
    enum MlirSparseTensorEnumAttribute kind) {
  switch (kind) {
    case MLIR_SPARSE_TENSOR_ENUM_ATTRIBUTE_STORAGE_SPECIFIER_KIND:
      return getEnumAttributeValue<mlir::sparse_tensor::StorageSpecifierKindAttr>(attribute);
    case MLIR_SPARSE_TENSOR_ENUM_ATTRIBUTE_SORT_KIND:
      return getEnumAttributeValue<mlir::sparse_tensor::SparseTensorSortKindAttr>(attribute);
    case MLIR_SPARSE_TENSOR_ENUM_ATTRIBUTE_CRD_TRANS_DIRECTION:
      return getEnumAttributeValue<mlir::sparse_tensor::CrdTransDirectionKindAttr>(attribute);
  }
  return 0;
}

bool mlirTypeIsASparseTensorStorageSpecifierType(MlirType type) {
  return isType<mlir::sparse_tensor::StorageSpecifierType>(type);
}

MlirType mlirSparseTensorStorageSpecifierTypeGet(MlirContext context, MlirAttribute encoding) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  auto encodingAttribute = llvm::dyn_cast<mlir::sparse_tensor::SparseTensorEncodingAttr>(unwrap(encoding));
  if (!encodingAttribute) {
    return {nullptr};
  }
  return wrap(mlir::sparse_tensor::StorageSpecifierType::get(unwrap(context), encodingAttribute));
}

MlirAttribute mlirSparseTensorStorageSpecifierTypeGetEncoding(MlirType type) {
  auto storageSpecifierType = llvm::dyn_cast<mlir::sparse_tensor::StorageSpecifierType>(unwrap(type));
  if (!storageSpecifierType) {
    return {nullptr};
  }
  return wrap(storageSpecifierType.getEncoding());
}

bool mlirTypeIsASparseTensorIterSpaceType(MlirType type) {
  return isType<mlir::sparse_tensor::IterSpaceType>(type);
}

MlirType mlirSparseTensorIterSpaceTypeGet(
    MlirContext context,
    MlirAttribute encoding,
    uint64_t lowerLevel,
    uint64_t upperLevel) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  auto encodingAttribute = llvm::dyn_cast<mlir::sparse_tensor::SparseTensorEncodingAttr>(unwrap(encoding));
  if (!encodingAttribute) {
    return {nullptr};
  }
  return wrap(mlir::sparse_tensor::IterSpaceType::get(unwrap(context), encodingAttribute, lowerLevel, upperLevel));
}

MlirAttribute mlirSparseTensorIterSpaceTypeGetEncoding(MlirType type) {
  auto iterSpaceType = llvm::dyn_cast<mlir::sparse_tensor::IterSpaceType>(unwrap(type));
  if (!iterSpaceType) {
    return {nullptr};
  }
  return wrap(iterSpaceType.getEncoding());
}

uint64_t mlirSparseTensorIterSpaceTypeGetLowerLevel(MlirType type) {
  auto iterSpaceType = llvm::dyn_cast<mlir::sparse_tensor::IterSpaceType>(unwrap(type));
  if (!iterSpaceType) {
    return 0;
  }
  return iterSpaceType.getLoLvl();
}

uint64_t mlirSparseTensorIterSpaceTypeGetUpperLevel(MlirType type) {
  auto iterSpaceType = llvm::dyn_cast<mlir::sparse_tensor::IterSpaceType>(unwrap(type));
  if (!iterSpaceType) {
    return 0;
  }
  return iterSpaceType.getHiLvl();
}

bool mlirTypeIsASparseTensorIteratorType(MlirType type) {
  return isType<mlir::sparse_tensor::IteratorType>(type);
}

MlirType mlirSparseTensorIteratorTypeGet(
    MlirContext context,
    MlirAttribute encoding,
    uint64_t lowerLevel,
    uint64_t upperLevel) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  auto encodingAttribute = llvm::dyn_cast<mlir::sparse_tensor::SparseTensorEncodingAttr>(unwrap(encoding));
  if (!encodingAttribute) {
    return {nullptr};
  }
  return wrap(mlir::sparse_tensor::IteratorType::get(unwrap(context), encodingAttribute, lowerLevel, upperLevel));
}

MlirAttribute mlirSparseTensorIteratorTypeGetEncoding(MlirType type) {
  auto iteratorType = llvm::dyn_cast<mlir::sparse_tensor::IteratorType>(unwrap(type));
  if (!iteratorType) {
    return {nullptr};
  }
  return wrap(iteratorType.getEncoding());
}

uint64_t mlirSparseTensorIteratorTypeGetLowerLevel(MlirType type) {
  auto iteratorType = llvm::dyn_cast<mlir::sparse_tensor::IteratorType>(unwrap(type));
  if (!iteratorType) {
    return 0;
  }
  return iteratorType.getLoLvl();
}

uint64_t mlirSparseTensorIteratorTypeGetUpperLevel(MlirType type) {
  auto iteratorType = llvm::dyn_cast<mlir::sparse_tensor::IteratorType>(unwrap(type));
  if (!iteratorType) {
    return 0;
  }
  return iteratorType.getHiLvl();
}
