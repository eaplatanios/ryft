#include "transform.h"

#include <optional>

#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Transform/IR/TransformAttrs.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace {

template <typename TypeT>
bool isType(MlirType type) {
  return type.ptr != nullptr && llvm::isa<TypeT>(unwrap(type));
}

template <typename AttributeT>
bool isAttribute(MlirAttribute attribute) {
  return attribute.ptr != nullptr && llvm::isa<AttributeT>(unwrap(attribute));
}

}  // namespace

bool mlirTypeIsATransformAffineMapParamType(MlirType type) {
  return isType<mlir::transform::AffineMapParamType>(type);
}

MlirType mlirTransformAffineMapParamTypeGet(MlirContext context) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  return wrap(mlir::transform::AffineMapParamType::get(unwrap(context)));
}

bool mlirTypeIsATransformTypeParamType(MlirType type) {
  return isType<mlir::transform::TypeParamType>(type);
}

MlirType mlirTransformTypeParamTypeGet(MlirContext context) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  return wrap(mlir::transform::TypeParamType::get(unwrap(context)));
}

bool mlirAttributeIsATransformEnumAttr(MlirAttribute attribute, MlirTransformEnumAttribute kind) {
  switch (kind) {
    case MLIR_TRANSFORM_ENUM_ATTRIBUTE_FAILURE_PROPAGATION_MODE:
      return isAttribute<mlir::transform::FailurePropagationModeAttr>(attribute);
    case MLIR_TRANSFORM_ENUM_ATTRIBUTE_MATCH_CMP_I_PREDICATE:
      return isAttribute<mlir::transform::MatchCmpIPredicateAttr>(attribute);
  }
  return false;
}

MlirAttribute mlirTransformEnumAttrGet(MlirContext context, MlirTransformEnumAttribute kind, uint32_t value) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  switch (kind) {
    case MLIR_TRANSFORM_ENUM_ATTRIBUTE_FAILURE_PROPAGATION_MODE: {
      std::optional<mlir::transform::FailurePropagationMode> mode =
          mlir::transform::symbolizeFailurePropagationMode(value);
      if (!mode) {
        return {nullptr};
      }
      return wrap(mlir::transform::FailurePropagationModeAttr::get(unwrap(context), *mode));
    }
    case MLIR_TRANSFORM_ENUM_ATTRIBUTE_MATCH_CMP_I_PREDICATE: {
      std::optional<mlir::transform::MatchCmpIPredicate> predicate =
          mlir::transform::symbolizeMatchCmpIPredicate(value);
      if (!predicate) {
        return {nullptr};
      }
      return wrap(mlir::transform::MatchCmpIPredicateAttr::get(unwrap(context), *predicate));
    }
  }
  return {nullptr};
}

uint32_t mlirTransformEnumAttrGetValue(MlirAttribute attribute, MlirTransformEnumAttribute kind) {
  if (attribute.ptr == nullptr) {
    return 0;
  }
  switch (kind) {
    case MLIR_TRANSFORM_ENUM_ATTRIBUTE_FAILURE_PROPAGATION_MODE: {
      auto typedAttribute = llvm::dyn_cast<mlir::transform::FailurePropagationModeAttr>(unwrap(attribute));
      return typedAttribute ? static_cast<uint32_t>(typedAttribute.getValue()) : 0;
    }
    case MLIR_TRANSFORM_ENUM_ATTRIBUTE_MATCH_CMP_I_PREDICATE: {
      auto typedAttribute = llvm::dyn_cast<mlir::transform::MatchCmpIPredicateAttr>(unwrap(attribute));
      return typedAttribute ? static_cast<uint32_t>(typedAttribute.getValue()) : 0;
    }
  }
  return 0;
}

bool mlirAttributeIsATransformParamOperandAttr(MlirAttribute attribute) {
  return isAttribute<mlir::transform::ParamOperandAttr>(attribute);
}

MlirAttribute mlirTransformParamOperandAttrGet(MlirContext context, MlirAttribute index) {
  if (context.ptr == nullptr || index.ptr == nullptr) {
    return {nullptr};
  }
  auto indexAttribute = llvm::dyn_cast<mlir::IntegerAttr>(unwrap(index));
  if (!indexAttribute) {
    return {nullptr};
  }
  return wrap(mlir::transform::ParamOperandAttr::get(unwrap(context), indexAttribute));
}

MlirAttribute mlirTransformParamOperandAttrGetIndex(MlirAttribute attribute) {
  if (attribute.ptr == nullptr) {
    return {nullptr};
  }
  auto typedAttribute = llvm::dyn_cast<mlir::transform::ParamOperandAttr>(unwrap(attribute));
  if (!typedAttribute) {
    return {nullptr};
  }
  return wrap(typedAttribute.getIndex());
}
