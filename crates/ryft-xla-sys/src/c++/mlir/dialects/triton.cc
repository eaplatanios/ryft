#include "triton.h"

#include <cstdint>
#include <optional>

#include "llvm/ADT/StringRef.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TritonTt, tt, mlir::triton::TritonDialect)

namespace {

template <typename AttributeT>
bool isAttribute(MlirAttribute attribute) {
  return attribute.ptr != nullptr && llvm::isa<AttributeT>(unwrap(attribute));
}

template <typename EnumT, typename AttributeT>
MlirAttribute getEnumAttribute(MlirContext context, MlirStringRef value) {
  if (context.ptr == nullptr || value.data == nullptr) {
    return {nullptr};
  }
  std::optional<EnumT> enumValue = mlir::triton::symbolizeEnum<EnumT>(unwrap(value));
  if (!enumValue) {
    return {nullptr};
  }
  return wrap(AttributeT::get(unwrap(context), *enumValue));
}

template <typename AttributeT>
MlirStringRef getEnumAttributeValue(MlirAttribute attribute) {
  if (attribute.ptr == nullptr) {
    return {nullptr, 0};
  }
  auto typedAttribute = llvm::dyn_cast<AttributeT>(unwrap(attribute));
  if (!typedAttribute) {
    return {nullptr, 0};
  }
  return wrap(mlir::triton::stringifyEnum(typedAttribute.getValue()));
}

}  // namespace

bool mlirTypeIsATritonTtPointerType(MlirType type) {
  return type.ptr != nullptr && llvm::isa<mlir::triton::PointerType>(unwrap(type));
}

MlirType mlirTritonTtPointerTypeGet(MlirType pointeeType, int32_t addressSpace) {
  if (pointeeType.ptr == nullptr) {
    return {nullptr};
  }
  return wrap(mlir::triton::PointerType::get(unwrap(pointeeType), static_cast<int>(addressSpace)));
}

MlirType mlirTritonTtPointerTypeGetPointeeType(MlirType type) {
  if (type.ptr == nullptr) {
    return {nullptr};
  }
  auto pointerType = llvm::dyn_cast<mlir::triton::PointerType>(unwrap(type));
  if (!pointerType) {
    return {nullptr};
  }
  return wrap(pointerType.getPointeeType());
}

int32_t mlirTritonTtPointerTypeGetAddressSpace(MlirType type) {
  if (type.ptr == nullptr) {
    return 0;
  }
  auto pointerType = llvm::dyn_cast<mlir::triton::PointerType>(unwrap(type));
  if (!pointerType) {
    return 0;
  }
  return static_cast<int32_t>(pointerType.getAddressSpace());
}

bool mlirTypeIsATritonTtTensorDescType(MlirType type) {
  return type.ptr != nullptr && llvm::isa<mlir::triton::TensorDescType>(unwrap(type));
}

MlirType mlirTritonTtTensorDescTypeGet(MlirType blockType) {
  if (blockType.ptr == nullptr) {
    return {nullptr};
  }
  auto rankedTensorType = llvm::dyn_cast<mlir::RankedTensorType>(unwrap(blockType));
  if (!rankedTensorType) {
    return {nullptr};
  }
  return wrap(mlir::triton::TensorDescType::get(rankedTensorType.getContext(), rankedTensorType));
}

MlirType mlirTritonTtTensorDescTypeGetBlockType(MlirType type) {
  if (type.ptr == nullptr) {
    return {nullptr};
  }
  auto tensorDescType = llvm::dyn_cast<mlir::triton::TensorDescType>(unwrap(type));
  if (!tensorDescType) {
    return {nullptr};
  }
  return wrap(tensorDescType.getBlockType());
}

bool mlirAttributeIsATritonTtEnumAttr(MlirAttribute attribute, enum MlirTritonTtEnumAttribute kind) {
  switch (kind) {
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_CACHE_MODIFIER:
      return isAttribute<mlir::triton::CacheModifierAttr>(attribute);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_MEM_SEMANTIC:
      return isAttribute<mlir::triton::MemSemanticAttr>(attribute);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_EVICTION_POLICY:
      return isAttribute<mlir::triton::EvictionPolicyAttr>(attribute);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_PADDING_OPTION:
      return isAttribute<mlir::triton::PaddingOptionAttr>(attribute);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_RMW_OP:
      return isAttribute<mlir::triton::RMWOpAttr>(attribute);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_DESCRIPTOR_REDUCE_KIND:
      return isAttribute<mlir::triton::DescriptorReduceKindAttr>(attribute);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_MEM_SYNC_SCOPE:
      return isAttribute<mlir::triton::MemSyncScopeAttr>(attribute);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_PROGRAM_ID_DIM:
      return isAttribute<mlir::triton::ProgramIDDimAttr>(attribute);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_ROUNDING_MODE:
      return isAttribute<mlir::triton::RoundingModeAttr>(attribute);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_PROPAGATE_NAN:
      return isAttribute<mlir::triton::PropagateNanAttr>(attribute);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_INPUT_PRECISION:
      return isAttribute<mlir::triton::InputPrecisionAttr>(attribute);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_SCALE_DOT_ELEM_TYPE:
      return isAttribute<mlir::triton::ScaleDotElemTypeAttr>(attribute);
  }
  return false;
}

MlirAttribute mlirTritonTtEnumAttrGet(
    MlirContext context,
    enum MlirTritonTtEnumAttribute kind,
    MlirStringRef value) {
  switch (kind) {
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_CACHE_MODIFIER:
      return getEnumAttribute<mlir::triton::CacheModifier, mlir::triton::CacheModifierAttr>(context, value);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_MEM_SEMANTIC:
      return getEnumAttribute<mlir::triton::MemSemantic, mlir::triton::MemSemanticAttr>(context, value);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_EVICTION_POLICY:
      return getEnumAttribute<mlir::triton::EvictionPolicy, mlir::triton::EvictionPolicyAttr>(context, value);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_PADDING_OPTION:
      return getEnumAttribute<mlir::triton::PaddingOption, mlir::triton::PaddingOptionAttr>(context, value);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_RMW_OP:
      return getEnumAttribute<mlir::triton::RMWOp, mlir::triton::RMWOpAttr>(context, value);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_DESCRIPTOR_REDUCE_KIND:
      return getEnumAttribute<
          mlir::triton::DescriptorReduceKind,
          mlir::triton::DescriptorReduceKindAttr>(context, value);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_MEM_SYNC_SCOPE:
      return getEnumAttribute<mlir::triton::MemSyncScope, mlir::triton::MemSyncScopeAttr>(context, value);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_PROGRAM_ID_DIM:
      return getEnumAttribute<mlir::triton::ProgramIDDim, mlir::triton::ProgramIDDimAttr>(context, value);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_ROUNDING_MODE:
      return getEnumAttribute<mlir::triton::RoundingMode, mlir::triton::RoundingModeAttr>(context, value);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_PROPAGATE_NAN:
      return getEnumAttribute<mlir::triton::PropagateNan, mlir::triton::PropagateNanAttr>(context, value);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_INPUT_PRECISION:
      return getEnumAttribute<mlir::triton::InputPrecision, mlir::triton::InputPrecisionAttr>(context, value);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_SCALE_DOT_ELEM_TYPE:
      return getEnumAttribute<mlir::triton::ScaleDotElemType, mlir::triton::ScaleDotElemTypeAttr>(context, value);
  }
  return {nullptr};
}

MlirStringRef mlirTritonTtEnumAttrGetValue(MlirAttribute attribute, enum MlirTritonTtEnumAttribute kind) {
  switch (kind) {
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_CACHE_MODIFIER:
      return getEnumAttributeValue<mlir::triton::CacheModifierAttr>(attribute);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_MEM_SEMANTIC:
      return getEnumAttributeValue<mlir::triton::MemSemanticAttr>(attribute);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_EVICTION_POLICY:
      return getEnumAttributeValue<mlir::triton::EvictionPolicyAttr>(attribute);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_PADDING_OPTION:
      return getEnumAttributeValue<mlir::triton::PaddingOptionAttr>(attribute);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_RMW_OP:
      return getEnumAttributeValue<mlir::triton::RMWOpAttr>(attribute);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_DESCRIPTOR_REDUCE_KIND:
      return getEnumAttributeValue<mlir::triton::DescriptorReduceKindAttr>(attribute);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_MEM_SYNC_SCOPE:
      return getEnumAttributeValue<mlir::triton::MemSyncScopeAttr>(attribute);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_PROGRAM_ID_DIM:
      return getEnumAttributeValue<mlir::triton::ProgramIDDimAttr>(attribute);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_ROUNDING_MODE:
      return getEnumAttributeValue<mlir::triton::RoundingModeAttr>(attribute);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_PROPAGATE_NAN:
      return getEnumAttributeValue<mlir::triton::PropagateNanAttr>(attribute);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_INPUT_PRECISION:
      return getEnumAttributeValue<mlir::triton::InputPrecisionAttr>(attribute);
    case MLIR_TRITON_TT_ENUM_ATTRIBUTE_SCALE_DOT_ELEM_TYPE:
      return getEnumAttributeValue<mlir::triton::ScaleDotElemTypeAttr>(attribute);
  }
  return {nullptr, 0};
}
