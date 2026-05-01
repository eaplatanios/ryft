#include "nvgpu.h"

#include <cstdint>
#include <optional>

#include "llvm/ADT/StringRef.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace {

template <typename TypeT>
bool isType(MlirType type) {
  return type.ptr != nullptr && llvm::isa<TypeT>(unwrap(type));
}

template <typename TypeT>
TypeT dynCastType(MlirType type) {
  if (type.ptr == nullptr) {
    return {};
  }
  return llvm::dyn_cast<TypeT>(unwrap(type));
}

template <typename AttributeT>
bool isAttribute(MlirAttribute attribute) {
  return attribute.ptr != nullptr && llvm::isa<AttributeT>(unwrap(attribute));
}

template <typename EnumT>
std::optional<EnumT> symbolizeEnumValue(uint32_t value);

template <>
std::optional<mlir::nvgpu::TensorMapSwizzleKind> symbolizeEnumValue(uint32_t value) {
  return mlir::nvgpu::symbolizeTensorMapSwizzleKind(value);
}

template <>
std::optional<mlir::nvgpu::TensorMapL2PromoKind> symbolizeEnumValue(uint32_t value) {
  return mlir::nvgpu::symbolizeTensorMapL2PromoKind(value);
}

template <>
std::optional<mlir::nvgpu::TensorMapOOBKind> symbolizeEnumValue(uint32_t value) {
  return mlir::nvgpu::symbolizeTensorMapOOBKind(value);
}

template <>
std::optional<mlir::nvgpu::TensorMapInterleaveKind> symbolizeEnumValue(uint32_t value) {
  return mlir::nvgpu::symbolizeTensorMapInterleaveKind(value);
}

template <>
std::optional<mlir::nvgpu::RcpRoundingMode> symbolizeEnumValue(uint32_t value) {
  return mlir::nvgpu::symbolizeRcpRoundingMode(value);
}

template <typename EnumT, typename AttributeT>
MlirAttribute getEnumAttribute(MlirContext context, uint32_t value) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  std::optional<EnumT> enumValue = symbolizeEnumValue<EnumT>(value);
  if (!enumValue) {
    return {nullptr};
  }
  return wrap(AttributeT::get(unwrap(context), *enumValue));
}

template <typename AttributeT>
uint32_t getEnumAttributeValue(MlirAttribute attribute) {
  if (attribute.ptr == nullptr) {
    return 0;
  }
  auto typedAttribute = llvm::dyn_cast<AttributeT>(unwrap(attribute));
  return typedAttribute ? static_cast<uint32_t>(typedAttribute.getValue()) : 0;
}

}  // namespace

bool mlirTypeIsANvgpuDeviceAsyncTokenType(MlirType type) {
  return isType<mlir::nvgpu::DeviceAsyncTokenType>(type);
}

MlirType mlirNvgpuDeviceAsyncTokenTypeGet(MlirContext context) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  return wrap(mlir::nvgpu::DeviceAsyncTokenType::get(unwrap(context)));
}

bool mlirTypeIsANvgpuMBarrierGroupType(MlirType type) {
  return isType<mlir::nvgpu::MBarrierGroupType>(type);
}

MlirType mlirNvgpuMBarrierGroupTypeGet(MlirContext context, MlirAttribute memory_space, uint32_t num_barriers) {
  if (context.ptr == nullptr || memory_space.ptr == nullptr) {
    return {nullptr};
  }
  return wrap(mlir::nvgpu::MBarrierGroupType::get(unwrap(context), unwrap(memory_space), num_barriers));
}

MlirAttribute mlirNvgpuMBarrierGroupTypeGetMemorySpace(MlirType type) {
  auto typedType = dynCastType<mlir::nvgpu::MBarrierGroupType>(type);
  return typedType ? wrap(typedType.getMemorySpace()) : MlirAttribute{nullptr};
}

uint32_t mlirNvgpuMBarrierGroupTypeGetNumBarriers(MlirType type) {
  auto typedType = dynCastType<mlir::nvgpu::MBarrierGroupType>(type);
  return typedType ? typedType.getNumBarriers() : 0;
}

bool mlirTypeIsANvgpuMBarrierTokenType(MlirType type) {
  return isType<mlir::nvgpu::MBarrierTokenType>(type);
}

MlirType mlirNvgpuMBarrierTokenTypeGet(MlirContext context) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  return wrap(mlir::nvgpu::MBarrierTokenType::get(unwrap(context)));
}

MlirType mlirNvgpuTensorMapDescriptorTypeGetTensor(MlirType type) {
  auto typedType = dynCastType<mlir::nvgpu::TensorMapDescriptorType>(type);
  return typedType ? wrap(typedType.getTensor()) : MlirType{nullptr};
}

uint32_t mlirNvgpuTensorMapDescriptorTypeGetSwizzle(MlirType type) {
  auto typedType = dynCastType<mlir::nvgpu::TensorMapDescriptorType>(type);
  return typedType ? static_cast<uint32_t>(typedType.getSwizzle()) : 0;
}

uint32_t mlirNvgpuTensorMapDescriptorTypeGetL2Promo(MlirType type) {
  auto typedType = dynCastType<mlir::nvgpu::TensorMapDescriptorType>(type);
  return typedType ? static_cast<uint32_t>(typedType.getL2promo()) : 0;
}

uint32_t mlirNvgpuTensorMapDescriptorTypeGetOob(MlirType type) {
  auto typedType = dynCastType<mlir::nvgpu::TensorMapDescriptorType>(type);
  return typedType ? static_cast<uint32_t>(typedType.getOob()) : 0;
}

uint32_t mlirNvgpuTensorMapDescriptorTypeGetInterleave(MlirType type) {
  auto typedType = dynCastType<mlir::nvgpu::TensorMapDescriptorType>(type);
  return typedType ? static_cast<uint32_t>(typedType.getInterleave()) : 0;
}

bool mlirTypeIsANvgpuWarpgroupMatrixDescriptorType(MlirType type) {
  return isType<mlir::nvgpu::WarpgroupMatrixDescriptorType>(type);
}

MlirType mlirNvgpuWarpgroupMatrixDescriptorTypeGet(MlirContext context, MlirType tensor) {
  auto tensorType = dynCastType<mlir::MemRefType>(tensor);
  if (context.ptr == nullptr || !tensorType) {
    return {nullptr};
  }
  return wrap(mlir::nvgpu::WarpgroupMatrixDescriptorType::get(unwrap(context), tensorType));
}

MlirType mlirNvgpuWarpgroupMatrixDescriptorTypeGetTensor(MlirType type) {
  auto typedType = dynCastType<mlir::nvgpu::WarpgroupMatrixDescriptorType>(type);
  return typedType ? wrap(typedType.getTensor()) : MlirType{nullptr};
}

bool mlirTypeIsANvgpuWarpgroupAccumulatorType(MlirType type) {
  return isType<mlir::nvgpu::WarpgroupAccumulatorType>(type);
}

MlirType mlirNvgpuWarpgroupAccumulatorTypeGet(MlirContext context, MlirType fragmented) {
  auto vectorType = dynCastType<mlir::VectorType>(fragmented);
  if (context.ptr == nullptr || !vectorType) {
    return {nullptr};
  }
  return wrap(mlir::nvgpu::WarpgroupAccumulatorType::get(unwrap(context), vectorType));
}

MlirType mlirNvgpuWarpgroupAccumulatorTypeGetFragmented(MlirType type) {
  auto typedType = dynCastType<mlir::nvgpu::WarpgroupAccumulatorType>(type);
  return typedType ? wrap(typedType.getFragmented()) : MlirType{nullptr};
}

bool mlirAttributeIsANvgpuEnumAttr(MlirAttribute attribute, enum MlirNvgpuEnumAttribute kind) {
  switch (kind) {
    case RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_TENSOR_MAP_SWIZZLE_KIND:
      return isAttribute<mlir::nvgpu::TensorMapSwizzleKindAttr>(attribute);
    case RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_TENSOR_MAP_L2_PROMO_KIND:
      return isAttribute<mlir::nvgpu::TensorMapL2PromoKindAttr>(attribute);
    case RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_TENSOR_MAP_OOB_KIND:
      return isAttribute<mlir::nvgpu::TensorMapOOBKindAttr>(attribute);
    case RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_TENSOR_MAP_INTERLEAVE_KIND:
      return isAttribute<mlir::nvgpu::TensorMapInterleaveKindAttr>(attribute);
    case RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_RCP_ROUNDING_MODE:
      return isAttribute<mlir::nvgpu::RcpRoundingModeAttr>(attribute);
  }
  return false;
}

MlirAttribute mlirNvgpuEnumAttrGet(MlirContext context, enum MlirNvgpuEnumAttribute kind, uint32_t value) {
  switch (kind) {
    case RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_TENSOR_MAP_SWIZZLE_KIND:
      return getEnumAttribute<mlir::nvgpu::TensorMapSwizzleKind, mlir::nvgpu::TensorMapSwizzleKindAttr>(
          context,
          value);
    case RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_TENSOR_MAP_L2_PROMO_KIND:
      return getEnumAttribute<mlir::nvgpu::TensorMapL2PromoKind, mlir::nvgpu::TensorMapL2PromoKindAttr>(
          context,
          value);
    case RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_TENSOR_MAP_OOB_KIND:
      return getEnumAttribute<mlir::nvgpu::TensorMapOOBKind, mlir::nvgpu::TensorMapOOBKindAttr>(context, value);
    case RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_TENSOR_MAP_INTERLEAVE_KIND:
      return getEnumAttribute<mlir::nvgpu::TensorMapInterleaveKind, mlir::nvgpu::TensorMapInterleaveKindAttr>(
          context,
          value);
    case RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_RCP_ROUNDING_MODE:
      return getEnumAttribute<mlir::nvgpu::RcpRoundingMode, mlir::nvgpu::RcpRoundingModeAttr>(context, value);
  }
  return {nullptr};
}

uint32_t mlirNvgpuEnumAttrGetValue(MlirAttribute attribute, enum MlirNvgpuEnumAttribute kind) {
  switch (kind) {
    case RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_TENSOR_MAP_SWIZZLE_KIND:
      return getEnumAttributeValue<mlir::nvgpu::TensorMapSwizzleKindAttr>(attribute);
    case RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_TENSOR_MAP_L2_PROMO_KIND:
      return getEnumAttributeValue<mlir::nvgpu::TensorMapL2PromoKindAttr>(attribute);
    case RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_TENSOR_MAP_OOB_KIND:
      return getEnumAttributeValue<mlir::nvgpu::TensorMapOOBKindAttr>(attribute);
    case RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_TENSOR_MAP_INTERLEAVE_KIND:
      return getEnumAttributeValue<mlir::nvgpu::TensorMapInterleaveKindAttr>(attribute);
    case RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_RCP_ROUNDING_MODE:
      return getEnumAttributeValue<mlir::nvgpu::RcpRoundingModeAttr>(attribute);
  }
  return 0;
}
