#include "mosaic_gpu.h"

#include <optional>

#include "llvm/ADT/StringRef.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/Attributes.h"
#include "jaxlib/mosaic/dialect/gpu/mosaic_gpu.h"

namespace {

template <typename AttributeT>
bool isAttribute(MlirAttribute attribute) {
  return attribute.ptr != nullptr && llvm::isa<AttributeT>(unwrap(attribute));
}

template <typename AttributeT>
AttributeT dynCastAttribute(MlirAttribute attribute) {
  if (attribute.ptr == nullptr) {
    return {};
  }
  return llvm::dyn_cast<AttributeT>(unwrap(attribute));
}

template <typename EnumT, typename AttributeT>
MlirAttribute getEnumAttribute(MlirContext context, MlirStringRef value) {
  if (context.ptr == nullptr || value.data == nullptr) {
    return {nullptr};
  }
  std::optional<EnumT> enumValue = mosaic_gpu::symbolizeEnum<EnumT>(unwrap(value));
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
  return wrap(mosaic_gpu::stringifyEnum(typedAttribute.getValue()));
}

}  // namespace

bool mlirAttributeIsAMosaicGpuEnumAttr(MlirAttribute attribute, enum MlirMosaicGpuEnumAttribute kind) {
  switch (kind) {
    case RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_DIMENSION:
      return isAttribute<mosaic_gpu::DimensionAttr>(attribute);
    case RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_SWIZZLING_MODE:
      return isAttribute<mosaic_gpu::SwizzlingModeAttr>(attribute);
    case RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_TMA_REDUCTION:
      return isAttribute<mosaic_gpu::TMAReductionAttr>(attribute);
    case RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_OOB_FILL_MODE:
      return isAttribute<mosaic_gpu::OOBFillModeAttr>(attribute);
    case RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_MULTIMEM_LOAD_REDUCTION_TYPE:
      return isAttribute<mosaic_gpu::MultimemLoadReductionTypeAttr>(attribute);
    case RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_ATOMIC_OP_TYPE:
      return isAttribute<mosaic_gpu::AtomicOpTypeAttr>(attribute);
  }
  return false;
}

MlirAttribute mlirMosaicGpuEnumAttrGet(
    MlirContext context,
    enum MlirMosaicGpuEnumAttribute kind,
    MlirStringRef value) {
  switch (kind) {
    case RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_DIMENSION:
      return getEnumAttribute<mosaic_gpu::Dimension, mosaic_gpu::DimensionAttr>(context, value);
    case RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_SWIZZLING_MODE:
      return getEnumAttribute<mosaic_gpu::SwizzlingMode, mosaic_gpu::SwizzlingModeAttr>(context, value);
    case RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_TMA_REDUCTION:
      return getEnumAttribute<mosaic_gpu::TMAReduction, mosaic_gpu::TMAReductionAttr>(context, value);
    case RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_OOB_FILL_MODE:
      return getEnumAttribute<mosaic_gpu::OOBFillMode, mosaic_gpu::OOBFillModeAttr>(context, value);
    case RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_MULTIMEM_LOAD_REDUCTION_TYPE:
      return getEnumAttribute<mosaic_gpu::MultimemLoadReductionType, mosaic_gpu::MultimemLoadReductionTypeAttr>(
          context,
          value);
    case RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_ATOMIC_OP_TYPE:
      return getEnumAttribute<mosaic_gpu::AtomicOpType, mosaic_gpu::AtomicOpTypeAttr>(context, value);
  }
  return {nullptr};
}

MlirStringRef mlirMosaicGpuEnumAttrGetValue(MlirAttribute attribute, enum MlirMosaicGpuEnumAttribute kind) {
  switch (kind) {
    case RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_DIMENSION:
      return getEnumAttributeValue<mosaic_gpu::DimensionAttr>(attribute);
    case RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_SWIZZLING_MODE:
      return getEnumAttributeValue<mosaic_gpu::SwizzlingModeAttr>(attribute);
    case RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_TMA_REDUCTION:
      return getEnumAttributeValue<mosaic_gpu::TMAReductionAttr>(attribute);
    case RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_OOB_FILL_MODE:
      return getEnumAttributeValue<mosaic_gpu::OOBFillModeAttr>(attribute);
    case RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_MULTIMEM_LOAD_REDUCTION_TYPE:
      return getEnumAttributeValue<mosaic_gpu::MultimemLoadReductionTypeAttr>(attribute);
    case RYFT_MLIR_MOSAIC_GPU_ENUM_ATTRIBUTE_ATOMIC_OP_TYPE:
      return getEnumAttributeValue<mosaic_gpu::AtomicOpTypeAttr>(attribute);
  }
  return {nullptr, 0};
}

bool mlirAttributeIsAMosaicGpuTmemAttr(MlirAttribute attribute) {
  return isAttribute<mosaic_gpu::TmemAttr>(attribute);
}

MlirAttribute mlirMosaicGpuTmemAttrGet(MlirContext context) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  return wrap(mosaic_gpu::TmemAttr::get(unwrap(context)));
}
