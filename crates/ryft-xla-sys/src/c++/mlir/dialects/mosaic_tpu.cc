#include "mosaic_tpu.h"

#include <cstddef>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"

namespace {

template <typename TypeT>
bool isType(MlirType type) {
  return type.ptr != nullptr && llvm::isa<TypeT>(unwrap(type));
}

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
  std::optional<EnumT> enumValue = mlir::tpu::symbolizeEnum<EnumT>(unwrap(value));
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
  return wrap(mlir::tpu::stringifyEnum(typedAttribute.getValue()));
}

llvm::ArrayRef<int64_t> getArrayRef(const int64_t *values, intptr_t size) {
  if (values == nullptr || size <= 0) {
    return {};
  }
  return llvm::ArrayRef<int64_t>(values, static_cast<size_t>(size));
}

MlirAttribute getDenseI64ArrayAttr(mlir::MLIRContext *context, llvm::ArrayRef<int64_t> values) {
  return wrap(mlir::DenseI64ArrayAttr::get(context, values));
}

}  // namespace

bool mlirTpuIsASemaphoreType(MlirType type) {
  return isType<mlir::tpu::SemaphoreType>(type);
}

MlirType mlirTpuSemaphoreTypeGet(MlirContext context) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  return wrap(mlir::tpu::SemaphoreType::get(unwrap(context)));
}

bool mlirTpuIsADmaSemaphoreType(MlirType type) {
  return isType<mlir::tpu::DMASemaphoreType>(type);
}

MlirType mlirTpuDmaSemaphoreTypeGet(MlirContext context) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  return wrap(mlir::tpu::DMASemaphoreType::get(unwrap(context)));
}

bool mlirAttributeIsAMosaicTpuEnumAttr(MlirAttribute attribute, enum MlirMosaicTpuEnumAttribute kind) {
  switch (kind) {
    case RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_CORE_TYPE:
      return isAttribute<mlir::tpu::CoreTypeAttr>(attribute);
    case RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_PIPELINE_MODE:
      return isAttribute<mlir::tpu::PipelineModeAttr>(attribute);
    case RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_REVISIT_MODE:
      return isAttribute<mlir::tpu::RevisitModeAttr>(attribute);
    case RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_DIMENSION_SEMANTICS:
      return isAttribute<mlir::tpu::DimensionSemanticsAttr>(attribute);
    case RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_CONTRACT_PRECISION:
      return isAttribute<mlir::tpu::ContractPrecisionAttr>(attribute);
    case RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_PACK_FORMAT:
      return isAttribute<mlir::tpu::PackFormatAttr>(attribute);
    case RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_REDUCTION_KIND:
      return isAttribute<mlir::tpu::ReductionKindAttr>(attribute);
    case RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_ROUNDING_MODE:
      return isAttribute<mlir::tpu::RoundingModeAttr>(attribute);
  }
  return false;
}

MlirAttribute mlirMosaicTpuEnumAttrGet(
    MlirContext context,
    enum MlirMosaicTpuEnumAttribute kind,
    MlirStringRef value) {
  switch (kind) {
    case RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_CORE_TYPE:
      return getEnumAttribute<mlir::tpu::CoreType, mlir::tpu::CoreTypeAttr>(context, value);
    case RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_PIPELINE_MODE:
      return getEnumAttribute<mlir::tpu::PipelineMode, mlir::tpu::PipelineModeAttr>(context, value);
    case RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_REVISIT_MODE:
      return getEnumAttribute<mlir::tpu::RevisitMode, mlir::tpu::RevisitModeAttr>(context, value);
    case RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_DIMENSION_SEMANTICS:
      return getEnumAttribute<mlir::tpu::DimensionSemantics, mlir::tpu::DimensionSemanticsAttr>(context, value);
    case RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_CONTRACT_PRECISION:
      return getEnumAttribute<mlir::tpu::ContractPrecision, mlir::tpu::ContractPrecisionAttr>(context, value);
    case RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_PACK_FORMAT:
      return getEnumAttribute<mlir::tpu::PackFormat, mlir::tpu::PackFormatAttr>(context, value);
    case RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_REDUCTION_KIND:
      return getEnumAttribute<mlir::tpu::ReductionKind, mlir::tpu::ReductionKindAttr>(context, value);
    case RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_ROUNDING_MODE:
      return getEnumAttribute<mlir::tpu::RoundingMode, mlir::tpu::RoundingModeAttr>(context, value);
  }
  return {nullptr};
}

MlirStringRef mlirMosaicTpuEnumAttrGetValue(MlirAttribute attribute, enum MlirMosaicTpuEnumAttribute kind) {
  switch (kind) {
    case RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_CORE_TYPE:
      return getEnumAttributeValue<mlir::tpu::CoreTypeAttr>(attribute);
    case RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_PIPELINE_MODE:
      return getEnumAttributeValue<mlir::tpu::PipelineModeAttr>(attribute);
    case RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_REVISIT_MODE:
      return getEnumAttributeValue<mlir::tpu::RevisitModeAttr>(attribute);
    case RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_DIMENSION_SEMANTICS:
      return getEnumAttributeValue<mlir::tpu::DimensionSemanticsAttr>(attribute);
    case RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_CONTRACT_PRECISION:
      return getEnumAttributeValue<mlir::tpu::ContractPrecisionAttr>(attribute);
    case RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_PACK_FORMAT:
      return getEnumAttributeValue<mlir::tpu::PackFormatAttr>(attribute);
    case RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_REDUCTION_KIND:
      return getEnumAttributeValue<mlir::tpu::ReductionKindAttr>(attribute);
    case RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_ROUNDING_MODE:
      return getEnumAttributeValue<mlir::tpu::RoundingModeAttr>(attribute);
  }
  return {nullptr, 0};
}

bool mlirAttributeIsAMosaicTpuDotDimensionNumbersAttr(MlirAttribute attribute) {
  return isAttribute<mlir::tpu::DotDimensionNumbersAttr>(attribute);
}

MlirAttribute mlirMosaicTpuDotDimensionNumbersAttrGet(
    MlirContext context,
    const int64_t *lhs_contracting_dims,
    intptr_t lhs_contracting_dims_size,
    const int64_t *rhs_contracting_dims,
    intptr_t rhs_contracting_dims_size,
    const int64_t *lhs_non_contracting_dims,
    intptr_t lhs_non_contracting_dims_size,
    const int64_t *rhs_non_contracting_dims,
    intptr_t rhs_non_contracting_dims_size,
    const int64_t *output_dim_order,
    intptr_t output_dim_order_size,
    const int64_t *lhs_batch_dims,
    intptr_t lhs_batch_dims_size,
    const int64_t *rhs_batch_dims,
    intptr_t rhs_batch_dims_size) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  return wrap(mlir::tpu::DotDimensionNumbersAttr::get(
      unwrap(context),
      getArrayRef(lhs_contracting_dims, lhs_contracting_dims_size),
      getArrayRef(rhs_contracting_dims, rhs_contracting_dims_size),
      getArrayRef(lhs_non_contracting_dims, lhs_non_contracting_dims_size),
      getArrayRef(rhs_non_contracting_dims, rhs_non_contracting_dims_size),
      getArrayRef(output_dim_order, output_dim_order_size),
      getArrayRef(lhs_batch_dims, lhs_batch_dims_size),
      getArrayRef(rhs_batch_dims, rhs_batch_dims_size)));
}

MlirAttribute mlirMosaicTpuDotDimensionNumbersAttrGetLhsContractingDims(MlirAttribute attribute) {
  auto typedAttribute = dynCastAttribute<mlir::tpu::DotDimensionNumbersAttr>(attribute);
  return typedAttribute ? getDenseI64ArrayAttr(typedAttribute.getContext(), typedAttribute.getLhsContractingDims())
                        : MlirAttribute{nullptr};
}

MlirAttribute mlirMosaicTpuDotDimensionNumbersAttrGetRhsContractingDims(MlirAttribute attribute) {
  auto typedAttribute = dynCastAttribute<mlir::tpu::DotDimensionNumbersAttr>(attribute);
  return typedAttribute ? getDenseI64ArrayAttr(typedAttribute.getContext(), typedAttribute.getRhsContractingDims())
                        : MlirAttribute{nullptr};
}

MlirAttribute mlirMosaicTpuDotDimensionNumbersAttrGetLhsNonContractingDims(MlirAttribute attribute) {
  auto typedAttribute = dynCastAttribute<mlir::tpu::DotDimensionNumbersAttr>(attribute);
  return typedAttribute ? getDenseI64ArrayAttr(typedAttribute.getContext(), typedAttribute.getLhsNonContractingDims())
                        : MlirAttribute{nullptr};
}

MlirAttribute mlirMosaicTpuDotDimensionNumbersAttrGetRhsNonContractingDims(MlirAttribute attribute) {
  auto typedAttribute = dynCastAttribute<mlir::tpu::DotDimensionNumbersAttr>(attribute);
  return typedAttribute ? getDenseI64ArrayAttr(typedAttribute.getContext(), typedAttribute.getRhsNonContractingDims())
                        : MlirAttribute{nullptr};
}

MlirAttribute mlirMosaicTpuDotDimensionNumbersAttrGetOutputDimOrder(MlirAttribute attribute) {
  auto typedAttribute = dynCastAttribute<mlir::tpu::DotDimensionNumbersAttr>(attribute);
  return typedAttribute ? getDenseI64ArrayAttr(typedAttribute.getContext(), typedAttribute.getOutputDimOrder())
                        : MlirAttribute{nullptr};
}

MlirAttribute mlirMosaicTpuDotDimensionNumbersAttrGetLhsBatchDims(MlirAttribute attribute) {
  auto typedAttribute = dynCastAttribute<mlir::tpu::DotDimensionNumbersAttr>(attribute);
  return typedAttribute ? getDenseI64ArrayAttr(typedAttribute.getContext(), typedAttribute.getLhsBatchDims())
                        : MlirAttribute{nullptr};
}

MlirAttribute mlirMosaicTpuDotDimensionNumbersAttrGetRhsBatchDims(MlirAttribute attribute) {
  auto typedAttribute = dynCastAttribute<mlir::tpu::DotDimensionNumbersAttr>(attribute);
  return typedAttribute ? getDenseI64ArrayAttr(typedAttribute.getContext(), typedAttribute.getRhsBatchDims())
                        : MlirAttribute{nullptr};
}

bool mlirAttributeIsAMosaicTpuElementWindowAttr(MlirAttribute attribute) {
  return isAttribute<mlir::tpu::ElementWindowAttr>(attribute);
}

MlirAttribute mlirMosaicTpuElementWindowAttrGet(
    MlirContext context,
    const int64_t *pad_low,
    intptr_t pad_low_size,
    const int64_t *pad_high,
    intptr_t pad_high_size) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  return wrap(mlir::tpu::ElementWindowAttr::get(
      unwrap(context),
      getArrayRef(pad_low, pad_low_size),
      getArrayRef(pad_high, pad_high_size)));
}

MlirAttribute mlirMosaicTpuElementWindowAttrGetPadLow(MlirAttribute attribute) {
  auto typedAttribute = dynCastAttribute<mlir::tpu::ElementWindowAttr>(attribute);
  return typedAttribute ? getDenseI64ArrayAttr(typedAttribute.getContext(), typedAttribute.getPadLow())
                        : MlirAttribute{nullptr};
}

MlirAttribute mlirMosaicTpuElementWindowAttrGetPadHigh(MlirAttribute attribute) {
  auto typedAttribute = dynCastAttribute<mlir::tpu::ElementWindowAttr>(attribute);
  return typedAttribute ? getDenseI64ArrayAttr(typedAttribute.getContext(), typedAttribute.getPadHigh())
                        : MlirAttribute{nullptr};
}

bool mlirAttributeIsAMosaicTpuVectorLayoutAttr(MlirAttribute attribute) {
  return isAttribute<mlir::tpu::VectorLayoutAttr>(attribute);
}

bool mlirAttributeIsAMosaicTpuTiledLayoutAttr(MlirAttribute attribute) {
  return isAttribute<mlir::tpu::TiledLayoutAttr>(attribute);
}

bool mlirAttributeIsAMosaicTpuMemorySpaceAttr(MlirAttribute attribute) {
  return isAttribute<mlir::tpu::MemorySpaceAttr>(attribute);
}

MlirAttribute mlirMosaicTpuMemorySpaceAttrGet(
    MlirContext context,
    MlirStringRef value,
    MlirStringRef core_type) {
  if (context.ptr == nullptr || value.data == nullptr) {
    return {nullptr};
  }
  std::optional<mlir::tpu::MemorySpace> memorySpace = mlir::tpu::symbolizeEnum<mlir::tpu::MemorySpace>(unwrap(value));
  if (!memorySpace) {
    return {nullptr};
  }
  std::optional<mlir::tpu::CoreType> coreType = std::nullopt;
  if (core_type.data != nullptr) {
    coreType = mlir::tpu::symbolizeEnum<mlir::tpu::CoreType>(unwrap(core_type));
    if (!coreType) {
      return {nullptr};
    }
  }
  return wrap(mlir::tpu::MemorySpaceAttr::get(unwrap(context), *memorySpace, coreType));
}

MlirStringRef mlirMosaicTpuMemorySpaceAttrGetValue(MlirAttribute attribute) {
  auto typedAttribute = dynCastAttribute<mlir::tpu::MemorySpaceAttr>(attribute);
  if (!typedAttribute) {
    return {nullptr, 0};
  }
  return wrap(mlir::tpu::stringifyEnum(typedAttribute.getValue()));
}

bool mlirMosaicTpuMemorySpaceAttrHasCoreType(MlirAttribute attribute) {
  auto typedAttribute = dynCastAttribute<mlir::tpu::MemorySpaceAttr>(attribute);
  return typedAttribute && typedAttribute.getCoreType().has_value();
}

MlirStringRef mlirMosaicTpuMemorySpaceAttrGetCoreType(MlirAttribute attribute) {
  auto typedAttribute = dynCastAttribute<mlir::tpu::MemorySpaceAttr>(attribute);
  if (!typedAttribute || !typedAttribute.getCoreType()) {
    return {nullptr, 0};
  }
  return wrap(mlir::tpu::stringifyEnum(*typedAttribute.getCoreType()));
}
