#include "gpu.h"

#include <cstddef>
#include <cstdint>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace {

template <typename AttributeT>
bool isAttribute(MlirAttribute attribute) {
  return attribute.ptr != nullptr && llvm::isa<AttributeT>(unwrap(attribute));
}

template <typename EnumT, typename AttributeT>
MlirAttribute getEnumAttribute(MlirContext context, MlirStringRef value) {
  std::optional<EnumT> enumValue = mlir::gpu::symbolizeEnum<EnumT>(unwrap(value));
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
  return wrap(mlir::gpu::stringifyEnum(typedAttribute.getValue()));
}

template <typename AttributeT>
MlirAttribute getMappingAttribute(MlirContext context, MlirStringRef value) {
  std::optional<mlir::gpu::MappingId> mappingId = mlir::gpu::symbolizeEnum<mlir::gpu::MappingId>(unwrap(value));
  if (!mappingId) {
    return {nullptr};
  }
  return wrap(AttributeT::get(unwrap(context), *mappingId));
}

}  // namespace

bool mlirTypeIsAGpuMmaMatrixType(MlirType type) {
  return type.ptr != nullptr && llvm::isa<mlir::gpu::MMAMatrixType>(unwrap(type));
}

MlirType mlirGpuMmaMatrixTypeGet(
    MlirType elementType,
    const int64_t *shape,
    intptr_t shapeSize,
    MlirStringRef operand) {
  if (elementType.ptr == nullptr || shape == nullptr || shapeSize < 0) {
    return {nullptr};
  }
  llvm::ArrayRef<int64_t> shapeRef(shape, static_cast<size_t>(shapeSize));
  return wrap(mlir::gpu::MMAMatrixType::get(shapeRef, unwrap(elementType), unwrap(operand)));
}

intptr_t mlirGpuMmaMatrixTypeGetNumDims(MlirType type) {
  auto matrixType = llvm::dyn_cast<mlir::gpu::MMAMatrixType>(unwrap(type));
  if (!matrixType) {
    return 0;
  }
  return static_cast<intptr_t>(matrixType.getNumDims());
}

int64_t mlirGpuMmaMatrixTypeGetDimSize(MlirType type, intptr_t dimension) {
  auto matrixType = llvm::dyn_cast<mlir::gpu::MMAMatrixType>(unwrap(type));
  if (!matrixType || dimension < 0 || dimension >= static_cast<intptr_t>(matrixType.getShape().size())) {
    return 0;
  }
  return matrixType.getShape()[static_cast<size_t>(dimension)];
}

MlirType mlirGpuMmaMatrixTypeGetElementType(MlirType type) {
  auto matrixType = llvm::dyn_cast<mlir::gpu::MMAMatrixType>(unwrap(type));
  if (!matrixType) {
    return {nullptr};
  }
  return wrap(matrixType.getElementType());
}

MlirStringRef mlirGpuMmaMatrixTypeGetOperand(MlirType type) {
  auto matrixType = llvm::dyn_cast<mlir::gpu::MMAMatrixType>(unwrap(type));
  if (!matrixType) {
    return {nullptr, 0};
  }
  return wrap(matrixType.getOperand());
}

bool mlirTypeIsAGpuSparseHandleType(MlirType type, enum MlirGpuSparseHandleType kind) {
  if (type.ptr == nullptr) {
    return false;
  }
  switch (kind) {
    case RYFT_MLIR_GPU_SPARSE_DN_TENSOR_HANDLE_TYPE:
      return llvm::isa<mlir::gpu::SparseDnTensorHandleType>(unwrap(type));
    case RYFT_MLIR_GPU_SPARSE_SP_MAT_HANDLE_TYPE:
      return llvm::isa<mlir::gpu::SparseSpMatHandleType>(unwrap(type));
    case RYFT_MLIR_GPU_SPARSE_SP_GEMM_OPERATION_HANDLE_TYPE:
      return llvm::isa<mlir::gpu::SparseSpGEMMOpHandleType>(unwrap(type));
  }
  return false;
}

MlirType mlirGpuSparseHandleTypeGet(MlirContext context, enum MlirGpuSparseHandleType kind) {
  switch (kind) {
    case RYFT_MLIR_GPU_SPARSE_DN_TENSOR_HANDLE_TYPE:
      return wrap(mlir::gpu::SparseDnTensorHandleType::get(unwrap(context)));
    case RYFT_MLIR_GPU_SPARSE_SP_MAT_HANDLE_TYPE:
      return wrap(mlir::gpu::SparseSpMatHandleType::get(unwrap(context)));
    case RYFT_MLIR_GPU_SPARSE_SP_GEMM_OPERATION_HANDLE_TYPE:
      return wrap(mlir::gpu::SparseSpGEMMOpHandleType::get(unwrap(context)));
  }
  return {nullptr};
}

bool mlirAttributeIsAGpuKernelMetadataAttr(MlirAttribute attribute) {
  return isAttribute<mlir::gpu::KernelMetadataAttr>(attribute);
}

MlirAttribute mlirGpuKernelMetadataAttrGet(
    MlirStringRef name,
    MlirType functionType,
    MlirAttribute argumentAttributes,
    MlirAttribute metadata) {
  if (functionType.ptr == nullptr) {
    return {nullptr};
  }
  mlir::Type functionTypeValue = unwrap(functionType);
  mlir::MLIRContext *context = functionTypeValue.getContext();
  mlir::ArrayAttr argumentAttributesValue;
  if (argumentAttributes.ptr != nullptr) {
    argumentAttributesValue = llvm::dyn_cast<mlir::ArrayAttr>(unwrap(argumentAttributes));
    if (!argumentAttributesValue) {
      return {nullptr};
    }
  }
  mlir::DictionaryAttr metadataValue;
  if (metadata.ptr != nullptr) {
    metadataValue = llvm::dyn_cast<mlir::DictionaryAttr>(unwrap(metadata));
    if (!metadataValue) {
      return {nullptr};
    }
  }
  return wrap(mlir::gpu::KernelMetadataAttr::get(
      mlir::StringAttr::get(context, unwrap(name)),
      functionTypeValue,
      argumentAttributesValue,
      metadataValue));
}

bool mlirAttributeIsAGpuKernelTableAttr(MlirAttribute attribute) {
  return isAttribute<mlir::gpu::KernelTableAttr>(attribute);
}

MlirAttribute mlirGpuKernelTableAttrGet(
    MlirContext context,
    intptr_t kernelCount,
    const MlirAttribute *kernels) {
  if (kernelCount < 0 || (kernelCount > 0 && kernels == nullptr)) {
    return {nullptr};
  }
  llvm::SmallVector<mlir::gpu::KernelMetadataAttr, 4> kernelAttributes;
  kernelAttributes.reserve(static_cast<size_t>(kernelCount));
  for (intptr_t index = 0; index < kernelCount; ++index) {
    auto kernel = llvm::dyn_cast<mlir::gpu::KernelMetadataAttr>(unwrap(kernels[index]));
    if (!kernel) {
      return {nullptr};
    }
    kernelAttributes.push_back(kernel);
  }
  return wrap(mlir::gpu::KernelTableAttr::get(unwrap(context), kernelAttributes));
}

bool mlirAttributeIsAGpuSelectObjectAttr(MlirAttribute attribute) {
  return isAttribute<mlir::gpu::SelectObjectAttr>(attribute);
}

MlirAttribute mlirGpuSelectObjectAttrGet(MlirContext context, MlirAttribute target) {
  mlir::Attribute targetValue = target.ptr == nullptr ? mlir::Attribute() : unwrap(target);
  return wrap(mlir::gpu::SelectObjectAttr::get(unwrap(context), targetValue));
}

bool mlirAttributeIsAGpuEnumAttr(MlirAttribute attribute, enum MlirGpuEnumAttribute kind) {
  switch (kind) {
    case RYFT_MLIR_GPU_ENUM_ATTRIBUTE_ADDRESS_SPACE:
      return isAttribute<mlir::gpu::AddressSpaceAttr>(attribute);
    case RYFT_MLIR_GPU_ENUM_ATTRIBUTE_DIMENSION:
      return isAttribute<mlir::gpu::DimensionAttr>(attribute);
    case RYFT_MLIR_GPU_ENUM_ATTRIBUTE_ALL_REDUCE_OPERATION_KIND:
      return isAttribute<mlir::gpu::AllReduceOperationAttr>(attribute);
    case RYFT_MLIR_GPU_ENUM_ATTRIBUTE_SHUFFLE_MODE:
      return isAttribute<mlir::gpu::ShuffleModeAttr>(attribute);
    case RYFT_MLIR_GPU_ENUM_ATTRIBUTE_MMA_ELEMENTWISE_OPERATION:
      return isAttribute<mlir::gpu::MMAElementwiseOpAttr>(attribute);
    case RYFT_MLIR_GPU_ENUM_ATTRIBUTE_PRUNE_2_TO_4_SPARSE_MATRIX_FLAG:
      return isAttribute<mlir::gpu::Prune2To4SpMatFlagAttr>(attribute);
    case RYFT_MLIR_GPU_ENUM_ATTRIBUTE_MATRIX_TRANSPOSE_MODE:
      return isAttribute<mlir::gpu::TransposeModeAttr>(attribute);
    case RYFT_MLIR_GPU_ENUM_ATTRIBUTE_SP_GEMM_WORK_KIND:
      return isAttribute<mlir::gpu::SpGEMMWorkEstimationOrComputeKindAttr>(attribute);
    case RYFT_MLIR_GPU_ENUM_ATTRIBUTE_BROADCAST_TYPE:
      return isAttribute<mlir::gpu::BroadcastTypeAttr>(attribute);
  }
  return false;
}

MlirAttribute mlirGpuEnumAttrGet(
    MlirContext context,
    enum MlirGpuEnumAttribute kind,
    MlirStringRef value) {
  switch (kind) {
    case RYFT_MLIR_GPU_ENUM_ATTRIBUTE_ADDRESS_SPACE:
      return getEnumAttribute<mlir::gpu::AddressSpace, mlir::gpu::AddressSpaceAttr>(context, value);
    case RYFT_MLIR_GPU_ENUM_ATTRIBUTE_DIMENSION:
      return getEnumAttribute<mlir::gpu::Dimension, mlir::gpu::DimensionAttr>(context, value);
    case RYFT_MLIR_GPU_ENUM_ATTRIBUTE_ALL_REDUCE_OPERATION_KIND:
      return getEnumAttribute<mlir::gpu::AllReduceOperation, mlir::gpu::AllReduceOperationAttr>(context, value);
    case RYFT_MLIR_GPU_ENUM_ATTRIBUTE_SHUFFLE_MODE:
      return getEnumAttribute<mlir::gpu::ShuffleMode, mlir::gpu::ShuffleModeAttr>(context, value);
    case RYFT_MLIR_GPU_ENUM_ATTRIBUTE_MMA_ELEMENTWISE_OPERATION:
      return getEnumAttribute<mlir::gpu::MMAElementwiseOp, mlir::gpu::MMAElementwiseOpAttr>(context, value);
    case RYFT_MLIR_GPU_ENUM_ATTRIBUTE_PRUNE_2_TO_4_SPARSE_MATRIX_FLAG:
      return getEnumAttribute<mlir::gpu::Prune2To4SpMatFlag, mlir::gpu::Prune2To4SpMatFlagAttr>(context, value);
    case RYFT_MLIR_GPU_ENUM_ATTRIBUTE_MATRIX_TRANSPOSE_MODE:
      return getEnumAttribute<mlir::gpu::TransposeMode, mlir::gpu::TransposeModeAttr>(context, value);
    case RYFT_MLIR_GPU_ENUM_ATTRIBUTE_SP_GEMM_WORK_KIND:
      return getEnumAttribute<
          mlir::gpu::SpGEMMWorkEstimationOrComputeKind,
          mlir::gpu::SpGEMMWorkEstimationOrComputeKindAttr>(context, value);
    case RYFT_MLIR_GPU_ENUM_ATTRIBUTE_BROADCAST_TYPE:
      return getEnumAttribute<mlir::gpu::BroadcastType, mlir::gpu::BroadcastTypeAttr>(context, value);
  }
  return {nullptr};
}

MlirStringRef mlirGpuEnumAttrGetValue(MlirAttribute attribute, enum MlirGpuEnumAttribute kind) {
  switch (kind) {
    case RYFT_MLIR_GPU_ENUM_ATTRIBUTE_ADDRESS_SPACE:
      return getEnumAttributeValue<mlir::gpu::AddressSpaceAttr>(attribute);
    case RYFT_MLIR_GPU_ENUM_ATTRIBUTE_DIMENSION:
      return getEnumAttributeValue<mlir::gpu::DimensionAttr>(attribute);
    case RYFT_MLIR_GPU_ENUM_ATTRIBUTE_ALL_REDUCE_OPERATION_KIND:
      return getEnumAttributeValue<mlir::gpu::AllReduceOperationAttr>(attribute);
    case RYFT_MLIR_GPU_ENUM_ATTRIBUTE_SHUFFLE_MODE:
      return getEnumAttributeValue<mlir::gpu::ShuffleModeAttr>(attribute);
    case RYFT_MLIR_GPU_ENUM_ATTRIBUTE_MMA_ELEMENTWISE_OPERATION:
      return getEnumAttributeValue<mlir::gpu::MMAElementwiseOpAttr>(attribute);
    case RYFT_MLIR_GPU_ENUM_ATTRIBUTE_PRUNE_2_TO_4_SPARSE_MATRIX_FLAG:
      return getEnumAttributeValue<mlir::gpu::Prune2To4SpMatFlagAttr>(attribute);
    case RYFT_MLIR_GPU_ENUM_ATTRIBUTE_MATRIX_TRANSPOSE_MODE:
      return getEnumAttributeValue<mlir::gpu::TransposeModeAttr>(attribute);
    case RYFT_MLIR_GPU_ENUM_ATTRIBUTE_SP_GEMM_WORK_KIND:
      return getEnumAttributeValue<mlir::gpu::SpGEMMWorkEstimationOrComputeKindAttr>(attribute);
    case RYFT_MLIR_GPU_ENUM_ATTRIBUTE_BROADCAST_TYPE:
      return getEnumAttributeValue<mlir::gpu::BroadcastTypeAttr>(attribute);
  }
  return {nullptr, 0};
}

bool mlirAttributeIsAGpuMappingAttr(MlirAttribute attribute, enum MlirGpuMappingAttribute kind) {
  switch (kind) {
    case RYFT_MLIR_GPU_MAPPING_ATTRIBUTE_BLOCK:
      return isAttribute<mlir::gpu::GPUBlockMappingAttr>(attribute);
    case RYFT_MLIR_GPU_MAPPING_ATTRIBUTE_WARPGROUP:
      return isAttribute<mlir::gpu::GPUWarpgroupMappingAttr>(attribute);
    case RYFT_MLIR_GPU_MAPPING_ATTRIBUTE_WARP:
      return isAttribute<mlir::gpu::GPUWarpMappingAttr>(attribute);
    case RYFT_MLIR_GPU_MAPPING_ATTRIBUTE_THREAD:
      return isAttribute<mlir::gpu::GPUThreadMappingAttr>(attribute);
    case RYFT_MLIR_GPU_MAPPING_ATTRIBUTE_LANE:
      return isAttribute<mlir::gpu::GPULaneMappingAttr>(attribute);
  }
  return false;
}

MlirAttribute mlirGpuMappingAttrGet(
    MlirContext context,
    enum MlirGpuMappingAttribute kind,
    MlirStringRef value) {
  switch (kind) {
    case RYFT_MLIR_GPU_MAPPING_ATTRIBUTE_BLOCK:
      return getMappingAttribute<mlir::gpu::GPUBlockMappingAttr>(context, value);
    case RYFT_MLIR_GPU_MAPPING_ATTRIBUTE_WARPGROUP:
      return getMappingAttribute<mlir::gpu::GPUWarpgroupMappingAttr>(context, value);
    case RYFT_MLIR_GPU_MAPPING_ATTRIBUTE_WARP:
      return getMappingAttribute<mlir::gpu::GPUWarpMappingAttr>(context, value);
    case RYFT_MLIR_GPU_MAPPING_ATTRIBUTE_THREAD:
      return getMappingAttribute<mlir::gpu::GPUThreadMappingAttr>(context, value);
    case RYFT_MLIR_GPU_MAPPING_ATTRIBUTE_LANE:
      return getMappingAttribute<mlir::gpu::GPULaneMappingAttr>(context, value);
  }
  return {nullptr};
}

MlirStringRef mlirGpuMappingAttrGetValue(MlirAttribute attribute, enum MlirGpuMappingAttribute kind) {
  switch (kind) {
    case RYFT_MLIR_GPU_MAPPING_ATTRIBUTE_BLOCK: {
      auto typedAttribute = llvm::dyn_cast<mlir::gpu::GPUBlockMappingAttr>(unwrap(attribute));
      return typedAttribute ? wrap(mlir::gpu::stringifyEnum(typedAttribute.getBlock())) : MlirStringRef{nullptr, 0};
    }
    case RYFT_MLIR_GPU_MAPPING_ATTRIBUTE_WARPGROUP: {
      auto typedAttribute = llvm::dyn_cast<mlir::gpu::GPUWarpgroupMappingAttr>(unwrap(attribute));
      return typedAttribute ? wrap(mlir::gpu::stringifyEnum(typedAttribute.getWarpgroup()))
                            : MlirStringRef{nullptr, 0};
    }
    case RYFT_MLIR_GPU_MAPPING_ATTRIBUTE_WARP: {
      auto typedAttribute = llvm::dyn_cast<mlir::gpu::GPUWarpMappingAttr>(unwrap(attribute));
      return typedAttribute ? wrap(mlir::gpu::stringifyEnum(typedAttribute.getWarp())) : MlirStringRef{nullptr, 0};
    }
    case RYFT_MLIR_GPU_MAPPING_ATTRIBUTE_THREAD: {
      auto typedAttribute = llvm::dyn_cast<mlir::gpu::GPUThreadMappingAttr>(unwrap(attribute));
      return typedAttribute ? wrap(mlir::gpu::stringifyEnum(typedAttribute.getThread()))
                            : MlirStringRef{nullptr, 0};
    }
    case RYFT_MLIR_GPU_MAPPING_ATTRIBUTE_LANE: {
      auto typedAttribute = llvm::dyn_cast<mlir::gpu::GPULaneMappingAttr>(unwrap(attribute));
      return typedAttribute ? wrap(mlir::gpu::stringifyEnum(typedAttribute.getLane())) : MlirStringRef{nullptr, 0};
    }
  }
  return {nullptr, 0};
}

bool mlirAttributeIsAGpuMappingMaskAttr(MlirAttribute attribute) {
  return isAttribute<mlir::gpu::GPUMappingMaskAttr>(attribute);
}

MlirAttribute mlirGpuMappingMaskAttrGet(MlirContext context, uint64_t mask) {
  return wrap(mlir::gpu::GPUMappingMaskAttr::get(unwrap(context), mask));
}

uint64_t mlirGpuMappingMaskAttrGetMask(MlirAttribute attribute) {
  auto typedAttribute = llvm::dyn_cast<mlir::gpu::GPUMappingMaskAttr>(unwrap(attribute));
  return typedAttribute ? typedAttribute.getMask() : 0;
}

bool mlirAttributeIsAGpuMemorySpaceMappingAttr(MlirAttribute attribute) {
  return isAttribute<mlir::gpu::GPUMemorySpaceMappingAttr>(attribute);
}

MlirAttribute mlirGpuMemorySpaceMappingAttrGet(MlirContext context, MlirStringRef addressSpace) {
  std::optional<mlir::gpu::AddressSpace> value = mlir::gpu::symbolizeEnum<mlir::gpu::AddressSpace>(unwrap(addressSpace));
  if (!value) {
    return {nullptr};
  }
  return wrap(mlir::gpu::GPUMemorySpaceMappingAttr::get(unwrap(context), *value));
}

MlirStringRef mlirGpuMemorySpaceMappingAttrGetAddressSpace(MlirAttribute attribute) {
  auto typedAttribute = llvm::dyn_cast<mlir::gpu::GPUMemorySpaceMappingAttr>(unwrap(attribute));
  if (!typedAttribute) {
    return {nullptr, 0};
  }
  return wrap(mlir::gpu::stringifyEnum(typedAttribute.getAddressSpace()));
}

bool mlirAttributeIsAGpuParallelLoopDimMappingAttr(MlirAttribute attribute) {
  return isAttribute<mlir::gpu::ParallelLoopDimMappingAttr>(attribute);
}

MlirAttribute mlirGpuParallelLoopDimMappingAttrGet(
    MlirContext context,
    MlirStringRef processor,
    MlirAffineMap map,
    MlirAffineMap bound) {
  std::optional<mlir::gpu::Processor> processorValue = mlir::gpu::symbolizeEnum<mlir::gpu::Processor>(unwrap(processor));
  if (!processorValue || map.ptr == nullptr || bound.ptr == nullptr) {
    return {nullptr};
  }
  return wrap(mlir::gpu::ParallelLoopDimMappingAttr::get(unwrap(context), *processorValue, unwrap(map), unwrap(bound)));
}

MlirStringRef mlirGpuParallelLoopDimMappingAttrGetProcessor(MlirAttribute attribute) {
  auto typedAttribute = llvm::dyn_cast<mlir::gpu::ParallelLoopDimMappingAttr>(unwrap(attribute));
  if (!typedAttribute) {
    return {nullptr, 0};
  }
  return wrap(mlir::gpu::stringifyEnum(typedAttribute.getProcessor()));
}

MlirAffineMap mlirGpuParallelLoopDimMappingAttrGetMap(MlirAttribute attribute) {
  auto typedAttribute = llvm::dyn_cast<mlir::gpu::ParallelLoopDimMappingAttr>(unwrap(attribute));
  if (!typedAttribute) {
    return {nullptr};
  }
  return wrap(typedAttribute.getMap());
}

MlirAffineMap mlirGpuParallelLoopDimMappingAttrGetBound(MlirAttribute attribute) {
  auto typedAttribute = llvm::dyn_cast<mlir::gpu::ParallelLoopDimMappingAttr>(unwrap(attribute));
  if (!typedAttribute) {
    return {nullptr};
  }
  return wrap(typedAttribute.getBound());
}
