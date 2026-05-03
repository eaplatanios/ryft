#include "llvm.h"

#include <cstddef>
#include <cstdint>

#include "llvm/ADT/SmallVector.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
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

template <typename TypeT>
TypeT dynCastType(MlirType type) {
  if (type.ptr == nullptr) {
    return {};
  }
  return llvm::dyn_cast<TypeT>(unwrap(type));
}

}  // namespace

#define MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(functionName, attributeName) \
  bool functionName(MlirAttribute attribute) { return isAttribute<mlir::LLVM::attributeName>(attribute); }

MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmAddressSpaceAttr, AddressSpaceAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmCConvAttr, CConvAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmComdatAttr, ComdatAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmLinkageAttr, LinkageAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmFramePointerKindAttr, FramePointerKindAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmLoopVectorizeAttr, LoopVectorizeAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmLoopInterleaveAttr, LoopInterleaveAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmLoopUnrollAttr, LoopUnrollAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmLoopUnrollAndJamAttr, LoopUnrollAndJamAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmLoopLicmAttr, LoopLICMAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmLoopDistributeAttr, LoopDistributeAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmLoopPipelineAttr, LoopPipelineAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmLoopPeeledAttr, LoopPeeledAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmLoopUnswitchAttr, LoopUnswitchAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmLoopAnnotationAttr, LoopAnnotationAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmDiExpressionElemAttr, DIExpressionElemAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmDiExpressionAttr, DIExpressionAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmDiNullTypeAttr, DINullTypeAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmDiBasicTypeAttr, DIBasicTypeAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmDiCompileUnitAttr, DICompileUnitAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmDiCompositeTypeAttr, DICompositeTypeAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmDiDerivedTypeAttr, DIDerivedTypeAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmDiFileAttr, DIFileAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmDiGlobalVariableExpressionAttr, DIGlobalVariableExpressionAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmDiGlobalVariableAttr, DIGlobalVariableAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmDiLexicalBlockAttr, DILexicalBlockAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmDiLexicalBlockFileAttr, DILexicalBlockFileAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmDiLocalVariableAttr, DILocalVariableAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmDiSubprogramAttr, DISubprogramAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmDiModuleAttr, DIModuleAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmDiNamespaceAttr, DINamespaceAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmDiImportedEntityAttr, DIImportedEntityAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmDiAnnotationAttr, DIAnnotationAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmDiSubrangeAttr, DISubrangeAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmDiCommonBlockAttr, DICommonBlockAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmDiGenericSubrangeAttr, DIGenericSubrangeAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmDiSubroutineTypeAttr, DISubroutineTypeAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmDiLabelAttr, DILabelAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmDiStringTypeAttr, DIStringTypeAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmMemoryEffectsAttr, MemoryEffectsAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmDenormalFpEnvAttr, DenormalFPEnvAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmAliasScopeDomainAttr, AliasScopeDomainAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmAliasScopeAttr, AliasScopeAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmAccessGroupAttr, AccessGroupAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmTbaaRootAttr, TBAARootAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmTbaaMemberAttr, TBAAMemberAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmTbaaTypeDescriptorAttr, TBAATypeDescriptorAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmTbaaTagAttr, TBAATagAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmMmraTagAttr, MMRATagAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmConstantRangeAttr, ConstantRangeAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmVScaleRangeAttr, VScaleRangeAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmTargetFeaturesAttr, TargetFeaturesAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmTargetAttr, TargetAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmUndefAttr, UndefAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmPoisonAttr, PoisonAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmDsoLocalEquivalentAttr, DSOLocalEquivalentAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmBlockTagAttr, BlockTagAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmBlockAddressAttr, BlockAddressAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmVecTypeHintAttr, VecTypeHintAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmZeroAttr, ZeroAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmTailCallKindAttr, TailCallKindAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmWorkgroupAttributionAttr, WorkgroupAttributionAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmDereferenceableAttr, DereferenceableAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmModuleFlagAttr, ModuleFlagAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmModuleFlagCgProfileEntryAttr, ModuleFlagCGProfileEntryAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(
    mlirAttributeIsALlvmModuleFlagProfileSummaryDetailedAttr,
    ModuleFlagProfileSummaryDetailedAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmModuleFlagProfileSummaryAttr, ModuleFlagProfileSummaryAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmDependentLibrariesAttr, DependentLibrariesAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmUwTableKindAttr, UWTableKindAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmMdStringAttr, MDStringAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmMdConstantAttr, MDConstantAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmMdFuncAttr, MDFuncAttr)
MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION(mlirAttributeIsALlvmMdNodeAttr, MDNodeAttr)

#undef MLIR_LLVM_ATTRIBUTE_ISA_FUNCTION

MlirAttribute mlirLlvmAddressSpaceAttrGet(MlirContext context, uint32_t addressSpace) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  return wrap(mlir::LLVM::AddressSpaceAttr::get(unwrap(context), addressSpace));
}

uint32_t mlirLlvmAddressSpaceAttrGetAddressSpace(MlirAttribute attribute) {
  if (!isAttribute<mlir::LLVM::AddressSpaceAttr>(attribute)) {
    return 0;
  }
  return llvm::cast<mlir::LLVM::AddressSpaceAttr>(unwrap(attribute)).getAddressSpace();
}

uint32_t mlirLlvmCConvAttrGetValue(MlirAttribute attribute) {
  if (!isAttribute<mlir::LLVM::CConvAttr>(attribute)) {
    return 0;
  }
  return static_cast<uint32_t>(llvm::cast<mlir::LLVM::CConvAttr>(unwrap(attribute)).getCallingConv());
}

uint32_t mlirLlvmComdatAttrGetValue(MlirAttribute attribute) {
  if (!isAttribute<mlir::LLVM::ComdatAttr>(attribute)) {
    return 0;
  }
  return static_cast<uint32_t>(llvm::cast<mlir::LLVM::ComdatAttr>(unwrap(attribute)).getComdat());
}

uint32_t mlirLlvmLinkageAttrGetValue(MlirAttribute attribute) {
  if (!isAttribute<mlir::LLVM::LinkageAttr>(attribute)) {
    return 0;
  }
  return static_cast<uint32_t>(llvm::cast<mlir::LLVM::LinkageAttr>(unwrap(attribute)).getLinkage());
}

MlirAttribute mlirLlvmFramePointerKindAttrGet(MlirContext context, MlirLlvmFramePointerKind kind) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  return wrap(mlir::LLVM::FramePointerKindAttr::get(
      unwrap(context), static_cast<mlir::LLVM::framePointerKind::FramePointerKind>(kind)));
}

MlirLlvmFramePointerKind mlirLlvmFramePointerKindAttrGetValue(MlirAttribute attribute) {
  if (!isAttribute<mlir::LLVM::FramePointerKindAttr>(attribute)) {
    return MLIR_LLVM_FRAME_POINTER_KIND_NONE;
  }
  auto kind = llvm::cast<mlir::LLVM::FramePointerKindAttr>(unwrap(attribute)).getFramePointerKind();
  return static_cast<MlirLlvmFramePointerKind>(kind);
}

MlirAttribute mlirLlvmUndefAttrGet(MlirContext context) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  return wrap(mlir::LLVM::UndefAttr::get(unwrap(context)));
}

MlirAttribute mlirLlvmPoisonAttrGet(MlirContext context) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  return wrap(mlir::LLVM::PoisonAttr::get(unwrap(context)));
}

MlirAttribute mlirLlvmZeroAttrGet(MlirContext context) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  return wrap(mlir::LLVM::ZeroAttr::get(unwrap(context)));
}

bool mlirTypeIsALlvmVoidType(MlirType type) {
  return isType<mlir::LLVM::LLVMVoidType>(type);
}

bool mlirTypeIsALlvmTokenType(MlirType type) {
  return isType<mlir::LLVM::LLVMTokenType>(type);
}

MlirType mlirLlvmTokenTypeGet(MlirContext context) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  return wrap(mlir::LLVM::LLVMTokenType::get(unwrap(context)));
}

bool mlirTypeIsALlvmLabelType(MlirType type) {
  return isType<mlir::LLVM::LLVMLabelType>(type);
}

MlirType mlirLlvmLabelTypeGet(MlirContext context) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  return wrap(mlir::LLVM::LLVMLabelType::get(unwrap(context)));
}

bool mlirTypeIsALlvmMetadataType(MlirType type) {
  return isType<mlir::LLVM::LLVMMetadataType>(type);
}

MlirType mlirLlvmMetadataTypeGet(MlirContext context) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  return wrap(mlir::LLVM::LLVMMetadataType::get(unwrap(context)));
}

bool mlirTypeIsALlvmTargetExtType(MlirType type) {
  return isType<mlir::LLVM::LLVMTargetExtType>(type);
}

MlirType mlirLlvmTargetExtTypeGet(
    MlirContext context,
    MlirStringRef name,
    intptr_t typeParamCount,
    const MlirType *typeParams,
    intptr_t intParamCount,
    const uint32_t *intParams) {
  if (context.ptr == nullptr || typeParamCount < 0 || intParamCount < 0) {
    return {nullptr};
  }
  if ((typeParamCount > 0 && typeParams == nullptr) || (intParamCount > 0 && intParams == nullptr)) {
    return {nullptr};
  }

  llvm::SmallVector<mlir::Type, 4> typeParamValues;
  typeParamValues.reserve(static_cast<size_t>(typeParamCount));
  for (intptr_t index = 0; index < typeParamCount; ++index) {
    if (typeParams[index].ptr == nullptr) {
      return {nullptr};
    }
    typeParamValues.push_back(unwrap(typeParams[index]));
  }

  llvm::SmallVector<unsigned int, 4> intParamValues;
  intParamValues.reserve(static_cast<size_t>(intParamCount));
  for (intptr_t index = 0; index < intParamCount; ++index) {
    intParamValues.push_back(static_cast<unsigned int>(intParams[index]));
  }

  return wrap(mlir::LLVM::LLVMTargetExtType::get(unwrap(context), unwrap(name), typeParamValues, intParamValues));
}

MlirStringRef mlirLlvmTargetExtTypeGetName(MlirType type) {
  auto typedType = dynCastType<mlir::LLVM::LLVMTargetExtType>(type);
  return typedType ? wrap(typedType.getExtTypeName()) : MlirStringRef{nullptr, 0};
}

intptr_t mlirLlvmTargetExtTypeGetNumTypeParams(MlirType type) {
  auto typedType = dynCastType<mlir::LLVM::LLVMTargetExtType>(type);
  return typedType ? static_cast<intptr_t>(typedType.getTypeParams().size()) : 0;
}

MlirType mlirLlvmTargetExtTypeGetTypeParam(MlirType type, intptr_t position) {
  auto typedType = dynCastType<mlir::LLVM::LLVMTargetExtType>(type);
  if (!typedType || position < 0 || position >= static_cast<intptr_t>(typedType.getTypeParams().size())) {
    return {nullptr};
  }
  return wrap(typedType.getTypeParams()[static_cast<size_t>(position)]);
}

intptr_t mlirLlvmTargetExtTypeGetNumIntParams(MlirType type) {
  auto typedType = dynCastType<mlir::LLVM::LLVMTargetExtType>(type);
  return typedType ? static_cast<intptr_t>(typedType.getIntParams().size()) : 0;
}

uint32_t mlirLlvmTargetExtTypeGetIntParam(MlirType type, intptr_t position) {
  auto typedType = dynCastType<mlir::LLVM::LLVMTargetExtType>(type);
  if (!typedType || position < 0 || position >= static_cast<intptr_t>(typedType.getIntParams().size())) {
    return 0;
  }
  return static_cast<uint32_t>(typedType.getIntParams()[static_cast<size_t>(position)]);
}

bool mlirTypeIsALlvmX86AmxType(MlirType type) {
  return isType<mlir::LLVM::LLVMX86AMXType>(type);
}

MlirType mlirLlvmX86AmxTypeGet(MlirContext context) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  return wrap(mlir::LLVM::LLVMX86AMXType::get(unwrap(context)));
}

bool mlirTypeIsALlvmPpcFp128Type(MlirType type) {
  return isType<mlir::LLVM::LLVMPPCFP128Type>(type);
}

MlirType mlirLlvmPpcFp128TypeGet(MlirContext context) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  return wrap(mlir::LLVM::LLVMPPCFP128Type::get(unwrap(context)));
}
