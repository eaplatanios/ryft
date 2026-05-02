#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  MLIR_LLVM_FRAME_POINTER_KIND_NONE = 0,
  MLIR_LLVM_FRAME_POINTER_KIND_NON_LEAF = 1,
  MLIR_LLVM_FRAME_POINTER_KIND_ALL = 2,
  MLIR_LLVM_FRAME_POINTER_KIND_RESERVED = 3,
  MLIR_LLVM_FRAME_POINTER_KIND_NON_LEAF_NO_RESERVE = 4,
} MlirLlvmFramePointerKind;

bool mlirAttributeIsALlvmAddressSpaceAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmCConvAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmComdatAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmLinkageAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmFramePointerKindAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmLoopVectorizeAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmLoopInterleaveAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmLoopUnrollAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmLoopUnrollAndJamAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmLoopLicmAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmLoopDistributeAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmLoopPipelineAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmLoopPeeledAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmLoopUnswitchAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmLoopAnnotationAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmDiExpressionElemAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmDiExpressionAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmDiNullTypeAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmDiBasicTypeAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmDiCompileUnitAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmDiCompositeTypeAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmDiDerivedTypeAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmDiFileAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmDiGlobalVariableExpressionAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmDiGlobalVariableAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmDiLexicalBlockAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmDiLexicalBlockFileAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmDiLocalVariableAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmDiSubprogramAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmDiModuleAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmDiNamespaceAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmDiImportedEntityAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmDiAnnotationAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmDiSubrangeAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmDiCommonBlockAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmDiGenericSubrangeAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmDiSubroutineTypeAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmDiLabelAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmDiStringTypeAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmMemoryEffectsAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmDenormalFpEnvAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmAliasScopeDomainAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmAliasScopeAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmAccessGroupAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmTbaaRootAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmTbaaMemberAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmTbaaTypeDescriptorAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmTbaaTagAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmMmraTagAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmConstantRangeAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmVScaleRangeAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmTargetFeaturesAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmTargetAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmUndefAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmPoisonAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmDsoLocalEquivalentAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmBlockTagAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmBlockAddressAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmVecTypeHintAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmZeroAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmTailCallKindAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmWorkgroupAttributionAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmDereferenceableAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmModuleFlagAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmModuleFlagCgProfileEntryAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmModuleFlagProfileSummaryDetailedAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmModuleFlagProfileSummaryAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmDependentLibrariesAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmUwTableKindAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmMdStringAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmMdConstantAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmMdFuncAttr(MlirAttribute attribute);
bool mlirAttributeIsALlvmMdNodeAttr(MlirAttribute attribute);

MlirAttribute mlirLlvmAddressSpaceAttrGet(MlirContext context, uint32_t addressSpace);
uint32_t mlirLlvmAddressSpaceAttrGetAddressSpace(MlirAttribute attribute);

uint32_t mlirLlvmCConvAttrGetValue(MlirAttribute attribute);
uint32_t mlirLlvmComdatAttrGetValue(MlirAttribute attribute);
uint32_t mlirLlvmLinkageAttrGetValue(MlirAttribute attribute);

MlirAttribute mlirLlvmFramePointerKindAttrGet(MlirContext context, MlirLlvmFramePointerKind kind);
MlirLlvmFramePointerKind mlirLlvmFramePointerKindAttrGetValue(MlirAttribute attribute);

MlirAttribute mlirLlvmUndefAttrGet(MlirContext context);
MlirAttribute mlirLlvmPoisonAttrGet(MlirContext context);
MlirAttribute mlirLlvmZeroAttrGet(MlirContext context);

bool mlirTypeIsALlvmVoidType(MlirType type);

bool mlirTypeIsALlvmTokenType(MlirType type);
MlirType mlirLlvmTokenTypeGet(MlirContext context);

bool mlirTypeIsALlvmLabelType(MlirType type);
MlirType mlirLlvmLabelTypeGet(MlirContext context);

bool mlirTypeIsALlvmMetadataType(MlirType type);
MlirType mlirLlvmMetadataTypeGet(MlirContext context);

bool mlirTypeIsALlvmTargetExtType(MlirType type);
MlirType mlirLlvmTargetExtTypeGet(
    MlirContext context,
    MlirStringRef name,
    intptr_t typeParamCount,
    const MlirType *typeParams,
    intptr_t intParamCount,
    const uint32_t *intParams);
MlirStringRef mlirLlvmTargetExtTypeGetName(MlirType type);
intptr_t mlirLlvmTargetExtTypeGetNumTypeParams(MlirType type);
MlirType mlirLlvmTargetExtTypeGetTypeParam(MlirType type, intptr_t position);
intptr_t mlirLlvmTargetExtTypeGetNumIntParams(MlirType type);
uint32_t mlirLlvmTargetExtTypeGetIntParam(MlirType type, intptr_t position);

bool mlirTypeIsALlvmX86AmxType(MlirType type);
MlirType mlirLlvmX86AmxTypeGet(MlirContext context);

bool mlirTypeIsALlvmPpcFp128Type(MlirType type);
MlirType mlirLlvmPpcFp128TypeGet(MlirContext context);

#ifdef __cplusplus
}
#endif
