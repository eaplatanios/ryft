#![allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]

use std::ffi::c_uint;

use crate::bindings::{MlirAttribute, MlirContext, MlirStringRef, MlirType, MlirTypeID};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum MlirLlvmFramePointerKind {
    MLIR_LLVM_FRAME_POINTER_KIND_NONE = 0,
    MLIR_LLVM_FRAME_POINTER_KIND_NON_LEAF = 1,
    MLIR_LLVM_FRAME_POINTER_KIND_ALL = 2,
    MLIR_LLVM_FRAME_POINTER_KIND_RESERVED = 3,
    MLIR_LLVM_FRAME_POINTER_KIND_NON_LEAF_NO_RESERVE = 4,
}

unsafe extern "C" {
    pub fn mlirAttributeIsALlvmAddressSpaceAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmCConvAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmComdatAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmLinkageAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmFramePointerKindAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmLoopVectorizeAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmLoopInterleaveAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmLoopUnrollAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmLoopUnrollAndJamAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmLoopLicmAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmLoopDistributeAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmLoopPipelineAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmLoopPeeledAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmLoopUnswitchAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmLoopAnnotationAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmDiExpressionElemAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmDiExpressionAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmDiNullTypeAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmDiBasicTypeAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmDiCompileUnitAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmDiCompositeTypeAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmDiDerivedTypeAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmDiFileAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmDiGlobalVariableExpressionAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmDiGlobalVariableAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmDiLexicalBlockAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmDiLexicalBlockFileAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmDiLocalVariableAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmDiSubprogramAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmDiModuleAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmDiNamespaceAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmDiImportedEntityAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmDiAnnotationAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmDiSubrangeAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmDiCommonBlockAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmDiGenericSubrangeAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmDiSubroutineTypeAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmDiLabelAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmDiStringTypeAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmMemoryEffectsAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmDenormalFpEnvAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmAliasScopeDomainAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmAliasScopeAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmAccessGroupAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmTbaaRootAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmTbaaMemberAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmTbaaTypeDescriptorAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmTbaaTagAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmMmraTagAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmConstantRangeAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmVScaleRangeAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmTargetFeaturesAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmTargetAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmUndefAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmPoisonAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmDsoLocalEquivalentAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmBlockTagAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmBlockAddressAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmVecTypeHintAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmZeroAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmTailCallKindAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmWorkgroupAttributionAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmDereferenceableAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmModuleFlagAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmModuleFlagCgProfileEntryAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmModuleFlagProfileSummaryDetailedAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmModuleFlagProfileSummaryAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmDependentLibrariesAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmUwTableKindAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmMdStringAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmMdConstantAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmMdFuncAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsALlvmMdNodeAttr(attribute: MlirAttribute) -> bool;

    pub fn mlirLlvmAddressSpaceAttrGet(context: MlirContext, address_space: u32) -> MlirAttribute;
    pub fn mlirLlvmAddressSpaceAttrGetAddressSpace(attribute: MlirAttribute) -> u32;

    pub fn mlirLlvmCConvAttrGetValue(attribute: MlirAttribute) -> u32;
    pub fn mlirLlvmComdatAttrGetValue(attribute: MlirAttribute) -> u32;
    pub fn mlirLlvmLinkageAttrGetValue(attribute: MlirAttribute) -> u32;

    pub fn mlirLlvmFramePointerKindAttrGet(context: MlirContext, kind: MlirLlvmFramePointerKind) -> MlirAttribute;
    pub fn mlirLlvmFramePointerKindAttrGetValue(attribute: MlirAttribute) -> MlirLlvmFramePointerKind;

    pub fn mlirLlvmUndefAttrGet(context: MlirContext) -> MlirAttribute;
    pub fn mlirLlvmPoisonAttrGet(context: MlirContext) -> MlirAttribute;
    pub fn mlirLlvmZeroAttrGet(context: MlirContext) -> MlirAttribute;

    pub fn mlirLLVMMDStringAttrGet(context: MlirContext, value: MlirStringRef) -> MlirAttribute;
    pub fn mlirLLVMMDStringAttrGetValue(attribute: MlirAttribute) -> MlirStringRef;
    pub fn mlirLLVMMDConstantAttrGet(context: MlirContext, value_attribute: MlirAttribute) -> MlirAttribute;
    pub fn mlirLLVMMDConstantAttrGetValue(attribute: MlirAttribute) -> MlirAttribute;
    pub fn mlirLLVMMDFuncAttrGet(context: MlirContext, name: MlirAttribute) -> MlirAttribute;
    pub fn mlirLLVMMDFuncAttrGetName(attribute: MlirAttribute) -> MlirAttribute;
    pub fn mlirLLVMMDNodeAttrGet(
        context: MlirContext,
        operand_count: isize,
        operands: *const MlirAttribute,
    ) -> MlirAttribute;
    pub fn mlirLLVMMDNodeAttrGetNumOperands(attribute: MlirAttribute) -> isize;
    pub fn mlirLLVMMDNodeAttrGetOperand(attribute: MlirAttribute, index: isize) -> MlirAttribute;

    pub fn mlirTypeIsALLVMArrayType(r#type: MlirType) -> bool;
    pub fn mlirLLVMArrayTypeGetTypeID() -> MlirTypeID;
    pub fn mlirLLVMArrayTypeGetNumElements(r#type: MlirType) -> c_uint;

    pub fn mlirTypeIsALLVMFunctionType(r#type: MlirType) -> bool;
    pub fn mlirLLVMFunctionTypeGetTypeID() -> MlirTypeID;
    pub fn mlirLLVMFunctionTypeIsVarArg(r#type: MlirType) -> bool;

    pub fn mlirTypeIsALlvmVoidType(r#type: MlirType) -> bool;

    pub fn mlirTypeIsALlvmLabelType(r#type: MlirType) -> bool;
    pub fn mlirLlvmLabelTypeGet(context: MlirContext) -> MlirType;

    pub fn mlirTypeIsALlvmMetadataType(r#type: MlirType) -> bool;
    pub fn mlirLlvmMetadataTypeGet(context: MlirContext) -> MlirType;

    pub fn mlirTypeIsALlvmTargetExtType(r#type: MlirType) -> bool;
    pub fn mlirLlvmTargetExtTypeGet(
        context: MlirContext,
        name: MlirStringRef,
        type_param_count: isize,
        type_params: *const MlirType,
        integer_param_count: isize,
        integer_params: *const u32,
    ) -> MlirType;
    pub fn mlirLlvmTargetExtTypeGetName(r#type: MlirType) -> MlirStringRef;
    pub fn mlirLlvmTargetExtTypeGetNumTypeParams(r#type: MlirType) -> isize;
    pub fn mlirLlvmTargetExtTypeGetTypeParam(r#type: MlirType, position: isize) -> MlirType;
    pub fn mlirLlvmTargetExtTypeGetNumIntParams(r#type: MlirType) -> isize;
    pub fn mlirLlvmTargetExtTypeGetIntParam(r#type: MlirType, position: isize) -> u32;

    pub fn mlirTypeIsALlvmX86AmxType(r#type: MlirType) -> bool;
    pub fn mlirLlvmX86AmxTypeGet(context: MlirContext) -> MlirType;

    pub fn mlirTypeIsALlvmPpcFp128Type(r#type: MlirType) -> bool;
    pub fn mlirLlvmPpcFp128TypeGet(context: MlirContext) -> MlirType;
}
