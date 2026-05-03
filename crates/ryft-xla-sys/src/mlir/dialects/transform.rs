#![allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]

use crate::bindings::{MlirAttribute, MlirContext, MlirType};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum MlirTransformEnumAttribute {
    MLIR_TRANSFORM_ENUM_ATTRIBUTE_FAILURE_PROPAGATION_MODE = 0,
    MLIR_TRANSFORM_ENUM_ATTRIBUTE_MATCH_CMP_I_PREDICATE = 1,
}

unsafe extern "C" {
    pub fn mlirTypeIsATransformAffineMapParamType(r#type: MlirType) -> bool;
    pub fn mlirTransformAffineMapParamTypeGet(context: MlirContext) -> MlirType;

    pub fn mlirTypeIsATransformTypeParamType(r#type: MlirType) -> bool;
    pub fn mlirTransformTypeParamTypeGet(context: MlirContext) -> MlirType;

    pub fn mlirAttributeIsATransformEnumAttr(attribute: MlirAttribute, kind: MlirTransformEnumAttribute) -> bool;
    pub fn mlirTransformEnumAttrGet(
        context: MlirContext,
        kind: MlirTransformEnumAttribute,
        value: u32,
    ) -> MlirAttribute;
    pub fn mlirTransformEnumAttrGetValue(attribute: MlirAttribute, kind: MlirTransformEnumAttribute) -> u32;

    pub fn mlirAttributeIsATransformParamOperandAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirTransformParamOperandAttrGet(context: MlirContext, index: MlirAttribute) -> MlirAttribute;
    pub fn mlirTransformParamOperandAttrGetIndex(attribute: MlirAttribute) -> MlirAttribute;
}
