#![allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]

use crate::bindings::{MlirAttribute, MlirContext, MlirDialectHandle, MlirStringRef, MlirType};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum MlirTritonTtEnumAttribute {
    MLIR_TRITON_TT_ENUM_ATTRIBUTE_CACHE_MODIFIER = 0,
    MLIR_TRITON_TT_ENUM_ATTRIBUTE_MEM_SEMANTIC = 1,
    MLIR_TRITON_TT_ENUM_ATTRIBUTE_EVICTION_POLICY = 2,
    MLIR_TRITON_TT_ENUM_ATTRIBUTE_PADDING_OPTION = 3,
    MLIR_TRITON_TT_ENUM_ATTRIBUTE_RMW_OP = 4,
    MLIR_TRITON_TT_ENUM_ATTRIBUTE_DESCRIPTOR_REDUCE_KIND = 5,
    MLIR_TRITON_TT_ENUM_ATTRIBUTE_MEM_SYNC_SCOPE = 6,
    MLIR_TRITON_TT_ENUM_ATTRIBUTE_PROGRAM_ID_DIM = 7,
    MLIR_TRITON_TT_ENUM_ATTRIBUTE_ROUNDING_MODE = 8,
    MLIR_TRITON_TT_ENUM_ATTRIBUTE_PROPAGATE_NAN = 9,
    MLIR_TRITON_TT_ENUM_ATTRIBUTE_INPUT_PRECISION = 10,
    MLIR_TRITON_TT_ENUM_ATTRIBUTE_SCALE_DOT_ELEM_TYPE = 11,
}

unsafe extern "C" {
    pub fn mlirGetDialectHandle__tt__() -> MlirDialectHandle;

    pub fn mlirTypeIsATritonTtPointerType(r#type: MlirType) -> bool;
    pub fn mlirTritonTtPointerTypeGet(pointee_type: MlirType, address_space: i32) -> MlirType;
    pub fn mlirTritonTtPointerTypeGetPointeeType(r#type: MlirType) -> MlirType;
    pub fn mlirTritonTtPointerTypeGetAddressSpace(r#type: MlirType) -> i32;

    pub fn mlirTypeIsATritonTtTensorDescType(r#type: MlirType) -> bool;
    pub fn mlirTritonTtTensorDescTypeGet(block_type: MlirType) -> MlirType;
    pub fn mlirTritonTtTensorDescTypeGetBlockType(r#type: MlirType) -> MlirType;

    pub fn mlirAttributeIsATritonTtEnumAttr(attribute: MlirAttribute, kind: MlirTritonTtEnumAttribute) -> bool;
    pub fn mlirTritonTtEnumAttrGet(
        context: MlirContext,
        kind: MlirTritonTtEnumAttribute,
        value: MlirStringRef,
    ) -> MlirAttribute;
    pub fn mlirTritonTtEnumAttrGetValue(attribute: MlirAttribute, kind: MlirTritonTtEnumAttribute) -> MlirStringRef;
}
