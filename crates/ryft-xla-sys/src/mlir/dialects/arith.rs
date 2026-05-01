#![allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]

use crate::bindings::{MlirAttribute, MlirContext};

unsafe extern "C" {
    pub fn mlirAttributeIsAArithAtomicRmwKindAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirArithAtomicRmwKindAttrGet(context: MlirContext, value: u64) -> MlirAttribute;
    pub fn mlirArithAtomicRmwKindAttrGetValue(attribute: MlirAttribute) -> u64;

    pub fn mlirAttributeIsAArithFastMathFlagsAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirArithFastMathFlagsAttrGet(context: MlirContext, value: u32) -> MlirAttribute;
    pub fn mlirArithFastMathFlagsAttrGetValue(attribute: MlirAttribute) -> u32;

    pub fn mlirAttributeIsAArithIntegerOverflowFlagsAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirArithIntegerOverflowFlagsAttrGet(context: MlirContext, value: u32) -> MlirAttribute;
    pub fn mlirArithIntegerOverflowFlagsAttrGetValue(attribute: MlirAttribute) -> u32;

    pub fn mlirAttributeIsAArithRoundingModeAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirArithRoundingModeAttrGet(context: MlirContext, value: u32) -> MlirAttribute;
    pub fn mlirArithRoundingModeAttrGetValue(attribute: MlirAttribute) -> u32;
}
