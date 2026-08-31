#![allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]

use crate::bindings::{MlirAttribute, MlirContext, MlirDialectHandle, MlirTypeID};

unsafe extern "C" {
    pub fn mlirGetDialectHandle__ub__() -> MlirDialectHandle;
    pub fn mlirAttributeIsAUbPoisonAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirUbPoisonAttrGet(context: MlirContext) -> MlirAttribute;
    pub fn mlirUbPoisonAttrGetTypeID() -> MlirTypeID;
}
