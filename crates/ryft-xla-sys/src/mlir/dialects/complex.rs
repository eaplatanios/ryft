#![allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]

use crate::bindings::{MlirAttribute, MlirContext, MlirDialectHandle, MlirLocation, MlirType, MlirTypeID};

unsafe extern "C" {
    pub fn mlirGetDialectHandle__complex__() -> MlirDialectHandle;
    pub fn mlirAttributeIsAComplex(attribute: MlirAttribute) -> bool;
    pub fn mlirComplexAttrDoubleGet(context: MlirContext, r#type: MlirType, real: f64, imaginary: f64)
    -> MlirAttribute;
    pub fn mlirComplexAttrDoubleGetChecked(
        location: MlirLocation,
        r#type: MlirType,
        real: f64,
        imaginary: f64,
    ) -> MlirAttribute;
    pub fn mlirComplexAttrGetRealDouble(attribute: MlirAttribute) -> f64;
    pub fn mlirComplexAttrGetImagDouble(attribute: MlirAttribute) -> f64;
    pub fn mlirComplexAttrGetTypeID() -> MlirTypeID;
}
