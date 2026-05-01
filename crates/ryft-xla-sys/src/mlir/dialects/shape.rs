#![allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]

use crate::bindings::{MlirContext, MlirType};

unsafe extern "C" {
    pub fn mlirTypeIsAShapeShapeType(r#type: MlirType) -> bool;
    pub fn mlirShapeShapeTypeGet(context: MlirContext) -> MlirType;

    pub fn mlirTypeIsAShapeSizeType(r#type: MlirType) -> bool;
    pub fn mlirShapeSizeTypeGet(context: MlirContext) -> MlirType;

    pub fn mlirTypeIsAShapeValueShapeType(r#type: MlirType) -> bool;
    pub fn mlirShapeValueShapeTypeGet(context: MlirContext) -> MlirType;

    pub fn mlirTypeIsAShapeWitnessType(r#type: MlirType) -> bool;
    pub fn mlirShapeWitnessTypeGet(context: MlirContext) -> MlirType;
}
