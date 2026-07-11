#![allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]

use crate::bindings::{MlirContext, MlirType};

unsafe extern "C" {
    pub fn mlirTypeIsAToken(r#type: MlirType) -> bool;
    pub fn mlirTokenTypeGet(context: MlirContext) -> MlirType;
}
