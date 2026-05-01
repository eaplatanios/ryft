#![allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]

use crate::bindings::MlirDialectHandle;

unsafe extern "C" {
    pub fn mlirGetDialectHandle__affine__() -> MlirDialectHandle;
}
