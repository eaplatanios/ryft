#![allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]

use crate::bindings::{
    MlirAffineMap, MlirAttribute, MlirContext, MlirSparseTensorLevelType, MlirType,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum MlirSparseTensorEnumAttribute {
    MLIR_SPARSE_TENSOR_ENUM_ATTRIBUTE_STORAGE_SPECIFIER_KIND = 0,
    MLIR_SPARSE_TENSOR_ENUM_ATTRIBUTE_SORT_KIND = 1,
    MLIR_SPARSE_TENSOR_ENUM_ATTRIBUTE_CRD_TRANS_DIRECTION = 2,
}

unsafe extern "C" {
    pub fn mlirAttributeIsASparseTensorDimSliceAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirSparseTensorDimSliceAttrGet(
        context: MlirContext,
        offset: i64,
        size: i64,
        stride: i64,
    ) -> MlirAttribute;
    pub fn mlirSparseTensorDimSliceAttrGetOffset(attribute: MlirAttribute) -> i64;
    pub fn mlirSparseTensorDimSliceAttrGetSize(attribute: MlirAttribute) -> i64;
    pub fn mlirSparseTensorDimSliceAttrGetStride(attribute: MlirAttribute) -> i64;

    pub fn mlirSparseTensorEncodingAttrGetWithDimSlices(
        context: MlirContext,
        level_rank: isize,
        level_types: *const MlirSparseTensorLevelType,
        dimension_to_level: MlirAffineMap,
        level_to_dimension: MlirAffineMap,
        position_width: ::std::os::raw::c_int,
        coordinate_width: ::std::os::raw::c_int,
        explicit_value: MlirAttribute,
        implicit_value: MlirAttribute,
        dimension_slice_count: isize,
        dimension_slices: *const MlirAttribute,
    ) -> MlirAttribute;
    pub fn mlirSparseTensorEncodingAttrGetDimSliceCount(attribute: MlirAttribute) -> isize;
    pub fn mlirSparseTensorEncodingAttrGetDimSlice(attribute: MlirAttribute, dimension: isize) -> MlirAttribute;

    pub fn mlirAttributeIsASparseTensorEnumAttr(
        attribute: MlirAttribute,
        kind: MlirSparseTensorEnumAttribute,
    ) -> bool;
    pub fn mlirSparseTensorEnumAttrGet(
        context: MlirContext,
        kind: MlirSparseTensorEnumAttribute,
        value: u32,
    ) -> MlirAttribute;
    pub fn mlirSparseTensorEnumAttrGetValue(
        attribute: MlirAttribute,
        kind: MlirSparseTensorEnumAttribute,
    ) -> u32;

    pub fn mlirTypeIsASparseTensorStorageSpecifierType(r#type: MlirType) -> bool;
    pub fn mlirSparseTensorStorageSpecifierTypeGet(context: MlirContext, encoding: MlirAttribute) -> MlirType;
    pub fn mlirSparseTensorStorageSpecifierTypeGetEncoding(r#type: MlirType) -> MlirAttribute;

    pub fn mlirTypeIsASparseTensorIterSpaceType(r#type: MlirType) -> bool;
    pub fn mlirSparseTensorIterSpaceTypeGet(
        context: MlirContext,
        encoding: MlirAttribute,
        lower_level: u64,
        upper_level: u64,
    ) -> MlirType;
    pub fn mlirSparseTensorIterSpaceTypeGetEncoding(r#type: MlirType) -> MlirAttribute;
    pub fn mlirSparseTensorIterSpaceTypeGetLowerLevel(r#type: MlirType) -> u64;
    pub fn mlirSparseTensorIterSpaceTypeGetUpperLevel(r#type: MlirType) -> u64;

    pub fn mlirTypeIsASparseTensorIteratorType(r#type: MlirType) -> bool;
    pub fn mlirSparseTensorIteratorTypeGet(
        context: MlirContext,
        encoding: MlirAttribute,
        lower_level: u64,
        upper_level: u64,
    ) -> MlirType;
    pub fn mlirSparseTensorIteratorTypeGetEncoding(r#type: MlirType) -> MlirAttribute;
    pub fn mlirSparseTensorIteratorTypeGetLowerLevel(r#type: MlirType) -> u64;
    pub fn mlirSparseTensorIteratorTypeGetUpperLevel(r#type: MlirType) -> u64;
}
