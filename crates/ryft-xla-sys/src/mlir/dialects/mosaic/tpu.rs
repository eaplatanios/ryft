#![allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]

use crate::bindings::{MlirAttribute, MlirContext, MlirDialectHandle, MlirOperation, MlirStringRef, MlirType};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum MlirMosaicTpuEnumAttribute {
    RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_CORE_TYPE = 0,
    RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_PIPELINE_MODE = 1,
    RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_REVISIT_MODE = 2,
    RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_DIMENSION_SEMANTICS = 3,
    RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_CONTRACT_PRECISION = 4,
    RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_PACK_FORMAT = 5,
    RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_REDUCTION_KIND = 6,
    RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_ROUNDING_MODE = 7,
}

unsafe extern "C" {
    pub fn mlirGetDialectHandle__tpu__() -> MlirDialectHandle;
    pub fn mlirTPUAnalyzePotentialCommunication(
        operation: MlirOperation,
        has_communication: *mut bool,
        has_custom_barrier: *mut bool,
    );
    pub fn mlirTpuRegisterMosaicSerdePass();

    pub fn mlirTpuFloat8EXMYTypeGetUnderlyingType(exmy_type: MlirType) -> MlirType;
    pub fn mlirTpuIsAFloat8EXMYType(r#type: MlirType) -> bool;
    pub fn mlirTpuFloat8EXMYTypeGet(context: MlirContext, exmy_type: MlirType) -> MlirType;

    pub fn mlirTpuIsASemaphoreType(r#type: MlirType) -> bool;
    pub fn mlirTpuSemaphoreTypeGet(context: MlirContext) -> MlirType;
    pub fn mlirTpuIsADmaSemaphoreType(r#type: MlirType) -> bool;
    pub fn mlirTpuDmaSemaphoreTypeGet(context: MlirContext) -> MlirType;

    pub fn mlirAttributeIsAMosaicTpuEnumAttr(attribute: MlirAttribute, kind: MlirMosaicTpuEnumAttribute) -> bool;
    pub fn mlirMosaicTpuEnumAttrGet(
        context: MlirContext,
        kind: MlirMosaicTpuEnumAttribute,
        value: MlirStringRef,
    ) -> MlirAttribute;
    pub fn mlirMosaicTpuEnumAttrGetValue(attribute: MlirAttribute, kind: MlirMosaicTpuEnumAttribute) -> MlirStringRef;

    pub fn mlirAttributeIsAMosaicTpuDotDimensionNumbersAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirMosaicTpuDotDimensionNumbersAttrGet(
        context: MlirContext,
        lhs_contracting_dims: *const i64,
        lhs_contracting_dims_size: isize,
        rhs_contracting_dims: *const i64,
        rhs_contracting_dims_size: isize,
        lhs_non_contracting_dims: *const i64,
        lhs_non_contracting_dims_size: isize,
        rhs_non_contracting_dims: *const i64,
        rhs_non_contracting_dims_size: isize,
        output_dim_order: *const i64,
        output_dim_order_size: isize,
        lhs_batch_dims: *const i64,
        lhs_batch_dims_size: isize,
        rhs_batch_dims: *const i64,
        rhs_batch_dims_size: isize,
    ) -> MlirAttribute;
    pub fn mlirMosaicTpuDotDimensionNumbersAttrGetLhsContractingDims(attribute: MlirAttribute) -> MlirAttribute;
    pub fn mlirMosaicTpuDotDimensionNumbersAttrGetRhsContractingDims(attribute: MlirAttribute) -> MlirAttribute;
    pub fn mlirMosaicTpuDotDimensionNumbersAttrGetLhsNonContractingDims(attribute: MlirAttribute) -> MlirAttribute;
    pub fn mlirMosaicTpuDotDimensionNumbersAttrGetRhsNonContractingDims(attribute: MlirAttribute) -> MlirAttribute;
    pub fn mlirMosaicTpuDotDimensionNumbersAttrGetOutputDimOrder(attribute: MlirAttribute) -> MlirAttribute;
    pub fn mlirMosaicTpuDotDimensionNumbersAttrGetLhsBatchDims(attribute: MlirAttribute) -> MlirAttribute;
    pub fn mlirMosaicTpuDotDimensionNumbersAttrGetRhsBatchDims(attribute: MlirAttribute) -> MlirAttribute;

    pub fn mlirAttributeIsAMosaicTpuElementWindowAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirMosaicTpuElementWindowAttrGet(
        context: MlirContext,
        pad_low: *const i64,
        pad_low_size: isize,
        pad_high: *const i64,
        pad_high_size: isize,
    ) -> MlirAttribute;
    pub fn mlirMosaicTpuElementWindowAttrGetPadLow(attribute: MlirAttribute) -> MlirAttribute;
    pub fn mlirMosaicTpuElementWindowAttrGetPadHigh(attribute: MlirAttribute) -> MlirAttribute;

    pub fn mlirAttributeIsAMosaicTpuVectorLayoutAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirAttributeIsAMosaicTpuTiledLayoutAttr(attribute: MlirAttribute) -> bool;

    pub fn mlirAttributeIsAMosaicTpuMemorySpaceAttr(attribute: MlirAttribute) -> bool;
    pub fn mlirMosaicTpuMemorySpaceAttrGet(
        context: MlirContext,
        value: MlirStringRef,
        core_type: MlirStringRef,
    ) -> MlirAttribute;
    pub fn mlirMosaicTpuMemorySpaceAttrGetValue(attribute: MlirAttribute) -> MlirStringRef;
    pub fn mlirMosaicTpuMemorySpaceAttrHasCoreType(attribute: MlirAttribute) -> bool;
    pub fn mlirMosaicTpuMemorySpaceAttrGetCoreType(attribute: MlirAttribute) -> MlirStringRef;
}
