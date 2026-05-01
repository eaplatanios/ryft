use ryft_xla_sys::bindings::{MlirAttribute, MlirStringRef};
use ryft_xla_sys::mlir::dialects::mosaic::tpu::{
    MlirMosaicTpuEnumAttribute, mlirAttributeIsAMosaicTpuDotDimensionNumbersAttr,
    mlirAttributeIsAMosaicTpuElementWindowAttr, mlirAttributeIsAMosaicTpuEnumAttr,
    mlirAttributeIsAMosaicTpuMemorySpaceAttr, mlirAttributeIsAMosaicTpuTiledLayoutAttr,
    mlirAttributeIsAMosaicTpuVectorLayoutAttr, mlirMosaicTpuDotDimensionNumbersAttrGet,
    mlirMosaicTpuDotDimensionNumbersAttrGetLhsBatchDims, mlirMosaicTpuDotDimensionNumbersAttrGetLhsContractingDims,
    mlirMosaicTpuDotDimensionNumbersAttrGetLhsNonContractingDims,
    mlirMosaicTpuDotDimensionNumbersAttrGetOutputDimOrder, mlirMosaicTpuDotDimensionNumbersAttrGetRhsBatchDims,
    mlirMosaicTpuDotDimensionNumbersAttrGetRhsContractingDims,
    mlirMosaicTpuDotDimensionNumbersAttrGetRhsNonContractingDims, mlirMosaicTpuElementWindowAttrGet,
    mlirMosaicTpuElementWindowAttrGetPadHigh, mlirMosaicTpuElementWindowAttrGetPadLow, mlirMosaicTpuEnumAttrGet,
    mlirMosaicTpuEnumAttrGetValue, mlirMosaicTpuMemorySpaceAttrGet, mlirMosaicTpuMemorySpaceAttrGetCoreType,
    mlirMosaicTpuMemorySpaceAttrGetValue, mlirMosaicTpuMemorySpaceAttrHasCoreType,
};

use crate::{Attribute, Context, DenseInteger64ArrayAttributeRef, DialectHandle, StringRef, mlir_subtype_trait_impls};

macro_rules! mosaic_tpu_enum_attribute {
    (
        enum_name = $enum_name:ident,
        attribute_name = $attribute_name:ident,
        context_method = $context_method:ident,
        ffi_kind = $ffi_kind:path,
        mnemonic = $mnemonic:literal,
        description = $description:literal,
        variants = { $($variant:ident => ($value:literal, $integer:literal)),+ $(,)* } $(,)*
    ) => {
        #[doc = "Mosaic TPU "]
        #[doc = $description]
        #[doc = "."]
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
        pub enum $enum_name {
            $($variant,)+
        }

        impl $enum_name {
            /// Returns the MLIR spelling of this enum value.
            pub fn as_str(&self) -> &'static str {
                match self {
                    $(Self::$variant => $value,)+
                }
            }

            /// Returns the integer value associated with this enum value in the Mosaic TPU dialect.
            pub fn as_i32(&self) -> i32 {
                match self {
                    $(Self::$variant => $integer,)+
                }
            }

            /// Parses the MLIR spelling of this enum value.
            pub fn from_str(value: &str) -> Option<Self> {
                match value {
                    $($value => Some(Self::$variant),)+
                    _ => None,
                }
            }

            /// Parses the integer value associated with this enum value.
            pub fn from_i32(value: i32) -> Option<Self> {
                match value {
                    $($integer => Some(Self::$variant),)+
                    _ => None,
                }
            }
        }

        #[doc = "Mosaic TPU "]
        #[doc = $description]
        #[doc = " [`Attribute`]."]
        #[derive(Copy, Clone)]
        pub struct $attribute_name<'c, 't> {
            /// Handle that represents this [`Attribute`] in the MLIR C API.
            handle: MlirAttribute,

            /// [`Context`] that owns this [`Attribute`].
            context: &'c Context<'t>,
        }

        impl $attribute_name<'_, '_> {
            /// Returns the enum value stored in this attribute.
            pub fn value(&self) -> $enum_name {
                let value = unsafe { StringRef::from_c_api(mlirMosaicTpuEnumAttrGetValue(self.handle, $ffi_kind)) };
                value
                    .as_str()
                    .ok()
                    .and_then($enum_name::from_str)
                    .expect(concat!("invalid Mosaic TPU `", $mnemonic, "` attribute"))
            }
        }

        impl<'c, 't> Attribute<'c, 't> for $attribute_name<'c, 't> {
            unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
                if !handle.ptr.is_null() && unsafe { mlirAttributeIsAMosaicTpuEnumAttr(handle, $ffi_kind) } {
                    Some(Self { handle, context })
                } else {
                    None
                }
            }

            unsafe fn to_c_api(&self) -> MlirAttribute {
                self.handle
            }

            fn context(&self) -> &'c Context<'t> {
                self.context
            }
        }

        mlir_subtype_trait_impls!($attribute_name<'c, 't> as Attribute, mlir_type = Attribute);

        impl<'t> Context<'t> {
            #[doc = "Creates a Mosaic TPU "]
            #[doc = $description]
            #[doc = " attribute owned by this [`Context`]."]
            pub fn $context_method<'c>(&'c self, value: $enum_name) -> $attribute_name<'c, 't> {
                self.load_dialect(DialectHandle::mosaic_tpu());
                let value = StringRef::from(value.as_str());
                unsafe {
                    $attribute_name::from_c_api(
                        mlirMosaicTpuEnumAttrGet(*self.handle.borrow_mut(), $ffi_kind, value.to_c_api()),
                        self,
                    )
                    .expect(concat!("invalid arguments to `Context::", stringify!($context_method), "`"))
                }
            }
        }
    };
}

mosaic_tpu_enum_attribute!(
    enum_name = CoreType,
    attribute_name = CoreTypeAttributeRef,
    context_method = mosaic_tpu_core_type_attribute,
    ffi_kind = MlirMosaicTpuEnumAttribute::RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_CORE_TYPE,
    mnemonic = "core_type",
    description = "core type",
    variants = {
        TensorCore => ("tc", 0),
        ScalarSubcore => ("sc_scalar_subcore", 1),
        VectorSubcore => ("sc_vector_subcore", 2),
    },
);

mosaic_tpu_enum_attribute!(
    enum_name = PipelineMode,
    attribute_name = PipelineModeAttributeRef,
    context_method = mosaic_tpu_pipeline_mode_attribute,
    ffi_kind = MlirMosaicTpuEnumAttribute::RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_PIPELINE_MODE,
    mnemonic = "pipeline_mode",
    description = "pipeline mode",
    variants = {
        Synchronous => ("synchronous", 1),
        DoubleBuffered => ("double_buffered", 2),
    },
);

mosaic_tpu_enum_attribute!(
    enum_name = RevisitMode,
    attribute_name = RevisitModeAttributeRef,
    context_method = mosaic_tpu_revisit_mode_attribute,
    ffi_kind = MlirMosaicTpuEnumAttribute::RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_REVISIT_MODE,
    mnemonic = "revisit_mode",
    description = "revisit mode",
    variants = {
        Immediate => ("immediate", 0),
        Any => ("any", 1),
    },
);

mosaic_tpu_enum_attribute!(
    enum_name = DimensionSemantics,
    attribute_name = DimensionSemanticsAttributeRef,
    context_method = mosaic_tpu_dimension_semantics_attribute,
    ffi_kind = MlirMosaicTpuEnumAttribute::RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_DIMENSION_SEMANTICS,
    mnemonic = "dimension_semantics",
    description = "dimension semantics",
    variants = {
        Parallel => ("parallel", 0),
        Arbitrary => ("arbitrary", 1),
        CoreParallel => ("core_parallel", 2),
        SubcoreParallel => ("subcore_parallel", 3),
    },
);

mosaic_tpu_enum_attribute!(
    enum_name = ContractPrecision,
    attribute_name = ContractPrecisionAttributeRef,
    context_method = mosaic_tpu_contract_precision_attribute,
    ffi_kind = MlirMosaicTpuEnumAttribute::RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_CONTRACT_PRECISION,
    mnemonic = "contract_precision",
    description = "contract precision",
    variants = {
        BFloat16 => ("bf16", 0),
        Float32 => ("fp32", 1),
    },
);

mosaic_tpu_enum_attribute!(
    enum_name = PackFormat,
    attribute_name = PackFormatAttributeRef,
    context_method = mosaic_tpu_pack_format_attribute,
    ffi_kind = MlirMosaicTpuEnumAttribute::RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_PACK_FORMAT,
    mnemonic = "pack_format",
    description = "pack format",
    variants = {
        Compressed => ("compressed", 0),
        Interleaved => ("interleaved", 1),
    },
);

mosaic_tpu_enum_attribute!(
    enum_name = ReductionKind,
    attribute_name = ReductionKindAttributeRef,
    context_method = mosaic_tpu_reduction_kind_attribute,
    ffi_kind = MlirMosaicTpuEnumAttribute::RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_REDUCTION_KIND,
    mnemonic = "reduction_kind",
    description = "reduction kind",
    variants = {
        Sum => ("sum", 0),
        Max => ("max", 1),
        Min => ("min", 2),
        ArgMax => ("arg_max", 3),
        ArgMin => ("arg_min", 4),
        FindFirstSet => ("find_first_set", 5),
    },
);

mosaic_tpu_enum_attribute!(
    enum_name = RoundingMode,
    attribute_name = RoundingModeAttributeRef,
    context_method = mosaic_tpu_rounding_mode_attribute,
    ffi_kind = MlirMosaicTpuEnumAttribute::RYFT_MLIR_MOSAIC_TPU_ENUM_ATTRIBUTE_ROUNDING_MODE,
    mnemonic = "rounding_mode",
    description = "rounding mode",
    variants = {
        TowardsZero => ("towards_zero", 0),
        ToNearestEven => ("to_nearest_even", 1),
    },
);

/// Mosaic TPU dot-dimension-number [`Attribute`].
#[derive(Copy, Clone)]
pub struct DotDimensionNumbersAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl DotDimensionNumbersAttributeRef<'_, '_> {
    /// Returns the left-hand side contracting dimensions.
    pub fn lhs_contracting_dims(&self) -> DenseInteger64ArrayAttributeRef<'_, '_> {
        unsafe {
            DenseInteger64ArrayAttributeRef::from_c_api(
                mlirMosaicTpuDotDimensionNumbersAttrGetLhsContractingDims(self.handle),
                self.context,
            )
            .expect("invalid Mosaic TPU dot-dimension-number lhs contracting dimensions")
        }
    }

    /// Returns the right-hand side contracting dimensions.
    pub fn rhs_contracting_dims(&self) -> DenseInteger64ArrayAttributeRef<'_, '_> {
        unsafe {
            DenseInteger64ArrayAttributeRef::from_c_api(
                mlirMosaicTpuDotDimensionNumbersAttrGetRhsContractingDims(self.handle),
                self.context,
            )
            .expect("invalid Mosaic TPU dot-dimension-number rhs contracting dimensions")
        }
    }

    /// Returns the left-hand side non-contracting dimensions.
    pub fn lhs_non_contracting_dims(&self) -> DenseInteger64ArrayAttributeRef<'_, '_> {
        unsafe {
            DenseInteger64ArrayAttributeRef::from_c_api(
                mlirMosaicTpuDotDimensionNumbersAttrGetLhsNonContractingDims(self.handle),
                self.context,
            )
            .expect("invalid Mosaic TPU dot-dimension-number lhs non-contracting dimensions")
        }
    }

    /// Returns the right-hand side non-contracting dimensions.
    pub fn rhs_non_contracting_dims(&self) -> DenseInteger64ArrayAttributeRef<'_, '_> {
        unsafe {
            DenseInteger64ArrayAttributeRef::from_c_api(
                mlirMosaicTpuDotDimensionNumbersAttrGetRhsNonContractingDims(self.handle),
                self.context,
            )
            .expect("invalid Mosaic TPU dot-dimension-number rhs non-contracting dimensions")
        }
    }

    /// Returns the output dimension order.
    pub fn output_dim_order(&self) -> DenseInteger64ArrayAttributeRef<'_, '_> {
        unsafe {
            DenseInteger64ArrayAttributeRef::from_c_api(
                mlirMosaicTpuDotDimensionNumbersAttrGetOutputDimOrder(self.handle),
                self.context,
            )
            .expect("invalid Mosaic TPU dot-dimension-number output dimension order")
        }
    }

    /// Returns the left-hand side batch dimensions.
    pub fn lhs_batch_dims(&self) -> DenseInteger64ArrayAttributeRef<'_, '_> {
        unsafe {
            DenseInteger64ArrayAttributeRef::from_c_api(
                mlirMosaicTpuDotDimensionNumbersAttrGetLhsBatchDims(self.handle),
                self.context,
            )
            .expect("invalid Mosaic TPU dot-dimension-number lhs batch dimensions")
        }
    }

    /// Returns the right-hand side batch dimensions.
    pub fn rhs_batch_dims(&self) -> DenseInteger64ArrayAttributeRef<'_, '_> {
        unsafe {
            DenseInteger64ArrayAttributeRef::from_c_api(
                mlirMosaicTpuDotDimensionNumbersAttrGetRhsBatchDims(self.handle),
                self.context,
            )
            .expect("invalid Mosaic TPU dot-dimension-number rhs batch dimensions")
        }
    }
}

impl<'c, 't> Attribute<'c, 't> for DotDimensionNumbersAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirAttributeIsAMosaicTpuDotDimensionNumbersAttr(handle) } {
            Some(Self { handle, context })
        } else {
            None
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(DotDimensionNumbersAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Mosaic TPU element-window [`Attribute`].
#[derive(Copy, Clone)]
pub struct ElementWindowAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl ElementWindowAttributeRef<'_, '_> {
    /// Returns the low padding values.
    pub fn pad_low(&self) -> DenseInteger64ArrayAttributeRef<'_, '_> {
        unsafe {
            DenseInteger64ArrayAttributeRef::from_c_api(
                mlirMosaicTpuElementWindowAttrGetPadLow(self.handle),
                self.context,
            )
            .expect("invalid Mosaic TPU element-window low padding")
        }
    }

    /// Returns the high padding values.
    pub fn pad_high(&self) -> DenseInteger64ArrayAttributeRef<'_, '_> {
        unsafe {
            DenseInteger64ArrayAttributeRef::from_c_api(
                mlirMosaicTpuElementWindowAttrGetPadHigh(self.handle),
                self.context,
            )
            .expect("invalid Mosaic TPU element-window high padding")
        }
    }
}

impl<'c, 't> Attribute<'c, 't> for ElementWindowAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirAttributeIsAMosaicTpuElementWindowAttr(handle) } {
            Some(Self { handle, context })
        } else {
            None
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(ElementWindowAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Mosaic TPU vector-layout [`Attribute`].
#[derive(Copy, Clone)]
pub struct VectorLayoutAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> Attribute<'c, 't> for VectorLayoutAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirAttributeIsAMosaicTpuVectorLayoutAttr(handle) } {
            Some(Self { handle, context })
        } else {
            None
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(VectorLayoutAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Mosaic TPU tiled-layout [`Attribute`].
#[derive(Copy, Clone)]
pub struct TiledLayoutAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> Attribute<'c, 't> for TiledLayoutAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirAttributeIsAMosaicTpuTiledLayoutAttr(handle) } {
            Some(Self { handle, context })
        } else {
            None
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(TiledLayoutAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Mosaic TPU memory-space [`Attribute`].
#[derive(Copy, Clone)]
pub struct MemorySpaceAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

/// Mosaic TPU memory-space values.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MemorySpace {
    Any,
    Vmem,
    Smem,
    Hbm,
    Cmem,
    SemaphoreMem,
    VmemShared,
    Host,
}

impl MemorySpace {
    /// Returns the MLIR spelling of this memory-space value.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Any => "any",
            Self::Vmem => "vmem",
            Self::Smem => "smem",
            Self::Hbm => "hbm",
            Self::Cmem => "cmem",
            Self::SemaphoreMem => "semaphore_mem",
            Self::VmemShared => "vmem_shared",
            Self::Host => "host",
        }
    }

    /// Parses the MLIR spelling of a memory-space value.
    pub fn from_str(value: &str) -> Option<Self> {
        match value {
            "any" => Some(Self::Any),
            "vmem" => Some(Self::Vmem),
            "smem" => Some(Self::Smem),
            "hbm" => Some(Self::Hbm),
            "cmem" => Some(Self::Cmem),
            "semaphore_mem" => Some(Self::SemaphoreMem),
            "vmem_shared" => Some(Self::VmemShared),
            "host" => Some(Self::Host),
            _ => None,
        }
    }
}

impl MemorySpaceAttributeRef<'_, '_> {
    /// Returns the memory-space value stored in this attribute.
    pub fn value(&self) -> MemorySpace {
        let value = unsafe { StringRef::from_c_api(mlirMosaicTpuMemorySpaceAttrGetValue(self.handle)) };
        value
            .as_str()
            .ok()
            .and_then(MemorySpace::from_str)
            .expect("invalid Mosaic TPU memory-space attribute")
    }

    /// Returns the optional core type associated with this memory space.
    pub fn core_type(&self) -> Option<CoreType> {
        if !unsafe { mlirMosaicTpuMemorySpaceAttrHasCoreType(self.handle) } {
            return None;
        }
        let value = unsafe { StringRef::from_c_api(mlirMosaicTpuMemorySpaceAttrGetCoreType(self.handle)) };
        value.as_str().ok().and_then(CoreType::from_str)
    }
}

impl<'c, 't> Attribute<'c, 't> for MemorySpaceAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirAttributeIsAMosaicTpuMemorySpaceAttr(handle) } {
            Some(Self { handle, context })
        } else {
            None
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(MemorySpaceAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'t> Context<'t> {
    /// Creates a Mosaic TPU [`DotDimensionNumbersAttributeRef`] owned by this [`Context`].
    pub fn mosaic_tpu_dot_dimension_numbers_attribute<'c>(
        &'c self,
        lhs_contracting_dims: &[i64],
        rhs_contracting_dims: &[i64],
        lhs_non_contracting_dims: &[i64],
        rhs_non_contracting_dims: &[i64],
        output_dim_order: &[i64],
        lhs_batch_dims: &[i64],
        rhs_batch_dims: &[i64],
    ) -> DotDimensionNumbersAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::mosaic_tpu());
        unsafe {
            DotDimensionNumbersAttributeRef::from_c_api(
                mlirMosaicTpuDotDimensionNumbersAttrGet(
                    *self.handle.borrow(),
                    lhs_contracting_dims.as_ptr(),
                    lhs_contracting_dims.len().cast_signed(),
                    rhs_contracting_dims.as_ptr(),
                    rhs_contracting_dims.len().cast_signed(),
                    lhs_non_contracting_dims.as_ptr(),
                    lhs_non_contracting_dims.len().cast_signed(),
                    rhs_non_contracting_dims.as_ptr(),
                    rhs_non_contracting_dims.len().cast_signed(),
                    output_dim_order.as_ptr(),
                    output_dim_order.len().cast_signed(),
                    lhs_batch_dims.as_ptr(),
                    lhs_batch_dims.len().cast_signed(),
                    rhs_batch_dims.as_ptr(),
                    rhs_batch_dims.len().cast_signed(),
                ),
                self,
            )
            .expect("invalid arguments to `Context::mosaic_tpu_dot_dimension_numbers_attribute`")
        }
    }

    /// Creates a Mosaic TPU [`ElementWindowAttributeRef`] owned by this [`Context`].
    pub fn mosaic_tpu_element_window_attribute<'c>(
        &'c self,
        pad_low: &[i64],
        pad_high: &[i64],
    ) -> ElementWindowAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::mosaic_tpu());
        unsafe {
            ElementWindowAttributeRef::from_c_api(
                mlirMosaicTpuElementWindowAttrGet(
                    *self.handle.borrow(),
                    pad_low.as_ptr(),
                    pad_low.len().cast_signed(),
                    pad_high.as_ptr(),
                    pad_high.len().cast_signed(),
                ),
                self,
            )
            .expect("invalid arguments to `Context::mosaic_tpu_element_window_attribute`")
        }
    }

    /// Creates a Mosaic TPU [`MemorySpaceAttributeRef`] owned by this [`Context`].
    pub fn mosaic_tpu_memory_space_attribute<'c>(
        &'c self,
        value: MemorySpace,
        core_type: Option<CoreType>,
    ) -> MemorySpaceAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::mosaic_tpu());
        let value = StringRef::from(value.as_str());
        let core_type_string = core_type.map(|core_type| StringRef::from(core_type.as_str()));
        let core_type = core_type_string
            .as_ref()
            .map(|core_type| unsafe { core_type.to_c_api() })
            .unwrap_or(MlirStringRef { data: std::ptr::null(), length: 0 });
        unsafe {
            MemorySpaceAttributeRef::from_c_api(
                mlirMosaicTpuMemorySpaceAttrGet(*self.handle.borrow_mut(), value.to_c_api(), core_type),
                self,
            )
            .expect("invalid arguments to `Context::mosaic_tpu_memory_space_attribute`")
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    macro_rules! test_enum_attribute {
        (
            $test_name:ident,
            $context_method:ident,
            $enum_name:ident,
            $attribute_name:ident,
            $value:ident,
            $expected:literal $(,)*
        ) => {
            #[test]
            fn $test_name() {
                let context = Context::new();
                let attribute = context.$context_method($enum_name::$value);
                assert_eq!(&context, attribute.context());
                assert_eq!(attribute.value(), $enum_name::$value);
                test_attribute_display_and_debug(attribute, $expected);
                test_attribute_casting(attribute);
                assert_eq!(attribute.as_ref().cast::<$attribute_name>().unwrap().value(), $enum_name::$value);
            }
        };
    }

    test_enum_attribute!(
        test_core_type_attribute,
        mosaic_tpu_core_type_attribute,
        CoreType,
        CoreTypeAttributeRef,
        TensorCore,
        "#tpu.core_type<tc>"
    );

    test_enum_attribute!(
        test_pipeline_mode_attribute,
        mosaic_tpu_pipeline_mode_attribute,
        PipelineMode,
        PipelineModeAttributeRef,
        DoubleBuffered,
        "#tpu.pipeline_mode<double_buffered>"
    );

    test_enum_attribute!(
        test_revisit_mode_attribute,
        mosaic_tpu_revisit_mode_attribute,
        RevisitMode,
        RevisitModeAttributeRef,
        Any,
        "#tpu.revisit_mode<any>"
    );

    test_enum_attribute!(
        test_dimension_semantics_attribute,
        mosaic_tpu_dimension_semantics_attribute,
        DimensionSemantics,
        DimensionSemanticsAttributeRef,
        CoreParallel,
        "#tpu.dimension_semantics<core_parallel>"
    );

    test_enum_attribute!(
        test_contract_precision_attribute,
        mosaic_tpu_contract_precision_attribute,
        ContractPrecision,
        ContractPrecisionAttributeRef,
        Float32,
        "#tpu.contract_precision<fp32>"
    );

    test_enum_attribute!(
        test_pack_format_attribute,
        mosaic_tpu_pack_format_attribute,
        PackFormat,
        PackFormatAttributeRef,
        Interleaved,
        "#tpu.pack_format<interleaved>"
    );

    test_enum_attribute!(
        test_reduction_kind_attribute,
        mosaic_tpu_reduction_kind_attribute,
        ReductionKind,
        ReductionKindAttributeRef,
        ArgMax,
        "#tpu.reduction_kind<arg_max>"
    );

    test_enum_attribute!(
        test_rounding_mode_attribute,
        mosaic_tpu_rounding_mode_attribute,
        RoundingMode,
        RoundingModeAttributeRef,
        ToNearestEven,
        "#tpu.rounding_mode<to_nearest_even>"
    );

    #[test]
    fn test_dot_dimension_numbers_attribute() {
        let context = Context::new();
        let attribute = context.mosaic_tpu_dot_dimension_numbers_attribute(&[1], &[0], &[0], &[1], &[0, 1], &[2], &[3]);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.lhs_contracting_dims().values().collect::<Vec<_>>(), vec![1]);
        assert_eq!(attribute.rhs_contracting_dims().values().collect::<Vec<_>>(), vec![0]);
        assert_eq!(attribute.lhs_non_contracting_dims().values().collect::<Vec<_>>(), vec![0]);
        assert_eq!(attribute.rhs_non_contracting_dims().values().collect::<Vec<_>>(), vec![1]);
        assert_eq!(attribute.output_dim_order().values().collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(attribute.lhs_batch_dims().values().collect::<Vec<_>>(), vec![2]);
        assert_eq!(attribute.rhs_batch_dims().values().collect::<Vec<_>>(), vec![3]);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_dot_dimension_numbers_attribute_with_empty_dims() {
        let context = Context::new();
        let attribute = context.mosaic_tpu_dot_dimension_numbers_attribute(&[1], &[0], &[0], &[], &[0, 1], &[], &[]);
        assert_eq!(attribute.rhs_non_contracting_dims().values().collect::<Vec<_>>(), Vec::<i64>::new());
        assert_eq!(attribute.lhs_batch_dims().values().collect::<Vec<_>>(), Vec::<i64>::new());
        assert_eq!(attribute.rhs_batch_dims().values().collect::<Vec<_>>(), Vec::<i64>::new());
    }

    #[test]
    fn test_element_window_attribute() {
        let context = Context::new();
        let attribute = context.mosaic_tpu_element_window_attribute(&[0, 1], &[2, 3]);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.pad_low().values().collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(attribute.pad_high().values().collect::<Vec<_>>(), vec![2, 3]);
        test_attribute_display_and_debug(attribute, "#tpu.element_window<[0, 1], [2, 3]>");
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_memory_space_attribute() {
        let context = Context::new();
        let attribute = context.mosaic_tpu_memory_space_attribute(MemorySpace::Smem, None);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value(), MemorySpace::Smem);
        assert_eq!(attribute.core_type(), None);
        test_attribute_display_and_debug(attribute, "#tpu.memory_space<smem>");
        test_attribute_casting(attribute);

        let attribute = context.mosaic_tpu_memory_space_attribute(MemorySpace::Vmem, Some(CoreType::VectorSubcore));
        assert_eq!(attribute.value(), MemorySpace::Vmem);
        assert_eq!(attribute.core_type(), Some(CoreType::VectorSubcore));
        test_attribute_display_and_debug(attribute, "#tpu.memory_space<vmem, sc_vector_subcore>");
    }
}
