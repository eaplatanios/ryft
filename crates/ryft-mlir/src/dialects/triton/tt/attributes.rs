use ryft_xla_sys::bindings::MlirAttribute;
use ryft_xla_sys::mlir::dialects::triton::tt::{
    MlirTritonTtEnumAttribute, mlirAttributeIsATritonTtEnumAttr, mlirTritonTtEnumAttrGet, mlirTritonTtEnumAttrGetValue,
};

use crate::{Attribute, Context, DialectHandle, Error, StringRef, mlir_subtype_trait_impls};

macro_rules! tt_enum_attribute {
    (
        enum_name = $enum_name:ident,
        attribute_name = $attribute_name:ident,
        context_method = $context_method:ident,
        ffi_kind = $ffi_kind:path,
        mnemonic = $mnemonic:literal,
        description = $description:literal,
        variants = { $($variant:ident => $value:literal),+ $(,)* } $(,)*
    ) => {
        #[doc = "Triton `tt` "]
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

            /// Parses the MLIR spelling of this enum value.
            pub fn from_str(value: &str) -> Option<Self> {
                match value {
                    $($value => Some(Self::$variant),)+
                    _ => None,
                }
            }
        }

        #[doc = "Triton `tt` "]
        #[doc = $description]
        #[doc = " [`Attribute`]."]
        #[derive(Copy, Clone)]
        pub struct $attribute_name<'c, 't> {
            /// Handle that represents this [`Attribute`] in the MLIR C API.
            handle: MlirAttribute,

            /// [`Context`] that owns this [`Attribute`].
            context: &'c Context<'t>,
        }

        impl<'c, 't> $attribute_name<'c, 't> {
            /// Returns the enum value stored in this attribute.
            pub fn value(&self) -> Result<$enum_name, Error> {
                let value = unsafe { StringRef::from_c_api(mlirTritonTtEnumAttrGetValue(self.handle, $ffi_kind)) };
                value
                    .as_str()
                    .ok()
                    .and_then($enum_name::from_str)
                    .ok_or_else(|| Error::invalid_argument(concat!("invalid Triton `tt` `", $mnemonic, "` attribute")))
            }
        }

        impl<'c, 't> Attribute<'c, 't> for $attribute_name<'c, 't> {
            unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Result<Self, Error> {
                if handle.ptr.is_null() {
                    return Err(Error::internal("expected non-null MLIR attribute handle"));
                }
                if unsafe { mlirAttributeIsATritonTtEnumAttr(handle, $ffi_kind) } {
                    Ok(Self { handle, context })
                } else {
                    Err(Error::invalid_argument("expected MLIR attribute handle"))
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
            #[doc = "Creates a Triton `tt` "]
            #[doc = $description]
            #[doc = " attribute owned by this [`Context`]."]
            pub fn $context_method<'c>(&'c self, value: $enum_name) -> Result<$attribute_name<'c, 't>, Error> {
                self.load_dialect(DialectHandle::triton_tt()?)?;
                let value = StringRef::from(value.as_str());
                unsafe {
                    $attribute_name::from_c_api(
                        mlirTritonTtEnumAttrGet(*self.handle.borrow_mut(), $ffi_kind, value.to_c_api()),
                        self,
                    )
                    .map_err(|_| Error::invalid_argument(concat!("invalid Triton `tt` `", $mnemonic, "` attribute")))
                }
            }
        }
    };
}

tt_enum_attribute!(
    enum_name = CacheModifier,
    attribute_name = CacheModifierAttributeRef,
    context_method = triton_tt_cache_modifier_attribute,
    ffi_kind = MlirTritonTtEnumAttribute::MLIR_TRITON_TT_ENUM_ATTRIBUTE_CACHE_MODIFIER,
    mnemonic = "cache_modifier",
    description = "cache modifier",
    variants = {
        None => "none",
        CacheAll => "ca",
        CacheGlobal => "cg",
        WriteBack => "wb",
        CacheStreaming => "cs",
        WriteThrough => "wt",
        CacheVolatile => "cv",
    },
);

tt_enum_attribute!(
    enum_name = MemSemantic,
    attribute_name = MemSemanticAttributeRef,
    context_method = triton_tt_mem_semantic_attribute,
    ffi_kind = MlirTritonTtEnumAttribute::MLIR_TRITON_TT_ENUM_ATTRIBUTE_MEM_SEMANTIC,
    mnemonic = "mem_semantic",
    description = "memory semantic",
    variants = {
        Relaxed => "relaxed",
        Acquire => "acquire",
        Release => "release",
        AcquireRelease => "acq_rel",
    },
);

tt_enum_attribute!(
    enum_name = EvictionPolicy,
    attribute_name = EvictionPolicyAttributeRef,
    context_method = triton_tt_eviction_policy_attribute,
    ffi_kind = MlirTritonTtEnumAttribute::MLIR_TRITON_TT_ENUM_ATTRIBUTE_EVICTION_POLICY,
    mnemonic = "eviction_policy",
    description = "eviction policy",
    variants = {
        Normal => "evict_normal",
        EvictFirst => "evict_first",
        EvictLast => "evict_last",
    },
);

tt_enum_attribute!(
    enum_name = PaddingOption,
    attribute_name = PaddingOptionAttributeRef,
    context_method = triton_tt_padding_option_attribute,
    ffi_kind = MlirTritonTtEnumAttribute::MLIR_TRITON_TT_ENUM_ATTRIBUTE_PADDING_OPTION,
    mnemonic = "padding_option",
    description = "padding option",
    variants = {
        Zero => "zero",
        Nan => "nan",
    },
);

tt_enum_attribute!(
    enum_name = RmwOp,
    attribute_name = RmwOpAttributeRef,
    context_method = triton_tt_rmw_op_attribute,
    ffi_kind = MlirTritonTtEnumAttribute::MLIR_TRITON_TT_ENUM_ATTRIBUTE_RMW_OP,
    mnemonic = "rmw_op",
    description = "atomic read-modify-write operation",
    variants = {
        And => "and",
        Or => "or",
        Xor => "xor",
        Add => "add",
        FloatAdd => "fadd",
        Max => "max",
        Min => "min",
        UnsignedMax => "umax",
        UnsignedMin => "umin",
        Exchange => "exch",
    },
);

tt_enum_attribute!(
    enum_name = DescriptorReduceKind,
    attribute_name = DescriptorReduceKindAttributeRef,
    context_method = triton_tt_descriptor_reduce_kind_attribute,
    ffi_kind = MlirTritonTtEnumAttribute::MLIR_TRITON_TT_ENUM_ATTRIBUTE_DESCRIPTOR_REDUCE_KIND,
    mnemonic = "descriptor_reduce_kind",
    description = "descriptor reduce kind",
    variants = {
        Add => "add",
        Min => "min",
        Max => "max",
        Increment => "inc",
        Decrement => "dec",
        And => "and",
        Or => "or",
        Xor => "xor",
    },
);

tt_enum_attribute!(
    enum_name = MemSyncScope,
    attribute_name = MemSyncScopeAttributeRef,
    context_method = triton_tt_mem_sync_scope_attribute,
    ffi_kind = MlirTritonTtEnumAttribute::MLIR_TRITON_TT_ENUM_ATTRIBUTE_MEM_SYNC_SCOPE,
    mnemonic = "mem_sync_scope",
    description = "memory synchronization scope",
    variants = {
        Gpu => "gpu",
        Cta => "cta",
        System => "sys",
    },
);

tt_enum_attribute!(
    enum_name = ProgramIdDim,
    attribute_name = ProgramIdDimAttributeRef,
    context_method = triton_tt_program_id_dim_attribute,
    ffi_kind = MlirTritonTtEnumAttribute::MLIR_TRITON_TT_ENUM_ATTRIBUTE_PROGRAM_ID_DIM,
    mnemonic = "program_id_dim",
    description = "program identifier dimension",
    variants = {
        X => "x",
        Y => "y",
        Z => "z",
    },
);

tt_enum_attribute!(
    enum_name = RoundingMode,
    attribute_name = RoundingModeAttributeRef,
    context_method = triton_tt_rounding_mode_attribute,
    ffi_kind = MlirTritonTtEnumAttribute::MLIR_TRITON_TT_ENUM_ATTRIBUTE_ROUNDING_MODE,
    mnemonic = "rounding_mode",
    description = "rounding mode",
    variants = {
        TowardsZero => "rtz",
        ToNearestEven => "rtne",
    },
);

tt_enum_attribute!(
    enum_name = PropagateNan,
    attribute_name = PropagateNanAttributeRef,
    context_method = triton_tt_propagate_nan_attribute,
    ffi_kind = MlirTritonTtEnumAttribute::MLIR_TRITON_TT_ENUM_ATTRIBUTE_PROPAGATE_NAN,
    mnemonic = "propagate_nan",
    description = "NaN propagation mode",
    variants = {
        None => "none",
        All => "all",
    },
);

tt_enum_attribute!(
    enum_name = InputPrecision,
    attribute_name = InputPrecisionAttributeRef,
    context_method = triton_tt_input_precision_attribute,
    ffi_kind = MlirTritonTtEnumAttribute::MLIR_TRITON_TT_ENUM_ATTRIBUTE_INPUT_PRECISION,
    mnemonic = "input_precision",
    description = "dot input precision",
    variants = {
        Tf32 => "tf32",
        Tf32x3 => "tf32x3",
        Ieee => "ieee",
        Bf16x3 => "bf16x3",
        Bf16x6 => "bf16x6",
    },
);

tt_enum_attribute!(
    enum_name = ScaleDotElemType,
    attribute_name = ScaleDotElemTypeAttributeRef,
    context_method = triton_tt_scale_dot_elem_type_attribute,
    ffi_kind = MlirTritonTtEnumAttribute::MLIR_TRITON_TT_ENUM_ATTRIBUTE_SCALE_DOT_ELEM_TYPE,
    mnemonic = "scale_dot_elem_type",
    description = "scaled dot element type",
    variants = {
        E4M3 => "e4m3",
        E5M2 => "e5m2",
        E2M3 => "e2m3",
        E3M2 => "e3m2",
        E2M1 => "e2m1",
        Bfloat16 => "bf16",
        Float16 => "fp16",
    },
);

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    macro_rules! tt_enum_attribute_tests {
        (
            $test_name:ident,
            $test_equality_name:ident,
            $test_display_name:ident,
            $test_casting_name:ident,
            $context_method:ident,
            $enum_name:ident,
            $attribute_name:ident,
            $value_1:ident,
            $value_2:ident,
            $expected:literal $(,)*
        ) => {
            #[test]
            fn $test_name() {
                let context = Context::new();
                let attribute = context.$context_method($enum_name::$value_1).unwrap();
                assert_eq!(&context, attribute.context());
                assert_eq!(attribute.value().unwrap(), $enum_name::$value_1);
                assert_eq!(attribute.value().unwrap().as_str(), $enum_name::$value_1.as_str());
                assert_eq!($enum_name::from_str($enum_name::$value_1.as_str()), Some($enum_name::$value_1));
                assert_eq!($enum_name::from_str("invalid"), None);
            }

            #[test]
            fn $test_equality_name() {
                let context = Context::new();

                // Same attributes from the same context must be equal because they are "uniqued".
                let attribute_1 = context.$context_method($enum_name::$value_1).unwrap();
                let attribute_2 = context.$context_method($enum_name::$value_1).unwrap();
                assert_eq!(attribute_1, attribute_2);

                // Different attributes from the same context must not be equal.
                let attribute_2 = context.$context_method($enum_name::$value_2).unwrap();
                assert_ne!(attribute_1, attribute_2);

                // Same attributes from different contexts must not be equal.
                let context = Context::new();
                let attribute_2 = context.$context_method($enum_name::$value_1).unwrap();
                assert_ne!(attribute_1, attribute_2);
            }

            #[test]
            fn $test_display_name() {
                let context = Context::new();
                let attribute = context.$context_method($enum_name::$value_1).unwrap();
                test_attribute_display_and_debug(attribute, $expected);
            }

            #[test]
            fn $test_casting_name() {
                let context = Context::new();
                let attribute = context.$context_method($enum_name::$value_1).unwrap();
                test_attribute_casting(attribute);
            }
        };
    }

    tt_enum_attribute_tests!(
        test_cache_modifier_attribute,
        test_cache_modifier_attribute_equality,
        test_cache_modifier_attribute_display_and_debug,
        test_cache_modifier_attribute_casting,
        triton_tt_cache_modifier_attribute,
        CacheModifier,
        CacheModifierAttributeRef,
        CacheAll,
        CacheGlobal,
        "2 : i32",
    );

    tt_enum_attribute_tests!(
        test_mem_semantic_attribute,
        test_mem_semantic_attribute_equality,
        test_mem_semantic_attribute_display_and_debug,
        test_mem_semantic_attribute_casting,
        triton_tt_mem_semantic_attribute,
        MemSemantic,
        MemSemanticAttributeRef,
        Acquire,
        Release,
        "2 : i32",
    );

    tt_enum_attribute_tests!(
        test_eviction_policy_attribute,
        test_eviction_policy_attribute_equality,
        test_eviction_policy_attribute_display_and_debug,
        test_eviction_policy_attribute_casting,
        triton_tt_eviction_policy_attribute,
        EvictionPolicy,
        EvictionPolicyAttributeRef,
        EvictFirst,
        EvictLast,
        "2 : i32",
    );

    tt_enum_attribute_tests!(
        test_padding_option_attribute,
        test_padding_option_attribute_equality,
        test_padding_option_attribute_display_and_debug,
        test_padding_option_attribute_casting,
        triton_tt_padding_option_attribute,
        PaddingOption,
        PaddingOptionAttributeRef,
        Zero,
        Nan,
        "1 : i32",
    );

    tt_enum_attribute_tests!(
        test_rmw_op_attribute,
        test_rmw_op_attribute_equality,
        test_rmw_op_attribute_display_and_debug,
        test_rmw_op_attribute_casting,
        triton_tt_rmw_op_attribute,
        RmwOp,
        RmwOpAttributeRef,
        Add,
        Xor,
        "4 : i32",
    );

    tt_enum_attribute_tests!(
        test_descriptor_reduce_kind_attribute,
        test_descriptor_reduce_kind_attribute_equality,
        test_descriptor_reduce_kind_attribute_display_and_debug,
        test_descriptor_reduce_kind_attribute_casting,
        triton_tt_descriptor_reduce_kind_attribute,
        DescriptorReduceKind,
        DescriptorReduceKindAttributeRef,
        Add,
        Xor,
        "1 : i32",
    );

    tt_enum_attribute_tests!(
        test_mem_sync_scope_attribute,
        test_mem_sync_scope_attribute_equality,
        test_mem_sync_scope_attribute_display_and_debug,
        test_mem_sync_scope_attribute_casting,
        triton_tt_mem_sync_scope_attribute,
        MemSyncScope,
        MemSyncScopeAttributeRef,
        Gpu,
        Cta,
        "1 : i32",
    );

    tt_enum_attribute_tests!(
        test_program_id_dim_attribute,
        test_program_id_dim_attribute_equality,
        test_program_id_dim_attribute_display_and_debug,
        test_program_id_dim_attribute_casting,
        triton_tt_program_id_dim_attribute,
        ProgramIdDim,
        ProgramIdDimAttributeRef,
        X,
        Y,
        "0 : i32",
    );

    tt_enum_attribute_tests!(
        test_rounding_mode_attribute,
        test_rounding_mode_attribute_equality,
        test_rounding_mode_attribute_display_and_debug,
        test_rounding_mode_attribute_casting,
        triton_tt_rounding_mode_attribute,
        RoundingMode,
        RoundingModeAttributeRef,
        TowardsZero,
        ToNearestEven,
        "0 : i32",
    );

    tt_enum_attribute_tests!(
        test_propagate_nan_attribute,
        test_propagate_nan_attribute_equality,
        test_propagate_nan_attribute_display_and_debug,
        test_propagate_nan_attribute_casting,
        triton_tt_propagate_nan_attribute,
        PropagateNan,
        PropagateNanAttributeRef,
        None,
        All,
        "0 : i32",
    );

    tt_enum_attribute_tests!(
        test_input_precision_attribute,
        test_input_precision_attribute_equality,
        test_input_precision_attribute_display_and_debug,
        test_input_precision_attribute_casting,
        triton_tt_input_precision_attribute,
        InputPrecision,
        InputPrecisionAttributeRef,
        Tf32,
        Ieee,
        "0 : i32",
    );

    tt_enum_attribute_tests!(
        test_scale_dot_elem_type_attribute,
        test_scale_dot_elem_type_attribute_equality,
        test_scale_dot_elem_type_attribute_display_and_debug,
        test_scale_dot_elem_type_attribute_casting,
        triton_tt_scale_dot_elem_type_attribute,
        ScaleDotElemType,
        ScaleDotElemTypeAttributeRef,
        E4M3,
        E5M2,
        "0 : i32",
    );
}
