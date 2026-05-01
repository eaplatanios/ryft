use ryft_xla_sys::bindings::MlirAttribute;
use ryft_xla_sys::mlir::dialects::nvgpu::{
    MlirNvgpuEnumAttribute, mlirAttributeIsANvgpuEnumAttr, mlirNvgpuEnumAttrGet, mlirNvgpuEnumAttrGetValue,
};

use crate::{Attribute, Context, DialectHandle, FromWithContext, mlir_subtype_trait_impls};

macro_rules! nvgpu_enum_attribute {
    (
        enum_name = $enum_name:ident,
        attribute_name = $attribute_name:ident,
        context_method = $context_method:ident,
        ffi_kind = $ffi_kind:path,
        description = $description:literal,
        variants = { $($variant:ident => ($value:literal, $spelling:literal)),+ $(,)* },
    ) => {
        /// Represents an NVGPU enum value.
        #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum $enum_name {
            $(
                #[doc = concat!("The `", $spelling, "` enum value.")]
                $variant,
            )+
        }

        impl $enum_name {
            /// Returns the integer representation used by MLIR for this enum value.
            pub fn value(&self) -> u32 {
                match self {
                    $(Self::$variant => $value,)+
                }
            }

            /// Creates this enum from the integer representation used by MLIR.
            pub fn from_value(value: u32) -> Option<Self> {
                match value {
                    $($value => Some(Self::$variant),)+
                    _ => None,
                }
            }

            /// Returns the textual MLIR spelling for this enum value.
            pub fn as_str(&self) -> &'static str {
                match self {
                    $(Self::$variant => $spelling,)+
                }
            }
        }

        #[doc = "MLIR [`Attribute`] that stores an NVGPU "]
        #[doc = $description]
        #[doc = "."]
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
                $enum_name::from_value(unsafe { mlirNvgpuEnumAttrGetValue(self.handle, $ffi_kind) })
                    .expect(concat!("invalid NVGPU ", $description, " attribute"))
            }
        }

        impl<'c, 't> Attribute<'c, 't> for $attribute_name<'c, 't> {
            unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
                if !handle.ptr.is_null() && unsafe { mlirAttributeIsANvgpuEnumAttr(handle, $ffi_kind) } {
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

        impl<'c, 't> FromWithContext<'c, 't, $enum_name> for $attribute_name<'c, 't> {
            fn from_with_context(value: $enum_name, context: &'c Context<'t>) -> Self {
                context.$context_method(value)
            }
        }

        impl<'t> Context<'t> {
            #[doc = "Creates a new NVGPU "]
            #[doc = $description]
            #[doc = " attribute owned by this [`Context`]."]
            pub fn $context_method<'c>(&'c self, value: $enum_name) -> $attribute_name<'c, 't> {
                self.load_dialect(DialectHandle::nvgpu());
                unsafe {
                    $attribute_name::from_c_api(
                        mlirNvgpuEnumAttrGet(*self.handle.borrow_mut(), $ffi_kind, value.value()),
                        self,
                    )
                    .expect(concat!("invalid arguments to `Context::", stringify!($context_method), "`"))
                }
            }
        }
    };
}

nvgpu_enum_attribute!(
    enum_name = TensorMapSwizzleKind,
    attribute_name = TensorMapSwizzleKindAttributeRef,
    context_method = nvgpu_tensor_map_swizzle_kind_attribute,
    ffi_kind = MlirNvgpuEnumAttribute::RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_TENSOR_MAP_SWIZZLE_KIND,
    description = "tensor map swizzle kind",
    variants = {
        None => (0, "none"),
        Swizzle32B => (1, "swizzle_32b"),
        Swizzle64B => (2, "swizzle_64b"),
        Swizzle128B => (3, "swizzle_128b"),
    },
);

nvgpu_enum_attribute!(
    enum_name = TensorMapL2PromoKind,
    attribute_name = TensorMapL2PromoKindAttributeRef,
    context_method = nvgpu_tensor_map_l2_promo_kind_attribute,
    ffi_kind = MlirNvgpuEnumAttribute::RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_TENSOR_MAP_L2_PROMO_KIND,
    description = "tensor map L2 promotion kind",
    variants = {
        None => (0, "none"),
        L2Promo64B => (1, "l2promo_64b"),
        L2Promo128B => (2, "l2promo_128b"),
        L2Promo256B => (3, "l2promo_256b"),
    },
);

nvgpu_enum_attribute!(
    enum_name = TensorMapOobKind,
    attribute_name = TensorMapOobKindAttributeRef,
    context_method = nvgpu_tensor_map_oob_kind_attribute,
    ffi_kind = MlirNvgpuEnumAttribute::RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_TENSOR_MAP_OOB_KIND,
    description = "tensor map out-of-bounds fill kind",
    variants = {
        Zero => (0, "zero"),
        Nan => (1, "nan"),
    },
);

nvgpu_enum_attribute!(
    enum_name = TensorMapInterleaveKind,
    attribute_name = TensorMapInterleaveKindAttributeRef,
    context_method = nvgpu_tensor_map_interleave_kind_attribute,
    ffi_kind = MlirNvgpuEnumAttribute::RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_TENSOR_MAP_INTERLEAVE_KIND,
    description = "tensor map interleave kind",
    variants = {
        None => (0, "none"),
        Interleave16B => (1, "interleave_16b"),
        Interleave32B => (2, "interleave_32b"),
    },
);

nvgpu_enum_attribute!(
    enum_name = RcpRoundingMode,
    attribute_name = RcpRoundingModeAttributeRef,
    context_method = nvgpu_rcp_rounding_mode_attribute,
    ffi_kind = MlirNvgpuEnumAttribute::RYFT_MLIR_NVGPU_ENUM_ATTRIBUTE_RCP_ROUNDING_MODE,
    description = "reciprocal rounding mode",
    variants = {
        Approx => (0, "approx"),
        NearestEven => (1, "rn"),
        TowardZero => (2, "rz"),
        Downward => (3, "rm"),
        Upward => (4, "rp"),
    },
);

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    macro_rules! nvgpu_enum_attribute_tests {
        ($test_name:ident, $constructor:ident, $enum_name:ident, $attribute_name:ident, $first:ident, $second:ident) => {
            paste::paste! {
                #[test]
                fn [<test_ $test_name _attribute>]() {
                    let context = Context::new();
                    let attribute = context.$constructor($enum_name::$first);
                    assert_eq!(&context, attribute.context());
                    assert_eq!(attribute.value(), $enum_name::$first);
                    assert_eq!($enum_name::from_value(attribute.value().value()), Some($enum_name::$first));
                }

                #[test]
                fn [<test_ $test_name _attribute_equality>]() {
                    let context = Context::new();
                    let attribute_1 = context.$constructor($enum_name::$first);
                    let attribute_2 = context.$constructor($enum_name::$first);
                    assert_eq!(attribute_1, attribute_2);

                    let attribute_2 = context.$constructor($enum_name::$second);
                    assert_ne!(attribute_1, attribute_2);

                    let context = Context::new();
                    let attribute_2 = context.$constructor($enum_name::$first);
                    assert_ne!(attribute_1, attribute_2);
                }

                #[test]
                fn [<test_ $test_name _attribute_display_and_debug>]() {
                    let context = Context::new();
                    let attribute = context.$constructor($enum_name::$first);
                    let expected = format!("#nvgpu<{} {}>", stringify!($test_name), $enum_name::$first.as_str());
                    test_attribute_display_and_debug(attribute, Box::leak(expected.into_boxed_str()));
                }

                #[test]
                fn [<test_ $test_name _attribute_parsing>]() {
                    let context = Context::new();
                    context.load_dialect(DialectHandle::nvgpu());
                    let attribute = context.$constructor($enum_name::$first);
                    let source = format!("#nvgpu<{} {}>", stringify!($test_name), $enum_name::$first.as_str());
                    assert_eq!(context.parse_attribute(&source).unwrap().cast::<$attribute_name>().unwrap(), attribute);
                }

                #[test]
                fn [<test_ $test_name _attribute_casting>]() {
                    let context = Context::new();
                    let attribute = context.$constructor($enum_name::$first);
                    test_attribute_casting(attribute);
                }
            }
        };
    }

    nvgpu_enum_attribute_tests!(
        swizzle,
        nvgpu_tensor_map_swizzle_kind_attribute,
        TensorMapSwizzleKind,
        TensorMapSwizzleKindAttributeRef,
        None,
        Swizzle32B
    );

    nvgpu_enum_attribute_tests!(
        l2promo,
        nvgpu_tensor_map_l2_promo_kind_attribute,
        TensorMapL2PromoKind,
        TensorMapL2PromoKindAttributeRef,
        None,
        L2Promo64B
    );

    nvgpu_enum_attribute_tests!(
        oob,
        nvgpu_tensor_map_oob_kind_attribute,
        TensorMapOobKind,
        TensorMapOobKindAttributeRef,
        Zero,
        Nan
    );

    nvgpu_enum_attribute_tests!(
        interleave,
        nvgpu_tensor_map_interleave_kind_attribute,
        TensorMapInterleaveKind,
        TensorMapInterleaveKindAttributeRef,
        None,
        Interleave16B
    );

    nvgpu_enum_attribute_tests!(
        rcp_rounding_mode,
        nvgpu_rcp_rounding_mode_attribute,
        RcpRoundingMode,
        RcpRoundingModeAttributeRef,
        Approx,
        NearestEven
    );
}
