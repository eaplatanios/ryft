use ryft_xla_sys::bindings::MlirAttribute;

use crate::{
    Attribute, AttributeRef, Context, DialectHandle, FromWithContext, IntegerAttributeRef, IntegerTypeRef, Type,
    mlir_subtype_trait_impls,
};

macro_rules! linalg_enum_attribute {
    (
        enum_name = $enum_name:ident,
        attribute_name = $attribute_name:ident,
        context_method = $context_method:ident,
        mnemonic = $mnemonic:literal,
        sentinel = $sentinel:literal,
        description = $description:literal,
        variants = { $($variant:ident => $value:literal),+ $(,)* },
    ) => {
        /// MLIR Linalg enum value.
        #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum $enum_name {
            $($variant,)+
        }

        impl $enum_name {
            /// Returns the MLIR spelling for this enum value.
            pub fn as_str(&self) -> &'static str {
                match self {
                    $(Self::$variant => $value,)+
                }
            }
        }

        impl TryFrom<&str> for $enum_name {
            type Error = String;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                match value {
                    $($value => Ok(Self::$variant),)+
                    _ => Err(format!("'{value}' is not a valid {}", $description)),
                }
            }
        }

        #[doc = "MLIR Linalg `"]
        #[doc = $description]
        #[doc = "` enum [`Attribute`] wrapper."]
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
                let source = self.to_string();
                source
                    .rsplit_once('<')
                    .and_then(|(_, value)| value.strip_suffix('>'))
                    .and_then(|value| $enum_name::try_from(value).ok())
                    .expect(concat!("invalid Linalg `", $description, "` attribute"))
            }
        }

        impl<'c, 't> Attribute<'c, 't> for $attribute_name<'c, 't> {
            unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
                if handle.ptr.is_null() {
                    return None;
                }
                context.load_dialect(DialectHandle::linalg());
                let expected = context.parse_attribute($sentinel)?;
                let attribute = unsafe { AttributeRef::from_c_api(handle, context) }?;
                if attribute.type_id() == expected.type_id() {
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
            #[doc = "Creates a new Linalg `"]
            #[doc = $description]
            #[doc = "` enum [`Attribute`] owned by this [`Context`]."]
            pub fn $context_method<'c>(&'c self, value: $enum_name) -> $attribute_name<'c, 't> {
                self.load_dialect(DialectHandle::linalg());
                let source = format!(concat!("#linalg.", $mnemonic, "<{}>"), value.as_str());
                let attribute = self.parse_attribute(&source).unwrap();
                unsafe { $attribute_name::from_c_api(attribute.to_c_api(), self).unwrap() }
            }
        }
    };
}

linalg_enum_attribute!(
    enum_name = ElementwiseKind,
    attribute_name = ElementwiseKindAttributeRef,
    context_method = linalg_elementwise_kind_attribute,
    mnemonic = "elementwise_kind",
    sentinel = "#linalg.elementwise_kind<add>",
    description = "elementwise kind",
    variants = {
        Exp => "exp",
        Log => "log",
        Abs => "abs",
        Ceil => "ceil",
        Floor => "floor",
        NegateFloat => "negf",
        Reciprocal => "reciprocal",
        Round => "round",
        Sqrt => "sqrt",
        Rsqrt => "rsqrt",
        Square => "square",
        Tanh => "tanh",
        Erf => "erf",
        Add => "add",
        Subtract => "sub",
        Multiply => "mul",
        Divide => "div",
        DivideUnsigned => "div_unsigned",
        MaximumSigned => "max_signed",
        MinimumSigned => "min_signed",
        MaximumUnsigned => "max_unsigned",
        MinimumUnsigned => "min_unsigned",
        PowerFloat => "powf",
        Select => "select",
    },
);

linalg_enum_attribute!(
    enum_name = UnaryFn,
    attribute_name = UnaryFnAttributeRef,
    context_method = linalg_unary_fn_attribute,
    mnemonic = "unary_fn",
    sentinel = "#linalg.unary_fn<exp>",
    description = "unary function",
    variants = {
        Exp => "exp",
        Log => "log",
        Abs => "abs",
        Ceil => "ceil",
        Floor => "floor",
        NegateFloat => "negf",
        Reciprocal => "reciprocal",
        Round => "round",
        Sqrt => "sqrt",
        Rsqrt => "rsqrt",
        Square => "square",
        Tanh => "tanh",
        Erf => "erf",
    },
);

linalg_enum_attribute!(
    enum_name = BinaryFn,
    attribute_name = BinaryFnAttributeRef,
    context_method = linalg_binary_fn_attribute,
    mnemonic = "binary_fn",
    sentinel = "#linalg.binary_fn<add>",
    description = "binary function",
    variants = {
        Add => "add",
        Subtract => "sub",
        Multiply => "mul",
        Divide => "div",
        DivideUnsigned => "div_unsigned",
        MaximumSigned => "max_signed",
        MinimumSigned => "min_signed",
        MaximumUnsigned => "max_unsigned",
        MinimumUnsigned => "min_unsigned",
        PowerFloat => "powf",
    },
);

linalg_enum_attribute!(
    enum_name = TernaryFn,
    attribute_name = TernaryFnAttributeRef,
    context_method = linalg_ternary_fn_attribute,
    mnemonic = "ternary_fn",
    sentinel = "#linalg.ternary_fn<select>",
    description = "ternary function",
    variants = {
        Select => "select",
    },
);

linalg_enum_attribute!(
    enum_name = TypeFn,
    attribute_name = TypeFnAttributeRef,
    context_method = linalg_type_fn_attribute,
    mnemonic = "type_fn",
    sentinel = "#linalg.type_fn<cast_signed>",
    description = "type function",
    variants = {
        CastSigned => "cast_signed",
        CastUnsigned => "cast_unsigned",
    },
);

linalg_enum_attribute!(
    enum_name = IteratorType,
    attribute_name = IteratorTypeAttributeRef,
    context_method = linalg_iterator_type_attribute,
    mnemonic = "iterator_type",
    sentinel = "#linalg.iterator_type<parallel>",
    description = "iterator type",
    variants = {
        Parallel => "parallel",
        Reduction => "reduction",
    },
);

/// Winograd Conv2D filter and output tile-size pair.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WinogradConv2DFmr {
    /// Winograd F(2, 3) transform.
    F2R3,

    /// Winograd F(4, 3) transform.
    F4R3,

    /// Winograd F(2, 5) transform.
    F2R5,
}

impl WinogradConv2DFmr {
    /// Returns the MLIR spelling for this Winograd transform size.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::F2R3 => "F_2_3",
            Self::F4R3 => "F_4_3",
            Self::F2R5 => "F_2_5",
        }
    }

    /// Returns the generated MLIR enum discriminant for this Winograd transform size.
    pub fn discriminant(&self) -> i64 {
        match self {
            Self::F2R3 => 0,
            Self::F4R3 => 1,
            Self::F2R5 => 2,
        }
    }

    /// Returns the Winograd transform size represented by a generated MLIR enum discriminant.
    pub fn from_discriminant(discriminant: i64) -> Option<Self> {
        match discriminant {
            0 => Some(Self::F2R3),
            1 => Some(Self::F4R3),
            2 => Some(Self::F2R5),
            _ => None,
        }
    }
}

impl TryFrom<&str> for WinogradConv2DFmr {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "F_2_3" => Ok(Self::F2R3),
            "F_4_3" => Ok(Self::F4R3),
            "F_2_5" => Ok(Self::F2R5),
            _ => Err(format!("'{value}' is not a valid Winograd Conv2D FMR value")),
        }
    }
}

/// MLIR Linalg Winograd Conv2D FMR enum [`Attribute`] wrapper.
#[derive(Copy, Clone)]
pub struct WinogradConv2DFmrAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl WinogradConv2DFmrAttributeRef<'_, '_> {
    /// Returns the Winograd Conv2D FMR value stored in this attribute.
    pub fn value(&self) -> WinogradConv2DFmr {
        let attribute = self.as_ref().cast::<IntegerAttributeRef>().unwrap();
        WinogradConv2DFmr::from_discriminant(attribute.signless_value())
            .expect("invalid Linalg Winograd Conv2D FMR attribute")
    }
}

impl<'c, 't> Attribute<'c, 't> for WinogradConv2DFmrAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        let attribute = unsafe { IntegerAttributeRef::from_c_api(handle, context) }?;
        let r#type = attribute.r#type().cast::<IntegerTypeRef>()?;
        if r#type.is_signless()
            && r#type.bit_width() == 32
            && WinogradConv2DFmr::from_discriminant(attribute.signless_value()).is_some()
        {
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

mlir_subtype_trait_impls!(WinogradConv2DFmrAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'c, 't> FromWithContext<'c, 't, WinogradConv2DFmr> for WinogradConv2DFmrAttributeRef<'c, 't> {
    fn from_with_context(value: WinogradConv2DFmr, context: &'c Context<'t>) -> Self {
        context.linalg_winograd_conv_2d_fmr_attribute(value)
    }
}

impl<'t> Context<'t> {
    /// Creates a new Linalg Winograd Conv2D FMR enum [`Attribute`] owned by this [`Context`].
    pub fn linalg_winograd_conv_2d_fmr_attribute<'c>(
        &'c self,
        value: WinogradConv2DFmr,
    ) -> WinogradConv2DFmrAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::linalg());
        let attribute = self.integer_attribute(self.signless_integer_type(32), value.discriminant());
        unsafe { WinogradConv2DFmrAttributeRef::from_c_api(attribute.to_c_api(), self).unwrap() }
    }
}

/// Elementwise arity groups used by MLIR Linalg helper logic.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ElementwiseArityGroup {
    /// Unary elementwise operation.
    Unary,

    /// Binary elementwise operation.
    Binary,

    /// Ternary elementwise operation.
    Ternary,
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Attribute, Context};

    use super::*;

    #[test]
    fn test_linalg_elementwise_kind_attribute() {
        let context = Context::new();
        let attribute = context.linalg_elementwise_kind_attribute(ElementwiseKind::Add);
        assert_eq!(attribute.value(), ElementwiseKind::Add);
        assert_eq!(attribute.to_string(), "#linalg.elementwise_kind<add>");
        assert_eq!(format!("{attribute:?}"), "ElementwiseKindAttributeRef[#linalg.elementwise_kind<add>]");
        assert_eq!(attribute.as_ref().cast::<ElementwiseKindAttributeRef>().unwrap(), attribute);
    }

    #[test]
    fn test_linalg_unary_fn_attribute() {
        let context = Context::new();
        let attribute = context.linalg_unary_fn_attribute(UnaryFn::Exp);
        assert_eq!(attribute.value(), UnaryFn::Exp);
        assert_eq!(attribute.to_string(), "#linalg.unary_fn<exp>");
        assert_eq!(attribute.as_ref().cast::<UnaryFnAttributeRef>().unwrap(), attribute);
    }

    #[test]
    fn test_linalg_binary_fn_attribute() {
        let context = Context::new();
        let attribute = context.linalg_binary_fn_attribute(BinaryFn::Multiply);
        assert_eq!(attribute.value(), BinaryFn::Multiply);
        assert_eq!(attribute.to_string(), "#linalg.binary_fn<mul>");
        assert_eq!(attribute.as_ref().cast::<BinaryFnAttributeRef>().unwrap(), attribute);
    }

    #[test]
    fn test_linalg_ternary_fn_attribute() {
        let context = Context::new();
        let attribute = context.linalg_ternary_fn_attribute(TernaryFn::Select);
        assert_eq!(attribute.value(), TernaryFn::Select);
        assert_eq!(attribute.to_string(), "#linalg.ternary_fn<select>");
        assert_eq!(attribute.as_ref().cast::<TernaryFnAttributeRef>().unwrap(), attribute);
    }

    #[test]
    fn test_linalg_type_fn_attribute() {
        let context = Context::new();
        let attribute = context.linalg_type_fn_attribute(TypeFn::CastUnsigned);
        assert_eq!(attribute.value(), TypeFn::CastUnsigned);
        assert_eq!(attribute.to_string(), "#linalg.type_fn<cast_unsigned>");
        assert_eq!(attribute.as_ref().cast::<TypeFnAttributeRef>().unwrap(), attribute);
    }

    #[test]
    fn test_linalg_iterator_type_attribute() {
        let context = Context::new();
        let attribute = context.linalg_iterator_type_attribute(IteratorType::Reduction);
        assert_eq!(attribute.value(), IteratorType::Reduction);
        assert_eq!(attribute.to_string(), "#linalg.iterator_type<reduction>");
        assert_eq!(attribute.as_ref().cast::<IteratorTypeAttributeRef>().unwrap(), attribute);
    }

    #[test]
    fn test_linalg_winograd_conv_2d_fmr_attribute() {
        let context = Context::new();
        let attribute = context.linalg_winograd_conv_2d_fmr_attribute(WinogradConv2DFmr::F2R3);
        assert_eq!(attribute.value(), WinogradConv2DFmr::F2R3);
        assert_eq!(attribute.to_string(), "0 : i32");
        assert_eq!(attribute.as_ref().cast::<WinogradConv2DFmrAttributeRef>().unwrap(), attribute);
    }

    #[test]
    fn test_linalg_enum_attribute_equality() {
        let context = Context::new();
        let attribute_1 = context.linalg_elementwise_kind_attribute(ElementwiseKind::Add);
        let attribute_2 = context.linalg_elementwise_kind_attribute(ElementwiseKind::Add);
        let attribute_3 = context.linalg_elementwise_kind_attribute(ElementwiseKind::Multiply);
        assert_eq!(attribute_1, attribute_2);
        assert_ne!(attribute_1, attribute_3);

        let other_context = Context::new();
        let attribute_4 = other_context.linalg_elementwise_kind_attribute(ElementwiseKind::Add);
        assert_ne!(attribute_1, attribute_4);
    }
}
