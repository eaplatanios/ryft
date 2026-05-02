use ryft_xla_sys::bindings::MlirAttribute;

use crate::{Attribute, AttributeRef, Context, DialectHandle, IntegerAttributeRef, mlir_subtype_trait_impls};

macro_rules! transform_enum_attribute {
    (
        enum_name = $enum_name:ident,
        attribute_name = $attribute_name:ident,
        context_method = $context_method:ident,
        doc = $doc:literal,
        variants = { $($variant:ident => ($value:literal, $string:literal)),+ $(,)* } $(,)?
    ) => {
        #[doc = $doc]
        #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum $enum_name {
            $($variant,)+
        }

        impl $enum_name {
            /// Returns the integer value used by the MLIR enum attribute.
            pub fn value(&self) -> u32 {
                match self {
                    $(Self::$variant => $value,)+
                }
            }

            /// Returns the textual spelling used in MLIR assembly.
            pub fn as_str(&self) -> &'static str {
                match self {
                    $(Self::$variant => $string,)+
                }
            }

            /// Returns the enum value corresponding to the provided MLIR integer value.
            pub fn from_value(value: u32) -> Option<Self> {
                match value {
                    $($value => Some(Self::$variant),)+
                    _ => None,
                }
            }
        }

        #[doc = "Transform dialect [`Attribute`] that stores a [`"]
        #[doc = stringify!($enum_name)]
        #[doc = "`]."]
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
                let attribute = unsafe { IntegerAttributeRef::from_c_api(self.handle, self.context).unwrap() };
                $enum_name::from_value(attribute.signless_value() as u32).expect("invalid Transform enum attribute value")
            }
        }

        impl<'c, 't> Attribute<'c, 't> for $attribute_name<'c, 't> {
            unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
                let attribute = unsafe { IntegerAttributeRef::from_c_api(handle, context)? };
                let value = attribute.signless_value();
                if value >= 0 && attribute.value_bit_width() == 32 && $enum_name::from_value(value as u32).is_some() {
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
            #[doc = "Creates a new [`"]
            #[doc = stringify!($attribute_name)]
            #[doc = "`] owned by this [`Context`]."]
            pub fn $context_method<'c>(&'c self, value: $enum_name) -> $attribute_name<'c, 't> {
                self.load_dialect(DialectHandle::transform());
                let attribute = self.integer_attribute(self.signless_integer_type(32), value.value() as i64);
                unsafe { $attribute_name::from_c_api(attribute.to_c_api(), self).unwrap() }
            }
        }
    };
}

transform_enum_attribute!(
    enum_name = FailurePropagationMode,
    attribute_name = FailurePropagationModeAttributeRef,
    context_method = transform_failure_propagation_mode_attribute,
    doc = "Policy used by Transform dialect container operations when nested operations produce silenceable failures.",
    variants = {
        Propagate => (1, "propagate"),
        Suppress => (2, "suppress"),
    },
);

transform_enum_attribute!(
    enum_name = MatchCmpIPredicate,
    attribute_name = MatchCmpIPredicateAttributeRef,
    context_method = transform_match_cmp_i_predicate_attribute,
    doc = "Signed integer comparison predicate used by `transform.match.param.cmpi`.",
    variants = {
        Equal => (0, "eq"),
        NotEqual => (1, "ne"),
        LessThan => (2, "lt"),
        LessThanOrEqual => (3, "le"),
        GreaterThan => (4, "gt"),
        GreaterThanOrEqual => (5, "ge"),
    },
);

/// Transform dialect [`Attribute`] that refers to a specific parameter operand by index.
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Transform/#paramoperandattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct ParamOperandAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> ParamOperandAttributeRef<'c, 't> {
    /// Returns the integer attribute that stores the referenced parameter operand index.
    pub fn index(&self) -> IntegerAttributeRef<'c, 't> {
        let source = self.to_string();
        let index = source
            .strip_prefix("#transform.param_operand<index = ")
            .and_then(|source| source.strip_suffix(">"))
            .expect("invalid `#transform.param_operand` assembly");
        self.context
            .parse_attribute(index)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .expect("invalid `index` parameter in `#transform.param_operand`")
    }
}

impl<'c, 't> Attribute<'c, 't> for ParamOperandAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        let attribute = unsafe { AttributeRef::from_c_api(handle, context)? };
        let is_param_operand = attribute.dialect().namespace().ok() == Some("transform")
            && attribute.to_string().starts_with("#transform.param_operand<");
        if is_param_operand { Some(Self { handle, context }) } else { None }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(ParamOperandAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'t> Context<'t> {
    /// Creates a new [`ParamOperandAttributeRef`] owned by this [`Context`].
    pub fn transform_param_operand_attribute<'c>(
        &'c self,
        index: IntegerAttributeRef<'c, 't>,
    ) -> ParamOperandAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::transform());
        let source = format!("#transform.param_operand<index = {index}>");
        self.parse_attribute(&source)
            .and_then(|attribute| attribute.cast())
            .expect("invalid arguments to `Context::transform_param_operand_attribute`")
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::Attribute;
    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    #[test]
    fn test_failure_propagation_mode_attribute() {
        let context = Context::new();
        let attribute = context.transform_failure_propagation_mode_attribute(FailurePropagationMode::Propagate);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value(), FailurePropagationMode::Propagate);
        assert_eq!(FailurePropagationMode::Propagate.value(), 1);
        assert_eq!(FailurePropagationMode::Propagate.as_str(), "propagate");
        assert_eq!(FailurePropagationMode::from_value(1), Some(FailurePropagationMode::Propagate));
    }

    #[test]
    fn test_failure_propagation_mode_attribute_equality() {
        let context = Context::new();
        let attribute_1 = context.transform_failure_propagation_mode_attribute(FailurePropagationMode::Propagate);
        let attribute_2 = context.transform_failure_propagation_mode_attribute(FailurePropagationMode::Propagate);
        assert_eq!(attribute_1, attribute_2);

        let attribute_2 = context.transform_failure_propagation_mode_attribute(FailurePropagationMode::Suppress);
        assert_ne!(attribute_1, attribute_2);

        let context = Context::new();
        let attribute_2 = context.transform_failure_propagation_mode_attribute(FailurePropagationMode::Propagate);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_failure_propagation_mode_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.transform_failure_propagation_mode_attribute(FailurePropagationMode::Propagate);
        test_attribute_display_and_debug(attribute, "1 : i32");
    }

    #[test]
    fn test_failure_propagation_mode_attribute_casting() {
        let context = Context::new();
        let attribute = context.transform_failure_propagation_mode_attribute(FailurePropagationMode::Propagate);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_match_cmp_i_predicate_attribute() {
        let context = Context::new();
        let attribute = context.transform_match_cmp_i_predicate_attribute(MatchCmpIPredicate::LessThanOrEqual);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value(), MatchCmpIPredicate::LessThanOrEqual);
        assert_eq!(MatchCmpIPredicate::LessThanOrEqual.value(), 3);
        assert_eq!(MatchCmpIPredicate::LessThanOrEqual.as_str(), "le");
        assert_eq!(MatchCmpIPredicate::from_value(3), Some(MatchCmpIPredicate::LessThanOrEqual));
    }

    #[test]
    fn test_match_cmp_i_predicate_attribute_equality() {
        let context = Context::new();
        let attribute_1 = context.transform_match_cmp_i_predicate_attribute(MatchCmpIPredicate::LessThanOrEqual);
        let attribute_2 = context.transform_match_cmp_i_predicate_attribute(MatchCmpIPredicate::LessThanOrEqual);
        assert_eq!(attribute_1, attribute_2);

        let attribute_2 = context.transform_match_cmp_i_predicate_attribute(MatchCmpIPredicate::GreaterThan);
        assert_ne!(attribute_1, attribute_2);

        let context = Context::new();
        let attribute_2 = context.transform_match_cmp_i_predicate_attribute(MatchCmpIPredicate::LessThanOrEqual);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_match_cmp_i_predicate_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.transform_match_cmp_i_predicate_attribute(MatchCmpIPredicate::LessThanOrEqual);
        test_attribute_display_and_debug(attribute, "3 : i32");
    }

    #[test]
    fn test_match_cmp_i_predicate_attribute_casting() {
        let context = Context::new();
        let attribute = context.transform_match_cmp_i_predicate_attribute(MatchCmpIPredicate::LessThanOrEqual);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_param_operand_attribute() {
        let context = Context::new();
        let index = context.integer_attribute(context.signless_integer_type(64), 7);
        let attribute = context.transform_param_operand_attribute(index);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.index(), index);
    }

    #[test]
    fn test_param_operand_attribute_equality() {
        let context = Context::new();
        let index = context.integer_attribute(context.signless_integer_type(64), 7);
        let attribute_1 = context.transform_param_operand_attribute(index);
        let attribute_2 = context.transform_param_operand_attribute(index);
        assert_eq!(attribute_1, attribute_2);

        let index = context.integer_attribute(context.signless_integer_type(64), 8);
        let attribute_2 = context.transform_param_operand_attribute(index);
        assert_ne!(attribute_1, attribute_2);

        let context = Context::new();
        let index = context.integer_attribute(context.signless_integer_type(64), 7);
        let attribute_2 = context.transform_param_operand_attribute(index);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_param_operand_attribute_display_and_debug() {
        let context = Context::new();
        let index = context.integer_attribute(context.signless_integer_type(64), 7);
        let attribute = context.transform_param_operand_attribute(index);
        test_attribute_display_and_debug(attribute, "#transform.param_operand<index = 7 : i64>");
    }

    #[test]
    fn test_param_operand_attribute_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::transform());
        let index = context.integer_attribute(context.signless_integer_type(64), 7);
        let attribute = context.transform_param_operand_attribute(index);
        assert_eq!(
            context
                .parse_attribute("#transform.param_operand<index = 7 : i64>")
                .unwrap()
                .cast::<ParamOperandAttributeRef>()
                .unwrap(),
            attribute
        );
    }

    #[test]
    fn test_param_operand_attribute_casting() {
        let context = Context::new();
        let index = context.integer_attribute(context.signless_integer_type(64), 7);
        let attribute = context.transform_param_operand_attribute(index);
        test_attribute_casting(attribute);
    }
}
