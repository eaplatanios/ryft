use ryft_xla_sys::bindings::MlirAttribute;

use crate::macros::mlir_subtype_trait_impls;
use crate::{ArrayAttributeRef, Attribute, AttributeRef, Context, DialectHandle, Error};

macro_rules! vector_enum_attribute {
    // Defines a vector enum and its context-owned MLIR attribute representation.
    (
        enum_name = $enum_name:ident,
        attribute_name = $attribute_name:ident,
        context_method = $context_method:ident,
        mnemonic = $mnemonic:literal,
        sentinel = $sentinel:literal,
        description = $description:literal,
        reference = $reference:literal,
        variants = { $($variant:ident => $spelling:literal => $meaning:literal),+ $(,)* },
    ) => {
        #[doc = concat!("A Vector dialect ", $description, ".")]
        #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum $enum_name {
            $(
                #[doc = $meaning]
                $variant,
            )+
        }

        impl $enum_name {
            /// Returns the canonical MLIR spelling of this value.
            pub fn as_str(&self) -> &'static str {
                match self {
                    $(Self::$variant => $spelling,)+
                }
            }
        }

        impl TryFrom<&str> for $enum_name {
            type Error = String;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                match value {
                    $($spelling => Ok(Self::$variant),)+
                    _ => Err(format!("`{value}` is not a valid Vector {}", $description)),
                }
            }
        }

        #[doc = concat!(
            "Vector dialect ",
            $description,
            " [`Attribute`], storing a [`",
            stringify!($enum_name),
            "`] value.",
        )]
        ///
        #[doc = concat!("For example, `", $sentinel, "` is the canonical MLIR representation of this attribute.")]
        ///
        #[doc = concat!(
            "Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Vector/#",
            $reference,
            ") for more information.",
        )]
        #[derive(Copy, Clone)]
        pub struct $attribute_name<'c, 't> {
            /// Handle that represents this [`Attribute`] in the MLIR C API.
            handle: MlirAttribute,

            /// [`Context`] that owns this [`Attribute`].
            context: &'c Context<'t>,
        }

        impl $attribute_name<'_, '_> {
            /// Returns the stored enum value.
            pub fn value(&self) -> Result<$enum_name, Error> {
                let source = self.to_string();
                let value = source.rsplit_once('<')
                    .and_then(|(_, value)| value.strip_suffix('>'))
                    .ok_or_else(|| Error::invalid_argument(concat!("invalid Vector ", $description, " attribute")))?;
                $enum_name::try_from(value).map_err(Error::invalid_argument)
            }
        }

        impl<'c, 't> Attribute<'c, 't> for $attribute_name<'c, 't> {
            unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Result<Self, Error> {
                if handle.ptr.is_null() {
                    return Err(Error::internal("expected non-null MLIR attribute handle"));
                }
                context.load_dialect(DialectHandle::vector()?)?;
                let expected = context.parse_attribute($sentinel)?;
                let attribute = unsafe { AttributeRef::from_c_api(handle, context) }?;
                if attribute.type_id() == expected.type_id() {
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
            #[doc = concat!(
            "Creates a [`",
            stringify!($attribute_name),
            "`] owned by this [`Context`]. Refer to its documentation for the attribute semantics.",
        )]
            pub fn $context_method<'c>(&'c self, value: $enum_name) -> Result<$attribute_name<'c, 't>, Error> {
                self.load_dialect(DialectHandle::vector()?)?;
                let source = format!(concat!("#vector.", $mnemonic, "<{}>"), value.as_str());
                let attribute = self.parse_attribute(&source)?;
                unsafe {
                    $attribute_name::from_c_api(attribute.to_c_api(), self).map_err(|_| {
                        Error::invalid_argument(concat!("invalid Vector ", $description, " attribute"))
                    })
                }
            }
        }
    };
}

vector_enum_attribute!(
    enum_name = CombiningKind,
    attribute_name = CombiningKindAttributeRef,
    context_method = vector_combining_kind_attribute,
    mnemonic = "kind",
    sentinel = "#vector.kind<add>",
    description = "combining kind",
    reference = "vectorcontract-vectorcontractionop",
    variants = {
        Add => "add" => "Addition.",
        Multiply => "mul" => "Multiplication.",
        MinimumUnsignedInteger => "minui" => "Minimum with unsigned integer comparison.",
        MinimumSignedInteger => "minsi" => "Minimum with signed integer comparison.",
        MinimumNumberFloat => "minnumf" => "Floating-point minimum with numeric preference over a single NaN.",
        MaximumUnsignedInteger => "maxui" => "Maximum with unsigned integer comparison.",
        MaximumSignedInteger => "maxsi" => "Maximum with signed integer comparison.",
        MaximumNumberFloat => "maxnumf" => "Floating-point maximum with numeric preference over a single NaN.",
        And => "and" => "Bitwise conjunction.",
        Or => "or" => "Bitwise disjunction.",
        Xor => "xor" => "Bitwise exclusive disjunction.",
        MinimumFloat => "minimumf" => "Floating-point minimum with NaN propagation.",
        MaximumFloat => "maximumf" => "Floating-point maximum with NaN propagation.",
    },
);

vector_enum_attribute!(
    enum_name = IteratorType,
    attribute_name = IteratorTypeAttributeRef,
    context_method = vector_iterator_type_attribute,
    mnemonic = "iterator_type",
    sentinel = "#vector.iterator_type<parallel>",
    description = "iterator type",
    reference = "vectorcontract-vectorcontractionop",
    variants = {
        Parallel => "parallel" => "Iteration over independent output coordinates.",
        Reduction => "reduction" => "Iteration that combines values into an output coordinate.",
    },
);

/// Vector dialect array of [`IteratorTypeAttributeRef`] values, preserving iteration-dimension order.
///
/// For example, `[#vector.iterator_type<parallel>, #vector.iterator_type<reduction>]` describes a parallel
/// dimension followed by a reduction dimension. Used by
/// [`ContractionOperation`](super::operations::ContractionOperation).
///
/// Refer to the
/// [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Vector/#vectorcontract-vectorcontractionop)
/// for more information.
#[derive(Copy, Clone)]
pub struct IteratorTypeArrayAttributeRef<'c, 't> {
    /// Built-in array attribute that stores the iterator-type attributes.
    attribute: ArrayAttributeRef<'c, 't>,
}

impl IteratorTypeArrayAttributeRef<'_, '_> {
    /// Returns the iterator types in source order.
    pub fn values(&self) -> Result<Vec<IteratorType>, Error> {
        self.attribute
            .elements()
            .map(|attribute| {
                attribute?
                    .cast::<IteratorTypeAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid Vector iterator type array attribute"))?
                    .value()
            })
            .collect()
    }
}

impl<'c, 't> Attribute<'c, 't> for IteratorTypeArrayAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Result<Self, Error> {
        let attribute = unsafe { ArrayAttributeRef::from_c_api(handle, context) }?;
        for element in attribute.elements() {
            if element?.cast::<IteratorTypeAttributeRef>().is_none() {
                return Err(Error::invalid_argument("expected Vector iterator type array attribute"));
            }
        }
        Ok(Self { attribute })
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        unsafe { self.attribute.to_c_api() }
    }

    fn context(&self) -> &'c Context<'t> {
        self.attribute.context()
    }
}

mlir_subtype_trait_impls!(IteratorTypeArrayAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'t> Context<'t> {
    /// Creates an [`IteratorTypeArrayAttributeRef`] owned by this [`Context`].
    /// The order of `values` determines iteration-dimension order; an empty slice creates an empty array.
    pub fn vector_iterator_type_array_attribute<'c>(
        &'c self,
        values: &[IteratorType],
    ) -> Result<IteratorTypeArrayAttributeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::vector()?)?;
        let attributes = values
            .iter()
            .map(|value| self.vector_iterator_type_attribute(*value))
            .collect::<Result<Vec<_>, _>>()?;
        unsafe { IteratorTypeArrayAttributeRef::from_c_api(self.array_attribute(&attributes).to_c_api(), self) }
    }
}

vector_enum_attribute!(
    enum_name = PrintPunctuation,
    attribute_name = PrintPunctuationAttributeRef,
    context_method = vector_print_punctuation_attribute,
    mnemonic = "punctuation",
    sentinel = "#vector.punctuation<newline>",
    description = "print punctuation",
    reference = "vectorprint-vectorprintop",
    variants = {
        None => "no_punctuation" => "No punctuation.",
        NewLine => "newline" => "A newline.",
        Comma => "comma" => "A comma.",
        Open => "open" => "An opening delimiter.",
        Close => "close" => "A closing delimiter.",
    },
);

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;

    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    #[test]
    fn test_combining_kind_as_str() {
        for (value, spelling) in [
            (CombiningKind::Add, "add"),
            (CombiningKind::Multiply, "mul"),
            (CombiningKind::MinimumUnsignedInteger, "minui"),
            (CombiningKind::MinimumSignedInteger, "minsi"),
            (CombiningKind::MinimumNumberFloat, "minnumf"),
            (CombiningKind::MaximumUnsignedInteger, "maxui"),
            (CombiningKind::MaximumSignedInteger, "maxsi"),
            (CombiningKind::MaximumNumberFloat, "maxnumf"),
            (CombiningKind::And, "and"),
            (CombiningKind::Or, "or"),
            (CombiningKind::Xor, "xor"),
            (CombiningKind::MinimumFloat, "minimumf"),
            (CombiningKind::MaximumFloat, "maximumf"),
        ] {
            assert_eq!(value.as_str(), spelling);
            let values = HashMap::from([(value, spelling)]);
            assert_eq!(values.get(&value), Some(&spelling));
            assert_eq!(CombiningKind::try_from(spelling), Ok(value));
        }
        assert_eq!(
            CombiningKind::try_from("invalid"),
            Err("`invalid` is not a valid Vector combining kind".to_string())
        );
    }

    #[test]
    fn test_combining_kind_attribute() {
        let context = Context::new();
        let attribute = context.vector_combining_kind_attribute(CombiningKind::Add).unwrap();
        assert_eq!(attribute.context(), &context);
        assert_eq!(attribute.value(), Ok(CombiningKind::Add));
    }

    #[test]
    fn test_combining_kind_attribute_equality() {
        let context = Context::new();
        let attribute = context.vector_combining_kind_attribute(CombiningKind::Add).unwrap();
        assert_eq!(attribute, context.vector_combining_kind_attribute(CombiningKind::Add).unwrap());
        assert_ne!(attribute, context.vector_combining_kind_attribute(CombiningKind::Multiply).unwrap());
        let other_context = Context::new();
        assert_ne!(attribute, other_context.vector_combining_kind_attribute(CombiningKind::Add).unwrap());
    }

    #[test]
    fn test_combining_kind_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.vector_combining_kind_attribute(CombiningKind::Add).unwrap();
        test_attribute_display_and_debug(attribute, "#vector.kind<add>");
    }

    #[test]
    fn test_combining_kind_attribute_casting() {
        let context = Context::new();
        let attribute = context.vector_combining_kind_attribute(CombiningKind::Add).unwrap();
        test_attribute_casting(attribute);
        assert_eq!(context.parse_attribute("#vector.kind<add>").unwrap(), attribute);
        assert_eq!(context.string_attribute("invalid").cast::<CombiningKindAttributeRef>(), None);
        assert!(matches!(
            unsafe {
                CombiningKindAttributeRef::from_c_api(MlirAttribute { ptr: std::ptr::null_mut() }, &context)
            },
            Err(Error::Internal { message, .. })
                if message == "expected non-null MLIR attribute handle",
        ));
    }

    #[test]
    fn test_iterator_type_as_str() {
        for (value, spelling) in [(IteratorType::Parallel, "parallel"), (IteratorType::Reduction, "reduction")] {
            assert_eq!(value.as_str(), spelling);
            let values = HashMap::from([(value, spelling)]);
            assert_eq!(values.get(&value), Some(&spelling));
            assert_eq!(IteratorType::try_from(spelling), Ok(value));
        }
        assert_eq!(IteratorType::try_from("invalid"), Err("`invalid` is not a valid Vector iterator type".to_string()));
    }

    #[test]
    fn test_iterator_type_attribute() {
        let context = Context::new();
        let attribute = context.vector_iterator_type_attribute(IteratorType::Parallel).unwrap();
        assert_eq!(attribute.context(), &context);
        assert_eq!(attribute.value(), Ok(IteratorType::Parallel));
    }

    #[test]
    fn test_iterator_type_attribute_equality() {
        let context = Context::new();
        let attribute = context.vector_iterator_type_attribute(IteratorType::Parallel).unwrap();
        assert_eq!(attribute, context.vector_iterator_type_attribute(IteratorType::Parallel).unwrap());
        assert_ne!(attribute, context.vector_iterator_type_attribute(IteratorType::Reduction).unwrap());
        let other_context = Context::new();
        assert_ne!(attribute, other_context.vector_iterator_type_attribute(IteratorType::Parallel).unwrap());
    }

    #[test]
    fn test_iterator_type_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.vector_iterator_type_attribute(IteratorType::Parallel).unwrap();
        test_attribute_display_and_debug(attribute, "#vector.iterator_type<parallel>");
    }

    #[test]
    fn test_iterator_type_attribute_casting() {
        let context = Context::new();
        let attribute = context.vector_iterator_type_attribute(IteratorType::Parallel).unwrap();
        test_attribute_casting(attribute);
        assert_eq!(context.parse_attribute("#vector.iterator_type<parallel>").unwrap(), attribute);
        assert_eq!(context.string_attribute("invalid").cast::<IteratorTypeAttributeRef>(), None);
        assert!(matches!(
            unsafe {
                IteratorTypeAttributeRef::from_c_api(MlirAttribute { ptr: std::ptr::null_mut() }, &context)
            },
            Err(Error::Internal { message, .. })
                if message == "expected non-null MLIR attribute handle",
        ));
    }

    #[test]
    fn test_iterator_type_array_attribute() {
        let context = Context::new();
        let attribute = context
            .vector_iterator_type_array_attribute(&[IteratorType::Parallel, IteratorType::Reduction])
            .unwrap();
        assert_eq!(attribute.context(), &context);
        assert_eq!(attribute.values(), Ok(vec![IteratorType::Parallel, IteratorType::Reduction]));
        assert_eq!(context.vector_iterator_type_array_attribute(&[]).unwrap().values(), Ok(vec![]));
    }

    #[test]
    fn test_iterator_type_array_attribute_equality() {
        let context = Context::new();
        let attribute = context
            .vector_iterator_type_array_attribute(&[IteratorType::Parallel, IteratorType::Reduction])
            .unwrap();
        assert_eq!(
            attribute,
            context
                .vector_iterator_type_array_attribute(&[IteratorType::Parallel, IteratorType::Reduction])
                .unwrap()
        );
        assert_ne!(
            attribute,
            context
                .vector_iterator_type_array_attribute(&[IteratorType::Reduction, IteratorType::Parallel])
                .unwrap()
        );
        let other_context = Context::new();
        assert_ne!(
            attribute,
            other_context
                .vector_iterator_type_array_attribute(&[IteratorType::Parallel, IteratorType::Reduction])
                .unwrap()
        );
    }

    #[test]
    fn test_iterator_type_array_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context
            .vector_iterator_type_array_attribute(&[IteratorType::Parallel, IteratorType::Reduction])
            .unwrap();
        test_attribute_display_and_debug(
            attribute,
            "[#vector.iterator_type<parallel>, #vector.iterator_type<reduction>]",
        );
    }

    #[test]
    fn test_iterator_type_array_attribute_casting() {
        let context = Context::new();
        let attribute = context
            .vector_iterator_type_array_attribute(&[IteratorType::Parallel, IteratorType::Reduction])
            .unwrap();
        test_attribute_casting(attribute);
        assert_eq!(
            context
                .parse_attribute("[#vector.iterator_type<parallel>, #vector.iterator_type<reduction>]")
                .unwrap(),
            attribute
        );
        assert_eq!(context.string_attribute("invalid").cast::<IteratorTypeArrayAttributeRef>(), None);
        assert_eq!(
            context
                .array_attribute(&[context.string_attribute("parallel")])
                .cast::<IteratorTypeArrayAttributeRef>(),
            None
        );
        assert!(matches!(
            unsafe {
                IteratorTypeArrayAttributeRef::from_c_api(
                    MlirAttribute { ptr: std::ptr::null_mut() },
                    &context,
                )
            },
            Err(Error::Internal { message, .. })
                if message == "expected non-null MLIR Attribute handle",
        ));
    }

    #[test]
    fn test_print_punctuation_as_str() {
        for (value, spelling) in [
            (PrintPunctuation::None, "no_punctuation"),
            (PrintPunctuation::NewLine, "newline"),
            (PrintPunctuation::Comma, "comma"),
            (PrintPunctuation::Open, "open"),
            (PrintPunctuation::Close, "close"),
        ] {
            assert_eq!(value.as_str(), spelling);
            let values = HashMap::from([(value, spelling)]);
            assert_eq!(values.get(&value), Some(&spelling));
            assert_eq!(PrintPunctuation::try_from(spelling), Ok(value));
        }
        assert_eq!(
            PrintPunctuation::try_from("invalid"),
            Err("`invalid` is not a valid Vector print punctuation".to_string())
        );
    }

    #[test]
    fn test_print_punctuation_attribute() {
        let context = Context::new();
        let attribute = context.vector_print_punctuation_attribute(PrintPunctuation::None).unwrap();
        assert_eq!(attribute.context(), &context);
        assert_eq!(attribute.value(), Ok(PrintPunctuation::None));
    }

    #[test]
    fn test_print_punctuation_attribute_equality() {
        let context = Context::new();
        let attribute = context.vector_print_punctuation_attribute(PrintPunctuation::None).unwrap();
        assert_eq!(attribute, context.vector_print_punctuation_attribute(PrintPunctuation::None).unwrap());
        assert_ne!(attribute, context.vector_print_punctuation_attribute(PrintPunctuation::NewLine).unwrap());
        let other_context = Context::new();
        assert_ne!(attribute, other_context.vector_print_punctuation_attribute(PrintPunctuation::None).unwrap());
    }

    #[test]
    fn test_print_punctuation_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.vector_print_punctuation_attribute(PrintPunctuation::None).unwrap();
        test_attribute_display_and_debug(attribute, "#vector.punctuation<no_punctuation>");
    }

    #[test]
    fn test_print_punctuation_attribute_casting() {
        let context = Context::new();
        let attribute = context.vector_print_punctuation_attribute(PrintPunctuation::None).unwrap();
        test_attribute_casting(attribute);
        assert_eq!(context.parse_attribute("#vector.punctuation<no_punctuation>").unwrap(), attribute);
        assert_eq!(context.string_attribute("invalid").cast::<PrintPunctuationAttributeRef>(), None);
        assert!(matches!(
            unsafe {
                PrintPunctuationAttributeRef::from_c_api(
                    MlirAttribute { ptr: std::ptr::null_mut() },
                    &context,
                )
            },
            Err(Error::Internal { message, .. })
                if message == "expected non-null MLIR attribute handle",
        ));
    }
}
