use std::collections::HashMap;

use ryft_xla_sys::bindings::{
    MlirAttribute, mlirDictionaryAttrGet, mlirDictionaryAttrGetElement, mlirDictionaryAttrGetElementByName,
    mlirDictionaryAttrGetNumElements, mlirDictionaryAttrGetTypeID,
};

use crate::{
    Attribute, AttributeRef, Context, Error, NamedAttributeRef, StringRef, TryFromWithContext, TypeId,
    mlir_subtype_trait_impls,
};

/// Built-in MLIR [`Attribute`] that represents a sorted collection of [`NamedAttributeRef`] values. The elements are
/// sorted by name, and each name must be unique within the collection.
///
/// # Examples
///
/// The following are examples of [`DictionaryAttributeRef`]s represented using their
/// [`Display`](std::fmt::Display) rendering:
///
/// ```text
/// {}
/// {attr_name = "string attribute"}
/// {int_attr = 10, "string attr name" = "string attribute"}
/// ```
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#arrayattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct DictionaryAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> DictionaryAttributeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`DictionaryAttributeRef`].
    pub fn type_id() -> Result<TypeId<'static>, Error> {
        unsafe { TypeId::from_c_api(mlirDictionaryAttrGetTypeID()) }
    }

    /// Returns the length of this [`DictionaryAttributeRef`] (i.e., the number of [`NamedAttributeRef`]s it contains).
    pub fn len(&self) -> usize {
        unsafe { mlirDictionaryAttrGetNumElements(self.handle).cast_unsigned() }
    }

    /// Returns `true` if this [`DictionaryAttributeRef`] is empty (i.e., it contains `0` [`NamedAttributeRef`]s).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the [`NamedAttributeRef`]s stored in this [`DictionaryAttributeRef`].
    pub fn elements(&self) -> impl Iterator<Item = NamedAttributeRef<'c, 't>> {
        (0..self.len()).map(|index| unsafe {
            NamedAttributeRef::from_c_api(mlirDictionaryAttrGetElement(self.handle, index.cast_signed()), self.context)
        })
    }

    /// Returns the [`NamedAttributeRef`] of this [`DictionaryAttributeRef`] at the specified index, or an error if the
    /// index is out of bounds.
    pub fn element(&self, index: usize) -> Result<NamedAttributeRef<'c, 't>, Error> {
        let len = self.len();
        if index >= len {
            return Err(Error::invalid_argument(format!(
                "dictionary attribute element index {index} is out of bounds for length {len}"
            )));
        }
        Ok(unsafe {
            NamedAttributeRef::from_c_api(mlirDictionaryAttrGetElement(self.handle, index.cast_signed()), self.context)
        })
    }

    /// Returns the [`AttributeRef`] with the specified name stored in this [`DictionaryAttributeRef`],
    /// and [`None`] if the provided name does not exist in this [`DictionaryAttributeRef`].
    pub fn element_by_name<S: AsRef<str>>(&self, name: S) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        let handle =
            unsafe { mlirDictionaryAttrGetElementByName(self.handle, StringRef::from(name.as_ref()).to_c_api()) };
        if handle.ptr.is_null() {
            Ok(None)
        } else {
            unsafe { AttributeRef::from_c_api(handle, self.context).map(Some) }
        }
    }
}

mlir_subtype_trait_impls!(
    DictionaryAttributeRef<'c, 't> as Attribute,
    mlir_type = Attribute,
    mlir_subtype = Dictionary,
);

impl<'c, 't> TryFromWithContext<'c, 't, &[NamedAttributeRef<'c, 't>]> for DictionaryAttributeRef<'c, 't> {
    fn try_from_with_context(value: &[NamedAttributeRef<'c, 't>], context: &'c Context<'t>) -> Result<Self, Error> {
        Ok(context.dictionary_attribute(value))
    }
}

impl<'c, 't, 's, A: Attribute<'c, 't>> TryFromWithContext<'c, 't, &HashMap<StringRef<'s>, A>>
    for DictionaryAttributeRef<'c, 't>
{
    fn try_from_with_context(value: &HashMap<StringRef<'s>, A>, context: &'c Context<'t>) -> Result<Self, Error> {
        Ok(context.dictionary_attribute(
            &value
                .iter()
                .map(|(name, attribute)| context.named_attribute(context.identifier(*name), *attribute))
                .collect::<Vec<_>>(),
        ))
    }
}

impl<'c, 't, A: Attribute<'c, 't>> TryFrom<DictionaryAttributeRef<'c, 't>> for HashMap<StringRef<'c>, A> {
    type Error = Error;

    fn try_from(value: DictionaryAttributeRef<'c, 't>) -> Result<Self, Self::Error> {
        value
            .elements()
            .map(|element| {
                let attribute = element.attribute()?;
                let attribute = attribute.cast().ok_or_else(|| {
                    Error::invalid_argument(format!("invalid dictionary attribute element `{}`", element.name()))
                })?;
                Ok((element.name().as_ref(), attribute))
            })
            .collect()
    }
}

impl<'t> Context<'t> {
    /// Creates a new [`DictionaryAttributeRef`] owned by this [`Context`].
    pub fn dictionary_attribute<'c>(
        &'c self,
        elements: &[NamedAttributeRef<'c, 't>],
    ) -> DictionaryAttributeRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe {
            let elements = elements.iter().map(|element| element.to_c_api()).collect::<Vec<_>>();
            let handle = mlirDictionaryAttrGet(
                *self.handle.borrow(),
                elements.len().cast_signed(),
                elements.as_ptr() as *const _,
            );
            DictionaryAttributeRef { handle, context: self }
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::TryIntoWithContext;
    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    #[test]
    fn test_dictionary_attribute_type_id() {
        let context = Context::new();
        let dictionary_attribute_id = DictionaryAttributeRef::type_id().unwrap();
        let dictionary_attribute_1 = context.dictionary_attribute(&[]);
        let dictionary_attribute_2 = context.dictionary_attribute(&[]);
        assert_eq!(dictionary_attribute_1.type_id().unwrap(), dictionary_attribute_2.type_id().unwrap());
        assert_eq!(dictionary_attribute_id, dictionary_attribute_1.type_id().unwrap());
    }

    #[test]
    fn test_dictionary_attribute() {
        let context = Context::new();

        // Test empty dictionary.
        let attribute = context.dictionary_attribute(&[]);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.len(), 0);

        // Test dictionary with named attributes.
        let i32_attribute = context.integer_attribute(context.signless_integer_type(32), 42);
        let string_attribute = context.string_attribute("test");
        let named_attributes = vec![
            context.named_attribute(context.identifier("int_key"), i32_attribute),
            context.named_attribute(context.identifier("str_key"), string_attribute),
        ];
        let attribute: DictionaryAttributeRef<'_, '_> =
            named_attributes.as_slice().try_into_with_context(&context).unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.len(), 2);
        assert_eq!(attribute.element(0).unwrap().attribute().unwrap(), i32_attribute);
        assert_eq!(attribute.element(1).unwrap().attribute().unwrap(), string_attribute);
        assert!(attribute.element(2).is_err());

        // Test [`DictionaryAttributeRef::element_by_name`].
        let element = attribute.element_by_name("int_key").unwrap().unwrap();
        assert_eq!(element.cast::<crate::IntegerAttributeRef>().unwrap(), i32_attribute);

        let element = attribute.element_by_name("str_key").unwrap().unwrap();
        assert_eq!(element.cast::<crate::StringAttributeRef>().unwrap(), string_attribute);
        assert_eq!(attribute.element_by_name("nonexistent").unwrap(), None);
    }

    #[test]
    fn test_dictionary_attribute_equality() {
        let context = Context::new();
        let named_attributes = vec![context.named_attribute(
            context.identifier("key"),
            context.integer_attribute(context.signless_integer_type(32), 42),
        )];

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.dictionary_attribute(&named_attributes);
        let attribute_2 = context.dictionary_attribute(&named_attributes);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.dictionary_attribute(&[context.named_attribute(
            context.identifier("key"),
            context.integer_attribute(context.signless_integer_type(32), 100),
        )]);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.dictionary_attribute(&[context.named_attribute(
            context.identifier("key"),
            context.integer_attribute(context.signless_integer_type(32), 42),
        )]);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_dictionary_attribute_display_and_debug() {
        let context = Context::new();

        // Empty dictionary.
        let attribute = context.dictionary_attribute(&[]);
        test_attribute_display_and_debug(attribute, "{}");

        // Dictionary with one element.
        let attribute = context.dictionary_attribute(&[context.named_attribute(
            context.identifier("key"),
            context.integer_attribute(context.signless_integer_type(32), 42),
        )]);
        test_attribute_display_and_debug(attribute, "{key = 42 : i32}");
    }

    #[test]
    fn test_dictionary_attribute_parsing() {
        let context = Context::new();

        // Test parsing empty dictionary.
        let attribute = context.dictionary_attribute(&[]);
        let parsed = context.parse_attribute("{}").unwrap();
        assert_eq!(parsed, attribute);

        // Test parsing dictionary with one element.
        let attribute = context.dictionary_attribute(&[context.named_attribute(
            context.identifier("key"),
            context.integer_attribute(context.signless_integer_type(32), 42),
        )]);
        let parsed = context.parse_attribute("{key = 42 : i32}").unwrap();
        assert_eq!(parsed, attribute);
    }

    #[test]
    fn test_dictionary_attribute_casting() {
        let context = Context::new();
        let attribute = context.dictionary_attribute(&[context.named_attribute(
            context.identifier("key"),
            context.integer_attribute(context.signless_integer_type(32), 42),
        )]);
        test_attribute_casting(attribute);
    }
}
