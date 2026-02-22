use ryft_xla_sys::bindings::{
    MlirAttribute, mlirArrayAttrGet, mlirArrayAttrGetElement, mlirArrayAttrGetNumElements, mlirArrayAttrGetTypeID,
};

use crate::{Attribute, AttributeRef, Context, FromWithContext, TypeId, mlir_subtype_trait_impls};

/// Built-in MLIR [`Attribute`] that stores an array of other [`Attribute`] values.
///
/// # Examples
///
/// The following are examples of [`ArrayAttributeRef`]s represented using their
/// [`Display`](std::fmt::Display) rendering:
///
/// ```text
/// []
/// [10, i32]
/// [affine_map<(d0, d1, d2) -> (d0, d1)>, i32, "string attribute"]
/// ```
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#arrayattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct ArrayAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> ArrayAttributeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`ArrayAttributeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirArrayAttrGetTypeID()).unwrap() }
    }

    /// Returns the length of this [`ArrayAttributeRef`] (i.e., the number of element [`Attribute`]s it contains).
    pub fn len(&self) -> usize {
        unsafe { mlirArrayAttrGetNumElements(self.handle).cast_unsigned() }
    }

    /// Returns the element [`AttributeRef`]s of this [`ArrayAttributeRef`].
    pub fn elements(&self) -> impl Iterator<Item = AttributeRef<'c, 't>> {
        (0..self.len()).map(|index| self.element(index))
    }

    /// Returns the element [`AttributeRef`] of this [`ArrayAttributeRef`] at the specified index.
    ///
    /// Note that this function will panic if the provided index is out of bounds.
    pub fn element(&self, index: usize) -> AttributeRef<'c, 't> {
        if index >= self.len() {
            panic!("index is out of bounds");
        }
        unsafe {
            AttributeRef::from_c_api(mlirArrayAttrGetElement(self.handle, index.cast_signed()), self.context).unwrap()
        }
    }
}

mlir_subtype_trait_impls!(ArrayAttributeRef<'c, 't> as Attribute, mlir_type = Attribute, mlir_subtype = Array);

impl<'c, 't, A: Attribute<'c, 't>> FromWithContext<'c, 't, &[A]> for ArrayAttributeRef<'c, 't> {
    fn from_with_context(value: &[A], context: &'c Context<'t>) -> Self {
        context.array_attribute(value)
    }
}

impl<'t> Context<'t> {
    /// Creates a new [`ArrayAttributeRef`] owned by this [`Context`].
    pub fn array_attribute<'c, A: Attribute<'c, 't>>(&'c self, elements: &[A]) -> ArrayAttributeRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe {
            let elements = elements.iter().map(|element| element.to_c_api()).collect::<Vec<_>>();
            ArrayAttributeRef::from_c_api(
                mlirArrayAttrGet(*self.handle.borrow(), elements.len().cast_signed(), elements.as_ptr() as *const _),
                self,
            )
            .unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    #[test]
    fn test_array_attribute_type_id() {
        let context = Context::new();
        let array_attribute_id = ArrayAttributeRef::type_id();
        let array_attribute_1 = context.array_attribute(&[context.unit_attribute()]);
        let array_attribute_2 = context.array_attribute(&[context.unit_attribute()]);
        assert_eq!(array_attribute_1.type_id(), array_attribute_2.type_id());
        assert_eq!(array_attribute_id, array_attribute_1.type_id());
    }

    #[test]
    fn test_array_attribute() {
        let context = Context::new();

        // Test empty array.
        let attribute = context.array_attribute(&[] as &[AttributeRef]);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.len(), 0);

        // Test array with integer attributes.
        let int_type = context.signless_integer_type(64);
        let elements = vec![
            context.integer_attribute(int_type, 1),
            context.integer_attribute(int_type, 2),
            context.integer_attribute(int_type, 3),
        ];
        let attribute = context.array_attribute(&elements);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.len(), 3);
        assert_eq!(attribute.element(0), elements[0]);
        assert_eq!(attribute.element(1), elements[1]);
        assert_eq!(attribute.element(2), elements[2]);

        // Test array with mixed attribute types.
        let mixed_elements = vec![
            context.integer_attribute(int_type, 42).as_ref(),
            context.string_attribute("test").as_ref(),
            context.boolean_attribute(true).as_ref(),
        ];
        let attribute = context.array_attribute(&mixed_elements);
        assert_eq!(attribute.len(), 3);
    }

    #[test]
    fn test_array_attribute_equality() {
        let context = Context::new();
        let int_type = context.signless_integer_type(64);

        // Same attributes from the same context must be equal because they are "uniqued".
        let elements = vec![context.integer_attribute(int_type, 1), context.integer_attribute(int_type, 2)];
        let attribute_1 = context.array_attribute(&elements);
        let attribute_2 = context.array_attribute(&elements);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let elements_2 = vec![context.integer_attribute(int_type, 3), context.integer_attribute(int_type, 4)];
        let attribute_2 = context.array_attribute(&elements_2);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let other_int_type = context.signless_integer_type(64);
        let other_elements =
            vec![context.integer_attribute(other_int_type, 1), context.integer_attribute(other_int_type, 2)];
        let attribute_2 = context.array_attribute(&other_elements);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_array_attribute_display_and_debug() {
        let context = Context::new();

        // Test empty array.
        let attribute = context.array_attribute(&[] as &[AttributeRef]);
        test_attribute_display_and_debug(attribute, "[]");

        // Test array with elements.
        let int_type = context.signless_integer_type(32);
        let elements = vec![context.integer_attribute(int_type, 1), context.integer_attribute(int_type, 2)];
        let attribute = context.array_attribute(&elements);
        test_attribute_display_and_debug(attribute, "[1 : i32, 2 : i32]");
    }

    #[test]
    fn test_array_attribute_parsing() {
        let context = Context::new();

        // Test parsing empty array.
        let attribute = context.array_attribute(&[] as &[AttributeRef]);
        let parsed = context.parse_attribute("[]").unwrap();
        assert_eq!(parsed, attribute);

        // Test parsing array with elements.
        let int_type = context.signless_integer_type(32);
        let elements = vec![context.integer_attribute(int_type, 1), context.integer_attribute(int_type, 2)];
        let attribute = context.array_attribute(&elements);
        let parsed = context.parse_attribute("[1 : i32, 2 : i32]").unwrap();
        assert_eq!(parsed, attribute);
    }

    #[test]
    fn test_array_attribute_casting() {
        let context = Context::new();
        let int_type = context.signless_integer_type(64);
        let elements = vec![context.integer_attribute(int_type, 1), context.integer_attribute(int_type, 2)];
        let attribute = context.array_attribute(&elements);
        test_attribute_casting(attribute);
    }
}
