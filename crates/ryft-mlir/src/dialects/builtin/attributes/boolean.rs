use ryft_xla_sys::bindings::{MlirAttribute, mlirBoolAttrGet, mlirBoolAttrGetValue};

use crate::{Attribute, Context, FromWithContext, mlir_subtype_trait_impls};

/// Built-in MLIR [`Attribute`] that stores a boolean value.
#[derive(Copy, Clone)]
pub struct BooleanAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> BooleanAttributeRef<'c, 't> {
    /// Returns the value stored in this [`BooleanAttributeRef`].
    pub fn value(&self) -> bool {
        unsafe { mlirBoolAttrGetValue(self.handle) }
    }
}

mlir_subtype_trait_impls!(BooleanAttributeRef<'c, 't> as Attribute, mlir_type = Attribute, mlir_subtype = Bool);

impl<'c, 't> FromWithContext<'c, 't, bool> for BooleanAttributeRef<'c, 't> {
    fn from_with_context(value: bool, context: &'c Context<'t>) -> Self {
        context.boolean_attribute(value)
    }
}

impl<'t> Context<'t> {
    /// Creates a new [`BooleanAttributeRef`] owned by this [`Context`].
    pub fn boolean_attribute<'c>(&'c self, value: bool) -> BooleanAttributeRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe {
            BooleanAttributeRef::from_c_api(mlirBoolAttrGet(*self.handle.borrow(), if value { 1 } else { 0 }), self)
                .unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::IntoWithContext;
    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    #[test]
    fn test_boolean_attribute() {
        let context = Context::new();

        let attribute = context.boolean_attribute(true);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value(), true);

        let attribute: BooleanAttributeRef<'_, '_> = false.into_with_context(&context);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value(), false);
    }

    #[test]
    fn test_boolean_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.boolean_attribute(true);
        let attribute_2 = context.boolean_attribute(true);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.boolean_attribute(false);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.boolean_attribute(true);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_boolean_attribute_display_and_debug() {
        let context = Context::new();

        let attribute = context.boolean_attribute(true);
        test_attribute_display_and_debug(attribute, "true");

        let attribute = context.boolean_attribute(false);
        test_attribute_display_and_debug(attribute, "false");
    }

    #[test]
    fn test_boolean_attribute_parsing() {
        let context = Context::new();
        assert_eq!(context.parse_attribute("true").unwrap(), context.boolean_attribute(true));
        assert_eq!(context.parse_attribute("false").unwrap(), context.boolean_attribute(false));
    }

    #[test]
    fn test_boolean_attribute_casting() {
        let context = Context::new();
        let attribute = context.boolean_attribute(true);
        test_attribute_casting(attribute);
    }
}
