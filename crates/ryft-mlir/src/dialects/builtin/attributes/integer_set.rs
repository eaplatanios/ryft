use ryft_xla_sys::bindings::{
    MlirAttribute, mlirIntegerSetAttrGet, mlirIntegerSetAttrGetTypeID, mlirIntegerSetAttrGetValue,
};

use crate::{Attribute, Context, FromWithContext, IntegerSet, TypeId, mlir_subtype_trait_impls};

/// Built-in MLIR [`Attribute`] that stores an [`IntegerSet`].
///
/// # Examples
///
/// The following is an example of an [`IntegerSetAttributeRef`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```text
/// affine_set<(d0) : (d0 - 2 >= 0)>
/// ```
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#integersetattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct IntegerSetAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> IntegerSetAttributeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`IntegerSetAttributeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirIntegerSetAttrGetTypeID()).unwrap() }
    }

    /// Returns the [`IntegerSet`] that is stored in this [`IntegerSetAttributeRef`].
    pub fn integer_set(&self) -> IntegerSet<'c, 't> {
        unsafe { IntegerSet::from_c_api(mlirIntegerSetAttrGetValue(self.handle), self.context) }
    }
}

mlir_subtype_trait_impls!(IntegerSetAttributeRef<'c, 't> as Attribute, mlir_type = Attribute, mlir_subtype = IntegerSet);

impl<'c, 't> From<IntegerSet<'c, 't>> for IntegerSetAttributeRef<'c, 't> {
    fn from(value: IntegerSet<'c, 't>) -> Self {
        value.context().integer_set_attribute(value)
    }
}

impl<'c, 't> FromWithContext<'c, 't, IntegerSet<'c, 't>> for IntegerSetAttributeRef<'c, 't> {
    fn from_with_context(value: IntegerSet<'c, 't>, context: &'c Context<'t>) -> Self {
        context.integer_set_attribute(value)
    }
}

impl<'t> Context<'t> {
    /// Creates a new [`IntegerSetAttributeRef`] owned by this [`Context`].
    pub fn integer_set_attribute<'c>(&'c self, integer_set: IntegerSet<'c, 't>) -> IntegerSetAttributeRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        let _guard = self.borrow();
        unsafe { IntegerSetAttributeRef::from_c_api(mlirIntegerSetAttrGet(integer_set.to_c_api()), &self).unwrap() }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::IntoWithContext;
    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    #[test]
    fn test_integer_set_attribute_type_id() {
        let context = Context::new();
        let integer_set_attribute_id = IntegerSetAttributeRef::type_id();
        let integer_set_attribute_1: IntegerSetAttributeRef<'_, '_> = context.empty_integer_set(0, 1).into();
        let integer_set_attribute_2: IntegerSetAttributeRef<'_, '_> =
            context.empty_integer_set(0, 1).into_with_context(&context);
        assert_eq!(integer_set_attribute_1.type_id(), integer_set_attribute_2.type_id());
        assert_eq!(integer_set_attribute_id, integer_set_attribute_1.type_id());
    }

    #[test]
    fn test_integer_set_attribute() {
        let context = Context::new();
        let attribute = context.integer_set_attribute(context.empty_integer_set(0, 1));
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.integer_set(), context.empty_integer_set(0, 1));
    }

    #[test]
    fn test_integer_set_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.integer_set_attribute(context.empty_integer_set(0, 1));
        let attribute_2 = context.integer_set_attribute(context.empty_integer_set(0, 1));
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.integer_set_attribute(context.empty_integer_set(1, 0));
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.integer_set_attribute(context.empty_integer_set(0, 1));
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_integer_set_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.integer_set_attribute(context.empty_integer_set(0, 1));
        test_attribute_display_and_debug(attribute, "affine_set<()[s0] : (1 == 0)>");
    }

    #[test]
    fn test_integer_set_attribute_parsing() {
        let context = Context::new();
        let attribute = context.integer_set_attribute(context.empty_integer_set(0, 1));
        let parsed = context.parse_attribute("affine_set<()[s0] : (1 == 0)>").unwrap();
        assert_eq!(parsed, attribute);
    }

    #[test]
    fn test_integer_set_attribute_casting() {
        let context = Context::new();
        let attribute = context.integer_set_attribute(context.empty_integer_set(0, 1));
        test_attribute_casting(attribute);
    }
}
