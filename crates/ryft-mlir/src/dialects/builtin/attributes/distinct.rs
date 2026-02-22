use ryft_xla_sys::bindings::{MlirAttribute, mlirDisctinctAttrCreate};

use crate::{Attribute, Context, mlir_subtype_trait_impls};

/// Built-in MLIR [`Attribute`] that associates an attribute with a unique identifier. As a result, multiple
/// [`DistinctAttributeRef`]s may point to the same underlying attribute. Every call to [`Context::distinct_attribute`]
/// allocates a new [`DistinctAttributeRef`] instance. The address of the attribute instance serves as a temporary
/// unique identifier. Similar to the names of SSA values, the final unique identifiers are generated during pretty
/// printing. This delayed numbering ensures the printed identifiers are deterministic even if multiple
/// [`DistinctAttributeRef`] instances are created in parallel.
///
/// This mechanism is meant to generate attributes with unique identifiers which can be used to mark groups of
/// operations that share a common property. For example, groups of aliasing memory operations may be marked using one
/// [`DistinctAttributeRef`] instance per alias group.
///
/// # Examples
///
/// The following are examples of [`DistinctAttributeRef`]s represented using their
/// [`Display`](std::fmt::Display) rendering:
///
/// ```text
/// #distinct = distinct[0]<42.0 : f32>
/// #distinct1 = distinct[1]<42.0 : f32>
/// #distinct2 = distinct[2]<array<i32: 10, 42>>
/// ```
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#distinctattribute)
/// for more information.
#[derive(Copy, Clone)]
pub struct DistinctAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> Attribute<'c, 't> for DistinctAttributeRef<'c, 't> {
    unsafe fn from_c_api(_handle: MlirAttribute, _context: &'c Context<'t>) -> Option<Self> {
        // Unfortunately, the MLIR C API does not provide a way to check if an [`MlirAttribute`] is a
        // [`DistinctAttributeRef`] or not and so we do not allow constructing [`DistinctAttributeRef`]s this way at all.
        // This means that downcasting [`Attribute`]s to [`DistinctAttributeRef`]s is not possible, for example. The
        // only way to construct a [`DistinctAttributeRef`] is to use the [`Context::distinct_attribute`] method, or
        // to call [`Attribute::into`] to wrap an existing [`Attribute`] into a new [`DistinctAttributeRef`].
        None
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        &self.context
    }
}

mlir_subtype_trait_impls!(DistinctAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'t> Context<'t> {
    /// Creates a new [`DistinctAttributeRef`] by wrapping the provided [`Attribute`].
    /// The resulting attribute is owned by this [`Context`].
    pub fn distinct_attribute<'c, A: Attribute<'c, 't>>(&'c self, attribute: A) -> DistinctAttributeRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        let _guard = self.borrow();
        unsafe { DistinctAttributeRef { handle: mlirDisctinctAttrCreate(attribute.to_c_api()), context: &self } }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_distinct_attribute() {
        let context = Context::new();

        // Test wrapping a boolean attribute.
        let attribute = context.distinct_attribute(context.boolean_attribute(true));
        assert_eq!(&context, attribute.context());

        // Test wrapping an integer attribute.
        let attribute = context.distinct_attribute(context.integer_attribute(context.signless_integer_type(32), 42));
        assert_eq!(&context, attribute.context());

        // Verify that [`DistinctAttributeRef::from_c_api`] always returns [`None`].
        assert_eq!(unsafe { DistinctAttributeRef::from_c_api(attribute.handle, &context) }, None);
    }

    #[test]
    fn test_distinct_attribute_equality() {
        let context = Context::new();
        let boolean_attribute = context.boolean_attribute(true);

        // Each call to [`Context::distinct_attribute`] creates a new distinct attribute instance, and so
        // the following two attributes should NOT be equal even if they are wrapping the same attribute.
        let attribute_1 = context.distinct_attribute(boolean_attribute);
        let attribute_2 = context.distinct_attribute(boolean_attribute);
        assert_ne!(attribute_1, attribute_2);

        // Different underlying attributes from the same context must not be equal.
        let attribute_2 = context.distinct_attribute(context.boolean_attribute(false));
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.distinct_attribute(context.boolean_attribute(true));
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_distinct_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.distinct_attribute(context.boolean_attribute(true));

        // The display format includes a unique ID which we can't predict, so we just check the prefix and the suffix.
        let display = attribute.to_string();
        assert!(display.starts_with("distinct["));
        assert!(display.ends_with("]<true>"));
    }
}
