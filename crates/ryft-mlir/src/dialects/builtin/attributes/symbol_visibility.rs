use ryft_xla_sys::bindings::{MlirAttribute, mlirStringAttrGet, mlirStringAttrGetValue};

use crate::{Attribute, Context, FromWithContext, StringRef, SymbolVisibility, mlir_subtype_trait_impls};

/// [`StringAttributeRef`](crate::StringAttributeRef) that stores the string rendering of a [`SymbolVisibility`]. This
/// is not really a built-in MLIR attribute type since MLIR uses [`StringAttributeRef`](crate::StringAttributeRef) for
/// storing [`SymbolVisibility`] values. However, it is provided for convenience due to how common its use is
/// within MLIR.
#[derive(Copy, Clone)]
pub struct SymbolVisibilityAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> SymbolVisibilityAttributeRef<'c, 't> {
    /// Returns the [`SymbolVisibility`] value that is stored in this [`SymbolVisibilityAttributeRef`].
    /// Note that this function may return [`None`] if the string value that is stored in this attribute does not
    /// correspond to a valid [`SymbolVisibility`] string rendering. If needed, that underlying value can be obtained
    /// by casting this [`SymbolVisibilityAttributeRef`] to a [`StringAttributeRef`](crate::StringAttributeRef) and
    /// obtaining the underlying string value.
    pub fn visibility(&self) -> Option<SymbolVisibility> {
        let visibility = unsafe { StringRef::from_c_api(mlirStringAttrGetValue(self.handle)) };
        visibility.as_str().ok().and_then(|visibility| SymbolVisibility::try_from(visibility).ok())
    }
}

mlir_subtype_trait_impls!(
    SymbolVisibilityAttributeRef<'c, 't> as Attribute,
    mlir_type = Attribute,
    // Note that this function will not actually check whether the underlying string value corresponds to a valid
    // [`SymbolVisibility`] string rendering. That is because [`SymbolVisibilityAttributeRef`] is just a convenient
    // helper for working with [`StringAttributeRef`](crate::StringAttributeRef) that are used to store
    // [`SymbolVisibility`] values, but it is the responsibility of the caller to make sure that this helper
    // is used correctly.
    mlir_subtype = String,
);

impl<'c, 't> FromWithContext<'c, 't, SymbolVisibility> for SymbolVisibilityAttributeRef<'c, 't> {
    fn from_with_context(value: SymbolVisibility, context: &'c Context<'t>) -> Self {
        context.symbol_visibility_attribute(value)
    }
}

impl<'t> Context<'t> {
    /// Creates a new [`SymbolVisibilityAttributeRef`] owned by this [`Context`].
    pub fn symbol_visibility_attribute<'c>(
        &'c self,
        visibility: SymbolVisibility,
    ) -> SymbolVisibilityAttributeRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        let visibility = match visibility {
            SymbolVisibility::Public => "public",
            SymbolVisibility::Private => "private",
            SymbolVisibility::Nested => "nested",
        };
        unsafe {
            SymbolVisibilityAttributeRef::from_c_api(
                mlirStringAttrGet(*self.handle.borrow(), StringRef::from(visibility).to_c_api()),
                &self,
            )
            .unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};
    use crate::{IntoWithContext, SymbolVisibility};

    use super::*;

    #[test]
    fn test_symbol_visibility_attribute() {
        let context = Context::new();

        // Test public visibility.
        let attribute = context.symbol_visibility_attribute(SymbolVisibility::Public);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.visibility(), Some(SymbolVisibility::Public));

        // Test private visibility.
        let attribute: SymbolVisibilityAttributeRef<'_, '_> = SymbolVisibility::Private.into_with_context(&context);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.visibility(), Some(SymbolVisibility::Private));

        // Test nested visibility.
        let attribute = context.symbol_visibility_attribute(SymbolVisibility::Nested);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.visibility(), Some(SymbolVisibility::Nested));
    }

    #[test]
    fn test_symbol_visibility_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.symbol_visibility_attribute(SymbolVisibility::Public);
        let attribute_2 = context.symbol_visibility_attribute(SymbolVisibility::Public);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.symbol_visibility_attribute(SymbolVisibility::Private);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.symbol_visibility_attribute(SymbolVisibility::Public);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_symbol_visibility_attribute_display_and_debug() {
        let context = Context::new();

        let attribute = context.symbol_visibility_attribute(SymbolVisibility::Public);
        test_attribute_display_and_debug(attribute, "\"public\"");

        let attribute = context.symbol_visibility_attribute(SymbolVisibility::Private);
        test_attribute_display_and_debug(attribute, "\"private\"");

        let attribute = context.symbol_visibility_attribute(SymbolVisibility::Nested);
        test_attribute_display_and_debug(attribute, "\"nested\"");
    }

    #[test]
    fn test_symbol_visibility_attribute_parsing() {
        let context = Context::new();

        // Test parsing public visibility.
        let attribute = context.symbol_visibility_attribute(SymbolVisibility::Public);
        let parsed = context.parse_attribute("\"public\"").unwrap();
        assert_eq!(parsed, attribute.as_ref());

        // Test parsing private visibility.
        let attribute = context.symbol_visibility_attribute(SymbolVisibility::Private);
        let parsed = context.parse_attribute("\"private\"").unwrap();
        assert_eq!(parsed, attribute.as_ref());

        // Test parsing nested visibility.
        let attribute = context.symbol_visibility_attribute(SymbolVisibility::Nested);
        let parsed = context.parse_attribute("\"nested\"").unwrap();
        assert_eq!(parsed, attribute.as_ref());
    }

    #[test]
    fn test_symbol_visibility_attribute_casting() {
        let context = Context::new();
        let attribute = context.symbol_visibility_attribute(SymbolVisibility::Public);
        test_attribute_casting(attribute);
    }
}
