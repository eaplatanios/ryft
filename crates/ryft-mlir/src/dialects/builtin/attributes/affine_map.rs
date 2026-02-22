use ryft_xla_sys::bindings::{
    MlirAttribute, mlirAffineMapAttrGet, mlirAffineMapAttrGetTypeID, mlirAffineMapAttrGetValue,
};

use crate::{AffineMap, Attribute, Context, FromWithContext, TypeId, mlir_subtype_trait_impls};

/// Built-in MLIR [`Attribute`] that stores an [`AffineMap`].
///
/// # Examples
///
/// The following are examples of [`AffineMapAttributeRef`]s represented using their
/// [`Display`](std::fmt::Display) rendering:
///
/// ```text
/// affine_map<(d0) -> (d0)>
/// affine_map<(d0, d1, d2) -> (d0, d1)>
/// ```
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#affinemapattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct AffineMapAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> AffineMapAttributeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`AffineMapAttributeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirAffineMapAttrGetTypeID()).unwrap() }
    }

    /// Returns the [`AffineMap`] that is stored in this [`AffineMapAttributeRef`].
    pub fn affine_map(&self) -> AffineMap<'c, 't> {
        unsafe { AffineMap::from_c_api(mlirAffineMapAttrGetValue(self.handle), self.context).unwrap() }
    }
}

mlir_subtype_trait_impls!(AffineMapAttributeRef<'c, 't> as Attribute, mlir_type = Attribute, mlir_subtype = AffineMap);

impl<'c, 't> From<AffineMap<'c, 't>> for AffineMapAttributeRef<'c, 't> {
    fn from(value: AffineMap<'c, 't>) -> Self {
        value.context().affine_map_attribute(value)
    }
}

impl<'c, 't> FromWithContext<'c, 't, AffineMap<'c, 't>> for AffineMapAttributeRef<'c, 't> {
    fn from_with_context(value: AffineMap<'c, 't>, context: &'c Context<'t>) -> Self {
        context.affine_map_attribute(value)
    }
}

impl<'t> Context<'t> {
    /// Creates a new [`AffineMapAttributeRef`] owned by this [`Context`].
    pub fn affine_map_attribute<'c>(&'c self, affine_map: AffineMap<'c, 't>) -> AffineMapAttributeRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        let _guard = self.borrow();
        unsafe { AffineMapAttributeRef::from_c_api(mlirAffineMapAttrGet(affine_map.to_c_api()), &self).unwrap() }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::IntoWithContext;
    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    #[test]
    fn test_affine_map_attribute_type_id() {
        let context = Context::new();
        let affine_map_attribute_id = AffineMapAttributeRef::type_id();
        let affine_map = context.zero_result_affine_map(1, 0);
        let affine_map_attribute_1: AffineMapAttributeRef<'_, '_> = affine_map.into_with_context(&context);
        let affine_map_attribute_2 = AffineMapAttributeRef::from(affine_map);
        assert_eq!(affine_map_attribute_1.type_id(), affine_map_attribute_2.type_id());
        assert_eq!(affine_map_attribute_id, affine_map_attribute_1.type_id());
    }

    #[test]
    fn test_affine_map_attribute() {
        let context = Context::new();
        let affine_map = context.zero_result_affine_map(1, 0);
        let attribute = context.affine_map_attribute(affine_map);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.affine_map(), affine_map);
    }

    #[test]
    fn test_affine_map_attribute_equality() {
        let context = Context::new();
        let affine_map = context.zero_result_affine_map(1, 0);

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.affine_map_attribute(affine_map);
        let attribute_2 = context.affine_map_attribute(affine_map);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let other_affine_map = context.zero_result_affine_map(2, 0);
        let attribute_2 = context.affine_map_attribute(other_affine_map);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let other_affine_map = context.zero_result_affine_map(1, 0);
        let attribute_2 = context.affine_map_attribute(other_affine_map);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_affine_map_attribute_display_and_debug() {
        let context = Context::new();
        let affine_map = context.zero_result_affine_map(1, 0);
        let attribute = context.affine_map_attribute(affine_map);
        test_attribute_display_and_debug(attribute, "affine_map<(d0) -> ()>");
    }

    #[test]
    fn test_affine_map_attribute_parsing() {
        let context = Context::new();
        let affine_map = context.zero_result_affine_map(1, 0);
        assert_eq!(
            context.parse_attribute("affine_map<(d0) -> ()>").unwrap(),
            context.affine_map_attribute(affine_map)
        );
    }

    #[test]
    fn test_affine_map_attribute_casting() {
        let context = Context::new();
        let affine_map = context.zero_result_affine_map(1, 0);
        let attribute = context.affine_map_attribute(affine_map);
        test_attribute_casting(attribute);
    }
}
