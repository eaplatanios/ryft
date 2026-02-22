use ryft_xla_sys::bindings::{MlirAttribute, mlirLocationFromAttribute, mlirLocationGetAttribute};

use crate::{Attribute, Context, Location, LocationRef, mlir_subtype_trait_impls};

/// Built-in MLIR [`Attribute`] that stores a [`Location`]. Refer to the
/// [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#location-attributes)
/// for more information.
#[derive(Copy, Clone)]
pub struct LocationAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> LocationAttributeRef<'c, 't> {
    /// Returns the [`Location`] that is stored in this [`LocationAttributeRef`].
    pub fn location(&self) -> LocationRef<'c, 't> {
        unsafe { LocationRef::from_c_api(mlirLocationFromAttribute(self.handle), self.context).unwrap() }
    }
}

mlir_subtype_trait_impls!(LocationAttributeRef<'c, 't> as Attribute, mlir_type = Attribute, mlir_subtype = Location);

impl<'t> Context<'t> {
    /// Creates a new [`LocationAttributeRef`] owned by this [`Context`].
    pub fn location_attribute<'c, L: Location<'c, 't>>(&'c self, location: L) -> LocationAttributeRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        let _guard = self.borrow();
        unsafe { LocationAttributeRef::from_c_api(mlirLocationGetAttribute(location.to_c_api()), self).unwrap() }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    #[test]
    fn test_location_attribute() {
        let context = Context::new();

        // Test with unknown location.
        let location = context.unknown_location();
        let attribute = context.location_attribute(location);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.location(), location);

        // Test with file location.
        let location = context.file_location("test.rs", 10, 5);
        let attribute = context.location_attribute(location);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.location(), location);
    }

    #[test]
    fn test_location_attribute_equality() {
        let context = Context::new();
        let location = context.unknown_location();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.location_attribute(location);
        let attribute_2 = context.location_attribute(location);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let other_location = context.file_location("test.rs", 10, 5);
        let attribute_2 = context.location_attribute(other_location);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.location_attribute(context.unknown_location());
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_location_attribute_display_and_debug() {
        let context = Context::new();
        let location = context.unknown_location();
        let attribute = context.location_attribute(location);
        test_attribute_display_and_debug(attribute, "loc(unknown)");
    }

    #[test]
    fn test_location_attribute_parsing() {
        let context = Context::new();
        let location = context.unknown_location();
        let attribute = context.location_attribute(location);
        let parsed = context.parse_attribute("loc(unknown)").unwrap();
        assert_eq!(parsed, attribute);
    }

    #[test]
    fn test_location_attribute_casting() {
        let context = Context::new();
        let location = context.unknown_location();
        let attribute = context.location_attribute(location);
        test_attribute_casting(attribute);
    }
}
