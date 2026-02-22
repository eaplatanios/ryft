use ryft_xla_sys::bindings::{MlirAttribute, mlirUnitAttrGet, mlirUnitAttrGetTypeID};

use crate::{Attribute, Context, TypeId, mlir_subtype_trait_impls};

/// Built-in MLIR [`Attribute`] that represents a value of `unit` type. The unit type allows only one value forming
/// a singleton set. This attribute value is used to represent attributes that only have meaning from their existence.
/// One example of such an attribute could be the `swift.self` attribute. This attribute indicates that a function
/// parameter is the self/context parameter. It could be represented as a
/// [`BooleanAttributeRef`](crate::BooleanAttributeRef) but a value of `false` does not really add any value.
/// The parameter either is the self/context or it is not.
///
/// # Examples
///
/// The following are examples of [`UnitAttributeRef`]s represented using their [`Display`](std::fmt::Display) rendering
/// in the context of the rendering of an operation:
///
/// ```text
/// // A unit attribute is defined with the `unit` value specifier:
/// func.func @verbose_form() attributes {dialectName.unitAttr = unit}
///
/// // A unit attribute in an attribute dictionary can also be defined without the value specifier:
/// func.func @simple_form() attributes {dialectName.unitAttr}
/// ```
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#unitattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct UnitAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> UnitAttributeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`UnitAttributeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirUnitAttrGetTypeID()).unwrap() }
    }
}

mlir_subtype_trait_impls!(UnitAttributeRef<'c, 't> as Attribute, mlir_type = Attribute, mlir_subtype = Unit);

impl<'t> Context<'t> {
    /// Creates a new [`UnitAttributeRef`] owned by this [`Context`].
    pub fn unit_attribute<'c>(&'c self) -> UnitAttributeRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe { UnitAttributeRef::from_c_api(mlirUnitAttrGet(*self.handle.borrow()), self).unwrap() }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::attributes::tests::test_attribute_display_and_debug;

    use super::*;

    #[test]
    fn test_unit_attribute_type_id() {
        let context = Context::new();
        let unit_attribute_id = UnitAttributeRef::type_id();
        let unit_attribute_1 = context.unit_attribute();
        let unit_attribute_2 = context.unit_attribute();
        assert_eq!(unit_attribute_1.type_id(), unit_attribute_2.type_id());
        assert_eq!(unit_attribute_id, unit_attribute_1.type_id());
    }

    #[test]
    fn test_unit_attribute() {
        let context = Context::new();
        let attribute = context.unit_attribute();
        assert_eq!(&context, attribute.context());
    }

    #[test]
    fn test_unit_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.unit_attribute();
        let attribute_2 = context.unit_attribute();
        assert_eq!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.unit_attribute();
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_unit_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.unit_attribute();
        test_attribute_display_and_debug(attribute, "unit");
    }

    #[test]
    fn test_unit_attribute_parsing() {
        let context = Context::new();
        assert_eq!(context.parse_attribute("unit").unwrap(), context.unit_attribute());
    }

    #[test]
    fn test_unit_attribute_casting() {
        let context = Context::new();
        let attribute = context.unit_attribute();
        let rendered_attribute = attribute.to_string();

        // Test upcasting.
        let attribute = attribute.as_ref();
        assert!(attribute.is::<UnitAttributeRef>());
        assert_eq!(attribute.to_string(), rendered_attribute);

        // Test downcasting.
        let attribute = attribute.cast::<UnitAttributeRef>().unwrap();
        assert!(attribute.is::<UnitAttributeRef>());
        assert_eq!(attribute.to_string(), rendered_attribute);

        // Invalid cast from specific attribute.
        let attribute = context.integer_attribute(context.signless_integer_type(1), 0);
        assert!(!attribute.is::<UnitAttributeRef>());
        assert_eq!(attribute.cast::<UnitAttributeRef>(), None);

        // Invalid cast from a generic attribute reference.
        let attribute = attribute.as_ref();
        assert!(!attribute.is::<UnitAttributeRef>());
        assert_eq!(attribute.cast::<UnitAttributeRef>(), None);
    }
}
