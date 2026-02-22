use half::{bf16, f16};

use ryft_xla_sys::bindings::{
    MlirAttribute, mlirFloatAttrDoubleGet, mlirFloatAttrDoubleGetChecked, mlirFloatAttrGetTypeID,
    mlirFloatAttrGetValueDouble,
};

use crate::{Attribute, Context, FloatType, FromWithContext, Location, TypeId, mlir_subtype_trait_impls};

/// Built-in MLIR [`Attribute`] that stores a floating-point value. This attribute can be represented in the
/// hexadecimal form where the hexadecimal value is interpreted as bits of the underlying binary representation.
/// This form is useful for representing infinity and NaN floating point values. To avoid confusion with integer
/// attributes, hexadecimal literals must be followed by a float type to define a [`FloatAttributeRef`].
///
/// # Examples
///
/// The following are examples of [`FloatAttributeRef`]s represented using their
/// [`Display`](std::fmt::Display) rendering:
///
/// ```text
/// 42.0         // float attribute defaults to f64 type
/// 42.0 : f32   // float attribute of f32 type
/// 0x7C00 : f16 // positive infinity
/// 0x7CFF : f16 // NaN (one of possible values)
/// 42 : f32     // Error: expected integer type
/// ```
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#floatattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct FloatAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> FloatAttributeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`FloatAttributeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirFloatAttrGetTypeID()).unwrap() }
    }

    /// Returns the floating-point value that is stored in this [`FloatAttributeRef`].
    pub fn value(&self) -> f64 {
        unsafe { mlirFloatAttrGetValueDouble(self.handle) }
    }
}

mlir_subtype_trait_impls!(FloatAttributeRef<'c, 't> as Attribute, mlir_type = Attribute, mlir_subtype = Float);

impl<'c, 't> FromWithContext<'c, 't, f16> for FloatAttributeRef<'c, 't> {
    fn from_with_context(value: f16, context: &'c Context<'t>) -> Self {
        context.float_attribute(context.float32_type(), value.to_f64())
    }
}

impl<'c, 't> FromWithContext<'c, 't, bf16> for FloatAttributeRef<'c, 't> {
    fn from_with_context(value: bf16, context: &'c Context<'t>) -> Self {
        context.float_attribute(context.bfloat16_type(), value.to_f64())
    }
}

impl<'c, 't> FromWithContext<'c, 't, f32> for FloatAttributeRef<'c, 't> {
    fn from_with_context(value: f32, context: &'c Context<'t>) -> Self {
        context.float_attribute(context.float32_type(), value as f64)
    }
}

impl<'c, 't> FromWithContext<'c, 't, f64> for FloatAttributeRef<'c, 't> {
    fn from_with_context(value: f64, context: &'c Context<'t>) -> Self {
        context.float_attribute(context.float64_type(), value)
    }
}

impl<'t> Context<'t> {
    /// Creates a new [`FloatAttributeRef`] owned by this [`Context`].
    pub fn float_attribute<'c, T: FloatType<'c, 't>>(&'c self, r#type: T, value: f64) -> FloatAttributeRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe {
            FloatAttributeRef::from_c_api(mlirFloatAttrDoubleGet(*self.handle.borrow(), r#type.to_c_api(), value), self)
                .unwrap()
        }
    }

    /// Creates a new [`FloatAttributeRef`] owned by this [`Context`]. If any of the arguments are invalid, then this
    /// function will return [`None`] and will also emit the appropriate diagnostics at the provided location.
    pub fn checked_float_attribute<'c, T: FloatType<'c, 't>, L: Location<'c, 't>>(
        &'c self,
        r#type: T,
        value: f64,
        location: L,
    ) -> Option<FloatAttributeRef<'c, 't>> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        let _guard = self.borrow();
        unsafe {
            FloatAttributeRef::from_c_api(
                mlirFloatAttrDoubleGetChecked(location.to_c_api(), r#type.to_c_api(), value),
                self,
            )
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
    fn test_float_attribute_type_id() {
        let context = Context::new();
        let float_attribute_id = FloatAttributeRef::type_id();
        let float_attribute_1: FloatAttributeRef<'_, '_> = 1.0.into_with_context(&context);
        let float_attribute_2: FloatAttributeRef<'_, '_> = f16::PI.into_with_context(&context);
        assert_eq!(float_attribute_1.type_id(), float_attribute_2.type_id());
        assert_eq!(float_attribute_id, float_attribute_1.type_id());
    }

    #[test]
    fn test_float_attribute() {
        let context = Context::new();

        // Test with `float32` type.
        let attribute: FloatAttributeRef<'_, '_> = std::f64::consts::PI.into_with_context(&context);
        assert_eq!(&context, attribute.context());
        assert!((attribute.value() - std::f64::consts::PI).abs() < 1e-6);

        // Test with `float64` type.
        let attribute: FloatAttributeRef<'_, '_> = bf16::E.into_with_context(&context);
        assert_eq!(&context, attribute.context());
        assert!((attribute.value() - std::f64::consts::E).abs() < 1e-3);

        // Test with `bfloat16` type.
        let attribute = context.float_attribute(context.bfloat16_type(), 1.5);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value(), 1.5);
    }

    #[test]
    fn test_float_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.float_attribute(context.float64_type(), std::f64::consts::PI);
        let attribute_2 = context.float_attribute(context.float64_type(), std::f64::consts::PI);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.float_attribute(context.float64_type(), std::f64::consts::E);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.float_attribute(context.float64_type(), std::f64::consts::PI);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_float_attribute_display_and_debug() {
        let context = Context::new();

        let attribute: FloatAttributeRef<'_, '_> = 42.0f32.into_with_context(&context);
        test_attribute_display_and_debug(attribute, "4.200000e+01 : f32");

        let attribute = context.float_attribute(context.float64_type(), 100.5);
        test_attribute_display_and_debug(attribute, "1.005000e+02 : f64");
    }

    #[test]
    fn test_float_attribute_parsing() {
        let context = Context::new();
        assert_eq!(
            context.parse_attribute("4.200000e+01 : f32").unwrap(),
            context.float_attribute(context.float32_type(), 42.0)
        );
        assert_eq!(
            context.parse_attribute("1.005000e+02 : f64").unwrap(),
            context.float_attribute(context.float64_type(), 100.5)
        );
    }

    #[test]
    fn test_float_attribute_casting() {
        let context = Context::new();
        let attribute = context.float_attribute(context.float64_type(), std::f64::consts::PI);
        test_attribute_casting(attribute);
    }
}
