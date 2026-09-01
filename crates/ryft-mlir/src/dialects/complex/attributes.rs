use ryft_xla_sys::bindings::MlirAttribute;
use ryft_xla_sys::mlir::dialects::complex::{
    mlirAttributeIsAComplex, mlirComplexAttrDoubleGet, mlirComplexAttrGetImagDouble, mlirComplexAttrGetRealDouble,
    mlirComplexAttrGetTypeID,
};

use crate::{Attribute, ComplexTypeRef, Context, DialectHandle, Error, Type, TypeId, mlir_subtype_trait_impls};

/// Complex number [`Attribute`] with floating-point real and imaginary components.
#[derive(Copy, Clone)]
pub struct NumberAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl NumberAttributeRef<'_, '_> {
    /// Gets the [`TypeId`] that corresponds to [`NumberAttributeRef`].
    pub fn type_id() -> Result<TypeId<'static>, Error> {
        unsafe { TypeId::from_c_api(mlirComplexAttrGetTypeID()) }
    }

    /// Returns the real component as an `f64`.
    pub fn real(&self) -> f64 {
        unsafe { mlirComplexAttrGetRealDouble(self.handle) }
    }

    /// Returns the imaginary component as an `f64`.
    pub fn imaginary(&self) -> f64 {
        unsafe { mlirComplexAttrGetImagDouble(self.handle) }
    }
}

impl<'c, 't> Attribute<'c, 't> for NumberAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Result<Self, Error> {
        if handle.ptr.is_null() {
            return Err(Error::internal("expected non-null MLIR attribute handle"));
        }
        if unsafe { mlirAttributeIsAComplex(handle) } {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected MLIR complex number attribute handle"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(NumberAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'t> Context<'t> {
    /// Creates a complex [`NumberAttributeRef`] owned by this [`Context`].
    pub fn complex_number_attribute<'c>(
        &'c self,
        r#type: ComplexTypeRef<'c, 't>,
        real: f64,
        imaginary: f64,
    ) -> Result<NumberAttributeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::complex()?)?;
        unsafe {
            NumberAttributeRef::from_c_api(
                mlirComplexAttrDoubleGet(*self.handle.borrow_mut(), r#type.to_c_api(), real, imaginary),
                self,
            )
            .map_err(|_| Error::invalid_argument("invalid arguments to `Context::complex_number_attribute`"))
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Attribute, Context};

    use super::*;

    #[test]
    fn test_number_attribute() {
        let context = Context::new();
        let r#type = context.complex_type(context.float64_type());
        let attribute = context.complex_number_attribute(r#type, 1.5, -2.25).unwrap();
        assert_eq!(attribute.real(), 1.5);
        assert_eq!(attribute.imaginary(), -2.25);
        assert_eq!(attribute.type_id().unwrap(), NumberAttributeRef::type_id().unwrap());
        assert_eq!(attribute.r#type().unwrap(), r#type);
        assert_eq!(attribute, context.complex_number_attribute(r#type, 1.5, -2.25).unwrap());
        assert_eq!(attribute.to_string(), "#complex.number<:f64 1.500000e+00, -2.250000e+00>");
        assert_eq!(format!("{attribute:?}"), "NumberAttributeRef[#complex.number<:f64 1.500000e+00, -2.250000e+00>]",);
        assert!(attribute.as_ref().is::<NumberAttributeRef>());
        assert_eq!(
            context
                .parse_attribute("#complex.number<:f64 1.500000e+00, -2.250000e+00>")
                .unwrap()
                .cast::<NumberAttributeRef>()
                .unwrap(),
            attribute,
        );
    }
}
