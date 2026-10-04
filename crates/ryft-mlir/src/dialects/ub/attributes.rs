use ryft_xla_sys::bindings::MlirAttribute;
use ryft_xla_sys::mlir::dialects::ub::{mlirAttributeIsAUbPoisonAttr, mlirUbPoisonAttrGet, mlirUbPoisonAttrGetTypeID};

use crate::{Attribute, Context, DialectHandle, Error, TypeId, mlir_subtype_trait_impls};

/// UB poison [`Attribute`] representing a fully poisoned value.
#[derive(Copy, Clone)]
pub struct PoisonAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl PoisonAttributeRef<'_, '_> {
    /// Gets the [`TypeId`] that corresponds to [`PoisonAttributeRef`].
    pub fn type_id() -> Result<TypeId<'static>, Error> {
        unsafe { TypeId::from_c_api(mlirUbPoisonAttrGetTypeID()) }
    }
}

impl<'c, 't> Attribute<'c, 't> for PoisonAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Result<Self, Error> {
        if handle.ptr.is_null() {
            return Err(Error::internal("expected non-null MLIR attribute handle"));
        }
        if unsafe { mlirAttributeIsAUbPoisonAttr(handle) } {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected MLIR UB poison attribute handle"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(PoisonAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'t> Context<'t> {
    /// Creates a UB [`PoisonAttributeRef`] owned by this [`Context`].
    pub fn ub_poison_attribute<'c>(&'c self) -> Result<PoisonAttributeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::ub()?)?;
        unsafe {
            PoisonAttributeRef::from_c_api(mlirUbPoisonAttrGet(*self.handle.borrow_mut()), self)
                .map_err(|_| Error::invalid_argument("invalid arguments to `Context::ub_poison_attribute`"))
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Attribute, Context};

    use super::*;

    #[test]
    fn test_poison_attribute() {
        let context = Context::new();
        let attribute = context.ub_poison_attribute().unwrap();
        assert_eq!(attribute, context.ub_poison_attribute().unwrap());
        assert_eq!(attribute.type_id().unwrap(), PoisonAttributeRef::type_id().unwrap());
        assert_eq!(attribute.to_string(), "#ub.poison");
        assert_eq!(format!("{attribute:?}"), "PoisonAttributeRef[#ub.poison]");
        assert!(attribute.as_ref().is::<PoisonAttributeRef>());
        assert_eq!(context.parse_attribute("#ub.poison").unwrap().cast::<PoisonAttributeRef>().unwrap(), attribute,);
    }
}
