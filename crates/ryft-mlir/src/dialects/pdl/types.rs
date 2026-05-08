use ryft_xla_sys::bindings::{
    MlirType, mlirPDLAttributeTypeGet, mlirPDLAttributeTypeGetTypeID, mlirPDLOperationTypeGet,
    mlirPDLOperationTypeGetTypeID, mlirPDLRangeTypeGet, mlirPDLRangeTypeGetElementType, mlirPDLRangeTypeGetTypeID,
    mlirPDLTypeTypeGet, mlirPDLTypeTypeGetTypeID, mlirPDLValueTypeGet, mlirPDLValueTypeGetTypeID,
    mlirTypeIsAPDLAttributeType, mlirTypeIsAPDLOperationType, mlirTypeIsAPDLRangeType, mlirTypeIsAPDLTypeType,
    mlirTypeIsAPDLValueType,
};

use crate::{Context, DialectHandle, Error, Type, TypeId, TypeRef, mlir_subtype_trait_impls};

/// PDL dialect [`Type`] that represents a handle to an [`Attribute`](crate::Attribute).
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/PDLOps/#type-definitions)
/// for more information.
#[derive(Copy, Clone)]
pub struct AttributeTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl AttributeTypeRef<'_, '_> {
    /// Gets the [`TypeId`] that corresponds to [`AttributeTypeRef`].
    pub fn type_id() -> Result<TypeId<'static>, Error> {
        unsafe { TypeId::from_c_api(mlirPDLAttributeTypeGetTypeID()) }
    }
}

impl<'c, 't> Type<'c, 't> for AttributeTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null() && unsafe { mlirTypeIsAPDLAttributeType(handle) } {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected MLIR type handle"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirType {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(AttributeTypeRef<'c, 't> as Type, mlir_type = Type);

/// PDL dialect [`Type`] that represents a handle to an [`Operation`](crate::Operation).
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/PDLOps/#type-definitions)
/// for more information.
#[derive(Copy, Clone)]
pub struct OperationTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl OperationTypeRef<'_, '_> {
    /// Gets the [`TypeId`] that corresponds to [`OperationTypeRef`].
    pub fn type_id() -> Result<TypeId<'static>, Error> {
        unsafe { TypeId::from_c_api(mlirPDLOperationTypeGetTypeID()) }
    }
}

impl<'c, 't> Type<'c, 't> for OperationTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null() && unsafe { mlirTypeIsAPDLOperationType(handle) } {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected MLIR type handle"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirType {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(OperationTypeRef<'c, 't> as Type, mlir_type = Type);

/// PDL dialect [`Type`] that represents a range of PDL entities with a shared element type.
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/PDLOps/#type-definitions)
/// for more information.
#[derive(Copy, Clone)]
pub struct RangeTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> RangeTypeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`RangeTypeRef`].
    pub fn type_id() -> Result<TypeId<'static>, Error> {
        unsafe { TypeId::from_c_api(mlirPDLRangeTypeGetTypeID()) }
    }

    /// Returns the PDL element [`Type`] stored by this range.
    pub fn element_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        unsafe {
            TypeRef::from_c_api(mlirPDLRangeTypeGetElementType(self.handle), self.context)
                .map_err(|_| Error::internal("MLIR returned an invalid PDL range element type"))
        }
    }
}

impl<'c, 't> Type<'c, 't> for RangeTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null() && unsafe { mlirTypeIsAPDLRangeType(handle) } {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected MLIR type handle"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirType {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(RangeTypeRef<'c, 't> as Type, mlir_type = Type);

/// PDL dialect [`Type`] that represents a handle to an MLIR [`Type`].
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/PDLOps/#type-definitions)
/// for more information.
#[derive(Copy, Clone)]
pub struct TypeTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl TypeTypeRef<'_, '_> {
    /// Gets the [`TypeId`] that corresponds to [`TypeTypeRef`].
    pub fn type_id() -> Result<TypeId<'static>, Error> {
        unsafe { TypeId::from_c_api(mlirPDLTypeTypeGetTypeID()) }
    }
}

impl<'c, 't> Type<'c, 't> for TypeTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null() && unsafe { mlirTypeIsAPDLTypeType(handle) } {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected MLIR type handle"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirType {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(TypeTypeRef<'c, 't> as Type, mlir_type = Type);

/// PDL dialect [`Type`] that represents a handle to an MLIR [`Value`](crate::Value).
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/PDLOps/#type-definitions)
/// for more information.
#[derive(Copy, Clone)]
pub struct ValueTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl ValueTypeRef<'_, '_> {
    /// Gets the [`TypeId`] that corresponds to [`ValueTypeRef`].
    pub fn type_id() -> Result<TypeId<'static>, Error> {
        unsafe { TypeId::from_c_api(mlirPDLValueTypeGetTypeID()) }
    }
}

impl<'c, 't> Type<'c, 't> for ValueTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null() && unsafe { mlirTypeIsAPDLValueType(handle) } {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected MLIR type handle"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirType {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(ValueTypeRef<'c, 't> as Type, mlir_type = Type);

impl<'t> Context<'t> {
    /// Creates a new [`AttributeTypeRef`] owned by this [`Context`].
    pub fn pdl_attribute_type<'c>(&'c self) -> Result<AttributeTypeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::pdl()?)?;
        unsafe {
            AttributeTypeRef::from_c_api(mlirPDLAttributeTypeGet(*self.handle.borrow()), self)
                .map_err(|_| Error::internal("MLIR returned an invalid PDL attribute type"))
        }
    }

    /// Creates a new [`OperationTypeRef`] owned by this [`Context`].
    pub fn pdl_operation_type<'c>(&'c self) -> Result<OperationTypeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::pdl()?)?;
        unsafe {
            OperationTypeRef::from_c_api(mlirPDLOperationTypeGet(*self.handle.borrow()), self)
                .map_err(|_| Error::internal("MLIR returned an invalid PDL operation type"))
        }
    }

    /// Creates a new [`RangeTypeRef`] owned by this [`Context`].
    pub fn pdl_range_type<'c, T: Type<'c, 't>>(&'c self, element_type: T) -> Result<RangeTypeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::pdl()?)?;
        unsafe {
            RangeTypeRef::from_c_api(mlirPDLRangeTypeGet(element_type.to_c_api()), self)
                .map_err(|_| Error::invalid_argument("invalid arguments to `Context::pdl_range_type`"))
        }
    }

    /// Creates a new [`TypeTypeRef`] owned by this [`Context`].
    pub fn pdl_type_type<'c>(&'c self) -> Result<TypeTypeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::pdl()?)?;
        unsafe {
            TypeTypeRef::from_c_api(mlirPDLTypeTypeGet(*self.handle.borrow()), self)
                .map_err(|_| Error::internal("MLIR returned an invalid PDL type type"))
        }
    }

    /// Creates a new [`ValueTypeRef`] owned by this [`Context`].
    pub fn pdl_value_type<'c>(&'c self) -> Result<ValueTypeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::pdl()?)?;
        unsafe {
            ValueTypeRef::from_c_api(mlirPDLValueTypeGet(*self.handle.borrow()), self)
                .map_err(|_| Error::internal("MLIR returned an invalid PDL value type"))
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::Type;
    use crate::types::tests::{test_type_casting, test_type_display_and_debug};

    use super::*;

    #[test]
    fn test_attribute_type() {
        let context = Context::new();
        let r#type = context.pdl_attribute_type().unwrap();
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.dialect().unwrap().namespace().unwrap(), "pdl");
        assert_eq!(AttributeTypeRef::type_id().unwrap(), r#type.type_id().unwrap());
    }

    #[test]
    fn test_attribute_type_equality() {
        let context = Context::new();
        let type_1 = context.pdl_attribute_type().unwrap();
        let type_2 = context.pdl_attribute_type().unwrap();
        assert_eq!(type_1, type_2);

        let context = Context::new();
        let type_2 = context.pdl_attribute_type().unwrap();
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_attribute_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.pdl_attribute_type().unwrap(), "!pdl.attribute");
    }

    #[test]
    fn test_attribute_type_parsing() {
        let context = Context::new();
        let r#type = context.pdl_attribute_type().unwrap();
        assert_eq!(context.parse_type("!pdl.attribute").unwrap(), r#type);
    }

    #[test]
    fn test_attribute_type_casting() {
        let context = Context::new();
        test_type_casting(context.pdl_attribute_type().unwrap());
    }

    #[test]
    fn test_operation_type() {
        let context = Context::new();
        let r#type = context.pdl_operation_type().unwrap();
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.dialect().unwrap().namespace().unwrap(), "pdl");
        assert_eq!(OperationTypeRef::type_id().unwrap(), r#type.type_id().unwrap());
    }

    #[test]
    fn test_operation_type_equality() {
        let context = Context::new();
        let type_1 = context.pdl_operation_type().unwrap();
        let type_2 = context.pdl_operation_type().unwrap();
        assert_eq!(type_1, type_2);

        let context = Context::new();
        let type_2 = context.pdl_operation_type().unwrap();
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_operation_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.pdl_operation_type().unwrap(), "!pdl.operation");
    }

    #[test]
    fn test_operation_type_parsing() {
        let context = Context::new();
        let r#type = context.pdl_operation_type().unwrap();
        assert_eq!(context.parse_type("!pdl.operation").unwrap(), r#type);
    }

    #[test]
    fn test_operation_type_casting() {
        let context = Context::new();
        test_type_casting(context.pdl_operation_type().unwrap());
    }

    #[test]
    fn test_range_type() {
        let context = Context::new();
        let element_type = context.pdl_value_type().unwrap();
        let r#type = context.pdl_range_type(element_type).unwrap();
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.dialect().unwrap().namespace().unwrap(), "pdl");
        assert_eq!(r#type.element_type().unwrap(), element_type.as_ref());
        assert_eq!(r#type.element_type().unwrap(), element_type.as_ref());
    }

    #[test]
    fn test_range_type_equality() {
        let context = Context::new();
        let type_1 = context.pdl_range_type(context.pdl_value_type().unwrap()).unwrap();
        let type_2 = context.pdl_range_type(context.pdl_value_type().unwrap()).unwrap();
        assert_eq!(type_1, type_2);

        let type_2 = context.pdl_range_type(context.pdl_type_type().unwrap()).unwrap();
        assert_ne!(type_1, type_2);

        let context = Context::new();
        let type_2 = context.pdl_range_type(context.pdl_value_type().unwrap()).unwrap();
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_range_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(
            context.pdl_range_type(context.pdl_value_type().unwrap()).unwrap(),
            "!pdl.range<value>",
        );
    }

    #[test]
    fn test_range_type_parsing() {
        let context = Context::new();
        let r#type = context.pdl_range_type(context.pdl_value_type().unwrap()).unwrap();
        assert_eq!(context.parse_type("!pdl.range<value>").unwrap(), r#type);
    }

    #[test]
    fn test_range_type_casting() {
        let context = Context::new();
        test_type_casting(context.pdl_range_type(context.pdl_value_type().unwrap()).unwrap());
    }

    #[test]
    fn test_type_type() {
        let context = Context::new();
        let r#type = context.pdl_type_type().unwrap();
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.dialect().unwrap().namespace().unwrap(), "pdl");
        assert_eq!(TypeTypeRef::type_id().unwrap(), r#type.type_id().unwrap());
    }

    #[test]
    fn test_type_type_equality() {
        let context = Context::new();
        let type_1 = context.pdl_type_type().unwrap();
        let type_2 = context.pdl_type_type().unwrap();
        assert_eq!(type_1, type_2);

        let context = Context::new();
        let type_2 = context.pdl_type_type().unwrap();
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_type_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.pdl_type_type().unwrap(), "!pdl.type");
    }

    #[test]
    fn test_type_type_parsing() {
        let context = Context::new();
        let r#type = context.pdl_type_type().unwrap();
        assert_eq!(context.parse_type("!pdl.type").unwrap(), r#type);
    }

    #[test]
    fn test_type_type_casting() {
        let context = Context::new();
        test_type_casting(context.pdl_type_type().unwrap());
    }

    #[test]
    fn test_value_type() {
        let context = Context::new();
        let r#type = context.pdl_value_type().unwrap();
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.dialect().unwrap().namespace().unwrap(), "pdl");
        assert_eq!(ValueTypeRef::type_id().unwrap(), r#type.type_id().unwrap());
    }

    #[test]
    fn test_value_type_equality() {
        let context = Context::new();
        let type_1 = context.pdl_value_type().unwrap();
        let type_2 = context.pdl_value_type().unwrap();
        assert_eq!(type_1, type_2);

        let context = Context::new();
        let type_2 = context.pdl_value_type().unwrap();
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_value_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.pdl_value_type().unwrap(), "!pdl.value");
    }

    #[test]
    fn test_value_type_parsing() {
        let context = Context::new();
        let r#type = context.pdl_value_type().unwrap();
        assert_eq!(context.parse_type("!pdl.value").unwrap(), r#type);
    }

    #[test]
    fn test_value_type_casting() {
        let context = Context::new();
        test_type_casting(context.pdl_value_type().unwrap());
    }
}
