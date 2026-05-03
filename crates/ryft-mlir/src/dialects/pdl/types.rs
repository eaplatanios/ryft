use ryft_xla_sys::bindings::{
    MlirType, mlirPDLAttributeTypeGet, mlirPDLAttributeTypeGetTypeID, mlirPDLOperationTypeGet,
    mlirPDLOperationTypeGetTypeID, mlirPDLRangeTypeGet, mlirPDLRangeTypeGetElementType, mlirPDLRangeTypeGetTypeID,
    mlirPDLTypeTypeGet, mlirPDLTypeTypeGetTypeID, mlirPDLValueTypeGet, mlirPDLValueTypeGetTypeID,
    mlirTypeIsAPDLAttributeType, mlirTypeIsAPDLOperationType, mlirTypeIsAPDLRangeType, mlirTypeIsAPDLTypeType,
    mlirTypeIsAPDLValueType,
};

use crate::{Context, DialectHandle, Type, TypeId, TypeRef, mlir_subtype_trait_impls};

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
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirPDLAttributeTypeGetTypeID()).unwrap() }
    }
}

impl<'c, 't> Type<'c, 't> for AttributeTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirTypeIsAPDLAttributeType(handle) } {
            Some(Self { handle, context })
        } else {
            None
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
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirPDLOperationTypeGetTypeID()).unwrap() }
    }
}

impl<'c, 't> Type<'c, 't> for OperationTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirTypeIsAPDLOperationType(handle) } {
            Some(Self { handle, context })
        } else {
            None
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
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirPDLRangeTypeGetTypeID()).unwrap() }
    }

    /// Returns the PDL element [`Type`] stored by this range.
    pub fn element_type(&self) -> TypeRef<'c, 't> {
        unsafe { TypeRef::from_c_api(mlirPDLRangeTypeGetElementType(self.handle), self.context).unwrap() }
    }
}

impl<'c, 't> Type<'c, 't> for RangeTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirTypeIsAPDLRangeType(handle) } {
            Some(Self { handle, context })
        } else {
            None
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
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirPDLTypeTypeGetTypeID()).unwrap() }
    }
}

impl<'c, 't> Type<'c, 't> for TypeTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirTypeIsAPDLTypeType(handle) } {
            Some(Self { handle, context })
        } else {
            None
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
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirPDLValueTypeGetTypeID()).unwrap() }
    }
}

impl<'c, 't> Type<'c, 't> for ValueTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirTypeIsAPDLValueType(handle) } {
            Some(Self { handle, context })
        } else {
            None
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
    pub fn pdl_attribute_type<'c>(&'c self) -> AttributeTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::pdl());
        unsafe { AttributeTypeRef::from_c_api(mlirPDLAttributeTypeGet(*self.handle.borrow()), self).unwrap() }
    }

    /// Creates a new [`OperationTypeRef`] owned by this [`Context`].
    pub fn pdl_operation_type<'c>(&'c self) -> OperationTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::pdl());
        unsafe { OperationTypeRef::from_c_api(mlirPDLOperationTypeGet(*self.handle.borrow()), self).unwrap() }
    }

    /// Creates a new [`RangeTypeRef`] owned by this [`Context`].
    pub fn pdl_range_type<'c, T: Type<'c, 't>>(&'c self, element_type: T) -> RangeTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::pdl());
        unsafe { RangeTypeRef::from_c_api(mlirPDLRangeTypeGet(element_type.to_c_api()), self).unwrap() }
    }

    /// Creates a new [`TypeTypeRef`] owned by this [`Context`].
    pub fn pdl_type_type<'c>(&'c self) -> TypeTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::pdl());
        unsafe { TypeTypeRef::from_c_api(mlirPDLTypeTypeGet(*self.handle.borrow()), self).unwrap() }
    }

    /// Creates a new [`ValueTypeRef`] owned by this [`Context`].
    pub fn pdl_value_type<'c>(&'c self) -> ValueTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::pdl());
        unsafe { ValueTypeRef::from_c_api(mlirPDLValueTypeGet(*self.handle.borrow()), self).unwrap() }
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
        let r#type = context.pdl_attribute_type();
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.dialect().namespace().unwrap(), "pdl");
        assert_eq!(AttributeTypeRef::type_id(), r#type.type_id());
    }

    #[test]
    fn test_attribute_type_equality() {
        let context = Context::new();
        let type_1 = context.pdl_attribute_type();
        let type_2 = context.pdl_attribute_type();
        assert_eq!(type_1, type_2);

        let context = Context::new();
        let type_2 = context.pdl_attribute_type();
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_attribute_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.pdl_attribute_type(), "!pdl.attribute");
    }

    #[test]
    fn test_attribute_type_parsing() {
        let context = Context::new();
        let r#type = context.pdl_attribute_type();
        assert_eq!(context.parse_type("!pdl.attribute").unwrap(), r#type);
    }

    #[test]
    fn test_attribute_type_casting() {
        let context = Context::new();
        test_type_casting(context.pdl_attribute_type());
    }

    #[test]
    fn test_operation_type() {
        let context = Context::new();
        let r#type = context.pdl_operation_type();
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.dialect().namespace().unwrap(), "pdl");
        assert_eq!(OperationTypeRef::type_id(), r#type.type_id());
    }

    #[test]
    fn test_operation_type_equality() {
        let context = Context::new();
        let type_1 = context.pdl_operation_type();
        let type_2 = context.pdl_operation_type();
        assert_eq!(type_1, type_2);

        let context = Context::new();
        let type_2 = context.pdl_operation_type();
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_operation_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.pdl_operation_type(), "!pdl.operation");
    }

    #[test]
    fn test_operation_type_parsing() {
        let context = Context::new();
        let r#type = context.pdl_operation_type();
        assert_eq!(context.parse_type("!pdl.operation").unwrap(), r#type);
    }

    #[test]
    fn test_operation_type_casting() {
        let context = Context::new();
        test_type_casting(context.pdl_operation_type());
    }

    #[test]
    fn test_range_type() {
        let context = Context::new();
        let element_type = context.pdl_value_type();
        let r#type = context.pdl_range_type(element_type);
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.dialect().namespace().unwrap(), "pdl");
        assert_eq!(r#type.element_type(), element_type.as_ref());
        assert_eq!(RangeTypeRef::type_id(), r#type.type_id());
    }

    #[test]
    fn test_range_type_equality() {
        let context = Context::new();
        let type_1 = context.pdl_range_type(context.pdl_value_type());
        let type_2 = context.pdl_range_type(context.pdl_value_type());
        assert_eq!(type_1, type_2);

        let type_2 = context.pdl_range_type(context.pdl_type_type());
        assert_ne!(type_1, type_2);

        let context = Context::new();
        let type_2 = context.pdl_range_type(context.pdl_value_type());
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_range_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.pdl_range_type(context.pdl_value_type()), "!pdl.range<value>");
    }

    #[test]
    fn test_range_type_parsing() {
        let context = Context::new();
        let r#type = context.pdl_range_type(context.pdl_value_type());
        assert_eq!(context.parse_type("!pdl.range<value>").unwrap(), r#type);
    }

    #[test]
    fn test_range_type_casting() {
        let context = Context::new();
        test_type_casting(context.pdl_range_type(context.pdl_value_type()));
    }

    #[test]
    fn test_type_type() {
        let context = Context::new();
        let r#type = context.pdl_type_type();
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.dialect().namespace().unwrap(), "pdl");
        assert_eq!(TypeTypeRef::type_id(), r#type.type_id());
    }

    #[test]
    fn test_type_type_equality() {
        let context = Context::new();
        let type_1 = context.pdl_type_type();
        let type_2 = context.pdl_type_type();
        assert_eq!(type_1, type_2);

        let context = Context::new();
        let type_2 = context.pdl_type_type();
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_type_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.pdl_type_type(), "!pdl.type");
    }

    #[test]
    fn test_type_type_parsing() {
        let context = Context::new();
        let r#type = context.pdl_type_type();
        assert_eq!(context.parse_type("!pdl.type").unwrap(), r#type);
    }

    #[test]
    fn test_type_type_casting() {
        let context = Context::new();
        test_type_casting(context.pdl_type_type());
    }

    #[test]
    fn test_value_type() {
        let context = Context::new();
        let r#type = context.pdl_value_type();
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.dialect().namespace().unwrap(), "pdl");
        assert_eq!(ValueTypeRef::type_id(), r#type.type_id());
    }

    #[test]
    fn test_value_type_equality() {
        let context = Context::new();
        let type_1 = context.pdl_value_type();
        let type_2 = context.pdl_value_type();
        assert_eq!(type_1, type_2);

        let context = Context::new();
        let type_2 = context.pdl_value_type();
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_value_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.pdl_value_type(), "!pdl.value");
    }

    #[test]
    fn test_value_type_parsing() {
        let context = Context::new();
        let r#type = context.pdl_value_type();
        assert_eq!(context.parse_type("!pdl.value").unwrap(), r#type);
    }

    #[test]
    fn test_value_type_casting() {
        let context = Context::new();
        test_type_casting(context.pdl_value_type());
    }
}
