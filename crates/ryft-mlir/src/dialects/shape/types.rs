use ryft_xla_sys::bindings::MlirType;
use ryft_xla_sys::mlir::dialects::shape::{
    mlirShapeShapeTypeGet, mlirShapeSizeTypeGet, mlirShapeValueShapeTypeGet, mlirShapeWitnessTypeGet,
    mlirTypeIsAShapeShapeType, mlirTypeIsAShapeSizeType, mlirTypeIsAShapeValueShapeType, mlirTypeIsAShapeWitnessType,
};

use crate::{Context, DialectHandle, Type, mlir_subtype_trait_impls};

/// Shape dialect [`Type`] that represents a possibly unranked, partially unknown, or invalid shape.
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/ShapeDialect/#shapetype)
/// for more information.
#[derive(Copy, Clone)]
pub struct ShapeTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> Type<'c, 't> for ShapeTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirTypeIsAShapeShapeType(handle) } {
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

mlir_subtype_trait_impls!(ShapeTypeRef<'c, 't> as Type, mlir_type = Type);

/// Shape dialect [`Type`] that represents a non-negative size with support for unknown and invalid values.
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/ShapeDialect/#sizetype)
/// for more information.
#[derive(Copy, Clone)]
pub struct SizeTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> Type<'c, 't> for SizeTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirTypeIsAShapeSizeType(handle) } {
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

mlir_subtype_trait_impls!(SizeTypeRef<'c, 't> as Type, mlir_type = Type);

/// Shape dialect [`Type`] that pairs a value with shape information.
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/ShapeDialect/#valueshapetype)
/// for more information.
#[derive(Copy, Clone)]
pub struct ValueShapeTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> Type<'c, 't> for ValueShapeTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirTypeIsAShapeValueShapeType(handle) } {
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

mlir_subtype_trait_impls!(ValueShapeTypeRef<'c, 't> as Type, mlir_type = Type);

/// Shape dialect [`Type`] that represents a constraint witness.
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/ShapeDialect/#witnesstype)
/// for more information.
#[derive(Copy, Clone)]
pub struct WitnessTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> Type<'c, 't> for WitnessTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirTypeIsAShapeWitnessType(handle) } {
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

mlir_subtype_trait_impls!(WitnessTypeRef<'c, 't> as Type, mlir_type = Type);

impl<'t> Context<'t> {
    /// Creates a new [`ShapeTypeRef`] owned by this [`Context`].
    pub fn shape_type<'c>(&'c self) -> ShapeTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::shape());
        unsafe { ShapeTypeRef::from_c_api(mlirShapeShapeTypeGet(*self.handle.borrow()), self).unwrap() }
    }

    /// Creates a new [`SizeTypeRef`] owned by this [`Context`].
    pub fn shape_size_type<'c>(&'c self) -> SizeTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::shape());
        unsafe { SizeTypeRef::from_c_api(mlirShapeSizeTypeGet(*self.handle.borrow()), self).unwrap() }
    }

    /// Creates a new [`ValueShapeTypeRef`] owned by this [`Context`].
    pub fn shape_value_shape_type<'c>(&'c self) -> ValueShapeTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::shape());
        unsafe { ValueShapeTypeRef::from_c_api(mlirShapeValueShapeTypeGet(*self.handle.borrow()), self).unwrap() }
    }

    /// Creates a new [`WitnessTypeRef`] owned by this [`Context`].
    pub fn shape_witness_type<'c>(&'c self) -> WitnessTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::shape());
        unsafe { WitnessTypeRef::from_c_api(mlirShapeWitnessTypeGet(*self.handle.borrow()), self).unwrap() }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::types::tests::{test_type_casting, test_type_display_and_debug};
    use crate::{Context, Type};

    use super::*;

    #[test]
    fn test_shape_type() {
        let context = Context::new();
        let r#type = context.shape_type();
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.dialect().namespace().unwrap(), "shape");
    }

    #[test]
    fn test_shape_type_equality() {
        let context = Context::new();
        let type_1 = context.shape_type();
        let type_2 = context.shape_type();
        assert_eq!(type_1, type_2);

        let context = Context::new();
        let type_2 = context.shape_type();
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_shape_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.shape_type(), "!shape.shape");
    }

    #[test]
    fn test_shape_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::shape());
        assert_eq!(context.parse_type("!shape.shape").unwrap(), context.shape_type());
    }

    #[test]
    fn test_shape_type_casting() {
        let context = Context::new();
        test_type_casting(context.shape_type());
    }

    #[test]
    fn test_size_type() {
        let context = Context::new();
        let r#type = context.shape_size_type();
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.dialect().namespace().unwrap(), "shape");
    }

    #[test]
    fn test_size_type_equality() {
        let context = Context::new();
        let type_1 = context.shape_size_type();
        let type_2 = context.shape_size_type();
        assert_eq!(type_1, type_2);

        let context = Context::new();
        let type_2 = context.shape_size_type();
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_size_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.shape_size_type(), "!shape.size");
    }

    #[test]
    fn test_size_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::shape());
        assert_eq!(context.parse_type("!shape.size").unwrap(), context.shape_size_type());
    }

    #[test]
    fn test_size_type_casting() {
        let context = Context::new();
        test_type_casting(context.shape_size_type());
    }

    #[test]
    fn test_value_shape_type() {
        let context = Context::new();
        let r#type = context.shape_value_shape_type();
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.dialect().namespace().unwrap(), "shape");
    }

    #[test]
    fn test_value_shape_type_equality() {
        let context = Context::new();
        let type_1 = context.shape_value_shape_type();
        let type_2 = context.shape_value_shape_type();
        assert_eq!(type_1, type_2);

        let context = Context::new();
        let type_2 = context.shape_value_shape_type();
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_value_shape_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.shape_value_shape_type(), "!shape.value_shape");
    }

    #[test]
    fn test_value_shape_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::shape());
        assert_eq!(context.parse_type("!shape.value_shape").unwrap(), context.shape_value_shape_type());
    }

    #[test]
    fn test_value_shape_type_casting() {
        let context = Context::new();
        test_type_casting(context.shape_value_shape_type());
    }

    #[test]
    fn test_witness_type() {
        let context = Context::new();
        let r#type = context.shape_witness_type();
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.dialect().namespace().unwrap(), "shape");
    }

    #[test]
    fn test_witness_type_equality() {
        let context = Context::new();
        let type_1 = context.shape_witness_type();
        let type_2 = context.shape_witness_type();
        assert_eq!(type_1, type_2);

        let context = Context::new();
        let type_2 = context.shape_witness_type();
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_witness_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.shape_witness_type(), "!shape.witness");
    }

    #[test]
    fn test_witness_type_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::shape());
        assert_eq!(context.parse_type("!shape.witness").unwrap(), context.shape_witness_type());
    }

    #[test]
    fn test_witness_type_casting() {
        let context = Context::new();
        test_type_casting(context.shape_witness_type());
    }
}
