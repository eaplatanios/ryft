use ryft_xla_sys::bindings::{
    MlirType, mlirTransformAnyOpTypeGet, mlirTransformAnyParamTypeGet, mlirTransformAnyValueTypeGet,
    mlirTransformOperationTypeGet, mlirTransformOperationTypeGetOperationName, mlirTransformParamTypeGet,
    mlirTransformParamTypeGetType, mlirTypeIsATransformAnyOpType, mlirTypeIsATransformAnyParamType,
    mlirTypeIsATransformAnyValueType, mlirTypeIsATransformOperationType, mlirTypeIsATransformParamType,
};
use ryft_xla_sys::mlir::dialects::transform::{
    mlirTransformAffineMapParamTypeGet, mlirTransformTypeParamTypeGet, mlirTypeIsATransformAffineMapParamType,
    mlirTypeIsATransformTypeParamType,
};

use crate::{Context, DialectHandle, StringRef, Type, TypeRef, mlir_subtype_trait_impls};

macro_rules! transform_unit_type {
    ($type_ref:ident, $context_method:ident, $is_a:path, $get:path, $display:literal, $doc:literal) => {
        #[doc = $doc]
        ///
        /// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Transform/#type-definitions)
        /// for more information.
        #[derive(Copy, Clone)]
        pub struct $type_ref<'c, 't> {
            /// Handle that represents this [`Type`] in the MLIR C API.
            handle: MlirType,

            /// [`Context`] that owns this [`Type`].
            context: &'c Context<'t>,
        }

        impl<'c, 't> Type<'c, 't> for $type_ref<'c, 't> {
            unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
                if !handle.ptr.is_null() && unsafe { $is_a(handle) } {
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

        mlir_subtype_trait_impls!($type_ref<'c, 't> as Type, mlir_type = Type);

        impl<'t> Context<'t> {
            #[doc = "Creates a new `"]
            #[doc = $display]
            #[doc = "` Transform dialect [`Type`] owned by this [`Context`]."]
            pub fn $context_method<'c>(&'c self) -> $type_ref<'c, 't> {
                self.load_dialect(DialectHandle::transform());
                unsafe { $type_ref::from_c_api($get(*self.handle.borrow_mut()), self).unwrap() }
            }
        }
    };
}

transform_unit_type!(
    AffineMapParamTypeRef,
    transform_affine_map_param_type,
    mlirTypeIsATransformAffineMapParamType,
    mlirTransformAffineMapParamTypeGet,
    "!transform.affine_map",
    "Transform dialect [`Type`] for parameters associated with affine map attributes."
);

transform_unit_type!(
    AnyOpTypeRef,
    transform_any_op_type,
    mlirTypeIsATransformAnyOpType,
    mlirTransformAnyOpTypeGet,
    "!transform.any_op",
    "Transform dialect [`Type`] for handles associated with arbitrary payload operations."
);

transform_unit_type!(
    AnyValueTypeRef,
    transform_any_value_type,
    mlirTypeIsATransformAnyValueType,
    mlirTransformAnyValueTypeGet,
    "!transform.any_value",
    "Transform dialect [`Type`] for handles associated with arbitrary payload values."
);

/// Transform dialect [`Type`] for handles associated with payload operations with a specific operation name.
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Transform/#operationtype)
/// for more information.
#[derive(Copy, Clone)]
pub struct OperationTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl OperationTypeRef<'_, '_> {
    /// Returns the name of the payload operation accepted by this Transform handle type.
    pub fn operation_name(&self) -> StringRef<'_> {
        unsafe { StringRef::from_c_api(mlirTransformOperationTypeGetOperationName(self.handle)) }
    }
}

impl<'c, 't> Type<'c, 't> for OperationTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirTypeIsATransformOperationType(handle) } {
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

transform_unit_type!(
    AnyParamTypeRef,
    transform_any_param_type,
    mlirTypeIsATransformAnyParamType,
    mlirTransformAnyParamTypeGet,
    "!transform.any_param",
    "Transform dialect [`Type`] for parameters associated with attributes of any type."
);

/// Transform dialect [`Type`] for parameters associated with attributes of the specified underlying type.
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Transform/#paramtype)
/// for more information.
#[derive(Copy, Clone)]
pub struct ParamTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> ParamTypeRef<'c, 't> {
    /// Returns the underlying MLIR type of attributes accepted by this Transform parameter type.
    pub fn element_type(&self) -> TypeRef<'c, 't> {
        unsafe { TypeRef::from_c_api(mlirTransformParamTypeGetType(self.handle), self.context).unwrap() }
    }
}

impl<'c, 't> Type<'c, 't> for ParamTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirTypeIsATransformParamType(handle) } {
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

mlir_subtype_trait_impls!(ParamTypeRef<'c, 't> as Type, mlir_type = Type);

transform_unit_type!(
    TypeParamTypeRef,
    transform_type_param_type,
    mlirTypeIsATransformTypeParamType,
    mlirTransformTypeParamTypeGet,
    "!transform.type",
    "Transform dialect [`Type`] for parameters associated with type attributes."
);

impl<'t> Context<'t> {
    /// Creates a new [`OperationTypeRef`] owned by this [`Context`].
    pub fn transform_operation_type<'c, 's, S: Into<StringRef<'s>>>(
        &'c self,
        operation_name: S,
    ) -> OperationTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::transform());
        unsafe {
            OperationTypeRef::from_c_api(
                mlirTransformOperationTypeGet(*self.handle.borrow_mut(), operation_name.into().to_c_api()),
                self,
            )
            .unwrap()
        }
    }

    /// Creates a new [`ParamTypeRef`] owned by this [`Context`].
    pub fn transform_param_type<'c, T: Type<'c, 't>>(&'c self, element_type: T) -> ParamTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::transform());
        unsafe {
            ParamTypeRef::from_c_api(
                mlirTransformParamTypeGet(*self.handle.borrow_mut(), element_type.to_c_api()),
                self,
            )
            .unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Type;
    use crate::types::tests::{test_type_casting, test_type_display_and_debug};

    use super::*;

    macro_rules! test_unit_transform_type {
        ($module:ident, $method:ident, $expected:literal) => {
            mod $module {
                use super::*;

                #[test]
                fn test_construction() {
                    let context = Context::new();
                    let r#type = context.$method();
                    assert_eq!(&context, r#type.context());
                    assert_eq!(r#type.dialect().namespace().unwrap(), "transform");
                }

                #[test]
                fn test_equality() {
                    let context = Context::new();
                    let type_1 = context.$method();
                    let type_2 = context.$method();
                    assert_eq!(type_1, type_2);

                    let context = Context::new();
                    let type_2 = context.$method();
                    assert_ne!(type_1, type_2);
                }

                #[test]
                fn test_display_and_debug() {
                    let context = Context::new();
                    let r#type = context.$method();
                    test_type_display_and_debug(r#type, $expected);
                }

                #[test]
                fn test_parsing() {
                    let context = Context::new();
                    let r#type = context.$method();
                    assert_eq!(context.parse_type($expected).unwrap(), r#type);
                }

                #[test]
                fn test_casting() {
                    let context = Context::new();
                    let r#type = context.$method();
                    test_type_casting(r#type);
                }
            }
        };
    }

    test_unit_transform_type!(affine_map_param_type, transform_affine_map_param_type, "!transform.affine_map");
    test_unit_transform_type!(any_op_type, transform_any_op_type, "!transform.any_op");
    test_unit_transform_type!(any_value_type, transform_any_value_type, "!transform.any_value");
    test_unit_transform_type!(any_param_type, transform_any_param_type, "!transform.any_param");
    test_unit_transform_type!(type_param_type, transform_type_param_type, "!transform.type");

    mod operation_type {
        use super::*;

        #[test]
        fn test_construction_and_accessors() {
            let context = Context::new();
            let r#type = context.transform_operation_type("func.func");
            assert_eq!(&context, r#type.context());
            assert_eq!(r#type.dialect().namespace().unwrap(), "transform");
            assert_eq!(r#type.operation_name().as_str(), Ok("func.func"));
        }

        #[test]
        fn test_equality() {
            let context = Context::new();
            let type_1 = context.transform_operation_type("func.func");
            let type_2 = context.transform_operation_type("func.func");
            assert_eq!(type_1, type_2);

            let type_2 = context.transform_operation_type("builtin.module");
            assert_ne!(type_1, type_2);

            let context = Context::new();
            let type_2 = context.transform_operation_type("func.func");
            assert_ne!(type_1, type_2);
        }

        #[test]
        fn test_display_and_debug() {
            let context = Context::new();
            let r#type = context.transform_operation_type("func.func");
            test_type_display_and_debug(r#type, "!transform.op<\"func.func\">");
        }

        #[test]
        fn test_parsing() {
            let context = Context::new();
            let r#type = context.transform_operation_type("func.func");
            assert_eq!(context.parse_type("!transform.op<\"func.func\">").unwrap(), r#type);
        }

        #[test]
        fn test_casting() {
            let context = Context::new();
            let r#type = context.transform_operation_type("func.func");
            test_type_casting(r#type);
        }
    }

    mod param_type {
        use super::*;

        #[test]
        fn test_construction_and_accessors() {
            let context = Context::new();
            let element_type = context.signless_integer_type(32);
            let r#type = context.transform_param_type(element_type);
            assert_eq!(&context, r#type.context());
            assert_eq!(r#type.dialect().namespace().unwrap(), "transform");
            assert_eq!(r#type.element_type(), element_type);
        }

        #[test]
        fn test_equality() {
            let context = Context::new();
            let type_1 = context.transform_param_type(context.signless_integer_type(32));
            let type_2 = context.transform_param_type(context.signless_integer_type(32));
            assert_eq!(type_1, type_2);

            let type_2 = context.transform_param_type(context.signless_integer_type(64));
            assert_ne!(type_1, type_2);

            let context = Context::new();
            let type_2 = context.transform_param_type(context.signless_integer_type(32));
            assert_ne!(type_1, type_2);
        }

        #[test]
        fn test_display_and_debug() {
            let context = Context::new();
            let r#type = context.transform_param_type(context.signless_integer_type(32));
            test_type_display_and_debug(r#type, "!transform.param<i32>");
        }

        #[test]
        fn test_parsing() {
            let context = Context::new();
            let r#type = context.transform_param_type(context.signless_integer_type(32));
            assert_eq!(context.parse_type("!transform.param<i32>").unwrap(), r#type);
        }

        #[test]
        fn test_casting() {
            let context = Context::new();
            let r#type = context.transform_param_type(context.signless_integer_type(32));
            test_type_casting(r#type);
        }
    }
}
