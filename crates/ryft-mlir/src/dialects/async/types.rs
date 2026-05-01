use ryft_xla_sys::bindings::MlirType;

use crate::{Context, DialectHandle, Type, TypeRef, mlir_subtype_trait_impls};

macro_rules! async_unit_type {
    (
        $name:ident,
        $context_method:ident,
        $mnemonic:literal,
        $summary:literal,
        $documentation:literal $(,)*
    ) => {
        #[doc = $summary]
        #[doc = ""]
        #[doc = $documentation]
        #[doc = ""]
        #[doc = "Refer to the [official MLIR async dialect documentation](https://mlir.llvm.org/docs/Dialects/AsyncDialect/#types)"]
        #[doc = "for more information."]
        #[derive(Copy, Clone)]
        pub struct $name<'c, 't> {
            /// Handle that represents this [`Type`] in the MLIR C API.
            handle: MlirType,

            /// [`Context`] that owns this [`Type`].
            context: &'c Context<'t>,
        }

        impl<'c, 't> Type<'c, 't> for $name<'c, 't> {
            unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
                if handle.ptr.is_null() {
                    return None;
                }
                let r#type = unsafe { TypeRef::from_c_api(handle, context) }?;
                if r#type.to_string() == $mnemonic { Some(Self { handle, context }) } else { None }
            }

            unsafe fn to_c_api(&self) -> MlirType {
                self.handle
            }

            fn context(&self) -> &'c Context<'t> {
                self.context
            }
        }

        mlir_subtype_trait_impls!($name<'c, 't> as Type, mlir_type = Type);

        impl<'t> Context<'t> {
            #[doc = "Creates a new async "]
            #[doc = $summary]
            #[doc = " owned by this [`Context`]."]
            pub fn $context_method<'c>(&'c self) -> $name<'c, 't> {
                self.load_dialect(DialectHandle::r#async());
                self.parse_type($mnemonic)
                    .and_then(|r#type| r#type.cast())
                    .unwrap_or_else(|| panic!("failed to parse async type `{}`", $mnemonic))
            }
        }
    };
}

async_unit_type!(
    TokenTypeRef,
    async_token_type,
    "!async.token",
    "Async token [`Type`].",
    "`async.token` represents completion of an asynchronous operation and can be used to express execution dependencies.",
);

/// Async value [`Type`].
///
/// `async.value<T>` represents a value of type `T` that may become available in the future.
///
/// Refer to the [official MLIR async dialect documentation](https://mlir.llvm.org/docs/Dialects/AsyncDialect/#valuetype)
/// for more information.
#[derive(Copy, Clone)]
pub struct ValueTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> ValueTypeRef<'c, 't> {
    /// Returns the underlying value [`Type`] wrapped by this async value type.
    pub fn value_type(&self) -> TypeRef<'c, 't> {
        let rendered_type = self.to_string();
        let value_type = rendered_type
            .strip_prefix("!async.value<")
            .and_then(|rendered_type| rendered_type.strip_suffix('>'))
            .expect("invalid `!async.value` type");
        self.context.parse_type(value_type).expect("invalid `!async.value` value type")
    }
}

impl<'c, 't> Type<'c, 't> for ValueTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() {
            return None;
        }
        let r#type = unsafe { TypeRef::from_c_api(handle, context) }?;
        let rendered_type = r#type.to_string();
        if rendered_type.starts_with("!async.value<") && rendered_type.ends_with('>') {
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

async_unit_type!(
    GroupTypeRef,
    async_group_type,
    "!async.group",
    "Async group [`Type`].",
    "`async.group` represents a runtime group of async tokens or values that can be awaited together.",
);

async_unit_type!(
    CoroIdTypeRef,
    async_coro_id_type,
    "!async.coro.id",
    "Async coroutine identifier [`Type`].",
    "`async.coro.id` identifies a switched-resume coroutine during async-to-LLVM lowering.",
);

async_unit_type!(
    CoroHandleTypeRef,
    async_coro_handle_type,
    "!async.coro.handle",
    "Async coroutine handle [`Type`].",
    "`async.coro.handle` represents a pointer-like handle to a coroutine frame.",
);

async_unit_type!(
    CoroStateTypeRef,
    async_coro_state_type,
    "!async.coro.state",
    "Async coroutine state [`Type`].",
    "`async.coro.state` represents saved coroutine suspension state.",
);

impl<'t> Context<'t> {
    /// Creates a new [`ValueTypeRef`] owned by this [`Context`] for the provided underlying value type.
    pub fn async_value_type<'c, T: Type<'c, 't>>(&'c self, value_type: T) -> ValueTypeRef<'c, 't> {
        self.load_dialect(DialectHandle::r#async());
        let source = format!("!async.value<{value_type}>");
        self.parse_type(source)
            .and_then(|r#type| r#type.cast())
            .expect("invalid arguments to `Context::async_value_type`")
    }
}

#[cfg(test)]
mod tests {
    use crate::Type;
    use crate::types::tests::{test_type_casting, test_type_display_and_debug};

    use super::*;

    macro_rules! async_unit_type_tests {
        ($constructor:ident, $test_prefix:ident, $expected:literal) => {
            paste::paste! {
                #[test]
                fn [<test_ $test_prefix>]() {
                    let context = Context::new();
                    let r#type = context.$constructor();
                    assert_eq!(&context, r#type.context());
                    assert_eq!(r#type.dialect().namespace().unwrap(), "async");
                }

                #[test]
                fn [<test_ $test_prefix _equality>]() {
                    let context = Context::new();
                    let type_1 = context.$constructor();
                    let type_2 = context.$constructor();
                    assert_eq!(type_1, type_2);

                    let context = Context::new();
                    let type_2 = context.$constructor();
                    assert_ne!(type_1, type_2);
                }

                #[test]
                fn [<test_ $test_prefix _display_and_debug>]() {
                    let context = Context::new();
                    let r#type = context.$constructor();
                    test_type_display_and_debug(r#type, $expected);
                }

                #[test]
                fn [<test_ $test_prefix _parsing>]() {
                    let context = Context::new();
                    let r#type = context.$constructor();
                    assert_eq!(context.parse_type($expected).unwrap(), r#type);
                }

                #[test]
                fn [<test_ $test_prefix _casting>]() {
                    let context = Context::new();
                    let r#type = context.$constructor();
                    test_type_casting(r#type);
                }
            }
        };
    }

    async_unit_type_tests!(async_token_type, token_type, "!async.token");
    async_unit_type_tests!(async_group_type, group_type, "!async.group");
    async_unit_type_tests!(async_coro_id_type, coro_id_type, "!async.coro.id");
    async_unit_type_tests!(async_coro_handle_type, coro_handle_type, "!async.coro.handle");
    async_unit_type_tests!(async_coro_state_type, coro_state_type, "!async.coro.state");

    #[test]
    fn test_value_type() {
        let context = Context::new();
        let value_type = context.async_value_type(context.float32_type());
        assert_eq!(&context, value_type.context());
        assert_eq!(value_type.dialect().namespace().unwrap(), "async");
        assert_eq!(value_type.value_type(), context.float32_type());
    }

    #[test]
    fn test_value_type_equality() {
        let context = Context::new();
        let value_type_1 = context.async_value_type(context.float32_type());
        let value_type_2 = context.async_value_type(context.float32_type());
        assert_eq!(value_type_1, value_type_2);

        let value_type_2 = context.async_value_type(context.float64_type());
        assert_ne!(value_type_1, value_type_2);

        let context = Context::new();
        let value_type_2 = context.async_value_type(context.float32_type());
        assert_ne!(value_type_1, value_type_2);
    }

    #[test]
    fn test_value_type_display_and_debug() {
        let context = Context::new();
        let value_type = context.async_value_type(context.float32_type());
        test_type_display_and_debug(value_type, "!async.value<f32>");
    }

    #[test]
    fn test_value_type_parsing() {
        let context = Context::new();
        let value_type = context.async_value_type(context.float32_type());
        assert_eq!(context.parse_type("!async.value<f32>").unwrap(), value_type);
    }

    #[test]
    fn test_value_type_casting() {
        let context = Context::new();
        let value_type = context.async_value_type(context.float32_type());
        test_type_casting(value_type);
    }
}
